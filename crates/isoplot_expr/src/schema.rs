use std::{marker::PhantomData, mem::MaybeUninit};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum FieldType {
    I32,
    F32,
}

#[derive(Copy, Clone, Debug)]
pub enum FieldValue {
    I32(i32),
    F32(f32),
}

impl FieldValue {
    pub(crate) fn ty(self) -> FieldType {
        match self {
            FieldValue::I32(_) => FieldType::I32,
            FieldValue::F32(_) => FieldType::F32,
        }
    }

    pub(crate) fn unwrap_i32(self) -> i32 {
        match self {
            Self::I32(value) => value,
            _ => panic!(),
        }
    }

    pub(crate) fn unwrap_f32(self) -> f32 {
        match self {
            Self::F32(value) => value,
            _ => panic!(),
        }
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Tile<T: Copy, const N: usize> {
    pub values: [T; N],
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Field {
    pub ty: FieldType,
    pub offset: usize,
}

impl Field {
    pub const fn new(ty: FieldType, offset: usize) -> Self {
        Self { ty, offset }
    }

    pub(crate) const fn offset_by(self, offset: usize) -> Self {
        Self {
            ty: self.ty,
            offset: self.offset.strict_add(offset),
        }
    }
}

pub unsafe trait Schema<const TILE_LEN: usize = 1>: Copy {
    /// Number of exported fields
    const NUM_FIELDS: usize;

    /// Returns the exported field at the specified index.
    fn field(index: usize) -> Field;

    fn fields() -> impl ExactSizeIterator<Item = Field> {
        (0..Self::NUM_FIELDS).map(Self::field)
    }
}

pub(crate) trait SchemaExt<const TILE_LEN: usize>: Schema<TILE_LEN> {
    fn get_i32(&self, index: usize) -> &[i32; TILE_LEN] {
        let field = Self::field(index);
        assert_eq!(field.ty, FieldType::I32);
        unsafe { field_ptr(self as *const _, index).as_ref_unchecked() }
    }

    fn get_f32(&self, index: usize) -> &[f32; TILE_LEN] {
        let field = Self::field(index);
        assert_eq!(field.ty, FieldType::F32);
        unsafe { field_ptr(self as *const _, index).as_ref_unchecked() }
    }
}

impl<const TILE_LEN: usize, S: Schema<TILE_LEN>> SchemaExt<TILE_LEN> for S {}

pub(crate) unsafe trait WriteSchema<const TILE_LEN: usize> {
    fn write_i32(&mut self, index: usize, value: [i32; TILE_LEN]);
    fn write_f32(&mut self, index: usize, value: [f32; TILE_LEN]);
}

unsafe impl<S: Schema<TILE_LEN>, const TILE_LEN: usize> WriteSchema<TILE_LEN> for MaybeUninit<S> {
    fn write_i32(&mut self, index: usize, value: [i32; TILE_LEN]) {
        assert_eq!(S::field(index).ty, FieldType::I32);
        unsafe { field_mut_ptr(self as *mut MaybeUninit<S> as *mut S, index).write(value) }
    }

    fn write_f32(&mut self, index: usize, value: [f32; TILE_LEN]) {
        assert_eq!(S::field(index).ty, FieldType::F32);
        unsafe { field_mut_ptr(self as *mut MaybeUninit<S> as *mut S, index).write(value) }
    }
}

unsafe fn field_ptr<S, T, const TILE_LEN: usize>(
    schema: *const S,
    index: usize,
) -> *const [T; TILE_LEN]
where
    S: Schema<TILE_LEN>,
{
    unsafe {
        schema
            .byte_add(S::field(index).offset)
            .cast::<[T; TILE_LEN]>()
    }
}

unsafe fn field_mut_ptr<S, T, const TILE_LEN: usize>(
    schema: *mut S,
    index: usize,
) -> *mut [T; TILE_LEN]
where
    S: Schema<TILE_LEN>,
{
    unsafe {
        schema
            .byte_add(S::field(index).offset)
            .cast::<[T; TILE_LEN]>()
    }
}

unsafe impl Schema for i32 {
    const NUM_FIELDS: usize = 1;

    fn field(index: usize) -> Field {
        assert_eq!(index, 0, "invalid index");
        Field::new(FieldType::I32, 0)
    }
}

unsafe impl Schema for f32 {
    const NUM_FIELDS: usize = 1;

    fn field(index: usize) -> Field {
        assert_eq!(index, 0, "invalid index");
        Field::new(FieldType::F32, 0)
    }
}

unsafe impl<const TILE_LEN: usize> Schema<TILE_LEN> for Tile<i32, TILE_LEN> {
    const NUM_FIELDS: usize = 1;

    fn field(index: usize) -> Field {
        assert_eq!(index, 0, "invalid index");
        Field::new(FieldType::I32, 0)
    }
}

unsafe impl<const TILE_LEN: usize> Schema<TILE_LEN> for Tile<f32, TILE_LEN> {
    const NUM_FIELDS: usize = 1;

    fn field(index: usize) -> Field {
        assert_eq!(index, 0, "invalid index");
        Field::new(FieldType::F32, 0)
    }
}

unsafe impl<S, const TILE_LEN: usize, const N: usize> Schema<TILE_LEN> for [S; N]
where
    S: Schema<TILE_LEN>,
{
    const NUM_FIELDS: usize = S::NUM_FIELDS.strict_mul(N);

    fn field(index: usize) -> Field {
        assert!(index < Self::NUM_FIELDS, "invalid index");
        let field = S::field(index % S::NUM_FIELDS);
        Field::new(
            field.ty,
            size_of::<S>().strict_mul(index / S::NUM_FIELDS) + field.offset,
        )
    }
}

// It is currently impossible to verify this at compile time because `Schema::field()` cannot be `const`.
pub(crate) unsafe trait CastSchemaSafe<const TILE_LEN: usize, S: Schema<TILE_LEN>>:
    Schema<TILE_LEN>
{
}

unsafe impl CastSchemaSafe<1, Tile<f32, 1>> for f32 {}
unsafe impl CastSchemaSafe<1, f32> for Tile<f32, 1> {}

unsafe impl<const TILE_LEN: usize, const N: usize, S, U> CastSchemaSafe<TILE_LEN, [S; N]> for [U; N]
where
    S: Schema<TILE_LEN> + CastSchemaSafe<TILE_LEN, U>,
    U: Schema<TILE_LEN>,
{
}

pub(crate) struct Slot<T> {
    _marker: (!, PhantomData<T>),
}

pub(crate) unsafe trait SchemaTemplate<const TILE_LEN: usize> {
    type With<I32, F32>;
}

unsafe impl<const TILE_LEN: usize> SchemaTemplate<TILE_LEN> for Slot<i32> {
    type With<I32, F32> = I32;
}

unsafe impl<const TILE_LEN: usize> SchemaTemplate<TILE_LEN> for Slot<f32> {
    type With<I32, F32> = F32;
}

unsafe impl<const TILE_LEN: usize, const N: usize> SchemaTemplate<TILE_LEN> for [Slot<i32>; N] {
    type With<I32, F32> = [I32; N];
}

unsafe impl<const TILE_LEN: usize, const N: usize> SchemaTemplate<TILE_LEN> for [Slot<f32>; N] {
    type With<I32, F32> = [F32; N];
}

pub(crate) trait SchemaWith<const TILE_LEN: usize, I32, F32>:
    SchemaTemplate<TILE_LEN, With<I32, F32>: Schema<TILE_LEN>>
{
}

impl<const TILE_LEN: usize, I32, F32, S> SchemaWith<TILE_LEN, I32, F32> for S where
    S: SchemaTemplate<TILE_LEN, With<I32, F32>: Schema<TILE_LEN>>
{
}
