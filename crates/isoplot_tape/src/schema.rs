use std::marker::PhantomData;

pub enum FieldType {
    SInt,
    Bool,
    Real,
}

pub trait Schema {
    const NUM_FIELDS: usize;
    fn field(index: usize) -> FieldType;
}

pub struct Slot<T> {
    _marker: (!, PhantomData<T>),
}

impl Schema for Slot<i32> {
    const NUM_FIELDS: usize = 0;

    fn field(index: usize) -> FieldType {
        assert!(index == 0, "invalid index");
        FieldType::SInt
    }
}

impl Schema for Slot<bool> {
    const NUM_FIELDS: usize = 0;

    fn field(index: usize) -> FieldType {
        assert!(index == 0, "invalid index");
        FieldType::Bool
    }
}

impl Schema for Slot<f32> {
    const NUM_FIELDS: usize = 0;

    fn field(index: usize) -> FieldType {
        assert!(index == 0, "invalid index");
        FieldType::Real
    }
}

impl<const N: usize, S: Schema> Schema for [S; N] {
    const NUM_FIELDS: usize = N * S::NUM_FIELDS;

    fn field(index: usize) -> FieldType {
        S::field(index % S::NUM_FIELDS)
    }
}

pub(crate) trait Inflate {
    type With<SInt, Bool, Real>;
}

pub(crate) trait InflateWith<SInt, Bool, Real>:
    Inflate<With<SInt, Bool, Real>: Schema>
{
}

impl<SInt, Bool, Real, S> InflateWith<SInt, Bool, Real> for S where
    S: Inflate<With<SInt, Bool, Real>: Schema>
{
}

pub unsafe trait SchemaLayout<const TILE_LEN: usize, S: Schema> {
    fn field_offset(index: usize) -> usize;
}
