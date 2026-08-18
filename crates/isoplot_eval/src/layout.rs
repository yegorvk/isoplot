use std::{iter, slice};

use bytemuck::{Pod, Zeroable};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ValueType {
    I32,
    F32,
}

#[derive(Copy, Clone)]
pub struct TypedValue {
    value: RawValue,
    ty: ValueType,
}

impl TypedValue {
    pub(crate) fn ty(&self) -> ValueType {
        self.ty
    }

    pub(crate) fn value(&self) -> RawValue {
        self.value
    }
}

impl From<i32> for TypedValue {
    fn from(value: i32) -> Self {
        Self {
            value: RawValue::from_i32(value),
            ty: ValueType::I32,
        }
    }
}

impl From<f32> for TypedValue {
    fn from(value: f32) -> Self {
        Self {
            value: RawValue::from_f32(value),
            ty: ValueType::F32,
        }
    }
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(transparent)]
pub(crate) struct RawValue(u32);

impl RawValue {
    pub(crate) const ZERO: Self = Self(0);

    pub(crate) const fn from_i32(value: i32) -> Self {
        Self(value as u32)
    }

    pub(crate) const fn from_f32(value: f32) -> Self {
        Self(value.to_bits())
    }

    pub(crate) const fn as_i32(self) -> i32 {
        self.0 as i32
    }

    pub(crate) const fn as_f32(self) -> f32 {
        f32::from_bits(self.0)
    }
}

/// Types that can be safely transmuted into `RawValue`.
trait IntoValue: Pod + Zeroable {
    const TYPE: ValueType;
}

impl IntoValue for i32 {
    const TYPE: ValueType = ValueType::I32;
}

impl IntoValue for f32 {
    const TYPE: ValueType = ValueType::F32;
}

mod private {
    use super::ValueType;
    use bytemuck::{Pod, Zeroable};

    /// Plain data made of exactly `LEN` consecutive `RawValue`s.
    #[doc(hidden)]
    pub trait Vector: Pod + Zeroable {
        type Scalar: Vector;
        const LEN: usize;
        fn types() -> impl Iterator<Item = ValueType>;
    }
}

pub(crate) use private::Vector;

pub trait Layout: Vector {}
impl<T: Vector> Layout for T {}

impl<T: IntoValue> Vector for T {
    type Scalar = T;
    const LEN: usize = 1;

    fn types() -> impl Iterator<Item = ValueType> {
        iter::once(T::TYPE)
    }
}

impl<T: Vector, const N: usize> Vector for [T; N] {
    type Scalar = T::Scalar;
    const LEN: usize = N * T::LEN;

    fn types() -> impl Iterator<Item = ValueType> {
        (0..N).flat_map(|_| T::types())
    }
}

mod sealed {
    pub(crate) trait Sealed {}
}

impl<T: Vector> sealed::Sealed for T {}

pub(crate) trait VectorExt: Vector + sealed::Sealed {
    fn raw_values(&self) -> &[RawValue];
    fn raw_values_mut(&mut self) -> &mut [RawValue];
}

impl<T: Vector> VectorExt for T {
    #[inline]
    fn raw_values(&self) -> &[RawValue] {
        bytemuck::must_cast_slice(slice::from_ref(self))
    }

    #[inline]
    fn raw_values_mut(&mut self) -> &mut [RawValue] {
        bytemuck::must_cast_slice_mut(slice::from_mut(self))
    }
}
