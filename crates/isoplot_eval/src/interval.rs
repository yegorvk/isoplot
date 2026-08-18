use std::iter;

use bytemuck::{Pod, Zeroable};

use crate::{
    autodiff::Gradient,
    layout::{ValueType, Vector},
    program::Program,
    tape::Tape,
};

/// A closed and bounded interval
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct Interval {
    min: f32,
    max: f32,
}

impl Interval {
    pub const fn new(min: f32, max: f32) -> Self {
        assert!(min <= max);
        Self { min, max }
    }

    pub const fn min(&self) -> f32 {
        self.min
    }

    pub const fn max(&self) -> f32 {
        self.max
    }

    pub const fn contains_zero(&self) -> bool {
        self.min <= 0.0 && self.max >= 0.0
    }
}

impl Vector for Interval {
    type Scalar = Interval;
    const LEN: usize = 2;

    fn types() -> impl Iterator<Item = ValueType> {
        iter::repeat_n(ValueType::F32, Self::LEN)
    }
}

mod private {
    use crate::layout::Layout;

    #[doc(hidden)]
    pub trait IntervalType: Layout {
        type Interval: Layout;
    }
}

pub(crate) use private::IntervalType;

impl IntervalType for f32 {
    type Interval = Interval;
}

impl<T: IntervalType, const N: usize> IntervalType for [T; N] {
    type Interval = [T::Interval; N];
}

impl<V: IntervalType> IntervalType for Gradient<V> {
    type Interval = Gradient<V::Interval>;
}

pub(crate) fn interval<Args, Ret>(
    program: &Program<Args, Ret>,
) -> Program<Args::Interval, Ret::Interval>
where
    Args: IntervalType,
    Ret: IntervalType,
{
    Program::new(translate(program.tape()))
}

fn translate(source: &Tape) -> Tape {
    _ = source;
    todo!("interval translation")
}
