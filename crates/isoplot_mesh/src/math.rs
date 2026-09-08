//! A small SIMD-agnostic math "library"

use fearless_simd::{
    Simd, SimdBase, SimdFloat, f32x4, f32x8, f32x16, f64x2, f64x4, f64x8, mask32x4, mask32x8,
    mask32x16, mask64x2, mask64x4, mask64x8,
};
use std::{f64, iter, marker::PhantomData, ops};

mod seal {
    #[doc(hidden)]
    pub trait Seal {}
}

use seal::Seal;

impl Seal for bool {}

impl Seal for u8 {}
impl Seal for u16 {}
impl Seal for u32 {}
impl Seal for u64 {}
impl Seal for usize {}
impl Seal for i8 {}
impl Seal for i16 {}
impl Seal for i32 {}
impl Seal for i64 {}
impl Seal for isize {}
impl Seal for f32 {}
impl Seal for f64 {}

impl Seal for Axis {}

impl<S: Simd> Seal for mask32x4<S> {}
impl<S: Simd> Seal for mask32x8<S> {}
impl<S: Simd> Seal for mask32x16<S> {}

impl<S: Simd> Seal for mask64x2<S> {}
impl<S: Simd> Seal for mask64x4<S> {}
impl<S: Simd> Seal for mask64x8<S> {}

impl<S: Simd> Seal for f32x4<S> {}
impl<S: Simd> Seal for f32x8<S> {}
impl<S: Simd> Seal for f32x16<S> {}

impl<S: Simd> Seal for f64x2<S> {}
impl<S: Simd> Seal for f64x4<S> {}
impl<S: Simd> Seal for f64x8<S> {}

impl Seal for R<f32> {}
impl Seal for R<f64> {}

impl<S: Simd> Seal for RxS<S, f32> {}
impl<S: Simd> Seal for RxS<S, f64> {}

pub trait Mask:
    Seal
    + Copy
    + ops::Not<Output = Self>
    + ops::BitAnd<Output = Self>
    + ops::BitAndAssign
    + ops::BitOr<Output = Self>
    + ops::BitOrAssign
    + ops::BitXor<Output = Self>
    + ops::BitXorAssign
{
}

impl Mask for bool {}

impl<S: Simd> Mask for mask32x4<S> {}
impl<S: Simd> Mask for mask32x8<S> {}
impl<S: Simd> Mask for mask32x16<S> {}

impl<S: Simd> Mask for mask64x2<S> {}
impl<S: Simd> Mask for mask64x4<S> {}
impl<S: Simd> Mask for mask64x8<S> {}

pub trait Number:
    Seal
    + Copy
    + ops::Add<Output = Self>
    + ops::AddAssign
    + ops::Sub<Output = Self>
    + ops::SubAssign
    + ops::Mul<Output = Self>
    + ops::MulAssign
{
    /// Boolean mask type
    type Mask: Mask;

    /// Returns if `self` is equal to `rhs`.
    fn eq(self, rhs: Self) -> Self::Mask;

    /// Returns if `self` is not equal to `rhs`.
    fn ne(self, rhs: Self) -> Self::Mask;

    /// Returns if `self` is less than `rhs`.
    fn lt(self, rhs: Self) -> Self::Mask;

    /// Returns if `self` is less than or equal to `rhs`.
    fn le(self, rhs: Self) -> Self::Mask;

    /// Returns if `self` is greater than `rhs`.
    fn gt(self, rhs: Self) -> Self::Mask;

    /// Returns if `self` is greater than or equal to `rhs`.
    fn ge(self, rhs: Self) -> Self::Mask;

    /// Returns the minimum of two numbers.
    fn min(self, rhs: Self) -> Self;

    /// Returns the maximum of two numbers.
    fn max(self, rhs: Self) -> Self;

    /// Selects between two numbers using `mask`.
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;

    /// Returns `self / rhs` when `rhs` is not zero; zero otherwise.
    fn div_or_zero(self, rhs: Self) -> Self;
}

pub trait Real: Seal + Number + ops::Div<Output = Self> + ops::DivAssign {
    /// Returns the square root of this element.
    fn sqrt(self) -> Self;
}

pub trait ConstZero: Seal + Number {
    const ZERO: Self;
}

impl ConstZero for f32 {
    const ZERO: Self = 0f32;
}

impl ConstZero for f64 {
    const ZERO: Self = 0f64;
}

pub trait ConstOne: Seal + Number {
    const ONE: Self;
}

impl ConstOne for f32 {
    const ONE: Self = 1f32;
}

impl ConstOne for f64 {
    const ONE: Self = 1f64;
}

pub trait Cast<T>: Seal {
    fn cast(self) -> T;
}

impl<T: Seal> Cast<T> for T {
    #[inline(always)]
    fn cast(self) -> T {
        self
    }
}

macro_rules! scalar_cast_impls {
    ($($from:ty => [$($to:ty),* $(,)?]),* $(,)?) => {
        $(
            $(
                impl Cast<$to> for $from {
                    #[inline(always)]
                    fn cast(self) -> $to {
                        self as $to
                    }
                }
            )*
        )*
    };
}

scalar_cast_impls! {
    // `u*` types
    u8 => [u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64],
    u16 => [u8, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64],
    u32 => [u8, u16, u64, usize, i8, i16, i32, i64, isize, f32, f64],
    u64 => [u8, u16, u32, usize, i8, i16, i32, i64, isize, f32, f64],
    usize => [u8, u16, u32, u64, i8, i16, i32, i64, isize, f32, f64],

    // `i*` types
    i8 => [u8, u16, u32, u64, usize, i16, i32, i64, isize, f32, f64],
    i16 => [u8, u16, u32, u64, usize, i8, i32, i64, isize, f32, f64],
    i32 => [u8, u16, u32, u64, usize, i8, i16, i64, isize, f32, f64],
    i64 => [u8, u16, u32, u64, usize, i8, i16, i32, f32, isize, f64],
    isize => [u8, u16, u32, u64, usize, i8, i16, i32, i64, f32, f64],

    // `f*` types
    f32 => [u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f64],
    f64 => [u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32],
}

impl<T> Cast<T> for bool
where
    u8: Cast<T>,
{
    #[inline(always)]
    fn cast(self) -> T {
        (self as u8).cast()
    }
}

pub(crate) trait NumberField: Seal + Copy {
    /// Field element type
    type Element: Number;

    /// Return the `0` element.
    fn zero(self) -> Self::Element;

    /// Return the `1` element.
    fn one(self) -> Self::Element;
}

macro_rules! scalar_number_impls {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Number for $ty {
                type Mask = bool;

                #[inline(always)]
                fn eq(self, rhs: Self) -> Self::Mask {
                    self == rhs
                }

                #[inline(always)]
                fn ne(self, rhs: Self) -> Self::Mask {
                    self != rhs
                }

                #[inline(always)]
                fn lt(self, rhs: Self) -> Self::Mask {
                    self < rhs
                }

                #[inline(always)]
                fn le(self, rhs: Self) -> Self::Mask {
                    self <= rhs
                }

                #[inline(always)]
                fn gt(self, rhs: Self) -> Self::Mask {
                    self > rhs
                }

                #[inline(always)]
                fn ge(self, rhs: Self) -> Self::Mask {
                    self >= rhs
                }

                #[inline(always)]
                fn min(self, rhs: Self) -> Self {
                    ScalarDispatch::<$ty>::min(self, rhs)
                }

                #[inline(always)]
                fn max(self, rhs: Self) -> Self {
                    ScalarDispatch::<$ty>::max(self, rhs)
                }

                #[inline(always)]
                fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
                    std::hint::select_unpredictable(mask, if_true, if_false)
                }

                #[inline(always)]
                fn div_or_zero(self, rhs: Self) -> Self {
                    ScalarDispatch::<$ty>::div_or_zero(self, rhs)
                }
            }
        )*
    };
}

struct ScalarDispatch<T>(PhantomData<fn() -> T>);

macro_rules! scalar_dispatch_int {
    ($($ty:ty),* $(,)?) => {
        $(
            #[allow(dead_code)]
            impl ScalarDispatch<$ty> {
                #[inline(always)]
                fn min(a: $ty, b: $ty) -> $ty {
                    std::cmp::min(a, b)
                }

                #[inline(always)]
                fn max(a: $ty, b: $ty) -> $ty {
                    std::cmp::max(a, b)
                }

                #[inline(always)]
                fn div_or_zero(a: $ty, b: $ty) -> $ty {
                    a.checked_div(b).unwrap_or(0)
                }
            }
        )*
    };
}

scalar_dispatch_int!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

impl ScalarDispatch<f32> {
    #[inline(always)]
    fn min(a: f32, b: f32) -> f32 {
        a.min(b)
    }

    #[inline(always)]
    fn max(a: f32, b: f32) -> f32 {
        a.max(b)
    }

    #[inline(always)]
    fn div_or_zero(a: f32, b: f32) -> f32 {
        let c = a / b;
        f32::select(c.is_finite(), c, 0.0)
    }
}

impl ScalarDispatch<f64> {
    #[inline(always)]
    fn min(a: f64, b: f64) -> f64 {
        a.min(b)
    }

    #[inline(always)]
    fn max(a: f64, b: f64) -> f64 {
        a.max(b)
    }

    #[inline(always)]
    fn div_or_zero(a: f64, b: f64) -> f64 {
        let c = a / b;
        f64::select(c.is_finite(), c, 0.0)
    }
}

macro_rules! scalar_real_impls {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Real for $ty {
                #[inline(always)]
                fn sqrt(self) -> Self {
                    <$ty>::sqrt(self)
                }
            }
        )*
    };
}

scalar_number_impls!(u8, u16, u32, u64, f32, f64);
scalar_real_impls!(f32, f64);

macro_rules! simd_number_impls {
    ($([$simd_ty:ident, $mask_ty:ident]),* $(,)?) => {
        $(
            impl<S: Simd> Number for $simd_ty<S> {
                type Mask = $mask_ty<S>;

                #[inline(always)]
                fn eq(self, rhs: Self) -> Self::Mask {
                    SimdBase::simd_eq(self, rhs)
                }

                #[inline(always)]
                fn ne(self, rhs: Self) -> Self::Mask {
                    !SimdBase::simd_eq(self, rhs)
                }

                #[inline(always)]
                fn lt(self, rhs: Self) -> Self::Mask {
                    SimdBase::simd_lt(self, rhs)
                }

                #[inline(always)]
                fn le(self, rhs: Self) -> Self::Mask {
                    SimdBase::simd_le(self, rhs)
                }

                #[inline(always)]
                fn gt(self, rhs: Self) -> Self::Mask {
                    SimdBase::simd_gt(self, rhs)
                }

                #[inline(always)]
                fn ge(self, rhs: Self) -> Self::Mask {
                    SimdBase::simd_ge(self, rhs)
                }

                #[inline(always)]
                fn min(self, rhs: Self) -> Self {
                    SimdBase::min(self, rhs)
                }

                #[inline(always)]
                fn max(self, rhs: Self) -> Self {
                    SimdBase::max(self, rhs)
                }

                #[inline(always)]
                fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
                    use fearless_simd::Select;
                    <$mask_ty<S>>::select(mask, if_true, if_false)
                }

                #[inline(always)]
                fn div_or_zero(self, rhs: Self) -> Self {
                    let zero = SimdBase::splat(rhs.simd, 0 as <$simd_ty<S> as SimdBase<S>>::Element);
                    Self::select(rhs.ne(zero), self / rhs, zero)
                }
            }
        )*
    };
}

macro_rules! simd_real_impls {
    ($($simd_ty:ident),* $(,)?) => {
        $(
            impl<S: Simd> Real for $simd_ty<S> {
                #[inline(always)]
                fn sqrt(self) -> Self {
                    SimdFloat::sqrt(self)
                }
            }
        )*
    };
}

simd_number_impls! {
    // `f32`
    [f32x4, mask32x4],
    [f32x8, mask32x8],
    [f32x16, mask32x16],

    // `f64`
    [f64x2, mask64x2],
    [f64x4, mask64x4],
    [f64x8, mask64x8],
}

simd_real_impls! {
    // `f32`
    f32x4, f32x8, f32x16,
    // `f64`
    f64x2, f64x4, f64x8,
}

pub(crate) struct R<T>(PhantomData<fn() -> T>);

impl<T> R<T>
where
    Self: NumberField,
{
    #[inline(always)]
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T> NumberField for R<T>
where
    Self: Seal,
    T: Real + ConstZero + ConstOne,
{
    type Element = T;

    #[inline(always)]
    fn zero(self) -> Self::Element {
        T::ZERO
    }

    #[inline(always)]
    fn one(self) -> Self::Element {
        T::ONE
    }
}

impl<T> Clone for R<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for R<T> {}

pub(crate) struct RxS<S, T> {
    simd: S,
    _marker: PhantomData<fn() -> T>,
}

impl<S: Simd> NumberField for RxS<S, f32>
where
    S::f32s: Real,
{
    type Element = S::f32s;

    #[inline(always)]
    fn zero(self) -> Self::Element {
        SimdBase::splat(self.simd, 0f32)
    }

    #[inline(always)]
    fn one(self) -> Self::Element {
        SimdBase::splat(self.simd, 1f32)
    }
}

impl<S: Simd> NumberField for RxS<S, f64>
where
    S::f64s: Real,
{
    type Element = S::f64s;

    #[inline(always)]
    fn zero(self) -> Self::Element {
        SimdBase::splat(self.simd, 0f64)
    }

    #[inline(always)]
    fn one(self) -> Self::Element {
        SimdBase::splat(self.simd, 1f64)
    }
}

impl<S: Copy, T> Clone for RxS<S, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: Copy, T> Copy for RxS<S, T> {}

#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum Axis {
    X = 0,
    Y = 1,
    Z = 2,
}

impl<T> Cast<T> for Axis
where
    u8: Cast<T>,
{
    #[inline(always)]
    fn cast(self) -> T {
        (self as u8).cast()
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(C)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Vec3<T> {
    #[inline(always)]
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    #[inline(always)]
    fn from_array(a: [T; 3]) -> Self {
        let [x, y, z] = a;
        Self { x, y, z }
    }

    #[inline(always)]
    pub fn to_array(self) -> [T; 3] {
        [self.x, self.y, self.z]
    }

    #[inline(always)]
    pub fn map<F, B>(self, mut f: F) -> Vec3<B>
    where
        F: FnMut(T) -> B,
    {
        Vec3 {
            x: f(self.x),
            y: f(self.y),
            z: f(self.z),
        }
    }

    #[inline(always)]
    pub fn cast<B>(self) -> Vec3<B>
    where
        T: Cast<B>,
    {
        self.map(|x| x.cast())
    }

    #[inline(always)]
    pub fn any<F>(self, mut f: F) -> bool
    where
        F: FnMut(T) -> bool,
    {
        f(self.x) || f(self.y) || f(self.z)
    }

    #[inline(always)]
    pub fn all<F>(self, mut f: F) -> bool
    where
        F: FnMut(T) -> bool,
    {
        f(self.x) && f(self.y) && f(self.z)
    }
}

impl<T> From<[T; 3]> for Vec3<T> {
    #[inline(always)]
    fn from(value: [T; 3]) -> Self {
        Self::from_array(value)
    }
}

impl<T> From<Vec3<T>> for [T; 3] {
    #[inline(always)]
    fn from(value: Vec3<T>) -> Self {
        value.to_array()
    }
}

impl<T: Copy> Vec3<T> {
    #[inline(always)]
    pub const fn splat(value: T) -> Self {
        Self {
            x: value,
            y: value,
            z: value,
        }
    }
}

impl<T, Idx> ops::Index<Idx> for Vec3<T>
where
    Idx: Cast<usize>,
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: Idx) -> &Self::Output {
        match index.cast() {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index must be between 0 and 2"),
        }
    }
}

impl<T: ConstZero> Vec3<T> {
    pub const ZERO: Self = Vec3::splat(T::ZERO);
}

impl<T: ConstOne> Vec3<T> {
    pub const ONE: Self = Vec3::splat(T::ONE);
}

impl<T: ConstZero + ConstOne> Vec3<T> {
    pub const X: Self = Self::new(T::ONE, T::ZERO, T::ZERO);
    pub const Y: Self = Self::new(T::ZERO, T::ONE, T::ZERO);
    pub const Z: Self = Self::new(T::ZERO, T::ZERO, T::ONE);
}

impl<T: Number> Vec3<T> {
    #[inline(always)]
    pub(crate) fn zero<F>(field: F) -> Self
    where
        F: NumberField<Element = T>,
    {
        Self::splat(F::zero(field))
    }

    #[inline(always)]
    pub(crate) fn one<F>(field: F) -> Self
    where
        F: NumberField<Element = T>,
    {
        Self::splat(F::one(field))
    }

    #[inline(always)]
    pub(crate) fn x<F>(field: F) -> Self
    where
        F: NumberField<Element = T>,
    {
        Self::new(F::one(field), F::zero(field), F::zero(field))
    }

    #[inline(always)]
    pub(crate) fn y<F>(field: F) -> Self
    where
        F: NumberField<Element = T>,
    {
        Self::new(F::zero(field), F::one(field), F::zero(field))
    }

    #[inline(always)]
    pub(crate) fn z<F>(field: F) -> Self
    where
        F: NumberField<Element = T>,
    {
        Self::new(F::zero(field), F::zero(field), F::one(field))
    }

    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        Self {
            x: self.x.max(rhs.x),
            y: self.y.max(rhs.y),
            z: self.z.max(rhs.z),
        }
    }

    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        Self {
            x: self.x.min(rhs.x),
            y: self.y.min(rhs.y),
            z: self.z.min(rhs.z),
        }
    }

    #[inline(always)]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self {
            x: self.x.max(min.x).min(max.x),
            y: self.y.max(min.y).min(max.y),
            z: self.z.max(min.z).min(max.z),
        }
    }

    #[inline(always)]
    pub fn norm_squared(self) -> T {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> T {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }
}

impl<T: Real> Vec3<T> {
    #[inline(always)]
    pub fn norm(self) -> T {
        self.norm_squared().sqrt()
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        self / self.norm()
    }

    #[inline(always)]
    pub fn normalize_or_zero(self) -> Self {
        let norm = self.norm();
        self.map(|v| v.div_or_zero(norm))
    }
}

impl<T> iter::Sum for Vec3<T>
where
    T: Number + ConstZero,
{
    #[inline(always)]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, v| acc + v)
    }
}

macro_rules! imm_binop_impls {
    ($(($trait:ident, $method:ident)),* $(,)?) => {
        $(
            impl<T: ops::$trait> ops::$trait for Vec3<T> {
                type Output = Vec3<<T as ops::$trait>::Output>;

                #[inline(always)]
                fn $method(self, rhs: Self) -> Self::Output {
                    Vec3 {
                        x: <T as ops::$trait>::$method(self.x, rhs.x),
                        y: <T as ops::$trait>::$method(self.y, rhs.y),
                        z: <T as ops::$trait>::$method(self.z, rhs.z),
                    }
                }
            }

            impl<T: Copy + ops::$trait> ops::$trait<T> for Vec3<T> {
                type Output = Vec3<<T as ops::$trait<T>>::Output>;

                #[inline(always)]
                fn $method(self, rhs: T) -> Self::Output {
                    <Vec3<T> as ops::$trait>::$method(self, Vec3::splat(rhs))
                }
            }
        )*
    };
}

macro_rules! mut_binop_impls {
    ($(($trait:ident, $method:ident)),* $(,)?) => {
        $(
            impl<T: ops::$trait> ops::$trait for Vec3<T> {
                #[inline(always)]
                fn $method(&mut self, rhs: Self) {
                    <T as ops::$trait>::$method(&mut self.x, rhs.x);
                    <T as ops::$trait>::$method(&mut self.y, rhs.y);
                    <T as ops::$trait>::$method(&mut self.z, rhs.z);
                }
            }

            impl<T: Copy + ops::$trait> ops::$trait<T> for Vec3<T> {
                #[inline(always)]
                fn $method(&mut self, rhs: T) {
                    <Vec3<T> as ops::$trait>::$method(self, Vec3::splat(rhs))
                }
            }
        )*
    };
}

imm_binop_impls!((Add, add), (Sub, sub), (Mul, mul), (Div, div));

mut_binop_impls! {
    (AddAssign, add_assign), (SubAssign, sub_assign),
    (MulAssign, mul_assign), (DivAssign, div_assign)
}

macro_rules! scalar_rev_binop_impls {
    ($($ty:ty => [$(($trait:ident, $method:ident)),* $(,)?]),* $(,)?) => {
        $(
            $(
                impl ops::$trait<Vec3<$ty>> for $ty {
                    type Output = Vec3<$ty>;

                    #[inline(always)]
                    fn $method(self, rhs: Vec3<$ty>) -> Self::Output {
                        <Vec3<$ty> as ops::$trait<$ty>>::$method(rhs, self)
                    }
                }
            )*
        )*
    };
}

macro_rules! simd_rev_binop_impls {
    ($($ty:ident => [$(($trait:ident, $method:ident, $op:tt)),* $(,)?]),* $(,)?) => {
        $(
            $(
                impl<S: Simd> ops::$trait<Vec3<$ty<S>>> for $ty<S> {
                    type Output = Vec3<$ty<S>>;

                    #[inline(always)]
                    fn $method(self, rhs: Vec3<$ty<S>>) -> Self::Output {
                        <Vec3<$ty<S>> as ops::$trait<$ty<S>>>::$method(rhs, self)
                    }
                }
            )*
        )*
    };
}

scalar_rev_binop_impls! {
    f32 => [(Add, add), (Sub, sub), (Mul, mul), (Div, div)],
    f64 => [(Add, add), (Sub, sub), (Mul, mul), (Div, div)],
}

simd_rev_binop_impls! {
    // `f32`
    f32x4 => [(Add, add, +), (Sub, sub, -), (Mul, mul, *), (Div, div, /)],
    f32x8 => [(Add, add, +), (Sub, sub, -), (Mul, mul, *), (Div, div, /)],
    f32x16 => [(Add, add, +), (Sub, sub, -), (Mul, mul, *), (Div, div, /)],

    // `f64`
    f64x2 => [(Add, add, +), (Sub, sub, -), (Mul, mul, *), (Div, div, /)],
    f64x4 => [(Add, add, +), (Sub, sub, -), (Mul, mul, *), (Div, div, /)],
    f64x8 => [(Add, add, +), (Sub, sub, -), (Mul, mul, *), (Div, div, /)],
}
