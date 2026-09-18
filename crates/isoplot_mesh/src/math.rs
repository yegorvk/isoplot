//! A small SIMD-agnostic math "library"

use fearless_simd::{Simd, SimdFrom, SimdInto};
use std::{f64, iter, marker::PhantomData, ops};

use crate::simd::{f32s, i32s, mask32s, u32s};

mod seal {
    #[doc(hidden)]
    pub trait Seal {}
}

use seal::Seal;

macro_rules! scalar_seal_impls {
    ($($ty:ty),* $(,)?) => {
        $( impl Seal for $ty {} )*
    };
}

scalar_seal_impls!(
    bool, u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64,
);

macro_rules! simd_seal_impls {
    ($($ty:ident),* $(,)?) => {
        $(
            impl<S: Simd> Seal for $ty<S> {}
        )*
    };
}

simd_seal_impls!(mask32s, u32s, i32s, f32s);

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
impl<S: Simd> Mask for mask32s<S> {}

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

    /// Factory type that produced this number
    type Factory: NumberFactory<Self>;

    /// Returns the corresponding factory instance.
    fn factory(self) -> Self::Factory;

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
}

pub trait Real: Seal + Number + ops::Div<Output = Self> + ops::DivAssign {
    // /// Returns `self / rhs` when `rhs` is not zero; zero otherwise.
    // fn div_or_zero(self, rhs: Self) -> Self;

    /// Returns the square root of this element.
    fn sqrt(self) -> Self;
}

pub trait ConstZero: Seal + Number {
    const ZERO: Self;
}

pub trait ConstOne: Seal + Number {
    const ONE: Self;
}

macro_rules! const_zero_one_impls {
    ($($ty:ty),* $(,)?) => {
        $(
            impl ConstZero for $ty {
                const ZERO: Self = 0 as $ty;
            }

            impl ConstOne for $ty {
                const ONE: Self = 1 as $ty;
            }
        )*
    };
}

const_zero_one_impls!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);

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

pub trait NumberFactory<T>: Seal + Copy {
    /// Return the `0` element.
    fn zero(self) -> T;

    /// Return the `1` element.
    fn one(self) -> T;
}

#[derive(Copy, Clone)]
pub struct ConstFactory;

impl Seal for ConstFactory {}

impl<T> NumberFactory<T> for ConstFactory
where
    T: ConstZero + ConstOne,
{
    #[inline(always)]
    fn zero(self) -> T {
        T::ZERO
    }

    #[inline(always)]
    fn one(self) -> T {
        T::ONE
    }
}

macro_rules! scalar_number_impls {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Number for $ty {
                type Factory = ConstFactory;
                type Mask = bool;

                fn factory(self) -> Self::Factory {
                    ConstFactory
                }

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

scalar_number_impls!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);
scalar_real_impls!(f32, f64);

#[derive(Copy, Clone)]
pub struct SimdFactory<S: Simd>(S);

impl<S: Simd> SimdFactory<S> {
    pub const fn new(simd: S) -> Self {
        Self(simd)
    }
}

impl<S: Simd> From<S> for SimdFactory<S> {
    #[inline(always)]
    fn from(value: S) -> Self {
        Self::new(value)
    }
}

impl<S: Simd> Seal for SimdFactory<S> {}

impl<S: Simd> NumberFactory<u32s<S>> for SimdFactory<S> {
    #[inline(always)]
    fn zero(self) -> u32s<S> {
        u32s::splat(self.0, 0)
    }

    #[inline(always)]
    fn one(self) -> u32s<S> {
        u32s::splat(self.0, 1)
    }
}

impl<S: Simd> NumberFactory<i32s<S>> for SimdFactory<S> {
    #[inline(always)]
    fn zero(self) -> i32s<S> {
        i32s::splat(self.0, 0)
    }

    #[inline(always)]
    fn one(self) -> i32s<S> {
        i32s::splat(self.0, 1)
    }
}

impl<S: Simd> NumberFactory<f32s<S>> for SimdFactory<S> {
    #[inline(always)]
    fn zero(self) -> f32s<S> {
        f32s::splat(self.0, 0f32)
    }

    #[inline(always)]
    fn one(self) -> f32s<S> {
        f32s::splat(self.0, 1f32)
    }
}

macro_rules! simd_number_impls {
    ($(($simd_ty:ident, $lane_ty:ty, $mask_ty:ident)),* $(,)?) => {
        $(
            impl<S: Simd> Number for $simd_ty<S> {
                type Factory = SimdFactory<S>;
                type Mask = $mask_ty<S>;

                #[inline(always)]
                fn factory(self) -> Self::Factory {
                    self.witness().into()
                }

                #[inline(always)]
                fn eq(self, rhs: Self) -> Self::Mask {
                    <$simd_ty<S>>::simd_eq(self, rhs)
                }

                #[inline(always)]
                fn ne(self, rhs: Self) -> Self::Mask {
                    <$simd_ty<S>>::simd_ne(self, rhs)
                }

                #[inline(always)]
                fn lt(self, rhs: Self) -> Self::Mask {
                    <$simd_ty<S>>::simd_lt(self, rhs)
                }

                #[inline(always)]
                fn le(self, rhs: Self) -> Self::Mask {
                    <$simd_ty<S>>::simd_le(self, rhs)
                }

                #[inline(always)]
                fn gt(self, rhs: Self) -> Self::Mask {
                    <$simd_ty<S>>::simd_gt(self, rhs)
                }

                #[inline(always)]
                fn ge(self, rhs: Self) -> Self::Mask {
                    <$simd_ty<S>>::simd_ge(self, rhs)
                }

                #[inline(always)]
                fn min(self, rhs: Self) -> Self {
                    <$simd_ty<S>>::min(self, rhs)
                }

                #[inline(always)]
                fn max(self, rhs: Self) -> Self {
                    <$simd_ty<S>>::max(self, rhs)
                }

                #[inline(always)]
                fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
                    <$simd_ty<S>>::select(mask, if_true, if_false)
                }
            }
        )*
    };
}

simd_number_impls!(
    (u32s, u32, mask32s),
    (i32s, i32, mask32s),
    (f32s, f32, mask32s)
);

impl<S: Simd> Real for f32s<S> {
    #[inline(always)]
    fn sqrt(self) -> Self {
        <f32s<S>>::sqrt(self)
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

impl<T> Vec3<T> {
    #[inline(always)]
    pub(crate) fn simd_new<S: Simd, U>(simd: S, x: U, y: U, z: U) -> Self
    where
        U: SimdInto<T, S>,
    {
        Self {
            x: x.simd_into(simd),
            y: y.simd_into(simd),
            z: z.simd_into(simd),
        }
    }

    #[inline(always)]
    pub(crate) fn simd_splat<S: Simd, U>(simd: S, value: U) -> Self
    where
        U: Copy + SimdInto<T, S>,
    {
        Self::simd_new(simd, value, value, value)
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

impl<T: Mask> Vec3<T> {
    #[inline(always)]
    pub fn fold_and(self) -> T {
        self.x & self.y & self.z
    }

    #[inline(always)]
    pub fn fold_or(self) -> T {
        self.x | self.y | self.z
    }
}

impl<T: Number> Vec3<T> {
    #[inline(always)]
    pub(crate) fn zero<Factory>(field: Factory) -> Self
    where
        Factory: NumberFactory<T>,
    {
        Self::splat(Factory::zero(field))
    }

    #[inline(always)]
    pub(crate) fn one<Factory>(field: Factory) -> Self
    where
        Factory: NumberFactory<T>,
    {
        Self::splat(Factory::one(field))
    }

    #[inline(always)]
    pub(crate) fn x<Factory>(field: Factory) -> Self
    where
        Factory: NumberFactory<T>,
    {
        Self::new(
            Factory::one(field),
            Factory::zero(field),
            Factory::zero(field),
        )
    }

    #[inline(always)]
    pub(crate) fn y<F>(factory: F) -> Self
    where
        F: NumberFactory<T>,
    {
        Self::new(F::zero(factory), F::one(factory), F::zero(factory))
    }

    #[inline(always)]
    pub(crate) fn z<F>(factory: F) -> Self
    where
        F: NumberFactory<T>,
    {
        Self::new(F::zero(factory), F::zero(factory), F::one(factory))
    }

    #[inline(always)]
    pub fn eq_mask(self, rhs: Self) -> Vec3<T::Mask> {
        Vec3 {
            x: self.x.eq(rhs.x),
            y: self.y.eq(rhs.y),
            z: self.z.eq(rhs.z),
        }
    }

    #[inline(always)]
    pub fn ne_mask(self, rhs: Self) -> Vec3<T::Mask> {
        Vec3 {
            x: self.x.ne(rhs.x),
            y: self.y.ne(rhs.y),
            z: self.z.ne(rhs.z),
        }
    }

    #[inline(always)]
    pub fn lt_mask(self, rhs: Self) -> Vec3<T::Mask> {
        Vec3 {
            x: self.x.lt(rhs.x),
            y: self.y.lt(rhs.y),
            z: self.z.lt(rhs.z),
        }
    }

    #[inline(always)]
    pub fn le_mask(self, rhs: Self) -> Vec3<T::Mask> {
        Vec3 {
            x: self.x.le(rhs.x),
            y: self.y.le(rhs.y),
            z: self.z.le(rhs.z),
        }
    }

    #[inline(always)]
    pub fn gt_mask(self, rhs: Self) -> Vec3<T::Mask> {
        Vec3 {
            x: self.x.gt(rhs.x),
            y: self.y.gt(rhs.y),
            z: self.z.gt(rhs.z),
        }
    }

    #[inline(always)]
    pub fn ge_mask(self, rhs: Self) -> Vec3<T::Mask> {
        Vec3 {
            x: self.x.ge(rhs.x),
            y: self.y.ge(rhs.y),
            z: self.z.ge(rhs.z),
        }
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
        self.map(|v| {
            let zero = v.factory().zero();
            T::select(norm.eq(zero), zero, v / norm)
        })
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
