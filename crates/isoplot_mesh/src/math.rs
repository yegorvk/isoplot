use std::{fmt::Debug, iter, marker::PhantomData, ops};

use bytemuck::{AnyBitPattern, NoUninit, Zeroable, must_cast_mut, must_cast_ref};
use derive_where::derive_where;
use fearless_simd::{ExtractToken, Select, Simd, SimdBase, SimdFloat, SimdInto, SimdMask};

#[inline(always)]
pub(crate) fn u32s_splat<S: Simd>(simd: S, value: u32) -> S::u32s {
    S::u32s::splat(simd, value)
}

#[inline(always)]
pub(crate) fn i32s_splat<S: Simd>(simd: S, value: i32) -> S::i32s {
    S::i32s::splat(simd, value)
}

#[inline(always)]
pub(crate) fn f32s_splat<S: Simd>(simd: S, value: f32) -> S::f32s {
    S::f32s::splat(simd, value)
}

mod seal_number {
    pub trait SealNumber<Token> {}
}

mod seal_mask {
    pub trait SealMask<Token> {}
}

use seal_mask::SealMask;
use seal_number::SealNumber;

#[derive(Copy, Clone, Debug)]
pub struct ScalarToken;

macro_rules! seal {
    ($($trait:ident => [ $($ty:ty),* ]),*) => {
        $( $( impl $trait<ScalarToken> for $ty {} )* )*
    };
}

seal! {
    SealNumber => [u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64],
    SealMask => [bool]
}

pub trait Mask<Token>:
    SealMask<Token>
    + Copy
    + ops::Not<Output = Self>
    + ops::BitAnd<Output = Self>
    + ops::BitOr<Output = Self>
    + ops::BitXor<Output = Self>
{
}

impl Mask<ScalarToken> for bool {}

pub trait Number<Token: Copy>:
    SealNumber<Token>
    + Copy
    + ops::Add<Output = Self>
    + ops::Sub<Output = Self>
    + ops::Mul<Output = Self>
{
    /// The mask type used for conditional execution
    type Mask: Mask<Token>;

    /// Returns this number's token.
    fn token(self) -> Token;

    /// Returns the `0` value.
    fn zero(token: Token) -> Self;

    /// Returns the `1` value.
    fn one(token: Token) -> Self;

    /// Returns a mask indicating whether `self` is equal to `rhs`.
    fn eq(self, rhs: Self) -> Self::Mask;

    /// Returns a mask indicating whether `self` is not equal to `rhs`.
    fn ne(self, rhs: Self) -> Self::Mask;

    /// Returns a mask indicating whether `self` is less than `rhs`.
    fn lt(self, rhs: Self) -> Self::Mask;

    /// Returns a mask indicating whether `self` is less than or equal to `rhs`.
    fn le(self, rhs: Self) -> Self::Mask;

    /// Returns a mask indicating whether `self` is greater than `rhs`.
    fn gt(self, rhs: Self) -> Self::Mask;

    /// Returns a mask indicating whether `self` is greater than or equal to `rhs`.
    fn ge(self, rhs: Self) -> Self::Mask;

    /// Returns the minimum of two numbers.
    fn min(self, rhs: Self) -> Self;

    /// Returns the maximum of two numbers.
    fn max(self, rhs: Self) -> Self;

    /// Selects between two numbers using `mask`.
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;
}

pub trait Signed<Token: Copy>: Number<Token> + ops::Neg {
    /// Computes the absolute value of this number.
    fn abs(self) -> Self;
}

pub trait Real<Token: Copy>: Number<Token> + ops::Div<Output = Self> {
    /// Returns a number with the magnitude of `self` and the sign of `rhs`.
    fn copysign(self, rhs: Self) -> Self;

    /// Computes an approximate value of `1.0 / self`.
    fn approximate_recip(self) -> Self;

    /// Returns the square root of this element.
    fn sqrt(self) -> Self;
}

pub trait WithConstToken: Number<ScalarToken> {
    /// The `0` value
    const ZERO: Self;

    /// The `1` value
    const ONE: Self;
}

macro_rules! impl_const_token {
    ($($ty:ty),*) => {
        $(
            impl WithConstToken for $ty {
                const ZERO: $ty = 0 as $ty;
                const ONE: $ty = 1 as $ty;
            }
        )*
    };
}

impl_const_token!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);

struct DispatchNumber<T> {
    _marker: PhantomData<fn() -> T>,
}

macro_rules! dispatch_int {
    ($($ty:ty),*) => {
        $(
            #[allow(dead_code)]
            impl DispatchNumber<$ty> {
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

dispatch_int!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

impl DispatchNumber<f32> {
    #[inline(always)]
    fn min(a: f32, b: f32) -> f32 {
        a.min(b)
    }

    #[inline(always)]
    fn max(a: f32, b: f32) -> f32 {
        a.max(b)
    }
}

impl DispatchNumber<f64> {
    #[inline(always)]
    fn min(a: f64, b: f64) -> f64 {
        a.min(b)
    }

    #[inline(always)]
    fn max(a: f64, b: f64) -> f64 {
        a.max(b)
    }
}

macro_rules! impl_number {
    ($($ty:ty),*) => {
        $(
            impl Number<ScalarToken> for $ty {
                type Mask = bool;

                #[inline(always)]
                fn token(self) -> ScalarToken {
                    ScalarToken
                }

                #[inline(always)]
                fn zero(_token: ScalarToken) -> Self {
                    0 as $ty
                }

                #[inline(always)]
                fn one(_token: ScalarToken) -> Self {
                    1 as $ty
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
                    DispatchNumber::<$ty>::min(self, rhs)
                }

                #[inline(always)]
                fn max(self, rhs: Self) -> Self {
                    DispatchNumber::<$ty>::max(self, rhs)
                }

                #[inline(always)]
                fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
                    std::hint::select_unpredictable(mask, if_true, if_false)
                }
            }
        )*
    };
}

impl_number!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);

macro_rules! impl_signed {
    ($($ty:ty),*) => {
        $(
            impl Signed<ScalarToken> for $ty {
                #[inline(always)]
                fn abs(self) -> Self {
                    <$ty>::abs(self)
                }
            }
        )*
    };
}

impl_signed!(i8, i16, i32, i64, isize, f32, f64);

macro_rules! impl_real {
    ($($ty:ty),*) => {
        $(
            impl Real<ScalarToken> for $ty {
                #[inline(always)]
                fn copysign(self, rhs: Self) -> Self {
                    <$ty>::copysign(self, rhs)
                }

                #[inline(always)]
                fn approximate_recip(self) -> Self {
                    <$ty>::recip(self)
                }

                #[inline(always)]
                fn sqrt(self) -> Self {
                    <$ty>::sqrt(self)
                }
            }
        )*
    };
}

impl_real!(f32, f64);

#[derive(Copy, Clone)]
pub struct SimdToken<S: Simd>(S);

impl<S: Simd> SimdToken<S> {
    pub const fn new(simd: S) -> Self {
        Self(simd)
    }
}

impl<S: Simd> From<S> for SimdToken<S> {
    #[inline]
    fn from(value: S) -> Self {
        Self(value)
    }
}

impl<S: Simd, V: SimdMask<S>> SealMask<SimdToken<S>> for V {}
impl<S: Simd, V: SimdMask<S>> Mask<SimdToken<S>> for V {}

impl<S: Simd, V: SimdBase<S>> SealNumber<SimdToken<S>> for V {}

impl<S: Simd, V: SimdBase<S>> Number<SimdToken<S>> for V {
    type Mask = V::Mask;

    #[inline]
    fn token(self) -> SimdToken<S> {
        ExtractToken::token(&self).into()
    }

    #[inline(always)]
    fn zero(token: SimdToken<S>) -> Self {
        V::splat(token.0, false.into())
    }

    #[inline(always)]
    fn one(token: SimdToken<S>) -> Self {
        V::splat(token.0, true.into())
    }

    #[inline(always)]
    fn eq(self, rhs: Self) -> Self::Mask {
        self.simd_eq(rhs)
    }

    #[inline(always)]
    fn ne(self, rhs: Self) -> Self::Mask {
        !self.simd_eq(rhs)
    }

    #[inline(always)]
    fn lt(self, rhs: Self) -> Self::Mask {
        self.simd_lt(rhs)
    }

    #[inline(always)]
    fn le(self, rhs: Self) -> Self::Mask {
        self.simd_le(rhs)
    }

    #[inline(always)]
    fn gt(self, rhs: Self) -> Self::Mask {
        self.simd_gt(rhs)
    }

    #[inline(always)]
    fn ge(self, rhs: Self) -> Self::Mask {
        self.simd_ge(rhs)
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
        mask.select(if_true, if_false)
    }
}

impl<S: Simd, V> Signed<SimdToken<S>> for V
where
    V: SimdBase<S> + ops::Neg<Output = V>,
{
    #[inline(always)]
    fn abs(self) -> Self {
        SimdBase::abs(self)
    }
}

impl<S: Simd, V: SimdFloat<S>> Real<SimdToken<S>> for V {
    #[inline(always)]
    fn copysign(self, rhs: Self) -> Self {
        SimdFloat::copysign(self, rhs)
    }

    #[inline(always)]
    fn approximate_recip(self) -> Self {
        SimdFloat::approximate_recip(self)
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        SimdFloat::sqrt(self)
    }
}

mod seal_cast {
    pub trait SealCast {}
}

use seal_cast::SealCast;

pub trait Cast<T>: SealCast {
    fn cast(self) -> T;
}

impl<T: SealCast> Cast<T> for T {
    #[inline(always)]
    fn cast(self) -> T {
        self
    }
}

macro_rules! impl_cast {
    ($($from:ty => [$($to:ty),*]),*) => {
        $(
            impl SealCast for $from {}

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

impl_cast! {
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
    f64 => [u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32]
}

impl SealCast for bool {}

impl<T> Cast<T> for bool
where
    u8: Cast<T>,
{
    #[inline(always)]
    fn cast(self) -> T {
        (self as u8).cast()
    }
}

#[derive_where(Copy, Clone, Eq, PartialEq, Debug; T)]
#[repr(C)]
pub struct AVec3<T, Token = ScalarToken> {
    pub x: T,
    pub y: T,
    pub z: T,
    _tkn: PhantomData<Token>,
}

/// SAFETY: `AVec3<T>` is a `repr(C)` struct with exactly `N` fields of type `T`.
unsafe impl<T: Copy, const N: usize> Zeroable for AVec3<[T; N]> where [T; N]: Zeroable {}

/// SAFETY: `AVec3<T>` is a `repr(C)` struct with exactly `N` fields of type `T`.
unsafe impl<T: Copy, const N: usize> AnyBitPattern for AVec3<[T; N]> where [T; N]: AnyBitPattern {}

/// SAFETY: `AVec3<T>` is a `repr(C)` struct with exactly `N` fields of type `T`.
unsafe impl<T: Copy, const N: usize> NoUninit for AVec3<[T; N]> where [T; N]: NoUninit {}

impl<Token: Copy, T: Copy> AVec3<T, Token> {
    #[inline(always)]
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self {
            x,
            y,
            z,
            _tkn: PhantomData,
        }
    }

    #[inline(always)]
    pub const fn splat(value: T) -> Self {
        Self::new(value, value, value)
    }

    #[inline(always)]
    pub const fn from_array(a: [T; 3]) -> Self {
        let [x, y, z] = a;
        Self::new(x, y, z)
    }

    #[inline(always)]
    pub const fn to_array(self) -> [T; 3] {
        [self.x, self.y, self.z]
    }

    #[inline(always)]
    pub fn cast<U: Copy>(self) -> AVec3<U, Token>
    where
        T: Cast<U>,
    {
        self.map(|x| x.cast())
    }

    #[inline(always)]
    pub fn map<F, U: Copy>(self, mut f: F) -> AVec3<U, Token>
    where
        F: FnMut(T) -> U,
    {
        AVec3::new(f(self.x), f(self.y), f(self.z))
    }

    #[inline(always)]
    pub fn mix<F, B, U: Copy>(self, rhs: AVec3<B, Token>, mut f: F) -> AVec3<U, Token>
    where
        F: FnMut(T, B) -> U,
    {
        AVec3::new(f(self.x, rhs.x), f(self.y, rhs.y), f(self.z, rhs.z))
    }

    #[inline(always)]
    pub fn mix_apply<F, B>(&mut self, rhs: AVec3<B, Token>, mut f: F)
    where
        F: FnMut(&mut T, B),
    {
        f(&mut self.x, rhs.x);
        f(&mut self.y, rhs.y);
        f(&mut self.z, rhs.z);
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

    #[inline(always)]
    pub const fn as_array(&self) -> &[T; 3]
    where
        [T; 3]: AnyBitPattern,
        Self: NoUninit,
    {
        must_cast_ref(self)
    }

    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [T; 3]
    where
        [T; 3]: NoUninit + AnyBitPattern,
        Self: NoUninit + AnyBitPattern,
    {
        must_cast_mut(self)
    }

    #[inline(always)]
    pub const fn from_array_ref(a: &[T; 3]) -> &Self
    where
        Self: AnyBitPattern,
        [T; 3]: NoUninit,
    {
        must_cast_ref(a)
    }

    #[inline(always)]
    pub fn from_array_mut(a: &mut [T; 3]) -> &mut Self
    where
        Self: NoUninit + AnyBitPattern,
        [T; 3]: NoUninit + AnyBitPattern,
    {
        must_cast_mut(a)
    }
}

impl<S: Simd, V: Copy> AVec3<V, SimdToken<S>> {
    #[inline(always)]
    pub fn simd_new<T>(simd: S, x: T, y: T, z: T) -> Self
    where
        T: SimdInto<V, S>,
    {
        Self::new(x.simd_into(simd), y.simd_into(simd), z.simd_into(simd))
    }

    #[inline(always)]
    pub fn simd_splat<T>(simd: S, value: T) -> Self
    where
        T: SimdInto<V, S>,
    {
        let value = value.simd_into(simd);
        Self::simd_new(simd, value, value, value)
    }
}

impl<S: Simd, V: SimdBase<S>> AVec3<V, SimdToken<S>> {
    #[inline(always)]
    pub fn simd_from_slice(simd: S, slice: &[V::Element]) -> Self {
        assert_eq!(slice.len(), V::LEN * 3);
        Self::simd_new(
            simd,
            V::from_slice(simd, &slice[0..V::LEN]),
            V::from_slice(simd, &slice[V::LEN..(2 * V::LEN)]),
            V::from_slice(simd, &slice[(2 * V::LEN)..]),
        )
    }
}

impl<Token: Copy, T: Number<Token>> AVec3<T, Token> {
    #[inline(always)]
    pub fn zero(token: Token) -> Self {
        Self::splat(T::zero(token))
    }

    #[inline(always)]
    pub fn one(token: Token) -> Self {
        Self::splat(T::one(token))
    }

    #[inline(always)]
    pub fn axis_x(token: Token) -> Self {
        Self::new(T::one(token), T::zero(token), T::zero(token))
    }

    #[inline(always)]
    pub fn axis_y(token: Token) -> Self {
        Self::new(T::zero(token), T::one(token), T::zero(token))
    }

    #[inline(always)]
    pub fn axis_z(token: Token) -> Self {
        Self::new(T::zero(token), T::zero(token), T::one(token))
    }

    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        self.mix(rhs, |a, b| a.max(b))
    }

    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        self.mix(rhs, |a, b| a.min(b))
    }

    #[inline(always)]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }

    #[inline(always)]
    pub fn component_sum(self) -> T {
        self.x + self.y + self.z
    }

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> T {
        self.mix(rhs, |a, b| a * b).component_sum()
    }

    #[inline(always)]
    pub fn norm_squared(self) -> T {
        self.dot(self)
    }

    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
}

impl<Token: Copy, T: Number<Token>> AVec3<T, Token> {
    #[inline(always)]
    pub fn eq_mask(self, rhs: Self) -> AVec3<T::Mask, Token> {
        self.mix(rhs, Number::eq)
    }

    #[inline(always)]
    pub fn ne_mask(self, rhs: Self) -> AVec3<T::Mask, Token> {
        self.mix(rhs, Number::ne)
    }

    #[inline(always)]
    pub fn lt_mask(self, rhs: Self) -> AVec3<T::Mask, Token> {
        self.mix(rhs, Number::lt)
    }

    #[inline(always)]
    pub fn le_mask(self, rhs: Self) -> AVec3<T::Mask, Token> {
        self.mix(rhs, Number::le)
    }

    #[inline(always)]
    pub fn gt_mask(self, rhs: Self) -> AVec3<T::Mask, Token> {
        self.mix(rhs, Number::gt)
    }

    #[inline(always)]
    pub fn ge_mask(self, rhs: Self) -> AVec3<T::Mask, Token> {
        self.mix(rhs, Number::ge)
    }
}

impl<Token: Copy, T: Number<Token> + WithConstToken> AVec3<T, Token> {
    pub const ZERO: Self = AVec3::splat(T::ZERO);
    pub const ONE: Self = AVec3::splat(T::ONE);
    pub const X: Self = AVec3::new(T::ONE, T::ZERO, T::ZERO);
    pub const Y: Self = AVec3::new(T::ZERO, T::ONE, T::ZERO);
    pub const Z: Self = AVec3::new(T::ZERO, T::ZERO, T::ONE);
}

impl<Token: Copy, T: Real<Token>> AVec3<T, Token> {
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
        let norm: T = self.norm();
        self.map(|v| {
            let zero = T::zero(v.token());
            T::select(norm.eq(zero), zero, v / norm)
        })
    }
}

impl<Token: Copy, T: Copy> From<[T; 3]> for AVec3<T, Token> {
    #[inline(always)]
    fn from(value: [T; 3]) -> Self {
        Self::from_array(value)
    }
}

impl<Token: Copy, T: Copy> From<AVec3<T, Token>> for [T; 3] {
    #[inline(always)]
    fn from(value: AVec3<T, Token>) -> Self {
        value.to_array()
    }
}

impl<T> ops::Index<usize> for AVec3<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        match index.cast() {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index must be between 0 and 2"),
        }
    }
}

impl<T> ops::IndexMut<usize> for AVec3<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index.cast() {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("index must be between 0 and 2"),
        }
    }
}

impl<Token: Copy, T> iter::Sum for AVec3<T, Token>
where
    T: Number<Token> + WithConstToken,
{
    #[inline(always)]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, v| acc + v)
    }
}

macro_rules! impl_imm_binops {
    ($(($trait:ident, $method:ident)),* $(,)?) => {
        $(
            impl<Token: Copy, T: Copy + ops::$trait> ops::$trait for AVec3<T, Token>
            where
                <T as ops::$trait>::Output: Copy,
            {
                type Output = AVec3<<T as ops::$trait>::Output, Token>;

                #[inline(always)]
                fn $method(self, rhs: Self) -> Self::Output {
                    self.mix(rhs, <T as ops::$trait>::$method)
                }
            }

            impl<Token: Copy, T: Copy + ops::$trait> ops::$trait<T> for AVec3<T, Token>
            where
                <T as ops::$trait>::Output: Copy,
            {
                type Output = AVec3<<T as ops::$trait<T>>::Output, Token>;

                #[inline(always)]
                fn $method(self, rhs: T) -> Self::Output {
                    <AVec3<T, Token> as ops::$trait>::$method(self, AVec3::splat(rhs))
                }
            }
        )*
    };
}

macro_rules! impl_mut_binops {
    ($(($trait:ident, $method:ident)),* $(,)?) => {
        $(
            impl<Token: Copy, T: Copy + ops::$trait> ops::$trait for AVec3<T, Token> {
                #[inline(always)]
                fn $method(&mut self, rhs: Self) {
                    self.mix_apply(rhs, <T as ops::$trait>::$method)
                }
            }

            impl<Token: Copy, T: Copy + ops::$trait> ops::$trait<T> for AVec3<T, Token> {
                #[inline(always)]
                fn $method(&mut self, rhs: T) {
                    <AVec3<T, Token> as ops::$trait>::$method(self, AVec3::splat(rhs))
                }
            }
        )*
    };
}

impl_imm_binops!((Add, add), (Sub, sub), (Mul, mul), (Div, div));

impl_mut_binops! {
    (AddAssign, add_assign), (SubAssign, sub_assign),
    (MulAssign, mul_assign), (DivAssign, div_assign)
}

pub type Vec3<T> = AVec3<T, ScalarToken>;
pub type Vec3S<S, V> = AVec3<V, SimdToken<S>>;
