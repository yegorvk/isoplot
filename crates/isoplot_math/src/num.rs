use fearless_simd::{
    Bytes, Select as SimdSelect, Simd, SimdBase, SimdFloat, SimdMask, f32x8, u32x8,
};
use std::{marker::PhantomData, ops};

use crate::{
    seal::Sealed,
    token::{PhantomToken, SimdToken, WithToken},
};

pub trait Mask<Token: Copy>:
    Sealed<Token>
    + WithToken<Token>
    + Copy
    + ops::Not<Output = Self>
    + ops::BitAnd<Output = Self>
    + ops::BitOr<Output = Self>
    + ops::BitXor<Output = Self>
{
    /// Returns a mask where all lanes are equal to `value`.
    fn splat(token: Token, value: bool) -> Self;

    /// Returns if all lanes are `true`.
    fn all_true(self) -> bool;
}

impl Mask<PhantomToken> for bool {
    #[inline]
    fn splat(_token: PhantomToken, value: bool) -> Self {
        value
    }

    #[inline]
    fn all_true(self) -> bool {
        self
    }
}

pub trait Select<Token: Copy>: Sealed<Token> {
    /// The mask type used for conditional execution
    type Mask: Mask<Token>;

    /// Selects between two values using `mask`.
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;
}

pub trait Zero<Token>: Sealed<Token> + WithToken<Token> {
    fn zero(token: Token) -> Self;
}

pub trait ConstZero<Token>: Sealed<Token> + Zero<Token> {
    const ZERO: Self;
}

pub trait One<Token>: Sealed<Token> + WithToken<Token> {
    fn one(token: Token) -> Self;
}

pub trait ConstOne<Token>: Sealed<Token> + One<Token> {
    const ONE: Self;
}

pub trait Number<Token: Copy>:
    Sealed<Token>
    + WithToken<Token>
    + Copy
    + Zero<Token>
    + One<Token>
    + ops::Add<Output = Self>
    + ops::Sub<Output = Self>
    + ops::Mul<Output = Self>
    + Select<Token>
{
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
}

pub trait Signed<Token: Copy>: Sealed<Token> + Number<Token> + ops::Neg {
    /// Computes the absolute value of this number.
    fn abs(self) -> Self;
}

pub trait Real<Token: Copy>: Sealed<Token> + Number<Token> + ops::Div<Output = Self> {
    /// Returns a number with the magnitude of `self` and the sign of `rhs`.
    fn copysign(self, rhs: Self) -> Self;

    /// Returns whether this number if finite.
    fn is_finite(self) -> Self::Mask;

    /// Computes an approximate value of `1.0 / self`.
    #[inline(always)]
    fn approximate_recip(self) -> Self {
        Self::one(self.token()) / self
    }

    /// Returns the square root of this element.
    fn sqrt(self) -> Self;
}

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
            impl Select<PhantomToken > for $ty {
                type Mask = bool;

                #[inline(always)]
                fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
                    std::hint::select_unpredictable(mask, if_true, if_false)
                }
            }

            impl ConstZero<PhantomToken > for $ty {
                const ZERO: $ty = 0 as $ty;
            }

            impl Zero<PhantomToken > for $ty {
                #[inline]
                fn zero(_token: PhantomToken ) -> Self {
                    Self::ZERO
                }
            }

            impl ConstOne<PhantomToken > for $ty {
                const ONE: $ty = 1 as $ty;
            }

            impl One<PhantomToken > for $ty {
                #[inline]
                fn one(_token: PhantomToken ) -> Self {
                    Self::ONE
                }
            }

            impl Number<PhantomToken> for $ty {
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
            }
        )*
    };
}

impl_number!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);

macro_rules! impl_signed {
    ($($ty:ty),*) => {
        $(
            impl Signed<PhantomToken > for $ty {
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
            impl Real<PhantomToken > for $ty {
                #[inline(always)]
                fn copysign(self, rhs: Self) -> Self {
                    <$ty>::copysign(self, rhs)
                }

                #[inline]
                fn is_finite(self) -> Self::Mask {
                    <$ty>::is_finite(self)
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

impl<S: Simd, V: Sealed<SimdToken<S>> + SimdMask<S>> Mask<SimdToken<S>> for V {
    #[inline(always)]
    fn splat(token: SimdToken<S>, value: bool) -> Self {
        SimdMask::splat(token.0, value)
    }

    #[inline(always)]
    fn all_true(self) -> bool {
        SimdMask::all_true(self)
    }
}

impl<S: Simd, V: Sealed<SimdToken<S>> + SimdBase<S>> Zero<SimdToken<S>> for V {
    #[inline(always)]
    fn zero(token: SimdToken<S>) -> Self {
        V::splat(token.0, true.into())
    }
}

impl<S: Simd, V: Sealed<SimdToken<S>> + SimdBase<S>> One<SimdToken<S>> for V {
    #[inline(always)]
    fn one(token: SimdToken<S>) -> Self {
        V::splat(token.0, true.into())
    }
}

impl<S: Simd, V> Select<SimdToken<S>> for V
where
    V: Sealed<SimdToken<S>> + SimdBase<S, Mask: Mask<SimdToken<S>>>,
{
    type Mask = V::Mask;

    #[inline(always)]
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
        mask.select(if_true, if_false)
    }
}

impl<S: Simd, V> Number<SimdToken<S>> for V
where
    V: Sealed<SimdToken<S>> + SimdBase<S, Mask: Mask<SimdToken<S>>>,
{
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
}

impl<S: Simd, V> Signed<SimdToken<S>> for V
where
    V: Sealed<SimdToken<S>> + SimdBase<S, Mask: Mask<SimdToken<S>>> + ops::Neg<Output = V>,
{
    #[inline(always)]
    fn abs(self) -> Self {
        SimdBase::abs(self)
    }
}

impl<S: Simd> Real<SimdToken<S>> for f32x8<S> {
    #[inline(always)]
    fn copysign(self, rhs: Self) -> Self {
        SimdFloat::copysign(self, rhs)
    }

    #[inline(always)]
    fn is_finite(self) -> Self::Mask {
        const F32_EXP_MASK: u32 = 0xFF << 23;
        let bits: u32x8<S> = self.bitcast();
        !(bits & F32_EXP_MASK).simd_eq(F32_EXP_MASK)
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

pub trait PrimitiveCast<T>: Sealed<PhantomToken> {
    fn cast(self) -> T;
}

impl<T: Sealed<PhantomToken>> PrimitiveCast<T> for T {
    #[inline(always)]
    fn cast(self) -> T {
        self
    }
}

macro_rules! impl_cast {
    ($($from:ty => [$($to:ty),*]),*) => {
        $(
            $(
                impl PrimitiveCast<$to> for $from {
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

impl<T> PrimitiveCast<T> for bool
where
    u8: PrimitiveCast<T>,
{
    #[inline(always)]
    fn cast(self) -> T {
        (self as u8).cast()
    }
}
