use std::{iter, marker::PhantomData, ops};

use bytemuck::{AnyBitPattern, NoUninit, Zeroable, must_cast_mut, must_cast_ref};
use derive_where::derive_where;
use fearless_simd::{Simd, SimdBase, SimdInto};

use crate::{
    num::{ConstOne, ConstZero, Number, PrimitiveCast, Real},
    token::{PhantomToken, SimdToken},
};

#[derive_where(Copy, Clone, Eq, PartialEq, Debug; T)]
#[repr(C)]
pub struct AVec3<T, Token = PhantomToken> {
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
        T: PrimitiveCast<U>,
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
    pub fn reduce_sum(self) -> T {
        self.x + self.y + self.z
    }

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> T {
        self.mix(rhs, |a, b| a * b).reduce_sum()
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

impl<Token: Copy, T: Number<Token> + ConstZero<Token>> AVec3<T, Token> {
    pub const ZERO: Self = AVec3::splat(T::ZERO);
}

impl<Token: Copy, T: Number<Token> + ConstOne<Token>> AVec3<T, Token> {
    pub const ONE: Self = AVec3::splat(T::ONE);
}

impl<Token: Copy, T: Number<Token> + ConstZero<Token> + ConstOne<Token>> AVec3<T, Token> {
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
    T: Number<Token> + ConstZero<Token>,
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
