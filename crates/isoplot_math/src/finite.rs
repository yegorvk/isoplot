use std::marker::PhantomData;

use bytemuck::{NoUninit, Pod, Zeroable};
use fearless_simd::{Simd, SimdBase, f32x8, mask32x8};

use crate::{
    masked::{IntoMasked, Masked},
    num::{IntoNumber, Mask, One, Real, Select, Zero},
    seal::Sealed,
    token::{PhantomToken, SimdToken, WithToken},
};

/// A floating point value that is known to be finite
#[derive(Copy, Clone, NoUninit)]
#[repr(transparent)]
pub struct Finite<T: Real<Token>, Token: Copy = PhantomToken> {
    value: T,
    _tkn: PhantomData<Token>,
}

impl<Token: Copy, T: Real<Token>> Finite<T, Token> {
    #[inline(always)]
    fn new_unchecked(value: T) -> Self {
        debug_assert!(value.is_finite().all_true());
        Self {
            value,
            _tkn: PhantomData,
        }
    }

    #[inline(always)]
    pub fn get(self) -> T {
        self.value
    }
}

impl<Token: Copy, T: Real<Token>> Sealed<Token> for Finite<T, Token> {}

impl<Token: Copy, T: Real<Token>> WithToken<Token> for Finite<T, Token> {
    #[inline]
    fn token(&self) -> Token {
        self.value.token()
    }
}

impl<Token: Copy, T: Real<Token>> Zero<Token> for Finite<T, Token> {
    #[inline(always)]
    fn zero(token: Token) -> Self {
        Self {
            value: T::zero(token),
            _tkn: PhantomData,
        }
    }
}

impl<Token: Copy, T: Real<Token>> One<Token> for Finite<T, Token> {
    #[inline(always)]
    fn one(token: Token) -> Self {
        Self {
            value: T::one(token),
            _tkn: PhantomData,
        }
    }
}

impl<Token: Copy, T: Real<Token>> Select<Token> for Finite<T, Token> {
    type Mask = T::Mask;

    #[inline(always)]
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
        Self {
            value: T::select(mask, if_true.value, if_false.value),
            _tkn: PhantomData,
        }
    }
}

impl<Token: Copy, T: Real<Token>> IntoNumber<Token> for Finite<T, Token> {
    type Number = T;

    #[inline(always)]
    fn into_number(self) -> Self::Number {
        self.value
    }
}

/// A floating point value that may or may not be finite.
///
/// Unlike a regular float, this type allows explicitly treating non-finite numbers
/// as "forbidden" bit patterns for niche-like manual optimizations.
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(transparent)]
pub struct MaybeFinite<T: Real<Token>, Token: Copy = PhantomToken> {
    value: T,
    _tkn: PhantomData<Token>,
}

impl<Token: Copy, T: Real<Token>> MaybeFinite<T, Token> {
    #[inline(always)]
    pub fn new(value: T) -> Self {
        Self {
            value,
            _tkn: PhantomData,
        }
    }

    #[inline(always)]
    pub fn is_finite(self) -> T::Mask {
        self.value.is_finite()
    }

    #[inline(always)]
    pub fn into_parts(self) -> (T, T::Mask) {
        (self.value, self.is_finite())
    }

    #[inline(always)]
    pub fn filter_map<F>(self, f: F) -> Self
    where
        F: FnOnce(T) -> T,
    {
        Self::new(f(self.value))
    }
}

impl<Token: Copy, T: Real<Token, Mask = bool>> MaybeFinite<T, Token> {
    #[inline(always)]
    pub fn get(self) -> Option<Finite<T, Token>> {
        self.is_finite().then(|| Finite::new_unchecked(self.value))
    }
}

impl<Token: Copy, T: Real<Token>> Sealed<Token> for MaybeFinite<T, Token> {}

impl Masked<PhantomToken> for MaybeFinite<f32, PhantomToken> {
    type Value = Finite<f32>;
    type Mask = bool;

    #[inline]
    fn all_masked(_token: PhantomToken) -> Self {
        MaybeFinite::new(f32::INFINITY)
    }

    #[inline]
    fn mask(self) -> bool {
        self.is_finite()
    }

    #[inline]
    fn get_or(self, default: Self::Value) -> Self::Value {
        Finite::new_unchecked(f32::select(
            self.value.is_finite(),
            self.value,
            default.value,
        ))
    }

    #[inline]
    fn get_or_zero(self) -> Self::Value {
        self.get_or(Finite::zero(PhantomToken))
    }
}

impl<S: Simd> Masked<SimdToken<S>> for MaybeFinite<f32x8<S>, SimdToken<S>> {
    type Value = Finite<f32x8<S>, SimdToken<S>>;
    type Mask = mask32x8<S>;

    #[inline]
    fn all_masked(token: SimdToken<S>) -> Self {
        MaybeFinite::new(f32x8::splat(token.0, f32::INFINITY))
    }

    #[inline(always)]
    fn mask(self) -> Self::Mask {
        self.value.is_finite()
    }

    #[inline(always)]
    fn get_or(self, default: Self::Value) -> Self::Value {
        Finite::new_unchecked(f32x8::select(
            self.value.is_finite(),
            self.value,
            default.value,
        ))
    }

    // TODO: use bitwise operations?
    #[inline(always)]
    fn get_or_zero(self) -> Self::Value {
        self.get_or(Finite::zero(self.value.token()))
    }
}

impl<Token: Copy, T: Real<Token>> IntoMasked<Token> for Finite<T, Token>
where
    MaybeFinite<T, Token>: Masked<Token, Value = Finite<T, Token>>,
{
    type Masked = MaybeFinite<T, Token>;

    #[inline(always)]
    fn into_masked(self) -> Self::Masked {
        MaybeFinite::new(self.value)
    }
}

pub type FiniteS<S, V> = Finite<V, SimdToken<S>>;
pub type MaybeFiniteS<S, V> = MaybeFinite<V, SimdToken<S>>;
