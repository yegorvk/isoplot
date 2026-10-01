use std::{hint::select_unpredictable, marker::PhantomData};

use fearless_simd::Simd;

use crate::{
    MaskedVec3, Vec3,
    finite::{Finite, MaybeFinite},
    num::{IntoNumber, Mask, Number, Real, Select, Zero},
    seal::Sealed,
    token::{PhantomToken, SimdToken, WithToken},
    vec3::AVec3,
};

pub trait Masked<Token: Copy>: Sealed<Token> + Copy {
    /// The underlying value type
    type Value: Copy;

    /// The mask type
    type Mask: Mask<Token>;

    /// Creates a new value with all lanes masked.
    fn all_masked(token: Token) -> Self;

    // /// Creates a new masked value from a value and a mask.
    // fn from_parts(value: Self::Value, mask: Self::Mask) -> Self;

    /// Returns the mask.
    fn mask(self) -> Self::Mask;

    /// Returns a copy where all masked lanes are replaced with
    /// the corresponding lanes from `default`.
    fn get_or(self, default: Self::Value) -> Self::Value;

    /// Returns a copy where all masked lanes are replaced with zeroes.
    fn get_or_zero(self) -> Self::Value;
}

pub trait IntoMasked<Token: Copy>: Sealed<Token> {
    type Masked: Masked<Token, Value = Self>;
    fn into_masked(self) -> Self::Masked;
}

#[derive(Copy, Clone)]
pub struct MaskedAVec3<T: IntoMasked<Token>, Token: Copy> {
    x: T::Masked,
    y: T,
    z: T,
}

impl<Token: Copy, M: Mask<Token>, T: Copy> MaskedAVec3<T, Token>
where
    T: IntoMasked<Token, Masked: Masked<Token, Mask = M>>
        + IntoNumber<Token, Number: Number<Token, Mask = M>>
        + Zero<Token>
        + Select<Token, Mask = M>,
{
    #[inline(always)]
    pub fn get_or(self, default: T) -> AVec3<T, Token> {
        self.into_parts_or(default).0
    }

    #[inline(always)]
    pub fn get_or_zero(self) -> AVec3<T, Token> {
        self.get_or(T::zero(self.y.token()))
    }

    #[inline(always)]
    pub fn into_parts_or(self, default: T) -> (AVec3<T, Token>, T::Mask) {
        let mask = self.x.mask();
        (
            AVec3::new(
                self.x.get_or(default),
                T::select(mask, self.y, default),
                T::select(mask, self.z, default),
            ),
            mask,
        )
    }

    #[inline(always)]
    pub fn into_parts_or_zero(self) -> (AVec3<T, Token>, T::Mask) {
        self.into_parts_or(T::zero(self.y.token()))
    }
}

impl<Token: Copy, T: Real<Token>> MaskedAVec3<Finite<T, Token>, Token>
where
    Finite<T, Token>: IntoMasked<Token, Masked = MaybeFinite<T, Token>>,
    MaybeFinite<T, Token>: Masked<Token, Value = Finite<T, Token>>,
{
    #[inline(always)]
    pub fn new(value: Vec3<T>) -> Self {
        Self {
            x: MaybeFinite::new(value.x),
            y: MaybeFinite::new(value.y).get_or_zero(),
            z: MaybeFinite::new(value.z).get_or_zero(),
        }
    }

    #[inline(always)]
    pub fn filter_map<F>(self, f: F) -> Self
    where
        F: FnOnce(AVec3<T, Token>) -> AVec3<T, Token>,
    {
        let mut y = T::zero(self.y.token());
        let mut z = T::zero(self.z.token());

        let x = self.x.filter_map(|x| {
            let v = AVec3::new(x, self.y.get(), self.z.get());
            let new = f(v);
            (y, z) = (new.y, new.z);
            new.x
        });

        Self {
            x,
            y: MaybeFinite::new(y).get_or_zero(),
            z: MaybeFinite::new(z).get_or_zero(),
        }
    }
}

impl<Token: Copy, T: Real<Token, Mask = bool>> MaskedAVec3<Finite<T, Token>, Token>
where
    MaybeFinite<T, Token>: Masked<Token, Value = Finite<T, Token>, Mask = bool>,
    Finite<T, Token>: IntoMasked<Token, Masked = MaybeFinite<T, Token>>,
{
    #[inline]
    pub fn get(self) -> Option<AVec3<T, Token>> {
        self.x
            .get()
            .map(|x| AVec3::new(x.get(), self.y.get(), self.z.get()))
    }
}

impl From<Option<Vec3<f32>>> for MaskedVec3<Finite<f32>> {
    #[inline]
    fn from(value: Option<Vec3<f32>>) -> Self {
        match value {
            Some(v) => Self {
                x: MaybeFinite::new(v.x),
                y: MaybeFinite::<f32>::new(v.y).get_or_zero(),
                z: MaybeFinite::<f32>::new(v.z).get_or_zero(),
            },
            None => Self {
                x: MaybeFinite::all_masked(PhantomToken),
                y: Finite::zero(PhantomToken),
                z: Finite::zero(PhantomToken),
            },
        }
    }
}
