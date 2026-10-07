use crate::token::{PhantomToken, SimdToken};
use fearless_simd::{Simd, f32x8, mask32x8, u32x8};

// Rust orphan rules ensure that we do not need trait-specific seals.
mod private {
    /// Prevents downstream crates from implementing internal traits
    pub trait Sealed<Token> {}
}

pub(crate) use private::Sealed;

macro_rules! seal {
    ($($ty:ty),*) => {
        $( impl Sealed<PhantomToken> for $ty {} )*
    };
}

seal!(
    bool, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64
);

// SIMD vectors
impl<S: Simd> Sealed<SimdToken<S>> for u32x8<S> {}
impl<S: Simd> Sealed<SimdToken<S>> for f32x8<S> {}

// SIMD masks
impl<S: Simd> Sealed<SimdToken<S>> for mask32x8<S> {}
