use fearless_simd::{ExtractToken, Simd};

/// A token that is always available
#[derive(Copy, Clone, Debug)]
pub struct PhantomToken;

/// A token that wraps a hardware capability token from `fearless_simd` crate
#[derive(Copy, Clone, Debug)]
pub struct SimdToken<S: Simd>(pub S);

impl<S: Simd> SimdToken<S> {
    #[inline]
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

/// A type that holds a token
pub trait WithToken<Token> {
    fn token(&self) -> Token;
}

macro_rules! impl_with_token {
    ($($ty:ty),*) => {
        $(
            impl WithToken<PhantomToken> for $ty {
                #[inline]
                fn token(&self) -> PhantomToken {
                    PhantomToken
                }
            }
        )*
    };
}

impl_with_token!(
    bool, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64
);

impl<S: Simd, T: Copy + ExtractToken<S = S>> WithToken<SimdToken<S>> for T {
    #[inline]
    fn token(&self) -> SimdToken<S> {
        ExtractToken::token(&self).into()
    }
}

impl<S: Simd> ExtractToken for SimdToken<S> {
    type S = S;

    #[inline]
    fn token(&self) -> Self::S {
        WithToken::token(self).0
    }
}
