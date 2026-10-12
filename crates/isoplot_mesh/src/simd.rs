use bytemuck::{Pod, Zeroable};
use fearless_simd::{mask32x8, prelude::*};

#[derive(Copy, Clone, Eq, PartialEq, Pod, Zeroable)]
#[repr(transparent)]
pub struct BMask8(u8);

impl BMask8 {
    pub(crate) const ZERO: Self = Self(0);

    #[inline]
    pub(crate) fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> bool,
    {
        let mut mask = 0u8;

        for i in 0..8u8 {
            mask |= (f(i as usize) as u8) << i;
        }

        Self(mask)
    }

    pub(crate) const fn get(self, index: u8) -> bool {
        self.0 & (1u8 << index) != 0
    }

    #[inline(always)]
    fn unpack_32x8<S: Simd>(self, simd: S) -> mask32x8<S> {
        mask32x8::from_bitmask(simd, self.0 as u64)
    }
}

impl<S: Simd> From<mask32x8<S>> for BMask8 {
    #[inline(always)]
    fn from(value: mask32x8<S>) -> Self {
        Self(value.to_bitmask() as u8)
    }
}

impl<S: Simd> SimdFrom<BMask8, S> for mask32x8<S> {
    #[inline(always)]
    fn simd_from(simd: S, value: BMask8) -> Self {
        value.unpack_32x8(simd)
    }
}
