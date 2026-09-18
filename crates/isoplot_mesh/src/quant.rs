use bilge::prelude::*;
use fearless_simd::{Simd, SimdInto};

use crate::{
    math::Vec3,
    octree::{ChildIndex, Payload},
    simd::{Bitcast, f32s, i32s, mask32s, u32s},
};

/// A quantized point in a unit cube
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[repr(transparent)]
pub(crate) struct Quant(RawQuant);

impl Quant {
    /// Maximum number of consequent subdivisions
    pub(crate) const MAX_SUBDIV: u8 = 10;

    pub(crate) fn root() -> Self {
        Self(RawQuant::root())
    }

    pub(crate) fn level(self) -> u8 {
        self.0.level()
    }

    pub(crate) fn child(self, which: ChildIndex) -> Option<Quant> {
        self.0.child(which.0).map(Quant)
    }

    pub(crate) fn min_point_size(self) -> (Vec3<f32>, f32) {
        let (parts, level) = self.0.parts_level();

        let x = fract_u32_to_f32(parts.x as u32, level as u32);
        let y = fract_u32_to_f32(parts.y as u32, level as u32);
        let z = fract_u32_to_f32(parts.z as u32, level as u32);

        (Vec3::new(x, y, z), f32_exp2_small(-(level as i8)))
    }

    pub(crate) fn center_point(self) -> Vec3<f32> {
        let (min_point, size) = self.min_point_size();
        min_point + size * 0.5
    }
}

impl Payload for Quant {
    fn into_bits(self) -> u31 {
        self.0.value
    }

    unsafe fn from_bits(bits: u31) -> Self {
        Self(RawQuant { value: bits })
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct QuantS<S: Simd>(RawQuantS<S>);

impl<S: Simd> QuantS<S> {
    /// Maximum number of consequent subdivisions
    pub(crate) const MAX_SUBDIV: u8 = Quant::MAX_SUBDIV;

    #[inline(always)]
    pub(crate) fn root(simd: S) -> Self {
        Self(RawQuantS::root(simd))
    }

    #[inline(always)]
    pub(crate) fn level(self) -> u32s<S> {
        self.0.level()
    }

    #[inline(always)]
    pub(crate) fn is_leaf(self) -> mask32s<S> {
        self.0.is_leaf()
    }

    #[inline(always)]
    pub(crate) fn child(self, which: ChildIndex) -> Self {
        let which = which.0.value() as u32;
        Self(self.0.child(which.simd_into(self.0.witness())))
    }

    #[inline(always)]
    pub(crate) fn min_point_size(self) -> (Vec3<f32s<S>>, f32s<S>) {
        let (parts, level) = self.0.parts_level();

        let x = fract_u32_to_f32_s(parts.x, level);
        let y = fract_u32_to_f32_s(parts.y, level);
        let z = fract_u32_to_f32_s(parts.z, level);

        (Vec3::new(x, y, z), f32_exp2_small_i32_s(-level.to_signed()))
    }

    #[inline(always)]
    pub(crate) fn center_point(self) -> Vec3<f32s<S>> {
        let (min_point, size) = self.min_point_size();
        min_point + size * 0.5
    }
}

#[repr(transparent)]
#[bitsize(31)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, DebugBits)]
struct RawQuant {
    x: u11,
    y: u10,
    z: u10,
}

impl RawQuant {
    fn root() -> Self {
        Self::new(u11::new(1), u10::ZERO, u10::ZERO)
    }

    fn from_raw_parts(raw_x: u16, y: u16, z: u16) -> Self {
        debug_assert!(raw_x != 0);
        Self::new(u11::new(raw_x), u10::new(y), u10::new(z))
    }

    fn level(self) -> u8 {
        quant_level(self.x().value())
    }

    fn child(self, which: u3) -> Option<RawQuant> {
        let (mut raw_x, mut y, mut z) = self.raw_parts();

        if quant_level(raw_x) == Quant::MAX_SUBDIV {
            return None;
        }

        let which = which.value();

        raw_x = (raw_x << 1) | ((which & 0x1 != 0) as u16);
        y = (y << 1) | ((which & 0x2 != 0) as u16);
        z = (z << 1) | ((which & 0x4 != 0) as u16);

        Some(Self::from_raw_parts(raw_x, y, z))
    }

    fn parts_level(self) -> (Vec3<u16>, u8) {
        let ((raw_x, y, z), level) = self.raw_parts_level();
        (Vec3::new(raw_x ^ (1u16 << level), y, z), level)
    }

    fn raw_parts_level(self) -> ((u16, u16, u16), u8) {
        let (raw_x, y, z) = self.raw_parts();
        ((raw_x, y, z), quant_level(raw_x))
    }

    fn raw_parts(self) -> (u16, u16, u16) {
        (self.x().value(), self.y().value(), self.z().value())
    }
}

#[repr(transparent)]
#[derive(Copy, Clone)]
struct RawQuantS<S: Simd>(u32s<S>);

impl<S: Simd> RawQuantS<S> {
    const X_BITS: u32 = 11;
    const Y_BITS: u32 = 10;
    const Z_BITS: u32 = 10;

    const Y_SHIFT: u32 = Self::X_BITS;
    const Z_SHIFT: u32 = Self::X_BITS + Self::Y_BITS;

    #[inline(always)]
    fn root(simd: S) -> Self {
        Self::from_raw_parts(
            1u32.simd_into(simd),
            0u32.simd_into(simd),
            0u32.simd_into(simd),
        )
    }

    #[inline(always)]
    fn from_raw_parts(raw_x: u32s<S>, y: u32s<S>, z: u32s<S>) -> Self {
        debug_assert!(
            raw_x.simd_eq(0).all_false()
                && raw_x.simd_le(mask(Self::X_BITS)).all_true()
                && y.simd_le(mask(Self::Y_BITS)).all_true()
                && z.simd_le(mask(Self::Z_BITS)).all_true()
        );

        Self(raw_x | (y << Self::Y_SHIFT) | (z << Self::Z_SHIFT))
    }

    #[inline(always)]
    fn level(self) -> u32s<S> {
        quant_level_s::<S>(self.raw_x())
    }

    #[inline(always)]
    fn is_leaf(self) -> mask32s<S> {
        self.level().simd_lt(Quant::MAX_SUBDIV as u32)
    }

    #[inline(always)]
    fn child(self, which: u32s<S>) -> Self {
        let raw_x = (self.raw_x() << 1) | (which & 0x1);
        let y = (self.y() << 1) | ((which & 0x2) >> 1);
        let z = (self.z() << 1) | ((which & 0x4) >> 2);
        Self::from_raw_parts(raw_x, y, z)
    }

    #[inline(always)]
    fn parts_level(self) -> (Vec3<u32s<S>>, u32s<S>) {
        let [raw_x, y, z] = self.raw_parts();

        let level = quant_level_s::<S>(raw_x);
        let one: u32s<S> = 1u32.simd_into(self.witness());
        let level_bit: u32s<S> = one << level;

        (Vec3::new(raw_x ^ level_bit, y, z), level)
    }

    #[inline(always)]
    fn raw_parts(self) -> [u32s<S>; 3] {
        [self.raw_x(), self.y(), self.z()]
    }

    #[inline(always)]
    fn raw_x(self) -> u32s<S> {
        self.0 & mask(Self::X_BITS)
    }

    #[inline(always)]
    fn y(self) -> u32s<S> {
        (self.0 >> Self::Y_SHIFT) & mask(Self::Y_BITS)
    }

    #[inline(always)]
    fn z(self) -> u32s<S> {
        (self.0 >> Self::Z_SHIFT) & mask(Self::Z_BITS)
    }

    #[inline(always)]
    fn witness(self) -> S {
        self.0.witness()
    }
}

const fn mask(n_bits: u32) -> u32 {
    (1u32 << n_bits) - 1
}

#[inline(always)]
fn quant_level(raw_x: u16) -> u8 {
    let level = (15 - raw_x.leading_zeros()) as u8;
    debug_assert!(level < Quant::MAX_SUBDIV);
    level
}

#[inline(always)]
fn quant_level_s<S: Simd>(raw_x: u32s<S>) -> u32s<S> {
    let level = msb_index_u32_s(raw_x);
    debug_assert!(level.simd_lt(Quant::MAX_SUBDIV as u32).all_true());
    level
}

#[inline(always)]
fn msb_index_u32_s<S: Simd>(v: u32s<S>) -> u32s<S> {
    u32s::select(v.simd_eq(0), 0, msb_index_non_zero_u32_s(v))
}

#[inline(always)]
fn msb_index_non_zero_u32_s<S: Simd>(v: u32s<S>) -> u32s<S> {
    let v_f32: f32s<S> = v.to_float();
    let v_exp = v_f32.bitcast::<u32s<S>>() >> 23;
    let msb = v_exp - 127;
    u32s::select((v >> msb).simd_eq(1), msb, msb - 1)
}

/// Computes `2^exp` for an integer `exp`.
///
/// If `exp` is less than -126, this function
/// will behave incorrectly and may panic.
#[inline]
const fn f32_exp2_small(exp: i8) -> f32 {
    debug_assert!(exp >= -126);
    f32::from_bits(((exp as i32 + 127) as u32) << 23)
}

/// Converts a fraction-only fixed-point `u32` to an `f32`.
///
/// If `len` is greater than 23 or `num` is greater than or equal to
/// `2^len`, this function will behave incorrectly and may panic.
#[inline]
const fn fract_u32_to_f32(num: u32, len: u32) -> f32 {
    debug_assert!((num == 1 && len == 0) || (len <= 23 && num < (1u32 << len)));
    let fract = num << (23u32 - len);
    f32::from_bits((127u32 << 23u32) + fract) - 1f32
}

#[inline(always)]
fn f32_exp2_small_i32_s<S: Simd>(exp: i32s<S>) -> f32s<S> {
    debug_assert!(exp.simd_ge(-126).all_true());
    ((exp + 127).bitcast::<u32s<S>>() << 23).bitcast()
}

#[inline(always)]
fn fract_u32_to_f32_s<S: Simd>(num: u32s<S>, len: u32s<S>) -> f32s<S> {
    debug_assert!(
        ((num.simd_eq(1) & len.simd_eq(0)) | (len.simd_le(23) & num.simd_lt(1 << len))).all_true()
    );
    let fract = num << (23 - len);
    ((127u32 << 23u32) + fract).bitcast::<f32s<S>>() - 1f32
}

#[cfg(test)]
mod tests {
    use super::*;

    trait SimdTest {
        fn run<S: Simd>(self, simd: S);
    }

    fn simd_test(test: impl SimdTest) {
        use fearless_simd::{Level, dispatch};
        dispatch!(Level::new(), simd => test.run(simd));
    }

    #[test]
    fn test_f32_exp2_small() {
        // Zero
        assert_eq!(f32_exp2_small(0), 1.0);

        // Positive
        assert_eq!(f32_exp2_small(1), 2.0);
        assert_eq!(f32_exp2_small(3), 8.0);
        assert_eq!(f32_exp2_small(4), 16.0);

        // Negative
        assert_eq!(f32_exp2_small(-1), 0.5);
        assert_eq!(f32_exp2_small(-3), 0.125);
        assert_eq!(f32_exp2_small(-4), 0.0625);
    }

    #[test]
    fn test_fract_u32_to_f32() {
        // num == 0
        assert_eq!(fract_u32_to_f32(0, 0), 0.0);
        assert_eq!(fract_u32_to_f32(0, 1), 0.0);
        assert_eq!(fract_u32_to_f32(0, 23), 0.0);

        // 0 bits of precision
        assert_eq!(fract_u32_to_f32(1, 0), 1.0);

        // 1 bit of precisions
        assert_eq!(fract_u32_to_f32(0, 1), 0.0);
        assert_eq!(fract_u32_to_f32(1, 1), 0.5);

        // 2 bits of precision
        assert_eq!(fract_u32_to_f32(0, 2), 0.0);
        assert_eq!(fract_u32_to_f32(1, 2), 0.25);
        assert_eq!(fract_u32_to_f32(2, 2), 0.5);
        assert_eq!(fract_u32_to_f32(3, 2), 0.75);
    }

    #[test]
    fn test_quant_root() {
        assert_eq!(Quant::root().min_point_size(), (Vec3::ZERO, 1.0));
    }

    #[test]
    fn test_quant_child() {
        assert_eq!(
            Quant::root()
                .child(ChildIndex::new(0))
                .unwrap()
                .min_point_size(),
            (Vec3::ZERO, 0.5)
        );

        assert_eq!(
            Quant::root()
                .child(ChildIndex::new(0))
                .unwrap()
                .child(ChildIndex::new(0))
                .unwrap()
                .min_point_size(),
            (Vec3::ZERO, 0.25)
        );

        assert_eq!(
            Quant::root()
                .child(ChildIndex::new(5))
                .unwrap()
                .min_point_size(),
            (Vec3::new(0.5, 0.0, 0.5), 0.5)
        );

        assert_eq!(
            Quant::root()
                .child(ChildIndex::new(3))
                .unwrap()
                .min_point_size(),
            (Vec3::new(0.5, 0.5, 0.0), 0.5)
        );

        assert_eq!(
            Quant::root()
                .child(ChildIndex::new(3))
                .unwrap()
                .child(ChildIndex::new(6))
                .unwrap()
                .min_point_size(),
            (Vec3::new(0.5, 0.75, 0.25), 0.25)
        );
    }

    #[test]
    fn test_u32_msb_s() {
        struct Test;
        impl SimdTest for Test {
            #[inline(always)]
            fn run<S: Simd>(self, simd: S) {
                assert_eq!(msb_index_u32_s(u32s::splat(simd, 0)), 0);
                assert_eq!(msb_index_u32_s(u32s::splat(simd, 2)), 1);
                assert_eq!(msb_index_u32_s(u32s::splat(simd, 6)), 2);
                assert_eq!(msb_index_u32_s(u32s::splat(simd, (1u32 << 20) - 1)), 19);
                assert_eq!(msb_index_u32_s(u32s::splat(simd, (1u32 << 29) - 1)), 28);
                assert_eq!(msb_index_u32_s(u32s::splat(simd, 1u32 << 29)), 29);
                assert_eq!(msb_index_u32_s(u32s::splat(simd, 1u32 << 31)), 31);
                assert_eq!(msb_index_u32_s(u32s::splat(simd, u32::MAX)), 31);
            }
        }
        simd_test(Test);
    }

    #[test]
    fn test_f32_exp2_small_i32_s() {
        struct Test;
        impl SimdTest for Test {
            #[inline(always)]
            fn run<S: Simd>(self, simd: S) {
                // Zero
                assert_eq!(f32_exp2_small_i32_s(i32s::splat(simd, 0)), 1.0);

                // Positive
                assert_eq!(f32_exp2_small_i32_s(i32s::splat(simd, 1)), 2.0);
                assert_eq!(f32_exp2_small_i32_s(i32s::splat(simd, 3)), 8.0);
                assert_eq!(f32_exp2_small_i32_s(i32s::splat(simd, 4)), 16.0);

                // Negative
                assert_eq!(f32_exp2_small_i32_s(i32s::splat(simd, -1)), 0.5);
                assert_eq!(f32_exp2_small_i32_s(i32s::splat(simd, -3)), 0.125);
                assert_eq!(f32_exp2_small_i32_s(i32s::splat(simd, -4)), 0.0625);
            }
        }
        simd_test(Test);
    }

    #[test]
    fn test_fract_u32_to_f32_s() {
        struct Test;
        impl SimdTest for Test {
            #[inline(always)]
            fn run<S: Simd>(self, simd: S) {
                // num == 0
                assert_eq!(
                    fract_u32_to_f32_s(u32s::splat(simd, 0), u32s::splat(simd, 0)),
                    0.0
                );
                assert_eq!(
                    fract_u32_to_f32_s(u32s::splat(simd, 0), u32s::splat(simd, 1)),
                    0.0
                );
                assert_eq!(
                    fract_u32_to_f32_s(u32s::splat(simd, 0), u32s::splat(simd, 23)),
                    0.0
                );

                // 0 bits of precision
                assert_eq!(
                    fract_u32_to_f32_s(u32s::splat(simd, 1), u32s::splat(simd, 0)),
                    1.0
                );

                // 1 bit of precision
                assert_eq!(
                    fract_u32_to_f32_s(u32s::splat(simd, 0), u32s::splat(simd, 1)),
                    0.0
                );
                assert_eq!(
                    fract_u32_to_f32_s(u32s::splat(simd, 1), u32s::splat(simd, 1)),
                    0.5
                );

                // 2 bits of precision
                assert_eq!(
                    fract_u32_to_f32_s(u32s::splat(simd, 0), u32s::splat(simd, 2)),
                    0.0
                );
                assert_eq!(
                    fract_u32_to_f32_s(u32s::splat(simd, 1), u32s::splat(simd, 2)),
                    0.25
                );
                assert_eq!(
                    fract_u32_to_f32_s(u32s::splat(simd, 2), u32s::splat(simd, 2)),
                    0.5
                );
                assert_eq!(
                    fract_u32_to_f32_s(u32s::splat(simd, 3), u32s::splat(simd, 2)),
                    0.75
                );
            }
        }
        simd_test(Test);
    }

    #[test]
    fn test_quant_root_s() {
        struct Test;
        impl SimdTest for Test {
            #[inline(always)]
            fn run<S: Simd>(self, simd: S) {
                assert_eq!(
                    QuantS::root(simd).min_point_size(),
                    (Vec3::splat(f32s::splat(simd, 0.0)), f32s::splat(simd, 1.0))
                );
            }
        }
        simd_test(Test);
    }

    #[test]
    fn test_quant_child_s() {
        struct Test;
        impl SimdTest for Test {
            #[inline(always)]
            fn run<S: Simd>(self, simd: S) {
                assert_eq!(
                    QuantS::root(simd)
                        .child(ChildIndex::new(0))
                        .min_point_size(),
                    (Vec3::simd_splat(simd, 0.0), f32s::splat(simd, 0.5))
                );

                assert_eq!(
                    QuantS::root(simd)
                        .child(ChildIndex::new(0))
                        .child(ChildIndex::new(0))
                        .min_point_size(),
                    (Vec3::simd_splat(simd, 0.0), f32s::splat(simd, 0.25))
                );

                assert_eq!(
                    QuantS::root(simd)
                        .child(ChildIndex::new(5))
                        .min_point_size(),
                    (Vec3::simd_new(simd, 0.5, 0.0, 0.5), f32s::splat(simd, 0.5))
                );

                assert_eq!(
                    QuantS::root(simd)
                        .child(ChildIndex::new(3))
                        .min_point_size(),
                    (Vec3::simd_new(simd, 0.5, 0.5, 0.0), f32s::splat(simd, 0.5))
                );

                assert_eq!(
                    QuantS::root(simd)
                        .child(ChildIndex::new(3))
                        .child(ChildIndex::new(6))
                        .min_point_size(),
                    (
                        Vec3::simd_new(simd, 0.5, 0.75, 0.25),
                        f32s::splat(simd, 0.25)
                    )
                );
            }
        }
        simd_test(Test);
    }
}
