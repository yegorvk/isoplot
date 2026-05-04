use bilge::prelude::*;
use glam::{U16Vec3, Vec3, u16vec3, vec3};

use crate::octree::{ChildIndex, Payload};

/// A quantized point in a unit cube
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Quant(RawQuant);

impl Quant {
    /// Maximum number of consequent subdivisions
    pub const MAX_SUBDIV: u8 = 10;

    pub fn root() -> Self {
        Self(RawQuant::root())
    }

    pub fn child(self, index: ChildIndex) -> Option<Quant> {
        self.0.child(index.0).map(|child| Quant(child))
    }

    pub fn min_point_size(self) -> (Vec3, f32) {
        let (parts, level) = self.0.parts_level();

        let x = fract_u32_to_f32(parts.x as u32, level as u32);
        let y = fract_u32_to_f32(parts.y as u32, level as u32);
        let z = fract_u32_to_f32(parts.z as u32, level as u32);

        (vec3(x, y, z), f32_exp2_small(-(level as i8)))
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

#[bitsize(31)]
#[derive(Copy, Clone, DebugBits)]
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

    fn parts_level(self) -> (U16Vec3, u8) {
        let ((raw_x, y, z), level) = self.raw_parts_level();
        (u16vec3(raw_x ^ (1u16 << level), y, z), level)
    }

    fn raw_parts_level(self) -> ((u16, u16, u16), u8) {
        let (raw_x, y, z) = self.raw_parts();
        ((raw_x, y, z), quant_level(raw_x))
    }

    fn raw_parts(self) -> (u16, u16, u16) {
        (self.x().value(), self.y().value(), self.z().value())
    }
}

fn quant_level(raw_x: u16) -> u8 {
    let level = (15 - raw_x.leading_zeros()) as u8;
    debug_assert!(level < Quant::MAX_SUBDIV);
    level
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
    let fract = (num as u32) << (23u32 - len);
    f32::from_bits((127u32 << 23u32) + fract) - 1f32
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Zero
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
        assert_eq!(Quant::root().min_point_size(), (Vec3::ZERO, 1f32));
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
            (vec3(0.5, 0.0, 0.5), 0.5)
        );

        assert_eq!(
            Quant::root()
                .child(ChildIndex::new(3))
                .unwrap()
                .min_point_size(),
            (vec3(0.5, 0.5, 0.0), 0.5)
        );

        assert_eq!(
            Quant::root()
                .child(ChildIndex::new(3))
                .unwrap()
                .child(ChildIndex::new(6))
                .unwrap()
                .min_point_size(),
            (vec3(0.5, 0.75, 0.25), 0.25)
        );
    }
}
