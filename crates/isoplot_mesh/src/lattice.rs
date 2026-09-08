use std::ops::BitOr;

mod tables;
mod traverse;

use crate::math::Vec3;
pub(crate) use tables::{
    Corner, Edge, EdgeKey, EdgeKind, EdgeSlot, Face, FaceKey, FaceKind, FaceSlot, edge_corners,
    face_edge_slot, for_each_cell_edge, for_each_cell_face,
};
pub(crate) use traverse::{Edges, Faces, MinimalEdges, TraverseOctree};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum AxisKind {
    X = 0,
    Y = 1,
    Z = 2,
}

impl AxisKind {
    pub const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Offset(u8);

impl Offset {
    const ZERO: Self = Self(0);

    const X: Self = Self(1);
    const Y: Self = Self(2);
    const Z: Self = Self(4);

    pub(crate) const ALL: [Offset; 8] = {
        let mut a = [Self::ZERO; 8];
        let mut i = 0u8;
        while i < 8 {
            a[i as usize] = Self(i);
            i += 1;
        }
        a
    };

    pub const fn new(axis: AxisKind) -> Self {
        match axis {
            AxisKind::X => Self::X,
            AxisKind::Y => Self::Y,
            AxisKind::Z => Self::Z,
        }
    }

    const fn from_components(x: bool, y: bool, z: bool) -> Self {
        Self(x as u8 | (y as u8 * 2) | (z as u8 * 4))
    }

    pub const fn as_vec3(self) -> Vec3<bool> {
        Vec3::new(
            self.0 & Self::X.0 != 0,
            self.0 & Self::Y.0 != 0,
            self.0 & Self::Z.0 != 0,
        )
    }

    pub(crate) const fn as_u8(self) -> u8 {
        self.0
    }
}

impl BitOr for Offset {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}
