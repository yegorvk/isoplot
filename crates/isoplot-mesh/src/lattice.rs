use std::ops::BitOr;

use glam::{UVec3, uvec3};

mod tables;
mod traverse;

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

    pub const fn new(axis: AxisKind) -> Self {
        match axis {
            AxisKind::X => Self::X,
            AxisKind::Y => Self::Y,
            AxisKind::Z => Self::Z,
        }
    }

    pub(crate) fn enumerate() -> impl Iterator<Item = Offset> {
        (0..8).map(Offset)
    }

    const fn from_components(x: bool, y: bool, z: bool) -> Self {
        Self(x as u8 | (y as u8 * 2) | (z as u8 * 4))
    }

    pub const fn as_uvec3(self) -> UVec3 {
        let x = (self.0 & Self::X.0 != 0) as u32;
        let y = (self.0 & Self::Y.0 != 0) as u32;
        let z = (self.0 & Self::Z.0 != 0) as u32;
        uvec3(x, y, z)
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
