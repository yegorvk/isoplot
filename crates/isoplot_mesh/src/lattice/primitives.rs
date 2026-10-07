use std::ops;

use crate::utils::array_map_index;
use isoplot_math::Vec3;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[repr(u8)]
pub enum AxisKind {
    X = 0,
    Y = 1,
    Z = 2,
}

impl AxisKind {
    #[inline]
    const fn complement(self) -> [AxisKind; 2] {
        match self {
            AxisKind::X => [AxisKind::Y, AxisKind::Z],
            AxisKind::Y => [AxisKind::X, AxisKind::Z],
            AxisKind::Z => [AxisKind::X, AxisKind::Y],
        }
    }

    #[inline]
    const fn from_u8(value: u8) -> Self {
        assert!(value < 3);
        match value {
            0 => AxisKind::X,
            1 => AxisKind::Y,
            2 => AxisKind::Z,
            _ => unreachable!(),
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Offset(u8);

impl Offset {
    pub(crate) const ZERO: Self = Self(0);

    pub(crate) const ALL: [Offset; 8] = {
        let mut a = [Self::ZERO; 8];
        let mut i = 0u8;
        while i < 8 {
            a[i as usize] = Self(i);
            i += 1;
        }
        a
    };

    #[inline]
    pub(crate) const fn new(axis: AxisKind) -> Self {
        Self(1u8 << (axis as u8))
    }

    #[inline]
    pub(crate) const fn from_components(x: bool, y: bool, z: bool) -> Self {
        Self(x as u8 | ((y as u8) << 1) | ((z as u8) << 2))
    }

    #[inline]
    pub const fn as_vec3(self) -> Vec3<bool> {
        Vec3::new(self.0 & 1 != 0, self.0 & 2 != 0, self.0 & 4 != 0)
    }

    #[inline]
    pub(crate) const fn as_u8(self) -> u8 {
        self.0
    }
}

impl std::ops::BitOr for Offset {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
struct Offset2(u8);

impl Offset2 {
    const ZERO: Self = Self(0);

    #[inline]
    const fn from_gray(index: u8) -> Self {
        assert!(index < 4);
        Self(index ^ (index >> 1))
    }

    #[inline]
    const fn gray_index(self) -> u8 {
        self.0 ^ (self.0 >> 1)
    }

    #[inline]
    const fn complement(self) -> Self {
        Self(3 ^ self.0)
    }

    #[inline]
    const fn project(self, u: AxisKind, v: AxisKind) -> Offset {
        let u_bit = (self.0 & 1 != 0) as u8;
        let v_bit = (self.0 & 2 != 0) as u8;
        Offset((u_bit << (u as u8)) | (v_bit << (v as u8)))
    }
}

impl std::ops::BitOr for Offset2 {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Corner(pub(crate) Offset);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct FaceKind(pub(crate) AxisKind);

impl FaceKind {
    pub(crate) const X: Self = Self(AxisKind::X);
    pub(crate) const Y: Self = Self(AxisKind::Y);
    pub(crate) const Z: Self = Self(AxisKind::Z);

    pub const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    #[inline]
    pub(crate) const fn middle_edge(self) -> EdgeKind {
        EdgeKind(self.0)
    }

    #[inline]
    pub(crate) fn adjacent_edges(self) -> [EdgeKind; 2] {
        self.0.complement().map(EdgeKind)
    }

    #[inline]
    pub(crate) const fn normal(self) -> Offset {
        Offset::new(self.0)
    }

    #[inline]
    pub(crate) const fn slot_offset(self, slot: FaceSlot) -> Offset {
        Offset(slot.0 << (self.0 as u8))
    }

    #[inline]
    pub fn offsets(self) -> [Offset; 2] {
        FaceSlot::ALL.map(|slot| self.slot_offset(slot))
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct EdgeKind(pub(crate) AxisKind);

impl EdgeKind {
    pub(crate) const X: Self = Self(AxisKind::X);
    pub(crate) const Y: Self = Self(AxisKind::Y);
    pub(crate) const Z: Self = Self(AxisKind::Z);

    pub const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    #[inline]
    pub(crate) const fn normal_face(self) -> FaceKind {
        FaceKind(self.0)
    }

    #[inline]
    pub(crate) fn adjacent_faces(self) -> [FaceKind; 2] {
        self.0.complement().map(FaceKind)
    }

    #[inline]
    pub(crate) fn slot_offset(self, slot: EdgeSlot) -> Offset {
        let [a, b] = self.adjacent_faces();
        slot.0.project(a.0, b.0)
    }

    #[inline]
    pub fn offsets(self) -> [Offset; 4] {
        EdgeSlot::ALL.map(|slot| self.slot_offset(slot))
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) struct FaceSlot(u8);

impl FaceSlot {
    pub(crate) const ALL: [Self; 2] = [Self(0), Self(1)];

    #[inline]
    pub(crate) const fn new(which: u8) -> Self {
        assert!(which < 2);
        Self(which)
    }

    #[inline]
    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) struct EdgeSlot(Offset2);

impl EdgeSlot {
    pub(crate) const ALL: [Self; 4] = {
        let mut a = [Self(Offset2::ZERO); 4];

        let mut i = 0usize;
        while i < 4 {
            a[i] = Self(Offset2::from_gray(i as u8));
            i += 1;
        }

        a
    };

    #[inline]
    pub(crate) const fn new(which: u8) -> Self {
        assert!(which < 4);
        Self(Offset2::from_gray(which))
    }

    #[inline]
    pub(crate) const fn opposite(self) -> Self {
        Self(self.0.complement())
    }

    #[inline]
    pub(crate) const fn index(self) -> usize {
        self.0.gray_index() as usize
    }
}

pub(crate) struct Face<T> {
    pub(crate) kind: FaceKind,
    pub(crate) cells: [T; 2],
}

impl<T> Face<T> {
    #[inline]
    pub(crate) fn new(kind: FaceKind, cells: [T; 2]) -> Self {
        Self { kind, cells }
    }

    #[inline]
    pub(crate) fn try_from_fn<F, E>(mut key: FaceKey<T>, mut f: F) -> Result<Self, E>
    where
        F: FnMut(&mut T, Offset) -> Result<T, E>,
    {
        let p = f(&mut key.min_cell, key.kind.normal())?;
        Ok(Self {
            kind: key.kind,
            cells: [key.min_cell, p],
        })
    }

    #[inline]
    pub(crate) fn map<F, U>(self, f: F) -> Face<U>
    where
        F: FnMut(T) -> U,
    {
        Face::new(self.kind, self.cells.map(f))
    }

    #[inline]
    pub(crate) fn each_ref(&self) -> Face<&T> {
        let cells = self.cells.each_ref();
        Face {
            kind: self.kind,
            cells,
        }
    }
}

impl<T> ops::Index<FaceSlot> for Face<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: FaceSlot) -> &Self::Output {
        &self.cells[index.0 as usize]
    }
}

impl<T> ops::IndexMut<FaceSlot> for Face<T> {
    #[inline]
    fn index_mut(&mut self, index: FaceSlot) -> &mut Self::Output {
        &mut self.cells[index.0 as usize]
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Edge<T> {
    pub(crate) kind: EdgeKind,
    pub(crate) cells: [T; 4],
}

impl<T> Edge<T> {
    #[inline]
    pub(crate) fn new(kind: EdgeKind, cells: [T; 4]) -> Self {
        Self { kind, cells }
    }

    #[inline]
    pub(crate) fn try_from_fn<F, E>(mut key: EdgeKey<T>, mut f: F) -> Result<Self, E>
    where
        F: FnMut(&mut T, Offset) -> Result<T, E>,
    {
        let [a, b] = key.kind.adjacent_faces().map(FaceKind::normal);

        let pn = f(&mut key.min_cell, a)?;
        let pp = f(&mut key.min_cell, a | b)?;
        let np = f(&mut key.min_cell, b)?;

        Ok(Self {
            kind: key.kind,
            cells: [key.min_cell, pn, pp, np],
        })
    }

    #[inline]
    pub(crate) fn slot_cells(&self) -> impl Iterator<Item = (EdgeSlot, &T)> {
        EdgeSlot::ALL
            .into_iter()
            .map(|slot| (slot, &self.cells[slot.index()]))
    }

    #[inline]
    pub(crate) fn map_slot<F, U>(self, mut f: F) -> Edge<U>
    where
        F: FnMut(EdgeSlot, T) -> U,
    {
        let cells = array_map_index(self.cells, |i, cell| {
            let slot = EdgeSlot::new(i as u8);
            f(slot, cell)
        });

        Edge::new(self.kind, cells)
    }

    #[inline]
    pub(crate) fn map<F, U>(self, f: F) -> Edge<U>
    where
        F: FnMut(T) -> U,
    {
        Edge::new(self.kind, self.cells.map(f))
    }

    #[inline]
    pub(crate) fn each_ref(&self) -> Edge<&T> {
        let cells = self.cells.each_ref();
        Edge {
            kind: self.kind,
            cells,
        }
    }
}

impl<T> ops::Index<EdgeSlot> for Edge<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: EdgeSlot) -> &Self::Output {
        &self.cells[index.index()]
    }
}

impl<T> ops::IndexMut<EdgeSlot> for Edge<T> {
    #[inline]
    fn index_mut(&mut self, index: EdgeSlot) -> &mut Self::Output {
        &mut self.cells[index.index()]
    }
}

pub(crate) struct FaceKey<T> {
    kind: FaceKind,
    min_cell: T,
}

impl<T> FaceKey<T> {
    #[inline]
    pub(crate) const fn new(kind: FaceKind, min_cell: T) -> Self {
        Self { kind, min_cell }
    }
}

pub(crate) struct EdgeKey<T> {
    kind: EdgeKind,
    min_cell: T,
}

impl<T> EdgeKey<T> {
    #[inline]
    pub(crate) const fn new(kind: EdgeKind, min_cell: T) -> Self {
        Self { kind, min_cell }
    }
}

/// An index of an "interior" face in an octree cell.
///
/// Each face is composed of two face-adjacent sub-cells.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) struct FaceIndex(u8);

impl FaceIndex {
    pub(crate) const ALL: [Self; 12] = {
        let mut a = [Self(0u8); 12];

        let mut i = 0usize;
        while i < 12 {
            a[i] = Self(i as u8);
            i += 1;
        }

        a
    };

    #[inline]
    pub(crate) const fn new(kind: FaceKind, slot: EdgeSlot) -> Self {
        Self(((kind.0 as u8) << 2) | slot.0.0)
    }

    #[inline]
    pub(crate) const fn kind(self) -> FaceKind {
        FaceKind(AxisKind::from_u8(self.0 >> 2))
    }

    #[inline]
    pub(crate) const fn slot(self) -> EdgeSlot {
        EdgeSlot(Offset2(self.0 & 3))
    }

    #[inline]
    const fn into_parts(self) -> (FaceKind, EdgeSlot) {
        (self.kind(), self.slot())
    }
}

/// An index of a "boundary" edge of an octree cell.
///
/// It is just a newtype for [`FaceIndex`] since there is a natural
/// isomorphism between them.
pub(crate) struct BEdgeIndex(pub(crate) FaceIndex);

impl BEdgeIndex {
    #[inline]
    pub(crate) fn corners(self) -> [Corner; 2] {
        self.corner_offsets().map(Corner)
    }

    #[inline]
    fn corner_offsets(self) -> [Offset; 2] {
        let (kind, edge_slot) = self.0.into_parts();
        FaceSlot::ALL.map(|face_slot| {
            kind.middle_edge().slot_offset(edge_slot) | kind.slot_offset(face_slot)
        })
    }
}

impl From<FaceIndex> for BEdgeIndex {
    #[inline]
    fn from(value: FaceIndex) -> Self {
        Self(value)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) struct BEdgesMask(u16);

impl BEdgesMask {
    #[inline]
    pub(crate) fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(BEdgeIndex) -> bool,
    {
        let mut mask = 0u16;

        for index in FaceIndex::ALL {
            if f(BEdgeIndex(index)) {
                let raw_index = index.0;
                debug_assert!(raw_index < 16);
                mask |= 1u16 << raw_index;
            }
        }

        Self(mask)
    }

    #[inline]
    pub(crate) const fn contains(self, index: BEdgeIndex) -> bool {
        let raw_index = index.0.0;
        debug_assert!(raw_index < 16);
        self.0 & (1u16 << raw_index) != 0
    }
}
