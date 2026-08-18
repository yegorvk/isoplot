use super::{AxisKind, Offset};
use std::array;

#[derive(Copy, Clone, Debug)]
pub(crate) struct Corner(Offset);

impl Corner {
    pub(crate) const fn new(offset: Offset) -> Self {
        Self(offset)
    }

    pub(crate) const fn offset(self) -> Offset {
        self.0
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub(crate) struct FaceKind(AxisKind);

impl FaceKind {
    const X: Self = Self(AxisKind::X);
    const Y: Self = Self(AxisKind::Y);
    const Z: Self = Self(AxisKind::Z);

    pub(crate) const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    pub(crate) const fn axis(self) -> AxisKind {
        self.0
    }

    pub(crate) const fn tangent_edges(self) -> [EdgeKind; 2] {
        match self.0 {
            AxisKind::X => [EdgeKind::Y, EdgeKind::Z],
            AxisKind::Y => [EdgeKind::X, EdgeKind::Z],
            AxisKind::Z => [EdgeKind::X, EdgeKind::Y],
        }
    }

    const fn normal(self) -> Offset {
        Offset::new(self.0)
    }

    pub(crate) const fn slot_offset(self, slot: FaceSlot) -> Offset {
        match slot.0 {
            0 => Offset::ZERO,
            _ => self.normal(),
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub(crate) struct EdgeKind(AxisKind);

impl EdgeKind {
    const X: Self = Self(AxisKind::X);
    const Y: Self = Self(AxisKind::Y);
    const Z: Self = Self(AxisKind::Z);

    pub(crate) const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    pub(crate) const fn axis(self) -> AxisKind {
        self.0
    }

    const fn perp_faces(self) -> [FaceKind; 2] {
        match self.0 {
            AxisKind::X => [FaceKind::Y, FaceKind::Z],
            AxisKind::Y => [FaceKind::X, FaceKind::Z],
            AxisKind::Z => [FaceKind::X, FaceKind::Y],
        }
    }

    const fn perp_normals(self) -> [Offset; 2] {
        let [a, b] = self.perp_faces();
        [a.normal(), b.normal()]
    }

    pub(crate) fn slot_offset(self, slot: EdgeSlot) -> Offset {
        let [a, b] = self.perp_normals();

        match slot.0 {
            0 => Offset::ZERO,
            1 => a,
            2 => a | b,
            _ => b,
        }
    }
}

const fn c(corner: u8) -> Corner {
    Corner::new(Offset::from_components(
        corner & 1 != 0,
        corner & 2 != 0,
        corner & 4 != 0,
    ))
}

const CELL_FACES: [[[Corner; 2]; 4]; 3] = [
    [[c(0), c(1)], [c(2), c(3)], [c(6), c(7)], [c(4), c(5)]],
    [[c(0), c(2)], [c(1), c(3)], [c(5), c(7)], [c(4), c(6)]],
    [[c(0), c(4)], [c(1), c(5)], [c(3), c(7)], [c(2), c(6)]],
];

pub(crate) fn for_each_cell_face<B, R, F>(kind: FaceKind, mut refine: R, mut f: F)
where
    R: FnMut(Corner) -> B,
    F: FnMut([B; 2]),
{
    for indices in CELL_FACES[kind.axis() as usize] {
        f(indices.map(&mut refine))
    }
}

const CELL_EDGES: [[[Corner; 4]; 2]; 3] = [
    [[c(0), c(2), c(6), c(4)], [c(1), c(3), c(7), c(5)]],
    [[c(0), c(1), c(5), c(4)], [c(2), c(3), c(7), c(6)]],
    [[c(0), c(1), c(3), c(2)], [c(4), c(5), c(7), c(6)]],
];

pub(crate) fn for_each_cell_edge<B, R, F>(kind: EdgeKind, mut refine: R, mut f: F)
where
    R: FnMut(Corner) -> B,
    F: FnMut([B; 4]),
{
    for indices in CELL_EDGES[kind.axis() as usize] {
        f(indices.map(&mut refine))
    }
}

const SUB_FACES: [[[Corner; 2]; 4]; 3] = [
    [[c(1), c(0)], [c(3), c(2)], [c(7), c(6)], [c(5), c(4)]],
    [[c(2), c(0)], [c(3), c(1)], [c(7), c(5)], [c(6), c(4)]],
    [[c(4), c(0)], [c(5), c(1)], [c(7), c(3)], [c(6), c(2)]],
];

pub(crate) fn for_each_sub_face<B, R, F>(kind: FaceKind, mut refine: R, mut f: F)
where
    R: FnMut(FaceSlot, Corner) -> B,
    F: FnMut([B; 2]),
{
    for corners in SUB_FACES[kind.axis() as usize] {
        f(array::from_fn(|i| refine(FaceSlot(i as u8), corners[i])))
    }
}

#[allow(clippy::type_complexity)]
const FACE_EDGES: [[[[(u8, Corner); 4]; 2]; 2]; 3] = [
    [
        [
            [(0, c(1)), (1, c(0)), (1, c(4)), (0, c(5))],
            [(0, c(3)), (1, c(2)), (1, c(6)), (0, c(7))],
        ],
        [
            [(0, c(1)), (1, c(0)), (1, c(2)), (0, c(3))],
            [(0, c(5)), (1, c(4)), (1, c(6)), (0, c(7))],
        ],
    ],
    [
        [
            [(0, c(2)), (1, c(0)), (1, c(4)), (0, c(6))],
            [(0, c(3)), (1, c(1)), (1, c(5)), (0, c(7))],
        ],
        [
            [(0, c(2)), (0, c(3)), (1, c(1)), (1, c(0))],
            [(0, c(6)), (0, c(7)), (1, c(5)), (1, c(4))],
        ],
    ],
    [
        [
            [(0, c(4)), (0, c(6)), (1, c(2)), (1, c(0))],
            [(0, c(5)), (0, c(7)), (1, c(3)), (1, c(1))],
        ],
        [
            [(0, c(4)), (0, c(5)), (1, c(1)), (1, c(0))],
            [(0, c(6)), (0, c(7)), (1, c(3)), (1, c(2))],
        ],
    ],
];

const fn face_edges(kind: (FaceKind, EdgeKind)) -> [[(u8, Corner); 4]; 2] {
    match kind {
        (FaceKind::X, EdgeKind::Y) => FACE_EDGES[0][0],
        (FaceKind::X, EdgeKind::Z) => FACE_EDGES[0][1],
        (FaceKind::Y, EdgeKind::X) => FACE_EDGES[1][0],
        (FaceKind::Y, EdgeKind::Z) => FACE_EDGES[1][1],
        (FaceKind::Z, EdgeKind::X) => FACE_EDGES[2][0],
        (FaceKind::Z, EdgeKind::Y) => FACE_EDGES[2][1],
        _ => unreachable!(),
    }
}

pub(crate) fn for_each_face_edge<B, R, F>(kind: (FaceKind, EdgeKind), mut refine: R, mut f: F)
where
    R: FnMut(FaceSlot, Corner) -> B,
    F: FnMut([B; 4]),
{
    for indices in face_edges(kind) {
        f(indices.map(|(i, which)| refine(FaceSlot(i), which)));
    }
}

pub(crate) const fn face_edge_slot(kind: (FaceKind, EdgeKind), slot: EdgeSlot) -> FaceSlot {
    FaceSlot(face_edges(kind)[0][slot.0 as usize].0)
}

const SUB_EDGES: [[[Corner; 4]; 2]; 3] = [
    [[c(6), c(4), c(0), c(2)], [c(7), c(5), c(1), c(3)]],
    [[c(5), c(4), c(0), c(1)], [c(7), c(6), c(2), c(3)]],
    [[c(3), c(2), c(0), c(1)], [c(7), c(6), c(4), c(5)]],
];

pub(super) fn for_each_sub_edge<B, R, F>(kind: EdgeKind, mut refine: R, mut f: F)
where
    R: FnMut(EdgeSlot, Corner) -> B,
    F: FnMut([B; 4]),
{
    for indices in SUB_EDGES[kind.axis() as usize] {
        f(array::from_fn(|i| refine(EdgeSlot(i as u8), indices[i])));
    }
}

const EDGE_CORNERS: [[[Corner; 2]; 4]; 3] = [
    [[c(6), c(7)], [c(4), c(5)], [c(0), c(1)], [c(2), c(3)]],
    [[c(5), c(7)], [c(4), c(6)], [c(0), c(2)], [c(1), c(3)]],
    [[c(3), c(7)], [c(2), c(6)], [c(0), c(4)], [c(1), c(5)]],
];

pub(crate) fn edge_corners<B, R>(
    kind: EdgeKind,
    refine: R,
) -> impl Iterator<Item = (EdgeSlot, [B; 2])>
where
    R: Fn(EdgeSlot, Corner) -> B,
{
    EDGE_CORNERS[kind.axis() as usize]
        .iter()
        .enumerate()
        .map(move |(slot, corners)| {
            let slot = EdgeSlot(slot as u8);
            (slot, array::from_fn(|i| refine(slot, corners[i])))
        })
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) struct FaceSlot(u8);

impl FaceSlot {
    pub(crate) const ALL: [Self; 2] = [Self(0), Self(1)];

    pub(crate) fn as_usize(self) -> usize {
        self.0 as usize
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) struct EdgeSlot(u8);

impl EdgeSlot {
    pub(crate) const ALL: [Self; 4] = [Self(0), Self(1), Self(2), Self(3)];

    pub(crate) fn as_usize(self) -> usize {
        self.0 as usize
    }
}

pub(crate) struct Face<T>(pub [T; 2]);

impl<T> Face<T> {
    pub(crate) fn try_from_fn<F, E>(mut key: FaceKey<T>, mut f: F) -> Result<Self, E>
    where
        F: FnMut(&mut T, Offset) -> Result<T, E>,
    {
        let positive = f(&mut key.min_cell, key.kind.normal())?;
        Ok(Self([key.min_cell, positive]))
    }
}

pub(crate) struct Edge<T>(pub [T; 4]);

impl<T> Edge<T> {
    pub(crate) fn try_from_fn<F, E>(mut key: EdgeKey<T>, mut f: F) -> Result<Self, E>
    where
        F: FnMut(&mut T, Offset) -> Result<T, E>,
    {
        let [a, b] = key.kind.perp_normals();

        let pn = f(&mut key.min_cell, a)?;
        let pp = f(&mut key.min_cell, a | b)?;
        let np = f(&mut key.min_cell, b)?;

        Ok(Self([key.min_cell, pn, pp, np]))
    }
}

pub(crate) struct FaceKey<T> {
    pub kind: FaceKind,
    pub min_cell: T,
}

impl<T> FaceKey<T> {
    pub(crate) fn new(kind: FaceKind, min_cell: T) -> Self {
        Self { kind, min_cell }
    }
}

pub(crate) struct EdgeKey<T> {
    pub kind: EdgeKind,
    pub min_cell: T,
}

impl<T> EdgeKey<T> {
    pub(crate) fn new(kind: EdgeKind, min_cell: T) -> Self {
        Self { kind, min_cell }
    }
}
