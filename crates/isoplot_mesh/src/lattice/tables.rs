use super::primitives::{Corner, EdgeKind, EdgeSlot, FaceIndex, FaceKind, FaceSlot, Offset};
use std::array;

const fn c(corner: u8) -> Corner {
    Corner(Offset::from_components(
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
    for indices in CELL_FACES[kind.0 as usize] {
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
    for indices in CELL_EDGES[kind.0 as usize] {
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
    for corners in SUB_FACES[kind.0 as usize] {
        f(array::from_fn(|i| {
            refine(FaceSlot::new(i as u8), corners[i])
        }))
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
        f(indices.map(|(i, which)| refine(FaceSlot::new(i), which)));
    }
}

pub(crate) const fn face_edge_slot(kind: (FaceKind, EdgeKind), slot: EdgeSlot) -> FaceSlot {
    FaceSlot::new(face_edges(kind)[0][slot.index()].0)
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
    for indices in SUB_EDGES[kind.0 as usize] {
        f(array::from_fn(|i| {
            refine(EdgeSlot::new(i as u8), indices[i])
        }));
    }
}

#[inline]
pub(crate) const fn edge_normal_face(kind: EdgeKind, slot: EdgeSlot) -> FaceIndex {
    FaceIndex::new(kind.normal_face(), slot.opposite())
}
