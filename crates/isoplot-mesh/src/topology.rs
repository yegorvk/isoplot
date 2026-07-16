use glam::IVec3;
use std::{array, iter, ops::Index};

#[derive(Copy, Clone, Debug)]
pub struct Corner(u8);

impl Corner {
    pub const fn new(corner: u8) -> Self {
        assert!(corner < 8);
        Self(corner)
    }

    pub const fn as_u8(self) -> u8 {
        self.0
    }
}

#[derive(Copy, Clone, Debug)]
pub enum FaceKind {
    X = 0,
    Y = 1,
    Z = 2,
}

impl FaceKind {
    pub const fn tangent_edges(self) -> [EdgeKind; 2] {
        match self {
            FaceKind::X => [EdgeKind::Y, EdgeKind::Z],
            FaceKind::Y => [EdgeKind::X, EdgeKind::Z],
            FaceKind::Z => [EdgeKind::X, EdgeKind::Y],
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum EdgeKind {
    X = 0,
    Y = 1,
    Z = 2,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum Side {
    Negative = 0,
    Positive = 1,
}

impl Side {
    const fn opposite(self) -> Self {
        match self {
            Side::Negative => Side::Positive,
            Side::Positive => Side::Negative,
        }
    }
}

const fn c(corner: u8) -> Corner {
    Corner::new(corner)
}

const CELL_FACES: [[[Corner; 2]; 4]; 3] = [
    [[c(0), c(1)], [c(2), c(3)], [c(6), c(7)], [c(4), c(5)]],
    [[c(0), c(2)], [c(1), c(3)], [c(5), c(7)], [c(4), c(6)]],
    [[c(0), c(4)], [c(1), c(5)], [c(3), c(7)], [c(2), c(6)]],
];

pub fn for_each_cell_face<T, B, R, F>(cell: T, kind: FaceKind, mut refine: R, mut f: F)
where
    R: FnMut(&T, Corner) -> B,
    F: FnMut([B; 2]),
{
    for indices in CELL_FACES[kind as usize] {
        f(indices.map(|which| refine(&cell, which)))
    }
}

const CELL_EDGES: [[[Corner; 4]; 2]; 3] = [
    [[c(0), c(2), c(6), c(4)], [c(1), c(3), c(7), c(5)]],
    [[c(0), c(1), c(5), c(4)], [c(2), c(3), c(7), c(6)]],
    [[c(0), c(1), c(3), c(2)], [c(4), c(5), c(7), c(6)]],
];

pub fn for_each_cell_edge<T, B, R, F>(cell: T, kind: EdgeKind, mut refine: R, mut f: F)
where
    R: FnMut(&T, Corner) -> B,
    F: FnMut([B; 4]),
{
    for indices in CELL_EDGES[kind as usize] {
        f(indices.map(|which| refine(&cell, which)))
    }
}

/// An exterior face index
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct FaceIndex {
    /// The face index within the `CELL_EDGES` array
    ///
    /// Exterior faces are dual to interior edges during subdivision,
    /// so we can represent exterior faces using an index into the
    /// `CELL_EDGES` array (when flattened).
    idx: u8,
}

impl FaceIndex {
    const fn new(kind: FaceKind, which: u8) -> Self {
        // There are 2 faces per normal direction.
        assert!(which < 2);

        Self {
            idx: kind as u8 * 2 + which,
        }
    }

    fn for_each<F>(mut f: F)
    where
        F: FnMut(FaceIndex),
    {
        for idx in 0..6u8 {
            f(Self { idx })
        }
    }

    const fn kind(self) -> FaceKind {
        match self.idx / 2 {
            0 => FaceKind::X,
            1 => FaceKind::Y,
            _ => FaceKind::Z,
        }
    }

    const fn as_ivec3(self) -> IVec3 {
        let side = (self.idx % 2) as i32 * 2 - 1;

        match self.idx / 2 {
            0 => IVec3::new(side, 0, 0),
            1 => IVec3::new(0, side, 0),
            _ => IVec3::new(0, 0, side),
        }
    }

    const fn side(self) -> Side {
        if self.idx.is_multiple_of(2) {
            Side::Negative
        } else {
            Side::Positive
        }
    }
}

/// An exterior edge index
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct EdgeIndex {
    /// The edge index within the `CELL_FACES` array
    ///
    /// Exterior edges are dual to interior faces during subdivision,
    /// so we can represent exterior edges using an index into the
    /// `CELL_FACES` array (when flattened).
    idx: u8,
}

impl EdgeIndex {
    fn for_each<F>(mut f: F)
    where
        F: FnMut(EdgeIndex),
    {
        for idx in 0..12u8 {
            f(Self { idx })
        }
    }

    const fn kind(self) -> EdgeKind {
        match self.idx / 4 {
            0 => EdgeKind::X,
            1 => EdgeKind::Y,
            _ => EdgeKind::Z,
        }
    }

    const fn as_ivec3(self) -> IVec3 {
        let [a, b] = self.faces();
        let (a, b) = (a.as_ivec3(), b.as_ivec3());
        IVec3::new(a.x + b.x, a.y + b.y, a.z + b.z)
    }
}

const fn face_index_x(which: u8) -> FaceIndex {
    FaceIndex::new(FaceKind::X, which)
}

const fn face_index_y(which: u8) -> FaceIndex {
    FaceIndex::new(FaceKind::Y, which)
}

const fn face_index_z(which: u8) -> FaceIndex {
    FaceIndex::new(FaceKind::Z, which)
}

const CELL_EDGE_FACES: [[FaceIndex; 2]; 12] = [
    [face_index_y(0), face_index_z(0)],
    [face_index_y(1), face_index_z(0)],
    [face_index_y(1), face_index_z(1)],
    [face_index_y(0), face_index_z(1)],
    [face_index_x(0), face_index_z(0)],
    [face_index_x(1), face_index_z(0)],
    [face_index_x(1), face_index_z(1)],
    [face_index_x(0), face_index_z(1)],
    [face_index_x(0), face_index_y(0)],
    [face_index_x(1), face_index_y(0)],
    [face_index_x(1), face_index_y(1)],
    [face_index_x(0), face_index_y(1)],
];

impl EdgeIndex {
    const fn faces(self) -> [FaceIndex; 2] {
        CELL_EDGE_FACES[self.idx as usize]
    }
}

const SUB_FACES: [[[Corner; 2]; 4]; 3] = [
    [[c(1), c(0)], [c(3), c(2)], [c(7), c(6)], [c(5), c(4)]],
    [[c(2), c(0)], [c(3), c(1)], [c(7), c(5)], [c(6), c(4)]],
    [[c(4), c(0)], [c(5), c(1)], [c(7), c(3)], [c(6), c(2)]],
];

pub fn for_each_sub_face<T, B, R, F>(face: [T; 2], kind: FaceKind, refine: R, mut f: F)
where
    R: Fn(&T, Corner) -> B,
    F: FnMut([B; 2]),
{
    for indices in SUB_FACES[kind as usize] {
        f([refine(&face[0], indices[0]), refine(&face[1], indices[1])])
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

pub fn for_each_face_edge<T, B, R, F>(face: [T; 2], kind: (FaceKind, EdgeKind), refine: R, mut f: F)
where
    R: Fn(&T, Corner) -> B,
    F: FnMut([B; 4]),
{
    let edges = match kind {
        (FaceKind::X, EdgeKind::Y) => FACE_EDGES[0][0],
        (FaceKind::X, EdgeKind::Z) => FACE_EDGES[0][1],
        (FaceKind::Y, EdgeKind::X) => FACE_EDGES[1][0],
        (FaceKind::Y, EdgeKind::Z) => FACE_EDGES[1][1],
        (FaceKind::Z, EdgeKind::X) => FACE_EDGES[2][0],
        (FaceKind::Z, EdgeKind::Y) => FACE_EDGES[2][1],
        _ => return,
    };

    for indices in edges {
        f(indices.map(|(i, which)| refine(&face[i as usize], which)));
    }
}

const SUB_EDGES: [[[Corner; 4]; 2]; 3] = [
    [[c(6), c(4), c(0), c(2)], [c(7), c(5), c(1), c(3)]],
    [[c(5), c(4), c(0), c(1)], [c(7), c(6), c(2), c(3)]],
    [[c(3), c(2), c(0), c(1)], [c(7), c(6), c(4), c(5)]],
];

pub fn for_each_sub_edge<T, B, R, F>(edge: [T; 4], kind: EdgeKind, refine: R, mut f: F)
where
    R: Fn(&T, Corner) -> B,
    F: FnMut([B; 4]),
{
    for indices in SUB_EDGES[kind as usize] {
        f([
            refine(&edge[0], indices[0]),
            refine(&edge[1], indices[1]),
            refine(&edge[2], indices[2]),
            refine(&edge[3], indices[3]),
        ]);
    }
}

const EDGE_CORNERS: [[[Corner; 2]; 4]; 3] = [
    [[c(6), c(7)], [c(4), c(5)], [c(0), c(1)], [c(2), c(3)]],
    [[c(5), c(7)], [c(4), c(6)], [c(0), c(2)], [c(1), c(3)]],
    [[c(3), c(7)], [c(2), c(6)], [c(0), c(4)], [c(1), c(5)]],
];

pub fn edge_corners<T, B, R>(
    edge: [T; 4],
    kind: EdgeKind,
    refine: R,
) -> impl Iterator<Item = (T, [B; 2])>
where
    R: Fn(&T, Corner) -> B,
{
    iter::zip(edge, EDGE_CORNERS[kind as usize].iter()).map(move |(cell, indices)| {
        let sub_edges = [refine(&cell, indices[0]), refine(&cell, indices[1])];
        (cell, sub_edges)
    })
}

const fn edge_seam_slot(a: Side, b: Side) -> usize {
    match (a, b) {
        (Side::Negative, Side::Negative) => 0,
        (Side::Positive, Side::Negative) => 1,
        (Side::Positive, Side::Positive) => 2,
        (Side::Negative, Side::Positive) => 3,
    }
}

#[derive(Debug, Default)]
pub struct Neighbors<T> {
    faces: [T; 6],
    edges: [T; 12],
}

impl<T> Neighbors<T> {
    pub fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(IVec3) -> T,
    {
        Self {
            edges: array::from_fn(|i| f(EdgeIndex { idx: i as u8 }.as_ivec3())),
            faces: array::from_fn(|i| f(FaceIndex { idx: i as u8 }.as_ivec3())),
        }
    }

    pub fn as_ref(&self) -> Neighbors<&T> {
        Neighbors {
            faces: self.faces.each_ref(),
            edges: self.edges.each_ref(),
        }
    }

    pub fn map<U, F>(self, mut f: F) -> Neighbors<U>
    where
        F: FnMut(T) -> U,
    {
        Neighbors {
            faces: self.faces.map(&mut f),
            edges: self.edges.map(&mut f),
        }
    }
}

impl<T: Copy> Neighbors<T> {
    pub fn for_each_face_seam<F>(&self, this: T, mut f: F)
    where
        F: FnMut(FaceKind, [(IVec3, T); 2]),
    {
        FaceIndex::for_each(|index| {
            f(
                index.kind(),
                match index.side() {
                    Side::Negative => [(index.as_ivec3(), self[index]), (IVec3::ZERO, this)],
                    Side::Positive => [(IVec3::ZERO, this), (index.as_ivec3(), self[index])],
                },
            );
        });
    }

    pub fn for_each_edge_seam<F>(&self, this: T, mut f: F)
    where
        F: FnMut(EdgeKind, [(IVec3, T); 4]),
    {
        EdgeIndex::for_each(|index| {
            let [face_a, face_b] = index.faces();
            let (side_a, side_b) = (face_a.side(), face_b.side());

            let mut seam = [(IVec3::ZERO, this); 4];
            seam[edge_seam_slot(side_a, side_b)] = (index.as_ivec3(), self[index]);
            seam[edge_seam_slot(side_a, side_b.opposite())] = (face_a.as_ivec3(), self[face_a]);
            seam[edge_seam_slot(side_a.opposite(), side_b)] = (face_b.as_ivec3(), self[face_b]);
            seam[edge_seam_slot(side_a.opposite(), side_b.opposite())] = (IVec3::ZERO, this);

            f(index.kind(), seam);
        });
    }
}

impl<T> Index<FaceIndex> for Neighbors<T> {
    type Output = T;

    fn index(&self, index: FaceIndex) -> &Self::Output {
        &self.faces[index.idx as usize]
    }
}

impl<T> Index<EdgeIndex> for Neighbors<T> {
    type Output = T;

    fn index(&self, index: EdgeIndex) -> &Self::Output {
        &self.edges[index.idx as usize]
    }
}
