use std::iter;

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
