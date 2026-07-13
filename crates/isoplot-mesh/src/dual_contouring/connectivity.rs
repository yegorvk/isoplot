use derive_where::derive_where;

use crate::{
    topology::{
        Corner, EdgeKind, FaceKind, for_each_face_edge, for_each_sub_edge, for_each_sub_face,
    },
    utils::{array_transpose, traverse_ping_pong},
};

pub struct MinimalEdges<T> {
    faces: Faces<T>,
    edges: Edges<T>,
}

impl<T: Copy> MinimalEdges<T> {
    pub fn new(faces: Faces<T>, edges: Edges<T>) -> Self {
        Self { faces, edges }
    }

    pub fn traverse<L, R, F>(mut self, is_leaf: L, refine: R, mut f: F)
    where
        L: Fn(&T) -> bool,
        R: Fn(&T, Corner) -> Option<T>,
        F: FnMut(EdgeKind, [T; 4]),
    {
        for (kind, faces) in self.faces.into_axes() {
            traverse_ping_pong(faces, |current, next| {
                for keys in current.iter().copied() {
                    if keys.iter().all(&is_leaf) {
                        continue;
                    }

                    for_each_sub_face(keys, kind, &refine, |sub_face| {
                        if let Some(sub_face) = array_transpose(sub_face) {
                            next.push(sub_face);
                        }
                    });

                    for edge_kind in kind.tangent_edges() {
                        for_each_face_edge(keys, (kind, edge_kind), &refine, |edge| {
                            if let Some(edge) = array_transpose(edge) {
                                self.edges.insert(edge_kind, edge);
                            }
                        });
                    }
                }
            });
        }

        for (kind, edges) in self.edges.into_axes() {
            traverse_ping_pong(edges, |current, next| {
                for keys in current.iter().copied() {
                    if keys.iter().all(&is_leaf) {
                        f(kind, keys);
                        continue;
                    }

                    for_each_sub_edge(keys, kind, &refine, |sub_edge| {
                        if let Some(edge) = array_transpose(sub_edge) {
                            next.push(edge);
                        }
                    });
                }
            });
        }
    }
}

#[derive_where(Default)]
#[derive(Debug)]
pub struct Faces<T> {
    pub x: Vec<[T; 2]>,
    pub y: Vec<[T; 2]>,
    pub z: Vec<[T; 2]>,
}

impl<T> Faces<T> {
    pub fn insert(&mut self, kind: FaceKind, face: [T; 2]) {
        let faces = match kind {
            FaceKind::X => &mut self.x,
            FaceKind::Y => &mut self.y,
            FaceKind::Z => &mut self.z,
        };

        faces.push(face);
    }

    pub fn for_each_axis_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(FaceKind, &mut Vec<[T; 2]>),
    {
        f(FaceKind::X, &mut self.x);
        f(FaceKind::Y, &mut self.y);
        f(FaceKind::Z, &mut self.z);
    }

    pub fn into_axes(self) -> [(FaceKind, Vec<[T; 2]>); 3] {
        [
            (FaceKind::X, self.x),
            (FaceKind::Y, self.y),
            (FaceKind::Z, self.z),
        ]
    }
}

#[derive_where(Default)]
#[derive(Debug)]
pub struct Edges<T> {
    pub x: Vec<[T; 4]>,
    pub y: Vec<[T; 4]>,
    pub z: Vec<[T; 4]>,
}

impl<T> Edges<T> {
    pub fn insert(&mut self, kind: EdgeKind, edge: [T; 4]) {
        let edges = match kind {
            EdgeKind::X => &mut self.x,
            EdgeKind::Y => &mut self.y,
            EdgeKind::Z => &mut self.z,
        };

        edges.push(edge);
    }

    pub fn for_each_axis_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(EdgeKind, &mut Vec<[T; 4]>),
    {
        f(EdgeKind::X, &mut self.x);
        f(EdgeKind::Y, &mut self.y);
        f(EdgeKind::Z, &mut self.z);
    }

    pub fn into_axes(self) -> [(EdgeKind, Vec<[T; 4]>); 3] {
        [
            (EdgeKind::X, self.x),
            (EdgeKind::Y, self.y),
            (EdgeKind::Z, self.z),
        ]
    }
}
