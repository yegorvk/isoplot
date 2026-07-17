use derive_where::derive_where;
use std::{iter, marker::PhantomData};

use crate::{
    topology::{
        Corner, Edge, EdgeKind, EdgeSlot, FaceKind, FaceSlot, for_each_face_edge,
        for_each_sub_edge, for_each_sub_face,
    },
    utils::{array_transpose, traverse_ping_pong},
};

pub(crate) trait TraverseOctree {
    /// Octree node type
    type Node;

    /// Returns whether the face node is a leaf.
    fn is_face_leaf(&mut self, node: &Self::Node, slot: FaceSlot) -> bool;

    /// Returns whether the edge node is a leaf.
    fn is_edge_leaf(&mut self, node: &Self::Node, kind: EdgeKind, slot: EdgeSlot) -> bool;

    /// Returns the specified child of a face node, `None` it does not exist.
    fn refine_face(
        &mut self,
        node: &Self::Node,
        kind: FaceKind,
        slot: FaceSlot,
        corner: Corner,
    ) -> Option<Self::Node>;

    /// Returns the specified child of an edge node, `None` if it does not exist.
    fn refine_edge(
        &mut self,
        node: &Self::Node,
        kind: EdgeKind,
        slot: EdgeSlot,
        corner: Corner,
    ) -> Option<Self::Node>;
}

pub(crate) struct MinimalEdges<T> {
    faces: Faces<T>,
    edges: Edges<T>,
}

impl<T: Copy> MinimalEdges<T> {
    pub(crate) fn new(faces: Faces<T>, edges: Edges<T>) -> Self {
        Self { faces, edges }
    }

    pub(crate) fn traverse_single<L, R, F>(self, is_leaf: L, refine: R, f: F)
    where
        L: FnMut(&T) -> bool,
        R: FnMut(&T, Corner) -> Option<T>,
        F: FnMut(EdgeKind, Edge<T>),
    {
        let mut cx = TraverseSingleOctree {
            is_leaf,
            refine,
            _marker: PhantomData,
        };

        self.traverse(&mut cx, f);
    }

    pub(crate) fn traverse<C, F>(mut self, cx: &mut C, mut f: F)
    where
        C: TraverseOctree<Node = T>,
        F: FnMut(EdgeKind, Edge<T>),
    {
        for (kind, faces) in self.faces.into_axes() {
            traverse_ping_pong(faces, |current, next| {
                for keys in current.iter().copied() {
                    let leaf = iter::zip(&keys, FaceSlot::ALL)
                        .all(|(key, slot)| cx.is_face_leaf(key, slot));

                    if leaf {
                        continue;
                    }

                    for_each_sub_face(
                        kind,
                        |slot, corner| cx.refine_face(&keys[slot.as_usize()], kind, slot, corner),
                        |sub_face| {
                            if let Some(sub_face) = array_transpose(sub_face) {
                                next.push(sub_face);
                            }
                        },
                    );

                    for edge_kind in kind.tangent_edges() {
                        for_each_face_edge(
                            (kind, edge_kind),
                            |slot, corner| {
                                cx.refine_face(&keys[slot.as_usize()], kind, slot, corner)
                            },
                            |edge| {
                                if let Some(edge) = array_transpose(edge) {
                                    self.edges.insert(edge_kind, edge);
                                }
                            },
                        );
                    }
                }
            });
        }

        for (kind, edges) in self.edges.into_axes() {
            traverse_ping_pong(edges, |current, next| {
                for keys in current.iter().copied() {
                    let leaf = iter::zip(&keys, EdgeSlot::ALL)
                        .all(|(key, slot)| cx.is_edge_leaf(key, kind, slot));

                    if leaf {
                        f(kind, Edge(keys));
                        continue;
                    }

                    for_each_sub_edge(
                        kind,
                        |slot, corner| cx.refine_edge(&keys[slot.as_usize()], kind, slot, corner),
                        |sub_edge| {
                            if let Some(edge) = array_transpose(sub_edge) {
                                next.push(edge);
                            }
                        },
                    );
                }
            });
        }
    }
}

struct TraverseSingleOctree<T, L, R> {
    is_leaf: L,
    refine: R,
    _marker: PhantomData<fn(&T) -> T>,
}

impl<T, L, R> TraverseOctree for TraverseSingleOctree<T, L, R>
where
    L: FnMut(&T) -> bool,
    R: FnMut(&T, Corner) -> Option<T>,
{
    type Node = T;

    fn is_face_leaf(&mut self, node: &T, _slot: FaceSlot) -> bool {
        (self.is_leaf)(node)
    }

    fn is_edge_leaf(&mut self, node: &T, _kind: EdgeKind, _slot: EdgeSlot) -> bool {
        (self.is_leaf)(node)
    }

    fn refine_face(
        &mut self,
        node: &T,
        _kind: FaceKind,
        _slot: FaceSlot,
        corner: Corner,
    ) -> Option<T> {
        (self.refine)(node, corner)
    }

    fn refine_edge(
        &mut self,
        node: &T,
        _kind: EdgeKind,
        _slot: EdgeSlot,
        corner: Corner,
    ) -> Option<T> {
        (self.refine)(node, corner)
    }
}

#[derive_where(Default)]
#[derive(Debug)]
pub struct Faces<T> {
    axes: [Vec<[T; 2]>; 3],
}

impl<T> Faces<T> {
    pub fn insert(&mut self, kind: FaceKind, face: [T; 2]) {
        self.axes[kind.axis() as usize].push(face);
    }

    pub fn for_each_axis_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(FaceKind, &mut Vec<[T; 2]>),
    {
        for (kind, faces) in iter::zip(FaceKind::ALL, &mut self.axes) {
            f(kind, faces);
        }
    }

    pub fn into_axes(self) -> impl Iterator<Item = (FaceKind, Vec<[T; 2]>)> {
        iter::zip(FaceKind::ALL, self.axes)
    }
}

#[derive_where(Default)]
#[derive(Debug)]
pub struct Edges<T> {
    axes: [Vec<[T; 4]>; 3],
}

impl<T> Edges<T> {
    pub fn insert(&mut self, kind: EdgeKind, edge: [T; 4]) {
        self.axes[kind.axis() as usize].push(edge);
    }

    pub fn for_each_axis_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(EdgeKind, &mut Vec<[T; 4]>),
    {
        for (kind, edges) in iter::zip(EdgeKind::ALL, &mut self.axes) {
            f(kind, edges);
        }
    }

    pub fn into_axes(self) -> impl Iterator<Item = (EdgeKind, Vec<[T; 4]>)> {
        iter::zip(EdgeKind::ALL, self.axes)
    }
}
