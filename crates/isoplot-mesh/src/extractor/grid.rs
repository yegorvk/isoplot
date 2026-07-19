use glam::Vec3;
use std::array;

use crate::{
    lattice::{
        Corner, Edge, EdgeKind, EdgeSlot, Edges, Face, FaceKind, FaceSlot, Faces, MinimalEdges,
        Offset, TraverseOctree, edge_corners, face_edge_slot, for_each_cell_edge,
        for_each_cell_face,
    },
    octree::{BuildOctree, ChildIndex, Key, Node, Octree},
    quant::Quant,
    source::ScalarField,
    utils::array_transpose,
};

struct OctreeSource<S, P> {
    scalar_field: S,
    max_level: u8,
    place_feature: P,
}

impl<S, P> BuildOctree<Feature> for OctreeSource<S, P>
where
    S: ScalarField,
    P: Fn(Quant) -> Vec3,
{
    type Tag = Quant;

    fn root(&mut self) -> Self::Tag {
        Quant::root()
    }

    fn is_leaf(&mut self, tag: Self::Tag) -> bool {
        let (min_point, _) = tag.min_point_size();

        if (min_point.z.to_bits() + min_point.x.to_bits() + 56) % 3 == 1 {
            tag.level() >= self.max_level - 1
        } else {
            tag.level() >= self.max_level
        }
    }

    fn refine(&mut self, tag: Self::Tag, which: ChildIndex) -> Option<Self::Tag> {
        Some(tag.child(which).unwrap())
    }

    fn place_leaf(&mut self, tag: Self::Tag) -> Feature {
        let p_mask = {
            let (min_corner, size) = tag.min_point_size();

            Offset::enumerate().fold(0, |mask, offset| {
                let position = min_corner + offset.as_uvec3().as_vec3() * size;

                if self.scalar_field.sample(position).is_sign_positive() {
                    mask | (1u8 << offset.as_u8())
                } else {
                    mask
                }
            })
        };

        Feature {
            vertex: (self.place_feature)(tag),
            quant: tag,
            p_mask,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Feature {
    vertex: Vec3,
    quant: Quant,
    p_mask: u8,
}

impl Feature {
    fn contains_sign_change(&self, edge: (Corner, Corner)) -> bool {
        let (a, b) = edge;
        self.is_corner_sign_positive(a) != self.is_corner_sign_positive(b)
    }

    fn is_corner_sign_positive(&self, corner: Corner) -> bool {
        self.p_mask & (1u8 << corner.offset().as_u8()) != 0
    }
}

#[derive(Debug)]
pub struct AdaptiveGrid {
    octree: Octree<Feature>,
}

impl AdaptiveGrid {
    pub(crate) fn build<S, P>(field: S, max_level: u8, place_feature: P) -> Self
    where
        S: ScalarField,
        P: Fn(Quant) -> Vec3,
    {
        let mut source = OctreeSource {
            scalar_field: field,
            max_level,
            place_feature,
        };

        Self {
            octree: Octree::build(&mut source),
        }
    }

    pub(crate) fn for_each_quad<F>(&self, mut f: F)
    where
        F: FnMut([Vec3; 4]),
    {
        let mut faces = Faces::default();
        let mut edges = Edges::default();

        self.octree.for_each_branch(|branch| {
            let refine = |which: Corner| {
                let child = ChildIndex::new(which.offset().as_u8());
                branch.child(child)
            };

            faces.for_each_axis_mut(|kind, faces| {
                for_each_cell_face(kind, &refine, |keys| {
                    if let Some(keys) = array_transpose(keys) {
                        faces.push(keys);
                    }
                });
            });

            edges.for_each_axis_mut(|kind, edges| {
                for_each_cell_edge(kind, &refine, |keys| {
                    if let Some(keys) = array_transpose(keys) {
                        edges.push(keys);
                    }
                });
            });
        });

        let refine_node = |key: &Key, which: Corner| {
            if let Node::Branch(branch) = self.octree.get(*key).unwrap() {
                branch.child(ChildIndex::new(which.offset().as_u8()))
            } else {
                Some(*key)
            }
        };

        MinimalEdges::new(faces, edges).traverse_single(
            |key| self.octree.is_leaf(*key),
            refine_node,
            |kind, keys| {
                let features = keys.0.map(|key| self.get_feature(key).unwrap());

                if contains_sign_change(kind, features) {
                    f(features.map(|feature| feature.vertex));
                }
            },
        );
    }

    fn get_feature(&self, key: Key) -> Option<&Feature> {
        self.octree.get(key).and_then(|key| key.as_leaf().copied())
    }
}

pub(crate) struct FaceSeam<'a> {
    kind: FaceKind,
    face: Face<&'a AdaptiveGrid>,
}

impl<'a> FaceSeam<'a> {
    pub(crate) fn new(kind: FaceKind, face: Face<&'a AdaptiveGrid>) -> Self {
        Self { kind, face }
    }

    pub(crate) fn for_each_quad<F>(&self, mut f: F)
    where
        F: FnMut([Vec3; 4]),
    {
        let mut faces = Faces::default();
        faces.insert(self.kind, [Key::ROOT; 2]);

        let mut cx = TraverseFaceSeam {
            kind: self.kind,
            grids: self.face.0,
        };

        MinimalEdges::new(faces, Edges::default()).traverse(&mut cx, |kind, Edge(keys)| {
            let cells = EdgeSlot::ALL.map(|slot| {
                let slot = face_edge_slot((self.kind, kind), slot);
                (self.face.0[slot.as_usize()], self.kind.slot_offset(slot))
            });

            emit_seam_quad(kind, cells, keys, &mut f);
        });
    }
}

pub(crate) struct EdgeSeam<'a> {
    kind: EdgeKind,
    face: Edge<&'a AdaptiveGrid>,
}

impl<'a> EdgeSeam<'a> {
    pub(crate) fn new(kind: EdgeKind, face: Edge<&'a AdaptiveGrid>) -> Self {
        Self { kind, face }
    }

    pub(crate) fn for_each_quad<F>(&self, mut f: F)
    where
        F: FnMut([Vec3; 4]),
    {
        let mut edges = Edges::default();
        edges.insert(self.kind, [Key::ROOT; 4]);

        let mut cx = TraverseEdgeSeam { grids: self.face.0 };

        MinimalEdges::new(Faces::default(), edges).traverse(&mut cx, |kind, Edge(keys)| {
            let cells = EdgeSlot::ALL
                .map(|slot| (self.face.0[slot.as_usize()], self.kind.slot_offset(slot)));

            emit_seam_quad(kind, cells, keys, &mut f);
        });
    }
}

struct TraverseFaceSeam<'a> {
    kind: FaceKind,
    grids: [&'a AdaptiveGrid; 2],
}

impl TraverseOctree for TraverseFaceSeam<'_> {
    type Node = Key;

    fn is_face_leaf(&mut self, node: &Key, slot: FaceSlot) -> bool {
        self.grids[slot.as_usize()].octree.is_leaf(*node)
    }

    fn is_edge_leaf(&mut self, node: &Key, kind: EdgeKind, slot: EdgeSlot) -> bool {
        let slot = face_edge_slot((self.kind, kind), slot);
        self.grids[slot.as_usize()].octree.is_leaf(*node)
    }

    fn refine_face(
        &mut self,
        node: &Key,
        _kind: FaceKind,
        slot: FaceSlot,
        corner: Corner,
    ) -> Option<Key> {
        refine_key(self.grids[slot.as_usize()], node, corner)
    }

    fn refine_edge(
        &mut self,
        node: &Key,
        kind: EdgeKind,
        slot: EdgeSlot,
        corner: Corner,
    ) -> Option<Key> {
        let slot = face_edge_slot((self.kind, kind), slot);
        refine_key(self.grids[slot.as_usize()], node, corner)
    }
}

struct TraverseEdgeSeam<'a> {
    grids: [&'a AdaptiveGrid; 4],
}

impl TraverseOctree for TraverseEdgeSeam<'_> {
    type Node = Key;

    fn is_face_leaf(&mut self, _node: &Key, _slot: FaceSlot) -> bool {
        unreachable!()
    }

    fn is_edge_leaf(&mut self, node: &Key, _kind: EdgeKind, slot: EdgeSlot) -> bool {
        self.grids[slot.as_usize()].octree.is_leaf(*node)
    }

    fn refine_face(
        &mut self,
        _node: &Key,
        _kind: FaceKind,
        _slot: FaceSlot,
        _corner: Corner,
    ) -> Option<Key> {
        unreachable!()
    }

    fn refine_edge(
        &mut self,
        node: &Key,
        _kind: EdgeKind,
        slot: EdgeSlot,
        corner: Corner,
    ) -> Option<Key> {
        refine_key(self.grids[slot.as_usize()], node, corner)
    }
}

fn refine_key(grid: &AdaptiveGrid, key: &Key, which: Corner) -> Option<Key> {
    if let Node::Branch(branch) = grid.octree.get(*key).unwrap() {
        branch.child(ChildIndex::new(which.offset().as_u8()))
    } else {
        Some(*key)
    }
}

fn emit_seam_quad<F>(kind: EdgeKind, cells: [(&AdaptiveGrid, Offset); 4], keys: [Key; 4], f: &mut F)
where
    F: FnMut([Vec3; 4]),
{
    let features: [&Feature; 4] = array::from_fn(|i| cells[i].0.get_feature(keys[i]).unwrap());

    if contains_sign_change(kind, features) {
        f(array::from_fn(|i| {
            features[i].vertex + cells[i].1.as_uvec3().as_vec3()
        }));
    }
}

fn contains_sign_change(kind: EdgeKind, features: [&Feature; 4]) -> bool {
    let (max_feature, [a, b]) = edge_corners(kind, |_, corner| corner)
        .max_by_key(|(slot, _)| features[slot.as_usize()].quant.level())
        .unwrap();

    features[max_feature.as_usize()].contains_sign_change((a, b))
}
