use isoplot_math::Vec3;

use crate::{
    lattice::{
        Corner, Edge, EdgeKind, EdgeSlot, Edges, Face, FaceKind, FaceSlot, Faces, MinimalEdges,
        TraverseOctree, face_edge_slot,
    },
    octree::{ChildIndex, Node, OctreeKey},
};

use super::chunk::{Chunk, contains_intersection};

pub(crate) struct FaceSeam<'a> {
    face: Face<&'a Chunk>,
}

impl<'a> FaceSeam<'a> {
    pub(crate) fn new(face: Face<&'a Chunk>) -> Self {
        Self { face }
    }

    pub(crate) fn for_each_quad<F>(&self, mut f: F)
    where
        F: FnMut([Vec3<f32>; 4]),
    {
        let mut faces = Faces::default();
        faces.insert(self.face.kind, [OctreeKey::ROOT; 2]);

        let mut cx = TraverseFaceSeam(&self.face);

        MinimalEdges::new(faces, Edges::default()).traverse(&mut cx, |edge| {
            emit_face_seam_quad(&self.face, &edge, &mut f);
        });
    }
}

pub(crate) struct EdgeSeam<'a> {
    edge: Edge<&'a Chunk>,
}

impl<'a> EdgeSeam<'a> {
    pub(crate) fn new(edge: Edge<&'a Chunk>) -> Self {
        Self { edge }
    }

    pub(crate) fn for_each_quad<F>(&self, mut f: F)
    where
        F: FnMut([Vec3<f32>; 4]),
    {
        let mut edges = Edges::default();
        edges.insert(self.edge.kind, [OctreeKey::ROOT; 4]);

        let mut cx = TraverseEdgeSeam {
            cells: self.edge.cells,
        };

        MinimalEdges::new(Faces::default(), edges).traverse(&mut cx, |edge| {
            emit_edge_seam_quad(&edge.map_slot(|slot, key| (self.edge[slot], key)), &mut f);
        });
    }
}

struct TraverseFaceSeam<'a>(&'a Face<&'a Chunk>);

impl TraverseOctree for TraverseFaceSeam<'_> {
    type Node = OctreeKey;

    fn is_face_leaf(&mut self, node: &OctreeKey, slot: FaceSlot) -> bool {
        self.0.cells[slot.index()].octree.is_leaf(*node)
    }

    fn is_edge_leaf(&mut self, node: &OctreeKey, kind: EdgeKind, slot: EdgeSlot) -> bool {
        let slot = face_edge_slot((self.0.kind, kind), slot);
        self.0.cells[slot.index()].octree.is_leaf(*node)
    }

    fn refine_face(
        &mut self,
        node: &OctreeKey,
        _kind: FaceKind,
        slot: FaceSlot,
        corner: Corner,
    ) -> Option<OctreeKey> {
        refine_key(self.0.cells[slot.index()], node, corner)
    }

    fn refine_edge(
        &mut self,
        node: &OctreeKey,
        kind: EdgeKind,
        slot: EdgeSlot,
        corner: Corner,
    ) -> Option<OctreeKey> {
        let slot = face_edge_slot((self.0.kind, kind), slot);
        refine_key(self.0.cells[slot.index()], node, corner)
    }
}

struct TraverseEdgeSeam<'a> {
    cells: [&'a Chunk; 4],
}

impl TraverseOctree for TraverseEdgeSeam<'_> {
    type Node = OctreeKey;

    fn is_face_leaf(&mut self, _node: &OctreeKey, _slot: FaceSlot) -> bool {
        unreachable!()
    }

    fn is_edge_leaf(&mut self, node: &OctreeKey, _kind: EdgeKind, slot: EdgeSlot) -> bool {
        self.cells[slot.index()].octree.is_leaf(*node)
    }

    fn refine_face(
        &mut self,
        _node: &OctreeKey,
        _kind: FaceKind,
        _slot: FaceSlot,
        _corner: Corner,
    ) -> Option<OctreeKey> {
        unreachable!()
    }

    fn refine_edge(
        &mut self,
        node: &OctreeKey,
        _kind: EdgeKind,
        slot: EdgeSlot,
        corner: Corner,
    ) -> Option<OctreeKey> {
        refine_key(self.cells[slot.index()], node, corner)
    }
}

fn refine_key(chunk: &Chunk, key: &OctreeKey, which: Corner) -> Option<OctreeKey> {
    if let Node::Branch(branch) = chunk.octree.get(*key).unwrap() {
        branch.child(ChildIndex::new(which.0.as_u8()))
    } else {
        Some(*key)
    }
}

fn emit_face_seam_quad<F>(face: &Face<&Chunk>, edge: &Edge<OctreeKey>, mut f: F)
where
    F: FnMut([Vec3<f32>; 4]),
{
    let features = edge.map_slot(|slot, key| {
        let face_slot = face_edge_slot((face.kind, edge.kind), slot);
        face[face_slot].get_feature(key).unwrap()
    });

    if contains_intersection(&features) {
        f(features
            .map_slot(|slot, feature| {
                let face_slot = face_edge_slot((face.kind, edge.kind), slot);
                let offset = face.kind.slot_offset(face_slot);
                feature.vertex + offset.as_vec3().cast()
            })
            .cells);
    }
}

fn emit_edge_seam_quad<F>(edge: &Edge<(&Chunk, OctreeKey)>, mut f: F)
where
    F: FnMut([Vec3<f32>; 4]),
{
    let features = edge.map(|(chunk, key)| chunk.get_feature(key).unwrap());

    if contains_intersection(&features) {
        f(features
            .map_slot(|slot, feature| {
                let offset = edge.kind.slot_offset(slot);
                feature.vertex + offset.as_vec3().cast()
            })
            .cells)
    }
}
