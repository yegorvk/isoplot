mod primitives;
mod tables;
mod traverse;

pub use primitives::{AxisKind, EdgeKind, FaceKind, Offset};
pub(crate) use primitives::{
    BEdgeIndex, BEdgesMask, Corner, Edge, EdgeKey, EdgeSlot, Face, FaceKey, FaceSlot,
};

pub(crate) use tables::{edge_normal_face, face_edge_slot, for_each_cell_edge, for_each_cell_face};
pub(crate) use traverse::{Edges, Faces, MinimalEdges, TraverseOctree};
