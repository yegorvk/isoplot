mod adaptive_grid;
mod connectivity;

use glam::Vec3;

use crate::{
    AxisKind, ExtractError, NormalField, Offset, PopulateMesh, Vertex, should_flip_face,
    topology::{Edge, EdgeKey, EdgeKind, Face, FaceKey, FaceKind},
};
use adaptive_grid::{AdaptiveGrid, EdgeSeam, FaceSeam};

#[derive(Debug)]
pub struct Chunk(AdaptiveGrid);

pub trait BorrowChunk {
    fn borrow_chunk(&self) -> &Chunk;
}

impl BorrowChunk for Chunk {
    fn borrow_chunk(&self) -> &Chunk {
        self
    }
}

impl<T: BorrowChunk + ?Sized> BorrowChunk for &T {
    fn borrow_chunk(&self) -> &Chunk {
        T::borrow_chunk(self)
    }
}

/// Dual contouring algorithm
pub struct DualContouring<S> {
    scalar_field: S,
    max_level: u8,
}

impl<S> DualContouring<S> {
    pub fn new(scalar_field: S, max_level: u8) -> Self {
        Self {
            scalar_field,
            max_level,
        }
    }
}

impl<S> DualContouring<S>
where
    S: NormalField,
{
    pub fn extract_chunk<P>(&self, sink: &mut P) -> Result<Chunk, ExtractError>
    where
        P: PopulateMesh,
    {
        let grid = AdaptiveGrid::build(&self.scalar_field, self.max_level, |feature| {
            feature.center_point()
        });

        grid.for_each_quad(|vertices| self.add_quad(vertices, sink));
        Ok(Chunk(grid))
    }

    pub fn extract_face_seam<B, P>(
        &self,
        face: ChunkFace<B>,
        sink: &mut P,
    ) -> Result<(), ExtractError>
    where
        B: BorrowChunk,
        P: PopulateMesh,
    {
        let ChunkFace { kind, face } = face;
        let grids = Face(face.0.each_ref().map(|chunk| &chunk.borrow_chunk().0));

        FaceSeam::new(kind, grids).for_each_quad(|vertices| self.add_quad(vertices, sink));

        Ok(())
    }

    pub fn extract_edge_seam<B, P>(
        &self,
        edge: ChunkEdge<B>,
        sink: &mut P,
    ) -> Result<(), ExtractError>
    where
        B: BorrowChunk,
        P: PopulateMesh,
    {
        let ChunkEdge { kind, edge } = edge;
        let grids = Edge(edge.0.each_ref().map(|chunk| &chunk.borrow_chunk().0));

        EdgeSeam::new(kind, grids).for_each_quad(|vertices| self.add_quad(vertices, sink));

        Ok(())
    }

    fn add_quad<P>(&self, mut vertices: [Vec3; 4], sink: &mut P)
    where
        P: PopulateMesh,
    {
        if vertices[0] == vertices[1] {
            vertices[1] = vertices[3];
            vertices[3] = vertices[2];
        }

        if vertices[1] == vertices[2] {
            vertices[2] = vertices[3];
        }

        let n = self.scalar_field.sample_normal(vertices[0]);

        if should_flip_face(vertices[0], vertices[1], vertices[2], n) {
            vertices.reverse();
        }

        sink.add_quad(
            vertices
                .map(|position| Vertex::new(position, self.scalar_field.sample_normal(position))),
        );
    }
}

pub struct ChunkFace<T> {
    kind: FaceKind,
    face: Face<T>,
}

impl<T> ChunkFace<T> {
    pub fn from_fn<F>(axis: AxisKind, min_chunk: T, mut f: F) -> Option<Self>
    where
        F: FnMut(&mut T, Offset) -> Option<T>,
    {
        let kind = FaceKind::from_axis(axis);
        let key = FaceKey::new(kind, min_chunk);

        Face::try_from_fn(key, |min_cell, offset| f(min_cell, offset).ok_or(()))
            .ok()
            .map(|face| Self { kind, face })
    }
}

pub struct ChunkEdge<T> {
    kind: EdgeKind,
    edge: Edge<T>,
}

impl<T> ChunkEdge<T> {
    pub fn from_fn<F>(axis: AxisKind, min_chunk: T, mut f: F) -> Option<Self>
    where
        F: FnMut(&mut T, Offset) -> Option<T>,
    {
        let kind = EdgeKind::from_axis(axis);
        let key = EdgeKey::new(kind, min_chunk);

        Edge::try_from_fn(key, |min_chunk, offset| f(min_chunk, offset).ok_or(()))
            .ok()
            .map(|edge| Self { kind, edge })
    }
}
