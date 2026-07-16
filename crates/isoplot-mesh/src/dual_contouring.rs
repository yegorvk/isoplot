mod connectivity;
mod grid;

use glam::{IVec3, Vec3};

use crate::{
    ExtractError, NormalField, PopulateMesh, Vertex, should_flip_face, topology::Neighbors,
};
use grid::AdaptiveGrid;

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

        grid.for_each_interior_quad(|vertices| self.add_quad(vertices, sink));
        Ok(Chunk(grid))
    }

    pub fn extract_seam<B, N, P>(
        &self,
        this: &Chunk,
        peek: N,
        sink: &mut P,
    ) -> Result<(), ExtractError>
    where
        B: BorrowChunk,
        N: FnMut(IVec3) -> Option<B>,
        P: PopulateMesh,
    {
        let neighbors = Neighbors::from_fn(peek);

        let grids = neighbors
            .as_ref()
            .map(|neighbor| neighbor.as_ref().map(|chunk| &chunk.borrow_chunk().0));

        this.0.for_each_seam_quad(grids, |vertices| {
            self.add_quad(vertices, sink);
        });

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
