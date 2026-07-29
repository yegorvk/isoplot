mod grid;

use glam::Vec3;

use crate::{
    lattice::{
        Corner, Edge, EdgeKey, EdgeKind, EdgeSlot, Face, FaceKey, FaceKind, FaceSlot, Offset,
    },
    mesh::{PopulateMesh, Vertex},
    quant::Quant,
    source::{NormalField, ScalarField, Translate},
};
use grid::{AdaptiveGrid, Corners, EdgeSeam, FaceSeam};

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

pub struct Extractor<S> {
    scalar_field: S,
    max_level: u8,
}

impl<S> Extractor<S> {
    pub fn new(scalar_field: S, max_level: u8) -> Self {
        Self {
            scalar_field,
            max_level,
        }
    }
}

impl<S: ScalarField> Extractor<Translate<S>> {
    pub fn with_offset(scalar_field: S, offset: Vec3, max_level: u8) -> Self {
        Self::new(scalar_field.translated(offset), max_level)
    }
}

impl<S> Extractor<S>
where
    S: NormalField,
{
    pub fn extract_chunk<P>(&self, mut sink: P) -> Result<Chunk, ExtractError>
    where
        P: PopulateMesh,
    {
        let grid = AdaptiveGrid::build(&self.scalar_field, self.max_level, |cell, corners| {
            place_feature(&self.scalar_field, cell, corners)
        });

        grid.for_each_quad(|vertices| self.add_quad(vertices, &mut sink));
        Ok(Chunk(grid))
    }

    pub fn extract_face_seam<B, P>(
        &self,
        face: ChunkFace<B>,
        mut sink: P,
    ) -> Result<(), ExtractError>
    where
        B: BorrowChunk,
        P: PopulateMesh,
    {
        let ChunkFace { kind, face } = face;
        let grids = Face(face.0.each_ref().map(|chunk| &chunk.borrow_chunk().0));

        FaceSeam::new(kind, grids).for_each_quad(|vertices| self.add_quad(vertices, &mut sink));

        Ok(())
    }

    pub fn extract_edge_seam<B, P>(
        &self,
        edge: ChunkEdge<B>,
        mut sink: P,
    ) -> Result<(), ExtractError>
    where
        B: BorrowChunk,
        P: PopulateMesh,
    {
        let ChunkEdge { kind, edge } = edge;
        let grids = Edge(edge.0.each_ref().map(|chunk| &chunk.borrow_chunk().0));

        EdgeSeam::new(kind, grids).for_each_quad(|vertices| self.add_quad(vertices, &mut sink));

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

        sink.add_quad(
            vertices
                .map(|position| Vertex::new(position, self.scalar_field.sample_normal(position))),
        );
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SharedFaceKind(FaceKind);

impl SharedFaceKind {
    pub const ALL: [Self; 3] = {
        let [x, y, z] = FaceKind::ALL;
        [Self(x), Self(y), Self(z)]
    };

    pub fn slot_offsets(self) -> [Offset; 2] {
        FaceSlot::ALL.map(|slot| self.0.slot_offset(slot))
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SharedEdgeKind(EdgeKind);

impl SharedEdgeKind {
    pub const ALL: [Self; 3] = {
        let [x, y, z] = EdgeKind::ALL;
        [Self(x), Self(y), Self(z)]
    };

    pub fn slot_offsets(self) -> [Offset; 4] {
        EdgeSlot::ALL.map(|slot| self.0.slot_offset(slot))
    }
}

pub struct ChunkFace<T> {
    kind: FaceKind,
    face: Face<T>,
}

impl<T> ChunkFace<T> {
    pub fn from_fn<F>(kind: SharedFaceKind, min_chunk: T, mut f: F) -> Option<Self>
    where
        F: FnMut(&mut T, Offset) -> Option<T>,
    {
        let key = FaceKey::new(kind.0, min_chunk);

        Face::try_from_fn(key, |min_cell, offset| f(min_cell, offset).ok_or(()))
            .ok()
            .map(|face| Self { kind: kind.0, face })
    }
}

pub struct ChunkEdge<T> {
    kind: EdgeKind,
    edge: Edge<T>,
}

impl<T> ChunkEdge<T> {
    pub fn from_fn<F>(kind: SharedEdgeKind, min_chunk: T, mut f: F) -> Option<Self>
    where
        F: FnMut(&mut T, Offset) -> Option<T>,
    {
        let key = EdgeKey::new(kind.0, min_chunk);

        Edge::try_from_fn(key, |min_chunk, offset| f(min_chunk, offset).ok_or(()))
            .ok()
            .map(|edge| Self { kind: kind.0, edge })
    }
}

#[derive(Debug)]
pub struct ExtractError;

fn place_feature<S: NormalField>(field: &S, cell: Quant, corners: Corners) -> Vec3 {
    const ITERS: usize = 25;

    let (min_corner, size) = cell.min_point_size();

    let mut positions = [Vec3::ZERO; 8];
    let mut values = [0f32; 8];

    for i in 0..8 {
        let corner = Corner::new(Offset::ALL[i]);
        positions[i] = min_corner + size * corner.offset().as_vec3();
        values[i] = field.sample(positions[i]);
    }

    let mut points = [Vec3::ZERO; 12];
    let mut normals = [Vec3::ZERO; 12];
    let mut count = 0;

    for i in 0..8u8 {
        for axis in [1u8, 2, 4] {
            let j = i ^ axis;

            if i >= j {
                continue;
            }

            let (a, b) = (i as usize, j as usize);

            if !corners
                .contains_sign_change((Corner::new(Offset::ALL[a]), Corner::new(Offset::ALL[b])))
            {
                continue;
            }

            let (va, vb) = (values[a], values[b]);
            let t = if (va - vb).abs() > f32::EPSILON {
                (va / (va - vb)).clamp(0.0, 1.0)
            } else {
                0.5
            };

            let point = positions[a].lerp(positions[b], t);
            points[count] = point;
            normals[count] = field.sample_normal(point);
            count += 1;
        }
    }

    if count == 0 {
        return cell.center_point();
    }

    let mut x = Vec3::ZERO;
    for point in &points[..count] {
        x += *point;
    }
    x /= count as f32;

    let max_corner = min_corner + Vec3::splat(size);

    for _ in 0..ITERS {
        let mut force = Vec3::ZERO;

        for k in 0..count {
            let n = normals[k];
            force += n * n.dot(points[k] - x);
        }

        x = (x + force / count as f32).clamp(min_corner, max_corner);
    }

    x
}
