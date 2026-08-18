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

        let [a, b, c, d] = vertices;

        let mut emit_face = |face: [Vec3; 3]| {
            let c = face.iter().sum::<Vec3>() / 3.0;
            let n_c = self.scalar_field.sample_normal(c);
            sink.add_triangle(face.map(|position| Vertex::new(position, n_c)));
        };

        if c != d {
            for face in [[a, b, c], [a, c, d]] {
                emit_face(face);
            }
        } else {
            emit_face([a, b, c]);
        }
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

            let point = bisect(field, positions[a], positions[b], values[a], values[b]);

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

fn bisect<S>(field: &S, a: Vec3, b: Vec3, v_a: f32, v_b: f32) -> Vec3
where
    S: ScalarField,
{
    const ITERS: usize = 3;

    let (mut t_0, mut t_1) = (0.0f32, 1.0f32);
    let (mut v_0, mut v_1) = (v_a, v_b);

    for _ in 0..ITERS {
        let t_m = 0.5 * (t_0 + t_1);
        let v_m = field.sample(a.lerp(b, t_m));

        if (v_m < 0.0) == (v_0 < 0.0) {
            (t_0, v_0) = (t_m, v_m);
        } else {
            (t_1, v_1) = (t_m, v_m);
        }
    }

    let t = if (v_0 - v_1).abs() > f32::EPSILON {
        (t_0 + (t_1 - t_0) * v_0 / (v_0 - v_1)).clamp(t_0, t_1)
    } else {
        0.5 * (t_0 + t_1)
    };

    a.lerp(b, t)
}
