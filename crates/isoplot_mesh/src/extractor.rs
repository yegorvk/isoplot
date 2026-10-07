mod chunk;
mod seams;

use isoplot_math::Vec3;
use std::array;

use crate::{
    extractor::seams::{EdgeSeam, FaceSeam},
    lattice::{Corner, Edge, EdgeKey, EdgeKind, Face, FaceKey, FaceKind, Offset},
    mesh::{PopulateMesh, Vertex},
    quant::Quant,
    source::{NormalField, ScalarField, Translate},
};

pub use chunk::Chunk;

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
    pub fn with_offset(scalar_field: S, offset: Vec3<f32>, max_level: u8) -> Self {
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
        let chunk = Chunk::build(&self.scalar_field, self.max_level, |cell| {
            place_feature(&self.scalar_field, cell)
        });
        chunk.for_each_quad(|vertices| self.add_quad(vertices, &mut sink));
        Ok(chunk)
    }

    pub fn extract_face_seam<B, P>(
        &self,
        key: FaceSeamKey<B>,
        mut sink: P,
    ) -> Result<(), ExtractError>
    where
        P: PopulateMesh,
        B: BorrowChunk,
    {
        FaceSeam::new(key.borrow_chunks()).for_each_quad(|vertices| {
            self.add_quad(vertices, &mut sink);
        });
        Ok(())
    }

    pub fn extract_edge_seam<B, P>(
        &self,
        key: EdgeSeamKey<B>,
        mut sink: P,
    ) -> Result<(), ExtractError>
    where
        P: PopulateMesh,
        B: BorrowChunk,
    {
        EdgeSeam::new(key.borrow_chunks()).for_each_quad(|vertices| {
            self.add_quad(vertices, &mut sink);
        });
        Ok(())
    }

    fn add_quad<P>(&self, vertices: [Vec3<f32>; 4], sink: &mut P)
    where
        P: PopulateMesh,
    {
        let [a, mut b, mut c, mut d] = vertices;

        if a == b {
            b = d;
            d = c;
        }

        if b == c {
            c = d;
        }

        let mut emit_face = |face: [Vec3<f32>; 3]| {
            let c = face.iter().copied().sum::<Vec3<f32>>() / 3.0;
            let n_c = self.scalar_field.sample_normal(c);
            sink.add_triangle(face.map(|position| Vertex::new(position, n_c)));
        };

        if c == d {
            emit_face([a, b, c]);
            return;
        }

        emit_face([a, c, b]);
        emit_face([a, d, c]);
    }
}

pub struct FaceSeamKey<B: BorrowChunk>(Face<B>);

impl<B: BorrowChunk> FaceSeamKey<B> {
    pub fn from_fn<F>(kind: FaceKind, min_chunk: B, mut f: F) -> Option<Self>
    where
        F: FnMut(&mut B, Offset) -> Option<B>,
    {
        Face::try_from_fn(FaceKey::new(kind, min_chunk), |min_cell, offset| {
            f(min_cell, offset).ok_or(())
        })
        .ok()
        .map(|face| Self(face))
    }

    fn borrow_chunks(&self) -> Face<&Chunk> {
        self.0.each_ref().map(|cell| cell.borrow_chunk())
    }
}

pub struct EdgeSeamKey<B: BorrowChunk>(Edge<B>);

impl<B: BorrowChunk> EdgeSeamKey<B> {
    pub fn from_fn<F>(kind: EdgeKind, min_chunk: B, mut f: F) -> Option<Self>
    where
        F: FnMut(&mut B, Offset) -> Option<B>,
    {
        Edge::try_from_fn(EdgeKey::new(kind, min_chunk), |min_cell, offset| {
            f(min_cell, offset).ok_or(())
        })
        .ok()
        .map(|face| Self(face))
    }

    fn borrow_chunks(&self) -> Edge<&Chunk> {
        self.0.each_ref().map(|cell| cell.borrow_chunk())
    }
}

#[derive(Debug)]
pub struct ExtractError;

fn place_feature<S: NormalField>(field: &S, cell: Quant) -> Option<Vec3<f32>> {
    const ITERS: usize = 25;

    let (min_corner, size) = cell.min_point_size();

    let positions: [Vec3<f32>; 8] =
        array::from_fn(|i| min_corner + Corner(Offset::ALL[i]).0.as_vec3().cast() * size);

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

            let Some(point) = field.find_intersection(positions[a], positions[b]).get() else {
                continue;
            };

            points[count] = point;
            normals[count] = field.sample_normal(point);
            count += 1;
        }
    }

    if count == 0 {
        return None;
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

    x.all(f32::is_finite).then_some(x)
}
