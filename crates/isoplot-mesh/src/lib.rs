mod grid;
mod octree;
mod quant;
mod tables;
mod utils;

use glam::{Vec3, vec3};

use crate::grid::AdaptiveGrid;

/// A scalar field source for isosurface extraction
pub trait ScalarField {
    /// Samples the scalar field at the specified point.
    fn sample(&self, point: Vec3) -> f32;
}

impl<S: ?Sized + ScalarField> ScalarField for &S {
    fn sample(&self, point: Vec3) -> f32 {
        <S as ScalarField>::sample(self, point)
    }
}

/// A scalar field source with normals
pub trait NormalField: ScalarField {
    /// Samples the scalar field normal at the specified point.
    fn sample_normal(&self, point: Vec3) -> Vec3;
}

impl<S: ?Sized + NormalField> NormalField for &S {
    fn sample_normal(&self, point: Vec3) -> Vec3 {
        <S as NormalField>::sample_normal(self, point)
    }
}

pub struct Translated<'a, S: ?Sized> {
    field: &'a S,
    delta: Vec3,
}

impl<'a, S: ?Sized> Translated<'a, S> {
    pub fn new(field: &'a S, delta: Vec3) -> Self {
        Self { field, delta }
    }
}

impl<'a, S> ScalarField for Translated<'a, S>
where
    S: ?Sized + ScalarField,
{
    fn sample(&self, point: Vec3) -> f32 {
        self.field.sample(point - self.delta)
    }
}

impl<'a, S> NormalField for Translated<'a, S>
where
    S: ?Sized + NormalField,
{
    fn sample_normal(&self, point: Vec3) -> Vec3 {
        self.field.sample_normal(point - self.delta)
    }
}

pub struct CentralDifference<'a, S: ?Sized> {
    field: &'a S,
    epsilon: f32,
}

impl<'a, S: ?Sized> CentralDifference<'a, S> {
    pub fn new(field: &'a S, epsilon: f32) -> Self {
        Self { field, epsilon }
    }
}

impl<S: ?Sized + ScalarField> ScalarField for CentralDifference<'_, S> {
    fn sample(&self, point: Vec3) -> f32 {
        self.field.sample(point)
    }
}

impl<S: ?Sized + ScalarField> NormalField for CentralDifference<'_, S> {
    fn sample_normal(&self, point: Vec3) -> Vec3 {
        let p = point;
        let e = self.epsilon;

        let f = |p: Vec3| self.field.sample(p);

        let dx = f(p + Vec3::X * e) - f(p - Vec3::X * e);
        let dy = f(p + Vec3::Y * e) - f(p - Vec3::Y * e);
        let dz = f(p + Vec3::Z * e) - f(p - Vec3::Z * e);

        vec3(dx, dy, dz).normalize_or_zero()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    position: Vec3,
    normal: Vec3,
}

impl Vertex {
    fn new(position: Vec3, normal: Vec3) -> Self {
        Self { position, normal }
    }

    #[inline]
    pub fn position(&self) -> Vec3 {
        self.position
    }
}

pub trait PopulateMesh {
    /// A previous added vertex index
    type Index: Copy;

    /// Adds a vertex to the mesh.
    fn add_vertex(&mut self, vertex: Vertex) -> Self::Index;

    /// Adds a face to the mesh.
    fn add_face(&mut self, indices: [Self::Index; 3]);

    /// Add a triangle to the mesh.
    fn add_triangle(&mut self, vertices: [Vertex; 3]) {
        let indices = vertices.map(|v| self.add_vertex(v));
        self.add_face(indices);
    }

    /// Adds a quad to the mesh.
    fn add_quad(&mut self, vertices: [Vertex; 4]) {
        let [a, b, c, d] = vertices.map(|v| self.add_vertex(v));
        self.add_face([a, b, c]);
        self.add_face([a, c, d]);
    }
}

#[derive(Default)]
pub struct SeparateNormals {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<[u32; 3]>,
}

impl PopulateMesh for SeparateNormals {
    type Index = u32;

    fn add_vertex(&mut self, vertex: Vertex) -> Self::Index {
        self.positions.push(vertex.position().to_array());
        self.normals.push(vertex.normal.to_array());
        (self.positions.len() - 1) as u32
    }

    fn add_face(&mut self, indices: [Self::Index; 3]) {
        self.indices.push(indices);
    }
}

/// A mesh extraction error
#[derive(Debug)]
pub struct ExtractError;

/// Dual contouring algorithm
pub struct DualContouring<S> {
    scalar_field: S,
    max_level: u8,
}

impl<'a, S> DualContouring<S> {
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
    pub fn extract_with<P>(self, sink: &mut P) -> Result<(), ExtractError>
    where
        P: PopulateMesh,
    {
        let grid = AdaptiveGrid::build(&self.scalar_field, self.max_level, |feature| {
            feature.center_point()
        });

        grid.for_each_feature_edge(|mut vertices| {
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
                vertices.map(|position| {
                    Vertex::new(position, self.scalar_field.sample_normal(position))
                }),
            );
        });

        Ok(())
    }

    pub fn extract<P>(self) -> Result<P, ExtractError>
    where
        P: Default + PopulateMesh,
    {
        let mut extractor = P::default();
        self.extract_with(&mut extractor)?;
        Ok(extractor)
    }
}

fn should_flip_face(a: Vec3, b: Vec3, c: Vec3, n: Vec3) -> bool {
    (b - a).cross(c - a).dot(n) < 0.0
}
