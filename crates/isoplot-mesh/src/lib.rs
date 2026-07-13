mod octree;
mod quant;
mod topology;
mod utils;

// Public modules
pub mod dual_contouring;

use glam::{Vec3, vec3};

/// A scalar field source for isosurface extraction
pub trait ScalarField {
    /// Samples the scalar field at the specified point.
    fn sample(&self, point: Vec3) -> f32;

    fn translated(self, offset: Vec3) -> TranslateField<Self>
    where
        Self: Sized,
    {
        TranslateField::new(self, offset)
    }
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

pub struct TranslateField<S> {
    source: S,
    offset: Vec3,
}

impl<S> TranslateField<S> {
    pub fn new(source: S, offset: Vec3) -> Self {
        Self { source, offset }
    }
}

impl<S: ScalarField> ScalarField for TranslateField<S> {
    fn sample(&self, point: Vec3) -> f32 {
        self.source.sample(point + self.offset)
    }
}

impl<S: NormalField> NormalField for TranslateField<S> {
    fn sample_normal(&self, point: Vec3) -> Vec3 {
        self.source.sample_normal(point + self.offset)
    }
}

pub struct CentralDifference<S> {
    source: S,
    delta: f32,
}

impl<S: ScalarField> CentralDifference<S> {
    pub fn new(source: S, delta: f32) -> Self {
        Self { source, delta }
    }
}

impl<S: ScalarField> ScalarField for CentralDifference<S> {
    fn sample(&self, point: Vec3) -> f32 {
        self.source.sample(point)
    }
}

impl<S: ScalarField> NormalField for CentralDifference<S> {
    fn sample_normal(&self, point: Vec3) -> Vec3 {
        let (p, e) = (point, self.delta);
        let f = |p: Vec3| self.source.sample(p);

        let dx = f(p + Vec3::X * e) - f(p - Vec3::X * e);
        let dy = f(p + Vec3::Y * e) - f(p - Vec3::Y * e);
        let dz = f(p + Vec3::Z * e) - f(p - Vec3::Z * e);

        vec3(dx, dy, dz).normalize_or_zero()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
}

impl Vertex {
    fn new(position: Vec3, normal: Vec3) -> Self {
        Self { position, normal }
    }

    fn translated(self, offset: Vec3) -> Self {
        Self::new(self.position + offset, self.normal)
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

pub struct TranslateMesh<'a, P> {
    sink: &'a mut P,
    offset: Vec3,
}

impl<'a, P> TranslateMesh<'a, P> {
    pub fn new(sink: &'a mut P, offset: Vec3) -> Self {
        Self { sink, offset }
    }
}

impl<'a, P> PopulateMesh for TranslateMesh<'a, P>
where
    P: PopulateMesh,
{
    type Index = P::Index;

    fn add_vertex(&mut self, vertex: Vertex) -> Self::Index {
        self.sink.add_vertex(vertex.translated(self.offset))
    }

    fn add_face(&mut self, indices: [Self::Index; 3]) {
        self.sink.add_face(indices);
    }

    fn add_triangle(&mut self, vertices: [Vertex; 3]) {
        let vertices = vertices.map(|v| v.translated(self.offset));
        self.sink.add_triangle(vertices)
    }

    fn add_quad(&mut self, vertices: [Vertex; 4]) {
        let vertices = vertices.map(|v| v.translated(self.offset));
        self.sink.add_quad(vertices);
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
        self.positions.push(vertex.position.to_array());
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

fn should_flip_face(a: Vec3, b: Vec3, c: Vec3, n: Vec3) -> bool {
    (b - a).cross(c - a).dot(n) < 0.0
}
