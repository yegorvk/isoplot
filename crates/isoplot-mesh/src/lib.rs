use glam::{Vec3, vec3};

/// A scalar field source for isosurface extraction
pub trait Source {
    /// Samples the scalar field at the specified point.
    fn sample(&self, point: Vec3) -> f32;

    /// Samples the scalar field normal at the specified point.
    fn sample_normal(&self, point: Vec3) -> Vec3;
}

/// A mesh vertex in the local coordinate frame
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    position: Vec3,
}

impl Vertex {
    fn new(position: Vec3) -> Self {
        Self { position }
    }

    #[inline]
    pub fn position(&self) -> Vec3 {
        self.position
    }
}

/// A consumer of the generated mesh
pub trait Extractor {
    /// An extracted vertex index
    type Index: Copy;

    /// Consumes a vertex and returns its index.
    fn extract_vertex(&mut self, vertex: Vertex) -> Self::Index;

    /// Consumes a face made out of 3 vertex indices.
    fn extract_face(&mut self, indices: [Self::Index; 3]);

    /// Consumes a triangle made out of 3 vertices.
    fn extract_triangle(&mut self, vertices: [Vertex; 3]) {
        let indices = vertices.map(|v| self.extract_vertex(v));
        self.extract_face(indices);
    }

    /// Consumes a quad made out of 4 vertices.
    fn extract_quad(&mut self, vertices: [Vertex; 4]) {
        let [a, b, c, d] = vertices.map(|v| self.extract_vertex(v));
        self.extract_face([a, c, b]);
        self.extract_face([a, d, c]);
    }
}

/// Isosurface extraction error
#[derive(Debug)]
pub struct ExtractError;

/// Dual contouring algorithm
pub struct DualContouring<'a, S> {
    source: &'a S,
}

impl<'a, S> DualContouring<'a, S> {
    pub fn new(source: &'a S) -> Self {
        Self { source }
    }
}

impl<'a, S: Source> DualContouring<'a, S> {
    pub fn extract<E>(self, extractor: &mut E) -> Result<(), ExtractError>
    where
        E: Extractor,
    {
        extractor.extract_quad([
            Vertex::new(vec3(0.0, 0.5, 0.0)),
            Vertex::new(vec3(0.0, 0.5, 1.0)),
            Vertex::new(vec3(1.0, 0.5, 1.0)),
            Vertex::new(vec3(1.0, 0.5, 0.0)),
        ]);

        Ok(())
    }
}
