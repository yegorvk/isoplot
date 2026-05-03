use glam::{Vec3, vec3};

#[derive(Copy, Clone, Debug)]
pub struct Point(pub Vec3);

/// A scalar field with source for isosurface extraction
pub trait ScalarField {
    /// Samples the scalar field at the specified point.
    fn sample(&self, point: Point) -> f32;
}

/// A scalar field source with normals
pub trait NormalField: ScalarField {
    /// Samples the scalar field normal at the specified point.
    fn sample_normal(&self, point: Point) -> Vec3;
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
    fn sample(&self, point: Point) -> f32 {
        self.field.sample(point)
    }
}

impl<S: ?Sized + ScalarField> NormalField for CentralDifference<'_, S> {
    fn sample_normal(&self, point: Point) -> Vec3 {
        let p = point.0;
        let e = self.epsilon;

        let f = |p: Vec3| self.field.sample(Point(p));

        let dx = f(p + Vec3::X * e) - f(p - Vec3::X * e);
        let dy = f(p + Vec3::Y * e) - f(p - Vec3::Y * e);
        let dz = f(p + Vec3::Z * e) - f(p - Vec3::Z * e);

        vec3(dx, dy, dz).normalize_or_zero()
    }
}

/// A mesh vertex in the local coordinate frame
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
pub struct DualContouring<'a, S: ?Sized> {
    field: &'a S,
}

impl<'a, S: ?Sized> DualContouring<'a, S> {
    pub fn new(field: &'a S) -> Self {
        Self { field }
    }
}

impl<'a, S: ?Sized + NormalField> DualContouring<'a, S> {
    pub fn extract_with<P>(self, sink: &mut P) -> Result<(), ExtractError>
    where
        P: PopulateMesh,
    {
        const SUBDIV: u32 = 25;

        let cell_size = (SUBDIV as f32).recip();

        for i in 0..SUBDIV {
            for j in 0..SUBDIV {
                let x = i as f32 * cell_size;
                let z = j as f32 * cell_size;

                let positions = [
                    vec3(x, 0.5, z),
                    vec3(x, 0.5, z + cell_size),
                    vec3(x + cell_size, 0.5, z + cell_size),
                    vec3(x + cell_size, 0.5, z),
                ];

                let vertices = positions.map(|position| {
                    let normal = self.field.sample_normal(Point(position));
                    Vertex::new(position, normal)
                });

                sink.add_quad(vertices);
            }
        }

        Ok(())
    }

    pub fn extract<E>(self) -> Result<E, ExtractError>
    where
        E: Default + PopulateMesh,
    {
        let mut extractor = E::default();
        self.extract_with(&mut extractor)?;
        Ok(extractor)
    }
}
