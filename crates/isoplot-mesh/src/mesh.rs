use glam::Vec3;

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
}

impl Vertex {
    pub fn new(position: Vec3, normal: Vec3) -> Self {
        Self { position, normal }
    }

    pub fn translated(self, offset: Vec3) -> Self {
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

impl<T: PopulateMesh> PopulateMesh for &mut T {
    type Index = T::Index;

    fn add_vertex(&mut self, vertex: Vertex) -> Self::Index {
        T::add_vertex(self, vertex)
    }

    fn add_face(&mut self, indices: [Self::Index; 3]) {
        T::add_face(self, indices);
    }
}

pub struct TranslateMesh<P> {
    extractor: P,
    offset: Vec3,
}

impl<P> TranslateMesh<P> {
    pub fn new(extractor: P, offset: Vec3) -> Self {
        Self { extractor, offset }
    }
}

impl<P> PopulateMesh for TranslateMesh<P>
where
    P: PopulateMesh,
{
    type Index = P::Index;

    fn add_vertex(&mut self, vertex: Vertex) -> Self::Index {
        self.extractor.add_vertex(vertex.translated(self.offset))
    }

    fn add_face(&mut self, indices: [Self::Index; 3]) {
        self.extractor.add_face(indices);
    }

    fn add_triangle(&mut self, vertices: [Vertex; 3]) {
        let vertices = vertices.map(|v| v.translated(self.offset));
        self.extractor.add_triangle(vertices)
    }

    fn add_quad(&mut self, vertices: [Vertex; 4]) {
        let vertices = vertices.map(|v| v.translated(self.offset));
        self.extractor.add_quad(vertices);
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
