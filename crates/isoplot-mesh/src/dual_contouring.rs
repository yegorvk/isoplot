use glam::{IVec3, Vec3};

use crate::{
    ExtractError, NormalField, PopulateMesh, ScalarField, Vertex,
    octree::{Branch, BuildOctree, ChildIndex, Key, Node, Octree},
    quant::Quant,
    should_flip_face,
    topology::{
        Corner, EdgeKind, FaceKind, edge_corners, for_each_cell_edge, for_each_cell_face,
        for_each_face_edge, for_each_sub_edge, for_each_sub_face,
    },
    utils::{array_transpose, traverse_ping_pong},
};

pub struct Chunk {
    grid: AdaptiveGrid,
}

/// A region of space with the current chunk at the origin
pub trait Region {
    fn get_chunk(&mut self, offset: IVec3) -> Option<&Chunk>;
}

pub struct EmptyRegion;

impl Region for EmptyRegion {
    fn get_chunk(&mut self, _: IVec3) -> Option<&Chunk> {
        None
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
    pub fn extract_with<R, P>(&self, region: &mut R, sink: &mut P) -> Result<(), ExtractError>
    where
        R: Region,
        P: PopulateMesh,
    {
        let grid = AdaptiveGrid::build(&self.scalar_field, self.max_level, |feature| {
            feature.center_point()
        });

        grid.for_each_quad(|mut vertices| {
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

    pub fn extract<R, P>(&self, region: &mut R) -> Result<P, ExtractError>
    where
        R: Region,
        P: Default + PopulateMesh,
    {
        let mut extractor = P::default();
        self.extract_with(region, &mut extractor)?;
        Ok(extractor)
    }
}

struct OctreeSource<S, P> {
    scalar_field: S,
    max_level: u8,
    place_feature: P,
}

impl<S, P> BuildOctree<Feature> for OctreeSource<S, P>
where
    S: ScalarField,
    P: Fn(Quant) -> Vec3,
{
    type Tag = Quant;

    fn root(&mut self) -> Self::Tag {
        Quant::root()
    }

    fn is_leaf(&mut self, tag: Self::Tag) -> bool {
        let (min_point, _) = tag.min_point_size();

        if (min_point.z.to_bits() + min_point.x.to_bits() + 56) % 3 == 1 {
            tag.level() >= self.max_level - 1
        } else {
            tag.level() >= self.max_level
        }
    }

    fn refine(&mut self, tag: Self::Tag, which: ChildIndex) -> Option<Self::Tag> {
        Some(tag.child(which).unwrap())
    }

    fn place_leaf(&mut self, tag: Self::Tag) -> Feature {
        let p_mask = {
            let mut mask = 0u8;

            tag.for_each_corner(|corner, position| {
                let value = self.scalar_field.sample(position);

                if value.is_sign_positive() {
                    mask |= 1u8 << corner.as_u8();
                }
            });

            mask
        };

        Feature {
            vertex: (self.place_feature)(tag),
            quant: tag,
            p_mask,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Feature {
    vertex: Vec3,
    quant: Quant,
    p_mask: u8,
}

impl Feature {
    fn is_feature_edge(&self, a: u8, b: u8) -> bool {
        self.is_corner_sign_positive(a) != self.is_corner_sign_positive(b)
    }

    fn is_corner_sign_positive(&self, corner: u8) -> bool {
        self.p_mask & (1u8 << corner) != 0
    }
}

#[derive(Debug)]
struct AdaptiveGrid {
    octree: Octree<Feature>,
}

impl AdaptiveGrid {
    pub fn build<S, P>(field: S, max_level: u8, place_feature: P) -> Self
    where
        S: ScalarField,
        P: Fn(Quant) -> Vec3,
    {
        let mut source = OctreeSource {
            scalar_field: field,
            max_level,
            place_feature,
        };

        Self {
            octree: Octree::build(&mut source),
        }
    }

    pub fn for_each_quad<F>(&self, mut f: F)
    where
        F: FnMut([Vec3; 4]),
    {
        self.for_each_minimal_edge(|kind, keys| {
            let (feature, [a, b]) = edge_corners(keys, kind, |_, corner| corner)
                .map(|(key, [a, b])| {
                    let feature = self.octree.get(key).unwrap().unwrap_leaf();
                    (feature, [a, b])
                })
                .max_by_key(|(feature, _)| feature.quant.level())
                .unwrap();

            if feature.is_feature_edge(a.as_u8(), b.as_u8()) {
                f(keys.map(|key| self.octree.get(key).unwrap().unwrap_leaf().vertex));
            }
        });
    }

    fn for_each_minimal_edge<F>(&self, mut f: F)
    where
        F: FnMut(EdgeKind, [Key; 4]),
    {
        let refine_branch =
            |branch: &Branch, which: Corner| branch.child(ChildIndex::new(which.as_u8()));

        let mut faces = Faces::default();
        let mut edges = Edges::default();

        self.octree.for_each_branch(|branch| {
            faces.for_each_axis_mut(|kind, faces| {
                for_each_cell_face(branch, kind, refine_branch, |keys| {
                    if let Some(keys) = array_transpose(keys) {
                        faces.push(keys);
                    }
                });
            });

            edges.for_each_axis_mut(|kind, edges| {
                for_each_cell_edge(branch, kind, refine_branch, |keys| {
                    if let Some(keys) = array_transpose(keys) {
                        edges.push(keys);
                    }
                });
            });
        });

        let refine_node = |key: &Key, which: Corner| {
            if let Node::Branch(branch) = self.octree.get(*key).unwrap() {
                branch.child(ChildIndex::new(which.as_u8()))
            } else {
                Some(*key)
            }
        };

        for (kind, faces) in faces.into_axes() {
            traverse_ping_pong(faces, |current, next| {
                for keys in current.iter().copied() {
                    if keys.iter().all(|key| self.octree.is_leaf(*key)) {
                        continue;
                    }

                    for_each_sub_face(keys, kind, refine_node, |sub_face| {
                        if let Some(sub_face) = array_transpose(sub_face) {
                            next.push(sub_face);
                        }
                    });

                    for edge_kind in kind.tangent_edges() {
                        for_each_face_edge(keys, (kind, edge_kind), refine_node, |edge| {
                            if let Some(edge) = array_transpose(edge) {
                                edges.insert(edge_kind, edge);
                            }
                        });
                    }
                }
            });
        }

        for (kind, edges) in edges.into_axes() {
            traverse_ping_pong(edges, |current, next| {
                for keys in current.iter().copied() {
                    if keys.iter().all(|key| self.octree.is_leaf(*key)) {
                        f(kind, keys);
                        continue;
                    }

                    for_each_sub_edge(keys, kind, refine_node, |sub_edge| {
                        if let Some(edge) = array_transpose(sub_edge) {
                            next.push(edge);
                        }
                    });
                }
            });
        }
    }
}

#[derive(Debug, Default)]
struct Faces {
    x: Vec<[Key; 2]>,
    y: Vec<[Key; 2]>,
    z: Vec<[Key; 2]>,
}

impl Faces {
    fn for_each_axis_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(FaceKind, &mut Vec<[Key; 2]>),
    {
        f(FaceKind::X, &mut self.x);
        f(FaceKind::Y, &mut self.y);
        f(FaceKind::Z, &mut self.z);
    }

    fn into_axes(self) -> [(FaceKind, Vec<[Key; 2]>); 3] {
        [
            (FaceKind::X, self.x),
            (FaceKind::Y, self.y),
            (FaceKind::Z, self.z),
        ]
    }
}

#[derive(Debug, Default)]
struct Edges {
    x: Vec<[Key; 4]>,
    y: Vec<[Key; 4]>,
    z: Vec<[Key; 4]>,
}

impl Edges {
    fn for_each_axis_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(EdgeKind, &mut Vec<[Key; 4]>),
    {
        f(EdgeKind::X, &mut self.x);
        f(EdgeKind::Y, &mut self.y);
        f(EdgeKind::Z, &mut self.z);
    }

    fn into_axes(self) -> [(EdgeKind, Vec<[Key; 4]>); 3] {
        [
            (EdgeKind::X, self.x),
            (EdgeKind::Y, self.y),
            (EdgeKind::Z, self.z),
        ]
    }

    fn insert(&mut self, kind: EdgeKind, edge: [Key; 4]) {
        let edges = match kind {
            EdgeKind::X => &mut self.x,
            EdgeKind::Y => &mut self.y,
            EdgeKind::Z => &mut self.z,
        };

        edges.push(edge);
    }
}
