use std::array;

use derive_where::derive_where;
use glam::{IVec3, Vec3};

use crate::{
    ExtractError, NormalField, PopulateMesh, ScalarField, Vertex,
    octree::{Branch, BuildOctree, ChildIndex, Key, Node, Octree},
    quant::Quant,
    should_flip_face,
    topology::{
        Corner, EdgeKind, FaceKind, Neighbors, edge_corners, for_each_cell_edge,
        for_each_cell_face, for_each_face_edge, for_each_sub_edge, for_each_sub_face,
    },
    utils::{array_transpose, traverse_ping_pong},
};

pub trait World {
    fn get_neighbor(&self, this: &Chunk, offset: IVec3) -> Option<&Chunk>;
}

#[derive(Debug)]
pub struct Chunk(AdaptiveGrid);

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

    pub fn extract_seam<'a, N, P>(
        &self,
        this: &Chunk,
        mut peek: N,
        sink: &mut P,
    ) -> Result<(), ExtractError>
    where
        N: FnMut(IVec3) -> Option<&'a Chunk>,
        P: PopulateMesh,
    {
        this.0.for_each_seam_quad(
            |offset| peek(offset).map(|chunk| &chunk.0),
            |vertices| self.add_quad(vertices, sink),
        );

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
    fn contains_sign_change(&self, edge: (Corner, Corner)) -> bool {
        let (a, b) = edge;
        self.is_corner_sign_positive(a) != self.is_corner_sign_positive(b)
    }

    fn is_corner_sign_positive(&self, corner: Corner) -> bool {
        self.p_mask & (1u8 << corner.as_u8()) != 0
    }
}

#[derive(Debug)]
struct AdaptiveGrid {
    octree: Octree<Feature>,
}

impl AdaptiveGrid {
    fn build<S, P>(field: S, max_level: u8, place_feature: P) -> Self
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

    fn for_each_interior_quad<F>(&self, mut f: F)
    where
        F: FnMut([Vec3; 4]),
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

        MinimalEdges::new(faces, edges).traverse(
            |key| self.octree.is_leaf(*key),
            refine_node,
            |kind, keys| {
                let features = keys.map(|key| self.get_feature(key).unwrap());

                if contains_sign_change(kind, features) {
                    f(features.map(|feature| feature.vertex));
                }
            },
        );
    }

    fn for_each_seam_quad<'a, N, F>(&self, peek: N, mut f: F)
    where
        N: FnMut(IVec3) -> Option<&'a Self>,
        F: FnMut([Vec3; 4]),
    {
        let neighbors = Neighbors::from_fn(peek);

        neighbors.for_each_face_seam(Some(self), |kind, tagged| {
            if tagged.iter().any(|(_, cell)| cell.is_none()) {
                return;
            }

            let mut faces = Faces::default();
            faces.insert(kind, array::from_fn(|i| SeamCell::new(i as u8, Key::ROOT)));

            let slots = tagged.map(|(offset, grid)| SeamSlot::new(grid.unwrap(), offset));
            traverse_seam(&slots, faces, Edges::default(), &mut f);
        });

        neighbors.for_each_edge_seam(Some(self), |kind, tagged| {
            if tagged.iter().any(|(_, cell)| cell.is_none()) {
                return;
            }

            let mut edges = Edges::default();
            edges.insert(kind, array::from_fn(|i| SeamCell::new(i as u8, Key::ROOT)));

            let slots = tagged.map(|(offset, grid)| SeamSlot::new(grid.unwrap(), offset));
            traverse_seam(&slots, Faces::default(), edges, &mut f);
        });
    }

    fn get_feature(&self, key: Key) -> Option<&Feature> {
        self.octree.get(key).and_then(|key| key.as_leaf().copied())
    }
}

#[derive(Copy, Clone, Debug)]
struct SeamSlot<'a> {
    grid: &'a AdaptiveGrid,
    offset: IVec3,
}

impl<'a> SeamSlot<'a> {
    fn new(grid: &'a AdaptiveGrid, offset: IVec3) -> Self {
        Self { grid, offset }
    }
}

#[derive(Copy, Clone, Debug)]
struct SeamCell {
    which: u8,
    key: Key,
}

impl SeamCell {
    const fn new(which: u8, key: Key) -> Self {
        Self { which, key }
    }
}

fn traverse_seam<F>(
    slots: &[SeamSlot<'_>],
    faces: Faces<SeamCell>,
    edges: Edges<SeamCell>,
    mut f: F,
) where
    F: FnMut([Vec3; 4]),
{
    let is_leaf = |cell: &SeamCell| {
        let slot = slots[cell.which as usize];
        slot.grid.octree.is_leaf(cell.key)
    };

    let refine = |cell: &SeamCell, which: Corner| {
        let slot = slots[cell.which as usize];

        if let Node::Branch(branch) = slot.grid.octree.get(cell.key).unwrap() {
            let child = branch.child(ChildIndex::new(which.as_u8()));
            child.map(|key| SeamCell::new(cell.which, key))
        } else {
            Some(*cell)
        }
    };

    MinimalEdges::new(faces, edges).traverse(is_leaf, refine, |kind, cells| {
        let features = cells.map(|cell| {
            let slot = slots[cell.which as usize];
            slot.grid.get_feature(cell.key).unwrap()
        });

        if contains_sign_change(kind, features) {
            let vertices = cells.map(|cell| {
                let slot = slots[cell.which as usize];
                let node = slot.grid.octree.get(cell.key).unwrap();
                node.unwrap_leaf().quant.center_point() + slot.offset.as_vec3()
            });

            f(vertices);
        }
    });
}

fn contains_sign_change(kind: EdgeKind, features: [&Feature; 4]) -> bool {
    let (max_feature, [a, b]) = edge_corners(features, kind, |_, corner| corner)
        .max_by_key(|(feature, _)| feature.quant.level())
        .unwrap();

    max_feature.contains_sign_change((a, b))
}

struct MinimalEdges<T> {
    faces: Faces<T>,
    edges: Edges<T>,
}

impl<T: Copy> MinimalEdges<T> {
    fn new(faces: Faces<T>, edges: Edges<T>) -> Self {
        Self { faces, edges }
    }

    fn traverse<L, R, F>(mut self, is_leaf: L, refine: R, mut f: F)
    where
        L: Fn(&T) -> bool,
        R: Fn(&T, Corner) -> Option<T>,
        F: FnMut(EdgeKind, [T; 4]),
    {
        for (kind, faces) in self.faces.into_axes() {
            traverse_ping_pong(faces, |current, next| {
                for keys in current.iter().copied() {
                    if keys.iter().all(&is_leaf) {
                        continue;
                    }

                    for_each_sub_face(keys, kind, &refine, |sub_face| {
                        if let Some(sub_face) = array_transpose(sub_face) {
                            next.push(sub_face);
                        }
                    });

                    for edge_kind in kind.tangent_edges() {
                        for_each_face_edge(keys, (kind, edge_kind), &refine, |edge| {
                            if let Some(edge) = array_transpose(edge) {
                                self.edges.insert(edge_kind, edge);
                            }
                        });
                    }
                }
            });
        }

        for (kind, edges) in self.edges.into_axes() {
            traverse_ping_pong(edges, |current, next| {
                for keys in current.iter().copied() {
                    if keys.iter().all(&is_leaf) {
                        f(kind, keys);
                        continue;
                    }

                    for_each_sub_edge(keys, kind, &refine, |sub_edge| {
                        if let Some(edge) = array_transpose(sub_edge) {
                            next.push(edge);
                        }
                    });
                }
            });
        }
    }
}

#[derive_where(Default)]
#[derive(Debug)]
struct Faces<T> {
    x: Vec<[T; 2]>,
    y: Vec<[T; 2]>,
    z: Vec<[T; 2]>,
}

impl<T> Faces<T> {
    fn for_each_axis_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(FaceKind, &mut Vec<[T; 2]>),
    {
        f(FaceKind::X, &mut self.x);
        f(FaceKind::Y, &mut self.y);
        f(FaceKind::Z, &mut self.z);
    }

    fn into_axes(self) -> [(FaceKind, Vec<[T; 2]>); 3] {
        [
            (FaceKind::X, self.x),
            (FaceKind::Y, self.y),
            (FaceKind::Z, self.z),
        ]
    }

    fn insert(&mut self, kind: FaceKind, face: [T; 2]) {
        let faces = match kind {
            FaceKind::X => &mut self.x,
            FaceKind::Y => &mut self.y,
            FaceKind::Z => &mut self.z,
        };

        faces.push(face);
    }
}

#[derive_where(Default)]
#[derive(Debug)]
struct Edges<T> {
    x: Vec<[T; 4]>,
    y: Vec<[T; 4]>,
    z: Vec<[T; 4]>,
}

impl<T> Edges<T> {
    fn for_each_axis_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(EdgeKind, &mut Vec<[T; 4]>),
    {
        f(EdgeKind::X, &mut self.x);
        f(EdgeKind::Y, &mut self.y);
        f(EdgeKind::Z, &mut self.z);
    }

    fn into_axes(self) -> [(EdgeKind, Vec<[T; 4]>); 3] {
        [
            (EdgeKind::X, self.x),
            (EdgeKind::Y, self.y),
            (EdgeKind::Z, self.z),
        ]
    }

    fn insert(&mut self, kind: EdgeKind, edge: [T; 4]) {
        let edges = match kind {
            EdgeKind::X => &mut self.x,
            EdgeKind::Y => &mut self.y,
            EdgeKind::Z => &mut self.z,
        };

        edges.push(edge);
    }
}
