use glam::{IVec3, Vec3};
use std::array;

use crate::{
    ScalarField,
    octree::{Branch, BuildOctree, ChildIndex, Key, Node, Octree},
    quant::Quant,
    topology::{Corner, EdgeKind, Neighbors, edge_corners, for_each_cell_edge, for_each_cell_face},
    utils::array_transpose,
};

use super::connectivity::{Edges, Faces, MinimalEdges};

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
pub struct AdaptiveGrid {
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

    pub fn for_each_interior_quad<F>(&self, mut f: F)
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

    pub fn for_each_seam_quad<'a, N, F>(&self, peek: N, mut f: F)
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
