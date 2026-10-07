use isoplot_math::Vec3;

use crate::{
    lattice::{
        BEdgeIndex, BEdgesMask, Corner, Edge, Edges, Faces, MinimalEdges, edge_normal_face,
        for_each_cell_edge, for_each_cell_face,
    },
    octree::{BuildOctree, ChildIndex, Node, Octree, OctreeKey},
    quant::Quant,
    source::ScalarField,
    utils::array_transpose,
};

struct OctreeSource<S, P> {
    scalar_field: S,
    max_level: u8,
    place_feature: P,
}

impl<S, P> BuildOctree<Feature> for OctreeSource<S, P>
where
    S: ScalarField,
    P: Fn(Quant) -> Option<Vec3<f32>>,
{
    #[inline]
    fn is_empty(&mut self, tag: Quant) -> bool {
        let (min_p, size) = tag.min_point_size();
        self.scalar_field.is_empty(min_p, size)
    }

    #[inline]
    fn is_leaf(&mut self, tag: Quant) -> bool {
        if tag.level() >= self.max_level {
            return true;
        }

        let (min_p, size) = tag.min_point_size();
        self.scalar_field.is_flat(min_p, size)
    }

    #[inline]
    fn place_leaf(&mut self, tag: Quant) -> Feature {
        let (min_corner, size) = tag.min_point_size();

        let mask = BEdgesMask::from_fn(|index| {
            let [a, b] = index
                .corners()
                .map(|corner| min_corner + corner.0.as_vec3().cast() * size);

            self.scalar_field.find_intersection(a, b).get().is_some()
        });

        let vertex = (self.place_feature)(tag)
            .unwrap_or_else(|| tag.center_point())
            .clamp(min_corner, min_corner + size);

        Feature {
            vertex,
            level: tag.level(),
            mask,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub(super) struct Feature {
    pub(super) vertex: Vec3<f32>,
    level: u8,
    mask: BEdgesMask,
}

impl Feature {
    pub(super) const fn contains_intersection(&self, index: BEdgeIndex) -> bool {
        self.mask.contains(index)
    }
}

#[derive(Debug)]
pub struct Chunk {
    pub(super) octree: Octree<Feature>,
}

impl Chunk {
    pub(crate) fn build<S, P>(field: S, max_level: u8, place_feature: P) -> Self
    where
        S: ScalarField,
        P: Fn(Quant) -> Option<Vec3<f32>>,
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

    pub(crate) fn for_each_quad<F>(&self, mut f: F)
    where
        F: FnMut([Vec3<f32>; 4]),
    {
        let mut faces = Faces::default();
        let mut edges = Edges::default();

        self.octree.for_each_branch(|branch| {
            let refine = |which: Corner| {
                let child = ChildIndex::new(which.0.as_u8());
                branch.child(child)
            };

            faces.for_each_axis_mut(|kind, faces| {
                for_each_cell_face(kind, &refine, |keys| {
                    if let Some(keys) = array_transpose(keys) {
                        faces.push(keys);
                    }
                });
            });

            edges.for_each_axis_mut(|kind, edges| {
                for_each_cell_edge(kind, &refine, |keys| {
                    if let Some(keys) = array_transpose(keys) {
                        edges.push(keys);
                    }
                });
            });
        });

        let refine_node = |key: &OctreeKey, which: Corner| {
            if let Node::Branch(branch) = self.octree.get(*key).unwrap() {
                branch.child(ChildIndex::new(which.0.as_u8()))
            } else {
                Some(*key)
            }
        };

        MinimalEdges::new(faces, edges).traverse_single(
            |key| self.octree.is_leaf(*key),
            refine_node,
            |edge| {
                let edge = edge.map(|key| self.get_feature(key).unwrap());

                if contains_intersection(&edge) {
                    f(edge.cells.map(|feature| feature.vertex));
                }
            },
        );
    }

    pub(super) fn get_feature(&self, key: OctreeKey) -> Option<&Feature> {
        self.octree.get(key).and_then(|key| key.as_leaf().copied())
    }
}

pub(super) fn contains_intersection(edge: &Edge<&Feature>) -> bool {
    let (slot, feature) = edge
        .slot_cells()
        .max_by_key(|(_, feature)| feature.level)
        .unwrap();

    feature.contains_intersection(edge_normal_face(edge.kind, slot).into())
}
