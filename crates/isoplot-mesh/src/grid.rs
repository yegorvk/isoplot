use glam::Vec3;
use std::collections::VecDeque;

use crate::{
    ScalarField,
    octree::{Branch, ChildIndex, ImplicitOctree, Node, Octree},
    quant::Quant,
    tables::{
        Corner, EdgeKind, FaceKind, for_each_cell_edge, for_each_cell_face, for_each_face_edge,
        for_each_sub_edge, for_each_sub_face,
    },
    utils::array_transpose,
};

struct Implicit<S, P> {
    scalar_field: S,
    max_level: u8,
    place_feature: P,
}

impl<S, P> ImplicitOctree<Feature> for Implicit<S, P>
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
        let corners = {
            let mut corners = 0u8;

            tag.for_each_corner(|i, corner| {
                let value = self.scalar_field.sample(corner);

                if value.is_sign_positive() {
                    corners |= 1u8 << i;
                }
            });

            corners
        };

        Feature {
            vertex: (self.place_feature)(tag),
            quant: tag,
            corners,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Feature {
    vertex: Vec3,
    quant: Quant,
    corners: u8,
}

impl Feature {
    fn is_feature_edge(&self, a: u8, b: u8) -> bool {
        self.is_corner_sign_positive(a) != self.is_corner_sign_positive(b)
    }

    fn is_corner_sign_positive(&self, corner: u8) -> bool {
        self.corners & (1u8 << corner) != 0
    }
}

pub struct AdaptiveGrid {
    octree: Octree<Feature>,
}

impl AdaptiveGrid {
    pub fn build<S, P>(field: S, max_level: u8, place_feature: P) -> Self
    where
        S: ScalarField,
        P: Fn(Quant) -> Vec3,
    {
        let mut source = Implicit {
            scalar_field: field,
            max_level,
            place_feature,
        };

        Self {
            octree: Octree::build(&mut source),
        }
    }

    pub fn for_each_feature_edge<F>(&self, mut f: F)
    where
        F: FnMut([Vec3; 4]),
    {
        let mut f = |kind: EdgeKind, edge: [Feature; 4]| {
            let indices = match kind {
                EdgeKind::X => [[6, 7], [4, 5], [0, 1], [2, 3]],
                EdgeKind::Y => [[5, 7], [4, 6], [0, 2], [1, 3]],
                EdgeKind::Z => [[3, 7], [2, 6], [0, 4], [1, 5]],
            };

            let (min_index, _) = (edge.iter().enumerate())
                .max_by_key(|(_, feature)| feature.quant.level())
                .unwrap();

            let [a, b] = indices[min_index];

            if edge[min_index].is_feature_edge(a, b) {
                f(edge.map(|feature| feature.vertex));
            }
        };

        let mut faces_x = VecDeque::new();
        let mut faces_y = VecDeque::new();
        let mut faces_z = VecDeque::new();

        let mut edges_x = VecDeque::new();
        let mut edges_y = VecDeque::new();
        let mut edges_z = VecDeque::new();

        let refine_branch = |branch: &Branch, which: Corner| {
            branch
                .child(ChildIndex::new(which.as_u8()))
                .map(|key| self.octree.get(key).unwrap().copied())
        };

        self.octree.for_each_branch(|branch| {
            for_each_cell_face(branch, FaceKind::X, refine_branch, |face| {
                if let Some(face) = array_transpose(face) {
                    faces_x.push_back(face);
                }
            });

            for_each_cell_face(branch, FaceKind::Y, refine_branch, |face| {
                if let Some(face) = array_transpose(face) {
                    faces_y.push_back(face);
                }
            });

            for_each_cell_face(branch, FaceKind::Z, refine_branch, |face| {
                if let Some(face) = array_transpose(face) {
                    faces_z.push_back(face);
                }
            });

            for_each_cell_edge(branch, EdgeKind::X, refine_branch, |edge| {
                if let Some(edge) = array_transpose(edge) {
                    edges_x.push_back(edge);
                }
            });

            for_each_cell_edge(branch, EdgeKind::Y, refine_branch, |edge| {
                if let Some(edge) = array_transpose(edge) {
                    edges_y.push_back(edge);
                }
            });

            for_each_cell_edge(branch, EdgeKind::Z, refine_branch, |edge| {
                if let Some(edge) = array_transpose(edge) {
                    edges_z.push_back(edge);
                }
            });
        });

        let refine_node = |node: &Node<_>, which: Corner| -> Option<Node<Feature>> {
            if let Node::Branch(branch) = node {
                (branch.child(ChildIndex::new(which.as_u8())))
                    .map(|key| self.octree.get(key).unwrap().copied())
            } else {
                Some(*node)
            }
        };

        while let Some(face) = faces_x.pop_front() {
            if face.iter().all(|node| node.is_leaf()) {
                continue;
            }

            for_each_sub_face(face, FaceKind::X, refine_node, |face| {
                if let Some(face) = array_transpose(face) {
                    faces_x.push_back(face);
                }
            });

            for_each_face_edge(face, (FaceKind::X, EdgeKind::Y), refine_node, |edge| {
                if let Some(edge) = array_transpose(edge) {
                    edges_y.push_back(edge);
                }
            });

            for_each_face_edge(face, (FaceKind::X, EdgeKind::Z), refine_node, |edge| {
                if let Some(edge) = array_transpose(edge) {
                    edges_z.push_back(edge);
                }
            });
        }

        while let Some(face) = faces_y.pop_front() {
            if face.iter().all(|node| node.is_leaf()) {
                continue;
            }

            for_each_sub_face(face, FaceKind::Y, refine_node, |face| {
                if let Some(face) = array_transpose(face) {
                    faces_y.push_back(face);
                }
            });

            for_each_face_edge(face, (FaceKind::Y, EdgeKind::X), refine_node, |edge| {
                if let Some(edge) = array_transpose(edge) {
                    edges_x.push_back(edge);
                }
            });

            for_each_face_edge(face, (FaceKind::Y, EdgeKind::Z), refine_node, |edge| {
                if let Some(edge) = array_transpose(edge) {
                    edges_z.push_back(edge);
                }
            });
        }

        while let Some(face) = faces_z.pop_front() {
            if face.iter().all(|node| node.is_leaf()) {
                continue;
            }

            for_each_sub_face(face, FaceKind::Z, refine_node, |face| {
                if let Some(face) = array_transpose(face) {
                    faces_z.push_back(face);
                }
            });

            for_each_face_edge(face, (FaceKind::Z, EdgeKind::X), refine_node, |edge| {
                if let Some(edge) = array_transpose(edge) {
                    edges_x.push_back(edge);
                }
            });

            for_each_face_edge(face, (FaceKind::Z, EdgeKind::Y), refine_node, |edge| {
                if let Some(edge) = array_transpose(edge) {
                    edges_y.push_back(edge);
                }
            });
        }

        while let Some(edge) = edges_x.pop_front() {
            if edge.iter().all(|node| node.is_leaf()) {
                f(EdgeKind::X, edge.map(|node| node.unwrap_leaf()));
                continue;
            }

            for_each_sub_edge(edge, EdgeKind::X, refine_node, |edge| {
                if let Some(edge) = array_transpose(edge) {
                    edges_x.push_back(edge);
                }
            });
        }

        while let Some(edge) = edges_y.pop_front() {
            if edge.iter().all(|node| node.is_leaf()) {
                f(EdgeKind::Y, edge.map(|node| node.unwrap_leaf()));
                continue;
            }

            for_each_sub_edge(edge, EdgeKind::Y, refine_node, |edge| {
                if let Some(edge) = array_transpose(edge) {
                    edges_y.push_back(edge);
                }
            });
        }

        while let Some(edge) = edges_z.pop_front() {
            if edge.iter().all(|node| node.is_leaf()) {
                f(EdgeKind::Z, edge.map(|node| node.unwrap_leaf()));
                continue;
            }

            for_each_sub_edge(edge, EdgeKind::Z, refine_node, |edge| {
                if let Some(edge) = array_transpose(edge) {
                    edges_z.push_back(edge);
                }
            });
        }
    }
}
