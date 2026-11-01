use std::{array, iter};

use fearless_simd::{Simd, SimdBase, SimdFloat, SimdMask, f32x8, mask32x8};
use fearless_simd_macros::simd;
use isoplot_math::{Vec3, Vec3S};

use crate::{
    lattice::{
        BEdgeIndex, BEdges, BEdgesMask, BEdgesMaskS, Corner, Edge, Edges, Faces, MinimalEdges,
        edge_normal_face, for_each_cell_edge, for_each_cell_face,
    },
    octree::{ChildIndex, Node, Octree, OctreeKey, OctreeSource},
    quant::{Quant, QuantS},
    simd::{BMask8, VecExt, WriteBuf, WriteBufExt},
    source::ScalarField,
    utils::array_transpose,
};

#[derive(Copy, Clone, Debug)]
pub(crate) struct Cell {
    pub(crate) vol: Quant,
    mask: BEdgesMask,
}

#[derive(Copy, Clone)]
pub(crate) struct CellS<S: Simd> {
    pub(crate) vol: QuantS<S>,
    mask: BEdgesMaskS<S>,
}

impl<S: Simd> CellS<S> {
    #[inline(always)]
    pub(crate) fn contains_intersection(&self, index: BEdgeIndex) -> mask32x8<S> {
        self.mask.contains(index)
    }

    #[inline(always)]
    fn into_cells(self) -> [Cell; 8] {
        let vol = self.vol.to_array();
        let mask = self.mask.to_array();
        array::from_fn(|i| Cell {
            vol: vol[i],
            mask: mask[i],
        })
    }
}

impl Cell {
    pub(crate) fn contains_intersection(&self, index: BEdgeIndex) -> bool {
        self.mask.contains(index)
    }
}

#[derive(Copy, Clone, Debug)]
pub(super) struct Feature {
    pub(super) vertex: Vec3<f32>,
    level: u8,
    mask: BEdgesMask,
}

impl Feature {
    pub(super) fn contains_intersection(&self, index: BEdgeIndex) -> bool {
        self.mask.contains(index)
    }
}

pub(crate) trait PlaceFeature {
    /// Places a feature inside the given cell.
    fn place_feature(&mut self, cell: Cell) -> Vec3<f32>;

    #[inline]
    fn place_feature_s<'out, S: Simd, B>(&mut self, simd: S, cell: &[CellS<S>], out: B)
    where
        B: WriteBuf<'out, Item = Vec3S<f32x8<S>>>,
    {
        out.write_all_with(|out| {
            out.write_from_iter(cell.iter().map(|cell| {
                let vertices = cell.into_cells().map(|cell| self.place_feature(cell));

                Vec3S::simd_new(
                    simd,
                    vertices.map(|v| v.x),
                    vertices.map(|v| v.y),
                    vertices.map(|v| v.z),
                )
            }));
        });
    }
}

struct Source<F, P> {
    scalar_field: F,
    max_level: u8,
    place_feature: P,
    temp: Vec<(Vec3<[f32; 8]>, [f32; 8])>,
}

impl<F, P> OctreeSource<Feature> for Source<F, P>
where
    F: ScalarField,
    P: PlaceFeature,
{
    #[inline]
    fn is_empty(&mut self, tag: Quant) -> bool {
        let (min_p, size) = tag.min_point_size();
        self.scalar_field.is_empty(min_p, size)
    }

    #[simd]
    fn is_child_empty_mask_s<'out, S: Simd, B>(&mut self, simd: S, tag: &[Quant], out: B)
    where
        B: WriteBuf<'out, Item = BMask8>,
    {
        self.temp.clear();
        self.temp.extend(tag.iter().map(|tag| {
            let quant = tag.children_s(simd).unwrap();
            let (min_p, size) = quant.min_point_size();
            (min_p.simd_to_array(), size.to_array())
        }));

        self.scalar_field.is_empty_s(Vec3::ZERO, &self.temp, out);
    }

    #[inline]
    fn is_leaf(&mut self, tag: Quant) -> bool {
        if tag.level() >= self.max_level {
            return true;
        }

        let (min_p, size) = tag.min_point_size();
        self.scalar_field.is_flat(min_p, size)
    }

    #[simd]
    fn is_leaf_s<'out, S: Simd, B>(&mut self, simd: S, tag: &[[Quant; 8]], out: B)
    where
        B: WriteBuf<'out, Item = BMask8>,
    {
        self.temp.clear();
        self.temp.extend(tag.iter().map(|tag| {
            let quant = QuantS::load(simd, *tag);
            let (min_p, size) = quant.min_point_size();
            (min_p.simd_to_array(), size.to_array())
        }));

        self.scalar_field.is_flat_s(
            Vec3::ZERO,
            &self.temp,
            out.apply_index(|i, mask| {
                let level = QuantS::load(simd, tag[i]).level();
                *mask = (level.simd_ge(self.max_level as u32) | mask.unpack_32x8(simd)).into();
            }),
        );
    }

    #[inline]
    fn place_leaf(&mut self, tag: Quant) -> Feature {
        let (min_corner, size) = tag.min_point_size();

        let mask = BEdgesMask::from_fn(|index| {
            let [a, b] = index
                .corners()
                .map(|corner| min_corner + corner.0.as_vec3().cast() * size);

            if !self.scalar_field.may_contain_intersection(a, b) {
                return false;
            }

            let v_a = self.scalar_field.sample(a);
            let v_b = self.scalar_field.sample(b);

            v_a.is_sign_positive() != v_b.is_sign_positive()
        });

        let vertex = self
            .place_feature
            .place_feature(Cell { vol: tag, mask })
            .clamp(min_corner, min_corner + size);

        Feature {
            vertex,
            level: tag.level(),
            mask,
        }
    }

    #[simd]
    fn place_leaves_s<'out, S: Simd, B>(&mut self, simd: S, tags: &[Quant], out: B)
    where
        B: WriteBuf<'out, Item = Feature>,
    {
        let (chunks, tail) = tags.as_chunks::<8>();

        self.temp.clear();
        self.temp.extend(chunks.iter().map(|tag| {
            let quant = QuantS::load(simd, *tag);
            let (min_p, size) = quant.min_point_size();
            (min_p.simd_to_array(), size.to_array())
        }));

        let a_b: Vec<BEdges<[Vec3<[f32; 8]>; 2]>> = chunks
            .iter()
            .map(|chunk| {
                let (min_corner, size) = QuantS::load(simd, *chunk).min_point_size();

                BEdges::from_fn(|index| {
                    simd.vectorize(
                        #[inline(always)]
                        || {
                            index.corners().map(|corner| {
                                let corner = corner.0.as_vec3().cast_f32();
                                (min_corner + Vec3S::simd_spread(simd, corner) * size)
                                    .simd_to_array()
                            })
                        },
                    )
                })
            })
            .collect();

        let mut v_a_b: Vec<[[[f32; 8]; 2]; 12]> = vec![];
        self.scalar_field.sample_s(
            Vec3::ZERO,
            bytemuck::cast_slice(&a_b),
            v_a_b.extend_write(a_b.len()).flatten().flatten(),
        );

        let mut temp_masks: Vec<[BMask8; 12]> = vec![];
        self.scalar_field.may_contain_intersection_s(
            Vec3::ZERO,
            bytemuck::cast_slice(&a_b),
            temp_masks.extend_write(a_b.len()).flatten(),
        );

        _ = a_b;

        let cells: Vec<CellS<S>> = iter::zip(iter::zip(v_a_b, temp_masks), chunks)
            .map(|((v_a_b, is_candidate), chunk)| {
                let mask = BEdgesMaskS::from_fn(simd, |index| {
                    let [v_a, v_b] = v_a_b[index.as_usize()].map(|v| f32x8::load_array(simd, v));

                    is_candidate[index.as_usize()].unpack_32x8(simd)
                        & v_a.is_sign_positive().simd_ne(v_b.is_sign_positive())
                });

                CellS {
                    vol: QuantS::load(simd, *chunk),
                    mask,
                }
            })
            .collect();

        let mut vertices: Vec<Vec3S<f32x8<S>>> = vec![];
        self.place_feature
            .place_feature_s(simd, &cells, vertices.extend_write(cells.len()));

        out.write_all_with(|out| {
            out.write_from_iter(iter::zip(cells, vertices).flat_map(|(cell, vertex)| {
                iter::zip(cell.into_cells(), vertex.simd_transpose()).map(|(cell, vertex)| {
                    Feature {
                        vertex,
                        level: cell.vol.level(),
                        mask: cell.mask,
                    }
                })
            }));

            out.write_from_iter(tail.iter().map(|tag| self.place_leaf(*tag)));
        });
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
        P: PlaceFeature,
    {
        let mut source = Source {
            scalar_field: field,
            max_level,
            place_feature,
            temp: vec![],
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
