use isoplot_math::Vec3;
use std::array;

use crate::simd::{BMask8, WriteBuf, WriteBufExt};

/// A scalar field source for isosurface extraction
pub trait ScalarField {
    /// Samples the scalar field at the specified point.
    fn sample(&self, point: Vec3<f32>) -> f32;

    #[inline]
    fn sample_s<'out, B>(&self, offset: Vec3<f32>, points: &[Vec3<[f32; 8]>], out: B)
    where
        B: WriteBuf<'out, Item = [f32; 8]>,
    {
        out.write_all_with(|out| {
            out.write_from_iter(points.iter().map(|p| {
                array::from_fn(|i| {
                    let p = Vec3::new(p.x[i], p.y[i], p.z[i]);
                    self.sample(p + offset)
                })
            }));
        });
    }

    /// Returns `false` if the image of the interval `[start, end]` does *not* intersect the isosurface.
    #[inline]
    fn may_contain_intersection(&self, start: Vec3<f32>, end: Vec3<f32>) -> bool {
        _ = (start, end);
        false
    }

    #[inline]
    fn may_contain_intersection_s<'out, B>(
        &self,
        offset: Vec3<f32>,
        endpoints: &[[Vec3<[f32; 8]>; 2]],
        out: B,
    ) where
        B: WriteBuf<'out, Item = BMask8>,
    {
        out.write_all_with(|out| {
            out.write_from_iter(endpoints.iter().map(|endpoints| {
                BMask8::from_fn(|i| {
                    let i = i as usize;

                    let [a, b] = array::from_fn(|j| {
                        offset + Vec3::new(endpoints[j].x[i], endpoints[j].y[i], endpoints[j].z[i])
                    });

                    self.may_contain_intersection(a, b)
                })
            }));
        });
    }

    /// Returns `true` *only if* the given region is empty.
    #[inline]
    fn is_empty(&self, min: Vec3<f32>, size: f32) -> bool {
        _ = (min, size);
        false
    }

    #[inline]
    fn is_empty_s<'out, B>(
        &self,
        offset: Vec3<f32>,
        min_size: &[(Vec3<[f32; 8]>, [f32; 8])],
        out: B,
    ) where
        B: WriteBuf<'out, Item = BMask8>,
    {
        out.write_all_with(|out| {
            out.write_from_iter(min_size.iter().map(|(p, s)| {
                BMask8::from_fn(|i| {
                    let i = i as usize;
                    let p = offset + Vec3::new(p.x[i], p.y[i], p.z[i]);
                    self.is_empty(p, s[i])
                })
            }));
        });
    }

    /// Returns `true` *only if* the given region is flat.
    ///
    /// A region is considered flat when it does not need to be further
    /// subdivided. Consequently, it is always allowed to return `false`.
    #[inline]
    fn is_flat(&self, min: Vec3<f32>, size: f32) -> bool {
        _ = (min, size);
        false
    }

    #[inline]
    fn is_flat_s<'out, B>(&self, offset: Vec3<f32>, min_size: &[(Vec3<[f32; 8]>, [f32; 8])], out: B)
    where
        B: WriteBuf<'out, Item = BMask8>,
    {
        out.write_all_with(|out| {
            out.write_from_iter(min_size.iter().map(|(p, s)| {
                BMask8::from_fn(|i| {
                    let i = i as usize;
                    let p = offset + Vec3::new(p.x[i], p.y[i], p.z[i]);
                    self.is_flat(p, s[i])
                })
            }));
        });
    }

    #[inline]
    fn translated(self, offset: Vec3<f32>) -> Translate<Self>
    where
        Self: Sized,
    {
        Translate {
            field: self,
            offset,
        }
    }
}

impl<S: ?Sized + ScalarField> ScalarField for &S {
    #[inline]
    fn sample(&self, point: Vec3<f32>) -> f32 {
        <S as ScalarField>::sample(self, point)
    }

    #[inline]
    fn may_contain_intersection(&self, start: Vec3<f32>, end: Vec3<f32>) -> bool {
        <S as ScalarField>::may_contain_intersection(self, start, end)
    }

    #[inline]
    fn is_empty(&self, min: Vec3<f32>, size: f32) -> bool {
        <S as ScalarField>::is_empty(self, min, size)
    }

    #[inline]
    fn is_flat(&self, min: Vec3<f32>, size: f32) -> bool {
        <S as ScalarField>::is_flat(self, min, size)
    }

    #[inline(always)]
    fn sample_s<'out, B>(&self, offset: Vec3<f32>, points: &[Vec3<[f32; 8]>], out: B)
    where
        B: WriteBuf<'out, Item = [f32; 8]>,
    {
        <S as ScalarField>::sample_s(self, offset, points, out)
    }

    #[inline(always)]
    fn may_contain_intersection_s<'out, B>(
        &self,
        offset: Vec3<f32>,
        endpoints: &[[Vec3<[f32; 8]>; 2]],
        out: B,
    ) where
        B: WriteBuf<'out, Item = BMask8>,
    {
        <S as ScalarField>::may_contain_intersection_s(self, offset, endpoints, out);
    }

    #[inline(always)]
    fn is_empty_s<'out, B>(
        &self,
        offset: Vec3<f32>,
        min_size: &[(Vec3<[f32; 8]>, [f32; 8])],
        out: B,
    ) where
        B: WriteBuf<'out, Item = BMask8>,
    {
        <S as ScalarField>::is_empty_s(self, offset, min_size, out);
    }

    #[inline(always)]
    fn is_flat_s<'out, B>(&self, offset: Vec3<f32>, min_size: &[(Vec3<[f32; 8]>, [f32; 8])], out: B)
    where
        B: WriteBuf<'out, Item = BMask8>,
    {
        <S as ScalarField>::is_flat_s(self, offset, min_size, out);
    }
}

/// A scalar field source with normals
pub trait NormalField: ScalarField {
    /// Samples the scalar field normal at the specified point.
    fn sample_normal(&self, point: Vec3<f32>) -> Vec3<f32>;
}

impl<S: ?Sized + NormalField> NormalField for &S {
    #[inline]
    fn sample_normal(&self, point: Vec3<f32>) -> Vec3<f32> {
        <S as NormalField>::sample_normal(self, point)
    }
}

pub struct Translate<S> {
    field: S,
    offset: Vec3<f32>,
}

impl<S> Translate<S> {
    pub fn new(field: S, offset: Vec3<f32>) -> Self {
        Self { field, offset }
    }
}

impl<S: ScalarField> ScalarField for Translate<S> {
    #[inline]
    fn sample(&self, point: Vec3<f32>) -> f32 {
        self.field.sample(point + self.offset)
    }

    #[inline]
    fn may_contain_intersection(&self, start: Vec3<f32>, end: Vec3<f32>) -> bool {
        self.field
            .may_contain_intersection(start + self.offset, end + self.offset)
    }

    #[inline]
    fn is_empty(&self, min: Vec3<f32>, size: f32) -> bool {
        self.field.is_empty(min + self.offset, size)
    }

    #[inline]
    fn is_flat(&self, min: Vec3<f32>, size: f32) -> bool {
        self.field.is_flat(min + self.offset, size)
    }

    #[inline(always)]
    fn sample_s<'out, B>(&self, offset: Vec3<f32>, points: &[Vec3<[f32; 8]>], out: B)
    where
        B: WriteBuf<'out, Item = [f32; 8]>,
    {
        self.field.sample_s(offset + self.offset, points, out);
    }

    #[inline(always)]
    fn may_contain_intersection_s<'out, B>(
        &self,
        offset: Vec3<f32>,
        endpoints: &[[Vec3<[f32; 8]>; 2]],
        out: B,
    ) where
        B: WriteBuf<'out, Item = BMask8>,
    {
        self.field
            .may_contain_intersection_s(offset + self.offset, endpoints, out)
    }

    #[inline(always)]
    fn is_empty_s<'out, B>(
        &self,
        offset: Vec3<f32>,
        min_size: &[(Vec3<[f32; 8]>, [f32; 8])],
        out: B,
    ) where
        B: WriteBuf<'out, Item = BMask8>,
    {
        self.field.is_empty_s(offset + self.offset, min_size, out);
    }

    #[inline(always)]
    fn is_flat_s<'out, B>(&self, offset: Vec3<f32>, min_size: &[(Vec3<[f32; 8]>, [f32; 8])], out: B)
    where
        B: WriteBuf<'out, Item = BMask8>,
    {
        self.field.is_flat_s(offset + self.offset, min_size, out)
    }
}

impl<S: NormalField> NormalField for Translate<S> {
    #[inline]
    fn sample_normal(&self, point: Vec3<f32>) -> Vec3<f32> {
        self.field.sample_normal(point + self.offset)
    }
}
