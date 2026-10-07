use isoplot_math::Vec3;
use std::iter;

const TILE_SIZE: usize = 8;

/// A scalar field source for isosurface extraction
pub trait ScalarField {
    /// Samples the scalar field at the specified point.
    fn sample(&self, point: Vec3<f32>) -> f32;

    fn sample_batch(&self, point: &[Vec3<[f32; TILE_SIZE]>], out: &mut [[f32; TILE_SIZE]]) {
        for (tile, out) in iter::zip(point, out) {
            #[allow(clippy::needless_range_loop)]
            for i in 0..TILE_SIZE {
                out[i] = self.sample(Vec3::new(tile.x[i], tile.y[i], tile.z[i]));
            }
        }
    }

    /// Returns `true` if the image of the interval `[start, end]` *might* intersect the isosurface.
    #[inline]
    fn may_contain_intersection(&self, start: Vec3<f32>, end: Vec3<f32>) -> bool {
        _ = (start, end);
        false
    }

    /// Returns `true` *only if* the given region is empty.
    #[inline]
    fn is_empty(&self, min: Vec3<f32>, size: f32) -> bool {
        _ = (min, size);
        false
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
    fn translated(self, offset: Vec3<f32>) -> Translate<Self>
    where
        Self: Sized,
    {
        Translate::new(self, offset)
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
    source: S,
    offset: Vec3<f32>,
}

impl<S> Translate<S> {
    pub fn new(source: S, offset: Vec3<f32>) -> Self {
        Self { source, offset }
    }
}

impl<S: ScalarField> ScalarField for Translate<S> {
    #[inline]
    fn sample(&self, point: Vec3<f32>) -> f32 {
        self.source.sample(point + self.offset)
    }

    #[inline]
    fn may_contain_intersection(&self, start: Vec3<f32>, end: Vec3<f32>) -> bool {
        self.source
            .may_contain_intersection(start + self.offset, end + self.offset)
    }

    #[inline]
    fn is_empty(&self, min: Vec3<f32>, size: f32) -> bool {
        self.source.is_empty(min + self.offset, size)
    }

    #[inline]
    fn is_flat(&self, min: Vec3<f32>, size: f32) -> bool {
        self.source.is_flat(min + self.offset, size)
    }
}

impl<S: NormalField> NormalField for Translate<S> {
    #[inline]
    fn sample_normal(&self, point: Vec3<f32>) -> Vec3<f32> {
        self.source.sample_normal(point + self.offset)
    }
}

pub struct CentralDifference<S> {
    source: S,
    delta: f32,
}

impl<S: ScalarField> CentralDifference<S> {
    pub fn new(source: S, delta: f32) -> Self {
        Self { source, delta }
    }
}

impl<S: ScalarField> ScalarField for CentralDifference<S> {
    #[inline]
    fn sample(&self, point: Vec3<f32>) -> f32 {
        self.source.sample(point)
    }

    #[inline]
    fn may_contain_intersection(&self, start: Vec3<f32>, end: Vec3<f32>) -> bool {
        self.source.may_contain_intersection(start, end)
    }

    #[inline]
    fn is_empty(&self, min: Vec3<f32>, size: f32) -> bool {
        self.source.is_empty(min, size)
    }

    #[inline]
    fn is_flat(&self, min: Vec3<f32>, size: f32) -> bool {
        self.source.is_flat(min, size)
    }
}

impl<S: ScalarField> NormalField for CentralDifference<S> {
    #[inline]
    fn sample_normal(&self, point: Vec3<f32>) -> Vec3<f32> {
        let (p, e) = (point, self.delta);
        let f = |p: Vec3<f32>| self.source.sample(p);

        let dx = f(p + Vec3::X * e) - f(p - Vec3::X * e);
        let dy = f(p + Vec3::Y * e) - f(p - Vec3::Y * e);
        let dz = f(p + Vec3::Z * e) - f(p - Vec3::Z * e);

        Vec3::new(dx, dy, dz).normalize_or_zero()
    }
}
