use glam::{Vec3, vec3};

/// A scalar field source for isosurface extraction
pub trait ScalarField {
    /// Samples the scalar field at the specified point.
    fn sample(&self, point: Vec3) -> f32;

    /// Returns `true` *only if* the given region is flat.
    ///
    /// A region is considered flat when it does not need to be further
    /// subdivided. Note that this method is always allowed to return `false`.
    fn is_flat(&self, min: Vec3, size: f32) -> bool {
        _ = (min, size);
        false
    }

    fn translated(self, offset: Vec3) -> Translate<Self>
    where
        Self: Sized,
    {
        Translate::new(self, offset)
    }
}

impl<S: ?Sized + ScalarField> ScalarField for &S {
    fn sample(&self, point: Vec3) -> f32 {
        <S as ScalarField>::sample(self, point)
    }

    fn is_flat(&self, min: Vec3, size: f32) -> bool {
        <S as ScalarField>::is_flat(&self, min, size)
    }
}

/// A scalar field source with normals
pub trait NormalField: ScalarField {
    /// Samples the scalar field normal at the specified point.
    fn sample_normal(&self, point: Vec3) -> Vec3;
}

impl<S: ?Sized + NormalField> NormalField for &S {
    fn sample_normal(&self, point: Vec3) -> Vec3 {
        <S as NormalField>::sample_normal(self, point)
    }
}

pub struct Translate<S> {
    source: S,
    offset: Vec3,
}

impl<S> Translate<S> {
    pub fn new(source: S, offset: Vec3) -> Self {
        Self { source, offset }
    }
}

impl<S: ScalarField> ScalarField for Translate<S> {
    fn sample(&self, point: Vec3) -> f32 {
        self.source.sample(point + self.offset)
    }

    fn is_flat(&self, min: Vec3, size: f32) -> bool {
        self.source.is_flat(min + self.offset, size)
    }
}

impl<S: NormalField> NormalField for Translate<S> {
    fn sample_normal(&self, point: Vec3) -> Vec3 {
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
    fn sample(&self, point: Vec3) -> f32 {
        self.source.sample(point)
    }

    fn is_flat(&self, min: Vec3, size: f32) -> bool {
        self.source.is_flat(min, size)
    }
}

impl<S: ScalarField> NormalField for CentralDifference<S> {
    fn sample_normal(&self, point: Vec3) -> Vec3 {
        let (p, e) = (point, self.delta);
        let f = |p: Vec3| self.source.sample(p);

        let dx = f(p + Vec3::X * e) - f(p - Vec3::X * e);
        let dy = f(p + Vec3::Y * e) - f(p - Vec3::Y * e);
        let dz = f(p + Vec3::Z * e) - f(p - Vec3::Z * e);

        vec3(dx, dy, dz).normalize_or_zero()
    }
}
