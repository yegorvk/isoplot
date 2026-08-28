use std::array;

use bevy::math::Vec3;
use isoplot_eval::{
    Bounds, CompileError, DefaultBackend, Diagnostic, Evaluator, Gradient, Instance, Interval,
    Program, ProgramDesc,
};
use isoplot_mesh::{NormalField, ScalarField};

use crate::plot::PlotSource;

const MIN_NORMAL_ALIGNMENT: f32 = 0.98;

pub struct Equation {
    field: Instance<DefaultBackend, [f32; 3], f32>,
    grad: Instance<DefaultBackend, [f32; 3], Gradient<[f32; 3]>>,
    bounds: Instance<DefaultBackend, [Interval; 3], Bounds>,
    grad_bounds: Instance<DefaultBackend, [Interval; 3], Gradient<[Bounds; 3]>>,
}

impl Equation {
    pub fn new(equation: &str) -> Result<Self, CompileError> {
        let program = Program::compile(&ProgramDesc::new(&["x", "y", "z"], &[]), equation)?;
        let grad = program.autodiff();

        Ok(Self {
            grad_bounds: grad.interval().instantiate(),
            grad: grad.instantiate(),
            bounds: program.interval().instantiate(),
            field: program.instantiate(),
        })
    }

    fn create_source(&self) -> DynamicSource {
        DynamicSource {
            field: self.field.evaluator(),
            grad: self.grad.evaluator(),
            field_bounds: self.bounds.evaluator(),
            grad_bounds: self.grad_bounds.evaluator(),
        }
    }
}

impl PlotSource for Equation {
    fn field(&self) -> impl NormalField {
        self.create_source()
    }
}

struct DynamicSource {
    field: Evaluator<DefaultBackend, [f32; 3], f32>,
    field_bounds: Evaluator<DefaultBackend, [Interval; 3], Bounds>,
    grad: Evaluator<DefaultBackend, [f32; 3], Gradient<[f32; 3]>>,
    grad_bounds: Evaluator<DefaultBackend, [Interval; 3], Gradient<[Bounds; 3]>>,
}

impl ScalarField for DynamicSource {
    fn sample(&self, point: Vec3) -> f32 {
        self.field.evaluate(&point.to_array())
    }

    fn find_intersection(&self, start: Vec3, end: Vec3) -> Option<Vec3> {
        const MAX_ITERS: usize = 32;

        let (mut a, mut b) = (start, end);
        let (mut v_a, mut v_b) = (self.sample(a), self.sample(b));

        if 1f32.copysign(v_a) == 1f32.copysign(v_b) {
            return None;
        }

        // Shrink the segment until its bounds are free of discontinuities.
        for _ in 0..MAX_ITERS {
            let segment = array::from_fn(|i| Interval::new(a[i].min(b[i]), a[i].max(b[i])));

            if self.field_bounds.evaluate(&segment).get().is_some() {
                return Some(self.bisect(a, b, v_a, v_b));
            }

            self.refine(&mut a, &mut b, &mut v_a, &mut v_b);
        }

        None
    }

    fn is_flat(&self, min: Vec3, size: f32) -> bool {
        let max = min + Vec3::splat(size);

        let cell = [
            Interval::new(min.x, max.x),
            Interval::new(min.y, max.y),
            Interval::new(min.z, max.z),
        ];

        let Some(bounds) = self.field_bounds.evaluate(&cell).get() else {
            return false;
        };

        if !bounds.contains_zero() {
            return true;
        }

        let [x, y, z] = self.grad_bounds.evaluate(&cell).gradient.map(|x| x.get());

        let (Some(x), Some(y), Some(z)) = (x, y, z) else {
            return false;
        };

        let g = [x, y, z];

        if g.iter().all(|x| x.contains_zero()) {
            return false;
        }

        let c_n = Vec3::from_array(g.map(|x| x.center())).normalize_or_zero();

        (0..8u8).all(|i| {
            let pick = |x: Interval, bit: u8| {
                if i & bit != 0 { x.max() } else { x.min() }
            };

            let vertex = Vec3::from_array(array::from_fn(|i| pick(g[i], 1 << i)));

            if !vertex.is_finite() {
                return false;
            }

            vertex.normalize_or_zero().dot(c_n) > MIN_NORMAL_ALIGNMENT
        })
    }
}

impl DynamicSource {
    fn refine(&self, a: &mut Vec3, b: &mut Vec3, v_a: &mut f32, v_b: &mut f32) {
        let m = a.lerp(*b, 0.5);
        let v_m = self.sample(m);

        if (v_m < 0.0) == (*v_a < 0.0) {
            (*a, *v_a) = (m, v_m);
        } else {
            (*b, *v_b) = (m, v_m);
        }
    }

    fn bisect(&self, mut a: Vec3, mut b: Vec3, mut v_a: f32, mut v_b: f32) -> Vec3 {
        const ITERS: usize = 3;

        for _ in 0..ITERS {
            self.refine(&mut a, &mut b, &mut v_a, &mut v_b);
        }

        let t = if (v_a - v_b).abs() > f32::EPSILON {
            (v_a / (v_a - v_b)).clamp(0.0, 1.0)
        } else {
            0.5
        };

        a.lerp(b, t)
    }

    fn sample_with_gradient(&self, point: Vec3) -> (f32, Vec3) {
        let gradient = self.grad.evaluate(&point.to_array());
        (gradient.value, Vec3::from(gradient.gradient))
    }
}

impl NormalField for DynamicSource {
    fn sample_normal(&self, point: Vec3) -> Vec3 {
        self.sample_with_gradient(point).1.normalize_or_zero()
    }
}

pub fn render_diagnostics(source: &str, diagnostics: &[Diagnostic]) -> String {
    let mut out = String::new();

    for (i, diagnostic) in diagnostics.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }

        let span = diagnostic.location();
        let (start, end) = (span.start.index(), span.end.index());
        let offset = source[..start].chars().count();
        let width = source[start..end].chars().count().max(1);

        out.push_str(&format!(
            "error: {}\n  {source}\n  {}{}",
            diagnostic.message(),
            " ".repeat(offset),
            "^".repeat(width),
        ));
    }

    out
}
