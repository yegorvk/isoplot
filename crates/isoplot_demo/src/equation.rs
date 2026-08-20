use bevy::math::Vec3;
use isoplot_eval::{
    CompileError, DefaultBackend, Diagnostic, Evaluator, Gradient, Instance, Interval, Program,
    ProgramDesc,
};
use isoplot_mesh::{NormalField, ScalarField};

use crate::plot::PlotSource;

const MIN_NORMAL_ALIGNMENT: f32 = 0.98;

pub struct Equation {
    field: Instance<DefaultBackend, [f32; 3], f32>,
    grad: Instance<DefaultBackend, [f32; 3], Gradient<[f32; 3]>>,
    bounds: Instance<DefaultBackend, [Interval; 3], Interval>,
    grad_bounds: Instance<DefaultBackend, [Interval; 3], Gradient<[Interval; 3]>>,
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
    field_bounds: Evaluator<DefaultBackend, [Interval; 3], Interval>,
    grad: Evaluator<DefaultBackend, [f32; 3], Gradient<[f32; 3]>>,
    grad_bounds: Evaluator<DefaultBackend, [Interval; 3], Gradient<[Interval; 3]>>,
}

impl ScalarField for DynamicSource {
    fn sample(&self, point: Vec3) -> f32 {
        self.field.evaluate(&point.to_array())
    }

    fn is_flat(&self, min: Vec3, size: f32) -> bool {
        let max = min + Vec3::splat(size);

        let cell = [
            Interval::new(min.x, max.x),
            Interval::new(min.y, max.y),
            Interval::new(min.z, max.z),
        ];

        if !self.field_bounds.evaluate(&cell).contains_zero() {
            return true;
        }

        let [g_x, g_y, g_z] = self.grad_bounds.evaluate(&cell).gradient;
        let center = Vec3::new(g_x.center(), g_y.center(), g_z.center()).normalize_or_zero();

        (0..8u8).all(|i| {
            let pick = |range: Interval, bit: u8| {
                if i & bit != 0 {
                    range.max()
                } else {
                    range.min()
                }
            };

            let vertex = Vec3::new(pick(g_x, 1), pick(g_y, 2), pick(g_z, 4));
            vertex.normalize_or_zero().dot(center) > MIN_NORMAL_ALIGNMENT
        })
    }
}

impl DynamicSource {
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
