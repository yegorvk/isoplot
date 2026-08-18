use bevy::math::Vec3;
use isoplot_eval::{
    CompileError, DefaultBackend, Diagnostic, Evaluator, Gradient, Instance, Program, ProgramDesc,
};
use isoplot_mesh::{NormalField, ScalarField};

use crate::plot::PlotSource;

pub struct Equation {
    field: Instance<DefaultBackend, [f32; 3], f32>,
    grad: Instance<DefaultBackend, [f32; 3], Gradient<[f32; 3]>>,
}

impl Equation {
    pub fn new(equation: &str) -> Result<Self, CompileError> {
        let program = Program::compile(&ProgramDesc::new(&["x", "y", "z"], &[]), equation)?;

        Ok(Self {
            grad: program.autodiff().instantiate(),
            field: program.instantiate(),
        })
    }

    fn create_source(&self) -> DynamicSource {
        DynamicSource {
            field: self.field.evaluator(),
            grad: self.grad.evaluator(),
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
    grad: Evaluator<DefaultBackend, [f32; 3], Gradient<[f32; 3]>>,
}

impl ScalarField for DynamicSource {
    fn sample(&self, point: Vec3) -> f32 {
        self.field.evaluate(&point.to_array())
    }

    fn is_flat(&self, min: Vec3, size: f32) -> bool {
        let mut samples = [(0.0f32, Vec3::ZERO); 9];

        for (i, sample) in samples.iter_mut().enumerate() {
            let offset = if i < 8 {
                Vec3::new((i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32)
            } else {
                Vec3::splat(0.5)
            };

            *sample = self.sample_with_gradient(min + size * offset);
        }

        let has_negative = samples.iter().any(|(v, _)| v.is_sign_negative());
        let has_positive = samples.iter().any(|(v, _)| v.is_sign_positive());

        if !(has_negative && has_positive) {
            let min_abs = samples
                .iter()
                .map(|(v, _)| v.abs())
                .fold(f32::INFINITY, f32::min);
            let max_slope = samples.iter().map(|(_, g)| g.length()).fold(0.0, f32::max);
            return min_abs > 2.0 * max_slope * size * 3f32.sqrt();
        }

        let mean_normal = samples
            .iter()
            .map(|(_, g)| g.normalize_or_zero())
            .sum::<Vec3>()
            .normalize_or_zero();

        samples
            .iter()
            .all(|(_, g)| g.normalize_or_zero().dot(mean_normal) > 0.995)
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
