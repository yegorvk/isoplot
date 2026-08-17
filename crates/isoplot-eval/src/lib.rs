mod autodiff;
mod backend;
mod diag;
mod frontend;
mod interval;
mod program;
mod tape;

pub use diag::{BytePos, Diagnostic, Span};
pub use frontend::dump_ast;

pub use backend::{
    Backend, DefaultBackend, DefaultEvaluator, DefaultInstance, DefaultMultiBackend,
    DefaultMultiEvaluator, DefaultMultiInstance, Evaluator, Fallback, FallbackMulti, Instance,
    ScalarBackend,
};

pub use autodiff::{
    DefaultGradientEvaluator, DefaultGradientInstance, GradientEvaluator, GradientInstance,
};

pub use interval::{
    DefaultIntervalEvaluator, DefaultIntervalInstance, Interval, IntervalEvaluator,
    IntervalInstance,
};

pub use program::{
    CompileError, GradientProgram, InstantiateError, IntervalProgram, Program, ProgramDesc,
    ProgramShape, ShapeBuilder,
};

#[cfg(feature = "cranelift")]
pub use backend::{Cranelift, CraneliftMulti};
