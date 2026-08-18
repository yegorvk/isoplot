mod autodiff;
mod backend;
mod diag;
mod frontend;
mod interval;
mod layout;
mod program;
mod tape;

pub use diag::{BytePos, Diagnostic, Span};
pub use program::dump_ast;

pub use backend::{Backend, DefaultBackend, Evaluator, Fallback, Instance};

pub use autodiff::Gradient;
pub use interval::Interval;
pub use layout::{Layout, TypedValue, ValueType};
pub use program::{CompileError, Program, ProgramDesc};

#[cfg(feature = "cranelift")]
pub use backend::Cranelift;
