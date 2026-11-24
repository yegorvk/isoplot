mod artifact;
mod autodiff;
mod backend;
mod diag;
mod frontend;
mod interval;
mod kernel;
mod schema;

pub(crate) const MIN_F32_MAGNITUDE: f32 = 1e-6;

pub use artifact::{Artifact, CompileError, Shape, dump_ast};
pub use autodiff::Dual;
pub use backend::Fallback;
pub use diag::{BytePos, Diagnostic, Span};
pub use interval::Interval;
pub use kernel::{Evaluate, Kernel};
pub use schema::{FieldType, FieldValue, Schema};

#[cfg(feature = "cranelift")]
pub use backend::Cranelift;
