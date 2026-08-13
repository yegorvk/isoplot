use std::collections::HashMap;

use frontend::{ParseError, ValidateError, parse};
use tape::Tape;
use thiserror::Error;

#[cfg(feature = "cranelift")]
pub use crate::backend::Cranelift;

pub use crate::backend::{Backend, DefaultBackend, DefaultInstance, Evaluator, Fallback, Instance};
pub use crate::diag::{BytePos, Diagnostic, Span};
pub use crate::frontend::dump_ast;

mod backend;
mod diag;
mod frontend;
mod tape;

#[derive(Error, Debug)]
pub enum CompileError {
    #[error(transparent)]
    Parse(#[from] ParseError),

    #[error(transparent)]
    Validate(#[from] ValidateError),
}

impl CompileError {
    pub fn diagnostics(&self) -> &[diag::Diagnostic] {
        match self {
            CompileError::Parse(error) => error.diagnostics(),
            CompileError::Validate(error) => error.diagnostics(),
        }
    }
}

#[derive(Debug)]
pub struct InstantiateError(());

#[derive(Debug)]
pub struct ProgramShape {
    inputs: Vec<String>,
    consts: Vec<String>,
}

impl ProgramShape {
    pub fn builder() -> ShapeBuilder {
        ShapeBuilder {
            inputs: Vec::new(),
            consts: Vec::new(),
        }
    }
}

pub struct ShapeBuilder {
    inputs: Vec<String>,
    consts: Vec<String>,
}

impl ShapeBuilder {
    pub fn with_input(mut self, name: impl Into<String>) -> Self {
        self.inputs.push(name.into());
        self
    }

    pub fn with_const(mut self, name: impl Into<String>) -> Self {
        self.consts.push(name.into());
        self
    }

    pub fn build(self) -> ProgramShape {
        let all_vars = || self.inputs.iter().chain(&self.consts);
        for (i, name) in all_vars().enumerate() {
            if all_vars().take(i).any(|prev| prev == name) {
                panic!("duplicate variable `{name}`");
            }
        }

        ProgramShape {
            inputs: self.inputs,
            consts: self.consts,
        }
    }
}

pub struct ProgramDesc<'a> {
    shape: &'a ProgramShape,
    consts: HashMap<String, f32>,
}

impl<'a> ProgramDesc<'a> {
    pub fn new(shape: &'a ProgramShape, consts: HashMap<String, f32>) -> Self {
        for name in &shape.consts {
            assert!(
                consts.contains_key(name),
                "not all constants were specified"
            )
        }

        Self { shape, consts }
    }
}

pub struct Program {
    tape: Tape,
}

impl Program {
    pub fn compile(desc: &ProgramDesc, source: &str) -> Result<Self, CompileError> {
        let (parsed, mut diagnostics) = parse(source);
        let validated = parsed.validate(desc.shape);

        if !diagnostics.is_empty() {
            if let Err(error) = &validated {
                diagnostics.extend_from_slice(error.diagnostics());
            }
            return Err(ParseError::new(diagnostics).into());
        }

        let tape = validated?.lower_to_ir(|name| *desc.consts.get(name).unwrap());
        Ok(Self { tape })
    }

    pub fn instantiate<B: Backend>(self) -> Instance<B> {
        Instance::new(self.tape)
    }
}
