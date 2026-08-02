use std::collections::HashMap;

use crate::frontend::{ParseError, ValidateError, parse};
use tape::Tape;
use thiserror::Error;

#[cfg(feature = "cranelift")]
pub use crate::backend::Cranelift;

pub use crate::backend::{Backend, Evaluator, Instance, Interpreter};

mod backend;
mod frontend;
mod tape;

#[derive(Error, Debug)]
pub enum CompileError {
    #[error("failed to parse expression")]
    Parse(#[from] ParseError),

    #[error("failed to validate expression")]
    Validate(#[from] ValidateError),
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
        let tape = parse(source)?
            .validate(desc.shape)?
            .lower_to_ir(&desc.consts);

        Ok(Self { tape })
    }

    pub fn instantiate<B: Backend>(self) -> Instance<B> {
        Instance::new(self.tape)
    }
}
