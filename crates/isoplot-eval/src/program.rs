use std::collections::HashMap;
use thiserror::Error;

use crate::{
    autodiff::{GradientInstance, GradientTape},
    backend::{Backend, Instance},
    diag,
    frontend::{ParseError, ValidateError, parse},
    interval::{IntervalInstance, IntervalTape},
    tape::Tape,
};

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
    pub(crate) inputs: Vec<String>,
    pub(crate) consts: Vec<String>,
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

    pub fn autodiff(&self) -> GradientProgram {
        GradientProgram::new(GradientTape::build(&self.tape))
    }

    pub fn interval(&self) -> IntervalProgram {
        IntervalProgram::new(&self.tape)
    }

    pub fn instantiate<B: Backend>(self) -> Instance<B> {
        Instance::new(self.tape)
    }
}

pub struct GradientProgram {
    tape: GradientTape,
}

impl GradientProgram {
    fn new(tape: GradientTape) -> Self {
        Self { tape }
    }

    pub fn interval(&self) -> IntervalProgram {
        IntervalProgram::new(self.tape.tape())
    }

    pub fn instantiate<B: Backend>(self) -> GradientInstance<B> {
        GradientInstance::new(self.tape)
    }
}

pub struct IntervalProgram {
    tape: IntervalTape,
}

impl IntervalProgram {
    fn new(source: &Tape) -> Self {
        Self {
            tape: IntervalTape::build(source),
        }
    }

    pub fn instantiate<B: Backend>(self) -> IntervalInstance<B> {
        IntervalInstance::new(self.tape)
    }
}
