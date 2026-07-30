use std::collections::HashMap;

pub use crate::backend::{Backend, Instance, Interpreter};
use crate::{frontend::Expr, instrs::Instructions};

mod backend;
mod frontend;
mod instrs;

#[derive(Debug)]
pub struct BuildError;

#[derive(Debug)]
pub struct SpecializeError;

#[derive(Debug)]
pub struct InstantiateError;

#[derive(Debug)]
pub struct Shape {
    inputs: Vec<String>,
    consts: Vec<String>,
}

impl Shape {
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

    pub fn build(self) -> Shape {
        let all_vars = || self.inputs.iter().chain(&self.consts);
        for (i, name) in all_vars().enumerate() {
            if all_vars().take(i).any(|prev| prev == name) {
                panic!("duplicate variable `{name}`");
            }
        }

        Shape {
            inputs: self.inputs,
            consts: self.consts,
        }
    }
}

pub struct Template {
    expr: Expr,
}

impl Template {
    pub fn build(shape: &Shape, source: &str) -> Result<Self, BuildError> {
        let expr = Expr::build(shape, source).map_err(|_| BuildError)?;
        Ok(Self { expr })
    }

    pub fn specialize(&self, consts: HashMap<String, f32>) -> Result<Program, SpecializeError> {
        let instrs = self.expr.lower_to_ir(consts).map_err(|_| SpecializeError)?;
        Ok(Program { instrs })
    }
}

pub struct Program {
    instrs: Instructions,
}

impl Program {
    pub fn instantiate<B: Backend>(self) -> Instance<B> {
        Instance::new(self.instrs)
    }
}
