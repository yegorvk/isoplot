mod ast;
mod parser;
mod span;
mod symbol;
mod token;

use std::{
    collections::HashMap,
    error::Error,
    fmt::{self, Display},
};

use crate::{
    ProgramShape,
    tape::{Instr, Tape, TapeBuilder, Type, ValueId},
};
use ast::{Ast, BinOp, ExprId, Intrinsic, SparseMap, UnOp, Value, Visitor};
use symbol::{Interner, Symbol};

#[derive(Copy, Clone, Debug)]
enum VarSlot {
    Const(u32),
    Input(u32),
}

#[derive(Debug)]
pub struct ParseError(());

impl Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PARSE_ERROR")
    }
}

impl Error for ParseError {}

#[derive(Debug)]
pub struct ValidateError(());

impl Display for ValidateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("VALIDATE_ERROR")
    }
}

impl Error for ValidateError {}

pub(crate) fn parse(source: &str) -> Result<Parsed, ParseError> {
    let mut interner = Interner::default();
    Ok(Parsed {
        ast: parser::parse(token::tokenize(source), &mut interner),
        interner,
    })
}

pub(crate) struct Parsed {
    interner: Interner,
    ast: Ast,
}

impl Parsed {
    pub(crate) fn validate(&self, shape: &ProgramShape) -> Result<Validated<'_>, ValidateError> {
        let mut valid = true;
        let resolver = Resolver {
            interner: &self.interner,
            shape,
            valid: &mut valid,
        };

        let vars = SparseMap::build(&self.ast, resolver);

        if !valid {
            return Err(ValidateError(()));
        }

        Ok(Validated {
            ast: &self.ast,
            num_inputs: shape.inputs.len() as u32,
            consts: shape.consts.clone(),
            vars,
        })
    }
}

pub(crate) struct Validated<'a> {
    ast: &'a Ast,
    num_inputs: u32,
    consts: Vec<String>,
    vars: SparseMap<VarSlot>,
}

impl Validated<'_> {
    pub(crate) fn lower_to_ir(&self, consts: &HashMap<String, f32>) -> Tape {
        assert_eq!(consts.len(), self.consts.len());

        let consts: Vec<f32> = (self.consts.iter())
            .map(|name| *consts.get(name).unwrap())
            .collect();

        let mut builder = Tape::builder(vec![Type::F32; self.num_inputs as usize]);

        let generator = Generator {
            vars: &self.vars,
            consts: &consts,
            builder: &mut builder,
        };

        let root = self.ast.fold(generator);

        if (0..self.num_inputs).any(|index| builder.arg(index) == root) {
            builder.instr(Instr::F32Max(root, root));
        }

        builder.build().unwrap()
    }
}

struct Resolver<'a> {
    interner: &'a Interner,
    shape: &'a ProgramShape,
    valid: &'a mut bool,
}

impl Visitor for Resolver<'_> {
    type In<'a> = Option<&'a VarSlot>;
    type Out = Option<VarSlot>;

    fn un_op(&mut self, _id: ExprId, _op: UnOp, _operand: Option<&VarSlot>) -> Option<VarSlot> {
        None
    }

    fn bin_op(
        &mut self,
        _id: ExprId,
        _op: BinOp,
        _lhs: Option<&VarSlot>,
        _rhs: Option<&VarSlot>,
    ) -> Option<VarSlot> {
        None
    }

    fn intrinsic<'a, I>(&mut self, _id: ExprId, intrinsic: Intrinsic, args: I) -> Option<VarSlot>
    where
        I: ExactSizeIterator<Item = Option<&'a VarSlot>>,
    {
        if args.len() != intrinsic.num_args() {
            *self.valid = false;
        }

        None
    }

    fn var(&mut self, _id: ExprId, name: Symbol) -> Option<VarSlot> {
        let name = self.interner.resolve(name).unwrap();

        if let Some(index) = self.shape.inputs.iter().position(|input| input == name) {
            return Some(VarSlot::Input(index as u32));
        }

        if let Some(index) = self.shape.consts.iter().position(|cst| cst == name) {
            return Some(VarSlot::Const(index as u32));
        }

        *self.valid = false;
        None
    }

    fn lit(&mut self, _id: ExprId, _value: Value) -> Option<VarSlot> {
        None
    }

    fn map_error(&mut self, _id: ExprId, _inner: Option<Option<&VarSlot>>) -> Option<VarSlot> {
        *self.valid = false;
        None
    }
}

struct Generator<'a> {
    vars: &'a SparseMap<VarSlot>,
    consts: &'a [f32],
    builder: &'a mut TapeBuilder,
}

impl Visitor for Generator<'_> {
    type In<'a> = ValueId;
    type Out = ValueId;

    fn un_op(&mut self, _id: ExprId, op: UnOp, operand: Self::In<'_>) -> Self::Out {
        match op {
            UnOp::Plus => operand,
            UnOp::Minus => self.builder.instr(Instr::F32Neg(operand)),
        }
    }

    fn bin_op(
        &mut self,
        _id: ExprId,
        op: BinOp,
        lhs: Self::In<'_>,
        rhs: Self::In<'_>,
    ) -> Self::Out {
        let instr = match op {
            BinOp::Add => Instr::F32Add(lhs, rhs),
            BinOp::Sub => Instr::F32Sub(lhs, rhs),
            BinOp::Mul => Instr::F32Mul(lhs, rhs),
            BinOp::Div => Instr::F32Div(lhs, rhs),
            BinOp::Pow => Instr::F32Powf(lhs, rhs),
        };

        self.builder.instr(instr)
    }

    fn intrinsic<'a, I>(&mut self, _id: ExprId, kind: Intrinsic, mut args: I) -> Self::Out
    where
        I: ExactSizeIterator<Item = Self::In<'a>>,
    {
        let arg = args.next().unwrap();

        let instr = match kind {
            Intrinsic::Exp => Instr::F32Exp(arg),
            Intrinsic::Log => Instr::F32Lg(arg),
            Intrinsic::Ln => Instr::F32Ln(arg),
            Intrinsic::Sin => Instr::F32Sin(arg),
            Intrinsic::Cos => Instr::F32Cos(arg),
            Intrinsic::Tan => Instr::F32Tan(arg),
            Intrinsic::Cot => Instr::F32Cot(arg),
        };

        self.builder.instr(instr)
    }

    fn var(&mut self, id: ExprId, _name: Symbol) -> Self::Out {
        match self.vars.get(id).unwrap() {
            VarSlot::Const(index) => {
                let value = self.consts[*index as usize];
                self.builder.instr(Instr::F32Const(value))
            }
            VarSlot::Input(index) => self.builder.arg(*index),
        }
    }

    fn lit(&mut self, _id: ExprId, value: Value) -> Self::Out {
        let Value::F32(value) = value;
        self.builder.instr(Instr::F32Const(value))
    }

    fn map_error(&mut self, _id: ExprId, _inner: Option<Self::In<'_>>) -> Self::Out {
        // Should have been handled by the resolver.
        unreachable!()
    }
}
