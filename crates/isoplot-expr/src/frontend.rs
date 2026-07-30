mod ast;
mod parser;
mod span;
mod symbol;
mod token;

use std::collections::HashMap;

use crate::{
    Shape,
    instrs::{Instruction, Instructions, InstructionsBuilder, Type, ValueId},
};
use ast::{Ast, BinOp, ExprId, Intrinsic, SparseMap, UnOp, Value, Visitor};
use symbol::{Interner, Symbol};

#[derive(Copy, Clone, Debug)]
enum VarSlot {
    Const(u32),
    Input(u32),
}

#[derive(Debug)]
pub(crate) struct BuildError;

#[derive(Debug)]
pub(crate) struct LowerError;

pub(crate) struct Expr {
    ast: Ast,
    num_inputs: u32,
    consts: Vec<String>,
    vars: SparseMap<VarSlot>,
}

impl Expr {
    pub(crate) fn build(shape: &Shape, source: &str) -> Result<Expr, BuildError> {
        let mut interner = Interner::default();
        let ast = parser::parse(token::tokenize(source), &mut interner);

        let mut valid = true;
        let resolver = Resolver {
            interner: &interner,
            shape,
            valid: &mut valid,
        };

        let vars = SparseMap::build(&ast, resolver);

        if !valid {
            return Err(BuildError);
        }

        Ok(Expr {
            ast,
            num_inputs: shape.inputs.len() as u32,
            consts: shape.consts.clone(),
            vars,
        })
    }

    pub(crate) fn lower_to_ir(
        &self,
        consts: HashMap<String, f32>,
    ) -> Result<Instructions, LowerError> {
        if consts.len() != self.consts.len() {
            return Err(LowerError);
        }

        let consts = self
            .consts
            .iter()
            .map(|name| consts.get(name).copied().ok_or(LowerError))
            .collect::<Result<Vec<f32>, _>>()?;

        let mut builder = Instructions::builder(vec![Type::F32; self.num_inputs as usize]);

        let generator = Generator {
            vars: &self.vars,
            consts: &consts,
            builder: &mut builder,
        };

        let root = self.ast.fold(generator);

        if (0..self.num_inputs).any(|index| builder.arg(index) == root) {
            builder.instr(Instruction::F32Max(root, root));
        }

        Ok(builder.build().unwrap())
    }
}

struct Resolver<'a> {
    interner: &'a Interner,
    shape: &'a Shape,
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
    builder: &'a mut InstructionsBuilder,
}

impl Visitor for Generator<'_> {
    type In<'a> = ValueId;
    type Out = ValueId;

    fn un_op(&mut self, _id: ExprId, op: UnOp, operand: Self::In<'_>) -> Self::Out {
        match op {
            UnOp::Plus => operand,
            UnOp::Minus => self.builder.instr(Instruction::F32Neg(operand)),
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
            BinOp::Add => Instruction::F32Add(lhs, rhs),
            BinOp::Sub => Instruction::F32Sub(lhs, rhs),
            BinOp::Mul => Instruction::F32Mul(lhs, rhs),
            BinOp::Div => Instruction::F32Div(lhs, rhs),
            BinOp::Pow => Instruction::F32Powf(lhs, rhs),
        };

        self.builder.instr(instr)
    }

    fn intrinsic<'a, I>(&mut self, _id: ExprId, kind: Intrinsic, mut args: I) -> Self::Out
    where
        I: ExactSizeIterator<Item = Self::In<'a>>,
    {
        let arg = args.next().unwrap();

        let instr = match kind {
            Intrinsic::Exp => Instruction::F32Exp(arg),
            Intrinsic::Log => Instruction::F32Lg(arg),
            Intrinsic::Ln => Instruction::F32Ln(arg),
            Intrinsic::Sin => Instruction::F32Sin(arg),
            Intrinsic::Cos => Instruction::F32Cos(arg),
            Intrinsic::Tan => Instruction::F32Tan(arg),
            Intrinsic::Cot => Instruction::F32Cot(arg),
        };

        self.builder.instr(instr)
    }

    fn var(&mut self, id: ExprId, _name: Symbol) -> Self::Out {
        match self.vars.get(id).unwrap() {
            VarSlot::Const(index) => {
                let value = self.consts[*index as usize];
                self.builder.instr(Instruction::F32Const(value))
            }
            VarSlot::Input(index) => self.builder.arg(*index),
        }
    }

    fn lit(&mut self, _id: ExprId, value: Value) -> Self::Out {
        let Value::F32(value) = value;
        self.builder.instr(Instruction::F32Const(value))
    }

    fn map_error(&mut self, _id: ExprId, _inner: Option<Self::In<'_>>) -> Self::Out {
        // Should have been handled by the resolver.
        unreachable!()
    }
}
