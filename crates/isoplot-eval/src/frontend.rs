mod ast;
mod parser;
mod pretty;
mod symbol;
mod token;

use std::{
    error::Error,
    fmt::{self, Display},
};

use crate::{
    ProgramShape,
    diag::Diagnostic,
    frontend::ast::DenseMap,
    tape::{Instr, Tape, TapeBuilder, Type, ValueId},
};
use ast::{Ast, BinOp, ExprId, Intrinsic, SparseMap, UnOp, Value, Visitor};
use pretty::PrettyPrintAst;
use symbol::{Interner, Symbol};

#[derive(Debug)]
pub struct ParseError {
    diagnostics: Vec<Diagnostic>,
}

impl ParseError {
    pub(crate) fn new(diagnostics: Vec<Diagnostic>) -> Self {
        Self { diagnostics }
    }

    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }
}

impl Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("failed to parse expression")
    }
}

impl Error for ParseError {}

#[derive(Debug)]
pub struct ValidateError {
    diagnostics: Vec<Diagnostic>,
}

impl ValidateError {
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }
}

impl Display for ValidateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("failed to validate expression")
    }
}

impl Error for ValidateError {}

pub fn dump_ast(shape: &ProgramShape, source: &str) -> (Parsed, Vec<Diagnostic>) {
    let (parsed, mut diagnostics) = parse(source);

    if let Err(error) = parsed.validate(shape) {
        diagnostics.extend(error.diagnostics);
    }

    (parsed, diagnostics)
}

pub(crate) fn parse(source: &str) -> (Parsed, Vec<Diagnostic>) {
    let mut interner = Interner::default();
    let mut diagnostics = Vec::new();
    let ast = parser::parse(token::tokenize(source), &mut interner, &mut diagnostics);

    (Parsed { ast, interner }, diagnostics)
}

pub struct Parsed {
    interner: Interner,
    ast: Ast,
}

impl Parsed {
    pub fn pretty_printer(&self) -> impl Display + '_ {
        PrettyPrintAst::new(&self.ast, &self.interner)
    }

    pub(crate) fn validate(&self, shape: &ProgramShape) -> Result<Validated<'_>, ValidateError> {
        let mut diagnostics = Vec::new();

        let resolver = VarResolver {
            interner: &self.interner,
            shape,
            ast: &self.ast,
            diags: &mut diagnostics,
        };

        let vars = SparseMap::build(&self.ast, resolver);
        let ty = DenseMap::build(&self.ast, TypeChecker);

        if !diagnostics.is_empty() {
            return Err(ValidateError { diagnostics });
        }

        Ok(Validated {
            ast: &self.ast,
            num_inputs: shape.inputs.len() as u32,
            consts: shape.consts.clone(),
            vars,
            ty,
        })
    }
}

pub(crate) struct Validated<'a> {
    ast: &'a Ast,
    num_inputs: u32,
    consts: Vec<String>,
    vars: SparseMap<VarSlot>,
    ty: DenseMap<Type>,
}

impl Validated<'_> {
    pub(crate) fn lower_to_ir<C>(&self, mut resolve_const: C) -> Tape
    where
        C: FnMut(&str) -> f32,
    {
        let consts: Vec<f32> = (self.consts.iter())
            .map(|name| resolve_const(name))
            .collect();

        let mut builder = Tape::builder(vec![Type::F32; self.num_inputs as usize], vec![Type::F32]);

        let generator = Generator {
            vars: &self.vars,
            consts: &consts,
            ty: &self.ty,
            builder: &mut builder,
        };

        let (root, root_ty) = self.ast.fold(generator);

        let root = match root_ty {
            Type::I32 => builder.instr(Instr::F32FromI32(root)),
            Type::F32 => root,
        };

        if (0..self.num_inputs).any(|index| builder.argument(index) == root) {
            builder.instr(Instr::F32Max(root, root));
        }

        builder.build().unwrap()
    }
}

struct VarResolver<'a> {
    interner: &'a Interner,
    shape: &'a ProgramShape,
    ast: &'a Ast,
    diags: &'a mut Vec<Diagnostic>,
}

impl Visitor for VarResolver<'_> {
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

    fn intrinsic<'a, I>(&mut self, id: ExprId, intrinsic: Intrinsic, args: I) -> Option<VarSlot>
    where
        I: ExactSizeIterator<Item = Option<&'a VarSlot>>,
    {
        if args.len() != intrinsic.num_args() {
            self.diags.push(Diagnostic::new(
                format!(
                    "`{}` expects {} argument(s), got {}",
                    intrinsic.name(),
                    intrinsic.num_args(),
                    args.len(),
                ),
                self.ast[id].span,
            ));
        }

        None
    }

    fn var(&mut self, id: ExprId, name: Symbol) -> Option<VarSlot> {
        let name = self.interner.resolve(name).unwrap();

        if let Some(index) = self.shape.inputs.iter().position(|input| input == name) {
            return Some(VarSlot::Input(index as u32));
        }

        if let Some(index) = self.shape.consts.iter().position(|cst| cst == name) {
            return Some(VarSlot::Const(index as u32));
        }

        self.diags.push(Diagnostic::new(
            format!("unknown variable `{name}`"),
            self.ast[id].span,
        ));

        None
    }

    fn lit(&mut self, _id: ExprId, _value: Value) -> Option<VarSlot> {
        None
    }

    fn map_error(&mut self, _id: ExprId, _inner: Option<Option<&VarSlot>>) -> Option<VarSlot> {
        None
    }
}

struct TypeChecker;

impl Visitor for TypeChecker {
    type In<'a> = &'a Type;
    type Out = Type;

    fn un_op(&mut self, _id: ExprId, _op: UnOp, operand: &Type) -> Type {
        *operand
    }

    fn bin_op(&mut self, _id: ExprId, op: BinOp, lhs: &Type, rhs: &Type) -> Type {
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul => match (lhs, rhs) {
                (Type::I32, Type::I32) => Type::I32,
                _ => Type::F32,
            },
            BinOp::Div | BinOp::Pow => Type::F32,
        }
    }

    fn intrinsic<'a, I>(&mut self, _id: ExprId, _kind: Intrinsic, _args: I) -> Type
    where
        I: ExactSizeIterator<Item = Self::In<'a>>,
    {
        Type::F32
    }

    fn var(&mut self, _id: ExprId, _name: Symbol) -> Type {
        Type::F32
    }

    fn lit(&mut self, _id: ExprId, value: Value) -> Type {
        match value {
            Value::I32(_) => Type::I32,
            Value::F32(_) => Type::F32,
        }
    }

    fn map_error(&mut self, _id: ExprId, inner: Option<&Type>) -> Type {
        inner.copied().unwrap_or(Type::F32)
    }
}

struct Generator<'a> {
    vars: &'a SparseMap<VarSlot>,
    consts: &'a [f32],
    ty: &'a DenseMap<Type>,
    builder: &'a mut TapeBuilder,
}

impl Generator<'_> {
    fn promote(&mut self, (value, ty): (ValueId, Type)) -> ValueId {
        match ty {
            Type::I32 => self.builder.instr(Instr::F32FromI32(value)),
            Type::F32 => value,
        }
    }
}

impl Visitor for Generator<'_> {
    type In<'a> = (ValueId, Type);
    type Out = (ValueId, Type);

    fn un_op(&mut self, id: ExprId, op: UnOp, operand: Self::In<'_>) -> Self::Out {
        let ty = self.ty[id];

        let value = match op {
            UnOp::Plus => operand.0,
            UnOp::Minus => match ty {
                Type::I32 => {
                    let zero = self.builder.instr(Instr::I32Const(0));
                    self.builder.instr(Instr::I32Sub(zero, operand.0))
                }
                Type::F32 => {
                    let operand = self.promote(operand);
                    self.builder.instr(Instr::F32Neg(operand))
                }
            },
        };

        (value, ty)
    }

    fn bin_op(&mut self, id: ExprId, op: BinOp, lhs: Self::In<'_>, rhs: Self::In<'_>) -> Self::Out {
        let ty = self.ty[id];

        let instr = match (op, ty) {
            (BinOp::Add, Type::I32) => Instr::I32Add(lhs.0, rhs.0),
            (BinOp::Sub, Type::I32) => Instr::I32Sub(lhs.0, rhs.0),
            (BinOp::Mul, Type::I32) => Instr::I32Mul(lhs.0, rhs.0),

            (BinOp::Add, Type::F32) => Instr::F32Add(self.promote(lhs), self.promote(rhs)),
            (BinOp::Sub, Type::F32) => Instr::F32Sub(self.promote(lhs), self.promote(rhs)),
            (BinOp::Mul, Type::F32) => Instr::F32Mul(self.promote(lhs), self.promote(rhs)),
            (BinOp::Div, Type::F32) => Instr::F32Div(self.promote(lhs), self.promote(rhs)),

            (BinOp::Pow, Type::F32) => match rhs.1 {
                Type::I32 => Instr::F32Powi(self.promote(lhs), rhs.0),
                Type::F32 => Instr::F32Powf(self.promote(lhs), rhs.0),
            },

            (BinOp::Div | BinOp::Pow, Type::I32) => unreachable!(),
        };

        (self.builder.instr(instr), ty)
    }

    fn intrinsic<'a, I>(&mut self, id: ExprId, kind: Intrinsic, mut args: I) -> Self::Out
    where
        I: ExactSizeIterator<Item = Self::In<'a>>,
    {
        let arg = args.next().unwrap();
        let arg = self.promote(arg);

        let instr = match kind {
            Intrinsic::Exp => Instr::F32Exp(arg),
            Intrinsic::Log => Instr::F32Lg(arg),
            Intrinsic::Ln => Instr::F32Ln(arg),
            Intrinsic::Sin => Instr::F32Sin(arg),
            Intrinsic::Cos => Instr::F32Cos(arg),
            Intrinsic::Tan => Instr::F32Tan(arg),
            Intrinsic::Cot => Instr::F32Cot(arg),
            Intrinsic::Abs => Instr::F32Abs(arg),
            Intrinsic::Min => {
                let rhs = args.next().unwrap();
                let rhs = self.promote(rhs);
                Instr::F32Min(arg, rhs)
            }
            Intrinsic::Max => {
                let rhs = args.next().unwrap();
                let rhs = self.promote(rhs);
                Instr::F32Max(arg, rhs)
            }
        };

        (self.builder.instr(instr), self.ty[id])
    }

    fn var(&mut self, id: ExprId, _name: Symbol) -> Self::Out {
        let value = match self.vars.get(id).unwrap() {
            VarSlot::Const(index) => {
                let value = self.consts[*index as usize];
                self.builder.instr(Instr::F32Const(value))
            }
            VarSlot::Input(index) => self.builder.argument(*index),
        };

        (value, self.ty[id])
    }

    fn lit(&mut self, id: ExprId, value: Value) -> Self::Out {
        let instr = match value {
            Value::I32(value) => Instr::I32Const(value),
            Value::F32(value) => Instr::F32Const(value),
        };

        (self.builder.instr(instr), self.ty[id])
    }

    fn map_error(&mut self, _id: ExprId, _inner: Option<Self::In<'_>>) -> Self::Out {
        unreachable!()
    }
}

#[derive(Copy, Clone, Debug)]
enum VarSlot {
    Const(u32),
    Input(u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> ProgramShape {
        ProgramShape::builder().with_input("x").build()
    }

    fn diagnostics(source: &str) -> Vec<(String, usize, usize)> {
        let (parsed, mut diagnostics) = parse(source);

        if let Err(error) = parsed.validate(&shape()) {
            diagnostics.extend(error.diagnostics);
        }

        diagnostics
            .iter()
            .map(|diag| {
                let span = diag.location();
                (
                    diag.message().to_owned(),
                    span.start.index(),
                    span.end.index(),
                )
            })
            .collect()
    }

    #[test]
    fn parse_diagnostics() {
        assert_eq!(
            diagnostics(""),
            [(
                "expected an expression, but found end of input".to_owned(),
                0,
                0
            )]
        );

        assert_eq!(
            diagnostics("1 + $"),
            [("expected an expression, but found `$`".to_owned(), 4, 5)]
        );

        assert_eq!(
            diagnostics("(x + 1"),
            [("expected `)`, but found end of input".to_owned(), 6, 6)]
        );

        assert_eq!(
            diagnostics("sin x"),
            [
                ("expected `(`, but found `x`".to_owned(), 4, 5),
                ("expected end of input, but found `x`".to_owned(), 4, 5),
            ]
        );

        assert_eq!(
            diagnostics("2147483648"),
            [("integer literal is out of range".to_owned(), 0, 10)]
        );
    }

    #[test]
    fn validate_diagnostics() {
        assert_eq!(
            diagnostics("x + y"),
            [("unknown variable `y`".to_owned(), 4, 5)]
        );
        assert_eq!(
            diagnostics("min(x)"),
            [("`min` expects 2 argument(s), got 1".to_owned(), 0, 5)]
        );

        assert_eq!(diagnostics("1 / 2"), []);
        assert_eq!(diagnostics("x / 2"), []);
        assert_eq!(diagnostics("x ^ -2"), []);
    }

    #[test]
    fn validation_with_error_nodes() {
        assert_eq!(
            diagnostics("sin(w"),
            [
                ("expected `)`, but found end of input".to_owned(), 5, 5),
                ("unknown variable `w`".to_owned(), 4, 5),
            ]
        );
    }
}
