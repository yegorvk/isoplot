mod ast;
mod parser;
mod pretty;
mod symbol;
mod token;

use std::{error::Error, fmt};

use crate::{
    diag::Diagnostic,
    layout::{Layout, TypedValue, ValueType},
    tape::{NewId, Tape, TapeBuilder, Type},
};

use ast::{Ast, BinOp, DenseMap, ExprId, Intrinsic, SparseMap, UnOp, Value, Visitor};
use pretty::PrettyPrintAst;
use symbol::{Interner, Symbol};

#[derive(Debug)]
pub(crate) struct ParseError {
    diagnostics: Vec<Diagnostic>,
}

impl ParseError {
    pub(crate) fn new(diagnostics: Vec<Diagnostic>) -> Self {
        Self { diagnostics }
    }

    pub(crate) fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("failed to parse expression")
    }
}

impl Error for ParseError {}

#[derive(Debug)]
pub(crate) struct ValidateError {
    diagnostics: Vec<Diagnostic>,
}

impl ValidateError {
    pub(crate) fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }
}

impl fmt::Display for ValidateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("failed to validate expression")
    }
}

impl Error for ValidateError {}

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
    pub fn pretty_printer(&self) -> impl fmt::Display + '_ {
        PrettyPrintAst::new(&self.ast, &self.interner)
    }

    pub(crate) fn validate(&self, bindings: &Bindings) -> Result<Validated<'_>, ValidateError> {
        let mut diagnostics = Vec::new();

        let resolver = VarResolver {
            interner: &self.interner,
            bindings,
            ast: &self.ast,
            diags: &mut diagnostics,
        };

        let vars = SparseMap::build(&self.ast, resolver);
        let ty = DenseMap::build(&self.ast, TypeChecker { vars: &vars });

        if !diagnostics.is_empty() {
            return Err(ValidateError { diagnostics });
        }

        Ok(Validated {
            ast: &self.ast,
            arg_types: bindings.args.iter().map(|&(_, ty)| ty).collect(),
            consts: bindings.consts.iter().map(|&(_, value)| value).collect(),
            vars,
            ty,
        })
    }
}

pub(crate) struct Validated<'a> {
    ast: &'a Ast,
    arg_types: Vec<Type>,
    consts: Vec<TypedValue>,
    vars: SparseMap<VarSlot>,
    ty: DenseMap<Type>,
}

impl Validated<'_> {
    pub(crate) fn lower_to_ir(&self) -> Tape {
        let mut builder = Tape::builder(self.arg_types.clone(), vec![Type::F32]);

        let generator = Generator {
            vars: &self.vars,
            consts: &self.consts,
            ty: &self.ty,
            builder: &mut builder,
        };

        let root = self.ast.fold(generator);
        let root = root.promote(&mut builder);

        if root.index() < self.arg_types.len() {
            builder.copy_f32(root);
        }

        builder.build().unwrap()
    }
}

struct VarResolver<'a> {
    interner: &'a Interner,
    bindings: &'a Bindings<'a>,
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

        if let Some(index) = self.bindings.args.iter().position(|&(var, _)| var == name) {
            let ty = self.bindings.args[index].1;
            return Some(VarSlot::Arg(VarIndex(index as u32), ty));
        }

        if let Some(index) = self
            .bindings
            .consts
            .iter()
            .position(|&(var, _)| var == name)
        {
            let ty = lower_type(self.bindings.consts[index].1.ty());
            return Some(VarSlot::Const(VarIndex(index as u32), ty));
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

struct TypeChecker<'a> {
    vars: &'a SparseMap<VarSlot>,
}

impl Visitor for TypeChecker<'_> {
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

    fn var(&mut self, id: ExprId, _name: Symbol) -> Type {
        match self.vars.get(id) {
            Some(VarSlot::Const(_, ty) | VarSlot::Arg(_, ty)) => *ty,
            None => Type::F32,
        }
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
    consts: &'a [TypedValue],
    ty: &'a DenseMap<Type>,
    builder: &'a mut TapeBuilder,
}

#[derive(Copy, Clone)]
enum Operand {
    I32(NewId<i32>),
    F32(NewId<f32>),
}

impl Operand {
    fn i32(self) -> NewId<i32> {
        match self {
            Operand::I32(value) => value,
            Operand::F32(_) => unreachable!("expected an i32 operand"),
        }
    }

    fn promote(self, builder: &mut TapeBuilder) -> NewId<f32> {
        match self {
            Operand::I32(value) => builder.f32_from_i32(value),
            Operand::F32(value) => value,
        }
    }
}

impl Visitor for Generator<'_> {
    type In<'a> = Operand;
    type Out = Operand;

    fn un_op(&mut self, id: ExprId, op: UnOp, operand: Self::In<'_>) -> Self::Out {
        match (op, self.ty[id]) {
            (UnOp::Plus, _) => operand,
            (UnOp::Minus, Type::I32) => {
                let zero = self.builder.i32_const(0);
                Operand::I32(self.builder.i32_sub(zero, operand.i32()))
            }
            (UnOp::Minus, Type::Bool) => unreachable!(),
            (UnOp::Minus, Type::F32) => {
                let operand = operand.promote(self.builder);
                Operand::F32(self.builder.f32_neg(operand))
            }
        }
    }

    fn bin_op(&mut self, id: ExprId, op: BinOp, lhs: Self::In<'_>, rhs: Self::In<'_>) -> Self::Out {
        match (op, self.ty[id]) {
            (BinOp::Add, Type::I32) => Operand::I32(self.builder.i32_add(lhs.i32(), rhs.i32())),
            (BinOp::Sub, Type::I32) => Operand::I32(self.builder.i32_sub(lhs.i32(), rhs.i32())),
            (BinOp::Mul, Type::I32) => Operand::I32(self.builder.i32_mul(lhs.i32(), rhs.i32())),

            (BinOp::Pow, Type::F32) => {
                let base = lhs.promote(self.builder);
                Operand::F32(match rhs {
                    Operand::I32(exp) => self.builder.f32_powi(base, exp),
                    Operand::F32(exp) => self.builder.f32_powf(base, exp),
                })
            }

            (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div, Type::F32) => {
                let (lhs, rhs) = (lhs.promote(self.builder), rhs.promote(self.builder));
                Operand::F32(match op {
                    BinOp::Add => self.builder.f32_add(lhs, rhs),
                    BinOp::Sub => self.builder.f32_sub(lhs, rhs),
                    BinOp::Mul => self.builder.f32_mul(lhs, rhs),
                    BinOp::Div => self.builder.f32_div(lhs, rhs),
                    BinOp::Pow => unreachable!(),
                })
            }

            (BinOp::Div | BinOp::Pow, Type::I32) | (_, Type::Bool) => unreachable!(),
        }
    }

    fn intrinsic<'a, I>(&mut self, _id: ExprId, kind: Intrinsic, mut args: I) -> Self::Out
    where
        I: ExactSizeIterator<Item = Self::In<'a>>,
    {
        let arg = args.next().unwrap().promote(self.builder);

        let value = match kind {
            Intrinsic::Exp => self.builder.f32_exp(arg),
            Intrinsic::Log => self.builder.f32_lg(arg),
            Intrinsic::Ln => self.builder.f32_ln(arg),
            Intrinsic::Sin => self.builder.f32_sin(arg),
            Intrinsic::Cos => self.builder.f32_cos(arg),
            Intrinsic::Tan => self.builder.f32_tan(arg),
            Intrinsic::Cot => self.builder.f32_cot(arg),
            Intrinsic::Abs => self.builder.f32_abs(arg),
            Intrinsic::Min => {
                let rhs = args.next().unwrap().promote(self.builder);
                self.builder.f32_min(arg, rhs)
            }
            Intrinsic::Max => {
                let rhs = args.next().unwrap().promote(self.builder);
                self.builder.f32_max(arg, rhs)
            }
        };

        Operand::F32(value)
    }

    fn var(&mut self, id: ExprId, _name: Symbol) -> Self::Out {
        match *self.vars.get(id).unwrap() {
            VarSlot::Const(index, ty) => {
                let value = self.consts[index.0 as usize].value();
                match ty {
                    Type::I32 => Operand::I32(self.builder.i32_const(value.as_i32())),
                    Type::Bool => unreachable!(),
                    Type::F32 => Operand::F32(self.builder.f32_const(value.as_f32())),
                }
            }
            VarSlot::Arg(index, ty) => match ty {
                Type::I32 => Operand::I32(self.builder.arg(index.0 as usize)),
                Type::Bool => unreachable!(),
                Type::F32 => Operand::F32(self.builder.arg(index.0 as usize)),
            },
        }
    }

    fn lit(&mut self, _id: ExprId, value: Value) -> Self::Out {
        match value {
            Value::I32(value) => Operand::I32(self.builder.i32_const(value)),
            Value::F32(value) => Operand::F32(self.builder.f32_const(value)),
        }
    }

    fn map_error(&mut self, _id: ExprId, _inner: Option<Self::In<'_>>) -> Self::Out {
        unreachable!()
    }
}

pub(crate) struct Bindings<'a> {
    args: Vec<(&'a str, Type)>,
    consts: &'a [(&'a str, TypedValue)],
}

impl<'a> Bindings<'a> {
    pub(crate) fn new<Args: Layout>(args: &[&'a str], consts: &'a [(&'a str, TypedValue)]) -> Self {
        assert_eq!(
            args.len(),
            Args::LEN,
            "argument names do not match the argument layout"
        );

        let all_vars = || {
            args.iter()
                .copied()
                .chain(consts.iter().map(|&(name, _)| name))
        };
        for (i, name) in all_vars().enumerate() {
            if all_vars().take(i).any(|prev| prev == name) {
                panic!("duplicate variable `{name}`");
            }
        }

        Self {
            args: args
                .iter()
                .copied()
                .zip(Args::types().map(lower_type))
                .collect(),
            consts,
        }
    }
}

pub(crate) fn lower_type(ty: ValueType) -> Type {
    match ty {
        ValueType::I32 => Type::I32,
        ValueType::F32 => Type::F32,
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct VarIndex(u32);

#[derive(Copy, Clone, Debug)]
enum VarSlot {
    Const(VarIndex, Type),
    Arg(VarIndex, Type),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagnostics(source: &str) -> Vec<(String, usize, usize)> {
        let (parsed, mut diagnostics) = parse(source);

        if let Err(error) = parsed.validate(&Bindings::new::<f32>(&["x"], &[])) {
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
