mod ast;
mod parser;
mod pretty;
mod symbol;
mod token;

use std::{collections::HashSet, error::Error, fmt, hash::Hash, marker::PhantomData};

use crate::{
    diag::Diagnostic,
    schema::{FieldType, FieldValue, Schema},
};
use isoplot_tape::{NewValueId, Tape, TapeBuilder, ValueType};

use self::{
    ast::{Ast, BinOp, DenseMap, ExprId, Intrinsic, SparseMap, UnOp, Value, Visitor},
    pretty::PrettyPrintAst,
    symbol::{Interner, Symbol},
};

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

    pub(crate) fn validate<Params: Schema>(
        &self,
        params: &ParamBindings<Params>,
    ) -> Result<Validated<'_>, ValidateError> {
        let mut diagnostics = Vec::new();

        let resolver = VarResolver {
            interner: &self.interner,
            params,
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
            arg_types: params
                .params
                .iter()
                .map(|&(_, ty)| field_type(ty))
                .collect(),
            vars,
            ty,
        })
    }
}

pub(crate) struct Validated<'a> {
    ast: &'a Ast,
    arg_types: Vec<ValueType>,
    vars: SparseMap<VarSlot>,
    ty: DenseMap<ValueType>,
}

impl Validated<'_> {
    pub(crate) fn lower_to_ir(&self) -> Tape {
        let mut builder = Tape::builder(self.arg_types.clone(), vec![ValueType::F32]);

        let generator = Generator {
            vars: &self.vars,
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

struct VarResolver<'a, Params> {
    interner: &'a Interner,
    params: &'a ParamBindings<'a, Params>,
    ast: &'a Ast,
    diags: &'a mut Vec<Diagnostic>,
}

impl<Params: Schema> Visitor for VarResolver<'_, Params> {
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

        if let Some((index, ty)) = self.params.get(name) {
            return Some(VarSlot::Param(index, ty));
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
    type In<'a> = &'a ValueType;
    type Out = ValueType;

    fn un_op(&mut self, _id: ExprId, _op: UnOp, operand: &ValueType) -> ValueType {
        *operand
    }

    fn bin_op(&mut self, _id: ExprId, op: BinOp, lhs: &ValueType, rhs: &ValueType) -> ValueType {
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul => match (lhs, rhs) {
                (ValueType::I32, ValueType::I32) => ValueType::I32,
                _ => ValueType::F32,
            },
            BinOp::Div | BinOp::Pow => ValueType::F32,
        }
    }

    fn intrinsic<'a, I>(&mut self, _id: ExprId, _kind: Intrinsic, _args: I) -> ValueType
    where
        I: ExactSizeIterator<Item = Self::In<'a>>,
    {
        ValueType::F32
    }

    fn var(&mut self, id: ExprId, _name: Symbol) -> ValueType {
        match self.vars.get(id) {
            Some(slot) => slot.ty(),
            None => ValueType::F32,
        }
    }

    fn lit(&mut self, _id: ExprId, value: Value) -> ValueType {
        match value {
            Value::I32(_) => ValueType::I32,
            Value::F32(_) => ValueType::F32,
        }
    }

    fn map_error(&mut self, _id: ExprId, inner: Option<&ValueType>) -> ValueType {
        inner.copied().unwrap_or(ValueType::F32)
    }
}

struct Generator<'a> {
    vars: &'a SparseMap<VarSlot>,
    ty: &'a DenseMap<ValueType>,
    builder: &'a mut TapeBuilder,
}

#[derive(Copy, Clone)]
enum Operand {
    I32(NewValueId<i32>),
    F32(NewValueId<f32>),
}

impl Operand {
    fn i32(self) -> NewValueId<i32> {
        match self {
            Operand::I32(value) => value,
            Operand::F32(_) => unreachable!("expected an i32 operand"),
        }
    }

    fn promote(self, builder: &mut TapeBuilder) -> NewValueId<f32> {
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
            (UnOp::Minus, ValueType::I32) => {
                let zero = self.builder.i32_const(0);
                Operand::I32(self.builder.i32_sub(zero, operand.i32()))
            }
            (UnOp::Minus, ValueType::Bool) => unreachable!(),
            (UnOp::Minus, ValueType::F32) => {
                let operand = operand.promote(self.builder);
                Operand::F32(self.builder.f32_neg(operand))
            }
        }
    }

    fn bin_op(&mut self, id: ExprId, op: BinOp, lhs: Self::In<'_>, rhs: Self::In<'_>) -> Self::Out {
        match (op, self.ty[id]) {
            (BinOp::Add, ValueType::I32) => {
                Operand::I32(self.builder.i32_add(lhs.i32(), rhs.i32()))
            }
            (BinOp::Sub, ValueType::I32) => {
                Operand::I32(self.builder.i32_sub(lhs.i32(), rhs.i32()))
            }
            (BinOp::Mul, ValueType::I32) => {
                Operand::I32(self.builder.i32_mul(lhs.i32(), rhs.i32()))
            }

            (BinOp::Pow, ValueType::F32) => {
                let base = lhs.promote(self.builder);
                Operand::F32(match rhs {
                    Operand::I32(exp) => self.builder.f32_powi(base, exp),
                    Operand::F32(exp) => self.builder.f32_powf(base, exp),
                })
            }

            (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div, ValueType::F32) => {
                let (lhs, rhs) = (lhs.promote(self.builder), rhs.promote(self.builder));
                Operand::F32(match op {
                    BinOp::Add => self.builder.f32_add(lhs, rhs),
                    BinOp::Sub => self.builder.f32_sub(lhs, rhs),
                    BinOp::Mul => self.builder.f32_mul(lhs, rhs),
                    BinOp::Div => self.builder.f32_div(lhs, rhs),
                    BinOp::Pow => unreachable!(),
                })
            }

            (BinOp::Div | BinOp::Pow, ValueType::I32) | (_, ValueType::Bool) => unreachable!(),
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
            VarSlot::Param(index, ty) => match ty {
                FieldType::I32 => Operand::I32(self.builder.param(index.0 as usize)),
                FieldType::F32 => Operand::F32(self.builder.param(index.0 as usize)),
            },
            VarSlot::Const(value) => match value.ty() {
                FieldType::I32 => Operand::I32(self.builder.i32_const(value.unwrap_i32())),
                FieldType::F32 => Operand::F32(self.builder.f32_const(value.unwrap_f32())),
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

pub(crate) struct ParamBindings<'a, S> {
    params: &'a [(&'a str, FieldType)],
    _marker: PhantomData<fn() -> S>,
}

impl<'a, S: Schema> ParamBindings<'a, S> {
    pub(crate) fn new(params: &'a [(&'a str, FieldType)]) -> Self {
        let param_ty = params.iter().map(|(_, ty)| *ty);
        assert!(
            param_ty.eq(S::fields().map(|field| field.ty)),
            "parameters do not match the schema fields"
        );

        if let Some((name, _)) = find_duplicate(params) {
            panic!("duplicate parameter `{name}`");
        }

        Self {
            params,
            _marker: PhantomData,
        }
    }

    fn get(&self, name: &str) -> Option<(ParamIndex, FieldType)> {
        self.params
            .iter()
            .enumerate()
            .find_map(|(i, (field_name, ty))| {
                (name == *field_name).then_some((ParamIndex(i as u32), *ty))
            })
    }
}

const fn field_type(ty: FieldType) -> ValueType {
    match ty {
        FieldType::I32 => ValueType::I32,
        FieldType::F32 => ValueType::F32,
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct ParamIndex(u32);

#[derive(Copy, Clone, Debug)]
enum VarSlot {
    Param(ParamIndex, FieldType),
    Const(FieldValue),
}

impl VarSlot {
    fn ty(self) -> ValueType {
        field_type(match self {
            VarSlot::Param(_, ty) => ty,
            VarSlot::Const(value) => value.ty(),
        })
    }
}

fn find_duplicate<T, I>(elems: I) -> Option<T>
where
    T: Copy + Eq + Hash,
    I: IntoIterator<Item = T>,
{
    let mut seen = HashSet::new();

    for elem in elems.into_iter() {
        if !seen.insert(elem) {
            return Some(elem);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagnostics(source: &str) -> Vec<(String, usize, usize)> {
        let (parsed, mut diagnostics) = parse(source);

        if let Err(error) = parsed.validate(&ParamBindings::<f32>::new(&[("x", FieldType::F32)])) {
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
