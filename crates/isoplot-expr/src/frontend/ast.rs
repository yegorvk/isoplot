use std::{
    collections::HashMap,
    marker::PhantomData,
    ops::{Index, Range},
};

use super::{span::Span, symbol::Symbol};

#[derive(Copy, Clone, PartialEq, Debug)]
pub(super) enum Value {
    F32(f32),
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub(super) struct ExprId(u32);

#[derive(Copy, Clone, Debug)]
pub(super) struct ExprSetId {
    start: u32,
    end: u32,
}

impl ExprSetId {
    fn raw_range(self) -> Range<u32> {
        self.start..self.end
    }

    fn usize_range(self) -> Range<usize> {
        (self.start as usize)..(self.end as usize)
    }
}

#[derive(Debug, Default)]
struct Arena {
    nodes: Vec<Expr>,
}

impl Arena {
    fn get(&self, id: ExprId) -> &Expr {
        &self.nodes[id.0 as usize]
    }

    fn insert(&mut self, node: Expr) -> ExprId {
        let len = self.nodes.len();
        assert!(len < u32::MAX as usize);
        self.nodes.push(node);
        ExprId(len as u32)
    }
}

type Brand<'b> = PhantomData<fn(&'b ()) -> &'b ()>;

pub(super) struct AstBuilder<'b> {
    arena: Arena,
    isolated: u32,
    _brand: Brand<'b>,
}

impl<'b> AstBuilder<'b> {
    pub(super) fn un_op(&mut self, op_span: Span, op: UnOp, operand: NewId<'b>) -> NewId<'b> {
        let span = op_span.chain(operand.node.span);
        let operand = self.consume(operand);
        self.create(span, ExprKind::UnOp(op, operand))
    }

    pub(super) fn bin_op(&mut self, op: BinOp, lhs: NewId<'b>, rhs: NewId<'b>) -> NewId<'b> {
        let span = lhs.node.span.chain(rhs.node.span);
        let (lhs, rhs) = (self.consume(lhs), self.consume(rhs));
        self.create(span, ExprKind::BinOp(op, lhs, rhs))
    }

    pub(super) fn intrinsic(
        &mut self,
        mut span: Span,
        intrinsic: Intrinsic,
        args: impl IntoIterator<Item = NewId<'b>>,
    ) -> NewId<'b> {
        let start = self.arena.nodes.len() as u32;

        for arg in args {
            span = span.chain(arg.node.span);
            self.consume(arg);
        }

        let end = self.arena.nodes.len() as u32;
        self.create(
            span,
            ExprKind::Intrinsic(intrinsic, ExprSetId { start, end }),
        )
    }

    pub(super) fn lit(&mut self, span: Span, value: Value) -> NewId<'b> {
        self.create(span, ExprKind::Lit(value))
    }

    pub(super) fn var(&mut self, span: Span, name: Symbol) -> NewId<'b> {
        self.create(span, ExprKind::Var(name))
    }

    pub(super) fn error(&mut self, span: Span, inner: Option<NewId<'b>>) -> NewId<'b> {
        match inner {
            Some(inner) => {
                let span = inner.node.span.chain(span);
                let inner = self.consume(inner);
                self.create(span, ExprKind::Error(Some(inner)))
            }
            None => self.create(span, ExprKind::Error(None)),
        }
    }

    fn consume(&mut self, id: NewId<'b>) -> ExprId {
        self.isolated -= 1;
        self.arena.insert(id.node)
    }

    fn create(&mut self, span: Span, kind: ExprKind) -> NewId<'b> {
        self.isolated += 1;
        NewId {
            node: Expr { kind, span },
            _brand: PhantomData,
        }
    }
}

#[must_use]
pub(super) struct NewId<'b> {
    node: Expr,
    _brand: Brand<'b>,
}

pub(super) struct Ast {
    arena: Arena,
}

impl Ast {
    pub(super) fn build<F>(f: F) -> Self
    where
        F: for<'b> FnOnce(&mut AstBuilder<'b>) -> NewId<'b>,
    {
        let mut builder = AstBuilder {
            arena: Arena::default(),
            isolated: 0,
            _brand: PhantomData,
        };

        let root = f(&mut builder);

        if builder.isolated != 1 {
            panic!("`AstBuilder` contains orphaned nodes")
        }

        builder.consume(root);

        Ast {
            arena: builder.arena,
        }
    }

    pub(super) fn fold<F, Acc>(&self, mut folder: F) -> Acc
    where
        F: for<'a> Visitor<In<'a> = Acc, Out = Acc>,
    {
        let mut accs: Vec<Option<Acc>> = Vec::with_capacity(self.arena.nodes.len());

        for (index, node) in self.arena.nodes.iter().enumerate() {
            let id = ExprId(index as u32);

            let acc = match node.kind {
                ExprKind::UnOp(op, operand) => {
                    let operand = accs[operand.0 as usize].take().unwrap();
                    folder.un_op(id, op, operand)
                }
                ExprKind::BinOp(op, lhs, rhs) => {
                    let lhs = accs[lhs.0 as usize].take().unwrap();
                    let rhs = accs[rhs.0 as usize].take().unwrap();
                    folder.bin_op(id, op, lhs, rhs)
                }
                ExprKind::Intrinsic(intrinsic, args) => {
                    let args = accs[args.usize_range()]
                        .iter_mut()
                        .map(|arg| arg.take().unwrap());

                    folder.intrinsic(id, intrinsic, args)
                }
                ExprKind::Var(name) => folder.var(id, name),
                ExprKind::Lit(value) => folder.lit(id, value),
                ExprKind::Error(inner) => {
                    let inner = inner.map(|inner| accs[inner.0 as usize].take().unwrap());
                    folder.map_error(id, inner)
                }
            };

            accs.push(Some(acc));
        }

        accs.pop().unwrap().unwrap()
    }

    #[cfg(test)]
    fn root(&self) -> ExprId {
        ExprId(self.arena.nodes.len() as u32 - 1)
    }
}

impl Index<ExprId> for Ast {
    type Output = Expr;

    fn index(&self, index: ExprId) -> &Self::Output {
        self.arena.get(index)
    }
}

pub(super) struct DenseMap<T> {
    map: Vec<T>,
}

impl<T> DenseMap<T> {
    pub(super) fn build<B>(ast: &Ast, mut builder: B) -> Self
    where
        B: for<'a> Visitor<In<'a> = &'a T, Out = T>,
    {
        let mut map: Vec<T> = Vec::with_capacity(ast.arena.nodes.len());

        for (index, node) in ast.arena.nodes.iter().enumerate() {
            let id = ExprId(index as u32);

            let out = match node.kind {
                ExprKind::UnOp(op, operand) => builder.un_op(id, op, &map[operand.0 as usize]),
                ExprKind::BinOp(op, lhs, rhs) => {
                    builder.bin_op(id, op, &map[lhs.0 as usize], &map[rhs.0 as usize])
                }
                ExprKind::Intrinsic(intrinsic, args) => {
                    let args = map[args.usize_range()].iter();
                    builder.intrinsic(id, intrinsic, args)
                }
                ExprKind::Var(name) => builder.var(id, name),
                ExprKind::Lit(value) => builder.lit(id, value),
                ExprKind::Error(inner) => {
                    let inner = inner.map(|inner| &map[inner.0 as usize]);
                    builder.map_error(id, inner)
                }
            };

            map.push(out);
        }

        Self { map }
    }
}

impl<T> Index<ExprId> for DenseMap<T> {
    type Output = T;

    fn index(&self, index: ExprId) -> &Self::Output {
        &self.map[index.0 as usize]
    }
}

pub(super) struct SparseMap<T> {
    map: HashMap<ExprId, T>,
}

impl<T> SparseMap<T> {
    pub(super) fn build<B>(ast: &Ast, mut builder: B) -> Self
    where
        B: for<'a> Visitor<In<'a> = Option<&'a T>, Out = Option<T>>,
    {
        let mut map: HashMap<ExprId, T> = HashMap::new();

        for (index, node) in ast.arena.nodes.iter().enumerate() {
            let id = ExprId(index as u32);

            let out = match node.kind {
                ExprKind::UnOp(op, operand) => builder.un_op(id, op, map.get(&operand)),
                ExprKind::BinOp(op, lhs, rhs) => {
                    builder.bin_op(id, op, map.get(&lhs), map.get(&rhs))
                }
                ExprKind::Intrinsic(intrinsic, args) => {
                    let args = args.raw_range().map(|raw_id| map.get(&ExprId(raw_id)));
                    builder.intrinsic(id, intrinsic, args)
                }
                ExprKind::Var(name) => builder.var(id, name),
                ExprKind::Lit(value) => builder.lit(id, value),
                ExprKind::Error(inner) => {
                    let inner = inner.map(|inner| map.get(&inner));
                    builder.map_error(id, inner)
                }
            };

            if let Some(out) = out {
                map.insert(id, out);
            }
        }

        Self { map }
    }

    pub(super) fn get(&self, id: ExprId) -> Option<&T> {
        self.map.get(&id)
    }
}

#[derive(Copy, Clone, Debug)]
pub(super) struct Expr {
    pub(super) kind: ExprKind,
    pub(super) span: Span,
}

#[derive(Copy, Clone, Debug)]
pub(super) enum ExprKind {
    UnOp(UnOp, ExprId),
    BinOp(BinOp, ExprId, ExprId),
    Intrinsic(Intrinsic, ExprSetId),
    Var(Symbol),
    Lit(Value),
    Error(Option<ExprId>),
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(super) enum Intrinsic {
    Exp,
    Log,
    Ln,
    Sin,
    Cos,
    Tan,
    Cot,
}

impl Intrinsic {
    pub(super) fn num_args(self) -> usize {
        match self {
            Intrinsic::Exp
            | Intrinsic::Log
            | Intrinsic::Ln
            | Intrinsic::Sin
            | Intrinsic::Cos
            | Intrinsic::Tan
            | Intrinsic::Cot => 1,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(super) enum UnOp {
    Plus,
    Minus,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(super) enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

pub(super) trait Visitor {
    /// The input type
    type In<'a>;

    /// The output type
    type Out;

    /// Transforms a unary operator expression (e.g., `-A`).
    fn un_op(&mut self, id: ExprId, op: UnOp, operand: Self::In<'_>) -> Self::Out;

    /// Transforms a binary operator expression (e.g., `A + B`).
    fn bin_op(&mut self, id: ExprId, op: BinOp, lhs: Self::In<'_>, rhs: Self::In<'_>) -> Self::Out;

    fn intrinsic<'a, I>(&mut self, id: ExprId, kind: Intrinsic, args: I) -> Self::Out
    where
        I: ExactSizeIterator<Item = Self::In<'a>>;

    /// Transforms a variable expression (e.g., `x`).
    fn var(&mut self, id: ExprId, name: Symbol) -> Self::Out;

    /// Transforms a literal expression (e.g., `123.56`).
    fn lit(&mut self, id: ExprId, value: Value) -> Self::Out;

    /// Transforms an error node.
    fn map_error(&mut self, id: ExprId, inner: Option<Self::In<'_>>) -> Self::Out;
}

#[cfg(test)]
mod tests {
    use super::super::{span::BytePos, symbol::Interner};
    use super::*;
    use std::collections::HashMap;

    #[derive(Default)]
    struct Eval {
        env: HashMap<Symbol, f32>,
    }

    impl Visitor for Eval {
        type In<'a> = f32;
        type Out = f32;

        fn un_op(&mut self, _id: ExprId, op: UnOp, operand: f32) -> f32 {
            match op {
                UnOp::Plus => operand,
                UnOp::Minus => -operand,
            }
        }

        fn bin_op(&mut self, _id: ExprId, op: BinOp, lhs: f32, rhs: f32) -> f32 {
            match op {
                BinOp::Add => lhs + rhs,
                BinOp::Sub => lhs - rhs,
                BinOp::Mul => lhs * rhs,
                BinOp::Div => lhs / rhs,
                BinOp::Pow => lhs.powf(rhs),
            }
        }

        fn intrinsic<'a, I>(&mut self, _id: ExprId, intrinsic: Intrinsic, mut args: I) -> f32
        where
            I: ExactSizeIterator<Item = Self::In<'a>>,
        {
            let arg = args.next().unwrap();
            assert!(args.next().is_none());

            match intrinsic {
                Intrinsic::Exp => arg.exp(),
                Intrinsic::Log => arg.log10(),
                Intrinsic::Ln => arg.ln(),
                Intrinsic::Sin => arg.sin(),
                Intrinsic::Cos => arg.cos(),
                Intrinsic::Tan => arg.tan(),
                Intrinsic::Cot => arg.tan().recip(),
            }
        }

        fn var(&mut self, _id: ExprId, name: Symbol) -> f32 {
            self.env[&name]
        }

        fn lit(&mut self, _id: ExprId, value: Value) -> f32 {
            match value {
                Value::F32(x) => x,
            }
        }

        fn map_error(&mut self, _id: ExprId, _inner: Option<f32>) -> f32 {
            unreachable!()
        }
    }

    #[test]
    fn fold_eval() {
        let span = Span::new(BytePos(0), BytePos(1));

        // -(1 + 2 * 3)
        let ast = Ast::build(|b| {
            let one = b.lit(span, Value::F32(1.0));
            let two = b.lit(span, Value::F32(2.0));
            let three = b.lit(span, Value::F32(3.0));
            let mul = b.bin_op(BinOp::Mul, two, three);
            let sum = b.bin_op(BinOp::Add, one, mul);
            b.un_op(span, UnOp::Minus, sum)
        });

        assert_eq!(ast.fold(Eval::default()), -7.0);
    }

    #[test]
    fn fold_eval_vars() {
        let span = Span::new(BytePos(0), BytePos(1));

        let mut interner = Interner::default();
        let x = interner.get_or_insert("x");
        let y = interner.get_or_insert("y");

        // x ^ 2 + y ^ 2
        let ast = Ast::build(|b| {
            let x = b.var(span, x);
            let two = b.lit(span, Value::F32(2.0));
            let x2 = b.bin_op(BinOp::Pow, x, two);
            let y = b.var(span, y);
            let two = b.lit(span, Value::F32(2.0));
            let y2 = b.bin_op(BinOp::Pow, y, two);
            b.bin_op(BinOp::Add, x2, y2)
        });

        let env = HashMap::from([(x, 3.0), (y, 4.0)]);
        assert_eq!(ast.fold(Eval { env }), 25.0);
    }

    struct Depth;

    impl Visitor for Depth {
        type In<'a> = &'a u32;
        type Out = u32;

        fn un_op(&mut self, _id: ExprId, _op: UnOp, operand: &u32) -> u32 {
            operand + 1
        }

        fn bin_op(&mut self, _id: ExprId, _op: BinOp, lhs: &u32, rhs: &u32) -> u32 {
            lhs.max(rhs) + 1
        }

        fn intrinsic<'a, I>(&mut self, _id: ExprId, _intrinsic: Intrinsic, args: I) -> u32
        where
            I: ExactSizeIterator<Item = &'a u32>,
        {
            args.copied().max().unwrap_or(0) + 1
        }

        fn var(&mut self, _id: ExprId, _name: Symbol) -> u32 {
            1
        }

        fn lit(&mut self, _id: ExprId, _value: Value) -> u32 {
            1
        }

        fn map_error(&mut self, _id: ExprId, inner: Option<&u32>) -> u32 {
            inner.map_or(1, |inner| inner + 1)
        }
    }

    #[test]
    fn transform_depth() {
        let span = Span::new(BytePos(0), BytePos(1));

        // -(1 + 2 * 3)
        let ast = Ast::build(|b| {
            let one = b.lit(span, Value::F32(1.0));
            let two = b.lit(span, Value::F32(2.0));
            let three = b.lit(span, Value::F32(3.0));
            let mul = b.bin_op(BinOp::Mul, two, three);
            let sum = b.bin_op(BinOp::Add, one, mul);
            b.un_op(span, UnOp::Minus, sum)
        });

        let depths = DenseMap::build(&ast, Depth);
        assert_eq!(depths[ast.root()], 4);
    }

    #[test]
    fn intrinsic_arg_set() {
        let span = Span::new(BytePos(0), BytePos(1));

        // sin(1, 2 * 3) — arity is not enforced at the arena level
        let ast = Ast::build(|b| {
            let one = b.lit(span, Value::F32(1.0));
            let two = b.lit(span, Value::F32(2.0));
            let three = b.lit(span, Value::F32(3.0));
            let mul = b.bin_op(BinOp::Mul, two, three);
            b.intrinsic(span, Intrinsic::Sin, [one, mul])
        });

        let depths = DenseMap::build(&ast, Depth);
        assert_eq!(depths[ast.root()], 3);
    }

    struct Consts;

    impl Visitor for Consts {
        type In<'a> = Option<&'a f32>;
        type Out = Option<f32>;

        fn un_op(&mut self, _id: ExprId, op: UnOp, operand: Option<&f32>) -> Option<f32> {
            let operand = *operand?;
            Some(match op {
                UnOp::Plus => operand,
                UnOp::Minus => -operand,
            })
        }

        fn bin_op(
            &mut self,
            _id: ExprId,
            op: BinOp,
            lhs: Option<&f32>,
            rhs: Option<&f32>,
        ) -> Option<f32> {
            let (&lhs, &rhs) = lhs.zip(rhs)?;
            Some(match op {
                BinOp::Add => lhs + rhs,
                BinOp::Sub => lhs - rhs,
                BinOp::Mul => lhs * rhs,
                BinOp::Div => lhs / rhs,
                BinOp::Pow => lhs.powf(rhs),
            })
        }

        fn intrinsic<'a, I>(&mut self, _id: ExprId, _intrinsic: Intrinsic, _args: I) -> Option<f32>
        where
            I: ExactSizeIterator<Item = Option<&'a f32>>,
        {
            None
        }

        fn var(&mut self, _id: ExprId, _name: Symbol) -> Option<f32> {
            None
        }

        fn lit(&mut self, _id: ExprId, value: Value) -> Option<f32> {
            match value {
                Value::F32(x) => Some(x),
            }
        }

        fn map_error(&mut self, _id: ExprId, _inner: Option<Option<&f32>>) -> Option<f32> {
            unreachable!()
        }
    }

    #[test]
    fn sparse_consts() {
        let span = Span::new(BytePos(0), BytePos(1));

        let mut interner = Interner::default();
        let x = interner.get_or_insert("x");

        // x + 2 * 3
        let ast = Ast::build(|b| {
            let x = b.var(span, x);
            let two = b.lit(span, Value::F32(2.0));
            let three = b.lit(span, Value::F32(3.0));
            let mul = b.bin_op(BinOp::Mul, two, three);
            b.bin_op(BinOp::Add, x, mul)
        });

        let consts = SparseMap::build(&ast, Consts);

        // Insertion order: 0 = 2, 1 = 3, 2 = x, 3 = 2 * 3, 4 = root.
        assert_eq!(consts.get(ExprId(0)), Some(&2.0));
        assert_eq!(consts.get(ExprId(2)), None);
        assert_eq!(consts.get(ExprId(3)), Some(&6.0));
        assert_eq!(consts.get(ast.root()), None);
    }

    #[test]
    #[should_panic]
    fn orphans_detected() {
        let span = Span::new(BytePos(0), BytePos(1));

        Ast::build(|b| {
            let _orphan = b.lit(span, Value::F32(1.0));
            b.lit(span, Value::F32(2.0))
        });
    }
}
