use crate::{Value, span::Span, symbol::Symbol};
use std::{collections::HashMap, marker::PhantomData, ops::Index};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub(crate) struct ExprId(u32);

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

type Invariant<'b> = PhantomData<fn(&'b ()) -> &'b ()>;

pub(crate) struct AstBuilder<'b> {
    arena: Arena,
    isolated: u32,
    _brand: Invariant<'b>,
}

impl<'b> AstBuilder<'b> {
    pub(crate) fn un_op(&mut self, op_span: Span, op: UnOp, operand: NewId<'b>) -> NewId<'b> {
        let operand = self.consume(operand);
        let span = op_span.chain(self.span_of(operand));
        self.insert(span, ExprKind::UnOp(op, operand))
    }

    pub(crate) fn bin_op(&mut self, op: BinOp, lhs: NewId<'b>, rhs: NewId<'b>) -> NewId<'b> {
        let (lhs, rhs) = (self.consume(lhs), self.consume(rhs));
        let span = self.span_of(lhs).chain(self.span_of(rhs));
        self.insert(span, ExprKind::BinOp(op, lhs, rhs))
    }

    pub(crate) fn lit(&mut self, span: Span, value: Value) -> NewId<'b> {
        self.insert(span, ExprKind::Lit(value))
    }

    pub(crate) fn var(&mut self, span: Span, name: Symbol) -> NewId<'b> {
        self.insert(span, ExprKind::Var(name))
    }

    pub(crate) fn error(&mut self, span: Span, inner: Option<NewId<'b>>) -> NewId<'b> {
        match inner {
            Some(inner) => {
                let inner = self.consume(inner);
                let span = self.span_of(inner).chain(span);
                self.insert(span, ExprKind::Error(Some(inner)))
            }
            None => self.insert(span, ExprKind::Error(None)),
        }
    }

    fn span_of(&self, id: ExprId) -> Span {
        self.arena.get(id).span
    }

    fn consume(&mut self, id: NewId<'b>) -> ExprId {
        self.isolated -= 1;
        id.id
    }

    fn insert(&mut self, span: Span, kind: ExprKind) -> NewId<'b> {
        let id = self.arena.insert(Expr { kind, span });
        self.isolated += 1;
        NewId {
            id,
            _brand: PhantomData,
        }
    }
}

#[must_use]
pub(crate) struct NewId<'b> {
    id: ExprId,
    _brand: Invariant<'b>,
}

pub(crate) struct Ast {
    arena: Arena,
    root: ExprId,
}

impl Ast {
    pub(crate) fn build<F>(f: F) -> Self
    where
        F: for<'b> FnOnce(&mut AstBuilder<'b>) -> NewId<'b>,
    {
        let mut builder = AstBuilder {
            arena: Arena::default(),
            isolated: 0,
            _brand: PhantomData,
        };

        let root = f(&mut builder).id;

        if builder.isolated != 1 {
            panic!("`AstBuilder` contains orphaned nodes")
        }

        Ast {
            root,
            arena: builder.arena,
        }
    }

    pub(crate) fn fold<F, Acc>(&self, mut folder: F) -> Acc
    where
        F: for<'a> Transformer<In<'a> = Acc, Out = Acc>,
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
                ExprKind::Var(name) => folder.var(id, name),
                ExprKind::Lit(value) => folder.lit(id, value),
                ExprKind::Error(inner) => {
                    let inner = inner.map(|inner| accs[inner.0 as usize].take().unwrap());
                    folder.map_error(id, inner)
                }
            };

            accs.push(Some(acc));
        }

        accs[self.root.0 as usize].take().unwrap()
    }

    pub(crate) fn len(&self) -> usize {
        self.arena.nodes.len()
    }
}

impl Index<ExprId> for Ast {
    type Output = Expr;

    fn index(&self, index: ExprId) -> &Self::Output {
        self.arena.get(index)
    }
}

pub(crate) struct DenseMap<T> {
    map: Vec<T>,
}

impl<T> DenseMap<T> {
    pub(crate) fn build<B>(ast: &Ast, mut builder: B) -> Self
    where
        B: for<'a> Transformer<In<'a> = &'a T, Out = T>,
    {
        let mut map: Vec<T> = Vec::with_capacity(ast.arena.nodes.len());

        for (index, node) in ast.arena.nodes.iter().enumerate() {
            let id = ExprId(index as u32);

            let out = match node.kind {
                ExprKind::UnOp(op, operand) => builder.un_op(id, op, &map[operand.0 as usize]),
                ExprKind::BinOp(op, lhs, rhs) => {
                    builder.bin_op(id, op, &map[lhs.0 as usize], &map[rhs.0 as usize])
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

pub(crate) struct SparseMap<T> {
    map: HashMap<ExprId, T>,
}

impl<T> SparseMap<T> {
    pub(crate) fn build<B>(ast: &Ast, mut builder: B) -> Self
    where
        B: for<'a> Transformer<In<'a> = Option<&'a T>, Out = Option<T>>,
    {
        let mut map: HashMap<ExprId, T> = HashMap::new();

        for (index, node) in ast.arena.nodes.iter().enumerate() {
            let id = ExprId(index as u32);

            let out = match node.kind {
                ExprKind::UnOp(op, operand) => builder.un_op(id, op, map.get(&operand)),
                ExprKind::BinOp(op, lhs, rhs) => {
                    builder.bin_op(id, op, map.get(&lhs), map.get(&rhs))
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

    pub(crate) fn get(&self, id: ExprId) -> Option<&T> {
        self.map.get(&id)
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Expr {
    pub(crate) kind: ExprKind,
    pub(crate) span: Span,
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum ExprKind {
    UnOp(UnOp, ExprId),
    BinOp(BinOp, ExprId, ExprId),
    Var(Symbol),
    Lit(Value),
    Error(Option<ExprId>),
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) enum UnOp {
    Plus,
    Minus,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

pub(crate) trait Transformer {
    /// The input type
    type In<'a>;

    /// The output type
    type Out;

    /// Transforms a unary operator expression (e.g., `-A`).
    fn un_op(&mut self, id: ExprId, op: UnOp, operand: Self::In<'_>) -> Self::Out;

    /// Transforms a binary operator expression (e.g., `A + B`).
    fn bin_op(&mut self, id: ExprId, op: BinOp, lhs: Self::In<'_>, rhs: Self::In<'_>) -> Self::Out;

    /// Transforms a variable expression (e.g., `x`).
    fn var(&mut self, id: ExprId, name: Symbol) -> Self::Out;

    /// Transforms a literal expression (e.g., `123.56`).
    fn lit(&mut self, id: ExprId, value: Value) -> Self::Out;

    /// Transforms an error node.
    fn map_error(&mut self, id: ExprId, inner: Option<Self::In<'_>>) -> Self::Out;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{span::BytePos, symbol::Interner};
    use std::collections::HashMap;

    #[derive(Default)]
    struct Eval {
        env: HashMap<Symbol, f32>,
    }

    impl Transformer for Eval {
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

        fn var(&mut self, _id: ExprId, name: Symbol) -> f32 {
            self.env[&name]
        }

        fn lit(&mut self, _id: ExprId, value: Value) -> f32 {
            match value {
                Value::F32(x) => x,
                Value::Unit => unreachable!(),
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

    impl Transformer for Depth {
        type In<'a> = &'a u32;
        type Out = u32;

        fn un_op(&mut self, _id: ExprId, _op: UnOp, operand: &u32) -> u32 {
            operand + 1
        }

        fn bin_op(&mut self, _id: ExprId, _op: BinOp, lhs: &u32, rhs: &u32) -> u32 {
            lhs.max(rhs) + 1
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
        assert_eq!(depths[ast.root], 4);
    }

    struct Consts;

    impl Transformer for Consts {
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

        fn var(&mut self, _id: ExprId, _name: Symbol) -> Option<f32> {
            None
        }

        fn lit(&mut self, _id: ExprId, value: Value) -> Option<f32> {
            match value {
                Value::F32(x) => Some(x),
                Value::Unit => unreachable!(),
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

        // Post-order: 0 = x, 1 = 2, 2 = 3, 3 = 2 * 3, 4 = root.
        assert_eq!(consts.get(ExprId(0)), None);
        assert_eq!(consts.get(ExprId(1)), Some(&2.0));
        assert_eq!(consts.get(ExprId(3)), Some(&6.0));
        assert_eq!(consts.get(ast.root), None);
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
