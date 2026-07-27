use crate::{Value, span::Span};
use std::{marker::PhantomData, ops::Index};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
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

    pub(crate) fn fold<F>(&self, mut folder: F) -> F::Acc
    where
        F: Folder,
    {
        let mut accs: Vec<Option<F::Acc>> = Vec::with_capacity(self.arena.nodes.len());

        for (index, node) in self.arena.nodes.iter().enumerate() {
            let id = ExprId(index as u32);

            let acc = match node.kind {
                ExprKind::UnOp(op, operand) => {
                    let operand = accs[operand.0 as usize].take().unwrap();
                    folder.fold_un_op(id, op, operand)
                }
                ExprKind::BinOp(op, lhs, rhs) => {
                    let lhs = accs[lhs.0 as usize].take().unwrap();
                    let rhs = accs[rhs.0 as usize].take().unwrap();
                    folder.fold_bin_op(id, op, lhs, rhs)
                }
                ExprKind::Lit(value) => folder.fold_lit(id, value),
                ExprKind::Error(inner) => {
                    let inner = inner.map(|inner| accs[inner.0 as usize].take().unwrap());
                    folder.fold_error(id, inner)
                }
            };

            accs.push(Some(acc));
        }

        accs[self.root.0 as usize].take().unwrap()
    }
}

impl Index<ExprId> for Ast {
    type Output = Expr;

    fn index(&self, index: ExprId) -> &Self::Output {
        self.arena.get(index)
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

pub(crate) trait Folder {
    /// The accumulator type
    type Acc;

    /// Folds a unary operator expression (e.g., `-A`).
    fn fold_un_op(&mut self, id: ExprId, op: UnOp, operand: Self::Acc) -> Self::Acc;

    /// Folds a binary operator expression (e.g., `A + B`).
    fn fold_bin_op(&mut self, id: ExprId, op: BinOp, lhs: Self::Acc, rhs: Self::Acc) -> Self::Acc;

    /// Folds a literal expression (e.g., `123.56`).
    fn fold_lit(&mut self, id: ExprId, value: Value) -> Self::Acc;

    /// Folds an error node.
    fn fold_error(&mut self, id: ExprId, inner: Option<Self::Acc>) -> Self::Acc;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::BytePos;

    struct Eval;

    impl Folder for Eval {
        type Acc = f32;

        fn fold_un_op(&mut self, _id: ExprId, op: UnOp, operand: f32) -> f32 {
            match op {
                UnOp::Plus => operand,
                UnOp::Minus => -operand,
            }
        }

        fn fold_bin_op(&mut self, _id: ExprId, op: BinOp, lhs: f32, rhs: f32) -> f32 {
            match op {
                BinOp::Add => lhs + rhs,
                BinOp::Sub => lhs - rhs,
                BinOp::Mul => lhs * rhs,
                BinOp::Div => lhs / rhs,
                BinOp::Pow => lhs.powf(rhs),
            }
        }

        fn fold_lit(&mut self, _id: ExprId, value: Value) -> f32 {
            match value {
                Value::F32(x) => x,
                Value::Unit => unreachable!(),
            }
        }

        fn fold_error(&mut self, _id: ExprId, _inner: Option<f32>) -> Self::Acc {
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

        assert_eq!(ast.fold(Eval), -7.0);
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
