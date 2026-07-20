use crate::Value;
use crate::span::Span;
use std::ops::Index;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) struct ExprId(u32);

#[derive(Debug, Default)]
pub(crate) struct Ast {
    nodes: Vec<Expr>,
}

impl Ast {
    pub(crate) fn get(&self, id: ExprId) -> Option<&Expr> {
        self.nodes.get(id.0 as usize)
    }

    pub(crate) fn insert(&mut self, expr: Expr) -> ExprId {
        let len = self.nodes.len();
        assert!(len < u32::MAX as usize);
        self.nodes.push(expr);
        ExprId(len as u32)
    }
}

impl Index<ExprId> for Ast {
    type Output = Expr;

    fn index(&self, id: ExprId) -> &Self::Output {
        self.get(id).unwrap()
    }
}

#[derive(Debug)]
pub(crate) struct Expr {
    kind: ExprKind,
    span: Span,
}

impl Expr {
    pub(crate) fn new(kind: ExprKind, span: Span) -> Self {
        Self { kind, span }
    }
}

#[derive(Debug)]
pub(crate) enum ExprKind {
    Bin(BinOp, ExprId, ExprId),
    Lit(Lit),
    Error,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) enum BinOp {
    Add,
    Sub,
    Mul,
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Lit {
    pub(crate) value: Value,
}
