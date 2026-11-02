use std::fmt::{self, Display};

use super::{
    ast::{Ast, BinOp, ExprId, Intrinsic, UnOp, Value, Visitor},
    symbol::{Interner, Symbol},
};

#[derive(Copy, Clone)]
pub(super) struct PrettyPrintAst<'a> {
    syms: &'a Interner,
    ast: &'a Ast,
}

impl<'a> PrettyPrintAst<'a> {
    pub(super) fn new(ast: &'a Ast, syms: &'a Interner) -> Self {
        Self { ast, syms }
    }
}

impl Display for PrettyPrintAst<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let render = Render {
            syms: self.syms,
            ast: self.ast,
        };

        f.write_str(&self.ast.fold(render))
    }
}

struct Render<'a> {
    syms: &'a Interner,
    ast: &'a Ast,
}

impl Render<'_> {
    fn node(&self, id: ExprId, label: &str, children: &[String]) -> String {
        let span = self.ast[id].span;
        let mut out = format!("{label} @ {}..{}", span.start.index(), span.end.index());

        for (i, child) in children.iter().enumerate() {
            let (head, tail) = if i + 1 == children.len() {
                ("\u{2514}\u{2500} ", "   ")
            } else {
                ("\u{251c}\u{2500} ", "\u{2502}  ")
            };

            for (j, line) in child.lines().enumerate() {
                out.push('\n');
                out.push_str(if j == 0 { head } else { tail });
                out.push_str(line);
            }
        }

        out
    }
}

impl Visitor for Render<'_> {
    type In<'a> = String;
    type Out = String;

    fn un_op(&mut self, id: ExprId, op: UnOp, operand: String) -> String {
        let op = match op {
            UnOp::Plus => "+",
            UnOp::Minus => "-",
        };
        self.node(id, &format!("UnOp {op}"), &[operand])
    }

    fn bin_op(&mut self, id: ExprId, op: BinOp, lhs: String, rhs: String) -> String {
        let op = match op {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Pow => "^",
        };
        self.node(id, &format!("BinOp {op}"), &[lhs, rhs])
    }

    fn intrinsic<'a, I>(&mut self, id: ExprId, kind: Intrinsic, args: I) -> String
    where
        I: ExactSizeIterator<Item = Self::In<'a>>,
    {
        let args: Vec<String> = args.collect();
        self.node(id, &format!("Intrinsic {}", kind.name()), &args)
    }

    fn var(&mut self, id: ExprId, name: Symbol) -> String {
        let name = self.syms.resolve(name).unwrap();
        self.node(id, &format!("Var {name}"), &[])
    }

    fn lit(&mut self, id: ExprId, value: Value) -> String {
        let label = match value {
            Value::I32(value) => format!("I32 {value}"),
            Value::F32(value) => format!("F32 {value}"),
        };
        self.node(id, &label, &[])
    }

    fn map_error(&mut self, id: ExprId, inner: Option<String>) -> String {
        match inner {
            Some(inner) => self.node(id, "Error", &[inner]),
            None => self.node(id, "Error", &[]),
        }
    }
}
