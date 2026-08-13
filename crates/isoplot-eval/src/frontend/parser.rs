use std::{f32::consts::PI, mem};

use crate::diag::{BytePos, Diagnostic, Span};

use super::{
    ast::{Ast, AstBuilder, BinOp, Intrinsic, NewId, UnOp, Value},
    symbol::Interner,
    token::{Token, TokenKind},
};

pub(super) fn parse<'src, I>(
    mut input: I,
    interner: &mut Interner,
    diags: &mut Vec<Diagnostic>,
) -> Ast
where
    I: Iterator<Item = Token<'src>>,
{
    Ast::build(|builder| {
        let next = input.next().unwrap_or(Token {
            kind: TokenKind::Eof,
            span: Span::empty(BytePos(0)),
        });

        let mut parser = Parser {
            input,
            next,
            builder,
            interner,
            diags,
        };

        let root = parser.parse_expr(0);
        parser.expect(TokenKind::Eof, "expected end of input", root)
    })
}

struct Parser<'src, 'a, 'b, I> {
    input: I,
    next: Token<'src>,
    builder: &'a mut AstBuilder<'b>,
    interner: &'a mut Interner,
    diags: &'a mut Vec<Diagnostic>,
}

impl<'src, 'a, 'b, I> Parser<'src, 'a, 'b, I>
where
    I: Iterator<Item = Token<'src>>,
{
    fn parse_expr(&mut self, min_bp: u8) -> NewId<'b> {
        let mut lhs = self.parse_prefix();

        while let Some(op) = infix_op(self.next.kind) {
            let (lbp, rbp) = binop_bp(op);
            if lbp < min_bp {
                break;
            }
            self.advance();

            let rhs = self.parse_expr(rbp);
            lhs = self.builder.bin_op(op, lhs, rhs);
        }

        lhs
    }

    fn parse_prefix(&mut self) -> NewId<'b> {
        let token = self.advance();

        match token.kind {
            TokenKind::Int(lit) => match lit.parse_i32() {
                Ok(value) => self.builder.lit(token.span, Value::I32(value)),
                Err(_) => self.error(token.span, "integer literal is out of range", None),
            },
            TokenKind::Float(lit) => match lit.parse_f32() {
                Ok(value) => self.builder.lit(token.span, Value::F32(value)),
                Err(_) => self.error(token.span, "float literal is out of range", None),
            },
            TokenKind::Pi => self.builder.lit(token.span, Value::F32(PI)),
            TokenKind::Ident(name) => {
                let name = self.interner.get_or_insert(name);
                self.builder.var(token.span, name)
            }
            TokenKind::LParen => {
                let inner = self.parse_expr(0);
                self.expect(TokenKind::RParen, "expected `)`", inner)
            }
            kind => {
                if let Some(intrinsic) = intrinsic_op(kind) {
                    self.parse_call(token.span, intrinsic)
                } else if let Some(op) = prefix_op(kind) {
                    let operand = self.parse_expr(unop_bp(op));
                    self.builder.un_op(token.span, op, operand)
                } else {
                    let message = format!("expected an expression, but found {kind}");
                    self.error(token.span, &message, None)
                }
            }
        }
    }

    fn parse_call(&mut self, span: Span, intrinsic: Intrinsic) -> NewId<'b> {
        if self.next.kind != TokenKind::LParen {
            let message = format!("expected `(`, but found {}", self.next.kind);
            return self.error(self.next.span, &message, None);
        }
        self.advance();

        let mut args = vec![self.parse_expr(0)];

        while self.next.kind == TokenKind::Comma {
            self.advance();
            args.push(self.parse_expr(0));
        }

        let call = self.builder.intrinsic(span, intrinsic, args);
        self.expect(TokenKind::RParen, "expected `)`", call)
    }

    fn expect(&mut self, kind: TokenKind<'src>, message: &str, expr: NewId<'b>) -> NewId<'b> {
        if self.next.kind == kind {
            self.advance();
            expr
        } else {
            let message = format!("{message}, but found {}", self.next.kind);
            self.error(self.next.span, &message, Some(expr))
        }
    }

    fn error(&mut self, span: Span, message: &str, inner: Option<NewId<'b>>) -> NewId<'b> {
        self.diags.push(Diagnostic::new(message, span));
        self.builder.error(span, inner)
    }

    fn advance(&mut self) -> Token<'src> {
        let next = self.input.next().unwrap_or(Token {
            kind: TokenKind::Eof,
            span: self.next.span,
        });

        mem::replace(&mut self.next, next)
    }
}

fn infix_op(kind: TokenKind<'_>) -> Option<BinOp> {
    match kind {
        TokenKind::Plus => Some(BinOp::Add),
        TokenKind::Minus => Some(BinOp::Sub),
        TokenKind::Star => Some(BinOp::Mul),
        TokenKind::Slash => Some(BinOp::Div),
        TokenKind::Caret => Some(BinOp::Pow),
        _ => None,
    }
}

fn intrinsic_op(kind: TokenKind<'_>) -> Option<Intrinsic> {
    match kind {
        TokenKind::Exp => Some(Intrinsic::Exp),
        TokenKind::Log => Some(Intrinsic::Log),
        TokenKind::Ln => Some(Intrinsic::Ln),
        TokenKind::Sin => Some(Intrinsic::Sin),
        TokenKind::Cos => Some(Intrinsic::Cos),
        TokenKind::Tan => Some(Intrinsic::Tan),
        TokenKind::Cot => Some(Intrinsic::Cot),
        TokenKind::Abs => Some(Intrinsic::Abs),
        TokenKind::Min => Some(Intrinsic::Min),
        TokenKind::Max => Some(Intrinsic::Max),
        _ => None,
    }
}

fn prefix_op(kind: TokenKind<'_>) -> Option<UnOp> {
    match kind {
        TokenKind::Plus => Some(UnOp::Plus),
        TokenKind::Minus => Some(UnOp::Minus),
        _ => None,
    }
}

fn binop_bp(op: BinOp) -> (u8, u8) {
    match op {
        BinOp::Add => (1, 2),
        BinOp::Sub => (1, 2),
        BinOp::Mul => (3, 4),
        BinOp::Div => (3, 4),
        BinOp::Pow => (6, 5),
    }
}

fn unop_bp(op: UnOp) -> u8 {
    match op {
        UnOp::Plus => 5,
        UnOp::Minus => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        ast::{ExprId, Visitor},
        symbol::{Interner, Symbol},
        token::tokenize,
    };
    use super::*;

    #[derive(Default)]
    struct SExpr {
        interner: Interner,
    }

    impl Visitor for SExpr {
        type In<'a> = String;
        type Out = String;

        fn un_op(&mut self, _id: ExprId, op: UnOp, operand: String) -> String {
            let op = match op {
                UnOp::Plus => "+",
                UnOp::Minus => "-",
            };
            format!("({op} {operand})")
        }

        fn bin_op(&mut self, _id: ExprId, op: BinOp, lhs: String, rhs: String) -> String {
            let op = match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::Pow => "^",
            };
            format!("({op} {lhs} {rhs})")
        }

        fn intrinsic<'a, I>(&mut self, _id: ExprId, intrinsic: Intrinsic, args: I) -> String
        where
            I: ExactSizeIterator<Item = Self::In<'a>>,
        {
            let name = match intrinsic {
                Intrinsic::Exp => "exp",
                Intrinsic::Log => "log",
                Intrinsic::Ln => "ln",
                Intrinsic::Sin => "sin",
                Intrinsic::Cos => "cos",
                Intrinsic::Tan => "tan",
                Intrinsic::Cot => "cot",
                Intrinsic::Abs => "abs",
                Intrinsic::Min => "min",
                Intrinsic::Max => "max",
            };

            let parts: Vec<_> = args.collect();
            format!("({name} {})", parts.join(" "))
        }

        fn var(&mut self, _id: ExprId, name: Symbol) -> String {
            self.interner.resolve(name).unwrap().to_owned()
        }

        fn lit(&mut self, _id: ExprId, value: Value) -> String {
            match value {
                Value::F32(x) => format!("{x}"),
                Value::I32(x) => format!("{x}"),
            }
        }

        fn map_error(&mut self, _id: ExprId, _inner: Option<String>) -> String {
            panic!("AST contains an error node")
        }
    }

    fn sexpr(src: &str) -> String {
        let mut folder = SExpr::default();
        let ast = parse(tokenize(src), &mut folder.interner, &mut Vec::new());
        ast.fold(folder)
    }

    #[test]
    fn parse_sexpr() {
        assert_eq!(sexpr("1 + 2 * 3"), "(+ 1 (* 2 3))");
        assert_eq!(sexpr("(1 + 2) * 3"), "(* (+ 1 2) 3)");
        assert_eq!(sexpr("2 ^ 3 ^ 2"), "(^ 2 (^ 3 2))");
        assert_eq!(sexpr("-2 ^ 2"), "(- (^ 2 2))");
        assert_eq!(sexpr("-2 * 3"), "(* (- 2) 3)");
        assert_eq!(sexpr("2 ^ -3"), "(^ 2 (- 3))");
        assert_eq!(sexpr("6 / 2 / 3"), "(/ (/ 6 2) 3)");
        assert_eq!(sexpr("2.5"), "2.5");

        assert_eq!(sexpr("x + y * z"), "(+ x (* y z))");
        assert_eq!(sexpr("x ^ 2 + y ^ 2"), "(+ (^ x 2) (^ y 2))");
        assert_eq!(sexpr("-радиус ^ 2"), "(- (^ радиус 2))");
        assert_eq!(sexpr("2 * pi"), format!("(* 2 {PI})"));
        assert_eq!(sexpr("π ^ 2"), format!("(^ {PI} 2)"));

        assert_eq!(sexpr("sin(x)"), "(sin x)");
        assert_eq!(sexpr("log(x)"), "(log x)");
        assert_eq!(sexpr("ln(x)"), "(ln x)");
        assert_eq!(sexpr("-sin(x) ^ 2"), "(- (^ (sin x) 2))");
        assert_eq!(sexpr("cos(x + 1) * 2"), "(* (cos (+ x 1)) 2)");
        assert_eq!(sexpr("exp(sin(x), y)"), "(exp (sin x) y)");
        assert_eq!(sexpr("abs(x)"), "(abs x)");
        assert_eq!(sexpr("min(x, 2)"), "(min x 2)");
        assert_eq!(sexpr("max(x, min(y, z))"), "(max x (min y z))");
    }

    #[test]
    #[should_panic]
    fn empty_input() {
        sexpr("");
    }

    #[test]
    #[should_panic]
    fn missing_operand() {
        sexpr("1 + *");
    }

    #[test]
    #[should_panic]
    fn unknown_token() {
        sexpr("1 + $");
    }

    #[test]
    #[should_panic]
    fn int_overflow() {
        sexpr("2147483648");
    }

    #[test]
    #[should_panic]
    fn trailing_input() {
        sexpr("1 2");
    }

    #[test]
    #[should_panic]
    fn missing_rparen() {
        sexpr("(1 + 2");
    }

    #[test]
    #[should_panic]
    fn call_missing_lparen() {
        sexpr("sin x");
    }

    #[test]
    #[should_panic]
    fn call_missing_rparen() {
        sexpr("sin(x");
    }

    #[test]
    #[should_panic]
    fn call_empty_args() {
        sexpr("sin()");
    }
}
