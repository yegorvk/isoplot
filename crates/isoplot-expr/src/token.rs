use crate::span::{BytePos, Span};
use logos::Logos;

pub(crate) struct Token<'src> {
    pub(crate) kind: TokenKind<'src>,
    pub(crate) span: Span,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Logos)]
#[logos(skip r"[ \t\r\n\f]+")]
pub(crate) enum TokenKind<'src> {
    #[regex(r"[0-9]+\.[0-9]+", |lex| LitFloat(lex.slice()))]
    Float(LitFloat<'src>),

    #[regex(r"[0-9]+", |lex| LitInt(lex.slice()))]
    Int(LitInt<'src>),

    #[regex(r"[\p{XID_Start}_][\p{XID_Continue}]*")]
    Ident(&'src str),

    #[token("+")]
    Plus,

    #[token("-")]
    Minus,

    #[token("*")]
    Star,

    #[token("/")]
    Slash,

    #[token("^")]
    Caret,

    #[token("(")]
    LParen,

    #[token(")")]
    RParen,

    #[regex(".", priority = 0)]
    Unknown,

    /// Sentinel "EOF" token
    Eof,
}

pub(crate) fn tokenize<'src>(src: &'src str) -> impl Iterator<Item = Token<'src>> {
    assert!(src.len() < u32::MAX as usize, "`src` is too large");

    let eof_token = Token {
        kind: TokenKind::Eof,
        span: Span::empty(BytePos(src.len() as u32)),
    };

    TokenKind::lexer(src)
        .spanned()
        .map(|(kind, range)| Token {
            kind: kind.unwrap(),
            span: Span::new(BytePos(range.start as u32), BytePos(range.end as u32)),
        })
        .chain([eof_token])
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct OverflowError;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) struct LitInt<'src>(&'src str);

impl LitInt<'_> {
    pub(crate) fn parse_i32(self) -> Result<i32, OverflowError> {
        self.0.parse().map_err(|_| OverflowError)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) struct LitFloat<'src>(&'src str);

impl LitFloat<'_> {
    pub(crate) fn parse_f32(self) -> Result<f32, OverflowError> {
        let value: f32 = self.0.parse().unwrap();
        if value.is_finite() {
            Ok(value)
        } else {
            Err(OverflowError)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(src: &str) -> Vec<TokenKind<'_>> {
        tokenize(src).map(|token| token.kind).collect()
    }

    fn tokenize_int(src: &str) -> LitInt<'_> {
        match tokenize(src).next().unwrap().kind {
            TokenKind::Int(lit) => lit,
            kind => panic!("expected an int token, got {kind:?}"),
        }
    }

    fn tokenize_float(src: &str) -> LitFloat<'_> {
        match tokenize(src).next().unwrap().kind {
            TokenKind::Float(lit) => lit,
            kind => panic!("expected a float token, got {kind:?}"),
        }
    }

    #[test]
    fn test_tokenize() {
        use TokenKind::*;

        assert_eq!(
            tokens("1 + 2.5 * (30 - 4) / 5"),
            [
                Int(LitInt("1")),
                Plus,
                Float(LitFloat("2.5")),
                Star,
                LParen,
                Int(LitInt("30")),
                Minus,
                Int(LitInt("4")),
                RParen,
                Slash,
                Int(LitInt("5")),
                Eof
            ]
        );

        assert_eq!(tokens(" \t\r\n"), [Eof]);

        assert_eq!(tokens("1."), [Int(LitInt("1")), Unknown, Eof]);
        assert_eq!(tokens(".5"), [Unknown, Int(LitInt("5")), Eof]);

        assert_eq!(tokens("-1"), [Minus, Int(LitInt("1")), Eof]);

        assert_eq!(
            tokens("x + радиус_2"),
            [Ident("x"), Plus, Ident("радиус_2"), Eof]
        );
        assert_eq!(tokens("1x"), [Int(LitInt("1")), Ident("x"), Eof]);
    }

    #[test]
    fn test_ident_spans() {
        let tokens: Vec<_> = tokenize("р + ф2").collect();

        assert_eq!(tokens[0].span, Span::new(BytePos(0), BytePos(2)));
        assert_eq!(tokens[1].span, Span::new(BytePos(3), BytePos(4)));
        assert_eq!(tokens[2].span, Span::new(BytePos(5), BytePos(8)));
    }

    #[test]
    fn test_token_spans() {
        let tokens: Vec<_> = tokenize(" 12 + 3.5").collect();

        assert_eq!(tokens[0].span, Span::new(BytePos(1), BytePos(3)));
        assert_eq!(tokens[1].span, Span::new(BytePos(4), BytePos(5)));
        assert_eq!(tokens[2].span, Span::new(BytePos(6), BytePos(9)));
        assert_eq!(tokens[3].span, Span::new(BytePos(9), BytePos(9)));
    }

    #[test]
    fn test_parse_i32() {
        assert_eq!(tokenize_int("2147483647").parse_i32().unwrap(), i32::MAX);

        // i32::MAX + 1
        assert!(tokenize_int("2147483648").parse_i32().is_err());
    }

    #[test]
    fn test_parse_f32() {
        assert_eq!(tokenize_float("2.5").parse_f32().unwrap(), 2.5);

        // 1e39 > f32::MAX
        let src = format!("1{}.0", "0".repeat(39));
        assert!(tokenize_float(&src).parse_f32().is_err());
    }
}
