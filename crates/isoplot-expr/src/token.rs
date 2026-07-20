use std::ops::Range;

use crate::span::{BytePos, Span};
use logos::Logos;

#[derive(Copy, Clone, Debug)]
pub(crate) struct Token {
    pub(crate) kind: TokenKind,
    pub(crate) span: Span,
}

impl Token {
    pub(crate) fn unwrap_int(&self) -> LitInt {
        assert_eq!(self.kind, TokenKind::Int);
        LitInt(*self)
    }

    pub(crate) fn unwrap_float(&self) -> LitFloat {
        assert_eq!(self.kind, TokenKind::Float);
        LitFloat(*self)
    }
}

#[derive(Copy, Clone, PartialEq, Debug, Logos)]
#[logos(skip r"[ \t\r\n\f]+")]
pub(crate) enum TokenKind {
    #[regex(r"[0-9]+\.[0-9]+")]
    Float,

    #[regex(r"[0-9]+")]
    Int,

    #[token("+")]
    Plus,

    #[token("-")]
    Minus,

    #[token("*")]
    Star,

    #[token("/")]
    Slash,

    #[token("(")]
    LParen,

    #[token(")")]
    RParen,

    #[regex(".", priority = 0)]
    Unknown,

    /// Sentinel "EOF" token
    Eof,
}

pub(crate) fn tokenize(src: &str) -> Vec<Token> {
    let eof = Token {
        kind: TokenKind::Eof,
        span: Span::new(BytePos(src.len() as u32), BytePos(src.len() as u32)),
    };

    TokenKind::lexer(src)
        .spanned()
        .map(|(kind, range)| Token {
            kind: kind.unwrap(),
            span: token_span(range),
        })
        .chain([eof])
        .collect()
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct OverflowError;

#[derive(Copy, Clone, Debug)]
pub(crate) struct LitInt(Token);

impl LitInt {
    pub(crate) fn span(&self) -> Span {
        self.0.span
    }

    pub(crate) fn parse_i32(&self, src: &str) -> Result<i32, OverflowError> {
        src[span_range(self.0.span)]
            .parse()
            .map_err(|_| OverflowError)
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct LitFloat(Token);

impl LitFloat {
    pub(crate) fn span(&self) -> Span {
        self.0.span
    }

    pub(crate) fn parse_f32(&self, src: &str) -> Result<f32, OverflowError> {
        let value: f32 = src[span_range(self.0.span)].parse().unwrap();
        if value.is_finite() {
            Ok(value)
        } else {
            Err(OverflowError)
        }
    }
}

const fn token_span(range: Range<usize>) -> Span {
    Span::new(BytePos(range.start as u32), BytePos(range.end as u32))
}

fn span_range(span: Span) -> Range<usize> {
    span.start().as_usize()..span.end().as_usize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokenKind> {
        tokenize(src).iter().map(|token| token.kind).collect()
    }

    #[test]
    fn test_tokenize() {
        use TokenKind::*;

        assert_eq!(
            kinds("1 + 2.5 * (30 - 4) / 5"),
            [
                Int, Plus, Float, Star, LParen, Int, Minus, Int, RParen, Slash, Int, Eof
            ]
        );

        assert_eq!(kinds(" \t\r\n"), [Eof]);

        assert_eq!(kinds("1."), [Int, Unknown, Eof]);
        assert_eq!(kinds(".5"), [Unknown, Int, Eof]);

        assert_eq!(kinds("-1"), [Minus, Int, Eof]);
    }

    #[test]
    fn test_token_spans() {
        let tokens = tokenize(" 12 + 3.5");

        assert_eq!(tokens[0].span, Span::new(BytePos(1), BytePos(3)));
        assert_eq!(tokens[1].span, Span::new(BytePos(4), BytePos(5)));
        assert_eq!(tokens[2].span, Span::new(BytePos(6), BytePos(9)));
        assert_eq!(tokens[3].span, Span::new(BytePos(9), BytePos(9)));
    }

    #[test]
    fn test_parse_i32() {
        let src = "2147483647";
        assert_eq!(
            tokenize(src)[0].unwrap_int().parse_i32(src).unwrap(),
            i32::MAX
        );

        // i32::MAX + 1
        let src = "2147483648";
        assert!(tokenize(src)[0].unwrap_int().parse_i32(src).is_err());
    }

    #[test]
    fn test_parse_f32() {
        let src = "2.5";
        assert_eq!(tokenize(src)[0].unwrap_float().parse_f32(src).unwrap(), 2.5);

        // 1e39 > f32::MAX
        let src = format!("1{}.0", "0".repeat(39));
        assert!(tokenize(&src)[0].unwrap_float().parse_f32(&src).is_err());
    }
}
