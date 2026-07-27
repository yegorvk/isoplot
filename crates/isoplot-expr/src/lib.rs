use std::collections::HashMap;

use crate::{
    ast::{Ast, BinOp, ExprId, Transformer, UnOp},
    parser::parse,
    symbol::{Interner, Symbol},
    token::tokenize,
};

mod ast;
mod parser;
mod span;
mod symbol;
mod token;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Value {
    F32(f32),
    Unit,
}

#[derive(Debug, Default)]
pub struct Environment {
    vars: HashMap<String, Value>,
}

impl Environment {
    pub fn insert_var(&mut self, name: String, value: Value) {
        self.vars.insert(name, value);
    }
}

pub struct Program {
    interner: Interner,
    ast: Ast,
}

impl Program {
    pub fn create(source: &str) -> Self {
        let mut interner = Interner::default();

        Program {
            ast: parse(tokenize(source), &mut interner),
            interner,
        }
    }

    pub fn evaluate(&self, env: &Environment) -> Value {
        self.ast.fold(Evaluator {
            interner: &self.interner,
            env,
        })
    }

    pub fn validate(&self, env: &Environment) -> bool {
        self.ast.fold(Validator {
            interner: &self.interner,
            env,
        })
    }
}

struct Validator<'a> {
    interner: &'a Interner,
    env: &'a Environment,
}

impl Transformer for Validator<'_> {
    type In<'a> = bool;
    type Out = bool;

    fn un_op(&mut self, _id: ExprId, _op: UnOp, operand: bool) -> bool {
        operand
    }

    fn bin_op(&mut self, _id: ExprId, _op: BinOp, lhs: bool, rhs: bool) -> bool {
        lhs && rhs
    }

    fn var(&mut self, _id: ExprId, name: Symbol) -> bool {
        let name = self.interner.resolve(name).unwrap();
        self.env.vars.contains_key(name)
    }

    fn lit(&mut self, _id: ExprId, _value: Value) -> bool {
        true
    }

    fn map_error(&mut self, _id: ExprId, _inner: Option<bool>) -> bool {
        false
    }
}

struct Evaluator<'a> {
    interner: &'a Interner,
    env: &'a Environment,
}

impl Transformer for Evaluator<'_> {
    type In<'a> = Value;
    type Out = Value;

    fn un_op(&mut self, _id: ExprId, op: UnOp, operand: Value) -> Value {
        let Value::F32(operand) = operand else {
            panic!("cannot apply `{op:?}` to `{operand:?}`")
        };

        Value::F32(match op {
            UnOp::Plus => operand,
            UnOp::Minus => -operand,
        })
    }

    fn bin_op(&mut self, _id: ExprId, op: BinOp, lhs: Value, rhs: Value) -> Value {
        let (Value::F32(lhs), Value::F32(rhs)) = (lhs, rhs) else {
            panic!("cannot apply `{op:?}` to `{lhs:?}` and `{rhs:?}`")
        };

        Value::F32(match op {
            BinOp::Add => lhs + rhs,
            BinOp::Sub => lhs - rhs,
            BinOp::Mul => lhs * rhs,
            BinOp::Div => lhs / rhs,
            BinOp::Pow => lhs.powf(rhs),
        })
    }

    fn var(&mut self, _id: ExprId, name: Symbol) -> Value {
        let name = self.interner.resolve(name).unwrap();
        match self.env.vars.get(name) {
            Some(&value) => value,
            None => panic!("unbound variable `{name}`"),
        }
    }

    fn lit(&mut self, _id: ExprId, value: Value) -> Value {
        value
    }

    fn map_error(&mut self, _id: ExprId, _inner: Option<Value>) -> Value {
        panic!("AST contains an error node")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Program>();
    }

    #[test]
    fn test_eval() {
        let program = Program::create("x ^ 2 + y ^ 2");

        let mut env = Environment::default();
        env.insert_var("x".to_owned(), Value::F32(3.0));
        env.insert_var("y".to_owned(), Value::F32(4.0));

        assert_eq!(program.evaluate(&env), Value::F32(25.0));
    }

    #[test]
    fn test_validate() {
        let mut env = Environment::default();
        env.insert_var("x".to_owned(), Value::F32(0.0));

        assert!(Program::create("x + 1").validate(&env));
        assert!(!Program::create("x + y").validate(&env));
        assert!(!Program::create("x + *").validate(&env));
        assert!(!Program::create("").validate(&env));
    }

    #[test]
    #[should_panic]
    fn test_eval_unbound_var() {
        let program = Program::create("x + 1");
        program.evaluate(&Environment::default());
    }

    #[test]
    #[should_panic]
    fn test_eval_parse_error() {
        let program = Program::create("1 + *");
        program.evaluate(&Environment::default());
    }
}
