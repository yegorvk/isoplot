use std::num::NonZeroU32;

use crate::{
    ast::{Ast, BinOp, DenseMap, ExprId, Intrinsic, Transformer, UnOp},
    parser::parse,
    symbol::{Interner, Symbol},
    token::tokenize,
};

mod ast;
mod backend;
mod parser;
mod span;
mod symbol;
mod token;

#[derive(Copy, Clone, PartialEq, Debug)]
pub(crate) enum Value {
    F32(f32),
}

#[derive(Debug)]
pub struct ValidateError;

#[derive(Debug)]
pub struct InstantiateError;

#[derive(Debug)]
pub struct Shape {
    inputs: Vec<String>,
    consts: Vec<String>,
}

impl Shape {
    pub fn builder() -> ShapeBuilder {
        ShapeBuilder {
            inputs: Vec::new(),
            consts: Vec::new(),
        }
    }
}

pub struct ShapeBuilder {
    inputs: Vec<String>,
    consts: Vec<String>,
}

impl ShapeBuilder {
    pub fn with_input(mut self, name: impl Into<String>) -> Self {
        self.inputs.push(name.into());
        self
    }

    pub fn with_const(mut self, name: impl Into<String>) -> Self {
        self.consts.push(name.into());
        self
    }

    pub fn build(self) -> Shape {
        let vars = || self.inputs.iter().chain(&self.consts);

        for (i, name) in vars().enumerate() {
            if vars().take(i).any(|prev| prev == name) {
                panic!("duplicate variable `{name}`");
            }
        }

        Shape {
            inputs: self.inputs,
            consts: self.consts,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct VarSlot(NonZeroU32);

const _: () = assert!(size_of::<Option<VarSlot>>() == 4);

#[derive(Copy, Clone, Debug)]
enum VarSlotKind {
    Input(u32),
    Const(u32),
}

impl VarSlot {
    fn new(kind: VarSlotKind) -> Self {
        let (tag, index) = match kind {
            VarSlotKind::Input(index) => (0, index),
            VarSlotKind::Const(index) => (1, index),
        };

        assert!(index < u32::MAX >> 1);
        Self(NonZeroU32::new((index << 1 | tag) + 1).unwrap())
    }

    fn kind(self) -> VarSlotKind {
        let raw = self.0.get() - 1;
        let index = raw >> 1;

        match raw & 1 {
            0 => VarSlotKind::Input(index),
            _ => VarSlotKind::Const(index),
        }
    }
}

pub struct Program {
    shape: Shape,
    ast: Ast,
    var_slots: DenseMap<Option<VarSlot>>,
}

impl Program {
    pub fn create(shape: Shape, source: &str) -> Result<Self, ValidateError> {
        let mut interner = Interner::default();
        let ast = parse(tokenize(source), &mut interner);

        let mut valid = true;

        let var_slots = DenseMap::build(
            &ast,
            Resolver {
                interner: &interner,
                shape: &shape,
                valid: &mut valid,
            },
        );

        if !valid {
            return Err(ValidateError);
        }

        Ok(Program {
            shape,
            ast,
            var_slots,
        })
    }

    pub fn instantiate(self, consts: &[(&str, f32)]) -> Result<Instance, InstantiateError> {
        let consts = self.resolve_consts(consts)?;

        Ok(Instance {
            backend: backend::Instance::new(self, consts),
        })
    }

    fn resolve_consts(&self, consts: &[(&str, f32)]) -> Result<Vec<f32>, InstantiateError> {
        let mut values = vec![None; self.shape.consts.len()];

        for &(name, value) in consts {
            let index = self
                .shape
                .consts
                .iter()
                .position(|cst| cst == name)
                .ok_or(InstantiateError)?;

            if values[index].replace(value).is_some() {
                return Err(InstantiateError);
            }
        }

        values
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(InstantiateError)
    }
}

pub struct Instance {
    backend: backend::Instance,
}

impl Instance {
    pub fn call(&self, inputs: &[&[f32]], out: &mut [f32]) {
        self.backend.call(inputs, out)
    }
}

struct Resolver<'a> {
    interner: &'a Interner,
    shape: &'a Shape,
    valid: &'a mut bool,
}

impl Transformer for Resolver<'_> {
    type In<'a> = &'a Option<VarSlot>;
    type Out = Option<VarSlot>;

    fn un_op(&mut self, _id: ExprId, _op: UnOp, _operand: &Option<VarSlot>) -> Option<VarSlot> {
        None
    }

    fn bin_op(
        &mut self,
        _id: ExprId,
        _op: BinOp,
        _lhs: &Option<VarSlot>,
        _rhs: &Option<VarSlot>,
    ) -> Option<VarSlot> {
        None
    }

    fn intrinsic<'a, I>(&mut self, _id: ExprId, intrinsic: Intrinsic, args: I) -> Option<VarSlot>
    where
        I: ExactSizeIterator<Item = &'a Option<VarSlot>>,
    {
        if args.len() != intrinsic.num_args() {
            *self.valid = false;
        }

        None
    }

    fn var(&mut self, _id: ExprId, name: Symbol) -> Option<VarSlot> {
        let name = self.interner.resolve(name).unwrap();

        if let Some(index) = self.shape.inputs.iter().position(|input| input == name) {
            return Some(VarSlot::new(VarSlotKind::Input(index as u32)));
        }

        if let Some(index) = self.shape.consts.iter().position(|cst| cst == name) {
            return Some(VarSlot::new(VarSlotKind::Const(index as u32)));
        }

        *self.valid = false;
        None
    }

    fn lit(&mut self, _id: ExprId, _value: Value) -> Option<VarSlot> {
        None
    }

    fn map_error(&mut self, _id: ExprId, _inner: Option<&Option<VarSlot>>) -> Option<VarSlot> {
        *self.valid = false;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xy_shape() -> Shape {
        Shape::builder().with_input("x").with_const("y").build()
    }

    #[test]
    fn instance_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Program>();
        assert_send_sync::<Instance>();
    }

    #[test]
    fn test_create() {
        assert!(Program::create(xy_shape(), "x + 1").is_ok());
        assert!(Program::create(xy_shape(), "x + y").is_ok());
        assert!(Program::create(xy_shape(), "x + w").is_err());
        assert!(Program::create(xy_shape(), "x + *").is_err());
        assert!(Program::create(xy_shape(), "").is_err());
        assert!(Program::create(xy_shape(), "sin(x)").is_ok());
        assert!(Program::create(xy_shape(), "sin(x, y)").is_err());
        assert!(Program::create(xy_shape(), "sin x").is_err());
    }

    #[test]
    #[should_panic]
    fn duplicate_var() {
        Shape::builder().with_input("x").with_const("x").build();
    }

    #[test]
    fn test_instantiate() {
        let program = || Program::create(xy_shape(), "x").unwrap();

        assert!(program().instantiate(&[("y", 1.0)]).is_ok());
        assert!(program().instantiate(&[]).is_err());
        assert!(program().instantiate(&[("w", 1.0)]).is_err());
        assert!(program().instantiate(&[("y", 1.0), ("y", 2.0)]).is_err());
    }

    #[test]
    fn test_eval() {
        let program = Program::create(xy_shape(), "x ^ 2 + y ^ 2").unwrap();
        let instance = program.instantiate(&[("y", 4.0)]).unwrap();

        let mut out = [0.0];
        instance.call(&[&[3.0]], &mut out);

        assert_eq!(out, [25.0]);
    }

    #[test]
    fn test_eval_intrinsic() {
        let program = Program::create(xy_shape(), "cos(x) * y").unwrap();
        let instance = program.instantiate(&[("y", 2.0)]).unwrap();

        let mut out = [0.0];
        instance.call(&[&[0.0]], &mut out);

        assert_eq!(out, [2.0]);
    }

    #[test]
    fn test_eval_batch() {
        let program = Program::create(xy_shape(), "y * x + 1").unwrap();
        let instance = program.instantiate(&[("y", 2.0)]).unwrap();

        let mut out = [0.0; 4];
        instance.call(&[&[1.0, 2.0, 3.0, 4.0]], &mut out);

        assert_eq!(out, [3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    #[should_panic]
    fn test_wrong_lane_count() {
        let program = Program::create(xy_shape(), "x").unwrap();
        let instance = program.instantiate(&[("y", 0.0)]).unwrap();
        instance.call(&[], &mut [0.0]);
    }
}
