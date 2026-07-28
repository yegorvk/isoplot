use crate::{
    Program, Value, VarSlotKind,
    ast::{BinOp, ExprId, Transformer, UnOp},
    symbol::Symbol,
};

pub(crate) struct Instance {
    program: Program,
    consts: Vec<f32>,
}

impl Instance {
    pub(crate) fn new(program: Program, consts: Vec<f32>) -> Self {
        Self { program, consts }
    }

    pub(crate) fn call(&self, inputs: &[&[f32]], out: &mut [f32]) {
        assert_eq!(inputs.len(), self.program.shape.inputs.len());

        let batch_size = out.len();

        for lane in inputs {
            assert_eq!(lane.len(), batch_size);
        }

        let mut buf = Vec::with_capacity(self.program.ast.len() * batch_size);

        let root = self.program.ast.fold(Evaluator {
            program: &self.program,
            consts: &self.consts,
            inputs,
            batch_size,
            buf: &mut buf,
        });

        out.copy_from_slice(&buf[root..root + batch_size]);
    }
}

struct Evaluator<'a> {
    program: &'a Program,
    consts: &'a [f32],
    inputs: &'a [&'a [f32]],
    batch_size: usize,
    buf: &'a mut Vec<f32>,
}

impl Transformer for Evaluator<'_> {
    type In<'a> = usize;
    type Out = usize;

    fn un_op(&mut self, _id: ExprId, op: UnOp, operand: usize) -> usize {
        match op {
            UnOp::Plus => operand,
            UnOp::Minus => {
                let off = self.buf.len();

                for i in 0..self.batch_size {
                    let value = -self.buf[operand + i];
                    self.buf.push(value);
                }

                off
            }
        }
    }

    fn bin_op(&mut self, _id: ExprId, op: BinOp, lhs: usize, rhs: usize) -> usize {
        let off = self.buf.len();

        for i in 0..self.batch_size {
            let (lhs, rhs) = (self.buf[lhs + i], self.buf[rhs + i]);

            self.buf.push(match op {
                BinOp::Add => lhs + rhs,
                BinOp::Sub => lhs - rhs,
                BinOp::Mul => lhs * rhs,
                BinOp::Div => lhs / rhs,
                BinOp::Pow => lhs.powf(rhs),
            });
        }

        off
    }

    fn var(&mut self, id: ExprId, _name: Symbol) -> usize {
        let off = self.buf.len();

        match self.program.var_slots[id].unwrap().kind() {
            VarSlotKind::Input(index) => self.buf.extend_from_slice(self.inputs[index as usize]),
            VarSlotKind::Const(index) => {
                let value = self.consts[index as usize];
                self.buf.resize(off + self.batch_size, value);
            }
        }

        off
    }

    fn lit(&mut self, _id: ExprId, value: Value) -> usize {
        let Value::F32(value) = value else {
            panic!("cannot evaluate `{value:?}`")
        };

        let off = self.buf.len();
        self.buf.resize(off + self.batch_size, value);
        off
    }

    fn map_error(&mut self, _id: ExprId, _inner: Option<usize>) -> usize {
        panic!("AST contains an error node")
    }
}
