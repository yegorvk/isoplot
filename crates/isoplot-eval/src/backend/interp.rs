use std::{cell::RefCell, sync::Arc};

use crate::tape::{Instr, Tape, ValueId};

pub(super) struct Instance {
    tape: Arc<Tape>,
}

impl Clone for Instance {
    fn clone(&self) -> Self {
        Self {
            tape: Arc::clone(&self.tape),
        }
    }
}

impl Instance {
    pub(super) fn new(tape: Tape) -> Self {
        Self {
            tape: Arc::new(tape),
        }
    }

    pub(super) fn evaluator(&self) -> Evaluator {
        Evaluator {
            buf: vec![0f32; self.tape.num_inputs() + self.tape.instrs().len()].into(),
            tape: Arc::clone(&self.tape),
        }
    }
}

pub(super) struct Evaluator {
    tape: Arc<Tape>,
    buf: RefCell<Vec<f32>>,
}

impl Evaluator {
    pub(super) fn evaluate(&self, inputs: &[f32]) -> f32 {
        let num_params = self.tape.num_inputs();
        assert_eq!(inputs.len(), num_params);

        let mut buf = self.buf.borrow_mut();
        buf[..num_params].copy_from_slice(inputs);

        for (i, instr) in self.tape.instrs().iter().enumerate() {
            let value = |id: ValueId| buf[id.index()];

            let result = match *instr {
                Instr::I32Const(_)
                | Instr::I32Add(..)
                | Instr::I32Sub(..)
                | Instr::I32Mul(..)
                | Instr::F32FromI32(_)
                | Instr::F32Powi(..) => unimplemented!(),

                Instr::F32Const(value) => value,

                Instr::F32Neg(src) => -value(src),
                Instr::F32Abs(src) => value(src).abs(),
                Instr::F32Add(lhs, rhs) => value(lhs) + value(rhs),
                Instr::F32Sub(lhs, rhs) => value(lhs) - value(rhs),
                Instr::F32Mul(lhs, rhs) => value(lhs) * value(rhs),
                Instr::F32Div(lhs, rhs) => value(lhs) / value(rhs),
                Instr::F32Min(lhs, rhs) => value(lhs).min(value(rhs)),
                Instr::F32Max(lhs, rhs) => value(lhs).max(value(rhs)),
                Instr::F32Powf(lhs, rhs) => value(lhs).powf(value(rhs)),

                Instr::F32Exp(src) => value(src).exp(),
                Instr::F32Ln(src) => value(src).ln(),
                Instr::F32Lg(src) => value(src).log10(),
                Instr::F32Sin(src) => value(src).sin(),
                Instr::F32Cos(src) => value(src).cos(),
                Instr::F32Tan(src) => value(src).tan(),
                Instr::F32Cot(src) => value(src).tan().recip(),
            };

            buf[num_params + i] = result;
        }

        buf.last().copied().unwrap()
    }
}
