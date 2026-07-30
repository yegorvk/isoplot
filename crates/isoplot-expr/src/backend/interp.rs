use std::sync::Arc;

use crate::instrs::{Instruction, Instructions, ValueId};

pub(crate) struct Instance {
    instrs: Arc<Instructions>,
    buf: Vec<f32>,
}

impl Clone for Instance {
    fn clone(&self) -> Self {
        Self {
            instrs: Arc::clone(&self.instrs),
            buf: self.buf.clone(),
        }
    }
}

impl Instance {
    pub(super) fn new(instrs: Instructions) -> Self {
        Self {
            buf: vec![0f32; instrs.num_params() + instrs.instrs().len()],
            instrs: Arc::new(instrs),
        }
    }

    pub(super) fn evaluate(&mut self, inputs: &[f32]) -> f32 {
        let num_params = self.instrs.num_params();
        assert_eq!(inputs.len(), num_params);

        let buf = &mut self.buf;
        buf[..num_params].copy_from_slice(inputs);

        for (i, instr) in self.instrs.instrs().iter().enumerate() {
            let value = |id: ValueId| buf[id.index()];

            let result = match *instr {
                Instruction::I32Const(_)
                | Instruction::I32Add(..)
                | Instruction::I32Sub(..)
                | Instruction::I32Mul(..)
                | Instruction::F32FromI32(_)
                | Instruction::F32Powi(..) => unimplemented!(),

                Instruction::F32Const(value) => value,

                Instruction::F32Neg(src) => -value(src),
                Instruction::F32Abs(src) => value(src).abs(),
                Instruction::F32Add(lhs, rhs) => value(lhs) + value(rhs),
                Instruction::F32Sub(lhs, rhs) => value(lhs) - value(rhs),
                Instruction::F32Mul(lhs, rhs) => value(lhs) * value(rhs),
                Instruction::F32Div(lhs, rhs) => value(lhs) / value(rhs),
                Instruction::F32Min(lhs, rhs) => value(lhs).min(value(rhs)),
                Instruction::F32Max(lhs, rhs) => value(lhs).max(value(rhs)),
                Instruction::F32Powf(lhs, rhs) => value(lhs).powf(value(rhs)),

                Instruction::F32Exp(src) => value(src).exp(),
                Instruction::F32Ln(src) => value(src).ln(),
                Instruction::F32Lg(src) => value(src).log10(),
                Instruction::F32Sin(src) => value(src).sin(),
                Instruction::F32Cos(src) => value(src).cos(),
                Instruction::F32Tan(src) => value(src).tan(),
                Instruction::F32Cot(src) => value(src).tan().recip(),
            };

            buf[num_params + i] = result;
        }

        buf.last().copied().unwrap()
    }
}
