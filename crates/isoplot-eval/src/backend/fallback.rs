use std::{
    cell::{RefCell, RefMut},
    sync::Arc,
};

use bytemuck::{Pod, Zeroable};

use crate::tape::{Instr, Tape, ValueId};

pub(super) struct Fallback {
    tape: Arc<Tape>,
}

impl Clone for Fallback {
    fn clone(&self) -> Self {
        Self {
            tape: Arc::clone(&self.tape),
        }
    }
}

impl Fallback {
    pub(super) fn new(tape: Tape) -> Self {
        Self {
            tape: Arc::new(tape),
        }
    }

    pub(super) fn evaluator(&self) -> Evaluator {
        Evaluator {
            buf: vec![Value::ZERO; self.tape.num_arguments() + self.tape.instrs().len()].into(),
            tape: Arc::clone(&self.tape),
        }
    }
}

/// An untyped value (the tape has already been type checked).
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(transparent)]
struct Value(u32);

impl Value {
    const ZERO: Self = Value(0);

    const fn from_i32(value: i32) -> Self {
        Self(value as u32)
    }

    const fn from_f32(value: f32) -> Self {
        Self(value.to_bits())
    }

    const fn as_i32(self) -> i32 {
        self.0 as i32
    }

    const fn as_f32(self) -> f32 {
        f32::from_bits(self.0)
    }
}

pub(super) struct Evaluator {
    tape: Arc<Tape>,
    buf: RefCell<Vec<Value>>,
}

impl Evaluator {
    pub(super) fn evaluate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(self.tape.num_results(), 1);
        self.run(inputs).last().unwrap().as_f32()
    }

    pub(super) fn evaluate_into(&self, inputs: &[f32], outputs: &mut [f32]) {
        assert_eq!(outputs.len(), self.tape.num_results());
        let buf = self.run(inputs);
        let results = &buf[buf.len() - outputs.len()..];
        for (output, result) in outputs.iter_mut().zip(results) {
            *output = result.as_f32();
        }
    }

    fn run(&self, inputs: &[f32]) -> RefMut<'_, Vec<Value>> {
        let num_params = self.tape.num_arguments();
        assert_eq!(inputs.len(), num_params);

        let mut buf = self.buf.borrow_mut();
        buf[..num_params].copy_from_slice(bytemuck::cast_slice(inputs));

        for (i, instr) in self.tape.instrs().iter().enumerate() {
            let v_i32 = |id: ValueId| buf[id.index()].as_i32();
            let v_f32 = |id: ValueId| buf[id.index()].as_f32();

            let result = match *instr {
                Instr::I32Const(value) => Value::from_i32(value),
                Instr::F32Const(value) => Value::from_f32(value),

                Instr::I32Add(lhs, rhs) => Value::from_i32(v_i32(lhs).wrapping_add(v_i32(rhs))),
                Instr::I32Sub(lhs, rhs) => Value::from_i32(v_i32(lhs).wrapping_sub(v_i32(rhs))),
                Instr::I32Mul(lhs, rhs) => Value::from_i32(v_i32(lhs).wrapping_mul(v_i32(rhs))),

                Instr::F32FromI32(src) => Value::from_f32(v_i32(src) as f32),

                Instr::F32Neg(src) => Value::from_f32(-v_f32(src)),
                Instr::F32Abs(src) => Value::from_f32(v_f32(src).abs()),
                Instr::F32Add(lhs, rhs) => Value::from_f32(v_f32(lhs) + v_f32(rhs)),
                Instr::F32Sub(lhs, rhs) => Value::from_f32(v_f32(lhs) - v_f32(rhs)),
                Instr::F32Mul(lhs, rhs) => Value::from_f32(v_f32(lhs) * v_f32(rhs)),
                Instr::F32Div(lhs, rhs) => Value::from_f32(v_f32(lhs) / v_f32(rhs)),
                Instr::F32Min(lhs, rhs) => Value::from_f32(v_f32(lhs).min(v_f32(rhs))),
                Instr::F32Max(lhs, rhs) => Value::from_f32(v_f32(lhs).max(v_f32(rhs))),
                Instr::F32Powf(lhs, rhs) => Value::from_f32(v_f32(lhs).powf(v_f32(rhs))),
                Instr::F32Powi(lhs, rhs) => Value::from_f32(v_f32(lhs).powi(v_i32(rhs))),

                Instr::F32Exp(src) => Value::from_f32(v_f32(src).exp()),
                Instr::F32Ln(src) => Value::from_f32(v_f32(src).ln()),
                Instr::F32Lg(src) => Value::from_f32(v_f32(src).log10()),
                Instr::F32Sin(src) => Value::from_f32(v_f32(src).sin()),
                Instr::F32Cos(src) => Value::from_f32(v_f32(src).cos()),
                Instr::F32Tan(src) => Value::from_f32(v_f32(src).tan()),
                Instr::F32Cot(src) => Value::from_f32(v_f32(src).tan().recip()),
            };

            buf[num_params + i] = result;
        }

        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Type;

    #[test]
    fn multi_results() {
        let mut b = Tape::builder(vec![Type::F32, Type::F32], vec![Type::F32; 3]);
        let x = b.argument(0);
        let y = b.argument(1);
        b.instr(Instr::F32Add(x, y));
        b.instr(Instr::F32Mul(x, y));
        b.instr(Instr::F32Sub(x, y));
        let tape = b.build().unwrap();

        let (x, y) = (1.5f32, 2.25f32);
        let mut outputs = [0.0f32; 3];
        Fallback::new(tape).evaluator().evaluate_into(&[x, y], &mut outputs);
        assert_eq!(outputs, [x + y, x * y, x - y]);
    }
}
