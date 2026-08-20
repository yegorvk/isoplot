use std::{
    cell::{RefCell, RefMut},
    sync::Arc,
};

use crate::{
    layout::RawValue,
    tape::{Instr, Tape, ValueId},
};

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
            buf: vec![RawValue::ZERO; self.tape.num_args() + self.tape.instrs().len()].into(),
            tape: Arc::clone(&self.tape),
        }
    }
}

pub(super) struct Evaluator {
    tape: Arc<Tape>,
    buf: RefCell<Vec<RawValue>>,
}

impl Evaluator {
    pub(super) fn evaluate_into(&self, args: &[RawValue], results: &mut [RawValue]) {
        debug_assert_eq!(results.len(), self.tape.num_results());
        let buf = self.run(args);
        results.copy_from_slice(&buf[buf.len() - results.len()..]);
    }

    fn run(&self, args: &[RawValue]) -> RefMut<'_, Vec<RawValue>> {
        let num_args = self.tape.num_args();
        debug_assert_eq!(args.len(), num_args);

        let mut buf = self.buf.borrow_mut();
        buf[..num_args].copy_from_slice(args);

        for (i, instr) in self.tape.instrs().iter().enumerate() {
            let v_i32 = |id: ValueId| buf[id.index()].as_i32();
            let v_bool = |id: ValueId| buf[id.index()].as_bool();
            let v_f32 = |id: ValueId| buf[id.index()].as_f32();

            let result = match *instr {
                Instr::I32Const(value) => RawValue::from_i32(value),
                Instr::BoolConst(value) => RawValue::from_bool(value),
                Instr::F32Const(value) => RawValue::from_f32(value),

                Instr::I32Add(lhs, rhs) => RawValue::from_i32(v_i32(lhs).wrapping_add(v_i32(rhs))),
                Instr::I32Sub(lhs, rhs) => RawValue::from_i32(v_i32(lhs).wrapping_sub(v_i32(rhs))),
                Instr::I32Mul(lhs, rhs) => RawValue::from_i32(v_i32(lhs).wrapping_mul(v_i32(rhs))),

                Instr::I32Eq(lhs, rhs) => RawValue::from_bool(v_i32(lhs) == v_i32(rhs)),
                Instr::I32Ne(lhs, rhs) => RawValue::from_bool(v_i32(lhs) != v_i32(rhs)),
                Instr::I32Lt(lhs, rhs) => RawValue::from_bool(v_i32(lhs) < v_i32(rhs)),
                Instr::I32Le(lhs, rhs) => RawValue::from_bool(v_i32(lhs) <= v_i32(rhs)),
                Instr::I32Gt(lhs, rhs) => RawValue::from_bool(v_i32(lhs) > v_i32(rhs)),
                Instr::I32Ge(lhs, rhs) => RawValue::from_bool(v_i32(lhs) >= v_i32(rhs)),

                Instr::Not(src) => RawValue::from_bool(!v_bool(src)),
                Instr::And(lhs, rhs) => RawValue::from_bool(v_bool(lhs) & v_bool(rhs)),
                Instr::Or(lhs, rhs) => RawValue::from_bool(v_bool(lhs) | v_bool(rhs)),
                Instr::Xor(lhs, rhs) => RawValue::from_bool(v_bool(lhs) ^ v_bool(rhs)),

                Instr::I32FromBool(src) => RawValue::from_i32(v_bool(src) as i32),
                Instr::F32FromI32(src) => RawValue::from_f32(v_i32(src) as f32),
                Instr::F32FromBool(src) => RawValue::from_f32(v_bool(src) as i32 as f32),

                Instr::Copy(src) => buf[src.index()],

                Instr::F32Neg(src) => RawValue::from_f32(-v_f32(src)),
                Instr::F32Abs(src) => RawValue::from_f32(v_f32(src).abs()),
                Instr::F32Sign(src) => RawValue::from_f32(1.0f32.copysign(v_f32(src))),
                Instr::F32Floor(src) => RawValue::from_f32(v_f32(src).floor()),
                Instr::F32Add(lhs, rhs) => RawValue::from_f32(v_f32(lhs) + v_f32(rhs)),
                Instr::F32Sub(lhs, rhs) => RawValue::from_f32(v_f32(lhs) - v_f32(rhs)),
                Instr::F32Mul(lhs, rhs) => RawValue::from_f32(v_f32(lhs) * v_f32(rhs)),
                Instr::F32Div(lhs, rhs) => RawValue::from_f32(v_f32(lhs) / v_f32(rhs)),
                Instr::F32Min(lhs, rhs) => RawValue::from_f32(v_f32(lhs).min(v_f32(rhs))),
                Instr::F32Max(lhs, rhs) => RawValue::from_f32(v_f32(lhs).max(v_f32(rhs))),
                Instr::F32Powf(lhs, rhs) => RawValue::from_f32(v_f32(lhs).powf(v_f32(rhs))),
                Instr::F32Powi(lhs, rhs) => RawValue::from_f32(v_f32(lhs).powi(v_i32(rhs))),

                Instr::F32Exp(src) => RawValue::from_f32(v_f32(src).exp()),
                Instr::F32Ln(src) => RawValue::from_f32(v_f32(src).ln()),
                Instr::F32Lg(src) => RawValue::from_f32(v_f32(src).log10()),
                Instr::F32Sin(src) => RawValue::from_f32(v_f32(src).sin()),
                Instr::F32Cos(src) => RawValue::from_f32(v_f32(src).cos()),
                Instr::F32Tan(src) => RawValue::from_f32(v_f32(src).tan()),
                Instr::F32Cot(src) => RawValue::from_f32(v_f32(src).tan().recip()),

                Instr::F32Eq(lhs, rhs) => RawValue::from_bool(v_f32(lhs) == v_f32(rhs)),
                Instr::F32Ne(lhs, rhs) => RawValue::from_bool(v_f32(lhs) != v_f32(rhs)),
                Instr::F32Lt(lhs, rhs) => RawValue::from_bool(v_f32(lhs) < v_f32(rhs)),
                Instr::F32Le(lhs, rhs) => RawValue::from_bool(v_f32(lhs) <= v_f32(rhs)),
                Instr::F32Gt(lhs, rhs) => RawValue::from_bool(v_f32(lhs) > v_f32(rhs)),
                Instr::F32Ge(lhs, rhs) => RawValue::from_bool(v_f32(lhs) >= v_f32(rhs)),

                Instr::I32Sel(cond, v_true, v_false) | Instr::F32Sel(cond, v_true, v_false) => {
                    buf[if v_bool(cond) { v_true } else { v_false }.index()]
                }
            };

            buf[num_args + i] = result;
        }

        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Type;

    fn eval(evaluator: &Evaluator, args: &[f32]) -> f32 {
        let mut output = [RawValue::ZERO];
        evaluator.evaluate_into(bytemuck::cast_slice(args), &mut output);
        output[0].as_f32()
    }

    #[test]
    fn f32_ops() {
        let mut b = Tape::builder(vec![Type::F32, Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let y = b.arg(1);
        let sum = b.instr(Instr::F32Add(x, y));
        let prod = b.instr(Instr::F32Mul(sum, x));
        let diff = b.instr(Instr::F32Sub(prod, y));
        let quot = b.instr(Instr::F32Div(diff, y));
        let neg = b.instr(Instr::F32Neg(quot));
        let abs = b.instr(Instr::F32Abs(neg));
        let abs = b.instr(Instr::Copy(abs));
        let min = b.instr(Instr::F32Min(abs, x));
        let max = b.instr(Instr::F32Max(min, y));
        let c = b.instr(Instr::F32Const(0.5));
        b.instr(Instr::F32Add(max, c));
        let tape = b.build().unwrap();

        let (x, y) = (1.75f32, -0.5f32);
        let expected = (-((x + y) * x - y) / y).abs().min(x).max(y) + 0.5;
        assert_eq!(eval(&Fallback::new(tape).evaluator(), &[x, y]), expected);
    }

    #[test]
    fn f32_fn_calls() {
        let mut b = Tape::builder(vec![Type::F32, Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let y = b.arg(1);
        let sin = b.instr(Instr::F32Sin(x));
        let cos = b.instr(Instr::F32Cos(y));
        let tan = b.instr(Instr::F32Tan(x));
        let cot = b.instr(Instr::F32Cot(y));
        let exp = b.instr(Instr::F32Exp(x));
        let ln = b.instr(Instr::F32Ln(y));
        let lg = b.instr(Instr::F32Lg(x));
        let pow = b.instr(Instr::F32Powf(x, y));

        let mut acc = sin;
        for v in [cos, tan, cot, exp, ln, lg, pow] {
            acc = b.instr(Instr::F32Add(acc, v));
        }

        let tape = b.build().unwrap();
        let evaluator = Fallback::new(tape).evaluator();

        let (x, y) = (0.7f32, 1.3f32);
        let expected = x.sin()
            + y.cos()
            + x.tan()
            + y.tan().recip()
            + x.exp()
            + y.ln()
            + x.log10()
            + x.powf(y);

        assert_eq!(eval(&evaluator, &[x, y]), expected);
    }

    #[test]
    fn i32_ops() {
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let three = b.instr(Instr::I32Const(3));
        let two = b.instr(Instr::I32Const(2));
        let five = b.instr(Instr::I32Add(three, two));
        let diff = b.instr(Instr::I32Sub(five, two));
        let six = b.instr(Instr::I32Mul(diff, two));
        let sixf = b.instr(Instr::F32FromI32(six));
        let cube = b.instr(Instr::F32Powi(x, diff));
        b.instr(Instr::F32Add(sixf, cube));

        let tape = b.build().unwrap();
        let evaluator = Fallback::new(tape).evaluator();

        let x = 1.5f32;
        assert_eq!(eval(&evaluator, &[x]), 6.0 + x.powi(3));
    }

    #[test]
    fn f32_sign() {
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        b.instr(Instr::F32Sign(x));
        let tape = b.build().unwrap();
        let evaluator = Fallback::new(tape).evaluator();

        assert_eq!(eval(&evaluator, &[-3.5]), -1.0);
        assert_eq!(eval(&evaluator, &[2.0]), 1.0);
        assert_eq!(eval(&evaluator, &[0.0]), 1.0);
        assert_eq!(eval(&evaluator, &[-0.0]), -1.0);
    }

    #[test]
    fn f32_select() {
        // f = if 0 < x && x < y { x } else { y }
        let mut b = Tape::builder(vec![Type::F32, Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let y = b.arg(1);
        let zero = b.instr(Instr::F32Const(0.0));
        let lt = b.instr(Instr::F32Lt(x, y));
        let pos = b.instr(Instr::F32Gt(x, zero));
        let both = b.instr(Instr::And(lt, pos));
        b.instr(Instr::F32Sel(both, x, y));
        let tape = b.build().unwrap();
        let evaluator = Fallback::new(tape).evaluator();

        assert_eq!(eval(&evaluator, &[1.0, 2.0]), 1.0);
        assert_eq!(eval(&evaluator, &[-1.0, 2.0]), 2.0);
        assert_eq!(eval(&evaluator, &[3.0, 2.0]), 2.0);
    }

    #[test]
    fn bool_i32_ops() {
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let two = b.instr(Instr::I32Const(2));
        let three = b.instr(Instr::I32Const(3));
        let lt = b.instr(Instr::I32Lt(two, three)); // true
        let ge = b.instr(Instr::I32Ge(two, three)); // false
        let yes = b.instr(Instr::BoolConst(true));
        let xor = b.instr(Instr::Xor(lt, yes)); // false
        let any = b.instr(Instr::Or(xor, ge)); // false
        let not = b.instr(Instr::Not(any)); // true
        let sel = b.instr(Instr::I32Sel(not, two, three)); // 2
        let bump = b.instr(Instr::I32FromBool(lt)); // 1
        let sum = b.instr(Instr::I32Add(sel, bump)); // 3
        let sum = b.instr(Instr::F32FromI32(sum));
        let one = b.instr(Instr::F32FromBool(not)); // 1.0
        let sum = b.instr(Instr::F32Add(sum, one)); // 4.0
        b.instr(Instr::F32Add(sum, x));
        let tape = b.build().unwrap();
        let evaluator = Fallback::new(tape).evaluator();

        assert_eq!(eval(&evaluator, &[0.5]), 4.5);
    }

    #[test]
    fn multi_results() {
        let mut b = Tape::builder(vec![Type::F32, Type::F32], vec![Type::F32; 3]);
        let x = b.arg(0);
        let y = b.arg(1);
        b.instr(Instr::F32Add(x, y));
        b.instr(Instr::F32Mul(x, y));
        b.instr(Instr::F32Sub(x, y));
        let tape = b.build().unwrap();

        let (x, y) = (1.5f32, 2.25f32);
        let mut results = [0.0f32; 3];

        Fallback::new(tape).evaluator().evaluate_into(
            bytemuck::cast_slice(&[x, y]),
            bytemuck::cast_slice_mut(&mut results),
        );

        assert_eq!(results, [x + y, x * y, x - y]);
    }
}
