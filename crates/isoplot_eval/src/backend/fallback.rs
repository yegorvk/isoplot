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

        for r in self.tape.instrs() {
            let v_i32 = |id: ValueId<i32>| buf[id.index()].as_i32();
            let v_bool = |id: ValueId<bool>| buf[id.index()].as_bool();
            let v_f32 = |id: ValueId<f32>| buf[id.index()].as_f32();

            let result = match r.instr() {
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

                Instr::CopyI32(src) => buf[src.index()],
                Instr::CopyBool(src) => buf[src.index()],
                Instr::CopyF32(src) => buf[src.index()],

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

                Instr::I32Sel(cond, v_true, v_false) => {
                    buf[if v_bool(cond) { v_true } else { v_false }.index()]
                }
                Instr::F32Sel(cond, v_true, v_false) => {
                    buf[if v_bool(cond) { v_true } else { v_false }.index()]
                }
            };

            buf[r.index()] = result;
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
        let sum = b.f32_add(x, y);
        let prod = b.f32_mul(sum, x);
        let diff = b.f32_sub(prod, y);
        let quot = b.f32_div(diff, y);
        let neg = b.f32_neg(quot);
        let abs = b.f32_abs(neg);
        let abs = b.copy_f32(abs);
        let min = b.f32_min(abs, x);
        let max = b.f32_max(min, y);
        let c = b.f32_const(0.5);
        b.f32_add(max, c);
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
        let sin = b.f32_sin(x);
        let cos = b.f32_cos(y);
        let tan = b.f32_tan(x);
        let cot = b.f32_cot(y);
        let exp = b.f32_exp(x);
        let ln = b.f32_ln(y);
        let lg = b.f32_lg(x);
        let pow = b.f32_powf(x, y);

        let mut acc = sin;
        for v in [cos, tan, cot, exp, ln, lg, pow] {
            acc = b.f32_add(acc, v);
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
        let three = b.i32_const(3);
        let two = b.i32_const(2);
        let five = b.i32_add(three, two);
        let diff = b.i32_sub(five, two);
        let six = b.i32_mul(diff, two);
        let sixf = b.f32_from_i32(six);
        let cube = b.f32_powi(x, diff);
        b.f32_add(sixf, cube);

        let tape = b.build().unwrap();
        let evaluator = Fallback::new(tape).evaluator();

        let x = 1.5f32;
        assert_eq!(eval(&evaluator, &[x]), 6.0 + x.powi(3));
    }

    #[test]
    fn f32_sign() {
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        b.f32_sign(x);
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
        let zero = b.f32_const(0.0);
        let lt = b.f32_lt(x, y);
        let pos = b.f32_gt(x, zero);
        let both = b.and(lt, pos);
        b.f32_sel(both, x, y);
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
        let two = b.i32_const(2);
        let three = b.i32_const(3);
        let lt = b.i32_lt(two, three); // true
        let ge = b.i32_ge(two, three); // false
        let yes = b.bool_const(true);
        let xor = b.xor(lt, yes); // false
        let any = b.or(xor, ge); // false
        let not = b.not(any); // true
        let sel = b.i32_sel(not, two, three); // 2
        let bump = b.i32_from_bool(lt); // 1
        let sum = b.i32_add(sel, bump); // 3
        let sum = b.f32_from_i32(sum);
        let one = b.f32_from_bool(not); // 1.0
        let sum = b.f32_add(sum, one); // 4.0
        b.f32_add(sum, x);
        let tape = b.build().unwrap();
        let evaluator = Fallback::new(tape).evaluator();

        assert_eq!(eval(&evaluator, &[0.5]), 4.5);
    }

    #[test]
    fn multi_results() {
        let mut b = Tape::builder(vec![Type::F32, Type::F32], vec![Type::F32; 3]);
        let x = b.arg(0);
        let y = b.arg(1);
        b.f32_add(x, y);
        b.f32_mul(x, y);
        b.f32_sub(x, y);
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
