use std::fmt::{self, Debug, Formatter};

use bytemuck::{Pod, Zeroable};

use crate::{
    layout::{Layout, ValueType, Vector},
    program::Program,
    tape::{Inlined, Instr, NewId, Tape, TapeBuilder, Type, ValueId},
};

/// Value of a scalar function together with its gradient
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Gradient<V: Layout> {
    pub value: V::Scalar,
    pub gradient: V,
}

impl<V: Layout> Debug for Gradient<V>
where
    V: Debug,
    V::Scalar: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gradient")
            .field("value", &self.value)
            .field("gradient", &self.gradient)
            .finish()
    }
}

impl<V: Layout> PartialEq for Gradient<V>
where
    V: PartialEq,
    V::Scalar: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.gradient == other.gradient
    }
}

unsafe impl<V: Layout> Zeroable for Gradient<V> {}
unsafe impl<V: Layout> Pod for Gradient<V> {}

impl<V: Layout> Vector for Gradient<V> {
    const LEN: usize = <V::Scalar as Vector>::LEN + V::LEN;
    type Scalar = V::Scalar;

    fn types() -> impl Iterator<Item = ValueType> {
        V::Scalar::types().chain(V::types())
    }
}

pub(crate) fn autodiff<Args: Layout>(
    program: &Program<Args, f32>,
) -> Program<Args, Gradient<Args>> {
    Program::new(Autodiff::new(program.tape()).run())
}

struct Autodiff<'a> {
    source: &'a Tape,
    inlined: Inlined<'a>,
    builder: TapeBuilder,
    adjoints: Vec<Option<NewId<f32>>>,
}

impl<'a> Autodiff<'a> {
    fn new(source: &'a Tape) -> Self {
        assert!(
            source.arg_types().iter().all(|&ty| ty == Type::F32),
            "autodiff arguments must be f32"
        );

        assert_eq!(
            source.num_results(),
            1,
            "only scalar-valued functions are supported"
        );

        let num_args = source.num_args();
        let num_values = num_args + source.instrs().len();

        let mut builder = Tape::builder(vec![Type::F32; num_args], vec![Type::F32; 1 + num_args]);
        let inlined = builder.extend(source);

        Self {
            source,
            inlined,
            builder,
            adjoints: vec![None; num_values],
        }
    }

    fn run(mut self) -> Tape {
        let num_args = self.source.num_args();

        let ret = self.inlined.result(0);
        let seed = self.builder.f32_const(1.0);
        *self.adjoints.last_mut().unwrap() = Some(seed);

        for r in self.source.instrs().rev() {
            let Some(w_bar) = self.adjoints[r.index()] else {
                continue;
            };
            let w = self.inlined.convert(r.id());
            self.propagate(r.instr(), w, w_bar);
        }

        self.builder.copy_f32(ret);
        for i in 0..num_args {
            match self.adjoints[i] {
                None => self.builder.f32_const(0.0),
                Some(v) => self.builder.copy_f32(v),
            };
        }

        self.builder.build().unwrap()
    }

    fn propagate(&mut self, instr: Instr, w: NewId<f32>, w_bar: NewId<f32>) {
        match instr {
            Instr::I32Const(_)
            | Instr::BoolConst(_)
            | Instr::F32Const(_)
            | Instr::I32Add(..)
            | Instr::I32Sub(..)
            | Instr::I32Mul(..)
            | Instr::I32Eq(..)
            | Instr::I32Ne(..)
            | Instr::I32Lt(..)
            | Instr::I32Le(..)
            | Instr::I32Gt(..)
            | Instr::I32Ge(..)
            | Instr::Not(_)
            | Instr::And(..)
            | Instr::Or(..)
            | Instr::Xor(..)
            | Instr::I32FromBool(_)
            | Instr::F32FromI32(_)
            | Instr::F32FromBool(_)
            | Instr::F32Sign(_)
            | Instr::F32Floor(_)
            | Instr::F32Eq(..)
            | Instr::F32Ne(..)
            | Instr::F32Lt(..)
            | Instr::F32Le(..)
            | Instr::F32Gt(..)
            | Instr::F32Ge(..)
            | Instr::I32Sel(..)
            | Instr::CopyI32(_)
            | Instr::CopyBool(_) => {}

            Instr::CopyF32(src) => self.acc(src, w_bar),

            Instr::F32Neg(src) => {
                let c = self.builder.f32_neg(w_bar);
                self.acc(src, c);
            }

            Instr::F32Abs(src) => {
                let sign = self.builder.f32_sign(self.inlined.convert(src));
                let c = self.builder.f32_mul(w_bar, sign);
                self.acc(src, c);
            }

            Instr::F32Add(lhs, rhs) => {
                self.acc(lhs, w_bar);
                self.acc(rhs, w_bar);
            }

            Instr::F32Sub(lhs, rhs) => {
                self.acc(lhs, w_bar);
                let c = self.builder.f32_neg(w_bar);
                self.acc(rhs, c);
            }

            Instr::F32Mul(lhs, rhs) => {
                let c_l = self.builder.f32_mul(w_bar, self.inlined.convert(rhs));
                self.acc(lhs, c_l);
                let c_r = self.builder.f32_mul(w_bar, self.inlined.convert(lhs));
                self.acc(rhs, c_r);
            }

            Instr::F32Div(lhs, rhs) => {
                let y = self.inlined.convert(rhs);
                let c_l = self.builder.f32_div(w_bar, y);
                self.acc(lhs, c_l);
                let t = self.builder.f32_mul(w_bar, w);
                let t = self.builder.f32_div(t, y);
                let c_r = self.builder.f32_neg(t);
                self.acc(rhs, c_r);
            }

            Instr::F32Min(lhs, rhs) => {
                let (x, y) = (self.inlined.convert(lhs), self.inlined.convert(rhs));
                let (lhs_low, lhs_high) = self.select_weights(x, y);
                let c_l = self.builder.f32_mul(w_bar, lhs_low);
                self.acc(lhs, c_l);
                let c_r = self.builder.f32_mul(w_bar, lhs_high);
                self.acc(rhs, c_r);
            }

            Instr::F32Max(lhs, rhs) => {
                let (x, y) = (self.inlined.convert(lhs), self.inlined.convert(rhs));
                let (lhs_low, lhs_high) = self.select_weights(x, y);
                let c_l = self.builder.f32_mul(w_bar, lhs_high);
                self.acc(lhs, c_l);
                let c_r = self.builder.f32_mul(w_bar, lhs_low);
                self.acc(rhs, c_r);
            }

            Instr::F32Powf(lhs, rhs) => {
                let (x, y) = (self.inlined.convert(lhs), self.inlined.convert(rhs));
                let one = self.builder.f32_const(1.0);
                let e = self.builder.f32_sub(y, one);
                let p = self.builder.f32_powf(x, e);
                let t = self.builder.f32_mul(y, p);
                let c_l = self.builder.f32_mul(w_bar, t);
                self.acc(lhs, c_l);
                let ln = self.builder.f32_ln(x);
                let t = self.builder.f32_mul(w, ln);
                let cr = self.builder.f32_mul(w_bar, t);
                self.acc(rhs, cr);
            }

            Instr::F32Powi(lhs, rhs) => {
                let (x, n) = (self.inlined.convert(lhs), self.inlined.convert(rhs));
                let one = self.builder.i32_const(1);
                let e = self.builder.i32_sub(n, one);
                let p = self.builder.f32_powi(x, e);
                let n = self.builder.f32_from_i32(n);
                let t = self.builder.f32_mul(n, p);
                let c_l = self.builder.f32_mul(w_bar, t);
                self.acc(lhs, c_l);
            }

            Instr::F32Exp(src) => {
                let c = self.builder.f32_mul(w_bar, w);
                self.acc(src, c);
            }

            Instr::F32Ln(src) => {
                let c = self.builder.f32_div(w_bar, self.inlined.convert(src));
                self.acc(src, c);
            }

            Instr::F32Lg(src) => {
                let k = self.builder.f32_const(std::f32::consts::LOG10_E);
                let t = self.builder.f32_div(k, self.inlined.convert(src));
                let c = self.builder.f32_mul(w_bar, t);
                self.acc(src, c);
            }

            Instr::F32Sin(src) => {
                let cos = self.builder.f32_cos(self.inlined.convert(src));
                let c = self.builder.f32_mul(w_bar, cos);
                self.acc(src, c);
            }

            Instr::F32Cos(src) => {
                let sin = self.builder.f32_sin(self.inlined.convert(src));
                let m = self.builder.f32_mul(w_bar, sin);
                let c = self.builder.f32_neg(m);
                self.acc(src, c);
            }

            Instr::F32Tan(src) => {
                let sq = self.builder.f32_mul(w, w);
                let one = self.builder.f32_const(1.0);
                let t = self.builder.f32_add(sq, one);
                let c = self.builder.f32_mul(w_bar, t);
                self.acc(src, c);
            }

            Instr::F32Cot(src) => {
                let sq = self.builder.f32_mul(w, w);
                let one = self.builder.f32_const(1.0);
                let t = self.builder.f32_add(sq, one);
                let m = self.builder.f32_mul(w_bar, t);
                let c = self.builder.f32_neg(m);
                self.acc(src, c);
            }

            Instr::F32Sel(cond, v_true, v_false) => {
                let cond = self.inlined.convert(cond);
                let zero = self.builder.f32_const(0.0);
                let c_t = self.builder.f32_sel(cond, w_bar, zero);
                self.acc(v_true, c_t);
                let c_f = self.builder.f32_sel(cond, zero, w_bar);
                self.acc(v_false, c_f);
            }
        }
    }

    fn acc(&mut self, target: ValueId<f32>, contrib: NewId<f32>) {
        let slot = target.index();
        self.adjoints[slot] = Some(match self.adjoints[slot] {
            None => contrib,
            Some(prev) => self.builder.f32_add(prev, contrib),
        });
    }

    /// Returns (1.0, 0.0) if `lhs` <= `rhs`, or (0.0, 1.0) otherwise.
    fn select_weights(&mut self, lhs: NewId<f32>, rhs: NewId<f32>) -> (NewId<f32>, NewId<f32>) {
        let half = self.builder.f32_const(0.5);
        let diff = self.builder.f32_sub(rhs, lhs);
        let sign = self.builder.f32_sign(diff);
        let half_sign = self.builder.f32_mul(sign, half);
        let lhs_low = self.builder.f32_add(half, half_sign);
        let lhs_high = self.builder.f32_sub(half, half_sign);
        (lhs_low, lhs_high)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::{Fallback, Instance},
        layout::VectorExt,
    };

    fn gradient<const N: usize>(tape: &Tape, inputs: &[f32; N]) -> Vec<f32> {
        let grad = Autodiff::new(tape).run();
        assert_eq!(grad.num_args(), N);
        assert_eq!(grad.num_results(), 1 + N);

        let output = Instance::<Fallback, [f32; N], Gradient<[f32; N]>>::new(grad)
            .evaluator()
            .evaluate(inputs);

        bytemuck::cast_slice(output.raw_values()).to_vec()
    }

    #[test]
    fn polynomial() {
        // f = x*y + x
        let mut b = Tape::builder(vec![Type::F32; 2], vec![Type::F32]);
        let x = b.arg(0);
        let y = b.arg(1);
        let xy = b.f32_mul(x, y);
        b.f32_add(xy, x);
        let tape = b.build().unwrap();

        let (x, y) = (3.0f32, 5.0f32);
        assert_eq!(gradient(&tape, &[x, y]), [x * y + x, y + 1.0, x]);
    }

    #[test]
    fn shared_operand() {
        // f = x*x
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        b.f32_mul(x, x);
        let tape = b.build().unwrap();

        let x = 1.75f32;
        assert_eq!(gradient(&tape, &[x]), [x * x, x + x]);
    }

    #[test]
    fn chain_rule() {
        // f = sin(x) * exp(x)
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let sin = b.f32_sin(x);
        let exp = b.f32_exp(x);
        b.f32_mul(sin, exp);
        let tape = b.build().unwrap();

        let x = 0.7f32;
        let expected = x.sin() * x.exp() + x.exp() * x.cos();
        assert_eq!(gradient(&tape, &[x]), [x.sin() * x.exp(), expected]);
    }

    #[test]
    fn min_selects_branch() {
        // f = min(x, y*y)
        let mut b = Tape::builder(vec![Type::F32; 2], vec![Type::F32]);
        let x = b.arg(0);
        let y = b.arg(1);
        let yy = b.f32_mul(y, y);
        b.f32_min(x, yy);
        let tape = b.build().unwrap();

        // x selected
        assert_eq!(gradient(&tape, &[1.0, 2.0]), [1.0, 1.0, 0.0]);

        // y*y selected
        assert_eq!(gradient(&tape, &[5.0, 2.0]), [4.0, 0.0, 4.0]);

        // a tie, x wins
        assert_eq!(gradient(&tape, &[4.0, 2.0]), [4.0, 1.0, 0.0]);
    }

    #[test]
    fn equal_operand_min_max() {
        // f = max(x, x)
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        b.f32_max(x, x);
        let tape = b.build().unwrap();
        assert_eq!(gradient(&tape, &[3.0]), [3.0, 1.0]);
    }

    #[test]
    fn powi_and_abs() {
        // f = |x^3|
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let n = b.i32_const(3);
        let cube = b.f32_powi(x, n);
        b.f32_abs(cube);
        let tape = b.build().unwrap();

        assert_eq!(gradient(&tape, &[2.0]), [8.0, 12.0]);
        assert_eq!(gradient(&tape, &[-2.0]), [8.0, -12.0]);

        // must be finite everywhere
        assert_eq!(gradient(&tape, &[0.0]), [0.0, 0.0]);
    }

    #[test]
    fn select_gradient() {
        // f = if x < 2 { x*x } else { x }
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let two = b.f32_const(2.0);
        let lt = b.f32_lt(x, two);
        let sq = b.f32_mul(x, x);
        b.f32_sel(lt, sq, x);
        let tape = b.build().unwrap();

        assert_eq!(gradient(&tape, &[1.0]), [1.0, 2.0]);
        assert_eq!(gradient(&tape, &[3.0]), [3.0, 1.0]);
    }

    #[test]
    fn unused_argument() {
        // f = y, x unused
        let mut b = Tape::builder(vec![Type::F32; 2], vec![Type::F32]);
        let y = b.arg(1);
        b.copy_f32(y);
        let tape = b.build().unwrap();
        assert_eq!(gradient(&tape, &[3.0, 7.0]), [7.0, 0.0, 1.0]);
    }
}
