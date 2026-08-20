use std::fmt::{self, Debug, Formatter};

use bytemuck::{Pod, Zeroable};

use crate::{
    layout::{Layout, ValueType, Vector},
    program::Program,
    tape::{Instr, Tape, TapeBuilder, Type, ValueId},
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
    builder: TapeBuilder,
    values: Vec<ValueId>,
    adjoints: Vec<Option<ValueId>>,
}

impl<'a> Autodiff<'a> {
    fn new(source: &'a Tape) -> Self {
        assert_eq!(
            source.num_results(),
            1,
            "only scalar-valued functions are supported"
        );

        let num_args = source.num_args();
        let num_values = num_args + source.instrs().len();

        let mut builder = Tape::builder(vec![Type::F32; num_args], vec![Type::F32; 1 + num_args]);

        let mut values = Vec::with_capacity(num_values);
        for i in 0..num_args {
            values.push(builder.arg(i));
        }
        for &instr in source.instrs() {
            values.push(builder.instr(instr));
        }

        Self {
            source,
            builder,
            values,
            adjoints: vec![None; num_values],
        }
    }

    fn run(mut self) -> Tape {
        let num_args = self.source.num_args();

        let ret = *self.values.last().unwrap();
        let seed = self.c_f32(1.0);
        self.adjoints[ret.index()] = Some(seed);

        for (i, &instr) in self.source.instrs().iter().enumerate().rev() {
            let index = num_args + i;
            let Some(w_bar) = self.adjoints[index] else {
                continue;
            };
            self.propagate(instr, self.values[index], w_bar);
        }

        self.builder.instr(Instr::Copy(ret));
        for i in 0..num_args {
            match self.adjoints[i] {
                None => self.builder.instr(Instr::F32Const(0.0)),
                Some(v) => self.builder.instr(Instr::Copy(v)),
            };
        }

        self.builder.build().unwrap()
    }

    fn propagate(&mut self, instr: Instr, w: ValueId, w_bar: ValueId) {
        match instr {
            Instr::I32Const(_)
            | Instr::F32Const(_)
            | Instr::I32Add(..)
            | Instr::I32Sub(..)
            | Instr::I32Mul(..)
            | Instr::F32FromI32(_)
            | Instr::F32Sign(_)
            | Instr::F32Floor(_) => {}

            Instr::Copy(src) => self.acc(src, w_bar),

            Instr::F32Neg(src) => {
                let c = self.builder.instr(Instr::F32Neg(w_bar));
                self.acc(src, c);
            }

            Instr::F32Abs(src) => {
                let sign = self.builder.instr(Instr::F32Sign(src));
                let c = self.builder.instr(Instr::F32Mul(w_bar, sign));
                self.acc(src, c);
            }

            Instr::F32Add(lhs, rhs) => {
                self.acc(lhs, w_bar);
                self.acc(rhs, w_bar);
            }

            Instr::F32Sub(lhs, rhs) => {
                self.acc(lhs, w_bar);
                let c = self.builder.instr(Instr::F32Neg(w_bar));
                self.acc(rhs, c);
            }

            Instr::F32Mul(lhs, rhs) => {
                let c_l = self.builder.instr(Instr::F32Mul(w_bar, rhs));
                self.acc(lhs, c_l);
                let c_r = self.builder.instr(Instr::F32Mul(w_bar, lhs));
                self.acc(rhs, c_r);
            }

            Instr::F32Div(lhs, rhs) => {
                let c_l = self.builder.instr(Instr::F32Div(w_bar, rhs));
                self.acc(lhs, c_l);
                let t = self.builder.instr(Instr::F32Mul(w_bar, w));
                let t = self.builder.instr(Instr::F32Div(t, rhs));
                let c_r = self.builder.instr(Instr::F32Neg(t));
                self.acc(rhs, c_r);
            }

            Instr::F32Min(lhs, rhs) => {
                let (lhs_low, lhs_high) = self.select_weights(lhs, rhs);
                let c_l = self.builder.instr(Instr::F32Mul(w_bar, lhs_low));
                self.acc(lhs, c_l);
                let c_r = self.builder.instr(Instr::F32Mul(w_bar, lhs_high));
                self.acc(rhs, c_r);
            }

            Instr::F32Max(lhs, rhs) => {
                let (lhs_low, lhs_high) = self.select_weights(lhs, rhs);
                let c_l = self.builder.instr(Instr::F32Mul(w_bar, lhs_high));
                self.acc(lhs, c_l);
                let c_r = self.builder.instr(Instr::F32Mul(w_bar, lhs_low));
                self.acc(rhs, c_r);
            }

            Instr::F32Powf(lhs, rhs) => {
                let one = self.c_f32(1.0);
                let e = self.builder.instr(Instr::F32Sub(rhs, one));
                let p = self.builder.instr(Instr::F32Powf(lhs, e));
                let t = self.builder.instr(Instr::F32Mul(rhs, p));
                let c_l = self.builder.instr(Instr::F32Mul(w_bar, t));
                self.acc(lhs, c_l);
                let ln = self.builder.instr(Instr::F32Ln(lhs));
                let t = self.builder.instr(Instr::F32Mul(w, ln));
                let cr = self.builder.instr(Instr::F32Mul(w_bar, t));
                self.acc(rhs, cr);
            }

            Instr::F32Powi(lhs, rhs) => {
                let one = self.builder.instr(Instr::I32Const(1));
                let e = self.builder.instr(Instr::I32Sub(rhs, one));
                let p = self.builder.instr(Instr::F32Powi(lhs, e));
                let n = self.builder.instr(Instr::F32FromI32(rhs));
                let t = self.builder.instr(Instr::F32Mul(n, p));
                let c_l = self.builder.instr(Instr::F32Mul(w_bar, t));
                self.acc(lhs, c_l);
            }

            Instr::F32Exp(src) => {
                let c = self.builder.instr(Instr::F32Mul(w_bar, w));
                self.acc(src, c);
            }

            Instr::F32Ln(src) => {
                let c = self.builder.instr(Instr::F32Div(w_bar, src));
                self.acc(src, c);
            }

            Instr::F32Lg(src) => {
                let k = self.c_f32(std::f32::consts::LOG10_E);
                let t = self.builder.instr(Instr::F32Div(k, src));
                let c = self.builder.instr(Instr::F32Mul(w_bar, t));
                self.acc(src, c);
            }

            Instr::F32Sin(src) => {
                let cos = self.builder.instr(Instr::F32Cos(src));
                let c = self.builder.instr(Instr::F32Mul(w_bar, cos));
                self.acc(src, c);
            }

            Instr::F32Cos(src) => {
                let sin = self.builder.instr(Instr::F32Sin(src));
                let m = self.builder.instr(Instr::F32Mul(w_bar, sin));
                let c = self.builder.instr(Instr::F32Neg(m));
                self.acc(src, c);
            }

            Instr::F32Tan(src) => {
                let sq = self.builder.instr(Instr::F32Mul(w, w));
                let one = self.c_f32(1.0);
                let t = self.builder.instr(Instr::F32Add(sq, one));
                let c = self.builder.instr(Instr::F32Mul(w_bar, t));
                self.acc(src, c);
            }

            Instr::F32Cot(src) => {
                let sq = self.builder.instr(Instr::F32Mul(w, w));
                let one = self.c_f32(1.0);
                let t = self.builder.instr(Instr::F32Add(sq, one));
                let m = self.builder.instr(Instr::F32Mul(w_bar, t));
                let c = self.builder.instr(Instr::F32Neg(m));
                self.acc(src, c);
            }
        }
    }

    fn acc(&mut self, target: ValueId, contrib: ValueId) {
        let slot = target.index();
        self.adjoints[slot] = Some(match self.adjoints[slot] {
            None => contrib,
            Some(prev) => self.builder.instr(Instr::F32Add(prev, contrib)),
        });
    }

    /// Returns (1.0, 0.0) if `lhs` <= `rhs`, or (0.0, 1.0) otherwise.
    fn select_weights(&mut self, lhs: ValueId, rhs: ValueId) -> (ValueId, ValueId) {
        let half = self.c_f32(0.5);
        let diff = self.builder.instr(Instr::F32Sub(rhs, lhs));
        let sign = self.builder.instr(Instr::F32Sign(diff));
        let half_sign = self.builder.instr(Instr::F32Mul(sign, half));
        let lhs_low = self.builder.instr(Instr::F32Add(half, half_sign));
        let lhs_high = self.builder.instr(Instr::F32Sub(half, half_sign));
        (lhs_low, lhs_high)
    }

    fn c_f32(&mut self, value: f32) -> ValueId {
        self.builder.instr(Instr::F32Const(value))
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
        let xy = b.instr(Instr::F32Mul(x, y));
        b.instr(Instr::F32Add(xy, x));
        let tape = b.build().unwrap();

        let (x, y) = (3.0f32, 5.0f32);
        assert_eq!(gradient(&tape, &[x, y]), [x * y + x, y + 1.0, x]);
    }

    #[test]
    fn shared_operand() {
        // f = x*x
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        b.instr(Instr::F32Mul(x, x));
        let tape = b.build().unwrap();

        let x = 1.75f32;
        assert_eq!(gradient(&tape, &[x]), [x * x, x + x]);
    }

    #[test]
    fn chain_rule() {
        // f = sin(x) * exp(x)
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let sin = b.instr(Instr::F32Sin(x));
        let exp = b.instr(Instr::F32Exp(x));
        b.instr(Instr::F32Mul(sin, exp));
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
        let yy = b.instr(Instr::F32Mul(y, y));
        b.instr(Instr::F32Min(x, yy));
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
        b.instr(Instr::F32Max(x, x));
        let tape = b.build().unwrap();
        assert_eq!(gradient(&tape, &[3.0]), [3.0, 1.0]);
    }

    #[test]
    fn powi_and_abs() {
        // f = |x^3|
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let n = b.instr(Instr::I32Const(3));
        let cube = b.instr(Instr::F32Powi(x, n));
        b.instr(Instr::F32Abs(cube));
        let tape = b.build().unwrap();

        assert_eq!(gradient(&tape, &[2.0]), [8.0, 12.0]);
        assert_eq!(gradient(&tape, &[-2.0]), [8.0, -12.0]);

        // must be finite everywhere
        assert_eq!(gradient(&tape, &[0.0]), [0.0, 0.0]);
    }

    #[test]
    fn unused_argument() {
        // f = y, x unused
        let mut b = Tape::builder(vec![Type::F32; 2], vec![Type::F32]);
        let y = b.arg(1);
        b.instr(Instr::Copy(y));
        let tape = b.build().unwrap();
        assert_eq!(gradient(&tape, &[3.0, 7.0]), [7.0, 0.0, 1.0]);
    }
}
