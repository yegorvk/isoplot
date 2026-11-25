use std::{fmt::Debug, mem::offset_of};

use crate::{
    artifact::Artifact,
    schema::{CastSchemaSafe, Field, Schema, SchemaTemplate, Slot, Tile},
};
use isoplot_tape::{Inlined, Instr, NewValueId, Tape, TapeBuilder, ValueId, ValueType};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Dual<V, const N: usize> {
    /// The real (0th order) component
    pub value: V,
    /// The 1st-order derivative component
    pub derivative: [V; N],
}

impl<V, const N: usize> Dual<V, N> {
    fn new(value: V, derivative: [V; N]) -> Self {
        Self {
            value,
            derivative: derivative.into(),
        }
    }
}

impl<const N: usize> From<Artifact<[[f32; 1]; N], Dual<[f32; 1], N>>>
    for Artifact<[f32; N], Dual<f32, N>>
{
    fn from(value: Artifact<[[f32; 1]; N], Dual<[f32; 1], N>>) -> Self {
        Self::from_tape(value.into_tape())
    }
}

unsafe impl<V, const TILE_LEN: usize, const N: usize> Schema<TILE_LEN> for Dual<V, N>
where
    V: Schema<TILE_LEN>,
{
    const NUM_FIELDS: usize = (N + 1) * V::NUM_FIELDS;

    fn field(index: usize) -> Field {
        if index < V::NUM_FIELDS {
            V::field(index).offset_by(offset_of!(Self, value))
        } else {
            <[V; N]>::field(index - V::NUM_FIELDS).offset_by(offset_of!(Self, derivative))
        }
    }
}

unsafe impl<const TILE_LEN: usize, const N: usize, V, U> CastSchemaSafe<TILE_LEN, Dual<U, N>>
    for Dual<V, N>
where
    V: Schema<TILE_LEN> + CastSchemaSafe<TILE_LEN, U>,
    U: Schema<TILE_LEN>,
{
}

unsafe impl<const TILE_LEN: usize, const N: usize> SchemaTemplate<TILE_LEN> for Dual<Slot<f32>, N> {
    type With<I32, F32> = Dual<F32, N>;
}

pub(crate) fn lower_autodiff<const TILE_LEN: usize, const N: usize>(
    source: &Artifact<[Tile<f32, TILE_LEN>; N], Tile<f32, TILE_LEN>, TILE_LEN>,
) -> Artifact<[Tile<f32, TILE_LEN>; N], Dual<Tile<f32, TILE_LEN>, N>, TILE_LEN> {
    Artifact::from_tape(Autodiff::new(source.tape()).run())
}

struct Autodiff<'a> {
    source: &'a Tape,
    inlined: Inlined<'a>,
    builder: TapeBuilder,
    adjoints: Vec<Option<NewValueId<f32>>>,
}

impl<'a> Autodiff<'a> {
    fn new(source: &'a Tape) -> Self {
        assert!(
            source.param_types().iter().all(|&ty| ty == ValueType::F32),
            "autodiff arguments must be f32"
        );

        assert_eq!(
            source.num_results(),
            1,
            "only scalar-valued functions are supported"
        );

        let num_params = source.num_params();
        let num_values = num_params + source.instrs().len();

        let mut builder = Tape::builder(
            vec![ValueType::F32; num_params],
            vec![ValueType::F32; 1 + num_params],
        );
        let inlined = builder.extend(source);

        Self {
            source,
            inlined,
            builder,
            adjoints: vec![None; num_values],
        }
    }

    fn run(mut self) -> Tape {
        let num_params = self.source.num_params();

        let seed = self.builder.f32_const(1.0);
        *self.adjoints.last_mut().unwrap() = Some(seed);

        for r in self.source.iter_ref().rev() {
            let Some(w_bar) = self.adjoints[r.index()] else {
                continue;
            };
            let w = self
                .inlined
                .convert(r.id().typed().expect("adjoint of a non-f32 value"));
            self.propagate(r.instr(), w, w_bar);
        }

        self.builder.copy_f32(self.inlined.result(0));

        for i in 0..num_params {
            match self.adjoints[i] {
                None => self.builder.f32_const(0.0),
                Some(v) => self.builder.copy_f32(v),
            };
        }

        self.builder.build().unwrap()
    }

    fn propagate(&mut self, instr: Instr, w: NewValueId<f32>, w_bar: NewValueId<f32>) {
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
                let le = self.builder.f32_le(x, y);
                let zero = self.builder.f32_const(0.0);
                let c_l = self.builder.f32_sel(le, w_bar, zero);
                self.acc(lhs, c_l);
                let c_r = self.builder.f32_sel(le, zero, w_bar);
                self.acc(rhs, c_r);
            }

            Instr::F32Max(lhs, rhs) => {
                let (x, y) = (self.inlined.convert(lhs), self.inlined.convert(rhs));
                let le = self.builder.f32_le(x, y);
                let zero = self.builder.f32_const(0.0);
                let c_l = self.builder.f32_sel(le, zero, w_bar);
                self.acc(lhs, c_l);
                let c_r = self.builder.f32_sel(le, w_bar, zero);
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

    fn acc(&mut self, target: ValueId<f32>, contrib: NewValueId<f32>) {
        let slot = target.index();
        self.adjoints[slot] = Some(match self.adjoints[slot] {
            None => contrib,
            Some(prev) => self.builder.f32_add(prev, contrib),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::Fallback,
        kernel::{Evaluate, Kernel},
    };

    fn gradient<const N: usize>(tape: &Tape, params: impl Into<[f32; N]>) -> Dual<f32, N> {
        let tape = Autodiff::new(tape).run();

        assert_eq!(tape.num_params(), N);
        assert_eq!(tape.num_results(), 1 + N);

        Fallback::new(Artifact::from_tape(tape))
            .evaluator()
            .evaluate(params.into())
    }

    #[test]
    fn polynomial() {
        let mut b = Tape::builder(vec![ValueType::F32; 2], vec![ValueType::F32]);

        let x = b.param(0);
        let y = b.param(1);

        let x_mul_y = b.f32_mul(x, y);

        b.f32_add(x_mul_y, x);
        let tape = b.build().unwrap();

        let [v_x, v_y] = [3.0f32, 5.0f32];
        assert_eq!(
            gradient(&tape, [v_x, v_y]),
            Dual::new(v_x * v_y + v_x, [v_y + 1.0, v_x])
        );
    }

    #[test]
    fn shared_operand() {
        // f = x*x
        let mut b = Tape::builder(vec![ValueType::F32], vec![ValueType::F32]);
        let x = b.param(0);
        b.f32_mul(x, x);

        let tape = b.build().unwrap();

        let x = 1.75f32;
        assert_eq!(gradient(&tape, [x]), Dual::new(x * x, [x + x]));
    }

    #[test]
    fn chain_rule() {
        // f = sin(x) * exp(x)
        let mut b = Tape::builder(vec![ValueType::F32], vec![ValueType::F32]);
        let x = b.param(0);
        let sin = b.f32_sin(x);
        let exp = b.f32_exp(x);
        b.f32_mul(sin, exp);

        let tape = b.build().unwrap();

        let x = 0.7f32;
        assert_eq!(
            gradient(&tape, [x]),
            Dual::new(x.sin() * x.exp(), [x.sin() * x.exp() + x.exp() * x.cos()])
        );
    }

    #[test]
    fn min_branch() {
        // f = min(x, y*y)
        let mut b = Tape::builder(vec![ValueType::F32; 2], vec![ValueType::F32]);
        let x = b.param(0);
        let y = b.param(1);
        let yy = b.f32_mul(y, y);
        b.f32_min(x, yy);

        let tape = b.build().unwrap();

        assert_eq!(gradient(&tape, [1.0, 2.0]), Dual::new(1.0, [1.0, 0.0]));
        assert_eq!(gradient(&tape, [5.0, 2.0]), Dual::new(4.0, [0.0, 4.0]));
        assert_eq!(gradient(&tape, [4.0, 2.0]), Dual::new(4.0, [1.0, 0.0]));
    }

    #[test]
    fn powi_abs() {
        // f = |x^3|
        let mut b = Tape::builder(vec![ValueType::F32], vec![ValueType::F32]);
        let x = b.param(0);
        let n = b.i32_const(3);
        let cube = b.f32_powi(x, n);
        b.f32_abs(cube);

        let tape = b.build().unwrap();

        assert_eq!(gradient(&tape, [2.0]), Dual::new(8.0, [12.0]));
        assert_eq!(gradient(&tape, [-2.0]), Dual::new(8.0, [-12.0]));

        let m = crate::MIN_F32_MAGNITUDE;
        assert_eq!(
            gradient(&tape, [0.0]),
            Dual::new(m.powi(3), [3.0 * m.powi(2)])
        );
    }

    #[test]
    fn gradient_branch() {
        // f = if x < 2 { x*x } else { x }
        let mut b = Tape::builder(vec![ValueType::F32], vec![ValueType::F32]);
        let x = b.param(0);
        let two = b.f32_const(2.0);
        let lt = b.f32_lt(x, two);
        let sq = b.f32_mul(x, x);
        b.f32_sel(lt, sq, x);

        let tape = b.build().unwrap();

        assert_eq!(gradient(&tape, [1.0]), Dual::new(1.0, [2.0]));
        assert_eq!(gradient(&tape, [3.0]), Dual::new(3.0, [1.0]));
    }

    #[test]
    fn unused_param() {
        // f = y, x unused
        let mut b = Tape::builder(vec![ValueType::F32; 2], vec![ValueType::F32]);
        let y = b.param(1);
        b.copy_f32(y);
        let tape = b.build().unwrap();
        assert_eq!(gradient(&tape, [3.0, 7.0]), Dual::new(7.0, [0.0, 1.0]));
    }
}
