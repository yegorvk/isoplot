use std::{array, fmt::Debug, mem::offset_of};

use crate::{
    artifact::Artifact,
    schema::{CastSchemaSafe, Field, Schema, SchemaTemplate, Slot, Tile},
};
use isoplot_tape::{Instr, NewValueId, Tape, TapeBuilder, ValueId, ValueType};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Dual<V, const N: usize> {
    /// The real (0th order) component
    pub value: V,
    /// The 1st-order derivative component
    pub der: [V; N],
}

impl<V, const N: usize> Dual<V, N> {
    fn new(value: V, der: [V; N]) -> Self {
        Self { value, der: der }
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
            <[V; N]>::field(index - V::NUM_FIELDS).offset_by(offset_of!(Self, der))
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
    Artifact::from_tape(Translator::<N>::new(source.tape()).translate())
}

#[derive(Copy, Clone)]
struct NewDualId<const N: usize> {
    value: NewValueId<f32>,
    der: [NewValueId<f32>; N],
}

impl<const N: usize> NewDualId<N> {
    fn new(value: NewValueId<f32>, der: [NewValueId<f32>; N]) -> Self {
        Self { value, der }
    }

    fn from_fn<F>(value: NewValueId<f32>, mut f: F) -> Self
    where
        F: FnMut(usize) -> NewValueId<f32>,
    {
        Self {
            value,
            der: array::from_fn(|i| f(i)),
        }
    }

    fn parts(self) -> (NewValueId<f32>, [NewValueId<f32>; N]) {
        (self.value, self.der)
    }
}

#[derive(Copy, Clone)]
enum ValueSlot<const N: usize> {
    I32(NewValueId<i32>),
    Bool(NewValueId<bool>),
    F32(NewDualId<N>),
}

impl<const N: usize> ValueSlot<N> {
    fn unwrap_i32(self) -> NewValueId<i32> {
        match self {
            Self::I32(value) => value,
            _ => panic!("not an i32 value"),
        }
    }

    fn unwrap_bool(self) -> NewValueId<bool> {
        match self {
            Self::Bool(value) => value,
            _ => panic!("not a bool value"),
        }
    }

    fn unwrap_f32(self) -> NewDualId<N> {
        match self {
            Self::F32(value) => value,
            _ => panic!("not an f32 value"),
        }
    }
}

impl<const N: usize> From<NewValueId<i32>> for ValueSlot<N> {
    fn from(value: NewValueId<i32>) -> Self {
        Self::I32(value)
    }
}

impl<const N: usize> From<NewValueId<bool>> for ValueSlot<N> {
    fn from(value: NewValueId<bool>) -> Self {
        Self::Bool(value)
    }
}

impl<const N: usize> From<NewDualId<N>> for ValueSlot<N> {
    fn from(value: NewDualId<N>) -> Self {
        Self::F32(value)
    }
}

struct Translator<'a, const N: usize> {
    source: &'a Tape,
    builder: TapeBuilder,
    slots: Vec<ValueSlot<N>>,
}

impl<'a, const N: usize> Translator<'a, N> {
    fn new(source: &'a Tape) -> Self {
        assert!(
            source.param_types().iter().all(|&ty| ty == ValueType::F32),
            "all parameters must be of type `f32`"
        );

        assert_eq!(
            source.num_results(),
            1,
            "only a single return value is supported"
        );

        let num_params = source.num_params();

        let mut builder = Tape::builder(
            vec![ValueType::F32; num_params],
            vec![ValueType::F32; 1 + num_params],
        );

        let c_0 = builder.f32_const(0.0);
        let c_1 = builder.f32_const(1.0);

        let slots = Vec::from_fn(source.num_params(), |i| {
            NewDualId::from_fn(builder.param(i), |j| if i == j { c_1 } else { c_0 }).into()
        });

        Self {
            source,
            builder,
            slots,
        }
    }

    fn translate(mut self) -> Tape {
        for instr in self.source.instrs() {
            let slot = self.translate_instr(*instr);
            self.slots.push(slot);
        }

        for slot in self.slots.iter().rev().take(self.source.num_results()) {
            match *slot {
                ValueSlot::I32(value) => {
                    self.builder.copy_i32(value);
                }
                ValueSlot::Bool(value) => {
                    self.builder.copy_bool(value);
                }
                ValueSlot::F32(value) => {
                    self.builder.copy_f32(value.value);
                    for c in value.der {
                        self.builder.copy_f32(c);
                    }
                }
            };
        }

        self.builder.build().unwrap()
    }

    fn translate_instr(&mut self, instr: Instr) -> ValueSlot<N> {
        match instr {
            // Constants
            Instr::I32Const(value) => self.builder.i32_const(value).into(),
            Instr::BoolConst(value) => self.builder.bool_const(value).into(),
            Instr::F32Const(value) => NewDualId::new(
                self.builder.f32_const(value),
                [self.builder.f32_const(0.0); N],
            )
            .into(),

            // Copies
            Instr::CopyI32(value) => self.v_i32(value).into(),
            Instr::CopyBool(value) => self.v_bool(value).into(),
            Instr::CopyF32(value) => self.v_f32(value).into(),

            // Integer binary math operations
            Instr::I32Add(lhs, rhs) => self
                .builder
                .i32_add(self.v_i32(lhs), self.v_i32(rhs))
                .into(),
            Instr::I32Sub(lhs, rhs) => self
                .builder
                .i32_sub(self.v_i32(lhs), self.v_i32(rhs))
                .into(),
            Instr::I32Mul(lhs, rhs) => self
                .builder
                .i32_mul(self.v_i32(lhs), self.v_i32(rhs))
                .into(),

            // Integer comparison operations
            Instr::I32Eq(lhs, rhs) => self.builder.i32_eq(self.v_i32(lhs), self.v_i32(rhs)).into(),
            Instr::I32Ne(lhs, rhs) => self.builder.i32_ne(self.v_i32(lhs), self.v_i32(rhs)).into(),
            Instr::I32Lt(lhs, rhs) => self.builder.i32_lt(self.v_i32(lhs), self.v_i32(rhs)).into(),
            Instr::I32Gt(lhs, rhs) => self.builder.i32_gt(self.v_i32(lhs), self.v_i32(rhs)).into(),
            Instr::I32Le(lhs, rhs) => self.builder.i32_le(self.v_i32(lhs), self.v_i32(rhs)).into(),
            Instr::I32Ge(lhs, rhs) => self.builder.i32_ge(self.v_i32(lhs), self.v_i32(rhs)).into(),

            // Integer select
            Instr::I32Sel(cond, v_t, v_f) => self
                .builder
                .i32_sel(self.v_bool(cond), self.v_i32(v_t), self.v_i32(v_f))
                .into(),

            // Cast operations
            Instr::I32FromBool(value) => self.builder.i32_from_bool(self.v_bool(value)).into(),
            Instr::F32FromI32(value) => NewDualId::new(
                self.builder.f32_from_i32(self.v_i32(value)),
                [self.builder.f32_const(0.0); N],
            )
            .into(),
            Instr::F32FromBool(value) => NewDualId::new(
                self.builder.f32_from_bool(self.v_bool(value)),
                [self.builder.f32_const(0.0); N],
            )
            .into(),

            // Logical operations
            Instr::Not(value) => self.builder.not(self.v_bool(value)).into(),
            Instr::And(lhs, rhs) => self.builder.and(self.v_bool(lhs), self.v_bool(rhs)).into(),
            Instr::Or(lhs, rhs) => self.builder.or(self.v_bool(lhs), self.v_bool(rhs)).into(),
            Instr::Xor(lhs, rhs) => self.builder.xor(self.v_bool(lhs), self.v_bool(rhs)).into(),

            // Floating-point unary operations
            Instr::F32Neg(value) => {
                let (a, b) = self.v_f32(value).parts();
                NewDualId::from_fn(self.builder.f32_neg(a), |i| self.builder.f32_neg(b[i])).into()
            }
            Instr::F32Abs(value) => {
                let (a, b) = self.v_f32(value).parts();

                let c_0 = self.builder.f32_const(0.0);
                let c_1 = self.builder.f32_const(1.0);
                let c_neg_1 = self.builder.f32_const(-1.0);

                let a_ge_0 = self.builder.f32_ge(a, c_0);
                let coef = self.builder.f32_sel(a_ge_0, c_1, c_neg_1);

                NewDualId::from_fn(self.builder.f32_abs(a), |i| {
                    self.builder.f32_mul(b[i], coef)
                })
                .into()
            }
            Instr::F32Sign(value) => NewDualId::new(
                self.builder.f32_sign(self.v_f32(value).value),
                [self.builder.f32_const(0.0); N],
            )
            .into(),
            Instr::F32Floor(value) => NewDualId::new(
                self.builder.f32_floor(self.v_f32(value).value),
                [self.builder.f32_const(0.0); N],
            )
            .into(),
            Instr::F32Exp(value) => {
                let (a, b) = self.v_f32(value).parts();
                let exp_a = self.builder.f32_exp(a);
                NewDualId::from_fn(exp_a, |i| self.builder.f32_mul(exp_a, b[i])).into()
            }
            Instr::F32Ln(value) => {
                let (a, b) = self.v_f32(value).parts();

                let c_1 = self.builder.f32_const(1.0);
                let der = self.builder.f32_div(c_1, a);

                NewDualId::from_fn(self.builder.f32_ln(a), |i| self.builder.f32_mul(der, b[i]))
                    .into()
            }
            Instr::F32Lg(value) => {
                let (a, b) = self.v_f32(value).parts();

                let c_1 = self.builder.f32_const(1.0);
                let c_ln_10 = self.builder.f32_const(10f32.ln());

                let a_mul_ln_10 = self.builder.f32_mul(a, c_ln_10);
                let der = self.builder.f32_div(c_1, a_mul_ln_10);

                NewDualId::from_fn(self.builder.f32_lg(a), |i| self.builder.f32_mul(der, b[i]))
                    .into()
            }
            Instr::F32Sin(value) => {
                let (a, b) = self.v_f32(value).parts();
                let der = self.builder.f32_cos(a);
                NewDualId::from_fn(self.builder.f32_sin(a), |i| self.builder.f32_mul(der, b[i]))
                    .into()
            }
            Instr::F32Cos(value) => {
                let (a, b) = self.v_f32(value).parts();

                let sin_a = self.builder.f32_sin(a);
                let der = self.builder.f32_neg(sin_a);

                NewDualId::from_fn(self.builder.f32_cos(a), |i| self.builder.f32_mul(der, b[i]))
                    .into()
            }
            Instr::F32Tan(value) => {
                let (a, b) = self.v_f32(value).parts();

                let c_1 = self.builder.f32_const(1.0);
                let tan_a = self.builder.f32_tan(a);
                let tan_a_pow_2 = self.builder.f32_mul(tan_a, tan_a);
                let der = self.builder.f32_add(tan_a_pow_2, c_1);

                NewDualId::from_fn(tan_a, |i| self.builder.f32_mul(der, b[i])).into()
            }
            Instr::F32Cot(value) => {
                let (a, b) = self.v_f32(value).parts();

                let c_1 = self.builder.f32_const(1.0);
                let cot_a = self.builder.f32_cot(a);
                let cot_a_pow_2 = self.builder.f32_mul(cot_a, cot_a);
                let sum = self.builder.f32_add(cot_a_pow_2, c_1);
                let der = self.builder.f32_neg(sum);

                NewDualId::from_fn(cot_a, |i| self.builder.f32_mul(der, b[i])).into()
            }

            // Floating-point binary math operations
            Instr::F32Add(lhs, rhs) => {
                let (a, b) = self.v_f32(lhs).parts();
                let (c, d) = self.v_f32(rhs).parts();
                NewDualId::from_fn(self.builder.f32_add(a, c), |i| {
                    self.builder.f32_add(b[i], d[i])
                })
                .into()
            }
            Instr::F32Sub(lhs, rhs) => {
                let (a, b) = self.v_f32(lhs).parts();
                let (c, d) = self.v_f32(rhs).parts();
                NewDualId::from_fn(self.builder.f32_sub(a, c), |i| {
                    self.builder.f32_sub(b[i], d[i])
                })
                .into()
            }
            Instr::F32Mul(lhs, rhs) => {
                let (a, b) = self.v_f32(lhs).parts();
                let (c, d) = self.v_f32(rhs).parts();
                NewDualId::from_fn(self.builder.f32_mul(a, c), |i| {
                    let a_d = self.builder.f32_mul(a, d[i]);
                    let c_b = self.builder.f32_mul(c, b[i]);
                    self.builder.f32_add(a_d, c_b)
                })
                .into()
            }
            Instr::F32Div(lhs, rhs) => {
                let (a, b) = self.v_f32(lhs).parts();
                let (c, d) = self.v_f32(rhs).parts();

                let c_1 = self.builder.f32_const(1.0);
                let recip_c = self.builder.f32_div(c_1, c);

                let a_div_c = self.builder.f32_div(a, c);
                let a_div_c_2 = self.builder.f32_mul(a_div_c, recip_c);

                NewDualId::from_fn(a_div_c, |i| {
                    let b_div_c = self.builder.f32_mul(b[i], recip_c);
                    let a_d_div_c_2 = self.builder.f32_mul(d[i], a_div_c_2);
                    self.builder.f32_sub(b_div_c, a_d_div_c_2)
                })
                .into()
            }
            Instr::F32Powi(lhs, rhs) => {
                let (a, b) = self.v_f32(lhs).parts();
                let n = self.v_i32(rhs);

                let c_1 = self.builder.i32_const(1);
                let n_sub_1 = self.builder.i32_sub(n, c_1);
                let a_pow_n_sub_1 = self.builder.f32_powi(a, n_sub_1);

                let n_f = self.builder.f32_from_i32(n);
                let der = self.builder.f32_mul(a_pow_n_sub_1, n_f);

                NewDualId::from_fn(self.builder.f32_powi(a, n), |i| {
                    self.builder.f32_mul(der, b[i])
                })
                .into()
            }
            Instr::F32Powf(lhs, rhs) => {
                let (a, b) = self.v_f32(lhs).parts();
                let (c, d) = self.v_f32(rhs).parts();

                let a_pow_c = self.builder.f32_powf(a, c);
                let ln_a = self.builder.f32_ln(a);
                let c_div_a = self.builder.f32_div(c, a);

                NewDualId::from_fn(a_pow_c, |i| {
                    let d_ln_a = self.builder.f32_mul(d[i], ln_a);
                    let c_b_div_a = self.builder.f32_mul(b[i], c_div_a);
                    let sum = self.builder.f32_add(d_ln_a, c_b_div_a);
                    self.builder.f32_mul(a_pow_c, sum)
                })
                .into()
            }
            Instr::F32Min(lhs, rhs) => {
                let (a, b) = self.v_f32(lhs).parts();
                let (c, d) = self.v_f32(rhs).parts();
                let a_lt_c = self.builder.f32_lt(a, c);
                NewDualId::from_fn(self.builder.f32_min(a, c), |i| {
                    self.builder.f32_sel(a_lt_c, b[i], d[i])
                })
                .into()
            }
            Instr::F32Max(lhs, rhs) => {
                let (a, b) = self.v_f32(lhs).parts();
                let (c, d) = self.v_f32(rhs).parts();
                let a_gt_c = self.builder.f32_gt(a, c);
                NewDualId::from_fn(self.builder.f32_max(a, c), |i| {
                    self.builder.f32_sel(a_gt_c, b[i], d[i])
                })
                .into()
            }

            // Floating-point comparison operations
            Instr::F32Eq(lhs, rhs) => self
                .builder
                .f32_eq(self.v_f32(lhs).value, self.v_f32(rhs).value)
                .into(),
            Instr::F32Ne(lhs, rhs) => self
                .builder
                .f32_ne(self.v_f32(lhs).value, self.v_f32(rhs).value)
                .into(),
            Instr::F32Lt(lhs, rhs) => self
                .builder
                .f32_lt(self.v_f32(lhs).value, self.v_f32(rhs).value)
                .into(),
            Instr::F32Gt(lhs, rhs) => self
                .builder
                .f32_gt(self.v_f32(lhs).value, self.v_f32(rhs).value)
                .into(),
            Instr::F32Le(lhs, rhs) => self
                .builder
                .f32_le(self.v_f32(lhs).value, self.v_f32(rhs).value)
                .into(),
            Instr::F32Ge(lhs, rhs) => self
                .builder
                .f32_ge(self.v_f32(lhs).value, self.v_f32(rhs).value)
                .into(),

            // Floating-point select
            Instr::F32Sel(cond, v_t, v_f) => {
                let cond = self.v_bool(cond);

                let (a, b) = self.v_f32(v_t).parts();
                let (c, d) = self.v_f32(v_f).parts();

                NewDualId::from_fn(self.builder.f32_sel(cond, a, c), |i| {
                    self.builder.f32_sel(cond, b[i], d[i])
                })
                .into()
            }
        }
    }

    fn v_i32(&self, id: ValueId<i32>) -> NewValueId<i32> {
        self.slots[id.index()].unwrap_i32()
    }

    fn v_bool(&self, id: ValueId<bool>) -> NewValueId<bool> {
        self.slots[id.index()].unwrap_bool()
    }

    fn v_f32(&self, id: ValueId<f32>) -> NewDualId<N> {
        self.slots[id.index()].unwrap_f32()
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
        let tape = Translator::<N>::new(tape).translate();

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
        assert_eq!(gradient(&tape, [4.0, 2.0]), Dual::new(4.0, [0.0, 4.0]));
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
