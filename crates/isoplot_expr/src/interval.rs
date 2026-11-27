use std::{
    fmt::{self, Debug},
    iter,
};

use crate::{
    MIN_F32_MAGNITUDE,
    artifact::Artifact,
    schema::{Field, Schema, SchemaWith},
};
use isoplot_tape::{Instr, NewValueId, Tape, TapeBuilder, ValueId, ValueRepr, ValueType};

/// A closed and bounded interval
#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(transparent)]
pub struct Interval<S> {
    min_max: [S; 2],
}

impl<S: Copy> Interval<S> {
    pub const fn min(&self) -> S {
        self.min_max[0]
    }

    pub const fn max(&self) -> S {
        self.min_max[1]
    }
}

impl Interval<f32> {
    pub const fn new(min: f32, max: f32) -> Self {
        assert!(min <= max);
        Self {
            min_max: [min, max],
        }
    }

    pub const fn is_finite(self) -> bool {
        self.min_max[0].is_finite() && self.min_max[1].is_finite()
    }

    pub const fn contains_zero(self) -> bool {
        self.min_max[0] <= 0.0 && self.min_max[1] >= 0.0
    }

    pub const fn center(self) -> f32 {
        (self.min_max[0] + self.min_max[1]) * 0.5
    }
}

impl<const N: usize> Interval<[f32; N]> {
    pub fn new(min: [f32; N], max: [f32; N]) -> Self {
        assert!(iter::zip(min, max).all(|(min, max)| min <= max));
        Self {
            min_max: [min, max],
        }
    }
}

unsafe impl<S: Schema<TILE_LEN>, const TILE_LEN: usize> Schema<TILE_LEN> for Interval<S> {
    const NUM_FIELDS: usize = <[S; 2]>::NUM_FIELDS;

    fn field(index: usize) -> Field {
        <[S; 2]>::field(index)
    }
}

impl<const N: usize, const M: usize>
    From<Artifact<[Interval<[f32; 1]>; N], [Interval<[f32; 1]>; M]>>
    for Artifact<[Interval<f32>; N], [Interval<f32>; M]>
{
    fn from(value: Artifact<[Interval<[f32; 1]>; N], [Interval<[f32; 1]>; M]>) -> Self {
        Self::from_tape(value.into_tape())
    }
}

impl<S: Copy + Debug> fmt::Debug for Interval<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Interval({:?}, {:?})", self.min(), self.max())
    }
}

pub(crate) fn lower_interval<const TILE_LEN: usize, V, Params, Ret>(
    artifact: &Artifact<Params::With<!, V>, Ret::With<!, V>, TILE_LEN>,
) -> Artifact<Params::With<!, Interval<V>>, Ret::With<!, Interval<V>>, TILE_LEN>
where
    V: Schema<TILE_LEN>,
    Params: SchemaWith<TILE_LEN, !, V> + SchemaWith<TILE_LEN, !, Interval<V>>,
    Ret: SchemaWith<TILE_LEN, !, V> + SchemaWith<TILE_LEN, !, Interval<V>>,
{
    Artifact::from_tape(translate(artifact.tape()))
}

fn translate(source: &Tape) -> Tape {
    Translator::new(source).translate()
}

#[derive(Copy, Clone)]
struct NewRangeId<T> {
    min: NewValueId<T>,
    max: NewValueId<T>,
}

impl<T: Copy> NewRangeId<T> {
    fn new(min: NewValueId<T>, max: NewValueId<T>) -> Self {
        Self { min, max }
    }

    fn map<U: Copy, F>(&self, mut f: F) -> NewRangeId<U>
    where
        F: FnMut(NewValueId<T>) -> NewValueId<U>,
    {
        NewRangeId::new(f(self.min), f(self.max))
    }

    fn parts(&self) -> [NewValueId<T>; 2] {
        [self.min, self.max]
    }
}

#[derive(Copy, Clone)]
enum ValueSlot {
    I32(NewValueId<i32>),
    Bool(NewRangeId<bool>),
    F32(NewRangeId<f32>),
}

impl ValueSlot {
    fn unwrap_i32(self) -> NewValueId<i32> {
        match self {
            Self::I32(value) => value,
            _ => panic!("not an i32 value"),
        }
    }

    fn unwrap_bool(self) -> NewRangeId<bool> {
        match self {
            Self::Bool(value) => value,
            _ => panic!("not a boolean value"),
        }
    }

    fn unwrap_f32(self) -> NewRangeId<f32> {
        match self {
            Self::F32(value) => value,
            _ => panic!("not a f32 value"),
        }
    }
}

impl From<NewValueId<i32>> for ValueSlot {
    fn from(value: NewValueId<i32>) -> Self {
        Self::I32(value)
    }
}

impl From<NewRangeId<bool>> for ValueSlot {
    fn from(value: NewRangeId<bool>) -> Self {
        Self::Bool(value)
    }
}

impl From<NewRangeId<f32>> for ValueSlot {
    fn from(value: NewRangeId<f32>) -> Self {
        Self::F32(value)
    }
}

struct Translator<'a> {
    source: &'a Tape,
    builder: TapeBuilder,
    slots: Vec<ValueSlot>,
    poison: Vec<NewValueId<bool>>,
}

impl<'a> Translator<'a> {
    fn new(source: &'a Tape) -> Self {
        assert!(
            source.param_types().iter().all(|&ty| ty == ValueType::F32),
            "unsupported parameter type"
        );

        assert!(
            source.result_types().iter().all(|&ty| ty == ValueType::F32),
            "unsupported result type"
        );

        let num_params = source.num_params();

        let mut builder = Tape::builder(
            vec![ValueType::F32; 2 * num_params],
            (0..source.num_results())
                .flat_map(|_| [ValueType::F32, ValueType::F32])
                .collect(),
        );

        let slots = (0..num_params)
            .map(|i| {
                ValueSlot::F32(NewRangeId::new(
                    builder.param(2 * i),
                    builder.param(2 * i + 1),
                ))
            })
            .collect();

        let poison = vec![builder.bool_const(false); num_params];

        Self {
            source,
            builder,
            slots,
            poison,
        }
    }

    fn translate(mut self) -> Tape {
        for instr in self.source.instrs() {
            let (slot, poison) = self.translate_instr(*instr);
            self.slots.push(slot.into());
            self.poison.push(poison);
        }

        let c_nan = self.builder.f32_const(f32::NAN);

        iter::zip(self.slots.iter().copied(), self.poison.iter().copied())
            .rev()
            .take(self.source.num_results())
            .for_each(|(slot, is_poison)| {
                let ValueSlot::F32(value) = slot else {
                    unreachable!("only `f32` results are supported");
                };
                self.builder.f32_sel(is_poison, c_nan, value.min);
                self.builder.copy_f32(value.max);
            });

        self.builder.build().unwrap()
    }

    fn translate_instr(&mut self, instr: Instr) -> (ValueSlot, NewValueId<bool>) {
        match instr {
            Instr::I32Const(value) => (
                self.builder.i32_const(value).into(),
                self.builder.bool_const(false),
            ),

            Instr::BoolConst(value) => {
                let c = self.builder.bool_const(value);
                (NewRangeId::new(c, c).into(), self.builder.bool_const(false))
            }

            Instr::F32Const(value) => {
                let c = self.builder.f32_const(value);
                (NewRangeId::new(c, c).into(), self.builder.bool_const(false))
            }

            Instr::CopyI32(value) => (
                self.builder.copy_i32(self.v_i32(value)).into(),
                self.builder.bool_const(false),
            ),

            Instr::CopyBool(value) => {
                let [a, b] = self.v_bool(value).parts();
                (
                    NewRangeId::new(self.builder.copy_bool(a), self.builder.copy_bool(b)).into(),
                    self.is_poison(value),
                )
            }

            Instr::CopyF32(value) => {
                let [a, b] = self.v_f32(value).parts();
                (
                    NewRangeId::new(self.builder.copy_f32(a), self.builder.copy_f32(b)).into(),
                    self.is_poison(value),
                )
            }

            Instr::I32Add(lhs, rhs) => (
                self.builder
                    .i32_add(self.v_i32(lhs), self.v_i32(rhs))
                    .into(),
                self.builder.bool_const(false),
            ),

            Instr::I32Sub(lhs, rhs) => (
                self.builder
                    .i32_sub(self.v_i32(lhs), self.v_i32(rhs))
                    .into(),
                self.builder.bool_const(false),
            ),

            Instr::I32Mul(lhs, rhs) => (
                self.builder
                    .i32_mul(self.v_i32(lhs), self.v_i32(rhs))
                    .into(),
                self.builder.bool_const(false),
            ),

            Instr::I32Eq(lhs, rhs) => {
                let v = self.builder.i32_eq(self.v_i32(lhs), self.v_i32(rhs));
                (NewRangeId::new(v, v).into(), self.builder.bool_const(false))
            }

            Instr::I32Ne(lhs, rhs) => {
                let v = self.builder.i32_ne(self.v_i32(lhs), self.v_i32(rhs));
                (NewRangeId::new(v, v).into(), self.builder.bool_const(false))
            }

            Instr::I32Lt(lhs, rhs) => {
                let v = self.builder.i32_lt(self.v_i32(lhs), self.v_i32(rhs));
                (NewRangeId::new(v, v).into(), self.builder.bool_const(false))
            }

            Instr::I32Gt(lhs, rhs) => {
                let v = self.builder.i32_gt(self.v_i32(lhs), self.v_i32(rhs));
                (NewRangeId::new(v, v).into(), self.builder.bool_const(false))
            }

            Instr::I32Le(lhs, rhs) => {
                let v = self.builder.i32_le(self.v_i32(lhs), self.v_i32(rhs));
                (NewRangeId::new(v, v).into(), self.builder.bool_const(false))
            }

            Instr::I32Ge(lhs, rhs) => {
                let v = self.builder.i32_ge(self.v_i32(lhs), self.v_i32(rhs));
                (NewRangeId::new(v, v).into(), self.builder.bool_const(false))
            }

            Instr::I32Sel(cond, v_t, v_f) => {
                let [all, any] = self.v_bool(cond).parts();
                (
                    self.builder
                        .i32_sel(all, self.v_i32(v_t), self.v_i32(v_f))
                        .into(),
                    self.builder.xor(all, any),
                )
            }

            Instr::I32FromBool(value) => {
                let [all, any] = self.v_bool(value).parts();
                (
                    self.builder.i32_from_bool(all).into(),
                    self.builder.xor(all, any),
                )
            }

            Instr::Not(value) => {
                let [all, any] = self.v_bool(value).parts();
                (
                    NewRangeId::new(self.builder.not(any), self.builder.not(all)).into(),
                    self.is_poison(value),
                )
            }

            Instr::And(lhs, rhs) => {
                let [a, b] = self.v_bool(lhs).parts();
                let [c, d] = self.v_bool(rhs).parts();
                (
                    NewRangeId::new(self.builder.and(a, c), self.builder.and(b, d)).into(),
                    self.builder.or(self.is_poison(lhs), self.is_poison(rhs)),
                )
            }

            Instr::Or(lhs, rhs) => {
                let [a, b] = self.v_bool(lhs).parts();
                let [c, d] = self.v_bool(rhs).parts();
                (
                    NewRangeId::new(self.builder.or(a, c), self.builder.or(b, d)).into(),
                    self.builder.or(self.is_poison(lhs), self.is_poison(rhs)),
                )
            }

            Instr::Xor(_, _) => {
                todo!()
            }

            Instr::F32FromI32(value) => {
                let n = self.v_i32(value);
                (
                    NewRangeId::new(self.builder.f32_from_i32(n), self.builder.f32_from_i32(n))
                        .into(),
                    self.builder.bool_const(false),
                )
            }

            Instr::F32FromBool(value) => {
                let [a, b] = self.v_bool(value).parts();
                (
                    NewRangeId::new(self.builder.f32_from_bool(a), self.builder.f32_from_bool(b))
                        .into(),
                    self.is_poison(value),
                )
            }

            Instr::F32Add(lhs, rhs) => {
                let [a, b] = self.v_f32(lhs).parts();
                let [c, d] = self.v_f32(rhs).parts();
                (
                    NewRangeId::new(self.builder.f32_add(a, c), self.builder.f32_add(b, d)).into(),
                    self.builder.or(self.is_poison(lhs), self.is_poison(rhs)),
                )
            }

            Instr::F32Sub(lhs, rhs) => {
                let [a, b] = self.v_f32(lhs).parts();
                let [c, d] = self.v_f32(rhs).parts();
                (
                    NewRangeId::new(self.builder.f32_sub(a, d), self.builder.f32_sub(b, c)).into(),
                    self.builder.or(self.is_poison(lhs), self.is_poison(rhs)),
                )
            }

            Instr::F32Mul(lhs, rhs) => (
                self.mul_interval(self.v_f32(lhs), self.v_f32(rhs)).into(),
                self.builder.or(self.is_poison(lhs), self.is_poison(rhs)),
            ),

            Instr::F32Div(lhs, rhs) => {
                let [c, d] = self.v_f32(rhs).parts();

                let c_0 = self.builder.f32_const(0.0);
                let c_1 = self.builder.f32_const(1.0);

                let c_le_0 = self.builder.f32_le(c, c_0);
                let d_ge_0 = self.builder.f32_ge(d, c_0);

                let d_zero = self.builder.and(c_le_0, d_ge_0);

                let recip_c = self.builder.f32_div(c_1, c);
                let recip_d = self.builder.f32_div(c_1, d);

                let min_recip = self.builder.f32_min(recip_c, recip_d);
                let max_recip = self.builder.f32_max(recip_c, recip_d);

                let inherits_poison = self.builder.or(self.is_poison(lhs), self.is_poison(rhs));

                (
                    self.mul_interval(self.v_f32(lhs), NewRangeId::new(min_recip, max_recip))
                        .into(),
                    self.builder.or(d_zero, inherits_poison),
                )
            }

            Instr::F32Neg(value) => {
                let [a, b] = self.v_f32(value).parts();
                (
                    NewRangeId::new(self.builder.f32_neg(b), self.builder.f32_neg(a)).into(),
                    self.is_poison(value),
                )
            }

            Instr::F32Sign(value) => {
                let [a, b] = self.v_f32(value).parts();
                (
                    NewRangeId::new(self.builder.f32_sign(a), self.builder.f32_sign(b)).into(),
                    self.is_poison(value),
                )
            }

            Instr::F32Floor(value) => {
                let [a, b] = self.v_f32(value).parts();
                (
                    NewRangeId::new(self.builder.f32_floor(a), self.builder.f32_floor(b)).into(),
                    self.is_poison(value),
                )
            }

            Instr::F32Ln(value) => {
                let [a, b] = self.v_f32(value).parts();

                let c_min_safe = self.builder.f32_const(MIN_F32_MAGNITUDE);
                let a_le_0 = self.builder.f32_le(a, c_min_safe);

                (
                    NewRangeId::new(self.builder.f32_ln(a), self.builder.f32_ln(b)).into(),
                    self.builder.or(a_le_0, self.is_poison(value)),
                )
            }

            Instr::F32Lg(value) => {
                let [a, b] = self.v_f32(value).parts();

                let c_min_safe = self.builder.f32_const(MIN_F32_MAGNITUDE);
                let a_le_0 = self.builder.f32_le(a, c_min_safe);

                (
                    NewRangeId::new(self.builder.f32_lg(a), self.builder.f32_lg(b)).into(),
                    self.builder.or(a_le_0, self.is_poison(value)),
                )
            }

            Instr::F32Exp(p) => {
                let [a, b] = self.v_f32(p).parts();
                (
                    NewRangeId::new(self.builder.f32_exp(a), self.builder.f32_exp(b)).into(),
                    self.is_poison(p),
                )
            }

            Instr::F32Powi(base, p) => {
                let [a, b] = self.v_f32(base).parts();
                let n = self.v_i32(p);

                let a_pow_n = self.builder.f32_powi(a, n);
                let b_pow_n = self.builder.f32_powi(b, n);

                let c_zero_i32 = self.builder.i32_const(0);
                let n_neg = self.builder.i32_lt(n, c_zero_i32);

                let c_zero_f32 = self.builder.f32_const(0.0);

                let a_le_0 = self.builder.f32_le(a, c_zero_f32);
                let b_ge_0 = self.builder.f32_ge(b, c_zero_f32);

                let contains_zero = self.builder.and(a_le_0, b_ge_0);

                let min = self.builder.f32_min(a_pow_n, b_pow_n);
                let max = self.builder.f32_max(a_pow_n, b_pow_n);

                let min_0 = self.builder.f32_min(min, c_zero_f32);
                let max_0 = self.builder.f32_max(max, c_zero_f32);

                let poison_discont = self.builder.and(n_neg, contains_zero);

                (
                    NewRangeId::new(
                        self.builder.f32_sel(contains_zero, min_0, min),
                        self.builder.f32_sel(contains_zero, max_0, max),
                    )
                    .into(),
                    self.builder.or(poison_discont, self.is_poison(base)),
                )
            }

            Instr::F32Powf(base, p) => {
                let [a, b] = self.v_f32(base).parts();
                let [c, d] = self.v_f32(p).parts();

                let c_zero = self.builder.f32_const(0.0);
                let p_neg = self.builder.f32_lt(a, c_zero);

                let ln_a = self.builder.f32_ln(a);
                let ln_b = self.builder.f32_ln(b);

                let [min_exp, max_exp] = self
                    .mul_interval(NewRangeId::new(ln_a, ln_b), NewRangeId::new(c, d))
                    .parts();

                let inherits_poison = self.builder.or(self.is_poison(base), self.is_poison(p));

                (
                    NewRangeId::new(self.builder.f32_exp(min_exp), self.builder.f32_exp(max_exp))
                        .into(),
                    self.builder.or(inherits_poison, p_neg),
                )
            }

            Instr::F32Max(lhs, rhs) => {
                let [a, b] = self.v_f32(lhs).parts();
                let [c, d] = self.v_f32(rhs).parts();
                (
                    NewRangeId::new(self.builder.f32_max(a, c), self.builder.f32_max(b, d)).into(),
                    self.builder.or(self.is_poison(lhs), self.is_poison(rhs)),
                )
            }

            Instr::F32Min(lhs, rhs) => {
                let [a, b] = self.v_f32(lhs).parts();
                let [c, d] = self.v_f32(rhs).parts();
                (
                    NewRangeId::new(self.builder.f32_min(a, c), self.builder.f32_min(b, d)).into(),
                    self.builder.or(self.is_poison(lhs), self.is_poison(rhs)),
                )
            }

            Instr::F32Eq(lhs, rhs) => {
                let [a, b] = self.v_f32(lhs).parts();
                let [c, d] = self.v_f32(rhs).parts();

                let a_eq_b = self.builder.f32_eq(a, b);
                let c_eq_d = self.builder.f32_eq(c, d);
                let a_eq_c = self.builder.f32_eq(a, c);

                let a_b_c_eq = self.builder.and(a_eq_b, a_eq_c);
                let all_eq = self.builder.and(a_b_c_eq, c_eq_d);

                let c_le_b = self.builder.f32_le(c, b);
                let a_le_d = self.builder.f32_le(a, d);

                let any_eq = self.builder.or(c_le_b, a_le_d);

                (
                    NewRangeId::new(all_eq, any_eq).into(),
                    self.builder.or(self.is_poison(lhs), self.is_poison(rhs)),
                )
            }

            Instr::F32Ne(lhs, rhs) => {
                let [a, b] = self.v_f32(lhs).parts();
                let [c, d] = self.v_f32(rhs).parts();

                let b_lt_c = self.builder.f32_lt(b, c);
                let d_lt_a = self.builder.f32_lt(d, a);

                let none_eq = self.builder.and(b_lt_c, d_lt_a);

                let a_ne_b = self.builder.f32_ne(a, b);
                let c_ne_d = self.builder.f32_ne(c, d);
                let a_ne_c = self.builder.f32_ne(a, c);

                let a_b_c_ne = self.builder.or(a_ne_b, a_ne_c);
                let any_ne = self.builder.or(a_b_c_ne, c_ne_d);

                (
                    NewRangeId::new(none_eq, any_ne).into(),
                    self.builder.or(self.is_poison(lhs), self.is_poison(rhs)),
                )
            }

            Instr::F32Lt(lhs, rhs) => {
                let [a, b] = self.v_f32(lhs).parts();
                let [c, d] = self.v_f32(rhs).parts();
                (
                    NewRangeId::new(self.builder.f32_lt(b, c), self.builder.f32_lt(a, d)).into(),
                    self.builder.or(self.is_poison(lhs), self.is_poison(rhs)),
                )
            }

            Instr::F32Gt(lhs, rhs) => {
                let [a, b] = self.v_f32(lhs).parts();
                let [c, d] = self.v_f32(rhs).parts();
                (
                    NewRangeId::new(self.builder.f32_gt(a, d), self.builder.f32_gt(b, c)).into(),
                    self.builder.or(self.is_poison(lhs), self.is_poison(rhs)),
                )
            }

            Instr::F32Le(lhs, rhs) => {
                let [a, b] = self.v_f32(lhs).parts();
                let [c, d] = self.v_f32(rhs).parts();
                (
                    NewRangeId::new(self.builder.f32_le(b, c), self.builder.f32_le(a, d)).into(),
                    self.builder.or(self.is_poison(lhs), self.is_poison(rhs)),
                )
            }

            Instr::F32Ge(lhs, rhs) => {
                let [a, b] = self.v_f32(lhs).parts();
                let [c, d] = self.v_f32(rhs).parts();
                (
                    NewRangeId::new(self.builder.f32_ge(a, d), self.builder.f32_ge(b, c)).into(),
                    self.builder.or(self.is_poison(lhs), self.is_poison(rhs)),
                )
            }

            Instr::F32Sel(cond, v_t, v_f) => {
                let [all, any] = self.v_bool(cond).parts();

                let [a, b] = self.v_f32(v_t).parts();
                let [c, d] = self.v_f32(v_f).parts();

                let is_true = self.builder.and(any, all);

                let def_min = self.builder.f32_sel(is_true, a, c);
                let def_max = self.builder.f32_sel(is_true, b, d);

                let is_n_def = self.builder.xor(any, all);

                let n_def_min = self.builder.f32_min(a, c);
                let n_def_max = self.builder.f32_max(b, d);

                let poison_v_t = self.is_poison(v_t);
                let poison_v_f = self.is_poison(v_f);

                let poison_v_def = self.bool_sel(is_true, poison_v_t, poison_v_f);
                let poison_v_n_def = self.builder.or(poison_v_t, poison_v_f);

                let poison_v = self.bool_sel(is_n_def, poison_v_n_def, poison_v_def);

                (
                    NewRangeId::new(
                        self.builder.f32_sel(is_n_def, n_def_min, def_min),
                        self.builder.f32_sel(is_n_def, n_def_max, def_max),
                    )
                    .into(),
                    self.builder.or(poison_v, self.is_poison(cond)),
                )
            }

            Instr::F32Abs(value) => {
                let [a, b] = self.v_f32(value).parts();

                let a_abs = self.builder.f32_abs(a);
                let b_abs = self.builder.f32_abs(b);

                let min = self.builder.f32_min(a_abs, b_abs);
                let max = self.builder.f32_max(a_abs, b_abs);

                let c_zero = self.builder.f32_const(0.0);

                let a_lt_0 = self.builder.f32_lt(a, c_zero);
                let b_gt_0 = self.builder.f32_gt(b, c_zero);

                let changes_sign = self.builder.and(a_lt_0, b_gt_0);

                (
                    NewRangeId::new(self.builder.f32_sel(changes_sign, c_zero, min), max).into(),
                    self.is_poison(value),
                )
            }

            Instr::F32Sin(_) => {
                todo!()
            }

            Instr::F32Cos(_) => {
                todo!()
            }

            Instr::F32Tan(_) => {
                todo!()
            }

            Instr::F32Cot(_) => {
                todo!()
            }
        }
    }

    fn mul_interval(&mut self, lhs: NewRangeId<f32>, rhs: NewRangeId<f32>) -> NewRangeId<f32> {
        let [a, b] = lhs.parts();
        let [c, d] = rhs.parts();

        let ac = self.builder.f32_mul(a, c);
        let ad = self.builder.f32_mul(a, d);
        let bc = self.builder.f32_mul(b, c);
        let bd = self.builder.f32_mul(b, d);

        let min_ac_ad = self.builder.f32_min(ac, ad);
        let max_ac_ad = self.builder.f32_max(ac, ad);

        let min_bc_bd = self.builder.f32_min(bc, bd);
        let max_bc_bd = self.builder.f32_max(bc, bd);

        NewRangeId::new(
            self.builder.f32_min(min_ac_ad, min_bc_bd),
            self.builder.f32_max(max_ac_ad, max_bc_bd),
        )
    }

    fn bool_sel(
        &mut self,
        c: NewValueId<bool>,
        v_t: NewValueId<bool>,
        v_f: NewValueId<bool>,
    ) -> NewValueId<bool> {
        let d = self.builder.xor(v_t, v_f);
        let d_m = self.builder.and(d, c);
        self.builder.xor(d_m, v_f)
    }

    fn v_f32(&self, id: ValueId<f32>) -> NewRangeId<f32> {
        self.slots[id.index()].unwrap_f32()
    }

    fn v_bool(&self, id: ValueId<bool>) -> NewRangeId<bool> {
        self.slots[id.index()].unwrap_bool()
    }

    fn v_i32(&self, id: ValueId<i32>) -> NewValueId<i32> {
        self.slots[id.index()].unwrap_i32()
    }

    fn is_poison<T: ValueRepr>(&self, id: ValueId<T>) -> NewValueId<bool> {
        self.poison[id.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::Fallback,
        kernel::{Evaluate, Kernel},
    };

    fn eval<const N: usize>(tape: &Tape, params: &[Interval<f32>; N]) -> Interval<f32> {
        let tape = translate(tape);

        assert_eq!(tape.num_params(), 2 * N);
        assert_eq!(tape.num_results(), 2);

        let result: Interval<f32> = Fallback::new(Artifact::from_tape(tape))
            .evaluator()
            .evaluate(*params);

        result
    }

    fn un_op(f: impl FnOnce(&mut TapeBuilder, NewValueId<f32>) -> NewValueId<f32>) -> Tape {
        let mut b = Tape::builder(vec![ValueType::F32], vec![ValueType::F32]);
        let x = b.param(0);
        f(&mut b, x);
        b.build().unwrap()
    }

    fn bin_op(
        f: impl FnOnce(&mut TapeBuilder, NewValueId<f32>, NewValueId<f32>) -> NewValueId<f32>,
    ) -> Tape {
        let mut b = Tape::builder(vec![ValueType::F32; 2], vec![ValueType::F32]);
        let (x, y) = (b.param(0), b.param(1));
        f(&mut b, x, y);
        b.build().unwrap()
    }

    fn assert_close(actual: Interval<f32>, expected: Interval<f32>) {
        assert!(
            (actual.min() - expected.min()).abs() < 1e-5
                && (actual.max() - expected.max()).abs() < 1e-5,
            "expected {expected:?}, got {actual:?}"
        );
    }

    fn interval(min: f32, max: f32) -> Interval<f32> {
        Interval::<f32>::new(min, max)
    }

    #[test]
    fn test_f32_simple_ops() {
        let (x, y) = (interval(-1.0, 2.0), interval(3.0, 5.0));

        assert_eq!(
            eval(&bin_op(TapeBuilder::f32_add), &[x, y]),
            interval(2.0, 7.0)
        );

        assert_eq!(
            eval(&bin_op(TapeBuilder::f32_sub), &[x, y]),
            interval(-6.0, -1.0)
        );

        assert_eq!(
            eval(&bin_op(TapeBuilder::f32_mul), &[x, y]),
            interval(-5.0, 10.0)
        );

        assert_eq!(
            eval(&bin_op(TapeBuilder::f32_mul), &[x, x]),
            interval(-2.0, 4.0)
        );

        assert_eq!(
            eval(&bin_op(TapeBuilder::f32_min), &[x, y]),
            interval(-1.0, 2.0)
        );

        assert_eq!(
            eval(&bin_op(TapeBuilder::f32_max), &[x, y]),
            interval(3.0, 5.0)
        );

        assert_eq!(
            eval(&un_op(TapeBuilder::f32_neg), &[x]),
            interval(-2.0, 1.0)
        );

        assert_eq!(eval(&un_op(TapeBuilder::f32_abs), &[x]), interval(0.0, 2.0));

        assert_eq!(
            eval(&un_op(TapeBuilder::f32_abs), &[interval(-3.0, -1.0)]),
            interval(1.0, 3.0)
        );

        assert_eq!(
            eval(&un_op(TapeBuilder::f32_sign), &[x]),
            interval(-1.0, 1.0)
        );

        assert_eq!(
            eval(&un_op(TapeBuilder::f32_sign), &[y]),
            interval(1.0, 1.0)
        );

        assert_eq!(
            eval(&un_op(TapeBuilder::f32_floor), &[interval(-1.5, 2.3)]),
            interval(-2.0, 2.0)
        );
    }

    #[test]
    fn test_f32_div() {
        let tape = bin_op(TapeBuilder::f32_div);

        assert_close(
            eval(&tape, &[interval(1.0, 2.0), interval(2.0, 4.0)]),
            interval(0.25, 1.0),
        );

        assert_close(
            eval(&tape, &[interval(1.0, 2.0), interval(-4.0, -2.0)]),
            interval(-1.0, -0.25),
        );

        assert_close(
            eval(&tape, &[interval(-2.0, -1.0), interval(2.0, 4.0)]),
            interval(-1.0, -0.25),
        );

        assert_close(
            eval(&tape, &[interval(-2.0, -1.0), interval(-4.0, -2.0)]),
            interval(0.25, 1.0),
        );

        assert_close(
            eval(&tape, &[interval(-2.0, 1.0), interval(2.0, 4.0)]),
            interval(-1.0, 0.5),
        );

        assert_close(
            eval(&tape, &[interval(-2.0, 1.0), interval(-4.0, -2.0)]),
            interval(-0.5, 1.0),
        );

        assert!(
            eval(&tape, &[interval(1.0, 2.0), interval(-1.0, 1.0)])
                .min()
                .is_nan()
        );

        assert!(
            eval(&tape, &[interval(1.0, 2.0), interval(0.0, 1.0)])
                .min()
                .is_nan()
        );

        assert!(
            eval(&tape, &[interval(1.0, 2.0), interval(-1.0, 0.0)])
                .min()
                .is_nan()
        );
    }

    #[test]
    fn test_f32_powi() {
        let pow_n = |n: i32| {
            un_op(|builder, base| {
                let c_n = builder.i32_const(n);
                builder.f32_powi(base, c_n)
            })
        };

        let pow_2 = pow_n(2);

        assert_eq!(eval(&pow_2, &[interval(-1.0, 2.0)]), interval(0.0, 4.0));
        assert_eq!(eval(&pow_2, &[interval(1.0, 2.0)]), interval(1.0, 4.0));
        assert_eq!(eval(&pow_2, &[interval(-3.0, -2.0)]), interval(4.0, 9.0));

        let pow_3 = pow_n(3);
        assert_eq!(eval(&pow_3, &[interval(-1.0, 2.0)]), interval(-1.0, 8.0));

        let pow_neg_1 = pow_n(-1);
        assert_eq!(eval(&pow_neg_1, &[interval(1.0, 2.0)]), interval(0.5, 1.0));

        assert!(eval(&pow_neg_1, &[interval(-1.0, 2.0)]).min().is_nan());
        assert!(eval(&pow_neg_1, &[interval(0.0, 2.0)]).min().is_nan());

        assert!(eval(&pow_2, &[interval(-1.0, 2.0)]).min().is_finite());
    }

    #[test]
    fn test_f32_powf() {
        let tape = bin_op(TapeBuilder::f32_powf);

        assert_close(
            eval(&tape, &[interval(1.0, 2.0), interval(2.0, 3.0)]),
            interval(1.0, 8.0),
        );

        assert_close(
            eval(&tape, &[interval(2.0, 4.0), interval(-1.0, 0.5)]),
            interval(0.25, 2.0),
        );

        assert!(
            eval(&tape, &[interval(-1.0, 4.0), interval(2.0, 3.0)])
                .min()
                .is_nan()
        );

        assert!(
            !eval(&tape, &[interval(0.0, 4.0), interval(2.0, 3.0)])
                .min()
                .is_nan()
        );
    }

    // #[test]
    // fn transcendental() {
    //     assert_close(
    //         eval(&unary(TapeBuilder::f32_exp), &[interval(0.0, 1.0)]),
    //         interval(1.0, 1f32.exp()),
    //     );
    //     assert_close(
    //         eval(&unary(TapeBuilder::f32_ln), &[interval(1.0, 4.0)]),
    //         interval(0.0, 4f32.ln()),
    //     );
    //     assert_close(
    //         eval(&unary(TapeBuilder::f32_lg), &[interval(1.0, 100.0)]),
    //         interval(0.0, 2.0),
    //     );

    //     let lowest = MIN_F32_MAGNITUDE.ln();

    //     let ln = bounds(&unary(TapeBuilder::f32_ln), &[interval(-1.0, 4.0)]);
    //     assert!(ln.get().is_none());
    //     assert_close(ln.interval, interval(lowest, 4f32.ln()));

    //     let ln = bounds(&unary(TapeBuilder::f32_ln), &[interval(-2.0, -1.0)]);
    //     assert!(ln.get().is_none());
    //     assert_close(ln.interval, interval(lowest, lowest));

    //     assert!(poisoned(&unary(TapeBuilder::f32_lg), &[interval(0.0, 1.0)]));

    //     let sin = unary(TapeBuilder::f32_sin);
    //     assert_close(
    //         eval(&sin, &[interval(0.0, 0.2)]),
    //         interval(0.0, 0.2f32.sin()),
    //     );
    //     assert_close(eval(&sin, &[interval(1.0, 2.0)]), interval(1f32.sin(), 1.0));
    //     assert_close(
    //         eval(&sin, &[interval(3.0, 5.0)]),
    //         interval(-1.0, 3f32.sin()),
    //     );
    //     assert_eq!(eval(&sin, &[interval(-100.0, 100.0)]), interval(-1.0, 1.0));

    //     let cos = unary(TapeBuilder::f32_cos);
    //     assert_close(
    //         eval(&cos, &[interval(-0.1, 0.1)]),
    //         interval(0.1f32.cos(), 1.0),
    //     );
    //     assert_close(
    //         eval(&cos, &[interval(1.0, 2.0)]),
    //         interval(2f32.cos(), 1f32.cos()),
    //     );
    //     assert_close(eval(&cos, &[interval(3.0, 7.0)]), interval(-1.0, 1.0));
    // }

    // #[test]
    // fn tangent() {
    //     let tan = unary(TapeBuilder::f32_tan);
    //     assert_close(eval(&tan, &[interval(0.0, 1.0)]), interval(0.0, 1f32.tan()));
    //     assert_close(
    //         eval(&tan, &[interval(-1.0, -0.5)]),
    //         interval((-1f32).tan(), (-0.5f32).tan()),
    //     );
    //     assert!(poisoned(&tan, &[interval(1.0, 2.0)]));
    //     assert!(poisoned(&tan, &[interval(0.0, 4.0)]));

    //     let cot = unary(TapeBuilder::f32_cot);
    //     assert_close(
    //         eval(&cot, &[interval(0.5, 1.0)]),
    //         interval(1f32.tan().recip(), 0.5f32.tan().recip()),
    //     );
    //     assert!(poisoned(&cot, &[interval(-0.5, 0.5)]));
    // }

    #[test]
    fn test_f32_sel() {
        let mut b = Tape::builder(vec![ValueType::F32; 2], vec![ValueType::F32]);
        let [x, y] = [b.param(0), b.param(1)];

        let x_lt_y = b.f32_lt(x, y);

        b.f32_sel(x_lt_y, x, y);
        let tape = b.build().unwrap();

        // any(x < y) && all(x < y)
        assert_eq!(
            eval(&tape, &[interval(0.0, 1.0), interval(2.0, 3.0)]),
            interval(0.0, 1.0)
        );

        // any(x > y) && all(x > y)
        assert_eq!(
            eval(&tape, &[interval(4.0, 5.0), interval(2.0, 3.0)]),
            interval(2.0, 3.0)
        );

        // any(x < y) && !all(x < y)
        assert_eq!(
            eval(&tape, &[interval(0.0, 3.0), interval(2.0, 5.0)]),
            interval(0.0, 5.0)
        );
    }

    #[test]
    fn test_f32_sel_def() {
        let mut b = Tape::builder(vec![ValueType::F32], vec![ValueType::F32]);
        let x = b.param(0);

        let c_1 = b.f32_const(1.0);
        let c_2 = b.f32_const(2.0);

        let c_1_lt_2 = b.f32_lt(c_1, c_2);

        let c_3 = b.f32_const(3.0);
        let c_4 = b.f32_const(4.0);

        let scale = b.f32_sel(c_1_lt_2, c_3, c_4);

        b.f32_mul(x, scale);
        let tape = b.build().unwrap();

        assert_eq!(eval(&tape, &[interval(-1.0, 2.0)]), interval(-3.0, 6.0));
    }

    #[test]
    fn test_f32_sel_poison() {
        let mut b = Tape::builder(vec![ValueType::F32], vec![ValueType::F32]);
        let x = b.param(0);

        let c_0 = b.f32_const(0.0);
        let x_lt_0 = b.f32_lt(x, c_0);

        let c_1 = b.f32_const(1.0);
        let recip_x = b.f32_div(c_1, x);

        b.f32_sel(x_lt_0, recip_x, x);
        let tape = b.build().unwrap();

        // branch never taken (always >= 0)
        assert_eq!(eval(&tape, &[interval(0.0, 2.0)]), interval(0.0, 2.0));

        // branch always taken (always < 0)
        assert_eq!(eval(&tape, &[interval(-2.0, -1.0)]), interval(-1.0, -0.5));

        // taken and not taken at the same time (division by zero)
        assert!(eval(&tape, &[interval(-1.0, 1.0)]).min().is_nan());
    }
}
