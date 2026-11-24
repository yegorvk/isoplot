use std::{
    fmt::{self, Debug},
    iter,
};

use crate::{
    Dual,
    artifact::Artifact,
    schema::{Field, Schema},
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

pub(crate) fn interval<const TILE_LEN: usize, const N: usize, const M: usize>(
    artifact: &Artifact<[[f32; TILE_LEN]; N], [[f32; TILE_LEN]; M]>,
) -> Artifact<[Interval<[f32; TILE_LEN]>; N], [Interval<[f32; TILE_LEN]>; M]> {
    Artifact::from_tape(translate(artifact.tape()))
}

pub(crate) fn interval_d<const TILE_LEN: usize, const N: usize, const M: usize>(
    artifact: &Artifact<[[f32; TILE_LEN]; N], Dual<[f32; TILE_LEN], M>>,
) -> Artifact<[Interval<[f32; TILE_LEN]>; N], Dual<Interval<[f32; TILE_LEN]>, M>> {
    Artifact::from_tape(translate(artifact.tape()))
}

fn translate(source: &Tape) -> Tape {
    Translator::new(source).run()
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
enum Record {
    I32(NewValueId<i32>),
    Bool(NewRangeId<bool>),
    F32(NewRangeId<f32>),
}

impl Record {
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

impl From<NewValueId<i32>> for Record {
    fn from(value: NewValueId<i32>) -> Self {
        Self::I32(value)
    }
}

impl From<NewRangeId<bool>> for Record {
    fn from(value: NewRangeId<bool>) -> Self {
        Self::Bool(value)
    }
}

impl From<NewRangeId<f32>> for Record {
    fn from(value: NewRangeId<f32>) -> Self {
        Self::F32(value)
    }
}

struct Translator<'a> {
    source: &'a Tape,
    builder: TapeBuilder,
    records: Vec<Record>,
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

        let records = (0..num_params)
            .map(|i| {
                Record::F32(NewRangeId::new(
                    builder.param(2 * i),
                    builder.param(2 * i + 1),
                ))
            })
            .collect();

        let poison = vec![builder.bool_const(false); num_params];

        Self {
            source,
            builder,
            records,
            poison,
        }
    }

    fn run(mut self) -> Tape {
        for instr in self.source.instrs() {
            let (rec, poison) = self.translate(*instr);
            self.records.push(rec.into());
            self.poison.push(poison);
        }

        let num_results = self.source.num_results();
        let first = self.records.len() - num_results;

        let c_nan = self.builder.f32_const(f32::NAN);

        for i in first..(first + num_results) {
            let Record::F32(range) = self.records[i] else {
                unreachable!("interval results must be f32");
            };
            self.builder.f32_sel(self.poison[i], c_nan, range.min);
            self.builder.copy_f32(range.max);
        }

        self.builder.build().unwrap()
    }

    fn translate(&mut self, instr: Instr) -> (Record, NewValueId<bool>) {
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
                    NewRangeId::new(self.builder.and(a, c), self.builder.or(b, d)).into(),
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
                let [a, b] = self.v_f32(lhs).parts();
                let [c, d] = self.v_f32(rhs).parts();

                let zero = self.builder.f32_const(0.0);

                let a_le_0 = self.builder.f32_le(a, zero);
                let b_ge_0 = self.builder.f32_ge(b, zero);

                let c_le_0 = self.builder.f32_le(c, zero);
                let d_ge_0 = self.builder.f32_ge(d, zero);

                let d_zero = self.builder.and(c_le_0, d_ge_0);

                let n_min = self.builder.f32_sel(c_le_0, b, a);
                let n_max = self.builder.f32_sel(c_le_0, a, b);

                let d_min = self.builder.f32_sel(a_le_0, c, d);
                let d_max = self.builder.f32_sel(b_ge_0, c, d);

                let inherits_poison = self.builder.or(self.is_poison(lhs), self.is_poison(rhs));

                (
                    NewRangeId::new(
                        self.builder.f32_div(n_min, d_min),
                        self.builder.f32_div(n_max, d_max),
                    )
                    .into(),
                    self.builder.or(inherits_poison, d_zero),
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

                let c_zero = self.builder.f32_const(0.0);
                let a_le_0 = self.builder.f32_le(a, c_zero);

                (
                    NewRangeId::new(self.builder.f32_ln(a), self.builder.f32_ln(b)).into(),
                    self.builder.or(a_le_0, self.is_poison(value)),
                )
            }

            Instr::F32Lg(value) => {
                let [a, b] = self.v_f32(value).parts();

                let c_zero = self.builder.f32_const(0.0);
                let a_le_0 = self.builder.f32_le(a, c_zero);

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

                let a_eq_c = self.builder.f32_eq(a, c);
                let b_eq_d = self.builder.f32_eq(b, d);

                let max_ac = self.builder.f32_max(a, c);
                let min_bd = self.builder.f32_min(b, d);

                let any_eq = self.builder.f32_le(max_ac, min_bd);

                (
                    NewRangeId::new(self.builder.and(a_eq_c, b_eq_d), any_eq).into(),
                    self.builder.or(self.is_poison(lhs), self.is_poison(rhs)),
                )
            }

            Instr::F32Ne(lhs, rhs) => {
                let [a, b] = self.v_f32(lhs).parts();
                let [c, d] = self.v_f32(rhs).parts();

                let a_ne_c = self.builder.f32_ne(a, c);
                let b_ne_d = self.builder.f32_ne(b, d);

                let min_ac = self.builder.f32_min(a, c);
                let max_bd = self.builder.f32_max(b, d);

                let none_eq = self.builder.f32_gt(min_ac, max_bd);

                (
                    NewRangeId::new(none_eq, self.builder.or(a_ne_c, b_ne_d)).into(),
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
        self.records[id.index()].unwrap_f32()
    }

    fn v_bool(&self, id: ValueId<bool>) -> NewRangeId<bool> {
        self.records[id.index()].unwrap_bool()
    }

    fn v_i32(&self, id: ValueId<i32>) -> NewValueId<i32> {
        self.records[id.index()].unwrap_i32()
    }

    fn is_poison<T: ValueRepr>(&self, id: ValueId<T>) -> NewValueId<bool> {
        self.poison[id.index()]
    }

    // fn run(mut self) -> Tape {
    //     for instr in self.source.instrs().iter().copied() {
    //         let (value, is_nan) = self.translate(instr);
    //         self.records.push(value.into());
    //         self.poison.push(is_nan);
    //     }

    //     let num_results = self.source.num_results();
    //     let first = self.records.len() - num_results;

    //     for i in first..(first + num_results) {
    //         let Record::F32(range) = self.records[i] else {
    //             unreachable!("interval results must be f32");
    //         };

    //         self.builder.copy_f32(range.min);
    //         self.builder.copy_f32(range.max);
    //         self.builder.i32_from_bool(self.poison[i]);
    //     }

    //     self.builder.build().unwrap()
    // }

    // fn translate(&mut self, instr: Instr) -> (NewRangeId<f32>, NewValueId<bool>) {
    //     let value = match instr {
    //         Instr::F32Div(lhs, rhs) => {
    //             let divisor = self.f32(rhs);
    //             return (
    //                 self.mul_range(self.f32(lhs), self.recip(divisor)),
    //                 self.builder.or(
    //                     self.contains_zero(divisor),
    //                     self.builder.or(self.is_finite(lhs), self.is_finite(rhs)),
    //                 ),
    //             );
    //         }

    //         Instr::F32Powf(lhs, rhs) => {
    //             let base = self.f32(lhs);
    //             let ln_base = base.map(|v| self.builder.f32_ln(v));
    //             let exp = self.mul_range(ln_base, self.f32(rhs));
    //             return (
    //                 exp.map(|x| self.builder.f32_exp(x)),
    //                 self.builder.or(
    //                     self.below_min_magnitude(base),
    //                     self.builder.or(self.is_finite(lhs), self.is_finite(rhs)),
    //                 ),
    //             );
    //         }

    //         Instr::F32Powi(lhs, rhs) => {
    //             let base = self.f32(lhs);
    //             let pow = self.i32(rhs);
    //             let [a, b] = base.into_parts().map(|v| self.builder.f32_powi(v, pow));
    //             let range = self.f32_hull(a, b);

    //             // `range` might miss zero for even powers, e.g. [-1, 2]^2 = [1, 4].
    //             let zero_in_base = self.contains_zero(base);
    //             let range = self.include_zero_if(range, zero_in_base);

    //             return (
    //                 range.into(),
    //                 self.builder.or(
    //                     self.builder.and(
    //                         self.contains_zero(base),
    //                         self.builder.i32_lt(pow, self.builder.i32_const(0)),
    //                     ),
    //                     self.is_finite(lhs),
    //                 ),
    //             );
    //         }

    //         Instr::F32Ln(src) => {
    //             let x = self.f32(src);
    //             let range = x.map(|v| self.builder.f32_ln(v));
    //             return (range.into(), self.below_min_magnitude(x));
    //         }

    //         Instr::F32Lg(src) => {
    //             let x = self.f32(src);
    //             let range = x.map(|v| self.builder.f32_lg(v));
    //             return (range.into(), self.below_min_magnitude(x));
    //         }

    //         Instr::F32Tan(src) => {
    //             let x = self.f32(src);
    //             let [a, b] = x.into_parts().map(|v| self.builder.f32_tan(v));
    //             let range = self.f32_hull(a, b);
    //             return (range.into(), self.contains_modulo(x, FRAC_PI_2, PI));
    //         }

    //         Instr::F32Cot(src) => {
    //             let x = self.f32(src);
    //             let [a, b] = x.into_parts().map(|v| self.builder.f32_cot(v));
    //             let range = self.f32_hull(a, b);
    //             return (range.into(), self.contains_modulo(x, 0.0, PI));
    //         }

    //         Instr::F32Sel(cond, v_true, v_false) => {
    //             let [always, maybe] = self.bool(cond).into_parts();

    //             let (p_t, p_f) = (self.is_nan[v_true.index()], self.is_nan[v_false.index()]);
    //             let never_true = self.builder.not(always);
    //             let taken_t = self.builder.and(cond.any, p_t);
    //             let taken_f = self.builder.and(never_true, p_f);
    //             let taken = self.builder.or(taken_t, taken_f);

    //             let (t, f) = (self.f32(v_true), self.f32(v_false));

    //             let range = match cond.definite() {
    //                 Some(cond) => self.select_range(cond, t, f),
    //                 None => {
    //                     // An undecided condition yields the hull of both branches.
    //                     let merged = F32Range::new(
    //                         self.builder.f32_min(t.min, f.min),
    //                         self.builder.f32_max(t.max, f.max),
    //                     );
    //                     let unless_true = self.select_range(cond.any, merged, f);
    //                     self.select_range(cond.all, t, unless_true)
    //                 }
    //             };

    //             return (
    //                 range.into(),
    //                 self.builder.or(self.is_nan[cond.index()], taken),
    //             );
    //         }

    //         Instr::I32Const(c) => Record::I32(self.builder.i32_const(c)),
    //         Instr::BoolConst(c) => BoolRange::point(self.builder.bool_const(c)).into(),
    //         Instr::F32Const(c) => F32Range::point(self.builder.f32_const(c)).into(),

    //         Instr::I32Add(lhs, rhs) => {
    //             let (x, y) = (self.i32(lhs), self.i32(rhs));
    //             Record::I32(self.builder.i32_add(x, y))
    //         }

    //         Instr::I32Sub(lhs, rhs) => {
    //             let (x, y) = (self.i32(lhs), self.i32(rhs));
    //             Record::I32(self.builder.i32_sub(x, y))
    //         }

    //         Instr::I32Mul(lhs, rhs) => {
    //             let (x, y) = (self.i32(lhs), self.i32(rhs));
    //             Record::I32(self.builder.i32_mul(x, y))
    //         }

    //         Instr::I32Eq(lhs, rhs) => {
    //             let (x, y) = (self.i32(lhs), self.i32(rhs));
    //             BoolRange::point(self.builder.i32_eq(x, y)).into()
    //         }

    //         Instr::I32Ne(lhs, rhs) => {
    //             let (x, y) = (self.i32(lhs), self.i32(rhs));
    //             BoolRange::point(self.builder.i32_ne(x, y)).into()
    //         }

    //         Instr::I32Lt(lhs, rhs) => {
    //             let (x, y) = (self.i32(lhs), self.i32(rhs));
    //             BoolRange::point(self.builder.i32_lt(x, y)).into()
    //         }

    //         Instr::I32Le(lhs, rhs) => {
    //             let (x, y) = (self.i32(lhs), self.i32(rhs));
    //             BoolRange::point(self.builder.i32_le(x, y)).into()
    //         }

    //         Instr::I32Gt(lhs, rhs) => {
    //             let (x, y) = (self.i32(lhs), self.i32(rhs));
    //             BoolRange::point(self.builder.i32_gt(x, y)).into()
    //         }

    //         Instr::I32Ge(lhs, rhs) => {
    //             let (x, y) = (self.i32(lhs), self.i32(rhs));
    //             BoolRange::point(self.builder.i32_ge(x, y)).into()
    //         }

    //         Instr::Not(src) => {
    //             let x = self.bool(src);
    //             self.bool_not(x).into()
    //         }

    //         Instr::And(lhs, rhs) => {
    //             let (x, y) = (self.bool(lhs), self.bool(rhs));
    //             let all = self.builder.and(x.all, y.all);
    //             let any = if x.definite().is_some() && y.definite().is_some() {
    //                 all
    //             } else {
    //                 self.builder.and(x.any, y.any)
    //             };
    //             BoolRange { all, any }.into()
    //         }

    //         Instr::Or(lhs, rhs) => {
    //             let (x, y) = (self.bool(lhs), self.bool(rhs));
    //             let all = self.builder.or(x.all, y.all);
    //             let any = if x.definite().is_some() && y.definite().is_some() {
    //                 all
    //             } else {
    //                 self.builder.or(x.any, y.any)
    //             };
    //             BoolRange { all, any }.into()
    //         }

    //         Instr::Xor(lhs, rhs) => {
    //             let (x, y) = (self.bool(lhs), self.bool(rhs));
    //             let definite = self.builder.xor(x.all, y.all);
    //             if x.definite().is_some() && y.definite().is_some() {
    //                 BoolRange::point(definite).into()
    //             } else {
    //                 // Any uncertain operand makes the result fully uncertain.
    //                 let unc_x = self.builder.xor(x.all, x.any);
    //                 let unc_y = self.builder.xor(y.all, y.any);
    //                 let uncertain = self.builder.or(unc_x, unc_y);
    //                 let certain = self.builder.not(uncertain);
    //                 let all = self.builder.and(definite, certain);
    //                 let any = self.builder.or(definite, uncertain);
    //                 BoolRange { all, any }.into()
    //             }
    //         }

    //         Instr::CopyI32(src) => Record::I32(self.i32(src)),
    //         Instr::CopyBool(src) => self.bool(src).into(),
    //         Instr::CopyF32(src) => self.f32(src).into(),

    //         Instr::I32FromBool(src) => {
    //             let Some(src) = self.bool(src).definite() else {
    //                 unimplemented!(
    //                     "data-dependent i32 values are not supported in interval translation"
    //                 );
    //             };

    //             Record::I32(self.builder.i32_from_bool(src))
    //         }

    //         Instr::F32FromI32(src) => {
    //             let src = self.i32(src);
    //             F32Range::point(self.builder.f32_from_i32(src)).into()
    //         }

    //         Instr::F32FromBool(src) => {
    //             let x = self.bool(src);
    //             match x.definite() {
    //                 Some(v) => F32Range::point(self.builder.f32_from_bool(v)).into(),
    //                 None => {
    //                     let lo = self.builder.f32_from_bool(x.all);
    //                     let hi = self.builder.f32_from_bool(x.any);
    //                     F32Range::new(lo, hi).into()
    //                 }
    //             }
    //         }

    //         Instr::F32Neg(src) => {
    //             let x = self.f32(src);
    //             let [lo, hi] = x.values().map(|v| self.builder.f32_neg(v));
    //             F32Range::new(hi, lo).into()
    //         }

    //         Instr::F32Abs(src) => {
    //             // |[lo, hi]| = [max(lo, -hi, 0), max(hi, -lo)]
    //             let x = self.f32(src);
    //             let neg_lo = self.builder.f32_neg(x.min);
    //             let neg_hi = self.builder.f32_neg(x.max);
    //             let zero = self.builder.f32_const(0.0);
    //             let lo = self.max_of([x.min, neg_hi, zero]);
    //             let hi = self.builder.f32_max(x.max, neg_lo);
    //             F32Range::new(lo, hi).into()
    //         }

    //         Instr::F32Sign(src) => {
    //             let x = self.f32(src);
    //             x.map(|v| self.builder.f32_sign(v)).into()
    //         }

    //         Instr::F32Floor(src) => {
    //             let x = self.f32(src);
    //             x.map(|v| self.builder.f32_floor(v)).into()
    //         }

    //         Instr::F32Add(lhs, rhs) => {
    //             let (x, y) = (self.f32(lhs), self.f32(rhs));
    //             F32Range::new(
    //                 self.builder.f32_add(x.min, y.min),
    //                 self.builder.f32_add(x.max, y.max),
    //             )
    //             .into()
    //         }

    //         Instr::F32Sub(lhs, rhs) => {
    //             let (x, y) = (self.f32(lhs), self.f32(rhs));
    //             F32Range::new(
    //                 self.builder.f32_sub(x.min, y.max),
    //                 self.builder.f32_sub(x.max, y.min),
    //             )
    //             .into()
    //         }

    //         Instr::F32Mul(lhs, rhs) => {
    //             let (x, y) = (self.f32(lhs), self.f32(rhs));
    //             self.mul_range(x, y).into()
    //         }

    //         Instr::F32Min(lhs, rhs) => {
    //             let (x, y) = (self.f32(lhs), self.f32(rhs));
    //             F32Range::new(
    //                 self.builder.f32_min(x.min, y.min),
    //                 self.builder.f32_min(x.max, y.max),
    //             )
    //             .into()
    //         }

    //         Instr::F32Max(lhs, rhs) => {
    //             let (x, y) = (self.f32(lhs), self.f32(rhs));
    //             F32Range::new(
    //                 self.builder.f32_max(x.min, y.min),
    //                 self.builder.f32_max(x.max, y.max),
    //             )
    //             .into()
    //         }

    //         Instr::F32Exp(src) => {
    //             let x = self.f32(src);
    //             x.map(|v| self.builder.f32_exp(v)).into()
    //         }

    //         Instr::F32Sin(src) => {
    //             let x = self.f32(src);
    //             let [a, b] = x.values().map(|v| self.builder.f32_sin(v));
    //             let range = self.f32_hull(a, b);
    //             self.normalized_periodic_wave(x, range, FRAC_PI_2, PI)
    //                 .into()
    //         }

    //         Instr::F32Cos(src) => {
    //             let x = self.f32(src);
    //             let [a, b] = x.values().map(|v| self.builder.f32_cos(v));
    //             let range = self.f32_hull(a, b);
    //             self.normalized_periodic_wave(x, range, 0.0, PI).into()
    //         }

    //         Instr::F32Eq(lhs, rhs) => {
    //             let (x, y) = (self.f32(lhs), self.f32(rhs));
    //             self.f32_eq(x, y).into()
    //         }

    //         Instr::F32Ne(lhs, rhs) => {
    //             let (x, y) = (self.f32(lhs), self.f32(rhs));
    //             let eq = self.f32_eq(x, y);
    //             self.bool_not(eq).into()
    //         }

    //         Instr::F32Lt(lhs, rhs) => {
    //             let (x, y) = (self.f32(lhs), self.f32(rhs));
    //             self.f32_lt(x, y).into()
    //         }

    //         Instr::F32Le(lhs, rhs) => {
    //             let (x, y) = (self.f32(lhs), self.f32(rhs));
    //             self.f32_le(x, y).into()
    //         }

    //         Instr::F32Gt(lhs, rhs) => {
    //             let (x, y) = (self.f32(lhs), self.f32(rhs));
    //             self.f32_lt(y, x).into()
    //         }

    //         Instr::F32Ge(lhs, rhs) => {
    //             let (x, y) = (self.f32(lhs), self.f32(rhs));
    //             self.f32_le(y, x).into()
    //         }

    //         Instr::I32Sel(cond, v_true, v_false) => unimplemented!(),
    //     };

    //     (value, inherited)
    // }

    // fn f32_hull(&mut self, a: NewValueId<f32>, b: NewValueId<f32>) -> NewRangeId<f32> {
    //     NewRangeId::new(self.builder.f32_min(a, b), self.builder.f32_max(a, b))
    // }

    // fn min_of(&mut self, values: impl IntoIterator<Item = NewValueId<f32>>) -> NewValueId<f32> {
    //     values
    //         .into_iter()
    //         .reduce(|x, y| self.builder.f32_min(x, y))
    //         .unwrap()
    // }

    // fn max_of(&mut self, values: impl IntoIterator<Item = NewValueId<f32>>) -> NewValueId<f32> {
    //     values
    //         .into_iter()
    //         .reduce(|x, y| self.builder.f32_max(x, y))
    //         .unwrap()
    // }

    // fn bool_not(&mut self, x: NewRangeId<bool>) -> NewRangeId<bool> {
    //     NewRangeId::new(self.builder.not(x.max), self.builder.not(x.min))
    // }

    // fn f32_lt(&mut self, x: NewRangeId<f32>, y: NewRangeId<f32>) -> NewRangeId<bool> {
    //     NewRangeId::new(
    //         self.builder.f32_lt(x.max, y.min),
    //         self.builder.f32_lt(x.min, y.max),
    //     )
    // }

    // fn f32_le(&mut self, x: NewRangeId<f32>, y: NewRangeId<f32>) -> NewRangeId<bool> {
    //     NewRangeId::new(
    //         self.builder.f32_le(x.max, y.min),
    //         self.builder.f32_le(x.min, y.max),
    //     )
    // }

    // fn f32_eq(&mut self, x: NewRangeId<f32>, y: NewRangeId<f32>) -> NewRangeId<bool> {
    //     let xy = self.f32_le(x, y);
    //     let yx = self.f32_le(y, x);
    //     NewRangeId::new(
    //         self.builder.and(xy.min, yx.min),
    //         self.builder.and(xy.max, yx.max),
    //     )
    // }

    // fn contains_zero(&mut self, x: NewRangeId<f32>) -> NewValueId<bool> {
    //     let zero = self.builder.f32_const(0.0);
    //     let lo = self.builder.f32_le(x.min, zero);
    //     let hi = self.builder.f32_ge(x.max, zero);
    //     self.builder.and(lo, hi)
    // }

    // fn below_min_magnitude(&mut self, x: NewRangeId<f32>) -> NewValueId<bool> {
    //     let min = self.builder.f32_const(MIN_F32_MAGNITUDE);
    //     self.builder.f32_lt(x.min, min)
    // }

    // fn include_zero_if(&mut self, x: NewRangeId<f32>, cond: NewValueId<bool>) -> NewRangeId<f32> {
    //     let zero = self.builder.f32_const(0.0);
    //     let extended = NewRangeId::new(
    //         self.builder.f32_min(x.min, zero),
    //         self.builder.f32_max(x.max, zero),
    //     );
    //     self.select_range(cond, extended, x)
    // }

    // fn mul_range(&mut self, x: NewRangeId<f32>, y: NewRangeId<f32>) -> NewRangeId<f32> {
    //     let products = [
    //         self.builder.f32_mul(x.min, y.min),
    //         self.builder.f32_mul(x.min, y.max),
    //         self.builder.f32_mul(x.max, y.min),
    //         self.builder.f32_mul(x.max, y.max),
    //     ];
    //     NewRangeId::new(self.min_of(products), self.max_of(products))
    // }

    // // Sound only when `x` excludes zero; the caller poisons the result otherwise.
    // fn recip(&mut self, x: NewRangeId<f32>) -> NewRangeId<f32> {
    //     let one = self.builder.f32_const(1.0);
    //     let [a, b] = x.into_parts().map(|v| self.builder.f32_div(one, v));
    //     self.f32_hull(a, b)
    // }

    // fn select_range(
    //     &mut self,
    //     cond: NewValueId<bool>,
    //     if_true: NewRangeId<f32>,
    //     if_false: NewRangeId<f32>,
    // ) -> NewRangeId<f32> {
    //     NewRangeId::new(
    //         self.builder.f32_sel(cond, if_true.min, if_false.min),
    //         self.builder.f32_sel(cond, if_true.max, if_false.max),
    //     )
    // }

    // fn period_index(
    //     &mut self,
    //     v: NewValueId<f32>,
    //     offset: NewValueId<f32>,
    //     scale: NewValueId<f32>,
    // ) -> NewValueId<f32> {
    //     let shifted = self.builder.f32_sub(v, offset);
    //     let scaled = self.builder.f32_mul(shifted, scale);
    //     self.builder.f32_floor(scaled)
    // }

    // /// Returns whether the range contains a point equal to `v` modulo `period`.
    // fn contains_modulo(&mut self, x: NewRangeId<f32>, v: f32, period: f32) -> NewValueId<bool> {
    //     let offset = self.builder.f32_const(v);
    //     let scale = self.builder.f32_const(period.recip());
    //     let first = self.period_index(x.min, offset, scale);
    //     let last = self.period_index(x.max, offset, scale);
    //     let count = self.builder.f32_sub(last, first);
    //     let half = self.builder.f32_const(0.5);
    //     self.builder.f32_gt(count, half)
    // }

    // // Periodic waves reaching 1 at `peak` and -1 at `peak + half_period`, repeating every `2 * half_period`
    // fn normalized_periodic_wave(
    //     &mut self,
    //     x: NewRangeId<f32>,
    //     im: NewRangeId<f32>,
    //     peak: f32,
    //     half_period: f32,
    // ) -> NewRangeId<f32> {
    //     let has_max = self.contains_modulo(x, peak, 2.0 * half_period);
    //     let has_min = self.contains_modulo(x, peak + half_period, 2.0 * half_period);
    //     let one = self.builder.f32_const(1.0);
    //     let neg_one = self.builder.f32_const(-1.0);
    //     let min = self.builder.f32_sel(has_max, one, im.max);
    //     let max = self.builder.f32_sel(has_min, neg_one, im.min);
    //     NewRangeId::new(max, min)
    // }

    // fn i32(&self, id: ValueId<i32>) -> NewValueId<i32> {
    //     self.records[id.index()].unwrap_i32()
    // }

    // fn bool(&self, id: ValueId<bool>) -> NewRangeId<bool> {
    //     self.records[id.index()].unwrap_bool()
    // }

    // fn f32(&self, id: ValueId<f32>) -> NewRangeId<f32> {
    //     self.records[id.index()].unwrap_f32()
    // }

    // fn is_finite(&self, id: ValueId<f32>) -> NewValueId<bool> {
    //     self.poison[id.index()]
    // }
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

        assert_eq!(
            eval(&tape, &[interval(1.0, 2.0), interval(2.0, 4.0)]),
            interval(0.25, 1.0)
        );

        assert_eq!(
            eval(&tape, &[interval(1.0, 2.0), interval(-4.0, -2.0)]),
            interval(-1.0, -0.25)
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
