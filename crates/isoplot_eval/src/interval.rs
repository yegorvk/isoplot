use std::{
    f32::consts::{FRAC_PI_2, PI},
    iter,
};

use bytemuck::{Pod, Zeroable};

use crate::{
    autodiff::Gradient,
    layout::{ValueType, Vector},
    program::Program,
    tape::{Instr, NewId, Tape, TapeBuilder, Type, ValueId},
};

/// A closed and bounded `f32` interval
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct Interval {
    lo: f32,
    hi: f32,
}

impl Interval {
    pub const fn new(min: f32, max: f32) -> Self {
        assert!(min <= max);
        Self { lo: min, hi: max }
    }

    pub const fn min(&self) -> f32 {
        self.lo
    }

    pub const fn max(&self) -> f32 {
        self.hi
    }

    pub const fn center(&self) -> f32 {
        (self.lo + self.hi) * 0.5
    }

    pub const fn contains_zero(&self) -> bool {
        !(self.lo > 0.0 || self.hi < 0.0)
    }
}

impl Vector for Interval {
    type Scalar = Interval;
    const LEN: usize = 2;

    fn types() -> impl Iterator<Item = ValueType> {
        iter::repeat_n(ValueType::F32, Self::LEN)
    }
}

mod private {
    use crate::layout::Layout;

    #[doc(hidden)]
    pub trait IntervalType: Layout {
        type Interval: Layout;
    }
}

pub(crate) use private::IntervalType;

impl IntervalType for f32 {
    type Interval = Interval;
}

impl<T: IntervalType, const N: usize> IntervalType for [T; N] {
    type Interval = [T::Interval; N];
}

impl<V: IntervalType> IntervalType for Gradient<V> {
    type Interval = Gradient<V::Interval>;
}

type IntervalProgram<Args, Ret> =
    Program<<Args as IntervalType>::Interval, <Ret as IntervalType>::Interval>;

pub(crate) fn interval<Args, Ret>(program: &Program<Args, Ret>) -> IntervalProgram<Args, Ret>
where
    Args: IntervalType,
    Ret: IntervalType,
{
    Program::new(translate(program.tape()))
}

fn translate(source: &Tape) -> Tape {
    Translator::new(source).run()
}

#[derive(Copy, Clone)]
struct F32Range {
    lo: NewId<f32>,
    hi: NewId<f32>,
}

impl F32Range {
    fn new(lo: NewId<f32>, hi: NewId<f32>) -> Self {
        Self { lo, hi }
    }

    fn point(v: NewId<f32>) -> Self {
        Self { lo: v, hi: v }
    }

    fn map<F>(&self, mut f: F) -> Self
    where
        F: FnMut(NewId<f32>) -> NewId<f32>,
    {
        Self {
            lo: f(self.lo),
            hi: f(self.hi),
        }
    }

    fn values(&self) -> [NewId<f32>; 2] {
        [self.lo, self.hi]
    }
}

#[derive(Copy, Clone)]
struct BoolRange {
    all: NewId<bool>,
    any: NewId<bool>,
}

impl BoolRange {
    fn point(v: NewId<bool>) -> Self {
        Self { all: v, any: v }
    }

    fn definite(self) -> Option<NewId<bool>> {
        (self.all == self.any).then_some(self.all)
    }
}

#[derive(Copy, Clone)]
enum Value {
    I32(NewId<i32>),
    Bool(BoolRange),
    F32(F32Range),
}

impl From<BoolRange> for Value {
    fn from(range: BoolRange) -> Self {
        Self::Bool(range)
    }
}

impl From<F32Range> for Value {
    fn from(range: F32Range) -> Self {
        Self::F32(range)
    }
}

struct Translator<'a> {
    source: &'a Tape,
    builder: TapeBuilder,
    values: Vec<Value>,
}

impl<'a> Translator<'a> {
    fn new(source: &'a Tape) -> Self {
        assert!(
            source.arg_types().iter().all(|&ty| ty == Type::F32),
            "interval arguments must be f32"
        );

        assert!(
            source.result_types().iter().all(|&ty| ty == Type::F32),
            "interval results must be f32"
        );

        let num_args = source.num_args();
        let builder = Tape::builder(
            vec![Type::F32; 2 * num_args],
            vec![Type::F32; 2 * source.num_results()],
        );

        let values = (0..num_args)
            .map(|i| Value::F32(F32Range::new(builder.arg(2 * i), builder.arg(2 * i + 1))))
            .collect();

        Self {
            source,
            builder,
            values,
        }
    }

    fn run(mut self) -> Tape {
        for r in self.source.instrs() {
            let value = self.translate(r.instr());
            self.values.push(value);
        }

        let num_results = self.source.num_results();
        for &value in &self.values[self.values.len() - num_results..] {
            let Value::F32(range) = value else {
                unreachable!("interval results must be f32");
            };
            self.builder.copy_f32(range.lo);
            self.builder.copy_f32(range.hi);
        }

        self.builder.build().unwrap()
    }

    fn translate(&mut self, instr: Instr) -> Value {
        match instr {
            Instr::I32Const(c) => Value::I32(self.builder.i32_const(c)),
            Instr::BoolConst(c) => BoolRange::point(self.builder.bool_const(c)).into(),
            Instr::F32Const(c) => F32Range::point(self.builder.f32_const(c)).into(),

            Instr::I32Add(lhs, rhs) => {
                let (x, y) = (self.i32(lhs), self.i32(rhs));
                Value::I32(self.builder.i32_add(x, y))
            }

            Instr::I32Sub(lhs, rhs) => {
                let (x, y) = (self.i32(lhs), self.i32(rhs));
                Value::I32(self.builder.i32_sub(x, y))
            }

            Instr::I32Mul(lhs, rhs) => {
                let (x, y) = (self.i32(lhs), self.i32(rhs));
                Value::I32(self.builder.i32_mul(x, y))
            }

            Instr::I32Eq(lhs, rhs) => {
                let (x, y) = (self.i32(lhs), self.i32(rhs));
                BoolRange::point(self.builder.i32_eq(x, y)).into()
            }

            Instr::I32Ne(lhs, rhs) => {
                let (x, y) = (self.i32(lhs), self.i32(rhs));
                BoolRange::point(self.builder.i32_ne(x, y)).into()
            }

            Instr::I32Lt(lhs, rhs) => {
                let (x, y) = (self.i32(lhs), self.i32(rhs));
                BoolRange::point(self.builder.i32_lt(x, y)).into()
            }

            Instr::I32Le(lhs, rhs) => {
                let (x, y) = (self.i32(lhs), self.i32(rhs));
                BoolRange::point(self.builder.i32_le(x, y)).into()
            }

            Instr::I32Gt(lhs, rhs) => {
                let (x, y) = (self.i32(lhs), self.i32(rhs));
                BoolRange::point(self.builder.i32_gt(x, y)).into()
            }

            Instr::I32Ge(lhs, rhs) => {
                let (x, y) = (self.i32(lhs), self.i32(rhs));
                BoolRange::point(self.builder.i32_ge(x, y)).into()
            }

            Instr::Not(src) => {
                let x = self.bool(src);
                self.bool_not(x).into()
            }

            Instr::And(lhs, rhs) => {
                let (x, y) = (self.bool(lhs), self.bool(rhs));
                let all = self.builder.and(x.all, y.all);
                let any = if x.definite().is_some() && y.definite().is_some() {
                    all
                } else {
                    self.builder.and(x.any, y.any)
                };
                BoolRange { all, any }.into()
            }

            Instr::Or(lhs, rhs) => {
                let (x, y) = (self.bool(lhs), self.bool(rhs));
                let all = self.builder.or(x.all, y.all);
                let any = if x.definite().is_some() && y.definite().is_some() {
                    all
                } else {
                    self.builder.or(x.any, y.any)
                };
                BoolRange { all, any }.into()
            }

            Instr::Xor(lhs, rhs) => {
                let (x, y) = (self.bool(lhs), self.bool(rhs));
                let definite = self.builder.xor(x.all, y.all);
                if x.definite().is_some() && y.definite().is_some() {
                    BoolRange::point(definite).into()
                } else {
                    // Any uncertain operand makes the result fully uncertain.
                    let unc_x = self.builder.xor(x.all, x.any);
                    let unc_y = self.builder.xor(y.all, y.any);
                    let uncertain = self.builder.or(unc_x, unc_y);
                    let certain = self.builder.not(uncertain);
                    let all = self.builder.and(definite, certain);
                    let any = self.builder.or(definite, uncertain);
                    BoolRange { all, any }.into()
                }
            }

            Instr::CopyI32(src) => Value::I32(self.i32(src)),
            Instr::CopyBool(src) => self.bool(src).into(),
            Instr::CopyF32(src) => self.f32(src).into(),

            Instr::I32FromBool(src) => {
                let Some(src) = self.bool(src).definite() else {
                    unimplemented!(
                        "data-dependent i32 values are not supported in interval translation"
                    );
                };

                Value::I32(self.builder.i32_from_bool(src))
            }

            Instr::F32FromI32(src) => {
                let src = self.i32(src);
                F32Range::point(self.builder.f32_from_i32(src)).into()
            }

            Instr::F32FromBool(src) => {
                let x = self.bool(src);
                match x.definite() {
                    Some(v) => F32Range::point(self.builder.f32_from_bool(v)).into(),
                    None => {
                        let lo = self.builder.f32_from_bool(x.all);
                        let hi = self.builder.f32_from_bool(x.any);
                        F32Range::new(lo, hi).into()
                    }
                }
            }

            Instr::F32Neg(src) => {
                let x = self.f32(src);
                let [lo, hi] = x.values().map(|v| self.builder.f32_neg(v));
                F32Range::new(hi, lo).into()
            }

            Instr::F32Abs(src) => {
                // |[lo, hi]| = [max(lo, -hi, 0), max(hi, -lo)]
                let x = self.f32(src);
                let neg_lo = self.builder.f32_neg(x.lo);
                let neg_hi = self.builder.f32_neg(x.hi);
                let zero = self.builder.f32_const(0.0);
                let lo = self.max_of([x.lo, neg_hi, zero]);
                let hi = self.builder.f32_max(x.hi, neg_lo);
                F32Range::new(lo, hi).into()
            }

            Instr::F32Sign(src) => {
                let x = self.f32(src);
                x.map(|v| self.builder.f32_sign(v)).into()
            }

            Instr::F32Floor(src) => {
                let x = self.f32(src);
                x.map(|v| self.builder.f32_floor(v)).into()
            }

            Instr::F32Add(lhs, rhs) => {
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                F32Range::new(
                    self.builder.f32_add(x.lo, y.lo),
                    self.builder.f32_add(x.hi, y.hi),
                )
                .into()
            }

            Instr::F32Sub(lhs, rhs) => {
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                F32Range::new(
                    self.builder.f32_sub(x.lo, y.hi),
                    self.builder.f32_sub(x.hi, y.lo),
                )
                .into()
            }

            Instr::F32Mul(lhs, rhs) => {
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                self.mul_range(x, y).into()
            }

            Instr::F32Div(lhs, rhs) => {
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                let recip = self.recip(y);
                self.mul_range(x, recip).into()
            }

            Instr::F32Min(lhs, rhs) => {
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                F32Range::new(
                    self.builder.f32_min(x.lo, y.lo),
                    self.builder.f32_min(x.hi, y.hi),
                )
                .into()
            }

            Instr::F32Max(lhs, rhs) => {
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                F32Range::new(
                    self.builder.f32_max(x.lo, y.lo),
                    self.builder.f32_max(x.hi, y.hi),
                )
                .into()
            }

            Instr::F32Powf(lhs, rhs) => {
                // lhs^rhs = exp(rhs * ln(lhs))
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                let clamped = self.ensure_positive_safe(x);
                let ln = clamped.map(|v| self.builder.f32_ln(v));
                let exponent = self.mul_range(ln, y);
                exponent.map(|v| self.builder.f32_exp(v)).into()
            }

            Instr::F32Powi(lhs, rhs) => {
                let x = self.f32(lhs);
                let n = self.i32(rhs);
                let nonzero = self.ensure_nonzero(x);
                let [a, b] = nonzero.values().map(|v| self.builder.f32_powi(v, n));
                let range = self.f32_hull(a, b);

                // `range` might miss zero for even powers, e.g. [-1, 2]^2 = [1, 4].
                let zero_in_base = self.contains_zero(x);
                let range = self.include_zero_if(range, zero_in_base);

                // Negative powers blow up around zero.
                let zero = self.builder.i32_const(0);
                let negative_power = self.builder.i32_lt(n, zero);
                let pole = self.builder.and(zero_in_base, negative_power);
                self.widen_if(range, pole).into()
            }

            Instr::F32Exp(src) => {
                let x = self.f32(src);
                x.map(|v| self.builder.f32_exp(v)).into()
            }

            Instr::F32Ln(src) => {
                let x = self.f32(src);
                let clamped = self.ensure_positive_safe(x);
                clamped.map(|v| self.builder.f32_ln(v)).into()
            }

            Instr::F32Lg(src) => {
                let x = self.f32(src);
                let clamped = self.ensure_positive_safe(x);
                clamped.map(|v| self.builder.f32_lg(v)).into()
            }

            Instr::F32Sin(src) => {
                let x = self.f32(src);
                let [a, b] = x.values().map(|v| self.builder.f32_sin(v));
                let range = self.f32_hull(a, b);
                self.normalized_periodic_wave(x, range, FRAC_PI_2, PI)
                    .into()
            }

            Instr::F32Cos(src) => {
                let x = self.f32(src);
                let [a, b] = x.values().map(|v| self.builder.f32_cos(v));
                let range = self.f32_hull(a, b);
                self.normalized_periodic_wave(x, range, 0.0, PI).into()
            }

            Instr::F32Tan(src) => {
                let x = self.f32(src);
                let [a, b] = x.values().map(|v| self.builder.f32_tan(v));
                let range = self.f32_hull(a, b);
                self.periodic_monotone(x, range, FRAC_PI_2, PI).into()
            }

            Instr::F32Cot(src) => {
                let x = self.f32(src);
                let [a, b] = x.values().map(|v| self.builder.f32_cot(v));
                let range = self.f32_hull(a, b);
                self.periodic_monotone(x, range, 0.0, PI).into()
            }

            Instr::F32Eq(lhs, rhs) => {
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                self.f32_eq(x, y).into()
            }

            Instr::F32Ne(lhs, rhs) => {
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                let eq = self.f32_eq(x, y);
                self.bool_not(eq).into()
            }

            Instr::F32Lt(lhs, rhs) => {
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                self.f32_lt(x, y).into()
            }

            Instr::F32Le(lhs, rhs) => {
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                self.f32_le(x, y).into()
            }

            Instr::F32Gt(lhs, rhs) => {
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                self.f32_lt(y, x).into()
            }

            Instr::F32Ge(lhs, rhs) => {
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                self.f32_le(y, x).into()
            }

            Instr::I32Sel(cond, v_true, v_false) => {
                let Some(cond) = self.bool(cond).definite() else {
                    unimplemented!(
                        "data-dependent i32 values are not supported in interval translation"
                    );
                };
                let (t, f) = (self.i32(v_true), self.i32(v_false));
                Value::I32(self.builder.i32_sel(cond, t, f))
            }

            Instr::F32Sel(cond, v_true, v_false) => {
                let cond = self.bool(cond);
                let (t, f) = (self.f32(v_true), self.f32(v_false));
                match cond.definite() {
                    Some(cond) => self.select_range(cond, t, f).into(),
                    None => {
                        // An undecided condition yields the hull of both branches.
                        let merged = F32Range::new(
                            self.builder.f32_min(t.lo, f.lo),
                            self.builder.f32_max(t.hi, f.hi),
                        );
                        let unless_true = self.select_range(cond.any, merged, f);
                        self.select_range(cond.all, t, unless_true).into()
                    }
                }
            }
        }
    }

    fn i32(&self, id: ValueId<i32>) -> NewId<i32> {
        match self.values[id.index()] {
            Value::I32(v) => v,
            _ => unreachable!("expected an i32 value"),
        }
    }

    fn bool(&self, id: ValueId<bool>) -> BoolRange {
        match self.values[id.index()] {
            Value::Bool(range) => range,
            _ => unreachable!("expected a bool value"),
        }
    }

    fn f32(&self, id: ValueId<f32>) -> F32Range {
        match self.values[id.index()] {
            Value::F32(range) => range,
            _ => unreachable!("expected an f32 value"),
        }
    }

    fn f32_hull(&mut self, a: NewId<f32>, b: NewId<f32>) -> F32Range {
        F32Range::new(self.builder.f32_min(a, b), self.builder.f32_max(a, b))
    }

    fn min_of(&mut self, values: impl IntoIterator<Item = NewId<f32>>) -> NewId<f32> {
        values
            .into_iter()
            .reduce(|x, y| self.builder.f32_min(x, y))
            .unwrap()
    }

    fn max_of(&mut self, values: impl IntoIterator<Item = NewId<f32>>) -> NewId<f32> {
        values
            .into_iter()
            .reduce(|x, y| self.builder.f32_max(x, y))
            .unwrap()
    }

    fn bool_not(&mut self, x: BoolRange) -> BoolRange {
        let all = self.builder.not(x.any);
        let any = match x.definite() {
            Some(_) => all,
            None => self.builder.not(x.all),
        };
        BoolRange { all, any }
    }

    fn f32_lt(&mut self, x: F32Range, y: F32Range) -> BoolRange {
        BoolRange {
            all: self.builder.f32_lt(x.hi, y.lo),
            any: self.builder.f32_lt(x.lo, y.hi),
        }
    }

    fn f32_le(&mut self, x: F32Range, y: F32Range) -> BoolRange {
        BoolRange {
            all: self.builder.f32_le(x.hi, y.lo),
            any: self.builder.f32_le(x.lo, y.hi),
        }
    }

    fn f32_eq(&mut self, x: F32Range, y: F32Range) -> BoolRange {
        let xy = self.f32_le(x, y);
        let yx = self.f32_le(y, x);
        BoolRange {
            all: self.builder.and(xy.all, yx.all),
            any: self.builder.and(xy.any, yx.any),
        }
    }

    fn contains_zero(&mut self, x: F32Range) -> NewId<bool> {
        let signs = x.map(|v| self.builder.f32_sign(v));
        self.builder.f32_lt(signs.lo, signs.hi)
    }

    fn include_zero_if(&mut self, x: F32Range, cond: NewId<bool>) -> F32Range {
        let zero = self.builder.f32_const(0.0);
        let extended = F32Range::new(
            self.builder.f32_min(x.lo, zero),
            self.builder.f32_max(x.hi, zero),
        );
        self.select_range(cond, extended, x)
    }

    fn widen_if(&mut self, x: F32Range, cond: NewId<bool>) -> F32Range {
        let wide = F32Range::new(
            self.builder.f32_const(-f32::MAX),
            self.builder.f32_const(f32::MAX),
        );
        self.select_range(cond, wide, x)
    }

    fn mul_range(&mut self, x: F32Range, y: F32Range) -> F32Range {
        let products = [
            self.builder.f32_mul(x.lo, y.lo),
            self.builder.f32_mul(x.lo, y.hi),
            self.builder.f32_mul(x.hi, y.lo),
            self.builder.f32_mul(x.hi, y.hi),
        ];
        F32Range::new(self.min_of(products), self.max_of(products))
    }

    fn ensure_magnitude(&mut self, v: NewId<f32>, min: NewId<f32>) -> NewId<f32> {
        let sign = self.builder.f32_sign(v);
        let magnitude = self.builder.f32_abs(v);
        let clamped = self.builder.f32_max(magnitude, min);
        self.builder.f32_mul(sign, clamped)
    }

    fn ensure_nonzero(&mut self, x: F32Range) -> F32Range {
        let tiny = self.builder.f32_const(f32::MIN_POSITIVE);
        x.map(|v| self.ensure_magnitude(v, tiny))
    }

    fn recip(&mut self, x: F32Range) -> F32Range {
        let one = self.builder.f32_const(1.0);
        let nonzero = self.ensure_nonzero(x);
        let [a, b] = nonzero.values().map(|v| self.builder.f32_div(one, v));
        let range = self.f32_hull(a, b);
        let pole = self.contains_zero(x);
        self.widen_if(range, pole)
    }

    fn ensure_positive_safe(&mut self, x: F32Range) -> F32Range {
        let tiny = self.builder.f32_const(f32::MIN_POSITIVE);
        x.map(|v| self.builder.f32_max(v, tiny))
    }

    fn select_range(
        &mut self,
        cond: NewId<bool>,
        if_true: F32Range,
        if_false: F32Range,
    ) -> F32Range {
        F32Range::new(
            self.builder.f32_sel(cond, if_true.lo, if_false.lo),
            self.builder.f32_sel(cond, if_true.hi, if_false.hi),
        )
    }

    fn period_index(&mut self, v: NewId<f32>, offset: NewId<f32>, scale: NewId<f32>) -> NewId<f32> {
        let shifted = self.builder.f32_sub(v, offset);
        let scaled = self.builder.f32_mul(shifted, scale);
        self.builder.f32_floor(scaled)
    }

    /// Returns whether the range contains a point equal to `v` modulo `period`.
    fn contains_modulo(&mut self, x: F32Range, v: f32, period: f32) -> NewId<bool> {
        let offset = self.builder.f32_const(v);
        let scale = self.builder.f32_const(period.recip());
        let first = self.period_index(x.lo, offset, scale);
        let last = self.period_index(x.hi, offset, scale);
        let count = self.builder.f32_sub(last, first);
        let half = self.builder.f32_const(0.5);
        self.builder.f32_gt(count, half)
    }

    // Periodic waves reaching 1 at `peak` and -1 at `peak + half_period`, repeating every `2 * half_period`
    fn normalized_periodic_wave(
        &mut self,
        x: F32Range,
        im: F32Range,
        peak: f32,
        half_period: f32,
    ) -> F32Range {
        let has_max = self.contains_modulo(x, peak, 2.0 * half_period);
        let has_min = self.contains_modulo(x, peak + half_period, 2.0 * half_period);
        let one = self.builder.f32_const(1.0);
        let neg_one = self.builder.f32_const(-1.0);
        let hi = self.builder.f32_sel(has_max, one, im.hi);
        let lo = self.builder.f32_sel(has_min, neg_one, im.lo);
        F32Range::new(lo, hi)
    }

    // Periodic functions monotone between consecutive poles, at `pole` modulo `period`
    fn periodic_monotone(&mut self, x: F32Range, im: F32Range, pole: f32, period: f32) -> F32Range {
        let has_pole = self.contains_modulo(x, pole, period);
        self.widen_if(im, has_pole)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Fallback, Instance};

    fn eval<const N: usize>(tape: &Tape, args: &[Interval; N]) -> Interval {
        let tape = translate(tape);
        assert_eq!(tape.num_args(), 2 * N);
        assert_eq!(tape.num_results(), 2);

        Instance::<Fallback, [Interval; N], Interval>::new(tape)
            .evaluator()
            .evaluate(args)
    }

    fn interval(min: f32, max: f32) -> Interval {
        Interval::new(min, max)
    }

    fn unary(op: fn(&mut TapeBuilder, NewId<f32>) -> NewId<f32>) -> Tape {
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        op(&mut b, x);
        b.build().unwrap()
    }

    fn binary(op: fn(&mut TapeBuilder, NewId<f32>, NewId<f32>) -> NewId<f32>) -> Tape {
        let mut b = Tape::builder(vec![Type::F32; 2], vec![Type::F32]);
        let (x, y) = (b.arg(0), b.arg(1));
        op(&mut b, x, y);
        b.build().unwrap()
    }

    fn powi(n: i32) -> Tape {
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let n = b.i32_const(n);
        b.f32_powi(x, n);
        b.build().unwrap()
    }

    fn assert_close(actual: Interval, expected: Interval) {
        assert!(
            (actual.lo - expected.lo).abs() < 1e-5 && (actual.hi - expected.hi).abs() < 1e-5,
            "expected {expected:?}, got {actual:?}"
        );
    }

    fn assert_wide(actual: Interval) {
        assert!(actual.lo <= -1e30 && actual.hi >= 1e30, "{actual:?}");
    }

    #[test]
    fn arithmetic() {
        let (x, y) = (interval(-1.0, 2.0), interval(3.0, 5.0));
        assert_eq!(
            eval(&binary(TapeBuilder::f32_add), &[x, y]),
            interval(2.0, 7.0)
        );
        assert_eq!(
            eval(&binary(TapeBuilder::f32_sub), &[x, y]),
            interval(-6.0, -1.0)
        );
        assert_eq!(
            eval(&binary(TapeBuilder::f32_mul), &[x, y]),
            interval(-5.0, 10.0)
        );
        assert_eq!(
            eval(&binary(TapeBuilder::f32_mul), &[x, x]),
            interval(-2.0, 4.0)
        );
        assert_eq!(
            eval(&binary(TapeBuilder::f32_min), &[x, y]),
            interval(-1.0, 2.0)
        );
        assert_eq!(
            eval(&binary(TapeBuilder::f32_max), &[x, y]),
            interval(3.0, 5.0)
        );
        assert_eq!(
            eval(&unary(TapeBuilder::f32_neg), &[x]),
            interval(-2.0, 1.0)
        );
        assert_eq!(eval(&unary(TapeBuilder::f32_abs), &[x]), interval(0.0, 2.0));
        assert_eq!(
            eval(&unary(TapeBuilder::f32_abs), &[interval(-3.0, -1.0)]),
            interval(1.0, 3.0)
        );
        assert_eq!(
            eval(&unary(TapeBuilder::f32_sign), &[x]),
            interval(-1.0, 1.0)
        );
        assert_eq!(
            eval(&unary(TapeBuilder::f32_sign), &[y]),
            interval(1.0, 1.0)
        );
        assert_eq!(
            eval(&unary(TapeBuilder::f32_floor), &[interval(-1.5, 2.3)]),
            interval(-2.0, 2.0)
        );
    }

    #[test]
    fn division() {
        let tape = binary(TapeBuilder::f32_div);
        assert_eq!(
            eval(&tape, &[interval(1.0, 2.0), interval(2.0, 4.0)]),
            interval(0.25, 1.0)
        );
        assert_eq!(
            eval(&tape, &[interval(1.0, 2.0), interval(-4.0, -2.0)]),
            interval(-1.0, -0.25)
        );
        assert_wide(eval(&tape, &[interval(1.0, 2.0), interval(-1.0, 1.0)]));
    }

    #[test]
    fn integer_powers() {
        let square = powi(2);
        assert_eq!(eval(&square, &[interval(-1.0, 2.0)]), interval(0.0, 4.0));
        assert_eq!(eval(&square, &[interval(1.0, 2.0)]), interval(1.0, 4.0));
        assert_eq!(eval(&square, &[interval(-3.0, -2.0)]), interval(4.0, 9.0));

        let cube = powi(3);
        assert_eq!(eval(&cube, &[interval(-1.0, 2.0)]), interval(-1.0, 8.0));

        let inverse = powi(-1);
        assert_eq!(eval(&inverse, &[interval(1.0, 2.0)]), interval(0.5, 1.0));
        assert_wide(eval(&inverse, &[interval(-1.0, 2.0)]));

        let touching = eval(&inverse, &[interval(0.0, 2.0)]);
        assert!(
            touching.lo == 0.5 && touching.hi >= 1e30 && touching.hi.is_finite(),
            "{touching:?}"
        );
    }

    #[test]
    fn real_powers() {
        let tape = binary(TapeBuilder::f32_powf);
        assert_close(
            eval(&tape, &[interval(1.0, 2.0), interval(2.0, 3.0)]),
            interval(1.0, 8.0),
        );
        assert_close(
            eval(&tape, &[interval(2.0, 4.0), interval(-1.0, 0.5)]),
            interval(0.25, 2.0),
        );
    }

    #[test]
    fn transcendental() {
        assert_close(
            eval(&unary(TapeBuilder::f32_exp), &[interval(0.0, 1.0)]),
            interval(1.0, 1f32.exp()),
        );
        assert_close(
            eval(&unary(TapeBuilder::f32_ln), &[interval(1.0, 4.0)]),
            interval(0.0, 4f32.ln()),
        );
        assert_close(
            eval(&unary(TapeBuilder::f32_lg), &[interval(1.0, 100.0)]),
            interval(0.0, 2.0),
        );

        let lowest = f32::MIN_POSITIVE.ln();
        assert_close(
            eval(&unary(TapeBuilder::f32_ln), &[interval(-1.0, 4.0)]),
            interval(lowest, 4f32.ln()),
        );
        assert_close(
            eval(&unary(TapeBuilder::f32_ln), &[interval(-2.0, -1.0)]),
            interval(lowest, lowest),
        );

        let sin = unary(TapeBuilder::f32_sin);
        assert_close(
            eval(&sin, &[interval(0.0, 0.2)]),
            interval(0.0, 0.2f32.sin()),
        );
        assert_close(eval(&sin, &[interval(1.0, 2.0)]), interval(1f32.sin(), 1.0));
        assert_close(
            eval(&sin, &[interval(3.0, 5.0)]),
            interval(-1.0, 3f32.sin()),
        );
        assert_eq!(eval(&sin, &[interval(-100.0, 100.0)]), interval(-1.0, 1.0));

        let cos = unary(TapeBuilder::f32_cos);
        assert_close(
            eval(&cos, &[interval(-0.1, 0.1)]),
            interval(0.1f32.cos(), 1.0),
        );
        assert_close(
            eval(&cos, &[interval(1.0, 2.0)]),
            interval(2f32.cos(), 1f32.cos()),
        );
        assert_close(eval(&cos, &[interval(3.0, 7.0)]), interval(-1.0, 1.0));
    }

    #[test]
    fn tangent() {
        let tan = unary(TapeBuilder::f32_tan);
        assert_close(eval(&tan, &[interval(0.0, 1.0)]), interval(0.0, 1f32.tan()));
        assert_close(
            eval(&tan, &[interval(-1.0, -0.5)]),
            interval((-1f32).tan(), (-0.5f32).tan()),
        );
        assert_wide(eval(&tan, &[interval(1.0, 2.0)]));
        assert_wide(eval(&tan, &[interval(0.0, 4.0)]));

        let cot = unary(TapeBuilder::f32_cot);
        assert_close(
            eval(&cot, &[interval(0.5, 1.0)]),
            interval(1f32.tan().recip(), 0.5f32.tan().recip()),
        );
        assert_wide(eval(&cot, &[interval(-0.5, 0.5)]));
    }

    #[test]
    fn select() {
        // f = if x < y { x } else { y }
        let mut b = Tape::builder(vec![Type::F32; 2], vec![Type::F32]);
        let (x, y) = (b.arg(0), b.arg(1));
        let lt = b.f32_lt(x, y);
        b.f32_sel(lt, x, y);
        let tape = b.build().unwrap();

        // The condition is decided when the ranges are disjoint.
        assert_eq!(
            eval(&tape, &[interval(0.0, 1.0), interval(2.0, 3.0)]),
            interval(0.0, 1.0)
        );
        assert_eq!(
            eval(&tape, &[interval(4.0, 5.0), interval(2.0, 3.0)]),
            interval(2.0, 3.0)
        );

        // Overlapping ranges leave it undecided, so the branches merge.
        assert_eq!(
            eval(&tape, &[interval(0.0, 3.0), interval(2.0, 5.0)]),
            interval(0.0, 5.0)
        );
    }

    #[test]
    fn definite_conditions() {
        // f = x * float(if 1 < 2 { 3 } else { 4 })
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let one = b.i32_const(1);
        let two = b.i32_const(2);
        let lt = b.i32_lt(one, two);
        let three = b.i32_const(3);
        let four = b.i32_const(4);
        let sel = b.i32_sel(lt, three, four);
        let scale = b.f32_from_i32(sel);
        b.f32_mul(x, scale);
        let tape = b.build().unwrap();

        assert_eq!(eval(&tape, &[interval(-1.0, 2.0)]), interval(-3.0, 6.0));
    }

    #[test]
    fn mixed_i32() {
        // f = x * float(2 + 1)
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let two = b.i32_const(2);
        let one = b.i32_const(1);
        let three = b.i32_add(two, one);
        let three = b.f32_from_i32(three);
        b.f32_mul(x, three);
        let tape = b.build().unwrap();

        assert_eq!(eval(&tape, &[interval(-1.0, 2.0)]), interval(-3.0, 6.0));
    }
}
