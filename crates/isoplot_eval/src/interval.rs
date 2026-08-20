use std::{
    f32::consts::{FRAC_PI_2, PI},
    iter,
};

use bytemuck::{Pod, Zeroable};

use crate::{
    autodiff::Gradient,
    layout::{ValueType, Vector},
    program::Program,
    tape::{Instr, Tape, TapeBuilder, Type, ValueId},
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
struct ValueRange {
    lo: ValueId,
    hi: ValueId,
}

impl ValueRange {
    fn new(lo: ValueId, hi: ValueId) -> Self {
        Self { lo, hi }
    }

    fn point(v: ValueId) -> Self {
        Self { lo: v, hi: v }
    }
}

#[derive(Copy, Clone)]
enum Value {
    I32(ValueId),
    F32(ValueRange),
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
            .map(|i| Value::F32(ValueRange::new(builder.arg(2 * i), builder.arg(2 * i + 1))))
            .collect();

        Self {
            source,
            builder,
            values,
        }
    }

    fn run(mut self) -> Tape {
        for &instr in self.source.instrs() {
            let value = self.translate(instr);
            self.values.push(value);
        }

        let num_results = self.source.num_results();
        for &value in &self.values[self.values.len() - num_results..] {
            let Value::F32(range) = value else {
                unreachable!("interval results must be f32");
            };
            self.builder.instr(Instr::Copy(range.lo));
            self.builder.instr(Instr::Copy(range.hi));
        }

        self.builder.build().unwrap()
    }

    fn translate(&mut self, instr: Instr) -> Value {
        match instr {
            Instr::I32Const(_) => Value::I32(self.builder.instr(instr)),
            Instr::F32Const(_) => Value::F32(ValueRange::point(self.builder.instr(instr))),

            Instr::I32Add(lhs, rhs) => Value::I32(self.i32_binary(Instr::I32Add, lhs, rhs)),
            Instr::I32Sub(lhs, rhs) => Value::I32(self.i32_binary(Instr::I32Sub, lhs, rhs)),
            Instr::I32Mul(lhs, rhs) => Value::I32(self.i32_binary(Instr::I32Mul, lhs, rhs)),

            Instr::Copy(src) => self.values[src.index()],

            Instr::F32FromI32(src) => {
                let src = self.i32(src);
                let converted = self.builder.instr(Instr::F32FromI32(src));
                Value::F32(ValueRange::point(converted))
            }

            Instr::F32Neg(src) => {
                let x = self.f32(src);
                let negated = self.map(x, Instr::F32Neg);
                Value::F32(ValueRange::new(negated.hi, negated.lo))
            }

            Instr::F32Abs(src) => {
                // |[lo, hi]| = [max(lo, -hi, 0), max(hi, -lo)]
                let x = self.f32(src);
                let neg_lo = self.unary(Instr::F32Neg, x.lo);
                let neg_hi = self.unary(Instr::F32Neg, x.hi);
                let zero = self.c_f32(0.0);
                let lo = self.max_of([x.lo, neg_hi, zero]);
                let hi = self.max(x.hi, neg_lo);
                Value::F32(ValueRange::new(lo, hi))
            }

            Instr::F32Sign(src) => self.monotone(Instr::F32Sign, src),
            Instr::F32Floor(src) => self.monotone(Instr::F32Floor, src),

            Instr::F32Add(lhs, rhs) => {
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                Value::F32(ValueRange::new(self.add(x.lo, y.lo), self.add(x.hi, y.hi)))
            }

            Instr::F32Sub(lhs, rhs) => {
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                Value::F32(ValueRange::new(self.sub(x.lo, y.hi), self.sub(x.hi, y.lo)))
            }

            Instr::F32Mul(lhs, rhs) => {
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                Value::F32(self.mul_range(x, y))
            }

            Instr::F32Div(lhs, rhs) => {
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                let recip = self.recip(y);
                Value::F32(self.mul_range(x, recip))
            }

            Instr::F32Min(lhs, rhs) => {
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                Value::F32(ValueRange::new(self.min(x.lo, y.lo), self.min(x.hi, y.hi)))
            }

            Instr::F32Max(lhs, rhs) => {
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                Value::F32(ValueRange::new(self.max(x.lo, y.lo), self.max(x.hi, y.hi)))
            }

            Instr::F32Powf(lhs, rhs) => {
                // lhs^rhs = exp(rhs * ln(lhs))
                let (x, y) = (self.f32(lhs), self.f32(rhs));
                let ln = self.log(Instr::F32Ln, x);
                let exponent = self.mul_range(ln, y);
                Value::F32(self.map(exponent, Instr::F32Exp))
            }

            Instr::F32Powi(lhs, rhs) => {
                let x = self.f32(lhs);
                let n = self.i32(rhs);
                let tiny = self.c_f32(f32::MIN_POSITIVE);
                let lo = self.ensure_magnitude(x.lo, tiny);
                let hi = self.ensure_magnitude(x.hi, tiny);
                let a = self.builder.instr(Instr::F32Powi(lo, n));
                let b = self.builder.instr(Instr::F32Powi(hi, n));
                let range = self.hull(a, b);

                // `range` might miss zero for even powers, e.g. [-1, 2]^2 = [1, 4].
                let zero = self.contains_zero(x);
                let range = self.include_zero_if(range, zero);

                // Negative powers blow up around zero.
                let n_f32 = self.builder.instr(Instr::F32FromI32(n));
                let negative_power = self.is_negative(n_f32);
                let pole = self.mul(zero, negative_power);
                Value::F32(self.widen_if(range, pole))
            }

            Instr::F32Exp(src) => self.monotone(Instr::F32Exp, src),
            Instr::F32Ln(src) => self.monotone_log(Instr::F32Ln, src),
            Instr::F32Lg(src) => self.monotone_log(Instr::F32Lg, src),

            Instr::F32Sin(src) => self.periodic_wave(Instr::F32Sin, FRAC_PI_2, src),
            Instr::F32Cos(src) => self.periodic_wave(Instr::F32Cos, 0.0, src),
            Instr::F32Tan(src) => self.periodic_monotone(Instr::F32Tan, FRAC_PI_2, src),
            Instr::F32Cot(src) => self.periodic_monotone(Instr::F32Cot, 0.0, src),
        }
    }

    fn i32(&self, id: ValueId) -> ValueId {
        match self.values[id.index()] {
            Value::I32(v) => v,
            Value::F32(_) => unreachable!("expected an i32 value"),
        }
    }

    fn f32(&self, id: ValueId) -> ValueRange {
        match self.values[id.index()] {
            Value::F32(range) => range,
            Value::I32(_) => unreachable!("expected an f32 value"),
        }
    }

    fn i32_binary(
        &mut self,
        op: impl FnOnce(ValueId, ValueId) -> Instr,
        lhs: ValueId,
        rhs: ValueId,
    ) -> ValueId {
        let (lhs, rhs) = (self.i32(lhs), self.i32(rhs));
        self.builder.instr(op(lhs, rhs))
    }

    fn unary(&mut self, f: impl FnOnce(ValueId) -> Instr, src: ValueId) -> ValueId {
        self.builder.instr(f(src))
    }

    /// Applies a non-decreasing function to both endpoints.
    fn map(&mut self, x: ValueRange, mut op: impl FnMut(ValueId) -> Instr) -> ValueRange {
        ValueRange::new(self.unary(&mut op, x.lo), self.unary(&mut op, x.hi))
    }

    fn map_hull(&mut self, x: ValueRange, mut f: impl FnMut(ValueId) -> Instr) -> ValueRange {
        let (a, b) = (self.unary(&mut f, x.lo), self.unary(&mut f, x.hi));
        self.hull(a, b)
    }

    fn hull(&mut self, a: ValueId, b: ValueId) -> ValueRange {
        ValueRange::new(self.min(a, b), self.max(a, b))
    }

    fn c_f32(&mut self, value: f32) -> ValueId {
        self.builder.instr(Instr::F32Const(value))
    }

    fn add(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.builder.instr(Instr::F32Add(lhs, rhs))
    }

    fn sub(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.builder.instr(Instr::F32Sub(lhs, rhs))
    }

    fn mul(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.builder.instr(Instr::F32Mul(lhs, rhs))
    }

    fn min(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.builder.instr(Instr::F32Min(lhs, rhs))
    }

    fn max(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.builder.instr(Instr::F32Max(lhs, rhs))
    }

    fn min_of(&mut self, values: impl IntoIterator<Item = ValueId>) -> ValueId {
        values.into_iter().reduce(|x, y| self.min(x, y)).unwrap()
    }

    fn max_of(&mut self, values: impl IntoIterator<Item = ValueId>) -> ValueId {
        values.into_iter().reduce(|x, y| self.max(x, y)).unwrap()
    }

    fn one_minus(&mut self, v: ValueId) -> ValueId {
        let one = self.c_f32(1.0);
        self.sub(one, v)
    }

    /// Returns 1.0 if `v` is negative, 0.0 otherwise.
    fn is_negative(&mut self, v: ValueId) -> ValueId {
        let sign = self.unary(Instr::F32Sign, v);
        let zero = self.c_f32(0.0);
        let clamped = self.min(sign, zero);
        self.unary(Instr::F32Neg, clamped)
    }

    /// Returns 1.0 if the range contains zero, 0.0 otherwise.
    fn contains_zero(&mut self, x: ValueRange) -> ValueId {
        let signs = self.map(x, Instr::F32Sign);
        let product = self.mul(signs.lo, signs.hi);
        self.is_negative(product)
    }

    /// Extends the range to include zero when `cond` is 1.0.
    ///
    /// This method *must* only be called when `cond` is either `0.0` or `1.0`.
    fn include_zero_if(&mut self, x: ValueRange, cond: ValueId) -> ValueRange {
        let keep = self.one_minus(cond);
        let (lo_or_zero, hi_or_zero) = (self.mul(x.lo, keep), self.mul(x.hi, keep));
        ValueRange::new(self.min(x.lo, lo_or_zero), self.max(x.hi, hi_or_zero))
    }

    /// Extends the range to cover the entire finite `f32` range when `cond` is 1.0.
    ///
    /// This method *must* only be called when `cond` is either `0.0` or `1.0`.
    fn widen_if(&mut self, x: ValueRange, cond: ValueId) -> ValueRange {
        let huge = self.c_f32(f32::MAX);
        let neg_huge = self.c_f32(-f32::MAX);
        let delta = self.mul(cond, huge);
        let lo = self.sub(x.lo, delta);
        let hi = self.add(x.hi, delta);
        ValueRange::new(self.max(lo, neg_huge), self.min(hi, huge))
    }

    fn mul_range(&mut self, x: ValueRange, y: ValueRange) -> ValueRange {
        let products = [
            self.mul(x.lo, y.lo),
            self.mul(x.lo, y.hi),
            self.mul(x.hi, y.lo),
            self.mul(x.hi, y.hi),
        ];
        ValueRange::new(self.min_of(products), self.max_of(products))
    }

    fn ensure_magnitude(&mut self, v: ValueId, tiny: ValueId) -> ValueId {
        let sign = self.unary(Instr::F32Sign, v);
        let magnitude = self.unary(Instr::F32Abs, v);
        let clamped = self.max(magnitude, tiny);
        self.mul(sign, clamped)
    }

    fn recip(&mut self, x: ValueRange) -> ValueRange {
        let one = self.c_f32(1.0);
        let tiny = self.c_f32(f32::MIN_POSITIVE);
        let lo = self.ensure_magnitude(x.lo, tiny);
        let hi = self.ensure_magnitude(x.hi, tiny);
        let a = self.builder.instr(Instr::F32Div(one, lo));
        let b = self.builder.instr(Instr::F32Div(one, hi));
        let range = self.hull(a, b);
        let pole = self.contains_zero(x);
        self.widen_if(range, pole)
    }

    fn log(&mut self, op: impl FnMut(ValueId) -> Instr, x: ValueRange) -> ValueRange {
        let tiny = self.c_f32(f32::MIN_POSITIVE);
        let clamped = ValueRange::new(self.max(x.lo, tiny), self.max(x.hi, tiny));
        self.map(clamped, op)
    }

    fn monotone(&mut self, op: impl FnMut(ValueId) -> Instr, src: ValueId) -> Value {
        let x = self.f32(src);
        Value::F32(self.map(x, op))
    }

    fn monotone_log(&mut self, op: impl FnMut(ValueId) -> Instr, src: ValueId) -> Value {
        let x = self.f32(src);
        Value::F32(self.log(op, x))
    }

    /// Returns `a` when `cond` is 0.0 and `b` when it is 1.0.
    fn select(&mut self, a: ValueId, b: ValueId, cond: ValueId) -> ValueId {
        let diff = self.sub(b, a);
        let step = self.mul(cond, diff);
        self.add(a, step)
    }

    fn period_index(&mut self, v: ValueId, offset: ValueId, scale: ValueId) -> ValueId {
        let shifted = self.sub(v, offset);
        let scaled = self.mul(shifted, scale);
        self.unary(Instr::F32Floor, scaled)
    }

    /// Returns 1.0 if the range contains a point equal to `offset` modulo `period`, 0.0 otherwise.
    fn crosses(&mut self, x: ValueRange, offset: f32, period: f32) -> ValueId {
        let offset = self.c_f32(offset);
        let scale = self.c_f32(period.recip());
        let first = self.period_index(x.lo, offset, scale);
        let last = self.period_index(x.hi, offset, scale);
        let count = self.sub(last, first);
        let half = self.c_f32(0.5);
        let rem = self.sub(half, count);
        self.is_negative(rem)
    }

    // Periodic functions reaching 1 at `peak` and -1 at `peak + PI`, both modulo `2 * PI`
    fn periodic_wave(
        &mut self,
        op: impl FnMut(ValueId) -> Instr,
        peak: f32,
        src: ValueId,
    ) -> Value {
        let x = self.f32(src);
        let range = self.map_hull(x, op);
        let has_max = self.crosses(x, peak, 2.0 * PI);
        let has_min = self.crosses(x, peak + PI, 2.0 * PI);
        let one = self.c_f32(1.0);
        let neg_one = self.c_f32(-1.0);
        let hi = self.select(range.hi, one, has_max);
        let lo = self.select(range.lo, neg_one, has_min);
        Value::F32(ValueRange::new(lo, hi))
    }

    // Periodic functions monotone between poles spaced `PI` apart, at `pole` modulo `PI`
    fn periodic_monotone(
        &mut self,
        op: impl FnMut(ValueId) -> Instr,
        pole: f32,
        src: ValueId,
    ) -> Value {
        let x = self.f32(src);
        let range = self.map_hull(x, op);
        let has_pole = self.crosses(x, pole, PI);
        Value::F32(self.widen_if(range, has_pole))
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

    fn unary(op: fn(ValueId) -> Instr) -> Tape {
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        b.instr(op(x));
        b.build().unwrap()
    }

    fn binary(op: fn(ValueId, ValueId) -> Instr) -> Tape {
        let mut b = Tape::builder(vec![Type::F32; 2], vec![Type::F32]);
        let (x, y) = (b.arg(0), b.arg(1));
        b.instr(op(x, y));
        b.build().unwrap()
    }

    fn powi(n: i32) -> Tape {
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let n = b.instr(Instr::I32Const(n));
        b.instr(Instr::F32Powi(x, n));
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
        assert_eq!(eval(&binary(Instr::F32Add), &[x, y]), interval(2.0, 7.0));
        assert_eq!(eval(&binary(Instr::F32Sub), &[x, y]), interval(-6.0, -1.0));
        assert_eq!(eval(&binary(Instr::F32Mul), &[x, y]), interval(-5.0, 10.0));
        assert_eq!(eval(&binary(Instr::F32Mul), &[x, x]), interval(-2.0, 4.0));
        assert_eq!(eval(&binary(Instr::F32Min), &[x, y]), interval(-1.0, 2.0));
        assert_eq!(eval(&binary(Instr::F32Max), &[x, y]), interval(3.0, 5.0));
        assert_eq!(eval(&unary(Instr::F32Neg), &[x]), interval(-2.0, 1.0));
        assert_eq!(eval(&unary(Instr::F32Abs), &[x]), interval(0.0, 2.0));
        assert_eq!(
            eval(&unary(Instr::F32Abs), &[interval(-3.0, -1.0)]),
            interval(1.0, 3.0)
        );
        assert_eq!(eval(&unary(Instr::F32Sign), &[x]), interval(-1.0, 1.0));
        assert_eq!(eval(&unary(Instr::F32Sign), &[y]), interval(1.0, 1.0));
        assert_eq!(
            eval(&unary(Instr::F32Floor), &[interval(-1.5, 2.3)]),
            interval(-2.0, 2.0)
        );
    }

    #[test]
    fn division() {
        let tape = binary(Instr::F32Div);
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
        let tape = binary(Instr::F32Powf);
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
            eval(&unary(Instr::F32Exp), &[interval(0.0, 1.0)]),
            interval(1.0, 1f32.exp()),
        );
        assert_close(
            eval(&unary(Instr::F32Ln), &[interval(1.0, 4.0)]),
            interval(0.0, 4f32.ln()),
        );
        assert_close(
            eval(&unary(Instr::F32Lg), &[interval(1.0, 100.0)]),
            interval(0.0, 2.0),
        );

        let lowest = f32::MIN_POSITIVE.ln();
        assert_close(
            eval(&unary(Instr::F32Ln), &[interval(-1.0, 4.0)]),
            interval(lowest, 4f32.ln()),
        );
        assert_close(
            eval(&unary(Instr::F32Ln), &[interval(-2.0, -1.0)]),
            interval(lowest, lowest),
        );

        let sin = unary(Instr::F32Sin);
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

        let cos = unary(Instr::F32Cos);
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
        let tan = unary(Instr::F32Tan);
        assert_close(eval(&tan, &[interval(0.0, 1.0)]), interval(0.0, 1f32.tan()));
        assert_close(
            eval(&tan, &[interval(-1.0, -0.5)]),
            interval((-1f32).tan(), (-0.5f32).tan()),
        );
        assert_wide(eval(&tan, &[interval(1.0, 2.0)]));
        assert_wide(eval(&tan, &[interval(0.0, 4.0)]));

        let cot = unary(Instr::F32Cot);
        assert_close(
            eval(&cot, &[interval(0.5, 1.0)]),
            interval(1f32.tan().recip(), 0.5f32.tan().recip()),
        );
        assert_wide(eval(&cot, &[interval(-0.5, 0.5)]));
    }

    #[test]
    fn mixed_i32() {
        // f = x * float(2 + 1)
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let two = b.instr(Instr::I32Const(2));
        let one = b.instr(Instr::I32Const(1));
        let three = b.instr(Instr::I32Add(two, one));
        let three = b.instr(Instr::F32FromI32(three));
        b.instr(Instr::F32Mul(x, three));
        let tape = b.build().unwrap();

        assert_eq!(eval(&tape, &[interval(-1.0, 2.0)]), interval(-3.0, 6.0));
    }
}
