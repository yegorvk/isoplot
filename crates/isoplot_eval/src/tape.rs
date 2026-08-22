use std::marker::PhantomData;

mod private {
    #[doc(hidden)]
    pub(crate) trait Sealed {}
    impl Sealed for i32 {}
    impl Sealed for bool {}
    impl Sealed for f32 {}
}

/// A value type
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) enum Type {
    I32,
    Bool,
    F32,
}

pub(crate) trait ValuePrimitive: private::Sealed + Copy {
    const TYPE: Type;
}

impl ValuePrimitive for i32 {
    const TYPE: Type = Type::I32;
}

impl ValuePrimitive for bool {
    const TYPE: Type = Type::Bool;
}

impl ValuePrimitive for f32 {
    const TYPE: Type = Type::F32;
}

/// A value id (e.g., a temporary value)
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) struct ValueId<T> {
    raw_id: u16,
    _marker: PhantomData<fn() -> T>,
}

impl<T: ValuePrimitive> ValueId<T> {
    fn new(raw_id: u16) -> Self {
        Self {
            raw_id,
            _marker: PhantomData,
        }
    }

    fn shift_ids(self, delta: u16) -> Self {
        Self::new(self.raw_id + delta)
    }

    /// Returns the corresponding instruction index in the tape.
    pub(crate) fn index(self) -> usize {
        self.raw_id as usize
    }
}

/// A value id in a tape under construction
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) struct NewId<T> {
    raw_id: u16,
    _marker: PhantomData<fn() -> T>,
}

impl<T: ValuePrimitive> NewId<T> {
    fn new(raw_id: u16) -> Self {
        Self {
            raw_id,
            _marker: PhantomData,
        }
    }

    fn id(self) -> ValueId<T> {
        ValueId::new(self.raw_id)
    }

    /// Returns the corresponding instruction index in the tape.
    pub(crate) fn index(self) -> usize {
        self.raw_id as usize
    }
}

/// An instruction together with the id of the value it produces
#[derive(Copy, Clone, Debug)]
pub(crate) struct InstrRef {
    raw_id: u16,
    instr: Instr,
}

impl InstrRef {
    pub(crate) fn instr(&self) -> Instr {
        self.instr
    }

    pub(crate) fn id<T: ValuePrimitive>(&self) -> ValueId<T> {
        assert_eq!(self.instr.result_type(), T::TYPE);
        ValueId::new(self.raw_id)
    }

    pub(crate) fn index(&self) -> usize {
        self.raw_id as usize
    }
}

#[allow(dead_code)]
#[derive(Copy, Clone, Debug)]
pub(crate) enum Instr {
    // Constants
    I32Const(i32),
    BoolConst(bool),
    F32Const(f32),

    // Identity copies
    CopyI32(ValueId<i32>),
    CopyBool(ValueId<bool>),
    CopyF32(ValueId<f32>),

    // Basic `i32` operations
    I32Add(ValueId<i32>, ValueId<i32>),
    I32Sub(ValueId<i32>, ValueId<i32>),
    I32Mul(ValueId<i32>, ValueId<i32>),

    // `i32` comparison operations
    I32Eq(ValueId<i32>, ValueId<i32>),
    I32Ne(ValueId<i32>, ValueId<i32>),
    I32Lt(ValueId<i32>, ValueId<i32>),
    I32Le(ValueId<i32>, ValueId<i32>),
    I32Gt(ValueId<i32>, ValueId<i32>),
    I32Ge(ValueId<i32>, ValueId<i32>),

    // Basic `bool` operations
    Not(ValueId<bool>),
    And(ValueId<bool>, ValueId<bool>),
    Or(ValueId<bool>, ValueId<bool>),
    Xor(ValueId<bool>, ValueId<bool>),

    // Conversions
    I32FromBool(ValueId<bool>),
    F32FromI32(ValueId<i32>),
    F32FromBool(ValueId<bool>),

    // Basic `f32` operations
    F32Neg(ValueId<f32>),
    F32Abs(ValueId<f32>),
    F32Sign(ValueId<f32>),
    F32Floor(ValueId<f32>),
    F32Add(ValueId<f32>, ValueId<f32>),
    F32Sub(ValueId<f32>, ValueId<f32>),
    F32Mul(ValueId<f32>, ValueId<f32>),
    F32Div(ValueId<f32>, ValueId<f32>),
    F32Min(ValueId<f32>, ValueId<f32>),
    F32Max(ValueId<f32>, ValueId<f32>),
    F32Powf(ValueId<f32>, ValueId<f32>),
    F32Powi(ValueId<f32>, ValueId<i32>),

    // Common `f32` math functions
    F32Exp(ValueId<f32>),
    F32Ln(ValueId<f32>),
    F32Lg(ValueId<f32>),
    F32Sin(ValueId<f32>),
    F32Cos(ValueId<f32>),
    F32Tan(ValueId<f32>),
    F32Cot(ValueId<f32>),

    // `f32` comparison operations
    F32Eq(ValueId<f32>, ValueId<f32>),
    F32Ne(ValueId<f32>, ValueId<f32>),
    F32Lt(ValueId<f32>, ValueId<f32>),
    F32Le(ValueId<f32>, ValueId<f32>),
    F32Gt(ValueId<f32>, ValueId<f32>),
    F32Ge(ValueId<f32>, ValueId<f32>),

    // Branchless select (cond, v_true, v_false)
    I32Sel(ValueId<bool>, ValueId<i32>, ValueId<i32>),
    F32Sel(ValueId<bool>, ValueId<f32>, ValueId<f32>),
}

impl Instr {
    fn result_type(&self) -> Type {
        match self {
            Instr::I32Const(_)
            | Instr::CopyI32(_)
            | Instr::I32Add(..)
            | Instr::I32Sub(..)
            | Instr::I32Mul(..)
            | Instr::I32FromBool(_)
            | Instr::I32Sel(..) => Type::I32,

            Instr::BoolConst(_)
            | Instr::CopyBool(_)
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
            | Instr::F32Eq(..)
            | Instr::F32Ne(..)
            | Instr::F32Lt(..)
            | Instr::F32Le(..)
            | Instr::F32Gt(..)
            | Instr::F32Ge(..) => Type::Bool,

            Instr::F32Const(_)
            | Instr::CopyF32(_)
            | Instr::F32FromI32(_)
            | Instr::F32FromBool(_)
            | Instr::F32Sel(..)
            | Instr::F32Neg(_)
            | Instr::F32Abs(_)
            | Instr::F32Sign(_)
            | Instr::F32Floor(_)
            | Instr::F32Add(..)
            | Instr::F32Sub(..)
            | Instr::F32Mul(..)
            | Instr::F32Div(..)
            | Instr::F32Min(..)
            | Instr::F32Max(..)
            | Instr::F32Powf(..)
            | Instr::F32Powi(..)
            | Instr::F32Exp(_)
            | Instr::F32Ln(_)
            | Instr::F32Lg(_)
            | Instr::F32Sin(_)
            | Instr::F32Cos(_)
            | Instr::F32Tan(_)
            | Instr::F32Cot(_) => Type::F32,
        }
    }

    fn shift_ids(self, delta: u16) -> Self {
        match self {
            Instr::I32Const(_) | Instr::BoolConst(_) | Instr::F32Const(_) => self,

            Instr::CopyI32(src) => Instr::CopyI32(src.shift_ids(delta)),
            Instr::CopyBool(src) => Instr::CopyBool(src.shift_ids(delta)),
            Instr::CopyF32(src) => Instr::CopyF32(src.shift_ids(delta)),

            Instr::I32Add(lhs, rhs) => Instr::I32Add(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::I32Sub(lhs, rhs) => Instr::I32Sub(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::I32Mul(lhs, rhs) => Instr::I32Mul(lhs.shift_ids(delta), rhs.shift_ids(delta)),

            Instr::I32Eq(lhs, rhs) => Instr::I32Eq(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::I32Ne(lhs, rhs) => Instr::I32Ne(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::I32Lt(lhs, rhs) => Instr::I32Lt(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::I32Le(lhs, rhs) => Instr::I32Le(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::I32Gt(lhs, rhs) => Instr::I32Gt(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::I32Ge(lhs, rhs) => Instr::I32Ge(lhs.shift_ids(delta), rhs.shift_ids(delta)),

            Instr::Not(src) => Instr::Not(src.shift_ids(delta)),
            Instr::And(lhs, rhs) => Instr::And(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::Or(lhs, rhs) => Instr::Or(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::Xor(lhs, rhs) => Instr::Xor(lhs.shift_ids(delta), rhs.shift_ids(delta)),

            Instr::I32FromBool(src) => Instr::I32FromBool(src.shift_ids(delta)),
            Instr::F32FromI32(src) => Instr::F32FromI32(src.shift_ids(delta)),
            Instr::F32FromBool(src) => Instr::F32FromBool(src.shift_ids(delta)),

            Instr::F32Neg(src) => Instr::F32Neg(src.shift_ids(delta)),
            Instr::F32Abs(src) => Instr::F32Abs(src.shift_ids(delta)),
            Instr::F32Sign(src) => Instr::F32Sign(src.shift_ids(delta)),
            Instr::F32Floor(src) => Instr::F32Floor(src.shift_ids(delta)),
            Instr::F32Add(lhs, rhs) => Instr::F32Add(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::F32Sub(lhs, rhs) => Instr::F32Sub(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::F32Mul(lhs, rhs) => Instr::F32Mul(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::F32Div(lhs, rhs) => Instr::F32Div(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::F32Min(lhs, rhs) => Instr::F32Min(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::F32Max(lhs, rhs) => Instr::F32Max(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::F32Powf(lhs, rhs) => Instr::F32Powf(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::F32Powi(lhs, rhs) => Instr::F32Powi(lhs.shift_ids(delta), rhs.shift_ids(delta)),

            Instr::F32Exp(src) => Instr::F32Exp(src.shift_ids(delta)),
            Instr::F32Ln(src) => Instr::F32Ln(src.shift_ids(delta)),
            Instr::F32Lg(src) => Instr::F32Lg(src.shift_ids(delta)),
            Instr::F32Sin(src) => Instr::F32Sin(src.shift_ids(delta)),
            Instr::F32Cos(src) => Instr::F32Cos(src.shift_ids(delta)),
            Instr::F32Tan(src) => Instr::F32Tan(src.shift_ids(delta)),
            Instr::F32Cot(src) => Instr::F32Cot(src.shift_ids(delta)),

            Instr::F32Eq(lhs, rhs) => Instr::F32Eq(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::F32Ne(lhs, rhs) => Instr::F32Ne(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::F32Lt(lhs, rhs) => Instr::F32Lt(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::F32Le(lhs, rhs) => Instr::F32Le(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::F32Gt(lhs, rhs) => Instr::F32Gt(lhs.shift_ids(delta), rhs.shift_ids(delta)),
            Instr::F32Ge(lhs, rhs) => Instr::F32Ge(lhs.shift_ids(delta), rhs.shift_ids(delta)),

            Instr::I32Sel(cond, v_true, v_false) => Instr::I32Sel(
                cond.shift_ids(delta),
                v_true.shift_ids(delta),
                v_false.shift_ids(delta),
            ),
            Instr::F32Sel(cond, v_true, v_false) => Instr::F32Sel(
                cond.shift_ids(delta),
                v_true.shift_ids(delta),
                v_false.shift_ids(delta),
            ),
        }
    }
}

pub(crate) struct Tape {
    args: Vec<Type>,
    results: Vec<Type>,
    instrs: Vec<Instr>,
}

impl Tape {
    pub(crate) fn builder(args: Vec<Type>, results: Vec<Type>) -> TapeBuilder {
        assert!(
            args.len() <= u8::MAX as usize,
            "a tape can take at most {} arguments",
            u8::MAX
        );

        assert!(
            results.len() <= u8::MAX as usize,
            "a tape can produce at most {} results",
            u8::MAX
        );

        TapeBuilder {
            num_args: args.len(),
            values: args,
            instrs: vec![],
            results,
        }
    }

    pub(crate) fn num_args(&self) -> usize {
        self.args.len()
    }

    pub(crate) fn num_results(&self) -> usize {
        self.results.len()
    }

    pub(crate) fn arg_types(&self) -> &[Type] {
        &self.args
    }

    pub(crate) fn result_types(&self) -> &[Type] {
        &self.results
    }

    pub(crate) fn instrs(
        &self,
    ) -> impl DoubleEndedIterator<Item = InstrRef> + ExactSizeIterator + '_ {
        let num_args = self.args.len();
        self.instrs
            .iter()
            .enumerate()
            .map(move |(i, &instr)| InstrRef {
                raw_id: (num_args + i) as u16,
                instr,
            })
    }
}

pub(crate) struct Inlined<'a> {
    base: usize,
    tape: &'a Tape,
}

impl Inlined<'_> {
    /// Converts an inlined tape's value id into the corresponding new tape's one.
    pub(crate) fn convert<T: ValuePrimitive>(&self, src_id: ValueId<T>) -> NewId<T> {
        assert!(src_id.index() < self.tape.args.len() + self.tape.instrs.len());
        NewId::new((self.base + src_id.index()) as u16)
    }

    pub(crate) fn result<T: ValuePrimitive>(&self, index: usize) -> NewId<T> {
        let from_back = self.tape.results.len() - 1 - index;
        self.convert(self.tape.instrs().nth_back(from_back).unwrap().id())
    }
}

#[derive(Debug)]
pub(crate) struct ValidateError(());

pub(crate) struct TapeBuilder {
    num_args: usize,
    instrs: Vec<Instr>,
    values: Vec<Type>,
    results: Vec<Type>,
}

impl TapeBuilder {
    pub(crate) fn arg<T: ValuePrimitive>(&self, index: usize) -> NewId<T> {
        assert!(index < self.num_args);
        assert_eq!(self.values[index], T::TYPE);
        NewId::new(index as u16)
    }

    pub(crate) fn extend<'a>(&mut self, tape: &'a Tape) -> Inlined<'a> {
        assert!(
            self.values.ends_with(&tape.args),
            "tape arguments must match the last values in the builder"
        );
        assert!(self.values.len() + tape.instrs.len() <= u16::MAX as usize);
        let base = self.values.len() - tape.args.len();

        self.values
            .extend(tape.instrs.iter().map(Instr::result_type));

        self.instrs
            .extend(tape.instrs.iter().map(|instr| instr.shift_ids(base as u16)));

        Inlined { base, tape }
    }

    fn push<T: ValuePrimitive>(&mut self, instr: Instr) -> NewId<T> {
        let ty = instr.result_type();
        debug_assert_eq!(ty, T::TYPE);
        let index = self.values.len();
        assert!(index < u16::MAX as usize);
        self.values.push(ty);
        self.instrs.push(instr);
        NewId::new(index as u16)
    }

    pub(crate) fn build(mut self) -> Result<Tape, ValidateError> {
        self.validate()?;
        self.values.truncate(self.num_args);
        Ok(Tape {
            args: self.values,
            results: self.results,
            instrs: self.instrs,
        })
    }

    fn validate(&self) -> Result<(), ValidateError> {
        // Enforce that arguments are not implicitly used as results (not technically required by the backends).
        if self.values.len() < self.num_args + self.results.len() {
            return Err(ValidateError(()));
        }

        if self.values[(self.values.len() - self.results.len())..] != self.results {
            return Err(ValidateError(()));
        }

        for (index, &instr) in self.instrs.iter().enumerate() {
            let cur_dst = self.num_args + index;

            match instr {
                Instr::I32Const(_) | Instr::BoolConst(_) | Instr::F32Const(_) => {}

                Instr::CopyI32(src) | Instr::F32FromI32(src) => self.check(cur_dst, src)?,

                Instr::CopyBool(src)
                | Instr::Not(src)
                | Instr::I32FromBool(src)
                | Instr::F32FromBool(src) => self.check(cur_dst, src)?,

                Instr::CopyF32(src)
                | Instr::F32Neg(src)
                | Instr::F32Abs(src)
                | Instr::F32Sign(src)
                | Instr::F32Floor(src)
                | Instr::F32Exp(src)
                | Instr::F32Ln(src)
                | Instr::F32Lg(src)
                | Instr::F32Sin(src)
                | Instr::F32Cos(src)
                | Instr::F32Tan(src)
                | Instr::F32Cot(src) => self.check(cur_dst, src)?,

                Instr::I32Add(lhs, rhs)
                | Instr::I32Sub(lhs, rhs)
                | Instr::I32Mul(lhs, rhs)
                | Instr::I32Eq(lhs, rhs)
                | Instr::I32Ne(lhs, rhs)
                | Instr::I32Lt(lhs, rhs)
                | Instr::I32Le(lhs, rhs)
                | Instr::I32Gt(lhs, rhs)
                | Instr::I32Ge(lhs, rhs) => {
                    self.check(cur_dst, lhs)?;
                    self.check(cur_dst, rhs)?;
                }

                Instr::And(lhs, rhs) | Instr::Or(lhs, rhs) | Instr::Xor(lhs, rhs) => {
                    self.check(cur_dst, lhs)?;
                    self.check(cur_dst, rhs)?;
                }

                Instr::F32Add(lhs, rhs)
                | Instr::F32Sub(lhs, rhs)
                | Instr::F32Mul(lhs, rhs)
                | Instr::F32Div(lhs, rhs)
                | Instr::F32Min(lhs, rhs)
                | Instr::F32Max(lhs, rhs)
                | Instr::F32Powf(lhs, rhs)
                | Instr::F32Eq(lhs, rhs)
                | Instr::F32Ne(lhs, rhs)
                | Instr::F32Lt(lhs, rhs)
                | Instr::F32Le(lhs, rhs)
                | Instr::F32Gt(lhs, rhs)
                | Instr::F32Ge(lhs, rhs) => {
                    self.check(cur_dst, lhs)?;
                    self.check(cur_dst, rhs)?;
                }

                Instr::F32Powi(lhs, rhs) => {
                    self.check(cur_dst, lhs)?;
                    self.check(cur_dst, rhs)?;
                }

                Instr::I32Sel(cond, v_true, v_false) => {
                    self.check(cur_dst, cond)?;
                    self.check(cur_dst, v_true)?;
                    self.check(cur_dst, v_false)?;
                }

                Instr::F32Sel(cond, v_true, v_false) => {
                    self.check(cur_dst, cond)?;
                    self.check(cur_dst, v_true)?;
                    self.check(cur_dst, v_false)?;
                }
            }
        }

        Ok(())
    }

    fn check<T: ValuePrimitive>(
        &self,
        cur_dst: usize,
        value: ValueId<T>,
    ) -> Result<(), ValidateError> {
        if value.index() < cur_dst && self.values[value.index()] == T::TYPE {
            Ok(())
        } else {
            Err(ValidateError(()))
        }
    }
}

#[allow(dead_code)]
impl TapeBuilder {
    pub(crate) fn i32_const(&mut self, value: i32) -> NewId<i32> {
        self.push(Instr::I32Const(value))
    }

    pub(crate) fn bool_const(&mut self, value: bool) -> NewId<bool> {
        self.push(Instr::BoolConst(value))
    }

    pub(crate) fn f32_const(&mut self, value: f32) -> NewId<f32> {
        self.push(Instr::F32Const(value))
    }

    pub(crate) fn copy_i32(&mut self, src: NewId<i32>) -> NewId<i32> {
        self.push(Instr::CopyI32(src.id()))
    }

    pub(crate) fn copy_bool(&mut self, src: NewId<bool>) -> NewId<bool> {
        self.push(Instr::CopyBool(src.id()))
    }

    pub(crate) fn copy_f32(&mut self, src: NewId<f32>) -> NewId<f32> {
        self.push(Instr::CopyF32(src.id()))
    }

    pub(crate) fn i32_add(&mut self, lhs: NewId<i32>, rhs: NewId<i32>) -> NewId<i32> {
        self.push(Instr::I32Add(lhs.id(), rhs.id()))
    }

    pub(crate) fn i32_sub(&mut self, lhs: NewId<i32>, rhs: NewId<i32>) -> NewId<i32> {
        self.push(Instr::I32Sub(lhs.id(), rhs.id()))
    }

    pub(crate) fn i32_mul(&mut self, lhs: NewId<i32>, rhs: NewId<i32>) -> NewId<i32> {
        self.push(Instr::I32Mul(lhs.id(), rhs.id()))
    }

    pub(crate) fn i32_eq(&mut self, lhs: NewId<i32>, rhs: NewId<i32>) -> NewId<bool> {
        self.push(Instr::I32Eq(lhs.id(), rhs.id()))
    }

    pub(crate) fn i32_ne(&mut self, lhs: NewId<i32>, rhs: NewId<i32>) -> NewId<bool> {
        self.push(Instr::I32Ne(lhs.id(), rhs.id()))
    }

    pub(crate) fn i32_lt(&mut self, lhs: NewId<i32>, rhs: NewId<i32>) -> NewId<bool> {
        self.push(Instr::I32Lt(lhs.id(), rhs.id()))
    }

    pub(crate) fn i32_le(&mut self, lhs: NewId<i32>, rhs: NewId<i32>) -> NewId<bool> {
        self.push(Instr::I32Le(lhs.id(), rhs.id()))
    }

    pub(crate) fn i32_gt(&mut self, lhs: NewId<i32>, rhs: NewId<i32>) -> NewId<bool> {
        self.push(Instr::I32Gt(lhs.id(), rhs.id()))
    }

    pub(crate) fn i32_ge(&mut self, lhs: NewId<i32>, rhs: NewId<i32>) -> NewId<bool> {
        self.push(Instr::I32Ge(lhs.id(), rhs.id()))
    }

    pub(crate) fn not(&mut self, src: NewId<bool>) -> NewId<bool> {
        self.push(Instr::Not(src.id()))
    }

    pub(crate) fn and(&mut self, lhs: NewId<bool>, rhs: NewId<bool>) -> NewId<bool> {
        self.push(Instr::And(lhs.id(), rhs.id()))
    }

    pub(crate) fn or(&mut self, lhs: NewId<bool>, rhs: NewId<bool>) -> NewId<bool> {
        self.push(Instr::Or(lhs.id(), rhs.id()))
    }

    pub(crate) fn xor(&mut self, lhs: NewId<bool>, rhs: NewId<bool>) -> NewId<bool> {
        self.push(Instr::Xor(lhs.id(), rhs.id()))
    }

    pub(crate) fn i32_from_bool(&mut self, src: NewId<bool>) -> NewId<i32> {
        self.push(Instr::I32FromBool(src.id()))
    }

    pub(crate) fn f32_from_i32(&mut self, src: NewId<i32>) -> NewId<f32> {
        self.push(Instr::F32FromI32(src.id()))
    }

    pub(crate) fn f32_from_bool(&mut self, src: NewId<bool>) -> NewId<f32> {
        self.push(Instr::F32FromBool(src.id()))
    }

    pub(crate) fn f32_neg(&mut self, src: NewId<f32>) -> NewId<f32> {
        self.push(Instr::F32Neg(src.id()))
    }

    pub(crate) fn f32_abs(&mut self, src: NewId<f32>) -> NewId<f32> {
        self.push(Instr::F32Abs(src.id()))
    }

    pub(crate) fn f32_sign(&mut self, src: NewId<f32>) -> NewId<f32> {
        self.push(Instr::F32Sign(src.id()))
    }

    pub(crate) fn f32_floor(&mut self, src: NewId<f32>) -> NewId<f32> {
        self.push(Instr::F32Floor(src.id()))
    }

    pub(crate) fn f32_add(&mut self, lhs: NewId<f32>, rhs: NewId<f32>) -> NewId<f32> {
        self.push(Instr::F32Add(lhs.id(), rhs.id()))
    }

    pub(crate) fn f32_sub(&mut self, lhs: NewId<f32>, rhs: NewId<f32>) -> NewId<f32> {
        self.push(Instr::F32Sub(lhs.id(), rhs.id()))
    }

    pub(crate) fn f32_mul(&mut self, lhs: NewId<f32>, rhs: NewId<f32>) -> NewId<f32> {
        self.push(Instr::F32Mul(lhs.id(), rhs.id()))
    }

    pub(crate) fn f32_div(&mut self, lhs: NewId<f32>, rhs: NewId<f32>) -> NewId<f32> {
        self.push(Instr::F32Div(lhs.id(), rhs.id()))
    }

    pub(crate) fn f32_min(&mut self, lhs: NewId<f32>, rhs: NewId<f32>) -> NewId<f32> {
        self.push(Instr::F32Min(lhs.id(), rhs.id()))
    }

    pub(crate) fn f32_max(&mut self, lhs: NewId<f32>, rhs: NewId<f32>) -> NewId<f32> {
        self.push(Instr::F32Max(lhs.id(), rhs.id()))
    }

    pub(crate) fn f32_powf(&mut self, lhs: NewId<f32>, rhs: NewId<f32>) -> NewId<f32> {
        self.push(Instr::F32Powf(lhs.id(), rhs.id()))
    }

    pub(crate) fn f32_powi(&mut self, lhs: NewId<f32>, rhs: NewId<i32>) -> NewId<f32> {
        self.push(Instr::F32Powi(lhs.id(), rhs.id()))
    }

    pub(crate) fn f32_exp(&mut self, src: NewId<f32>) -> NewId<f32> {
        self.push(Instr::F32Exp(src.id()))
    }

    pub(crate) fn f32_ln(&mut self, src: NewId<f32>) -> NewId<f32> {
        self.push(Instr::F32Ln(src.id()))
    }

    pub(crate) fn f32_lg(&mut self, src: NewId<f32>) -> NewId<f32> {
        self.push(Instr::F32Lg(src.id()))
    }

    pub(crate) fn f32_sin(&mut self, src: NewId<f32>) -> NewId<f32> {
        self.push(Instr::F32Sin(src.id()))
    }

    pub(crate) fn f32_cos(&mut self, src: NewId<f32>) -> NewId<f32> {
        self.push(Instr::F32Cos(src.id()))
    }

    pub(crate) fn f32_tan(&mut self, src: NewId<f32>) -> NewId<f32> {
        self.push(Instr::F32Tan(src.id()))
    }

    pub(crate) fn f32_cot(&mut self, src: NewId<f32>) -> NewId<f32> {
        self.push(Instr::F32Cot(src.id()))
    }

    pub(crate) fn f32_eq(&mut self, lhs: NewId<f32>, rhs: NewId<f32>) -> NewId<bool> {
        self.push(Instr::F32Eq(lhs.id(), rhs.id()))
    }

    pub(crate) fn f32_ne(&mut self, lhs: NewId<f32>, rhs: NewId<f32>) -> NewId<bool> {
        self.push(Instr::F32Ne(lhs.id(), rhs.id()))
    }

    pub(crate) fn f32_lt(&mut self, lhs: NewId<f32>, rhs: NewId<f32>) -> NewId<bool> {
        self.push(Instr::F32Lt(lhs.id(), rhs.id()))
    }

    pub(crate) fn f32_le(&mut self, lhs: NewId<f32>, rhs: NewId<f32>) -> NewId<bool> {
        self.push(Instr::F32Le(lhs.id(), rhs.id()))
    }

    pub(crate) fn f32_gt(&mut self, lhs: NewId<f32>, rhs: NewId<f32>) -> NewId<bool> {
        self.push(Instr::F32Gt(lhs.id(), rhs.id()))
    }

    pub(crate) fn f32_ge(&mut self, lhs: NewId<f32>, rhs: NewId<f32>) -> NewId<bool> {
        self.push(Instr::F32Ge(lhs.id(), rhs.id()))
    }

    pub(crate) fn i32_sel(
        &mut self,
        cond: NewId<bool>,
        v_t: NewId<i32>,
        v_f: NewId<i32>,
    ) -> NewId<i32> {
        self.push(Instr::I32Sel(cond.id(), v_t.id(), v_f.id()))
    }

    pub(crate) fn f32_sel(
        &mut self,
        cond: NewId<bool>,
        v_t: NewId<f32>,
        v_f: NewId<f32>,
    ) -> NewId<f32> {
        self.push(Instr::F32Sel(cond.id(), v_t.id(), v_f.id()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Fallback, Instance};

    #[test]
    fn extend_inlines_at_tail() {
        // g(a) = a * a
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let a = b.arg(0);
        b.f32_mul(a, a);
        let square = b.build().unwrap();

        // f(x) = g(x + 1) - x
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let one = b.f32_const(1.0);
        b.f32_add(x, one);
        let squared = b.extend(&square).result(0);
        b.f32_sub(squared, x);
        let tape = b.build().unwrap();

        let result = Instance::<Fallback, f32, f32>::new(tape)
            .evaluator()
            .evaluate(&2.5);
        assert_eq!(result, 3.5 * 3.5 - 2.5);
    }
}
