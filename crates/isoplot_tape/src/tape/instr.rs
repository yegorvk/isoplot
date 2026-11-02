use std::{fmt, marker::PhantomData};

mod seal {
    pub trait Sealed {}
}
use seal::Sealed;

impl Sealed for i32 {}
impl Sealed for bool {}
impl Sealed for f32 {}

/// A value type
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ValueType {
    I32,
    Bool,
    F32,
}

pub trait ValueRepr: Copy + Sealed {
    const TYPE: ValueType;
}

impl ValueRepr for i32 {
    const TYPE: ValueType = ValueType::I32;
}

impl ValueRepr for bool {
    const TYPE: ValueType = ValueType::Bool;
}

impl ValueRepr for f32 {
    const TYPE: ValueType = ValueType::F32;
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(super) struct RawValueId(u16);

impl RawValueId {
    pub(super) fn new(index: usize) -> Self {
        Self(index.try_into().unwrap())
    }

    pub(super) fn index(self) -> usize {
        self.0 as usize
    }

    fn shifted(self, delta: u16) -> Self {
        Self(self.0 + delta)
    }
}

/// A value id (e.g., a temporary, or a "variable")
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct ValueId<T> {
    pub(super) id: RawValueId,
    _marker: PhantomData<fn() -> T>,
}

impl<T: ValueRepr> ValueId<T> {
    fn from_raw_id(id: RawValueId) -> Self {
        Self {
            id: id.into(),
            _marker: PhantomData,
        }
    }

    fn shifted(self, delta: u16) -> Self {
        Self::from_raw_id(self.id.shifted(delta))
    }

    /// Returns the instruction index in the tape.
    pub fn index(self) -> usize {
        self.id.index()
    }

    pub fn erased(self) -> AnyValueId {
        AnyValueId {
            id: self.id,
            ty: T::TYPE,
        }
    }
}

impl<T> From<ValueId<T>> for RawValueId {
    fn from(value: ValueId<T>) -> Self {
        value.id
    }
}

impl fmt::Debug for ValueId<i32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ValueId<i32>({})", self.id.0)
    }
}

impl fmt::Debug for ValueId<bool> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ValueId<bool>({})", self.id.0)
    }
}

impl fmt::Debug for ValueId<f32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ValueId<f32>({})", self.id.0)
    }
}

/// A value id whose type is only known at runtime
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct AnyValueId {
    pub(super) id: RawValueId,
    pub ty: ValueType,
}

impl AnyValueId {
    /// Returns the instruction index in the tape.
    pub fn index(self) -> usize {
        self.id.index()
    }

    pub fn typed<T: ValueRepr>(self) -> Option<ValueId<T>> {
        (self.ty == T::TYPE).then(|| ValueId::from_raw_id(self.id))
    }
}

impl<T: ValueRepr> From<ValueId<T>> for AnyValueId {
    fn from(value: ValueId<T>) -> Self {
        Self {
            id: value.id,
            ty: T::TYPE,
        }
    }
}

impl From<AnyValueId> for RawValueId {
    fn from(value: AnyValueId) -> Self {
        value.id
    }
}

/// A value id in a tape under construction
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct NewValueId<T> {
    pub(super) id: RawValueId,
    _marker: PhantomData<fn() -> T>,
}

impl<T: ValueRepr> NewValueId<T> {
    pub(super) fn from_raw_id(id: impl Into<RawValueId>) -> Self {
        Self {
            id: id.into(),
            _marker: PhantomData,
        }
    }

    pub(super) fn from_index(index: usize) -> Self {
        Self::from_raw_id(RawValueId::new(index))
    }

    pub(super) fn commit(self) -> ValueId<T> {
        ValueId::from_raw_id(self.id)
    }

    /// Returns the instruction index in the tape.
    pub fn index(self) -> usize {
        self.id.index()
    }
}

/// An instruction together with the id of the value it produces
#[derive(Copy, Clone, Debug)]
pub struct InstrRef {
    pub(super) id: RawValueId,
    pub(super) instr: Instr,
}

impl InstrRef {
    pub(super) fn new(id: impl Into<RawValueId>, instr: Instr) -> Self {
        Self {
            id: id.into(),
            instr,
        }
    }

    pub fn instr(&self) -> Instr {
        self.instr
    }

    pub fn id(&self) -> AnyValueId {
        AnyValueId {
            id: self.id,
            ty: self.instr.result_type(),
        }
    }

    pub fn index(&self) -> usize {
        self.id.index()
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Instr {
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
    pub fn result_type(&self) -> ValueType {
        match self {
            Instr::I32Const(_)
            | Instr::CopyI32(_)
            | Instr::I32Add(..)
            | Instr::I32Sub(..)
            | Instr::I32Mul(..)
            | Instr::I32FromBool(_)
            | Instr::I32Sel(..) => ValueType::I32,

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
            | Instr::F32Ge(..) => ValueType::Bool,

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
            | Instr::F32Cot(_) => ValueType::F32,
        }
    }

    pub fn sources(self) -> impl Iterator<Item = AnyValueId> {
        let ids: [Option<AnyValueId>; 3] = match self {
            Instr::I32Const(_) | Instr::BoolConst(_) | Instr::F32Const(_) => [None; 3],

            Instr::CopyI32(src) | Instr::F32FromI32(src) => [Some(src.erased()), None, None],

            Instr::CopyBool(src)
            | Instr::Not(src)
            | Instr::I32FromBool(src)
            | Instr::F32FromBool(src) => [Some(src.erased()), None, None],

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
            | Instr::F32Cot(src) => [Some(src.erased()), None, None],

            Instr::I32Add(lhs, rhs)
            | Instr::I32Sub(lhs, rhs)
            | Instr::I32Mul(lhs, rhs)
            | Instr::I32Eq(lhs, rhs)
            | Instr::I32Ne(lhs, rhs)
            | Instr::I32Lt(lhs, rhs)
            | Instr::I32Le(lhs, rhs)
            | Instr::I32Gt(lhs, rhs)
            | Instr::I32Ge(lhs, rhs) => [Some(lhs.erased()), Some(rhs.erased()), None],

            Instr::And(lhs, rhs) | Instr::Or(lhs, rhs) | Instr::Xor(lhs, rhs) => {
                [Some(lhs.erased()), Some(rhs.erased()), None]
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
            | Instr::F32Ge(lhs, rhs) => [Some(lhs.erased()), Some(rhs.erased()), None],

            Instr::F32Powi(lhs, rhs) => [Some(lhs.erased()), Some(rhs.erased()), None],

            Instr::I32Sel(cond, v_true, v_false) => [
                Some(cond.erased()),
                Some(v_true.erased()),
                Some(v_false.erased()),
            ],
            Instr::F32Sel(cond, v_true, v_false) => [
                Some(cond.erased()),
                Some(v_true.erased()),
                Some(v_false.erased()),
            ],
        };

        ids.into_iter().flatten()
    }

    pub(super) fn shifted_ids(self, new_base: impl Into<RawValueId>) -> Self {
        let offset = new_base.into().0;
        match self {
            Instr::I32Const(_) | Instr::BoolConst(_) | Instr::F32Const(_) => self,

            Instr::CopyI32(src) => Instr::CopyI32(src.shifted(offset)),
            Instr::CopyBool(src) => Instr::CopyBool(src.shifted(offset)),
            Instr::CopyF32(src) => Instr::CopyF32(src.shifted(offset)),

            Instr::I32Add(lhs, rhs) => Instr::I32Add(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::I32Sub(lhs, rhs) => Instr::I32Sub(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::I32Mul(lhs, rhs) => Instr::I32Mul(lhs.shifted(offset), rhs.shifted(offset)),

            Instr::I32Eq(lhs, rhs) => Instr::I32Eq(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::I32Ne(lhs, rhs) => Instr::I32Ne(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::I32Lt(lhs, rhs) => Instr::I32Lt(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::I32Le(lhs, rhs) => Instr::I32Le(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::I32Gt(lhs, rhs) => Instr::I32Gt(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::I32Ge(lhs, rhs) => Instr::I32Ge(lhs.shifted(offset), rhs.shifted(offset)),

            Instr::Not(src) => Instr::Not(src.shifted(offset)),
            Instr::And(lhs, rhs) => Instr::And(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::Or(lhs, rhs) => Instr::Or(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::Xor(lhs, rhs) => Instr::Xor(lhs.shifted(offset), rhs.shifted(offset)),

            Instr::I32FromBool(src) => Instr::I32FromBool(src.shifted(offset)),
            Instr::F32FromI32(src) => Instr::F32FromI32(src.shifted(offset)),
            Instr::F32FromBool(src) => Instr::F32FromBool(src.shifted(offset)),

            Instr::F32Neg(src) => Instr::F32Neg(src.shifted(offset)),
            Instr::F32Abs(src) => Instr::F32Abs(src.shifted(offset)),
            Instr::F32Sign(src) => Instr::F32Sign(src.shifted(offset)),
            Instr::F32Floor(src) => Instr::F32Floor(src.shifted(offset)),
            Instr::F32Add(lhs, rhs) => Instr::F32Add(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::F32Sub(lhs, rhs) => Instr::F32Sub(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::F32Mul(lhs, rhs) => Instr::F32Mul(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::F32Div(lhs, rhs) => Instr::F32Div(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::F32Min(lhs, rhs) => Instr::F32Min(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::F32Max(lhs, rhs) => Instr::F32Max(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::F32Powf(lhs, rhs) => Instr::F32Powf(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::F32Powi(lhs, rhs) => Instr::F32Powi(lhs.shifted(offset), rhs.shifted(offset)),

            Instr::F32Exp(src) => Instr::F32Exp(src.shifted(offset)),
            Instr::F32Ln(src) => Instr::F32Ln(src.shifted(offset)),
            Instr::F32Lg(src) => Instr::F32Lg(src.shifted(offset)),
            Instr::F32Sin(src) => Instr::F32Sin(src.shifted(offset)),
            Instr::F32Cos(src) => Instr::F32Cos(src.shifted(offset)),
            Instr::F32Tan(src) => Instr::F32Tan(src.shifted(offset)),
            Instr::F32Cot(src) => Instr::F32Cot(src.shifted(offset)),

            Instr::F32Eq(lhs, rhs) => Instr::F32Eq(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::F32Ne(lhs, rhs) => Instr::F32Ne(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::F32Lt(lhs, rhs) => Instr::F32Lt(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::F32Le(lhs, rhs) => Instr::F32Le(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::F32Gt(lhs, rhs) => Instr::F32Gt(lhs.shifted(offset), rhs.shifted(offset)),
            Instr::F32Ge(lhs, rhs) => Instr::F32Ge(lhs.shifted(offset), rhs.shifted(offset)),

            Instr::I32Sel(cond, v_true, v_false) => Instr::I32Sel(
                cond.shifted(offset),
                v_true.shifted(offset),
                v_false.shifted(offset),
            ),
            Instr::F32Sel(cond, v_true, v_false) => Instr::F32Sel(
                cond.shifted(offset),
                v_true.shifted(offset),
                v_false.shifted(offset),
            ),
        }
    }
}
