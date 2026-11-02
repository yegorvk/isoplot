use super::{
    RawValueId, Tape,
    instr::{Instr, NewValueId, ValueType, ValueId, ValueRepr},
};

#[derive(Debug)]
pub struct ValidateError(());

pub struct Inlined<'a> {
    base: RawValueId,
    tape: &'a Tape,
}

impl Inlined<'_> {
    /// Turns a `ValueId` from the inlined tape into a `NewValueId` in the main tape.
    pub fn convert<T: ValueRepr>(&self, src_id: ValueId<T>) -> NewValueId<T> {
        assert!(src_id.index() < self.tape.args.len() + self.tape.instrs.len());
        NewValueId::from_index(self.base.index() + src_id.index())
    }

    pub fn result<T: ValueRepr>(&self, index: usize) -> NewValueId<T> {
        let id = RawValueId::new(self.tape.instrs.len() - self.tape.results.len() + index);
        assert_eq!(self.tape.instrs[id.index()].result_type(), T::TYPE);
        NewValueId::from_raw_id(id)
    }
}

pub struct TapeBuilder {
    num_args: usize,
    instrs: Vec<Instr>,
    values: Vec<ValueType>,
    results: Vec<ValueType>,
}

impl TapeBuilder {
    pub(super) fn new(args: Vec<ValueType>, results: Vec<ValueType>) -> Self {
        const MAX_ARGS: usize = u8::MAX as usize;
        assert!(
            args.len() <= MAX_ARGS,
            "a tape can take at most {MAX_ARGS} arguments",
        );

        const MAX_RESULTS: usize = u8::MAX as usize;
        assert!(
            results.len() <= MAX_RESULTS as usize,
            "a tape can produce at most {MAX_RESULTS} results",
        );

        Self {
            num_args: args.len(),
            values: args,
            instrs: vec![],
            results,
        }
    }

    pub fn arg<T: ValueRepr>(&self, index: usize) -> NewValueId<T> {
        assert!(index < self.num_args && self.values[index] == T::TYPE);
        NewValueId::from_index(index)
    }

    pub fn extend<'a>(&mut self, tape: &'a Tape) -> Inlined<'a> {
        assert!(
            self.values.ends_with(&tape.args),
            "tape arguments must match the last values in the builder"
        );

        let base = RawValueId::new(self.values.len() - tape.args.len());

        self.values
            .extend(tape.instrs.iter().map(Instr::result_type));

        self.instrs
            .extend(tape.instrs.iter().map(|instr| instr.shifted_ids(base)));

        Inlined { base, tape }
    }

    fn push<T: ValueRepr>(&mut self, instr: Instr) -> NewValueId<T> {
        let ty = instr.result_type();
        debug_assert_eq!(ty, T::TYPE);
        let index = self.values.len();
        self.values.push(ty);
        self.instrs.push(instr);
        NewValueId::from_index(index)
    }

    pub fn build(mut self) -> Result<Tape, ValidateError> {
        self.validate()?;
        self.values.truncate(self.num_args);
        Ok(Tape {
            args: self.values,
            results: self.results,
            instrs: self.instrs,
        })
    }

    fn validate(&self) -> Result<(), ValidateError> {
        // Enforce that arguments are not implicitly used as results.
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

    fn check<T: ValueRepr>(&self, cur_dst: usize, value: ValueId<T>) -> Result<(), ValidateError> {
        if value.index() < cur_dst && self.values[value.index()] == T::TYPE {
            Ok(())
        } else {
            Err(ValidateError(()))
        }
    }
}

#[allow(dead_code)]
impl TapeBuilder {
    pub fn i32_const(&mut self, value: i32) -> NewValueId<i32> {
        self.push(Instr::I32Const(value))
    }

    pub fn bool_const(&mut self, value: bool) -> NewValueId<bool> {
        self.push(Instr::BoolConst(value))
    }

    pub fn f32_const(&mut self, value: f32) -> NewValueId<f32> {
        self.push(Instr::F32Const(value))
    }

    pub fn copy_i32(&mut self, src: NewValueId<i32>) -> NewValueId<i32> {
        self.push(Instr::CopyI32(src.commit()))
    }

    pub fn copy_bool(&mut self, src: NewValueId<bool>) -> NewValueId<bool> {
        self.push(Instr::CopyBool(src.commit()))
    }

    pub fn copy_f32(&mut self, src: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::CopyF32(src.commit()))
    }

    pub fn i32_add(&mut self, lhs: NewValueId<i32>, rhs: NewValueId<i32>) -> NewValueId<i32> {
        self.push(Instr::I32Add(lhs.commit(), rhs.commit()))
    }

    pub fn i32_sub(&mut self, lhs: NewValueId<i32>, rhs: NewValueId<i32>) -> NewValueId<i32> {
        self.push(Instr::I32Sub(lhs.commit(), rhs.commit()))
    }

    pub fn i32_mul(&mut self, lhs: NewValueId<i32>, rhs: NewValueId<i32>) -> NewValueId<i32> {
        self.push(Instr::I32Mul(lhs.commit(), rhs.commit()))
    }

    pub fn i32_eq(&mut self, lhs: NewValueId<i32>, rhs: NewValueId<i32>) -> NewValueId<bool> {
        self.push(Instr::I32Eq(lhs.commit(), rhs.commit()))
    }

    pub fn i32_ne(&mut self, lhs: NewValueId<i32>, rhs: NewValueId<i32>) -> NewValueId<bool> {
        self.push(Instr::I32Ne(lhs.commit(), rhs.commit()))
    }

    pub fn i32_lt(&mut self, lhs: NewValueId<i32>, rhs: NewValueId<i32>) -> NewValueId<bool> {
        self.push(Instr::I32Lt(lhs.commit(), rhs.commit()))
    }

    pub fn i32_le(&mut self, lhs: NewValueId<i32>, rhs: NewValueId<i32>) -> NewValueId<bool> {
        self.push(Instr::I32Le(lhs.commit(), rhs.commit()))
    }

    pub fn i32_gt(&mut self, lhs: NewValueId<i32>, rhs: NewValueId<i32>) -> NewValueId<bool> {
        self.push(Instr::I32Gt(lhs.commit(), rhs.commit()))
    }

    pub fn i32_ge(&mut self, lhs: NewValueId<i32>, rhs: NewValueId<i32>) -> NewValueId<bool> {
        self.push(Instr::I32Ge(lhs.commit(), rhs.commit()))
    }

    pub fn not(&mut self, src: NewValueId<bool>) -> NewValueId<bool> {
        self.push(Instr::Not(src.commit()))
    }

    pub fn and(&mut self, lhs: NewValueId<bool>, rhs: NewValueId<bool>) -> NewValueId<bool> {
        self.push(Instr::And(lhs.commit(), rhs.commit()))
    }

    pub fn or(&mut self, lhs: NewValueId<bool>, rhs: NewValueId<bool>) -> NewValueId<bool> {
        self.push(Instr::Or(lhs.commit(), rhs.commit()))
    }

    pub fn xor(&mut self, lhs: NewValueId<bool>, rhs: NewValueId<bool>) -> NewValueId<bool> {
        self.push(Instr::Xor(lhs.commit(), rhs.commit()))
    }

    pub fn i32_from_bool(&mut self, src: NewValueId<bool>) -> NewValueId<i32> {
        self.push(Instr::I32FromBool(src.commit()))
    }

    pub fn f32_from_i32(&mut self, src: NewValueId<i32>) -> NewValueId<f32> {
        self.push(Instr::F32FromI32(src.commit()))
    }

    pub fn f32_from_bool(&mut self, src: NewValueId<bool>) -> NewValueId<f32> {
        self.push(Instr::F32FromBool(src.commit()))
    }

    pub fn f32_neg(&mut self, src: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::F32Neg(src.commit()))
    }

    pub fn f32_abs(&mut self, src: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::F32Abs(src.commit()))
    }

    pub fn f32_sign(&mut self, src: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::F32Sign(src.commit()))
    }

    pub fn f32_floor(&mut self, src: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::F32Floor(src.commit()))
    }

    pub fn f32_add(&mut self, lhs: NewValueId<f32>, rhs: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::F32Add(lhs.commit(), rhs.commit()))
    }

    pub fn f32_sub(&mut self, lhs: NewValueId<f32>, rhs: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::F32Sub(lhs.commit(), rhs.commit()))
    }

    pub fn f32_mul(&mut self, lhs: NewValueId<f32>, rhs: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::F32Mul(lhs.commit(), rhs.commit()))
    }

    pub fn f32_div(&mut self, lhs: NewValueId<f32>, rhs: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::F32Div(lhs.commit(), rhs.commit()))
    }

    pub fn f32_min(&mut self, lhs: NewValueId<f32>, rhs: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::F32Min(lhs.commit(), rhs.commit()))
    }

    pub fn f32_max(&mut self, lhs: NewValueId<f32>, rhs: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::F32Max(lhs.commit(), rhs.commit()))
    }

    pub fn f32_powf(&mut self, lhs: NewValueId<f32>, rhs: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::F32Powf(lhs.commit(), rhs.commit()))
    }

    pub fn f32_powi(&mut self, lhs: NewValueId<f32>, rhs: NewValueId<i32>) -> NewValueId<f32> {
        self.push(Instr::F32Powi(lhs.commit(), rhs.commit()))
    }

    pub fn f32_exp(&mut self, src: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::F32Exp(src.commit()))
    }

    pub fn f32_ln(&mut self, src: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::F32Ln(src.commit()))
    }

    pub fn f32_lg(&mut self, src: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::F32Lg(src.commit()))
    }

    pub fn f32_sin(&mut self, src: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::F32Sin(src.commit()))
    }

    pub fn f32_cos(&mut self, src: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::F32Cos(src.commit()))
    }

    pub fn f32_tan(&mut self, src: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::F32Tan(src.commit()))
    }

    pub fn f32_cot(&mut self, src: NewValueId<f32>) -> NewValueId<f32> {
        self.push(Instr::F32Cot(src.commit()))
    }

    pub fn f32_eq(&mut self, lhs: NewValueId<f32>, rhs: NewValueId<f32>) -> NewValueId<bool> {
        self.push(Instr::F32Eq(lhs.commit(), rhs.commit()))
    }

    pub fn f32_ne(&mut self, lhs: NewValueId<f32>, rhs: NewValueId<f32>) -> NewValueId<bool> {
        self.push(Instr::F32Ne(lhs.commit(), rhs.commit()))
    }

    pub fn f32_lt(&mut self, lhs: NewValueId<f32>, rhs: NewValueId<f32>) -> NewValueId<bool> {
        self.push(Instr::F32Lt(lhs.commit(), rhs.commit()))
    }

    pub fn f32_le(&mut self, lhs: NewValueId<f32>, rhs: NewValueId<f32>) -> NewValueId<bool> {
        self.push(Instr::F32Le(lhs.commit(), rhs.commit()))
    }

    pub fn f32_gt(&mut self, lhs: NewValueId<f32>, rhs: NewValueId<f32>) -> NewValueId<bool> {
        self.push(Instr::F32Gt(lhs.commit(), rhs.commit()))
    }

    pub fn f32_ge(&mut self, lhs: NewValueId<f32>, rhs: NewValueId<f32>) -> NewValueId<bool> {
        self.push(Instr::F32Ge(lhs.commit(), rhs.commit()))
    }

    pub fn i32_sel(
        &mut self,
        cond: NewValueId<bool>,
        v_t: NewValueId<i32>,
        v_f: NewValueId<i32>,
    ) -> NewValueId<i32> {
        self.push(Instr::I32Sel(cond.commit(), v_t.commit(), v_f.commit()))
    }

    pub fn f32_sel(
        &mut self,
        cond: NewValueId<bool>,
        v_t: NewValueId<f32>,
        v_f: NewValueId<f32>,
    ) -> NewValueId<f32> {
        self.push(Instr::F32Sel(cond.commit(), v_t.commit(), v_f.commit()))
    }
}
