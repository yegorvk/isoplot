/// A value type
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) enum Type {
    I32,
    F32,
}

/// A value id (e.g., a temporary value)
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) struct ValueId(u16);

impl ValueId {
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum Instruction {
    // Constants
    I32Const(i32),
    F32Const(f32),

    // Basic `i32` operations
    I32Add(ValueId, ValueId),
    I32Sub(ValueId, ValueId),
    I32Mul(ValueId, ValueId),

    // Conversions
    F32FromI32(ValueId),

    // Basic `f32` operations
    F32Neg(ValueId),
    F32Abs(ValueId),
    F32Add(ValueId, ValueId),
    F32Sub(ValueId, ValueId),
    F32Mul(ValueId, ValueId),
    F32Div(ValueId, ValueId),
    F32Min(ValueId, ValueId),
    F32Max(ValueId, ValueId),
    F32Powf(ValueId, ValueId),
    F32Powi(ValueId, ValueId),

    // Common `f32` math functions
    F32Exp(ValueId),
    F32Ln(ValueId),
    F32Lg(ValueId),
    F32Sin(ValueId),
    F32Cos(ValueId),
    F32Tan(ValueId),
    F32Cot(ValueId),
}

impl Instruction {
    fn result_type(self) -> Type {
        match self {
            Instruction::I32Const(_)
            | Instruction::I32Add(..)
            | Instruction::I32Sub(..)
            | Instruction::I32Mul(..) => Type::I32,

            Instruction::F32Const(_)
            | Instruction::F32FromI32(_)
            | Instruction::F32Neg(_)
            | Instruction::F32Abs(_)
            | Instruction::F32Add(..)
            | Instruction::F32Sub(..)
            | Instruction::F32Mul(..)
            | Instruction::F32Div(..)
            | Instruction::F32Min(..)
            | Instruction::F32Max(..)
            | Instruction::F32Powf(..)
            | Instruction::F32Powi(..)
            | Instruction::F32Exp(_)
            | Instruction::F32Ln(_)
            | Instruction::F32Lg(_)
            | Instruction::F32Sin(_)
            | Instruction::F32Cos(_)
            | Instruction::F32Tan(_)
            | Instruction::F32Cot(_) => Type::F32,
        }
    }
}

pub(crate) struct Instructions {
    num_params: usize,
    instrs: Vec<Instruction>,
}

impl Instructions {
    pub(crate) fn builder(params: Vec<Type>) -> InstructionsBuilder {
        InstructionsBuilder {
            num_params: params.len(),
            values: params,
            instrs: Vec::new(),
        }
    }

    pub(crate) fn num_params(&self) -> usize {
        self.num_params
    }

    pub(crate) fn instrs(&self) -> &[Instruction] {
        &self.instrs
    }
}

#[derive(Debug)]
pub(crate) struct ValidateError;

pub(crate) struct InstructionsBuilder {
    num_params: usize,
    instrs: Vec<Instruction>,
    values: Vec<Type>,
}

impl InstructionsBuilder {
    pub(crate) fn arg(&self, index: u32) -> ValueId {
        assert!((index as usize) < self.num_params);
        ValueId(index as u16)
    }

    pub(crate) fn instr(&mut self, instr: Instruction) -> ValueId {
        let index = self.values.len();
        assert!(index < u16::MAX as usize);
        self.values.push(instr.result_type());
        self.instrs.push(instr);
        ValueId(index as u16)
    }

    pub(crate) fn build(self) -> Result<Instructions, ValidateError> {
        self.validate()?;
        Ok(Instructions {
            num_params: self.num_params,
            instrs: self.instrs,
        })
    }

    fn validate(&self) -> Result<(), ValidateError> {
        if self.values.last() != Some(&Type::F32) {
            return Err(ValidateError);
        }

        for (index, &instr) in self.instrs.iter().enumerate() {
            let cur_dst = self.num_params + index;

            let check = |value: ValueId, ty: Type| {
                let value = value.0 as usize;

                if value < cur_dst && self.values[value] == ty {
                    Ok(())
                } else {
                    Err(ValidateError)
                }
            };

            match instr {
                Instruction::I32Const(_) | Instruction::F32Const(_) => {}
                Instruction::I32Add(lhs, rhs)
                | Instruction::I32Sub(lhs, rhs)
                | Instruction::I32Mul(lhs, rhs) => {
                    check(lhs, Type::I32)?;
                    check(rhs, Type::I32)?;
                }

                Instruction::F32FromI32(src) => check(src, Type::I32)?,
                Instruction::F32Neg(src)
                | Instruction::F32Abs(src)
                | Instruction::F32Exp(src)
                | Instruction::F32Ln(src)
                | Instruction::F32Lg(src)
                | Instruction::F32Sin(src)
                | Instruction::F32Cos(src)
                | Instruction::F32Tan(src)
                | Instruction::F32Cot(src) => check(src, Type::F32)?,

                Instruction::F32Add(lhs, rhs)
                | Instruction::F32Sub(lhs, rhs)
                | Instruction::F32Mul(lhs, rhs)
                | Instruction::F32Div(lhs, rhs)
                | Instruction::F32Min(lhs, rhs)
                | Instruction::F32Max(lhs, rhs)
                | Instruction::F32Powf(lhs, rhs) => {
                    check(lhs, Type::F32)?;
                    check(rhs, Type::F32)?;
                }

                Instruction::F32Powi(lhs, rhs) => {
                    check(lhs, Type::F32)?;
                    check(rhs, Type::I32)?;
                }
            }
        }

        Ok(())
    }
}
