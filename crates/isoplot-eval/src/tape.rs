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
pub(crate) enum Instr {
    // Constants
    I32Const(i32),
    F32Const(f32),

    // Basic `i32` operations
    I32Add(ValueId, ValueId),
    I32Sub(ValueId, ValueId),
    I32Mul(ValueId, ValueId),

    // Conversions
    F32FromI32(ValueId),

    // Identity copy of any value
    Copy(ValueId),

    // Basic `f32` operations
    F32Neg(ValueId),
    F32Abs(ValueId),
    F32Sign(ValueId),
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

impl Instr {
    fn result_type(self, values: &[Type]) -> Type {
        match self {
            Instr::Copy(src) => values[src.index()],

            Instr::I32Const(_) | Instr::I32Add(..) | Instr::I32Sub(..) | Instr::I32Mul(..) => {
                Type::I32
            }

            Instr::F32Const(_)
            | Instr::F32FromI32(_)
            | Instr::F32Neg(_)
            | Instr::F32Abs(_)
            | Instr::F32Sign(_)
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
}

struct Shape {
    num_arguments: u8,
    num_results: u8,
}

pub(crate) struct Tape {
    shape: Shape,
    instrs: Vec<Instr>,
}

impl Tape {
    pub(crate) fn builder(arguments: Vec<Type>, results: Vec<Type>) -> TapeBuilder {
        assert!(
            arguments.len() <= u8::MAX as usize,
            "a tape can take at most {} arguments",
            u8::MAX
        );

        assert!(
            results.len() <= u8::MAX as usize,
            "a tape can produce at most {} results",
            u8::MAX
        );

        TapeBuilder {
            shape: Shape {
                num_arguments: arguments.len() as u8,
                num_results: results.len() as u8,
            },
            values: arguments,
            instrs: vec![],
            res_ty: results,
        }
    }

    pub(crate) fn num_arguments(&self) -> usize {
        self.shape.num_arguments as usize
    }

    pub(crate) fn num_results(&self) -> usize {
        self.shape.num_results as usize
    }

    pub(crate) fn instrs(&self) -> &[Instr] {
        &self.instrs
    }
}

#[derive(Debug)]
pub(crate) struct ValidateError(());

pub(crate) struct TapeBuilder {
    shape: Shape,
    instrs: Vec<Instr>,
    values: Vec<Type>,
    res_ty: Vec<Type>,
}

impl TapeBuilder {
    pub(crate) fn argument(&self, index: u32) -> ValueId {
        assert!(index < self.shape.num_arguments as u32);
        ValueId(index as u16)
    }

    pub(crate) fn instr(&mut self, instr: Instr) -> ValueId {
        let index = self.values.len();
        assert!(index < u16::MAX as usize);
        let ty = instr.result_type(&self.values);
        self.values.push(ty);
        self.instrs.push(instr);
        ValueId(index as u16)
    }

    pub(crate) fn build(self) -> Result<Tape, ValidateError> {
        self.validate()?;
        Ok(Tape {
            shape: self.shape,
            instrs: self.instrs,
        })
    }

    fn validate(&self) -> Result<(), ValidateError> {
        // Enforce that arguments are not implicitly used as results (not technically required by the backends).
        if self.values.len() < self.shape.num_arguments as usize + self.shape.num_results as usize {
            return Err(ValidateError(()));
        }

        if self.values[(self.values.len() - self.shape.num_results as usize)..] != self.res_ty {
            return Err(ValidateError(()));
        }

        for (index, &instr) in self.instrs.iter().enumerate() {
            let cur_dst = self.shape.num_arguments as usize + index;

            let check = |value: ValueId, ty: Type| {
                let value = value.0 as usize;
                if value < cur_dst && self.values[value] == ty {
                    Ok(())
                } else {
                    Err(ValidateError(()))
                }
            };

            match instr {
                Instr::I32Const(_) | Instr::F32Const(_) => {}
                Instr::Copy(src) => {
                    if src.0 as usize >= cur_dst {
                        return Err(ValidateError(()));
                    }
                }
                Instr::I32Add(lhs, rhs) | Instr::I32Sub(lhs, rhs) | Instr::I32Mul(lhs, rhs) => {
                    check(lhs, Type::I32)?;
                    check(rhs, Type::I32)?;
                }

                Instr::F32FromI32(src) => check(src, Type::I32)?,
                Instr::F32Neg(src)
                | Instr::F32Abs(src)
                | Instr::F32Sign(src)
                | Instr::F32Exp(src)
                | Instr::F32Ln(src)
                | Instr::F32Lg(src)
                | Instr::F32Sin(src)
                | Instr::F32Cos(src)
                | Instr::F32Tan(src)
                | Instr::F32Cot(src) => check(src, Type::F32)?,

                Instr::F32Add(lhs, rhs)
                | Instr::F32Sub(lhs, rhs)
                | Instr::F32Mul(lhs, rhs)
                | Instr::F32Div(lhs, rhs)
                | Instr::F32Min(lhs, rhs)
                | Instr::F32Max(lhs, rhs)
                | Instr::F32Powf(lhs, rhs) => {
                    check(lhs, Type::F32)?;
                    check(rhs, Type::F32)?;
                }

                Instr::F32Powi(lhs, rhs) => {
                    check(lhs, Type::F32)?;
                    check(rhs, Type::I32)?;
                }
            }
        }

        Ok(())
    }
}
