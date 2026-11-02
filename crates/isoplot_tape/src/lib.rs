mod tape;

pub use tape::{
    AnyValueId, Inlined, Instr, InstrRef, NewValueId, Tape, TapeBuilder, ValidateError, ValueId,
    ValueType,
};
