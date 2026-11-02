mod builder;
mod instr;

use instr::RawValueId;
use std::ops;

pub use builder::{Inlined, TapeBuilder, ValidateError};
pub use instr::{AnyValueId, Instr, InstrRef, NewValueId, ValueId, ValueType};

#[derive(Debug)]
pub struct Tape {
    args: Vec<ValueType>,
    results: Vec<ValueType>,
    instrs: Vec<Instr>,
}

impl Tape {
    pub fn builder(args: Vec<ValueType>, results: Vec<ValueType>) -> TapeBuilder {
        TapeBuilder::new(args, results)
    }

    pub fn num_args(&self) -> usize {
        self.args.len()
    }

    pub fn num_results(&self) -> usize {
        self.results.len()
    }

    pub fn arg_types(&self) -> &[ValueType] {
        &self.args
    }

    pub fn result_types(&self) -> &[ValueType] {
        &self.results
    }

    pub fn instrs(&self) -> &[Instr] {
        &self.instrs
    }

    pub fn iter_ref(&self) -> impl DoubleEndedIterator<Item = InstrRef> + ExactSizeIterator + '_ {
        let num_args = self.args.len();
        self.instrs
            .iter()
            .enumerate()
            .map(move |(i, &instr)| InstrRef::new(RawValueId::new(num_args + i), instr))
    }
}

impl<T: Into<AnyValueId>> ops::Index<T> for Tape {
    type Output = Instr;

    fn index(&self, index: T) -> &Self::Output {
        let id = index.into();
        let instr = &self.instrs[id.index()];
        assert_eq!(instr.result_type(), id.ty);
        instr
    }
}
