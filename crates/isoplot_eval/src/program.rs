use std::marker::PhantomData;
use thiserror::Error;

use crate::{
    autodiff,
    backend::{Backend, Instance},
    diag,
    frontend::{Bindings, ParseError, Parsed, ValidateError, lower_type, parse},
    interval,
    layout::{Layout, TypedValue},
    tape::{Tape, Type},
};

#[derive(Error, Debug)]
#[error(transparent)]
pub struct CompileError {
    #[from]
    kind: CompileErrorKind,
}

#[derive(Error, Debug)]
enum CompileErrorKind {
    #[error(transparent)]
    Parse(#[from] ParseError),

    #[error(transparent)]
    Validate(#[from] ValidateError),
}

impl CompileError {
    pub fn diagnostics(&self) -> &[diag::Diagnostic] {
        match &self.kind {
            CompileErrorKind::Parse(error) => error.diagnostics(),
            CompileErrorKind::Validate(error) => error.diagnostics(),
        }
    }
}

pub struct ProgramDesc<'a, Args> {
    bindings: Bindings<'a>,
    _marker: PhantomData<fn(&Args)>,
}

impl<'a, Args: Layout> ProgramDesc<'a, Args> {
    pub fn new(args: &[&'a str], consts: &'a [(&'a str, TypedValue)]) -> Self {
        Self {
            bindings: Bindings::new::<Args>(args, consts),
            _marker: PhantomData,
        }
    }

    pub(crate) fn bindings(&self) -> &Bindings<'a> {
        &self.bindings
    }
}

pub struct Program<Args, Ret> {
    tape: Tape,
    _marker: PhantomData<fn() -> (Args, Ret)>,
}

impl<Args: Layout> Program<Args, f32> {
    pub fn compile(desc: &ProgramDesc<Args>, source: &str) -> Result<Self, CompileError> {
        let (parsed, mut diagnostics) = parse(source);
        let validated = parsed.validate(desc.bindings());

        if !diagnostics.is_empty() {
            if let Err(error) = &validated {
                diagnostics.extend_from_slice(error.diagnostics());
            }
            return Err(CompileErrorKind::Parse(ParseError::new(diagnostics)).into());
        }

        let tape = validated.map_err(CompileErrorKind::from)?.lower_to_ir();
        Ok(Self::new(tape))
    }

    pub fn autodiff(&self) -> Program<Args, autodiff::Gradient<Args>> {
        autodiff::autodiff(self)
    }
}

impl<Args: Layout, Ret: Layout> Program<Args, Ret> {
    pub(crate) fn new(tape: Tape) -> Self {
        assert_eq!(
            tape.arg_types(),
            vector_types::<Args>(),
            "argument layout does not match the program"
        );

        assert_eq!(
            tape.result_types(),
            vector_types::<Ret>(),
            "result layout does not match the program"
        );

        Self {
            tape,
            _marker: PhantomData,
        }
    }

    pub(crate) fn tape(&self) -> &Tape {
        &self.tape
    }

    pub fn instantiate<B: Backend>(self) -> Instance<B, Args, Ret> {
        Instance::new(self.tape)
    }
}

fn vector_types<T: Layout>() -> Vec<Type> {
    T::types().map(lower_type).collect()
}

pub fn dump_ast<Args: Layout>(
    desc: &ProgramDesc<Args>,
    source: &str,
) -> (Parsed, Vec<diag::Diagnostic>) {
    let (parsed, mut diagnostics) = parse(source);

    if let Err(error) = parsed.validate(desc.bindings()) {
        diagnostics.extend_from_slice(error.diagnostics());
    }

    (parsed, diagnostics)
}

impl<Args, Ret> Program<Args, Ret>
where
    Args: interval::IntervalType,
    Ret: interval::IntervalType,
{
    pub fn interval(&self) -> Program<Args::Interval, Ret::Interval> {
        interval::interval(self)
    }
}
