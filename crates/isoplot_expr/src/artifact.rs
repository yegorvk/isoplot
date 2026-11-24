use std::{marker::PhantomData, mem};
use thiserror::Error;

use isoplot_tape::{Tape, ValueType};

use crate::{
    Interval,
    autodiff::{Dual, autodiff},
    diag::Diagnostic,
    frontend::{ParamBindings, ParseError, Parsed, ValidateError, parse},
    interval::{interval, interval_d},
    schema::{FieldType, Schema},
};

#[derive(Error, Debug)]
#[error(transparent)]
pub struct CompileError {
    #[from]
    kind: ErrorKind,
}

impl CompileError {
    pub fn diagnostics(&self) -> &[Diagnostic] {
        match &self.kind {
            ErrorKind::Parse(e) => e.diagnostics(),
            ErrorKind::Validate(e) => e.diagnostics(),
        }
    }
}

#[derive(Error, Debug)]
enum ErrorKind {
    #[error(transparent)]
    Parse(#[from] ParseError),

    #[error(transparent)]
    Validate(#[from] ValidateError),
}

pub struct Shape<'a> {
    pub params: &'a [(&'a str, FieldType)],
}

#[derive(Debug)]
pub struct Artifact<Params, Ret> {
    tape: Tape,
    _marker: PhantomData<fn(Params) -> Ret>,
}

impl<Params: Schema> Artifact<Params, f32> {
    pub fn compile(shape: &Shape<'_>, source: &str) -> Result<Self, CompileError> {
        let bindings = ParamBindings::<Params>::new(shape.params);
        let (parsed, mut diagnostics) = parse(source);
        let validated = parsed.validate(&bindings);

        if !diagnostics.is_empty() {
            if let Err(error) = &validated {
                diagnostics.extend_from_slice(error.diagnostics());
            }
            return Err(ErrorKind::Parse(ParseError::new(diagnostics)).into());
        }

        let tape = validated.map_err(ErrorKind::from)?.lower_to_ir();
        Ok(Self::from_tape(tape))
    }
}

impl<const N: usize> Artifact<[f32; N], f32> {
    pub fn autodiff(&self) -> Artifact<[f32; N], Dual<f32, N>> {
        autodiff(self.into()).into()
    }
}

impl<const N: usize> Artifact<[f32; N], f32> {
    pub fn interval(&self) -> Artifact<[Interval<f32>; N], Interval<f32>> {
        let artifact: &Artifact<[[f32; 1]; N], [[f32; 1]; 1]> = unsafe { mem::transmute(self) };
        Artifact::from_tape(interval(artifact).into_tape())
    }
}

impl<const N: usize, const M: usize> Artifact<[f32; N], [f32; M]> {
    pub fn interval(&self) -> Artifact<[Interval<f32>; N], [Interval<f32>; M]> {
        let artifact: &Artifact<[[f32; 1]; N], [[f32; 1]; M]> = self.into();
        interval(artifact).into()
    }
}

impl<const N: usize, const M: usize> Artifact<[f32; N], Dual<f32, M>> {
    pub fn interval(&self) -> Artifact<[Interval<f32>; N], Dual<Interval<f32>, M>> {
        let artifact: &Artifact<[[f32; 1]; N], Dual<[f32; 1], M>> = unsafe { mem::transmute(self) };
        Artifact::from_tape(interval_d(artifact).into_tape())
    }
}

impl<Params: Schema, Ret: Schema> Artifact<Params, Ret> {
    pub(crate) fn from_tape(tape: Tape) -> Self {
        assert!(
            tape.param_types()
                .iter()
                .copied()
                .eq(Params::fields().map(|field| field_type(field.ty))),
            "tape parameters are not compatible with `Params` schema"
        );

        assert!(
            tape.result_types()
                .iter()
                .copied()
                .eq(Ret::fields().map(|field| field_type(field.ty))),
            "tape return values are not compatible with `Ret` schema"
        );

        Self {
            tape,
            _marker: PhantomData,
        }
    }

    pub(crate) fn into_tape(self) -> Tape {
        self.tape
    }

    pub(crate) fn tape(&self) -> &Tape {
        &self.tape
    }
}

impl<const N: usize> From<Artifact<[f32; N], f32>> for Artifact<[[f32; 1]; N], [f32; 1]> {
    fn from(value: Artifact<[f32; N], f32>) -> Self {
        Self {
            tape: value.tape,
            _marker: PhantomData,
        }
    }
}

impl<const N: usize> From<&Artifact<[f32; N], f32>> for &Artifact<[[f32; 1]; N], [f32; 1]> {
    fn from(value: &Artifact<[f32; N], f32>) -> Self {
        unsafe { mem::transmute(value) }
    }
}

impl<const N: usize, const M: usize> From<Artifact<[f32; N], [f32; M]>>
    for Artifact<[[f32; 1]; N], [[f32; 1]; M]>
{
    fn from(value: Artifact<[f32; N], [f32; M]>) -> Self {
        Self {
            tape: value.tape,
            _marker: PhantomData,
        }
    }
}

impl<const N: usize, const M: usize> From<&Artifact<[f32; N], [f32; M]>>
    for &Artifact<[[f32; 1]; N], [[f32; 1]; M]>
{
    fn from(value: &Artifact<[f32; N], [f32; M]>) -> Self {
        unsafe { mem::transmute(value) }
    }
}

const fn field_type(ty: FieldType) -> ValueType {
    match ty {
        FieldType::I32 => ValueType::I32,
        FieldType::F32 => ValueType::F32,
    }
}

pub fn dump_ast<Params: Schema>(shape: &Shape, source: &str) -> (Parsed, Vec<Diagnostic>) {
    let (parsed, mut diagnostics) = parse(source);
    if let Err(error) = parsed.validate(&ParamBindings::<Params>::new(shape.params)) {
        diagnostics.extend_from_slice(error.diagnostics());
    }
    (parsed, diagnostics)
}
