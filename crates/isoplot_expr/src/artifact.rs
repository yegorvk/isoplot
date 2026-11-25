use std::marker::PhantomData;
use thiserror::Error;

use isoplot_tape::{Tape, ValueType};

use crate::{
    Interval,
    autodiff::{Dual, lower_autodiff},
    diag::Diagnostic,
    frontend::{ParamBindings, ParseError, Parsed, ValidateError, parse},
    interval::lower_interval,
    schema::{CastSchemaSafe, FieldType, Schema, Slot, Tile},
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
pub struct Artifact<Params, Ret, const TILE_LEN: usize = 1> {
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
    pub fn lower_autodiff(&self) -> Artifact<[f32; N], Dual<f32, N>> {
        lower_autodiff(self.cast_ref()).cast()
    }

    pub fn lower_interval(&self) -> Artifact<[Interval<f32>; N], Interval<f32>> {
        lower_interval::<1, f32, [Slot<f32>; N], Slot<f32>>(self)
    }
}

impl<const TILE_LEN: usize, const N: usize, const M: usize>
    Artifact<[Tile<f32, TILE_LEN>; N], [Tile<f32, TILE_LEN>; M], TILE_LEN>
{
    pub fn lower_interval(
        &self,
    ) -> Artifact<[Interval<Tile<f32, TILE_LEN>>; N], [Interval<Tile<f32, TILE_LEN>>; M], TILE_LEN>
    {
        lower_interval::<TILE_LEN, Tile<f32, TILE_LEN>, [Slot<f32>; N], [Slot<f32>; M]>(self)
    }
}

impl<const N: usize, const M: usize> Artifact<[f32; N], Dual<f32, M>> {
    pub fn lower_interval(&self) -> Artifact<[Interval<f32>; N], Dual<Interval<f32>, M>> {
        lower_interval::<1, f32, [Slot<f32>; N], Dual<Slot<f32>, M>>(self)
    }
}

impl<const TILE_LEN: usize, const N: usize, const M: usize>
    Artifact<[Tile<f32, TILE_LEN>; N], Dual<Tile<f32, TILE_LEN>, M>, TILE_LEN>
{
    pub fn lower_interval(
        &self,
    ) -> Artifact<
        [Interval<Tile<f32, TILE_LEN>>; N],
        Dual<Interval<Tile<f32, TILE_LEN>>, M>,
        TILE_LEN,
    > {
        lower_interval::<TILE_LEN, Tile<f32, TILE_LEN>, [Slot<f32>; N], Dual<Slot<f32>, M>>(self)
    }
}

impl<const TILE_LEN: usize, Params, Ret> Artifact<Params, Ret, TILE_LEN>
where
    Params: Schema<TILE_LEN>,
    Ret: Schema<TILE_LEN>,
{
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

    pub(crate) fn cast<P, R>(self) -> Artifact<P, R, TILE_LEN>
    where
        P: Schema<TILE_LEN> + CastSchemaSafe<TILE_LEN, Params>,
        R: Schema<TILE_LEN>,
        Ret: CastSchemaSafe<TILE_LEN, R>,
    {
        Artifact::from_tape(self.into_tape())
    }

    pub(crate) fn cast_ref<P, R>(&self) -> &Artifact<P, R, TILE_LEN>
    where
        P: Schema<TILE_LEN> + CastSchemaSafe<TILE_LEN, Params>,
        R: Schema<TILE_LEN>,
        Ret: CastSchemaSafe<TILE_LEN, R>,
    {
        let ptr: *const Artifact<P, R, TILE_LEN> = (self as *const Self).cast();
        unsafe { ptr.as_ref_unchecked() }
    }

    pub(crate) fn into_tape(self) -> Tape {
        self.tape
    }

    pub(crate) fn tape(&self) -> &Tape {
        &self.tape
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
