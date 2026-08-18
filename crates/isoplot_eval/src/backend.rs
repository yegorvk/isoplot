mod fallback;

#[cfg(feature = "cranelift")]
mod cranelift;

use std::marker::PhantomData;

use crate::{
    layout::{Layout, RawValue, VectorExt},
    tape::Tape,
};

mod private {
    use super::{Layout, RawValue, Tape};

    #[doc(hidden)]
    pub(super) trait Backend {
        type Instance<Ret: Layout>: Clone + Send + Sync;
        type Evaluator<Ret: Layout>;

        fn instantiate<Ret: Layout>(tape: Tape) -> Self::Instance<Ret>;
        fn evaluator<Ret: Layout>(instance: &Self::Instance<Ret>) -> Self::Evaluator<Ret>;

        fn evaluate_into<Ret: Layout>(
            evaluator: &Self::Evaluator<Ret>,
            args: &[RawValue],
            results: &mut [RawValue],
        );
    }
}

#[allow(private_bounds)]
pub trait Backend: private::Backend {}
impl<T: private::Backend> Backend for T {}

std::cfg_select! {
    feature = "default-cranelift" => {
        pub type DefaultBackend = Cranelift;
    }
    feature = "default-fallback" => {
        pub type DefaultBackend = Fallback;
    }
    _ => {
        compile_error!("no default backend feature enabled");
    }
}

/// Unoptimized bytecode interpreter
pub struct Fallback;

impl private::Backend for Fallback {
    type Instance<Ret: Layout> = fallback::Fallback;
    type Evaluator<Ret: Layout> = fallback::Evaluator;

    fn instantiate<Ret: Layout>(tape: Tape) -> Self::Instance<Ret> {
        fallback::Fallback::new(tape)
    }

    fn evaluator<Ret: Layout>(instance: &Self::Instance<Ret>) -> Self::Evaluator<Ret> {
        instance.evaluator()
    }

    #[inline(always)]
    fn evaluate_into<Ret: Layout>(
        evaluator: &Self::Evaluator<Ret>,
        args: &[RawValue],
        results: &mut [RawValue],
    ) {
        evaluator.evaluate_into(args, results)
    }
}

/// Cranelift-based JIT compiler
#[cfg(feature = "cranelift")]
pub struct Cranelift;

#[cfg(feature = "cranelift")]
impl private::Backend for Cranelift {
    type Instance<Ret: Layout> = cranelift::Instance<Ret>;
    type Evaluator<Ret: Layout> = cranelift::Instance<Ret>;

    fn instantiate<Ret: Layout>(tape: Tape) -> Self::Instance<Ret> {
        cranelift::Instance::new(&tape)
    }

    fn evaluator<Ret: Layout>(instance: &Self::Instance<Ret>) -> Self::Evaluator<Ret> {
        instance.clone()
    }

    #[inline(always)]
    fn evaluate_into<Ret: Layout>(
        evaluator: &Self::Evaluator<Ret>,
        args: &[RawValue],
        results: &mut [RawValue],
    ) {
        evaluator.evaluate_into(args, results)
    }
}

pub struct Instance<B, Args: Layout, Ret: Layout>
where
    B: Backend,
{
    instance: B::Instance<Ret>,
    _marker: PhantomData<fn(&Args)>,
}

impl<B: Backend, Args: Layout, Ret: Layout> Clone for Instance<B, Args, Ret> {
    fn clone(&self) -> Self {
        Self {
            instance: self.instance.clone(),
            _marker: PhantomData,
        }
    }
}

impl<B: Backend, Args: Layout, Ret: Layout> Instance<B, Args, Ret> {
    pub(crate) fn new(tape: Tape) -> Self {
        Self {
            instance: B::instantiate(tape),
            _marker: PhantomData,
        }
    }

    pub fn evaluator(&self) -> Evaluator<B, Args, Ret> {
        Evaluator {
            evaluator: B::evaluator(&self.instance),
            _marker: PhantomData,
        }
    }
}

pub struct Evaluator<B, Args: Layout, Ret: Layout>
where
    B: Backend,
{
    evaluator: B::Evaluator<Ret>,
    _marker: PhantomData<fn(&Args)>,
}

impl<B: Backend, Args: Layout, Ret: Layout> Evaluator<B, Args, Ret> {
    #[inline(always)]
    pub fn evaluate_into(&self, args: &Args, result: &mut Ret) {
        B::evaluate_into(&self.evaluator, args.raw_values(), result.raw_values_mut())
    }

    #[inline(always)]
    pub fn evaluate(&self, args: &Args) -> Ret {
        let mut result = Ret::zeroed();
        self.evaluate_into(args, &mut result);
        result
    }
}
