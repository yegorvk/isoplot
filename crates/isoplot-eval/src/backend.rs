mod interp;

#[cfg(feature = "cranelift")]
mod cranelift;

use crate::tape::Tape;

mod private {
    use super::Tape;

    #[doc(hidden)]
    pub(super) trait Backend {
        /// The underlying compiled instance type
        type Instance: Clone + Send + Sync;

        /// The underlying evaluator type
        type Evaluator;

        /// Compiles IR into a shareable instance.
        fn instantiate(program: Tape) -> Self::Instance;

        /// Creates an evaluator for the instance.
        fn evaluator(instance: &Self::Instance) -> Self::Evaluator;

        /// Evaluates the expression with given inputs.
        fn evaluate(evaluator: &Self::Evaluator, inputs: &[f32]) -> f32;
    }
}

#[allow(private_bounds)]
pub trait Backend: private::Backend {}
impl<T: private::Backend> Backend for T {}

/// Unoptimized bytecode interpreter
pub struct Interpreter;

impl private::Backend for Interpreter {
    type Instance = interp::Instance;
    type Evaluator = interp::Evaluator;

    fn instantiate(program: Tape) -> Self::Instance {
        interp::Instance::new(program)
    }

    fn evaluator(instance: &Self::Instance) -> Self::Evaluator {
        instance.evaluator()
    }

    fn evaluate(evaluator: &Self::Evaluator, inputs: &[f32]) -> f32 {
        evaluator.evaluate(inputs)
    }
}

/// Cranelift-based JIT compiler
#[cfg(feature = "cranelift")]
pub struct Cranelift;

#[cfg(feature = "cranelift")]
impl private::Backend for Cranelift {
    type Instance = cranelift::Instance;
    type Evaluator = cranelift::Instance;

    fn instantiate(program: Tape) -> Self::Instance {
        cranelift::Instance::new(&program)
    }

    fn evaluator(instance: &Self::Instance) -> Self::Evaluator {
        instance.clone()
    }

    fn evaluate(evaluator: &Self::Evaluator, inputs: &[f32]) -> f32 {
        evaluator.evaluate(inputs)
    }
}

pub struct Instance<B>
where
    B: Backend,
{
    instance: B::Instance,
}

impl<B: Backend> Clone for Instance<B> {
    fn clone(&self) -> Self {
        Self {
            instance: self.instance.clone(),
        }
    }
}

#[allow(private_bounds)]
impl<B: Backend> Instance<B> {
    pub(crate) fn new(instrs: Tape) -> Self {
        Self {
            instance: B::instantiate(instrs),
        }
    }

    pub fn evaluator(&self) -> Evaluator<B> {
        Evaluator {
            evaluator: B::evaluator(&self.instance),
        }
    }
}

pub struct Evaluator<B>
where
    B: Backend,
{
    evaluator: B::Evaluator,
}

#[allow(private_bounds)]
impl<B: Backend> Evaluator<B> {
    pub fn evaluate(&self, inputs: &[f32]) -> f32 {
        B::evaluate(&self.evaluator, inputs)
    }
}
