mod fallback;

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

        /// Evaluates the expression, writing the results into `outputs`.
        fn evaluate_into(evaluator: &Self::Evaluator, inputs: &[f32], outputs: &mut [f32]);
    }

    #[doc(hidden)]
    pub(super) trait ScalarBackend: Backend {
        /// Evaluates the expression with given inputs.
        fn evaluate(evaluator: &Self::Evaluator, inputs: &[f32]) -> f32;
    }
}

#[allow(private_bounds)]
pub trait Backend: private::Backend {}
impl<T: private::Backend> Backend for T {}

#[allow(private_bounds)]
pub trait ScalarBackend: Backend + private::ScalarBackend {}
impl<T: private::ScalarBackend> ScalarBackend for T {}

std::cfg_select! {
    feature = "default-cranelift" => {
        pub type DefaultBackend = Cranelift;
        pub type DefaultMultiBackend = CraneliftMulti;
    }
    feature = "default-fallback" => {
        pub type DefaultBackend = Fallback;
        pub type DefaultMultiBackend = FallbackMulti;
    }
    _ => {
        compile_error!("no default backend feature enabled");
    }
}

pub type DefaultInstance = Instance<DefaultBackend>;
pub type DefaultMultiInstance = Instance<DefaultMultiBackend>;

/// Unoptimized bytecode interpreter
pub struct Fallback;

impl private::Backend for Fallback {
    type Instance = fallback::Fallback;
    type Evaluator = fallback::Evaluator;

    fn instantiate(program: Tape) -> Self::Instance {
        assert_eq!(program.num_results(), 1);
        fallback::Fallback::new(program)
    }

    fn evaluator(instance: &Self::Instance) -> Self::Evaluator {
        instance.evaluator()
    }

    fn evaluate_into(evaluator: &Self::Evaluator, inputs: &[f32], outputs: &mut [f32]) {
        evaluator.evaluate_into(inputs, outputs)
    }
}

impl private::ScalarBackend for Fallback {
    fn evaluate(evaluator: &Self::Evaluator, inputs: &[f32]) -> f32 {
        evaluator.evaluate(inputs)
    }
}

pub struct FallbackMulti;

impl private::Backend for FallbackMulti {
    type Instance = fallback::Fallback;
    type Evaluator = fallback::Evaluator;

    fn instantiate(program: Tape) -> Self::Instance {
        fallback::Fallback::new(program)
    }

    fn evaluator(instance: &Self::Instance) -> Self::Evaluator {
        instance.evaluator()
    }

    fn evaluate_into(evaluator: &Self::Evaluator, inputs: &[f32], outputs: &mut [f32]) {
        evaluator.evaluate_into(inputs, outputs)
    }
}

/// Cranelift-based JIT compiler
#[cfg(feature = "cranelift")]
pub struct Cranelift;

#[cfg(feature = "cranelift")]
impl private::Backend for Cranelift {
    type Instance = cranelift::ScalarInstance;
    type Evaluator = cranelift::ScalarInstance;

    fn instantiate(program: Tape) -> Self::Instance {
        cranelift::ScalarInstance::new(&program)
    }

    fn evaluator(instance: &Self::Instance) -> Self::Evaluator {
        instance.clone()
    }

    fn evaluate_into(evaluator: &Self::Evaluator, inputs: &[f32], outputs: &mut [f32]) {
        evaluator.evaluate_into(inputs, outputs)
    }
}

#[cfg(feature = "cranelift")]
impl private::ScalarBackend for Cranelift {
    fn evaluate(evaluator: &Self::Evaluator, inputs: &[f32]) -> f32 {
        evaluator.evaluate(inputs)
    }
}

#[cfg(feature = "cranelift")]
pub struct CraneliftMulti;

#[cfg(feature = "cranelift")]
impl private::Backend for CraneliftMulti {
    type Instance = cranelift::MultiInstance;
    type Evaluator = cranelift::MultiInstance;

    fn instantiate(program: Tape) -> Self::Instance {
        cranelift::MultiInstance::new(&program)
    }

    fn evaluator(instance: &Self::Instance) -> Self::Evaluator {
        instance.clone()
    }

    fn evaluate_into(evaluator: &Self::Evaluator, inputs: &[f32], outputs: &mut [f32]) {
        evaluator.evaluate_into(inputs, outputs)
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
    pub fn evaluate_into(&self, inputs: &[f32], outputs: &mut [f32]) {
        B::evaluate_into(&self.evaluator, inputs, outputs)
    }
}

#[allow(private_bounds)]
impl<B: ScalarBackend> Evaluator<B> {
    pub fn evaluate(&self, inputs: &[f32]) -> f32 {
        B::evaluate(&self.evaluator, inputs)
    }
}
