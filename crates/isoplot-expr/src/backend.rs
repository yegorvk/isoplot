mod interp;

#[cfg(feature = "cranelift")]
mod cranelift;

use crate::instrs::Instructions;

mod private {
    use super::Instructions;

    #[doc(hidden)]
    pub(crate) trait Backend {
        /// The underlying instance type
        type Instance: Clone;

        /// Creates an instance from IR.
        fn instantiate(program: Instructions) -> Self::Instance;

        /// Evaluates the expression with given inputs.
        fn evaluate(instance: &mut Self::Instance, inputs: &[f32]) -> f32;
    }
}

pub trait Backend: private::Backend {}
impl<T: private::Backend> Backend for T {}

/// Unoptimized bytecode interpreter
pub struct Interpreter;

impl private::Backend for Interpreter {
    type Instance = interp::Instance;

    fn instantiate(program: Instructions) -> Self::Instance {
        interp::Instance::new(program)
    }

    fn evaluate(instance: &mut Self::Instance, inputs: &[f32]) -> f32 {
        instance.evaluate(inputs)
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
    pub(crate) fn new(instrs: Instructions) -> Self {
        Self {
            instance: B::instantiate(instrs),
        }
    }

    pub fn evaluate(&mut self, inputs: &[f32]) -> f32 {
        B::evaluate(&mut self.instance, inputs)
    }
}
