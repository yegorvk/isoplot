use crate::Program;

#[cfg(feature = "interpreter")]
pub(crate) mod interp;

#[cfg(feature = "cranelift")]
pub(crate) mod cranelift;

#[cfg(not(any(feature = "interpreter", feature = "cranelift")))]
compile_error!("at least one backend feature must be enabled");

pub(crate) enum Instance {
    #[cfg(feature = "interpreter")]
    Interp(interp::Instance),
}

impl Instance {
    pub(crate) fn new(program: Program, consts: Vec<f32>) -> Self {
        Instance::Interp(interp::Instance::new(program, consts))
    }

    pub(crate) fn call(&self, inputs: &[&[f32]], out: &mut [f32]) {
        match self {
            Instance::Interp(instance) => instance.call(inputs, out),
        }
    }
}
