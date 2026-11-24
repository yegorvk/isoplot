use crate::schema::Schema;
use std::{iter, mem::MaybeUninit};

pub trait Kernel<Params: Schema, Ret: Schema> {
    type Evaluator<'a>: Evaluate<Params, Ret>
    where
        Self: 'a;

    /// Creates an evaluator instance for this kernel.
    fn evaluator(&self) -> Self::Evaluator<'_>;
}

impl<Params: Schema, Ret: Schema, K: Kernel<Params, Ret>> Kernel<Params, Ret> for &K {
    type Evaluator<'a>
        = <K as Kernel<Params, Ret>>::Evaluator<'a>
    where
        Self: 'a;

    fn evaluator(&self) -> Self::Evaluator<'_> {
        <K>::evaluator(self)
    }
}

pub unsafe trait Evaluate<Params: Schema, Ret: Schema> {
    /// Evaluates the kernel with the given input.
    fn evaluate(&self, params: Params) -> Ret;

    #[inline]
    fn evaluate_batch(&self, params: &[Params], out: &mut [MaybeUninit<Ret>]) {
        assert_eq!(
            params.len(),
            out.len(),
            "input and output buffer length mismatch"
        );

        for (elem, out) in iter::zip(params, out) {
            out.write(self.evaluate(*elem));
        }
    }
}

unsafe impl<Params: Schema, Ret: Schema, E: Evaluate<Params, Ret>> Evaluate<Params, Ret> for &E {
    #[inline]
    fn evaluate(&self, params: Params) -> Ret {
        <E>::evaluate(self, params)
    }
}
