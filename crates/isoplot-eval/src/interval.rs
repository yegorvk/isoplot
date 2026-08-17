use bytemuck::{Pod, Zeroable};

use crate::{
    backend::{Backend, DefaultMultiBackend, Evaluator, Instance},
    tape::Tape,
};

/// A closed and bounded interval
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct Interval {
    min: f32,
    max: f32,
}

impl Interval {
    pub const fn new(min: f32, max: f32) -> Self {
        assert!(min <= max);
        Self { min, max }
    }

    pub const fn min(&self) -> f32 {
        self.min
    }

    pub const fn max(&self) -> f32 {
        self.max
    }

    pub const fn contains_zero(&self) -> bool {
        self.min <= 0.0 && self.max >= 0.0
    }
}

pub(crate) struct IntervalTape(Tape);

impl IntervalTape {
    pub(crate) fn num_inputs(&self) -> usize {
        self.0.num_arguments() / 2
    }

    pub(crate) fn num_outputs(&self) -> usize {
        self.0.num_results() / 2
    }

    pub(crate) fn into_tape(self) -> Tape {
        self.0
    }

    pub(crate) fn build(source: &Tape) -> Self {
        _ = source;
        todo!("interval translation")
    }
}

pub type DefaultIntervalInstance = IntervalInstance<DefaultMultiBackend>;
pub type DefaultIntervalEvaluator = IntervalEvaluator<DefaultMultiBackend>;

pub struct IntervalInstance<B: Backend> {
    instance: Instance<B>,
    num_inputs: usize,
    num_outputs: usize,
}

impl<B: Backend> IntervalInstance<B> {
    pub(crate) fn new(tape: IntervalTape) -> Self {
        Self {
            num_inputs: tape.num_inputs(),
            num_outputs: tape.num_outputs(),
            instance: Instance::new(tape.into_tape()),
        }
    }
}

impl<B: Backend> Clone for IntervalInstance<B> {
    fn clone(&self) -> Self {
        Self {
            instance: self.instance.clone(),
            num_inputs: self.num_inputs,
            num_outputs: self.num_outputs,
        }
    }
}

impl<B: Backend> IntervalInstance<B> {
    pub fn evaluator(&self) -> IntervalEvaluator<B> {
        IntervalEvaluator {
            evaluator: self.instance.evaluator(),
            num_inputs: self.num_inputs,
            num_outputs: self.num_outputs,
        }
    }
}

pub struct IntervalEvaluator<B: Backend> {
    evaluator: Evaluator<B>,
    num_inputs: usize,
    num_outputs: usize,
}

impl<B: Backend> IntervalEvaluator<B> {
    pub fn evaluate(&self, inputs: &[Interval]) -> Interval {
        assert_eq!(self.num_outputs, 1);
        let mut output = [Interval::new(0.0, 0.0)];
        self.evaluate_into(inputs, &mut output);
        output[0]
    }

    pub fn evaluate_into(&self, inputs: &[Interval], outputs: &mut [Interval]) {
        assert_eq!(inputs.len(), self.num_inputs);
        assert_eq!(outputs.len(), self.num_outputs);

        self.evaluator.evaluate_into(
            bytemuck::cast_slice(inputs),
            bytemuck::cast_slice_mut(outputs),
        );
    }
}
