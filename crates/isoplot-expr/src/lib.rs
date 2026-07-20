mod ast;
mod span;
mod token;

#[derive(Copy, Clone, Debug)]
pub enum Value {
    F32(f32),
    Unit,
}
