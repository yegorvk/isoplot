mod ast;
mod parser;
mod span;
mod symbol;
mod token;

#[derive(Copy, Clone, Debug)]
pub enum Value {
    F32(f32),
    Unit,
}
