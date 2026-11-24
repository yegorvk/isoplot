mod fallback;

#[cfg(feature = "cranelift")]
mod cranelift;

#[cfg(feature = "cranelift")]
pub use self::cranelift::Cranelift;

pub use self::fallback::Fallback;
