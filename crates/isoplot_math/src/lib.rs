use crate::{
    masked::MaskedAVec3,
    token::{PhantomToken, SimdToken},
    vec3::AVec3,
};

mod finite;
mod masked;
mod num;
mod seal;
mod token;
mod vec3;

pub use finite::{Finite, FiniteS, MaybeFinite, MaybeFiniteS};

pub type Vec3<T> = AVec3<T, PhantomToken>;
pub type Vec3S<S, V> = AVec3<V, SimdToken<S>>;

pub type MaskedVec3<T> = MaskedAVec3<T, PhantomToken>;
pub type MaskedVec3S<S, V> = MaskedAVec3<V, SimdToken<S>>;
