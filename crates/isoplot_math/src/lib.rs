use crate::{
    token::{PhantomToken, SimdToken},
    vec3::AVec3,
};

mod num;
mod seal;
mod token;
mod vec3;

pub type Vec3<T> = AVec3<T, PhantomToken>;
pub type Vec3S<S, V> = AVec3<V, SimdToken<S>>;
