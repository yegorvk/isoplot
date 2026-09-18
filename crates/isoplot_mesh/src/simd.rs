use fearless_simd::{
    Bytes, Select, Simd, SimdBase, SimdFloat, SimdFrom, SimdInt, SimdInto, SimdMask,
};

mod seal {
    pub(crate) trait Seal {}
}

use seal::Seal;

impl<S: Simd> Seal for u32s<S> {}
impl<S: Simd> Seal for i32s<S> {}
impl<S: Simd> Seal for f32s<S> {}

pub(crate) trait Bitcast: Copy + Seal {
    type Bytes;

    fn into_bytes(self) -> Self::Bytes;
    fn from_bytes(bytes: Self::Bytes) -> Self;

    #[inline(always)]
    fn bitcast<U>(self) -> U
    where
        U: Bitcast<Bytes = Self::Bytes>,
    {
        U::from_bytes(self.into_bytes())
    }
}

macro_rules! simd_base_impls {
    ($(($simd_ty:ident, $lane_ty:ident, $mask_ty:ident)),* $(,)?) => {
        $(
            impl<S: Simd> $simd_ty<S> {
                #[inline(always)]
                pub(crate) fn witness(self) -> S {
                    self.0.witness()
                }

                #[inline(always)]
                pub(crate) fn splat(simd: S, value: $lane_ty) -> Self {
                    Self(S::$simd_ty::splat(simd, value))
                }

                #[inline(always)]
                pub(crate) fn simd_eq(self, rhs: impl SimdInto<Self, S>) -> $mask_ty<S> {
                    let rhs: Self = rhs.simd_into(self.witness());
                    $mask_ty(self.0.simd_eq(rhs.0))
                }

                #[inline(always)]
                pub(crate) fn simd_ne(self, rhs: impl SimdInto<Self, S>) -> $mask_ty<S> {
                    !self.simd_eq(rhs)
                }
            }
        )*
    };
}

macro_rules! simd_num_impls {
    ($(($simd_ty:ident, $mask_ty:ident)),* $(,)?) => {
        $(
            impl<S: Simd> $simd_ty<S> {
                #[inline(always)]
                pub(crate) fn simd_lt(self, rhs: impl SimdInto<Self, S>) -> $mask_ty<S> {
                    let rhs: Self = rhs.simd_into(self.witness());
                    $mask_ty(self.0.simd_lt(rhs.0))
                }

                #[inline(always)]
                pub(crate) fn simd_le(self, rhs: impl SimdInto<Self, S>) -> $mask_ty<S> {
                    let rhs: Self = rhs.simd_into(self.witness());
                    $mask_ty(self.0.simd_le(rhs.0))
                }

                #[inline(always)]
                pub(crate) fn simd_gt(self, rhs: impl SimdInto<Self, S>) -> $mask_ty<S> {
                    let rhs: Self = rhs.simd_into(self.witness());
                    $mask_ty(self.0.simd_gt(rhs.0))
                }

                #[inline(always)]
                pub(crate) fn simd_ge(self, rhs: impl SimdInto<Self, S>) -> $mask_ty<S> {
                    let rhs: Self = rhs.simd_into(self.witness());
                    $mask_ty(self.0.simd_ge(rhs.0))
                }

                #[inline(always)]
                pub(crate) fn min(self, rhs: impl SimdInto<Self, S>) -> Self {
                    let rhs: Self = rhs.simd_into(self.witness());
                    Self(self.0.min(rhs.0))
                }

                #[inline(always)]
                pub(crate) fn max(self, rhs: impl SimdInto<Self, S>) -> Self {
                    let rhs: Self = rhs.simd_into(self.witness());
                    Self(self.0.max(rhs.0))
                }

                #[inline(always)]
                pub(crate) fn select(
                    mask: $mask_ty<S>,
                    if_true: impl SimdInto<Self, S>,
                    if_false: impl SimdInto<Self, S>,
                ) -> Self {
                    let simd = mask.witness();
                    Self(
                        mask.0
                            .select(if_true.simd_into(simd).0, if_false.simd_into(simd).0),
                    )
                }
            }
        )*
    };
}

macro_rules! simd_un_ops_impls {
    ($($simd_ty:ident => [$(($trait:ident, $method:ident)),* $(,)?]),* $(,)?) => {
        $(
            $(
                impl<S: Simd> ::core::ops::$trait for $simd_ty<S> {
                    type Output = Self;

                    #[inline(always)]
                    fn $method(self) -> Self::Output {
                        Self(::core::ops::$trait::$method(self.0))
                    }
                }
            )*
        )*
    };
}

macro_rules! simd_imm_bin_ops_impls {
    ($(($simd_ty:ident, $lane_ty:ident) => [$(($trait:ident, $method:ident)),* $(,)?]),* $(,)?) => {
        $(
            $(
                impl<S: Simd, Rhs> ::core::ops::$trait<Rhs> for $simd_ty<S>
                where
                    Rhs: SimdInto<Self, S>,
                {
                    type Output = Self;

                    #[inline(always)]
                    fn $method(self, rhs: Rhs) -> Self::Output {
                        let rhs = rhs.simd_into(self.witness()).0;
                        Self(::core::ops::$trait::$method(self.0, rhs))
                    }
                }

                impl<S: Simd> ::core::ops::$trait<$simd_ty<S>> for $lane_ty {
                    type Output = $simd_ty<S>;

                    #[inline(always)]
                    fn $method(self, rhs: $simd_ty<S>) -> Self::Output {
                        let lhs = $simd_ty::splat(rhs.witness(), self);
                        <$simd_ty<S> as ::core::ops::$trait>::$method(lhs, rhs)
                    }
                }
            )*
        )*
    };
}

macro_rules! simd_mut_bin_ops_impls {
    ($($simd_ty:ident => [$(($trait:ident, $method:ident)),* $(,)?]),* $(,)?) => {
        $(
            $(
                impl<S: Simd, Rhs> ::core::ops::$trait<Rhs> for $simd_ty<S>
                where
                    Rhs: SimdInto<Self, S>,
                {
                    #[inline(always)]
                    fn $method(&mut self, rhs: Rhs) {
                        let rhs = rhs.simd_into(self.witness()).0;
                        ::core::ops::$trait::$method(&mut self.0, rhs)
                    }
                }
            )*
        )*
    };
}

macro_rules! simd_float_impls {
    ($($simd_ty:ident),* $(,)?) => {
        $(
            impl<S: Simd> $simd_ty<S> {
                #[inline(always)]
                pub(crate) fn mul_add(
                    self,
                    op1: impl SimdInto<Self, S>,
                    op2: impl SimdInto<Self, S>,
                ) -> Self {
                    Self(self.0.mul_add(
                        op1.simd_into(self.witness()).0,
                        op2.simd_into(self.witness()).0,
                    ))
                }

                #[inline(always)]
                pub(crate) fn mul_sub(
                    self,
                    op1: impl SimdInto<Self, S>,
                    op2: impl SimdInto<Self, S>,
                ) -> Self {
                    Self(self.0.mul_sub(
                        op1.simd_into(self.witness()).0,
                        op2.simd_into(self.witness()).0,
                    ))
                }

                #[inline(always)]
                pub(crate) fn sqrt(self) -> Self {
                    Self(self.0.sqrt())
                }
            }
        )*
    };
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Eq)]
#[repr(transparent)]
pub(crate) struct mask32s<S: Simd>(S::mask32s);

impl<S: Simd> mask32s<S> {
    #[inline(always)]
    pub(crate) fn any_true(self) -> bool {
        self.0.any_true()
    }

    #[inline(always)]
    pub(crate) fn any_false(self) -> bool {
        self.0.any_false()
    }

    #[inline(always)]
    pub(crate) fn all_true(self) -> bool {
        self.0.all_true()
    }

    #[inline(always)]
    pub(crate) fn all_false(self) -> bool {
        self.0.all_false()
    }
}

impl<S: Simd> SimdFrom<bool, S> for mask32s<S> {
    #[inline(always)]
    fn simd_from(simd: S, value: bool) -> Self {
        Self(S::mask32s::splat(simd, value))
    }
}

impl<S: Simd> PartialEq for mask32s<S> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.simd_eq(*other).all_true()
    }
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Eq, Debug)]
#[repr(transparent)]
pub(crate) struct u32s<S: Simd>(S::u32s);

impl<S: Simd> u32s<S> {
    #[inline(always)]
    pub(crate) fn to_signed(self) -> i32s<S> {
        self.bitcast()
    }

    #[inline(always)]
    pub(crate) fn wrapping_neg(self) -> u32s<S> {
        (-self.to_signed()).bitcast()
    }

    #[inline(always)]
    pub(crate) fn to_float(self) -> f32s<S> {
        f32s(self.0.to_float())
    }
}

impl<S: Simd> SimdFrom<u32, S> for u32s<S> {
    #[inline(always)]
    fn simd_from(simd: S, value: u32) -> Self {
        Self(S::u32s::splat(simd, value))
    }
}

impl<S: Simd> Bitcast for u32s<S> {
    type Bytes = S::u8s;

    #[inline(always)]
    fn into_bytes(self) -> Self::Bytes {
        self.0.to_bytes()
    }

    #[inline(always)]
    fn from_bytes(bytes: Self::Bytes) -> Self {
        Self(S::u32s::from_bytes(bytes))
    }
}

impl<S: Simd> PartialEq for u32s<S> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.simd_eq(*other).all_true()
    }
}

impl<S: Simd> PartialEq<u32> for u32s<S> {
    #[inline(always)]
    fn eq(&self, other: &u32) -> bool {
        self.simd_eq(*other).all_true()
    }
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Eq, Debug)]
#[repr(transparent)]
pub(crate) struct i32s<S: Simd>(S::i32s);

impl<S: Simd> i32s<S> {
    #[inline(always)]
    pub(crate) fn to_unsigned(self) -> u32s<S> {
        self.bitcast()
    }

    #[inline(always)]
    pub(crate) fn to_float(self) -> f32s<S> {
        f32s(self.0.to_float())
    }
}

impl<S: Simd> SimdFrom<i32, S> for i32s<S> {
    #[inline(always)]
    fn simd_from(simd: S, value: i32) -> Self {
        Self(S::i32s::splat(simd, value))
    }
}

impl<S: Simd> Bitcast for i32s<S> {
    type Bytes = S::u8s;

    #[inline(always)]
    fn into_bytes(self) -> Self::Bytes {
        self.0.to_bytes()
    }

    #[inline(always)]
    fn from_bytes(bytes: Self::Bytes) -> Self {
        Self(S::i32s::from_bytes(bytes))
    }
}

impl<S: Simd> PartialEq for i32s<S> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.simd_eq(*other).all_true()
    }
}

impl<S: Simd> PartialEq<i32> for i32s<S> {
    #[inline(always)]
    fn eq(&self, other: &i32) -> bool {
        self.simd_eq(*other).all_true()
    }
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct f32s<S: Simd>(S::f32s);

impl<S: Simd> SimdFrom<f32, S> for f32s<S> {
    #[inline(always)]
    fn simd_from(simd: S, value: f32) -> Self {
        Self(S::f32s::splat(simd, value))
    }
}

impl<S: Simd> Bitcast for f32s<S> {
    type Bytes = S::u8s;

    #[inline(always)]
    fn into_bytes(self) -> Self::Bytes {
        self.0.to_bytes()
    }

    #[inline(always)]
    fn from_bytes(bytes: Self::Bytes) -> Self {
        Self(S::f32s::from_bytes(bytes))
    }
}

impl<S: Simd> PartialEq for f32s<S> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.simd_eq(*other).all_true()
    }
}

impl<S: Simd> PartialEq<f32> for f32s<S> {
    #[inline(always)]
    fn eq(&self, other: &f32) -> bool {
        self.simd_eq(*other).all_true()
    }
}

simd_base_impls!(
    (mask32s, bool, mask32s),
    (u32s, u32, mask32s),
    (i32s, i32, mask32s),
    (f32s, f32, mask32s)
);

simd_num_impls!((u32s, mask32s), (i32s, mask32s), (f32s, mask32s));
simd_float_impls!(f32s);

simd_un_ops_impls! {
    u32s => [(Not, not)],
    i32s => [(Not, not), (Neg, neg)],
    f32s => [(Neg, neg)],
    mask32s => [(Not, not)],
}

simd_imm_bin_ops_impls! {
    (u32s, u32) => [(Add, add),  (Sub, sub), (Mul, mul), (BitAnd, bitand), (BitOr, bitor), (BitXor, bitxor), (Shl, shl), (Shr, shr)],
    (i32s, i32) => [(Add, add),  (Sub, sub), (Mul, mul), (BitAnd, bitand), (BitOr, bitor), (BitXor, bitxor), (Shl, shl), (Shr, shr)],
    (f32s, f32) => [(Add, add),  (Sub, sub), (Mul, mul), (Div, div)],
    (mask32s, bool) => [(BitAnd, bitand), (BitOr, bitor), (BitXor, bitxor)],
}

simd_mut_bin_ops_impls! {
    u32s => [
        (AddAssign, add_assign), (SubAssign, sub_assign), (MulAssign, mul_assign),
        (BitAndAssign, bitand_assign), (BitOrAssign, bitor_assign),
        (BitXorAssign, bitxor_assign), (ShlAssign, shl_assign), (ShrAssign, shr_assign)
    ],
    i32s => [
        (AddAssign, add_assign), (SubAssign, sub_assign), (MulAssign, mul_assign),
        (BitAndAssign, bitand_assign), (BitOrAssign, bitor_assign),
        (BitXorAssign, bitxor_assign), (ShlAssign, shl_assign), (ShrAssign, shr_assign)
    ],
    f32s => [(AddAssign, add_assign), (SubAssign, sub_assign), (MulAssign, mul_assign), (DivAssign, div_assign)],
    mask32s => [(BitAndAssign, bitand_assign), (BitOrAssign, bitor_assign), (BitXorAssign, bitxor_assign)],
}
