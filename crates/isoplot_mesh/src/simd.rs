use std::{marker::PhantomData, mem::MaybeUninit, ops, ptr::NonNull};

use bytemuck::{Pod, Zeroable};
use fearless_simd::{mask32x8, prelude::*};

#[derive(Copy, Clone, Eq, PartialEq, Debug, Pod, Zeroable)]
#[repr(transparent)]
pub struct BMask8(pub(crate) u8);

impl BMask8 {
    #[inline]
    pub(crate) fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(u8) -> bool,
    {
        let mut mask = 0u8;

        for i in 0..8u8 {
            mask |= (f(i) as u8) << i;
        }

        Self(mask)
    }

    pub(crate) const fn contains(self, index: u8) -> bool {
        self.0 & (1u8 << index) != 0
    }

    #[inline]
    pub(crate) const fn count_set(self) -> u8 {
        self.0.count_ones() as u8
    }

    #[inline(always)]
    pub(crate) fn unpack_32x8<S: Simd>(self, simd: S) -> mask32x8<S> {
        mask32x8::from_bitmask(simd, self.0 as u64)
    }
}

impl ops::Not for BMask8 {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl<S: Simd> From<mask32x8<S>> for BMask8 {
    #[inline(always)]
    fn from(value: mask32x8<S>) -> Self {
        Self(value.to_bitmask() as u8)
    }
}

impl<S: Simd> SimdFrom<BMask8, S> for mask32x8<S> {
    #[inline(always)]
    fn simd_from(simd: S, value: BMask8) -> Self {
        value.unpack_32x8(simd)
    }
}

pub unsafe trait WriteBuf<'a>: Sized {
    /// The type of elements in the output buffer.
    type Item: Copy;

    /// Writes all elements using the provided closure.
    ///
    /// The closure is passed the exact current buffer content,
    /// which might contain uninitialized memory.
    ///
    /// # Safety
    /// The closure must initialize all elements.
    unsafe fn write_all_with_raw<F>(self, f: F) -> &'a mut [Self::Item]
    where
        F: FnOnce(&mut [MaybeUninit<Self::Item>]);
}

pub(crate) trait WriteBufExt<'a>: WriteBuf<'a> {
    #[inline]
    fn write_all_with<F>(self, f: F) -> &'a mut [Self::Item]
    where
        F: FnOnce(&mut InitSlice<'_, Self::Item>),
    {
        unsafe {
            self.write_all_with_raw(|slice| {
                let mut state = InitSlice {
                    left: slice.len(),
                    ptr: NonNull::from_mut(slice).cast::<Self::Item>(),
                    _marker: PhantomData,
                };

                f(&mut state);

                if state.left != 0 {
                    panic!("not all elements have been initialized");
                }
            })
        }
    }

    #[inline]
    fn flatten(self) -> Flatten<Self>
    where
        Self: Sized,
        Flatten<Self>: WriteBuf<'a>,
    {
        Flatten(self)
    }

    #[inline]
    fn apply_index<F>(self, f: F) -> ApplyIndex<Self, F>
    where
        Self: Sized,
        F: FnMut(usize, &mut Self::Item),
    {
        ApplyIndex { buf: self, f }
    }
}

impl<'a, T: WriteBuf<'a>> WriteBufExt<'a> for T {}

pub(crate) struct Flatten<B>(B);

unsafe impl<'a, T: Copy, B, const N: usize> WriteBuf<'a> for Flatten<B>
where
    B: WriteBuf<'a, Item = [T; N]>,
{
    type Item = T;

    #[inline]
    unsafe fn write_all_with_raw<F>(self, f: F) -> &'a mut [Self::Item]
    where
        F: FnOnce(&mut [MaybeUninit<Self::Item>]),
    {
        unsafe {
            self.0
                .write_all_with_raw(|slice| {
                    let chunked = transpose_uninit_slice_mut(slice);
                    f(chunked.as_flattened_mut())
                })
                .as_flattened_mut()
        }
    }
}

pub(crate) struct ApplyIndex<B, F> {
    buf: B,
    f: F,
}

unsafe impl<'a, B: WriteBuf<'a>, F> WriteBuf<'a> for ApplyIndex<B, F>
where
    F: FnMut(usize, &mut B::Item),
{
    type Item = B::Item;

    #[inline]
    unsafe fn write_all_with_raw<W>(mut self, f: W) -> &'a mut [Self::Item]
    where
        W: FnOnce(&mut [MaybeUninit<Self::Item>]),
    {
        unsafe {
            let out = self.buf.write_all_with_raw(f);
            for (i, value) in out.iter_mut().enumerate() {
                (self.f)(i, value);
            }
            out
        }
    }
}

struct WriteVec<'a, T> {
    vec: &'a mut Vec<T>,
    additional: usize,
}

unsafe impl<'a, T: Copy> WriteBuf<'a> for WriteVec<'a, T> {
    type Item = T;

    #[inline]
    unsafe fn write_all_with_raw<F>(self, f: F) -> &'a mut [Self::Item]
    where
        F: FnOnce(&mut [MaybeUninit<Self::Item>]),
    {
        let old_len = self.vec.len();

        self.vec.reserve(self.additional);
        f(&mut self.vec.spare_capacity_mut()[..self.additional]);

        unsafe {
            self.vec.set_len(old_len + self.additional);
        }

        &mut self.vec.as_mut_slice()[old_len..]
    }
}

pub(crate) trait VecExt<T> {
    fn extend_write<'a>(&'a mut self, additional: usize) -> impl WriteBuf<'a, Item = T>;
    fn clear_write<'a>(&'a mut self, new_len: usize) -> impl WriteBuf<'a, Item = T>;
}

impl<T: Copy> VecExt<T> for Vec<T> {
    #[inline]
    fn extend_write<'a>(&'a mut self, additional: usize) -> impl WriteBuf<'a, Item = T> {
        WriteVec {
            vec: self,
            additional,
        }
    }

    #[inline]
    fn clear_write<'a>(&'a mut self, new_len: usize) -> impl WriteBuf<'a, Item = T> {
        self.clear();
        WriteVec {
            vec: self,
            additional: new_len,
        }
    }
}

pub(crate) struct InitSlice<'a, T> {
    ptr: NonNull<T>,
    left: usize,
    _marker: PhantomData<&'a mut [MaybeUninit<T>]>,
}

impl<'a, T: Copy> InitSlice<'a, T> {
    #[inline]
    pub(crate) fn write(&mut self, elem: T) {
        self.write_array([elem])
    }

    #[inline]
    pub(crate) fn write_array<const N: usize>(&mut self, a: [T; N]) {
        if self.left < N {
            panic!("not enough elements left");
        }

        self.left -= N;

        if size_of::<T>() == 0 {
            return;
        }

        unsafe {
            self.ptr.cast::<[T; N]>().write(a);
            self.ptr = self.ptr.add(N);
        }
    }

    #[inline]
    pub(crate) fn write_from_iter<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for elem in iter.into_iter().take(self.left) {
            unsafe {
                self.ptr.write(elem);
                self.ptr = self.ptr.add(1);
                self.left -= 1;
            }
        }
    }
}

#[inline]
const fn transpose_uninit_slice_mut<T: Copy, const N: usize>(
    s: &mut [MaybeUninit<[T; N]>],
) -> &mut [[MaybeUninit<T>; N]] {
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr().cast(), s.len()) }
}
