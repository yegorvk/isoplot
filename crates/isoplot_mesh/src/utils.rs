use std::{
    array,
    mem::{self, MaybeUninit},
    ptr,
};

pub(crate) fn array_transpose<T: Copy, const N: usize>(array: [Option<T>; N]) -> Option<[T; N]> {
    array_try_from(|i| array[i].ok_or(())).ok()
}

fn array_try_from<T, F, E, const N: usize>(mut f: F) -> Result<[T; N], E>
where
    F: FnMut(usize) -> Result<T, E>,
{
    struct Guard<'a, T, const N: usize> {
        array: &'a mut [MaybeUninit<T>; N],
        init_count: usize,
    }

    impl<'a, T, const N: usize> Drop for Guard<'a, T, N> {
        fn drop(&mut self) {
            for elem in &mut self.array[0..self.init_count] {
                unsafe { ptr::drop_in_place(elem.assume_init_mut()) };
            }
        }
    }

    let mut array = [const { MaybeUninit::uninit() }; N];

    let mut guard = Guard {
        array: &mut array,
        init_count: 0,
    };

    for (i, elem) in guard.array.iter_mut().enumerate() {
        elem.write(f(i)?);
        guard.init_count += 1;
    }

    mem::forget(guard);
    Ok(unsafe { assume_init_array(array) })
}

unsafe fn assume_init_array<T, const N: usize>(array: [MaybeUninit<T>; N]) -> [T; N] {
    array.map(|elem| unsafe { elem.assume_init() })
}

#[inline]
pub(crate) fn array_map_index<T, F, U, const N: usize>(a: [T; N], mut f: F) -> [U; N]
where
    F: FnMut(usize, T) -> U,
{
    let mut iter = a.into_iter();

    array::from_fn(|i| {
        // SAFETY: both array contain exactly `N` elements.
        let old = unsafe { iter.next().unwrap_unchecked() };
        f(i, old)
    })
}

pub(crate) fn traverse_ping_pong<T, F>(roots: Vec<T>, max_level: Option<usize>, mut f: F) -> Vec<T>
where
    F: FnMut(usize, &[T], &mut Vec<T>),
{
    let mut current = roots;
    let mut next = Vec::new();

    let mut level = 0usize;

    while !current.is_empty() {
        if let Some(max_level) = max_level
            && level > max_level
        {
            return current;
        }
        next.clear();
        f(level, &current, &mut next);
        mem::swap(&mut current, &mut next);
        level += 1;
    }

    next
}

pub(crate) fn transmute_empty_vec<T: Copy, U: Copy>(vec: Vec<T>) -> Vec<U> {
    const {
        assert!(size_of::<T>() == size_of::<U>() && align_of::<U>() <= align_of::<T>());
    }
    assert!(vec.is_empty());
    let (ptr, _, cap) = vec.into_raw_parts();
    unsafe { Vec::from_raw_parts(ptr.cast(), 0, cap) }
}

pub(crate) fn map_vec_in_place<T: Copy, U: Copy, F>(vec: Vec<T>, mut f: F) -> Vec<U>
where
    F: FnMut(T) -> U,
{
    const {
        assert!(size_of::<T>() == size_of::<U>() && align_of::<U>() <= align_of::<T>());
    }

    let (mut ptr, len, cap) = vec.into_raw_parts();

    // Since `T` and `U` are `Copy`, we only need to free the memory if `f` panics.
    // INVARIANT: owns the raw memory of `vec`.
    struct Guard<T> {
        ptr: *mut T,
        cap: usize,
    }

    impl<T> Drop for Guard<T> {
        fn drop(&mut self) {
            // SAFETY: the guard logically owns the raw memory of the original `Vec`.
            unsafe { Vec::<T>::from_raw_parts(self.ptr, 0, self.cap) };
        }
    }

    let guard = Guard { ptr, cap };

    // SAFETY: `vec` contains at least `len` elements and advancing one past allocation is always safe.
    let end = unsafe { ptr.add(len) };

    while ptr != end {
        // SAFETY:
        // `T` is `Copy`, so we can move it out of `ptr`. We are only overwriting
        // the current element with a layout-compatible value of type `U`. We are
        // not holding any active reference to the memory denoted by `ptr` at that moment.
        // If `f` panics, the memory is freed in the guard. We have guaranteed `len` elements.
        unsafe {
            let old = *ptr.as_ref_unchecked();
            let new = f(old);
            ptr.cast::<U>().write(new);
            ptr = ptr.add(1);
        }
    }

    // Restore the original pointer.
    let ptr = guard.ptr;

    // `Drop` must not run (to avoid use-after-free)
    mem::forget(guard);

    // SAFETY: the tuple `(ptr, len, cap)` comes from a valid `Vec` and all elements of `vec` have been replaced
    // with values of type `U`, which is layout-compatible with `T`.
    unsafe { Vec::from_raw_parts(ptr.cast::<U>(), len, cap) }
}

pub(crate) fn filter_branchless_into_vec<T: Copy, F>(values: &[T], out: &mut Vec<T>, f: F) -> usize
where
    F: FnMut(T) -> bool,
{
    filter_map_branchless_into_vec(values, out, |elem| elem, f)
}

fn filter_map_branchless_into_vec<T: Copy, U: Copy, F, P>(
    values: &[T],
    out: &mut Vec<U>,
    mut map_f: F,
    mut pred_f: P,
) -> usize
where
    F: FnMut(T) -> U,
    P: FnMut(T) -> bool,
{
    let num_selected: usize = values.iter().map(|elem| pred_f(*elem) as usize).sum();

    if num_selected == 0 {
        return 0;
    }

    out.reserve(num_selected);
    let mut ptr = unsafe { out.as_mut_ptr().add(out.len()) };

    let mut remaining = num_selected;

    for elem in values.iter().copied() {
        if remaining == 0 {
            break;
        }

        unsafe {
            ptr.write(map_f(elem));
            let delta = pred_f(elem) as usize;
            remaining -= delta;
            ptr = ptr.add(delta);
        }
    }

    assert_eq!(remaining, 0);

    unsafe {
        out.set_len(out.len() + num_selected);
    }

    num_selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_branchless_into_vec() {
        {
            let mut v = vec![];
            assert_eq!(
                filter_branchless_into_vec(&[1, 2, 3, 4], &mut v, |x| x % 2 == 0),
                2
            );
            assert_eq!(v.as_slice(), &[2, 4]);
        }

        {
            let mut v = vec![];
            assert_eq!(
                filter_branchless_into_vec(&[1, 2, 3, 4], &mut v, |x| x == 2),
                1
            );
            assert_eq!(v.as_slice(), &[2]);
        }

        {
            let mut v = vec![6, 7];
            assert_eq!(
                filter_branchless_into_vec(&[1, 2, 3, 4], &mut v, |x| x % 2 == 0),
                2
            );
            assert_eq!(v.as_slice(), &[6, 7, 2, 4]);
        }

        {
            let mut v = vec![];
            assert_eq!(
                filter_branchless_into_vec(&[1, 2, 3, 4], &mut v, |_| true),
                4
            );
            assert_eq!(v.as_slice(), &[1, 2, 3, 4]);
        }

        {
            let mut v = vec![];
            assert_eq!(
                filter_branchless_into_vec(&[1, 2, 3, 4], &mut v, |_| false),
                0
            );
            assert_eq!(v.as_slice(), &[]);
        }
    }

    #[test]
    fn test_map_vec_in_place() {
        {
            let v = vec![1, 2, 3];
            assert_eq!(
                map_vec_in_place(v, |x| x as f32).as_slice(),
                &[1f32, 2f32, 3f32]
            );
        }

        {
            let v = vec![1];
            assert_eq!(map_vec_in_place(v, |x| x as f32).as_slice(), &[1f32]);
        }

        {
            let v: Vec<i32> = vec![];
            assert_eq!(map_vec_in_place(v, |x| x as f32).as_slice(), &[]);
        }
    }
}
