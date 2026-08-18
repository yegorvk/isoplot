use std::{
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

pub(crate) fn traverse_ping_pong<T, F>(roots: Vec<T>, mut f: F) -> Vec<T>
where
    F: FnMut(&[T], &mut Vec<T>),
{
    let mut current = roots;
    let mut next = Vec::new();

    while !current.is_empty() {
        next.clear();
        f(&current, &mut next);
        mem::swap(&mut current, &mut next);
    }

    next
}
