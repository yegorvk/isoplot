use arrayvec::ArrayVec;
use bilge::prelude::*;
use derive_where::derive_where;
use std::{marker::PhantomData, mem};

use crate::quant::Quant;

/// Maximum number of levels in an octree
///
/// Note that the root node is taken into account,
/// so there is always at least 1 level.
pub const MAX_LEVELS: u8 = 10;

const _: () = {
    assert!(MAX_LEVELS - 1 <= Quant::MAX_SUBDIV);
};

pub trait OctreeSource<T: Payload> {
    /// Returns `true` if the given node is empty (void).
    fn is_empty(&self, key: Quant) -> bool;

    /// Returns `true` if the given node is a leaf.
    fn is_leaf(&self, key: Quant) -> bool;

    /// Create a payload for the given leaf node.
    ///
    /// The provided node is guaranteed to be a leaf, meaning that `is_leaf`
    /// has previously returned `true` for it at least once.
    fn new_payload(&self, leaf_key: Quant) -> T;
}

pub struct Octree<T> {
    levels: ArrayVec<Level<T>, { MAX_LEVELS as usize }>,
}

impl<T: Payload> Octree<T> {
    pub fn build<S>(source: &S) -> Self
    where
        S: OctreeSource<T>,
    {
        let mut levels = ArrayVec::new();

        let mut this_keys = Vec::new();
        let mut next_keys = vec![Quant::root()];

        for _ in 0..MAX_LEVELS {
            let mut nodes = Vec::new();

            for key in next_keys.iter().copied() {
                if source.is_leaf(key) {
                    let payload = source.new_payload(key);
                    nodes.push(Leaf::new(payload).into());
                    continue;
                }

                let mut mask = 0u8;
                let offset = this_keys.len() as u32;

                for i in 0..8u8 {
                    let child_key = key.child(ChildIndex::new(i)).unwrap();

                    if !source.is_empty(child_key) {
                        mask |= 1u8 << i;
                        this_keys.push(child_key);
                    }
                }

                nodes.push(Branch::new(mask, offset).into());
            }

            levels.push(Level::new(nodes));

            mem::swap(&mut this_keys, &mut next_keys);
            this_keys.clear();

            if next_keys.is_empty() {
                break;
            }
        }

        Self { levels }
    }
}

#[derive(Debug)]
struct Level<T> {
    nodes: Vec<Node<T>>,
}

impl<T: Payload> Level<T> {
    fn new(nodes: Vec<Node<T>>) -> Self {
        Self { nodes }
    }
}

#[derive_where(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[repr(transparent)]
struct Node<T> {
    inner: RawNode,
    _ty: PhantomData<T>,
}

impl<T: Payload> Node<T> {
    #[inline]
    fn new_branch(branch: Branch) -> Self {
        Self {
            inner: RawNode::new_branch(branch),
            _ty: PhantomData,
        }
    }

    #[inline]
    fn new_leaf(leaf: Leaf<T>) -> Self {
        Self {
            inner: RawNode::new_leaf(leaf),
            _ty: PhantomData,
        }
    }

    fn is_leaf(self) -> bool {
        matches!(self.inner.kind(), NodeKind::Leaf)
    }

    fn as_leaf(self) -> Option<Leaf<T>> {
        if self.is_leaf() {
            Some(unsafe { Leaf::from_bits(self.inner.data()) })
        } else {
            None
        }
    }
}

impl<T: Payload> From<Branch> for Node<T> {
    fn from(value: Branch) -> Self {
        Self::new_branch(value)
    }
}

impl<T: Payload> From<Leaf<T>> for Node<T> {
    fn from(value: Leaf<T>) -> Self {
        Self::new_leaf(value)
    }
}

#[bitsize(32)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, DebugBits)]
struct RawNode {
    kind: NodeKind,
    data: u31,
}

impl RawNode {
    #[inline]
    fn new_branch(branch: Branch) -> Self {
        let data = branch.into_u31();
        Self::new(NodeKind::Branch, data)
    }

    #[inline]
    fn new_leaf<T>(leaf: Leaf<T>) -> Self {
        let data = leaf.payload;
        Self::new(NodeKind::Leaf, data)
    }
}

#[bitsize(1)]
#[derive(Copy, Clone, Debug, Hash, FromBits)]
enum NodeKind {
    Branch = 0,
    Leaf = 1,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[repr(transparent)]
pub struct Branch(RawBranch);

impl Branch {
    fn new(mask: u8, offset: u32) -> Self {
        Self(RawBranch::new(mask, u23::new(offset)))
    }

    #[inline]
    fn mask(self) -> u8 {
        self.0.mask()
    }

    #[inline]
    const fn into_u31(self) -> u31 {
        self.0.value
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct ChildIndex(pub u3);

impl ChildIndex {
    pub fn new(index: u8) -> Self {
        Self(u3::new(index))
    }
}

#[bitsize(31)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, DebugBits)]
struct RawBranch {
    /// Bitmask of non-empty children
    ///
    /// Children are stored contiguously and only those with set bits
    /// are present, in order.
    mask: u8,

    /// Offset (in nodes) to the first child in the next level
    offset: u23,
}

#[derive_where(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[repr(transparent)]
pub struct Leaf<T> {
    payload: u31,
    _ty: PhantomData<T>,
}

impl<T: Payload> Leaf<T> {
    #[inline]
    pub fn new(data: T) -> Self {
        Self {
            payload: data.into_bits(),
            _ty: PhantomData,
        }
    }

    unsafe fn from_bits(bits: u31) -> Self {
        Self {
            payload: bits,
            _ty: PhantomData,
        }
    }

    #[inline]
    pub fn get(self) -> T {
        unsafe { T::from_bits(self.payload) }
    }
}

pub trait Payload: Copy {
    /// Converts `Self` into a 31-bit unsigned integer.
    fn into_bits(self) -> u31;

    /// Reconstructs `Self` from a 31-bit unsigned integer.
    ///
    /// # Safety
    /// `bits` must have been previously returned by `into_bits`.
    unsafe fn from_bits(bits: u31) -> Self;
}

const CHILD_INDICES: [[ChildIndex; 8]; 256] = generate_child_indices();

const fn generate_child_indices() -> [[ChildIndex; 8]; 256] {
    let mut table = [[ChildIndex(u3::ZERO); 8]; 256];

    let mut mask = 0;
    while mask < 256 {
        let row = &mut table[mask];

        let mut next = 0;
        let mut bit = 0u8;

        while bit < 8 {
            if (mask >> bit) & 1 != 0 {
                row[next] = ChildIndex(u3::new(bit as u8));
                next += 1;
            }
            bit += 1;
        }

        mask += 1;
    }

    table
}

impl Branch {
    #[inline]
    fn child_indices(self) -> &'static [ChildIndex] {
        let mask = self.mask();
        let count = mask.count_ones() as usize;

        // SAFETY: `mask` cannot exceed 255 and `count` cannot exceed 8.
        unsafe {
            let indices = CHILD_INDICES.get_unchecked(mask as usize);
            indices.get_unchecked(0..count)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashSet};

    use super::*;

    #[test]
    fn test_octree_degenerate() {
        struct Source;

        impl OctreeSource<Quant> for Source {
            fn is_empty(&self, _: Quant) -> bool {
                false
            }

            fn is_leaf(&self, _: Quant) -> bool {
                true
            }

            fn new_payload(&self, leaf_key: Quant) -> Quant {
                leaf_key
            }
        }

        let octree = Octree::build(&Source);
        assert_eq!(octree.levels.len(), 1);

        let nodes = &octree.levels[0].nodes;
        assert_eq!(nodes.as_slice(), &[Leaf::new(Quant::root()).into()]);
    }

    #[test]
    fn test_octree_uniform() {
        struct Source;

        impl OctreeSource<Quant> for Source {
            fn is_empty(&self, _: Quant) -> bool {
                false
            }

            fn is_leaf(&self, key: Quant) -> bool {
                key.level() == 4
            }

            fn new_payload(&self, leaf_key: Quant) -> Quant {
                leaf_key
            }
        }

        let octree = Octree::build(&Source);
        assert_eq!(octree.levels.len(), 5);

        let mut leaves = HashSet::new();

        for (i, level) in octree.levels.iter().enumerate() {
            let nodes = &level.nodes;
            assert_eq!(nodes.len(), 1usize << (3 * i));

            if i < 4 {
                for node in nodes {
                    assert!(!node.is_leaf());
                }
            }

            if i == 4 {
                for node in nodes {
                    leaves.insert(node.as_leaf().unwrap());
                }
            }
        }

        assert_eq!(leaves.len(), octree.levels[4].nodes.len());

        for leaf in leaves {
            let payload = leaf.get();
            assert_eq!(payload.level(), 4);
        }
    }
}
