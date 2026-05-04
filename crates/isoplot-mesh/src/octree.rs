use arrayvec::ArrayVec;
use bilge::prelude::*;
use derive_where::derive_where;
use std::marker::PhantomData;

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
    fn is_empty(&self, node: Quant) -> bool;

    /// Returns `true` if the given node is a leaf.
    fn is_leaf(&self, node: Quant) -> bool;

    /// Create a payload for the given leaf node.
    ///
    /// The provided node is guaranteed to be a leaf, meaning that `is_leaf`
    /// has previously returned `true` for it at least once.
    fn new_payload(&self, leaf: Quant) -> T;
}

/// An octree construction error
#[derive(Debug)]
pub struct BuildError;

pub struct Octree<T> {
    levels: ArrayVec<Level<T>, { MAX_LEVELS as usize }>,
}

impl<T: Payload> Octree<T> {
    pub fn build<S>(source: &S) -> Result<Self, BuildError> {
        todo!()
    }
}

struct Level<T> {
    nodes: Vec<Node<T>>,
}

#[derive_where(Copy, Clone, Debug)]
#[repr(transparent)]
struct Node<T> {
    inner: RawNode,
    _ty: PhantomData<T>,
}

impl<T> Node<T> {
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
}

#[bitsize(32)]
#[derive(Copy, Clone, DebugBits)]
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
        Self::new(NodeKind::Branch, data)
    }
}

#[bitsize(1)]
#[derive(Copy, Clone, Debug, FromBits)]
enum NodeKind {
    Branch = 0,
    Leaf = 1,
}

#[repr(transparent)]
pub struct Branch(RawBranch);

impl Branch {
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
struct RawBranch {
    /// Offset (in nodes) to the first child in the next level
    offset: u23,

    /// Bitmask of non-empty children
    ///
    /// Children are stored contiguously and only those with set bits
    /// are present, in order.
    mask: u8,
}

#[derive(Copy, Clone, Debug)]
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
