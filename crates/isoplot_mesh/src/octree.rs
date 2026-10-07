use bilge::prelude::*;
use derive_where::derive_where;
use std::{iter, marker::PhantomData, mem, ops::Index};

use crate::quant::Quant;

/// Maximum number of levels in an octree
///
/// Note that the root node is taken into account,
/// so there is always at least 1 level.
pub(crate) const MAX_LEVELS: u8 = 10;

pub(crate) trait BuildOctree<T> {
    /// Returns `true` if the specified node is empty.
    fn is_empty(&mut self, tag: Quant) -> bool;

    /// Returns `true` if the specified node is a leaf.
    fn is_leaf(&mut self, tag: Quant) -> bool;

    /// Returns the leaf payload associated with the specified tag.
    fn place_leaf(&mut self, tag: Quant) -> T;
}

#[derive(Debug)]
pub(crate) struct InlineOctree<T> {
    nodes: Vec<InlineNode<T>>,
}

impl<T> InlineOctree<T> {
    #[inline]
    pub(crate) fn get(&self, key: OctreeKey) -> Option<&InlineNode<T>> {
        self.nodes.get(key.0.as_usize())
    }
}

impl<T: Payload> InlineOctree<T> {
    pub(crate) fn build<B>(source: &mut B) -> Self
    where
        B: ?Sized + BuildOctree<T>,
    {
        let mut nodes = vec![];

        let mut this_tags = vec![];
        let mut next_tags = vec![Quant::root()];

        let mut anchors = vec![];

        let mut pending_branches = vec![];
        let mut pending_leaves = vec![];
        let mut pending_tags = vec![];

        let mut next_offset = next_tags.len() as u32;

        for _ in 0..MAX_LEVELS {
            nodes.reserve(next_tags.len());

            for (i, tag) in next_tags.iter().copied().enumerate() {
                let is_leaf_bit = source.is_leaf(tag) as u8;
                nodes.push(InlineNode::new_branch(Branch::new(is_leaf_bit, i as u32)));
            }

            let total_nodes = nodes.len();
            let new_nodes = &mut nodes[(total_nodes - next_tags.len())..];

            anchors.extend(iter::zip(
                new_nodes.iter().copied(),
                next_tags.iter().copied(),
            ));

            anchors.sort_unstable_by_key(|(node, _)| node.is_leaf());

            let num_leaves = anchors
                .iter()
                .filter(|(anchor, _)| {
                    let fake_branch = anchor.as_branch().unwrap();
                    fake_branch.mask() != 0
                })
                .count();

            let num_branches = anchors.len() - num_leaves;

            pending_branches.reserve(num_branches);
            pending_tags.reserve(num_branches);

            pending_leaves.reserve(anchors.len());

            for (anchor, tag) in anchors.drain(..) {
                let fake_branch = anchor.as_branch().unwrap();

                let is_leaf = fake_branch.mask() != 0;
                let index = fake_branch.base_offset() as usize;

                if is_leaf {
                    let payload = source.place_leaf(tag);
                    pending_leaves.push((index, InlineLeaf::new(payload)));
                    continue;
                }

                let mut mask = 0u8;

                for child_index in ChildIndex::enumerate() {
                    let child = tag.child(child_index).unwrap();
                    mask |= (!source.is_empty(child) as u8) << child_index.0.value();
                    this_tags.reserve(mask.count_ones() as usize);
                    pending_tags.push((tag, mask));

                    if mask & (1u8 << child_index.0.value()) != 0 {
                        this_tags.push(child);
                    }
                }

                pending_branches.push((index, Branch::new(mask, next_offset)));
                next_offset += mask.count_ones();
            }

            {
                pending_branches.sort_unstable_by_key(|(i, _)| *i);

                for (index, branch) in pending_branches.drain(..) {
                    new_nodes[index] = branch.into();
                }

                pending_branches.clear();
            }

            {
                pending_leaves.sort_unstable_by_key(|(i, _)| *i);

                for (index, leaf) in pending_leaves.drain(..) {
                    new_nodes[index] = leaf.into();
                }

                pending_leaves.clear();
            }

            mem::swap(&mut this_tags, &mut next_tags);
            this_tags.clear();

            if next_tags.is_empty() {
                break;
            }
        }

        debug_assert_eq!(next_offset, nodes.len() as u32);

        Self { nodes }
    }

    #[inline]
    pub(crate) fn for_each_branch<F>(&self, mut f: F)
    where
        F: FnMut(Branch),
    {
        for node in &self.nodes {
            if let Some(branch) = node.as_branch() {
                f(branch);
            }
        }
    }
}

impl<T> Index<OctreeKey> for InlineOctree<T> {
    type Output = InlineNode<T>;

    #[inline]
    fn index(&self, index: OctreeKey) -> &Self::Output {
        self.get(index).unwrap()
    }
}

struct Collector<'a, T, I>
where
    I: BuildOctree<T>,
{
    source: &'a mut I,
    leaves: Vec<T>,
}

impl<T, I> BuildOctree<u31> for Collector<'_, T, I>
where
    I: BuildOctree<T>,
{
    #[inline]
    fn is_empty(&mut self, tag: Quant) -> bool {
        I::is_empty(self.source, tag)
    }

    #[inline]
    fn is_leaf(&mut self, tag: Quant) -> bool {
        I::is_leaf(self.source, tag)
    }

    #[inline]
    fn place_leaf(&mut self, tag: Quant) -> u31 {
        self.leaves.push(I::place_leaf(self.source, tag));
        u31::new(self.leaves.len() as u32 - 1)
    }
}

#[derive(Debug)]
pub(crate) struct Octree<T> {
    octree: InlineOctree<u31>,
    leaves: Vec<T>,
}

impl<T> Octree<T> {
    pub(crate) fn build<S>(source: &mut S) -> Self
    where
        S: BuildOctree<T>,
    {
        let mut collector = Collector {
            source,
            leaves: Vec::new(),
        };

        let octree = InlineOctree::build(&mut collector);

        Self {
            octree,
            leaves: collector.leaves,
        }
    }

    #[inline]
    pub(crate) fn for_each_branch<F>(&self, f: F)
    where
        F: FnMut(Branch),
    {
        self.octree.for_each_branch(f);
    }

    #[inline]
    pub(crate) fn get(&self, key: OctreeKey) -> Option<Node<&T>> {
        self.octree.get(key).map(|inline| {
            let node = inline.as_node();
            node.map_leaf(|i| &self.leaves[i.as_usize()])
        })
    }

    #[inline]
    pub(crate) fn is_leaf(&self, key: OctreeKey) -> bool {
        self.get(key).unwrap().is_leaf()
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(transparent)]
pub(crate) struct OctreeKey(u23);

impl OctreeKey {
    /// The key of the root node
    pub(crate) const ROOT: Self = Self(u23::new(0));
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum Node<T> {
    Branch(Branch),
    Leaf(T),
}

impl<T> Node<T> {
    #[inline]
    pub(crate) fn is_leaf(&self) -> bool {
        matches!(self, Node::Leaf(_))
    }

    #[inline]
    pub(crate) fn as_leaf(&self) -> Option<&T> {
        match self {
            Node::Leaf(leaf) => Some(leaf),
            _ => None,
        }
    }

    #[inline]
    fn map_leaf<B, F>(self, f: F) -> Node<B>
    where
        F: FnOnce(T) -> B,
    {
        match self {
            Node::Branch(branch) => Node::Branch(branch),
            Node::Leaf(leaf) => Node::Leaf(f(leaf)),
        }
    }
}

#[derive_where(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[repr(transparent)]
pub(crate) struct InlineNode<T> {
    inner: RawNode,
    _ty: PhantomData<T>,
}

impl<T: Payload> InlineNode<T> {
    #[inline]
    fn new_branch(branch: Branch) -> Self {
        Self {
            inner: RawNode::new_branch(branch),
            _ty: PhantomData,
        }
    }

    #[inline]
    fn new_leaf(leaf: InlineLeaf<T>) -> Self {
        Self {
            inner: RawNode::new_leaf(leaf),
            _ty: PhantomData,
        }
    }

    #[inline]
    pub(crate) fn is_leaf(self) -> bool {
        matches!(self.inner.kind(), NodeKind::Leaf)
    }

    #[inline]
    pub(crate) fn is_branch(self) -> bool {
        matches!(self.inner.kind(), NodeKind::Branch)
    }

    #[inline]
    pub(crate) fn as_leaf(self) -> Option<InlineLeaf<T>> {
        if self.is_leaf() {
            Some(unsafe { InlineLeaf::from_bits(self.inner.data()) })
        } else {
            None
        }
    }

    #[inline]
    pub(crate) fn as_branch(self) -> Option<Branch> {
        if self.is_branch() {
            Some(unsafe { Branch::from_bits(self.inner.data()) })
        } else {
            None
        }
    }

    #[inline]
    fn as_node(self) -> Node<T> {
        if let Some(branch) = self.as_branch() {
            Node::Branch(branch)
        } else {
            Node::Leaf(self.as_leaf().unwrap().get())
        }
    }
}

impl<T: Payload> From<Branch> for InlineNode<T> {
    #[inline]
    fn from(value: Branch) -> Self {
        Self::new_branch(value)
    }
}

impl<T: Payload> From<InlineLeaf<T>> for InlineNode<T> {
    #[inline]
    fn from(value: InlineLeaf<T>) -> Self {
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
    fn new_leaf<T>(leaf: InlineLeaf<T>) -> Self {
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
pub(crate) struct Branch(RawBranch);

impl Branch {
    #[inline]
    fn new(mask: u8, offset: u32) -> Self {
        Self(RawBranch::new(mask, u23::new(offset)))
    }

    #[inline]
    unsafe fn from_bits(bits: u31) -> Self {
        Self(RawBranch { value: bits })
    }

    #[inline]
    fn mask(self) -> u8 {
        self.0.mask()
    }

    const fn into_u31(self) -> u31 {
        self.0.value
    }

    #[inline]
    pub(crate) fn child(&self, which: ChildIndex) -> Option<OctreeKey> {
        if self.has_child(which) {
            Some(OctreeKey(u23::new(self.child_offset(which))))
        } else {
            None
        }
    }

    #[inline]
    fn has_child(self, which: ChildIndex) -> bool {
        self.mask() & (1u8 << which.0.value()) != 0
    }

    #[inline]
    fn base_offset(self) -> u32 {
        self.0.offset().value()
    }

    #[inline]
    fn child_offset(self, which: ChildIndex) -> u32 {
        let pref = (1u8 << which.0.value()) - 1;
        self.base_offset() + (self.mask() & pref).count_ones()
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct ChildIndex(pub(crate) u3);

impl ChildIndex {
    #[inline]
    pub(crate) const fn new(index: u8) -> Self {
        Self(u3::new(index))
    }

    #[inline]
    pub(crate) fn enumerate() -> impl Iterator<Item = Self> {
        (0..8u8).map(|i| ChildIndex(u3::new(i)))
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

    /// Offset (in nodes) of the first present child in the next level
    offset: u23,
}

#[derive_where(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[repr(transparent)]
pub(crate) struct InlineLeaf<T> {
    payload: u31,
    _ty: PhantomData<T>,
}

impl<T: Payload> InlineLeaf<T> {
    #[inline]
    fn new(data: T) -> Self {
        Self {
            payload: data.into_bits(),
            _ty: PhantomData,
        }
    }

    #[inline]
    unsafe fn from_bits(bits: u31) -> Self {
        Self {
            payload: bits,
            _ty: PhantomData,
        }
    }

    #[inline]
    pub(crate) fn get(self) -> T {
        unsafe { T::from_bits(self.payload) }
    }
}

pub(crate) trait Payload: Copy {
    /// Converts `Self` into a 31-bit unsigned integer.
    fn into_bits(self) -> u31;

    /// Reconstructs `Self` from a 31-bit unsigned integer.
    ///
    /// # Safety
    /// `bits` must have been previously returned by `into_bits`.
    unsafe fn from_bits(bits: u31) -> Self;
}

impl Payload for u31 {
    #[inline]
    fn into_bits(self) -> u31 {
        self
    }

    #[inline]
    unsafe fn from_bits(bits: u31) -> Self {
        bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octree_degenerate() {
        #[derive(Copy, Clone)]
        struct Dummy;

        impl Payload for Dummy {
            fn into_bits(self) -> u31 {
                u31::new(67)
            }

            unsafe fn from_bits(bits: u31) -> Self {
                assert_eq!(bits, u31::new(67));
                Dummy
            }
        }

        struct Degenerate;

        impl BuildOctree<Dummy> for Degenerate {
            fn is_empty(&mut self, _tag: Quant) -> bool {
                true
            }

            fn is_leaf(&mut self, _: Quant) -> bool {
                true
            }

            fn place_leaf(&mut self, tag: Quant) -> Dummy {
                assert_eq!(tag, Quant::root());
                Dummy
            }
        }

        let octree = InlineOctree::build(&mut Degenerate);
        assert_eq!(octree.nodes.as_slice(), &[InlineLeaf::new(Dummy).into()]);
    }

    #[test]
    fn test_octree_uniform() {
        use crate::quant::Quant;
        use std::collections::HashSet;

        struct Uniform;

        impl BuildOctree<Quant> for Uniform {
            fn is_empty(&mut self, _tag: Quant) -> bool {
                false
            }

            fn is_leaf(&mut self, tag: Quant) -> bool {
                tag.level() == 4
            }

            fn place_leaf(&mut self, tag: Quant) -> Quant {
                tag
            }
        }

        let octree = InlineOctree::build(&mut Uniform);
        let mut leaves = HashSet::new();

        for node in octree.nodes {
            if node.is_leaf() {
                leaves.insert(node.as_leaf().unwrap());
            }
        }

        assert_eq!(leaves.len(), 4096);

        for leaf in &leaves {
            assert_eq!(leaf.get().level(), 4);
        }
    }

    #[test]
    fn test_octree_child() {
        use crate::quant::Quant;

        struct Uniform;

        impl BuildOctree<Quant> for Uniform {
            fn is_empty(&mut self, _tag: Quant) -> bool {
                false
            }

            fn is_leaf(&mut self, tag: Quant) -> bool {
                tag.level() == 1
            }

            fn place_leaf(&mut self, tag: Quant) -> Quant {
                tag
            }
        }

        let octree = InlineOctree::build(&mut Uniform);
        let root = octree.nodes[0].as_branch().unwrap();

        for which in ChildIndex::enumerate() {
            let child = octree[root.child(which).unwrap()];

            assert_eq!(
                child.as_leaf().unwrap().get(),
                Quant::root().child(which).unwrap()
            );
        }
    }
}
