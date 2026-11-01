use bilge::prelude::*;
use derive_where::derive_where;
use fearless_simd::{Level, Simd, dispatch};
use fearless_simd_macros::simd;
use std::{hint::select_unpredictable, iter, marker::PhantomData, mem, ops::Index};

use crate::{
    quant::Quant,
    simd::{BMask8, VecExt, WriteBuf, WriteBufExt},
    utils::{
        filter_branchless_into_vec, map_vec_in_place, transmute_empty_vec, traverse_ping_pong,
    },
};

/// Maximum number of levels in an octree
///
/// The root node is always taken into account,
/// so that this values is always at least 1.
pub(crate) const MAX_LEVELS: u8 = 10;

/// The maximum octree level
///
/// The levels are numbered starting from zero.
pub(crate) const MAX_LEVEL: u8 = MAX_LEVELS - 1;

pub(crate) trait OctreeSource<T: Copy> {
    /// Returns `true` if the specified node is empty.
    fn is_empty(&mut self, tag: Quant) -> bool;

    #[inline]
    fn is_child_empty_mask(&mut self, tag: Quant) -> BMask8 {
        BMask8::from_fn(|i| self.is_empty(tag.child(ChildIndex::new(i)).unwrap()))
    }

    /// Returns `true` if the specified node is a leaf.
    fn is_leaf(&mut self, tag: Quant) -> bool;

    /// Returns the leaf payload associated with the specified tag.
    fn place_leaf(&mut self, tag: Quant) -> T;

    fn is_child_empty_mask_s<'out, S: Simd, B>(&mut self, simd: S, parent: &[Quant], out: B)
    where
        B: WriteBuf<'out, Item = BMask8>,
    {
        _ = simd;
        out.write_all_with(|out| {
            out.write_from_iter(parent.iter().map(|tag| {
                BMask8::from_fn(|i| {
                    let child_index = ChildIndex::new(i as u8);
                    self.is_empty(tag.child(child_index).unwrap())
                })
            }));
        });
    }

    fn is_leaf_s<'out, S: Simd, B>(&mut self, simd: S, tag: &[[Quant; 8]], out: B)
    where
        B: WriteBuf<'out, Item = BMask8>,
    {
        _ = simd;
        out.write_all_with(|out| {
            out.write_from_iter(
                tag.iter()
                    .map(|tag| BMask8::from_fn(|i| self.is_leaf(tag[i as usize]))),
            );
        });
    }

    fn place_leaves_s<'out, S: Simd, B>(&mut self, simd: S, tags: &[Quant], out: B)
    where
        T: 'out,
        B: WriteBuf<'out, Item = T>,
    {
        _ = simd;
        out.write_all_with(|out| {
            out.write_from_iter(tags.iter().map(|tag| self.place_leaf(*tag)));
        });
    }

    #[inline]
    fn should_vectorize(&mut self) -> bool {
        true
    }
}

#[derive(Debug)]
struct InlineOctree<T> {
    nodes: Vec<InlineNode<T>>,
}

impl<T> InlineOctree<T> {
    #[inline]
    fn get(&self, key: OctreeKey) -> Option<&InlineNode<T>> {
        self.nodes.get(key.0.as_usize())
    }
}

impl<T: Payload> InlineOctree<T> {
    pub(crate) fn build<I>(source: &mut I) -> Self
    where
        I: ?Sized + OctreeSource<T>,
    {
        let level = Level::new();

        if !source.should_vectorize() || level.is_fallback() {
            Self::build_scalar(source)
        } else {
            dispatch!(level, simd => Self::build_s(simd, source))
        }
    }

    fn build_scalar<I>(source: &mut I) -> Self
    where
        I: ?Sized + OctreeSource<T>,
    {
        let mut nodes: Vec<InlineNode<Quant>> = vec![];

        let mut next_branch_offset = 1u32;
        let mut total_leaves = 0usize;

        let mut temp_branches: Vec<Branch> = vec![];

        traverse_ping_pong(
            vec![Quant::root()],
            Some(MAX_LEVEL as usize),
            |_, old_tags, new_tags| {
                debug_assert!(temp_branches.is_empty());

                let old_node_count = nodes.len();
                let mut num_leaves = 0usize;

                nodes.extend(old_tags.iter().copied().enumerate().map(|(i, tag)| {
                    let is_leaf = source.is_leaf(tag);
                    num_leaves += is_leaf as usize;
                    InlineNode::new(is_leaf, Branch::new(0, i as u32), InlineLeaf::new(tag))
                }));

                let new_nodes = &mut nodes[old_node_count..];

                {
                    let mut temp = transmute_empty_vec(mem::take(&mut temp_branches));
                    filter_branchless_into_vec(new_nodes, &mut temp, |node| node.is_branch());
                    temp_branches = map_vec_in_place(temp, |node| node.as_branch().unwrap());
                }

                for branch in temp_branches.drain(..) {
                    let index = branch.base_offset() as usize;
                    let mask = !source.is_child_empty_mask(old_tags[index]);

                    new_nodes[index] = Branch::new(mask.0, next_branch_offset).into();
                    next_branch_offset += mask.count_set() as u32;

                    for i in 0..8u8 {
                        if mask.contains(i) {
                            new_tags.push(old_tags[index].child(ChildIndex::new(i)).unwrap());
                        }
                    }
                }

                total_leaves += num_leaves;
            },
        );

        _ = temp_branches;
        let mut leaf_tags: Vec<Quant> = Vec::with_capacity(total_leaves);

        {
            let mut temp = transmute_empty_vec(mem::take(&mut leaf_tags));
            filter_branchless_into_vec(&nodes, &mut temp, |node| node.is_leaf());
            leaf_tags = map_vec_in_place(temp, |node| node.as_leaf().unwrap().get())
        }

        let leaves = map_vec_in_place(leaf_tags, |tag| source.place_leaf(tag));

        let nodes: Vec<InlineNode<T>> = if total_leaves != 0 {
            let mut i = 0usize;

            let raw_nodes: Vec<RawNode> = nodes
                .into_iter()
                .map(|node| {
                    let is_leaf = node.is_leaf();
                    if i >= leaves.len() {
                        return node.inner;
                    }
                    let raw_node = select_unpredictable(
                        is_leaf,
                        RawNode::new_leaf(InlineLeaf::new(leaves[i])),
                        node.inner,
                    );
                    i += is_leaf as usize;
                    raw_node
                })
                .collect();

            map_vec_in_place(raw_nodes, |node| unsafe { InlineNode::from_raw(node) })
        } else {
            map_vec_in_place(nodes, |node| node.as_branch().unwrap().into())
        };

        Self { nodes }
    }

    #[allow(dead_code)]
    #[simd]
    fn build_s<S: Simd, I>(simd: S, source: &mut I) -> Self
    where
        I: ?Sized + OctreeSource<T>,
    {
        let mut nodes: Vec<InlineNode<Quant>> = vec![];

        let mut temp_branches: Vec<Branch> = vec![];
        let mut temp_masks: Vec<BMask8> = vec![];
        let mut temp_quants: Vec<Quant> = vec![];

        let mut next_branch_offset = 1u32;
        let mut total_leaves = 0usize;

        traverse_ping_pong(
            vec![Quant::root()],
            Some(MAX_LEVEL as usize),
            |_, old_tags, new_tags| {
                debug_assert!(
                    temp_branches.is_empty() && temp_masks.is_empty() && temp_quants.is_empty()
                );

                let old_node_count = nodes.len();

                nodes.reserve(old_tags.len());
                let num_leaves = {
                    let (chunks, tail) = old_tags.as_chunks::<8>();

                    source.is_leaf_s(simd, chunks, temp_masks.clear_write(chunks.len()));

                    let mut num_leaves = 0usize;
                    for i in 0..temp_masks.len() {
                        for j in 0..8 {
                            let is_leaf = temp_masks[i].contains(j as u8);
                            num_leaves += is_leaf as usize;
                            nodes.push(InlineNode::new(
                                is_leaf,
                                Branch::new(0, (i * 8 + j) as u32),
                                InlineLeaf::new(chunks[i][j]),
                            ));
                        }
                    }

                    temp_masks.clear();

                    for (i, tag) in tail.iter().copied().enumerate() {
                        let is_leaf = source.is_leaf(tag);
                        num_leaves += is_leaf as usize;
                        nodes.push(InlineNode::new(
                            is_leaf,
                            Branch::new(0, (8 * chunks.len() + i) as u32),
                            InlineLeaf::new(tag),
                        ));
                    }

                    num_leaves
                };

                let new_nodes = &mut nodes[old_node_count..];
                let num_branches = new_nodes.len() - num_leaves;

                {
                    let mut temp = transmute_empty_vec(mem::take(&mut temp_branches));
                    filter_branchless_into_vec(new_nodes, &mut temp, |node| node.is_branch());
                    temp_branches = map_vec_in_place(temp, |node| node.as_branch().unwrap());
                }

                temp_quants.clear();
                temp_quants.extend(
                    temp_branches
                        .iter()
                        .map(|branch| old_tags[branch.base_offset() as usize]),
                );

                source.is_child_empty_mask_s(
                    simd,
                    &temp_quants,
                    temp_masks.clear_write(num_branches),
                );

                temp_quants.clear();

                for (branch, mask) in iter::zip(temp_branches.drain(..), temp_masks.drain(..)) {
                    let index = branch.base_offset() as usize;
                    new_nodes[index] = Branch::new(!mask.0, next_branch_offset).into();
                    next_branch_offset += mask.0.count_zeros();

                    for i in 0..8u8 {
                        if mask.0 & (1u8 << i) == 0 {
                            new_tags.push(old_tags[index].child(ChildIndex::new(i)).unwrap());
                        }
                    }
                }

                total_leaves += num_leaves;
            },
        );

        _ = (temp_branches, temp_masks);
        let mut leaf_tags: Vec<Quant> = Vec::with_capacity(total_leaves);

        {
            let mut temp = transmute_empty_vec(mem::take(&mut leaf_tags));
            filter_branchless_into_vec(&nodes, &mut temp, |node| node.is_leaf());
            leaf_tags = map_vec_in_place(temp, |node| node.as_leaf().unwrap().get())
        }

        let mut leaves = Vec::with_capacity(total_leaves);
        source.place_leaves_s(simd, &leaf_tags, leaves.extend_write(total_leaves));
        _ = leaf_tags;

        let nodes: Vec<InlineNode<T>> = if total_leaves != 0 {
            let mut i = 0usize;

            let raw_nodes: Vec<RawNode> = nodes
                .into_iter()
                .map(|node| {
                    let is_leaf = node.is_leaf();
                    if i >= leaves.len() {
                        return node.inner;
                    }
                    let raw_node = select_unpredictable(
                        is_leaf,
                        RawNode::new_leaf(InlineLeaf::new(leaves[i])),
                        node.inner,
                    );
                    i += is_leaf as usize;
                    raw_node
                })
                .collect();

            map_vec_in_place(raw_nodes, |node| unsafe { InlineNode::from_raw(node) })
        } else {
            map_vec_in_place(nodes, |node| node.as_branch().unwrap().into())
        };

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

struct Collector<'a, T: Copy, I>
where
    I: OctreeSource<T>,
{
    source: &'a mut I,
    leaves: Vec<T>,
}

impl<T: Copy, I> OctreeSource<u31> for Collector<'_, T, I>
where
    I: OctreeSource<T>,
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

    #[inline(always)]
    fn is_child_empty_mask_s<'out, S: Simd, B>(&mut self, simd: S, tag: &[Quant], out: B)
    where
        B: WriteBuf<'out, Item = BMask8>,
    {
        I::is_child_empty_mask_s(self.source, simd, tag, out);
    }

    #[inline(always)]
    fn is_leaf_s<'out, S: Simd, B>(&mut self, simd: S, tag: &[[Quant; 8]], out: B)
    where
        B: WriteBuf<'out, Item = BMask8>,
    {
        I::is_leaf_s(self.source, simd, tag, out);
    }

    #[inline(always)]
    fn place_leaves_s<'out, S: Simd, B>(&mut self, simd: S, tags: &[Quant], mut out: B)
    where
        B: WriteBuf<'out, Item = u31>,
    {
        let old_len = self.leaves.len();

        self.source
            .place_leaves_s(simd, tags, self.leaves.extend_write(tags.len()));

        out.write_all_with(|out| {
            out.write_from_iter((old_len..self.leaves.len()).map(|i| u31::new(i as u32)));
        });
    }
}

#[derive(Debug)]
pub(crate) struct Octree<T> {
    octree: InlineOctree<u31>,
    leaves: Vec<T>,
}

impl<T: Copy> Octree<T> {
    pub(crate) fn build<S>(source: &mut S) -> Self
    where
        S: OctreeSource<T>,
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
        self.get(key).map(|node| node.is_leaf()).unwrap_or(false)
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
    fn new(is_leaf: bool, branch: Branch, leaf: InlineLeaf<T>) -> Self {
        Self {
            inner: RawNode::new_dyn(is_leaf, branch, leaf),
            _ty: PhantomData,
        }
    }

    #[inline]
    unsafe fn from_raw(raw_node: RawNode) -> Self {
        Self {
            inner: raw_node,
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

    #[inline]
    fn new_dyn<T>(is_leaf: bool, branch: Branch, leaf: InlineLeaf<T>) -> Self {
        // Dear compiler, please eliminate this branch.
        let kind = if is_leaf {
            NodeKind::Leaf
        } else {
            NodeKind::Branch
        };
        let payload = select_unpredictable(is_leaf, leaf.payload, branch.into_u31());
        Self::new(kind, payload)
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
    pub(crate) const ALL: [ChildIndex; 8] = {
        let mut a = [ChildIndex(u3::ZERO); 8];
        let mut i = 0usize;
        while i < 8 {
            a[i] = ChildIndex(u3::new(i as u8));
            i += 1;
        }
        a
    };

    #[inline]
    pub(crate) const fn new(index: u8) -> Self {
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

    /// Offset (in nodes) of the first present child in the octree
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
        #[allow(dead_code)]
        #[derive(Copy, Clone)]
        struct Dummy(u32);

        impl Payload for Dummy {
            fn into_bits(self) -> u31 {
                u31::new(67)
            }

            unsafe fn from_bits(bits: u31) -> Self {
                assert_eq!(bits, u31::new(67));
                Dummy(0)
            }
        }

        struct Degenerate;

        impl OctreeSource<Dummy> for Degenerate {
            fn is_empty(&mut self, _tag: Quant) -> bool {
                true
            }

            fn is_leaf(&mut self, _: Quant) -> bool {
                true
            }

            fn place_leaf(&mut self, tag: Quant) -> Dummy {
                assert_eq!(tag, Quant::root());
                Dummy(0)
            }
        }

        let octree = InlineOctree::build_scalar(&mut Degenerate);
        assert_eq!(octree.nodes.as_slice(), &[InlineLeaf::new(Dummy(0)).into()]);
    }

    #[test]
    fn test_octree_uniform() {
        use crate::quant::Quant;
        use std::collections::HashSet;

        struct Uniform;

        impl OctreeSource<Quant> for Uniform {
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

        let octree = InlineOctree::build_scalar(&mut Uniform);
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

        impl OctreeSource<Quant> for Uniform {
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

        let octree = InlineOctree::build_scalar(&mut Uniform);
        let root = octree.nodes[0].as_branch().unwrap();

        for which in ChildIndex::ALL {
            let child = octree[root.child(which).unwrap()];

            assert_eq!(
                child.as_leaf().unwrap().get(),
                Quant::root().child(which).unwrap()
            );
        }
    }

    #[test]
    fn test_octree_degenerate_simd() {
        #[allow(dead_code)]
        #[derive(Copy, Clone)]
        struct Dummy(u32);

        impl Payload for Dummy {
            fn into_bits(self) -> u31 {
                u31::new(67)
            }

            unsafe fn from_bits(bits: u31) -> Self {
                assert_eq!(bits, u31::new(67));
                Dummy(0)
            }
        }

        struct Degenerate;

        impl OctreeSource<Dummy> for Degenerate {
            fn is_empty(&mut self, _tag: Quant) -> bool {
                true
            }

            fn is_leaf(&mut self, _: Quant) -> bool {
                true
            }

            fn place_leaf(&mut self, tag: Quant) -> Dummy {
                assert_eq!(tag, Quant::root());
                Dummy(0)
            }
        }

        let octree = dispatch!(Level::new(), simd => InlineOctree::build_s(simd, &mut Degenerate));
        assert_eq!(octree.nodes.as_slice(), &[InlineLeaf::new(Dummy(0)).into()]);
    }

    #[test]
    fn test_octree_uniform_simd() {
        use crate::quant::Quant;
        use std::collections::HashSet;

        struct Uniform;

        impl OctreeSource<Quant> for Uniform {
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

        let octree = dispatch!(Level::new(), simd => InlineOctree::build_s(simd, &mut Uniform));
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
    fn test_octree_child_simd() {
        use crate::quant::Quant;

        struct Uniform;

        impl OctreeSource<Quant> for Uniform {
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

        let octree = dispatch!(Level::new(), simd => InlineOctree::build_s(simd, &mut Uniform));
        let root = octree.nodes[0].as_branch().unwrap();

        for which in ChildIndex::ALL {
            let child = octree[root.child(which).unwrap()];

            assert_eq!(
                child.as_leaf().unwrap().get(),
                Quant::root().child(which).unwrap()
            );
        }
    }
}
