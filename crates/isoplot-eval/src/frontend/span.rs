#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub(super) struct BytePos(pub(super) u32);

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(super) struct Span {
    start: BytePos,
    end: BytePos,
}

impl Span {
    pub(super) fn new(start: BytePos, end: BytePos) -> Self {
        assert!(start.0 <= end.0);
        Self { start, end }
    }

    pub(super) fn empty(position: BytePos) -> Self {
        Self::new(position, position)
    }

    pub(super) fn start(self) -> BytePos {
        self.start
    }

    pub(super) fn end(self) -> BytePos {
        self.end
    }

    pub(super) fn chain(self, other: Span) -> Self {
        Self::new(self.start, other.end)
    }
}
