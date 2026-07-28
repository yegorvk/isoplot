#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub(crate) struct BytePos(pub(crate) u32);

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) struct Span {
    start: BytePos,
    end: BytePos,
}

impl Span {
    pub(crate) fn new(start: BytePos, end: BytePos) -> Self {
        assert!(start.0 <= end.0);
        Self { start, end }
    }

    pub(crate) fn empty(position: BytePos) -> Self {
        Self::new(position, position)
    }

    pub(crate) fn start(self) -> BytePos {
        self.start
    }

    pub(crate) fn end(self) -> BytePos {
        self.end
    }

    pub(crate) fn chain(self, other: Span) -> Self {
        Self::new(self.start, other.end)
    }
}
