#[derive(Clone, Debug)]
pub struct Diagnostic {
    message: String,
    location: Span,
}

impl Diagnostic {
    pub(crate) fn new(message: impl Into<String>, location: Span) -> Self {
        Self {
            message: message.into(),
            location,
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn location(&self) -> Span {
        self.location
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Span {
    pub start: BytePos,
    pub end: BytePos,
}

impl Span {
    pub(crate) fn new(start: BytePos, end: BytePos) -> Self {
        Self { start, end }
    }

    pub(crate) fn empty(position: BytePos) -> Self {
        Self::new(position, position)
    }

    pub(crate) fn chain(self, other: Span) -> Self {
        Self::new(self.start, other.end)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct BytePos(pub(crate) u32);

impl BytePos {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}
