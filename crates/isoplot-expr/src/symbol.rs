use bumpalo::Bump;
use std::collections::HashMap;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub(crate) struct Symbol(u32);

#[derive(Debug, Default)]
pub(crate) struct Interner {
    bump: Bump,
    map: HashMap<&'static str, Symbol>,
    strings: Vec<&'static str>,
}

impl Interner {
    pub(crate) fn get_or_insert(&mut self, value: &str) -> Symbol {
        if let Some(&symbol) = self.map.get(value) {
            return symbol;
        }

        let len = self.strings.len();
        assert!(len < u32::MAX as usize);

        // SAFETY: the string is owned by `self.bump` and cannot outlast it.
        // Although it is stored with a `'static` lifetimes, `resolve`
        // constrains the returned reference lifetime to `&self`.
        let value = unsafe { &*(self.bump.alloc_str(value) as *const str) };

        let symbol = Symbol(len as u32);
        self.strings.push(value);
        self.map.insert(value, symbol);
        symbol
    }

    pub(crate) fn resolve(&self, symbol: Symbol) -> Option<&str> {
        // Invariant: constrains the lifetime to `&self`.
        self.strings.get(symbol.0 as usize).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_resolve() {
        let mut interner = Interner::default();

        let x = interner.get_or_insert("x");
        let y = interner.get_or_insert("y");

        assert_ne!(x, y);
        assert_eq!(interner.get_or_insert("x"), x);
        assert_eq!(interner.resolve(x).unwrap(), "x");
        assert_eq!(interner.resolve(y).unwrap(), "y");
    }
}
