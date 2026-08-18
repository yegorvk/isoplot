use std::collections::HashMap;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub(super) struct Symbol(u32);

#[derive(Debug, Default)]
pub(super) struct Interner {
    map: HashMap<String, Symbol>,
    strings: Vec<String>,
}

impl Interner {
    pub(super) fn get_or_insert(&mut self, value: &str) -> Symbol {
        if let Some(&symbol) = self.map.get(value) {
            return symbol;
        }

        let len = self.strings.len();
        assert!(len < u32::MAX as usize);

        let symbol = Symbol(len as u32);
        self.strings.push(value.to_owned());
        self.map.insert(value.to_owned(), symbol);
        symbol
    }

    pub(super) fn resolve(&self, symbol: Symbol) -> Option<&str> {
        self.strings.get(symbol.0 as usize).map(String::as_str)
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
