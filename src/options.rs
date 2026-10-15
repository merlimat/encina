use crate::error::{Error, Result};
use crate::node;

/// Configuration for a store.
#[derive(Clone, Debug)]
pub struct Options {
    pub(crate) page_size: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options { page_size: 4096 }
    }
}

impl Options {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the page size: a power of two from 4 KiB to 64 KiB, 4 KiB by default.
    ///
    /// The page size bounds entries: a put's key plus value may not exceed
    /// [`max_entry_size`](Self::max_entry_size), a bit under a quarter of a page.
    pub fn page_size(mut self, bytes: usize) -> Self {
        self.page_size = bytes;
        self
    }

    /// The largest key plus value a store with these options accepts. Larger
    /// entries make [`Db::write`](crate::Db::write) reject the whole batch with
    /// [`Error::EntryTooLarge`].
    pub fn max_entry_size(&self) -> usize {
        node::max_entry(self.page_size)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        // Tests use tiny pages to grow deep trees from few entries.
        let min = if cfg!(test) { 256 } else { 4096 };
        if !self.page_size.is_power_of_two() || !(min..=65536).contains(&self.page_size) {
            return Err(Error::InvalidOptions(
                "page size must be a power of two from 4 KiB to 64 KiB",
            ));
        }
        Ok(())
    }
}
