// Copyright 2026 Matteo Merli <matteo.merli@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::error::{Error, Result};
use crate::node;

/// The smallest cache: a write needs a few leaves in memory at once.
const MIN_CACHE_FRAMES: usize = 8;

/// The smallest page size. Tests use tiny pages to grow deep trees from few entries.
pub(crate) const MIN_PAGE_SIZE: usize = if cfg!(test) { 256 } else { 4096 };

/// Configuration for a store.
#[derive(Clone, Debug)]
pub struct Options {
    pub(crate) page_size: usize,
    pub(crate) cache_size: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            page_size: 4096,
            cache_size: 64 << 20,
        }
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

    /// Sets the memory budget for pages, 64 MiB by default.
    ///
    /// Internal nodes always stay in memory; leaves are cached within what's left,
    /// and evicted roughly least recently used first. A write needs the leaves it changes
    /// and the ones it builds in memory at once, so a large batch can exceed the
    /// budget until it commits.
    pub fn cache_size(mut self, bytes: usize) -> Self {
        self.cache_size = bytes;
        self
    }

    /// The largest key plus value a store with these options accepts. Larger
    /// entries make [`Db::write`](crate::Db::write) reject the whole batch with
    /// [`Error::EntryTooLarge`].
    pub fn max_entry_size(&self) -> usize {
        node::max_entry(self.page_size)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if !self.page_size.is_power_of_two() || !(MIN_PAGE_SIZE..=65536).contains(&self.page_size) {
            return Err(Error::InvalidOptions(
                "page size must be a power of two from 4 KiB to 64 KiB",
            ));
        }
        Ok(())
    }

    /// The number of page frames the cache budget allows.
    pub(crate) fn cache_frames(&self) -> usize {
        (self.cache_size / self.page_size).max(MIN_CACHE_FRAMES)
    }
}
