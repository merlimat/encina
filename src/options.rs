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

use opentelemetry::KeyValue;
use opentelemetry::metrics::MeterProvider;

use crate::error::{Error, Result};
use crate::metrics::SCOPE;
use crate::node;

/// The smallest cache: a write needs a few leaves in memory at once.
const MIN_CACHE_FRAMES: usize = 8;

/// The smallest page size. Tests use tiny pages to grow deep trees from few entries.
pub(crate) const MIN_PAGE_SIZE: usize = if cfg!(test) { 256 } else { 4096 };

/// The page size of a new store whose options don't set one.
pub(crate) const DEFAULT_PAGE_SIZE: usize = 4096;

/// Configuration for a store.
#[derive(Clone, Debug)]
pub struct Options {
    pub(crate) page_size: Option<usize>,
    pub(crate) min_run_pages: usize,
    pub(crate) cache_size: usize,
    pub(crate) meter: Option<nanometer::Meter>,
    pub(crate) metric_attributes: Vec<KeyValue>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            page_size: None,
            min_run_pages: 16,
            cache_size: 64 << 20,
            meter: None,
            metric_attributes: Vec::new(),
        }
    }
}

impl Options {
    /// The default options: no page size, runs of 16 pages, a 64 MiB cache, and no
    /// metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the page size: a power of two from 4 KiB to 64 KiB.
    ///
    /// A store keeps the page size it was created with. Without one, a new store
    /// gets 4 KiB pages, and an existing one opens with its own. With one,
    /// [`Db::open`](crate::Db::open) fails with [`Error::PageSizeMismatch`] on a
    /// store that has another. A [`ReadOnlyDb`](crate::ReadOnlyDb) writes no
    /// pages, so it opens any store with its own.
    ///
    /// The page size bounds keys: a put's key may not exceed
    /// [`max_key_size`](Self::max_key_size), a bit under a quarter of a page. An
    /// entry whose key plus value exceeds that too gets pages of its own, rounded up
    /// to whole pages, so a larger page size keeps more values in ordinary leaves.
    pub fn page_size(mut self, bytes: usize) -> Self {
        self.page_size = Some(bytes);
        self
    }

    /// Sets the shortest stretch of free pages that a write reuses, short of the
    /// end of the file: 16 by default.
    ///
    /// A write takes its pages in order from stretches of free pages, and writes
    /// each stretch with one request. It skips shorter ones and grows the file
    /// instead, until the pages around them are freed too. A smaller value keeps
    /// the file smaller and sends more write requests: 1 reuses every free page.
    /// Nothing about it is stored, so it can change from one open to the next.
    pub fn min_run_pages(mut self, pages: usize) -> Self {
        self.min_run_pages = pages;
        self
    }

    /// Sets the memory budget for pages, 64 MiB by default.
    ///
    /// Internal nodes always stay in memory; leaves are cached within what's left,
    /// and evicted roughly least recently used first. A write needs the leaves it changes
    /// and the ones it builds in memory at once, so a large batch can exceed the
    /// budget until it commits. Entries too large for a leaf take at most a
    /// quarter of the budget, and one larger than that isn't cached.
    pub fn cache_size(mut self, bytes: usize) -> Self {
        self.cache_size = bytes;
        self
    }

    /// Exports the store's metrics through `provider`, under the instrumentation
    /// scope `encina`. Without a provider, the store records none.
    ///
    /// [METRICS.md](https://github.com/merlimat/encina/blob/main/METRICS.md) lists
    /// the metrics. Each series has the attribute `encina.store`, the path the store
    /// was opened at, and those of [`metric_attributes`](Self::metric_attributes).
    /// Histograms export only through nanometer's `Producer`, which the app
    /// registers with its metric reader: see [nanometer](https://docs.rs/nanometer).
    ///
    /// The first store with metrics in a process calibrates the clock that times
    /// operations, which blocks its open for 10 to 200 ms.
    pub fn meter_provider<P: MeterProvider + ?Sized>(mut self, provider: &P) -> Self {
        self.meter = Some(nanometer::Meter::new(provider, SCOPE));
        self
    }

    /// Adds `attributes` to every series of the store's metrics, after
    /// `encina.store`: one of them with that key replaces the store's path.
    pub fn metric_attributes(mut self, attributes: &[KeyValue]) -> Self {
        self.metric_attributes.extend_from_slice(attributes);
        self
    }

    /// The longest key a store with these options accepts. Longer keys make
    /// [`Db::write`](crate::Db::write) reject the whole batch with
    /// [`Error::KeyTooLarge`]. Without a page size, this is the limit of a new
    /// store's 4 KiB pages: an existing store's comes from its own page size.
    pub fn max_key_size(&self) -> usize {
        node::max_entry(self.page_size.unwrap_or(DEFAULT_PAGE_SIZE))
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if let Some(size) = self.page_size
            && (!size.is_power_of_two() || !(MIN_PAGE_SIZE..=65536).contains(&size))
        {
            return Err(Error::InvalidOptions(
                "page size must be a power of two from 4 KiB to 64 KiB",
            ));
        }
        Ok(())
    }

    /// The number of page frames the cache budget allows.
    pub(crate) fn cache_frames(&self) -> usize {
        (self.cache_size / self.page_size.unwrap_or(DEFAULT_PAGE_SIZE)).max(MIN_CACHE_FRAMES)
    }
}
