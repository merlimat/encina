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

//! Page allocation.
//!
//! A bitmap tracks which pages are in use. Nothing about it is stored: opening a
//! store marks the pages its tree reaches. A page that a write replaces stays in
//! use until no reader of an older version, which might still read it, is left.

use std::collections::VecDeque;

pub(crate) struct Space {
    used: Vec<u64>,
    /// No page at or above this one is in use, or in the file's tree.
    high_water: u64,
    /// No page below this one is free.
    lowest_free: u64,
    /// Pages that published versions replaced, with the generation that replaced
    /// them, oldest first.
    dropped: VecDeque<(u64, u64)>,
    /// The generations of readers that may still read older versions.
    readers: Vec<u64>,
}

impl Space {
    /// Space for a file whose pages up to `high_water` may be in use. Mark the
    /// ones that are with [`mark_used`](Self::mark_used).
    pub(crate) fn new(high_water: u64) -> Space {
        Space {
            used: vec![0; high_water.div_ceil(64) as usize],
            high_water,
            lowest_free: 0,
            dropped: VecDeque::new(),
            readers: Vec::new(),
        }
    }

    pub(crate) fn high_water(&self) -> u64 {
        self.high_water
    }

    pub(crate) fn is_used(&self, page: u64) -> bool {
        page < self.high_water && self.used[(page / 64) as usize] & (1 << (page % 64)) != 0
    }

    pub(crate) fn mark_used(&mut self, page: u64) {
        debug_assert!(
            page < self.high_water && !self.is_used(page),
            "page {page} used twice"
        );
        self.used[(page / 64) as usize] |= 1 << (page % 64);
    }

    /// The lowest free page, which becomes used.
    pub(crate) fn alloc(&mut self) -> u64 {
        let mut word = (self.lowest_free / 64) as usize;
        while word < self.used.len() && self.used[word] == u64::MAX {
            word += 1;
        }
        let page = if word < self.used.len() {
            word as u64 * 64 + u64::from(self.used[word].trailing_ones())
        } else {
            self.high_water
        };
        if page >= self.high_water {
            // Extend the file by one page.
            self.high_water = page + 1;
            self.used.resize(self.high_water.div_ceil(64) as usize, 0);
        }
        self.mark_used(page);
        self.lowest_free = page + 1;
        page
    }

    /// Frees a page now. Only for pages no reader can reach and no durable
    /// version uses: those a write allocated but didn't commit.
    pub(crate) fn free(&mut self, page: u64) {
        debug_assert!(self.is_used(page), "page {page} freed twice");
        self.used[(page / 64) as usize] &= !(1 << (page % 64));
        self.lowest_free = self.lowest_free.min(page);
    }

    /// Records that the version published as `generation` no longer uses `page`.
    pub(crate) fn drop_page(&mut self, generation: u64, page: u64) {
        self.dropped.push_back((generation, page));
    }

    pub(crate) fn add_reader(&mut self, generation: u64) {
        self.readers.push(generation);
    }

    pub(crate) fn remove_reader(&mut self, generation: u64) {
        let i = self
            .readers
            .iter()
            .position(|&g| g == generation)
            .expect("unknown reader");
        self.readers.swap_remove(i);
    }

    /// Frees the dropped pages that no reader can still need, passing each to
    /// `freed`. A page dropped by generation `g` is needed by readers of earlier
    /// generations.
    pub(crate) fn reclaim(&mut self, mut freed: impl FnMut(u64)) {
        let oldest = self.readers.iter().copied().min().unwrap_or(u64::MAX);
        while let Some(&(generation, page)) = self.dropped.front() {
            if generation > oldest {
                break;
            }
            self.dropped.pop_front();
            self.free(page);
            freed(page);
        }
    }

    /// Pages dropped but not reclaimed yet.
    #[cfg(test)]
    pub(crate) fn dropped(&self) -> impl Iterator<Item = u64> + '_ {
        self.dropped.iter().map(|&(_, page)| page)
    }

    /// Pages in use.
    #[cfg(test)]
    pub(crate) fn used_count(&self) -> usize {
        self.used.iter().map(|w| w.count_ones() as usize).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_the_lowest_free_page_and_extends_the_file() {
        let mut space = Space::new(3);
        space.mark_used(0);
        space.mark_used(2);
        assert_eq!(space.alloc(), 1);
        assert_eq!(space.alloc(), 3);
        assert_eq!(space.high_water(), 4);
        space.free(1);
        assert_eq!(space.alloc(), 1);
        for page in 4..200 {
            assert_eq!(space.alloc(), page);
        }
    }

    #[test]
    fn dropped_pages_wait_for_older_readers() {
        let mut space = Space::new(10);
        for page in 0..10 {
            space.mark_used(page);
        }
        space.add_reader(4);
        space.drop_page(5, 7);
        space.drop_page(6, 8);
        let mut freed = Vec::new();
        space.reclaim(|page| freed.push(page));
        assert!(freed.is_empty());
        space.remove_reader(4);
        space.add_reader(5);
        space.reclaim(|page| freed.push(page));
        assert_eq!(freed, [7]);
        space.remove_reader(5);
        space.reclaim(|page| freed.push(page));
        assert_eq!(freed, [7, 8]);
        assert_eq!(space.alloc(), 7);
    }
}
