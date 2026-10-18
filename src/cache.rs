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

//! The leaf cache: which frame holds each cached leaf, and CLOCK eviction.
//!
//! The budget counts pages. A value leaf takes a large frame, of as many pages as
//! it spans, and value leaves take at most a quarter of the budget in all: a larger
//! one isn't cached.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::pool::{self, FrameId, Holds, Pool};

pub(crate) struct Cache {
    /// Each cached leaf's frame, by page id.
    map: HashMap<u64, FrameId, BuildHasherDefault<PageHasher>>,
    hand: usize,
    /// Pages the pool may hold before leaves get evicted.
    budget: usize,
}

impl Cache {
    pub(crate) fn new(budget: usize) -> Cache {
        let map = HashMap::with_capacity_and_hasher(budget, Default::default());
        Cache {
            map,
            hand: 0,
            budget,
        }
    }

    /// The frame holding a leaf page, marked as recently used.
    pub(crate) fn get(&self, pool: &mut Pool, page: u64) -> Option<FrameId> {
        let frame = *self.map.get(&page)?;
        pool.frame_mut(frame).clock = true;
        Some(frame)
    }

    pub(crate) fn insert(&mut self, page: u64, frame: FrameId) {
        let old = self.map.insert(page, frame);
        debug_assert!(old.is_none(), "page {page} cached twice");
    }

    pub(crate) fn remove(&mut self, page: u64) -> Option<FrameId> {
        self.map.remove(&page)
    }

    /// Pages the pool may hold before leaves get evicted.
    pub(crate) fn budget(&self) -> usize {
        self.budget
    }

    /// Whether a value leaf of `pages` pages is cached.
    pub(crate) fn caches(&self, pages: u64) -> bool {
        pages <= (self.budget / 4) as u64
    }

    /// A frame holding one reference, evicting a cold leaf first when the pool is
    /// at its budget. If every leaf is in use, the pool grows past the budget.
    pub(crate) fn alloc(&mut self, pool: &mut Pool, holds: Holds, page: u64) -> FrameId {
        self.try_alloc(pool, holds, page)
            .unwrap_or_else(|| pool.alloc(holds, page))
    }

    /// Like [`alloc`](Self::alloc), but returns `None` rather than growing the
    /// pool past its budget.
    pub(crate) fn try_alloc(
        &mut self,
        pool: &mut Pool,
        holds: Holds,
        page: u64,
    ) -> Option<FrameId> {
        if pool.in_use() >= self.budget {
            let victim = self.evict(pool, false)?;
            pool.free(victim);
        }
        Some(pool.alloc(holds, page))
    }

    /// A large frame of `pages` pages, for a value leaf the cache [`caches`](Self::caches),
    /// holding one reference. It evicts cold leaves first, until value leaves take
    /// at most a quarter of the budget, and the pool fits in it. If every leaf is in
    /// use, the pool grows past them.
    pub(crate) fn alloc_large(
        &mut self,
        pool: &mut Pool,
        holds: Holds,
        page: u64,
        pages: usize,
    ) -> FrameId {
        self.make_room(pool, pages);
        pool.alloc_large(holds, page, pages)
    }

    /// Like [`alloc_large`](Self::alloc_large), but returns `None` rather than
    /// growing the pool past its budget.
    pub(crate) fn try_alloc_large(
        &mut self,
        pool: &mut Pool,
        holds: Holds,
        page: u64,
        pages: usize,
    ) -> Option<FrameId> {
        self.make_room(pool, pages)
            .then(|| pool.alloc_large(holds, page, pages))
    }

    /// Evicts cold leaves until a large frame of `pages` pages fits: value leaves
    /// within a quarter of the budget, and the pool within the budget. Returns
    /// whether it fits.
    fn make_room(&mut self, pool: &mut Pool, pages: usize) -> bool {
        while pool.large_pages() + pages > self.budget / 4 {
            let Some(victim) = self.evict(pool, true) else {
                return false;
            };
            pool.free(victim);
        }
        while pool.in_use() + pages > self.budget {
            let Some(victim) = self.evict(pool, false) else {
                return false;
            };
            pool.free(victim);
        }
        true
    }

    /// Finds a frame to reuse, a large one if `large`: a clean, unreferenced leaf
    /// the clock hand finds cold, or one left over from a read whose future was
    /// dropped.
    fn evict(&mut self, pool: &mut Pool, large: bool) -> Option<FrameId> {
        let n = pool.slots();
        for _ in 0..2 * n {
            let id = pool.slot(self.hand % n);
            self.hand = (self.hand + 1) % n;
            let frame = *pool.frame(id);
            if frame.holds == Holds::Nothing || (large && !pool::is_large(id)) || pool.refs(id) != 0
            {
                continue;
            }
            match frame.holds {
                Holds::Loading => return Some(id),
                Holds::Leaf if !frame.dirty => {
                    if frame.clock {
                        pool.frame_mut(id).clock = false;
                    } else {
                        self.map.remove(&frame.page);
                        return Some(id);
                    }
                }
                _ => {}
            }
        }
        None
    }

    #[cfg(test)]
    pub(crate) fn entries(&self) -> impl Iterator<Item = (u64, FrameId)> + '_ {
        self.map.iter().map(|(&page, &frame)| (page, frame))
    }
}

/// Hashes page ids, which are already well spread, with one multiplication.
#[derive(Default)]
pub(crate) struct PageHasher(u64);

impl Hasher for PageHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 << 8 | u64::from(b)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        }
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = n.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}
