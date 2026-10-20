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
//! one isn't cached. The large frames have a clock hand of their own, so that
//! making room among them passes no other frame.
//!
//! A write keeps the leaves it changes and the ones it builds until it lets go of
//! them all, so in a write larger than the cache the hand finds nothing to evict,
//! and the pool grows past the budget. Once a sweep has found nothing while a write
//! holds leaves, the hand looks at no more than [`STEP`] frames for each frame
//! taken, until the write lets go: sweeping the whole pool for every frame would
//! make the write quadratic in its leaves. The leaves readers let go of meanwhile
//! are still found, a step at a time.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::metrics::Counter;
use crate::pool::{FrameId, Holds, Pool};

/// The most frames the clock hand looks at to find one to evict, once a sweep has
/// found none while a write holds leaves.
pub(crate) const STEP: usize = 64;

pub(crate) struct Cache {
    /// Each cached leaf's frame, by page id.
    map: HashMap<u64, FrameId, BuildHasherDefault<PageHasher>>,
    /// The clock hand over every frame, and the one over the large frames.
    hand: usize,
    large_hand: usize,
    /// Pages the pool may hold before leaves get evicted.
    budget: usize,
    /// Whether a write holds leaves, and whether a sweep has found nothing to
    /// evict since it began to.
    writing: bool,
    exhausted: bool,
    /// The leaves that reads and writes found cached and had to read, which they
    /// count themselves, and the leaves evicted, if the store records metrics.
    pub(crate) hits: Counter,
    pub(crate) misses: Counter,
    pub(crate) evictions: Counter,
    /// The frames the clock hands looked at.
    #[cfg(test)]
    pub(crate) looked: usize,
}

impl Cache {
    pub(crate) fn new(budget: usize) -> Cache {
        let map = HashMap::with_capacity_and_hasher(budget, Default::default());
        Cache {
            map,
            hand: 0,
            large_hand: 0,
            budget,
            writing: false,
            exhausted: false,
            hits: Counter::default(),
            misses: Counter::default(),
            evictions: Counter::default(),
            #[cfg(test)]
            looked: 0,
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

    /// Notes whether a write holds leaves: from when it has planned its batch until
    /// it lets go of the leaves it pinned and built. Either way, the clock hand
    /// sweeps the whole pool again.
    pub(crate) fn set_writing(&mut self, writing: bool) {
        self.writing = writing;
        self.exhausted = false;
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
            let victim = self.evict(pool)?;
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
            let Some(victim) = self.evict_large(pool) else {
                return false;
            };
            pool.free(victim);
        }
        while pool.in_use() + pages > self.budget {
            let Some(victim) = self.evict(pool) else {
                return false;
            };
            pool.free(victim);
        }
        true
    }

    /// Finds a frame to reuse. The hand sweeps the whole pool, unless a sweep has
    /// found nothing since the write in progress began to hold leaves: then it looks
    /// at the next [`STEP`] frames only.
    fn evict(&mut self, pool: &mut Pool) -> Option<FrameId> {
        let n = pool.slots();
        let look = if self.exhausted {
            STEP.min(2 * n)
        } else {
            2 * n
        };
        let victim = self.turn(pool, false, look);
        self.exhausted |= self.writing && victim.is_none();
        victim
    }

    /// Finds a large frame to reuse, with the hand that passes no other frame.
    fn evict_large(&mut self, pool: &mut Pool) -> Option<FrameId> {
        self.turn(pool, true, 2 * pool.large_slots())
    }

    /// Moves the clock hand over the large frames if `large`, or else the one over
    /// every frame, past at most `look` frames, to one to reuse: a clean,
    /// unreferenced leaf it finds cold, or one left over from a read whose future
    /// was dropped. It marks cold the leaves used since it last passed them, so two
    /// rounds find one if there is any.
    fn turn(&mut self, pool: &mut Pool, large: bool, look: usize) -> Option<FrameId> {
        let (n, hand) = if large {
            (pool.large_slots(), &mut self.large_hand)
        } else {
            (pool.slots(), &mut self.hand)
        };
        for _ in 0..look {
            let i = *hand % n;
            *hand = (i + 1) % n;
            let id = if large {
                pool.large_slot(i)
            } else {
                pool.slot(i)
            };
            #[cfg(test)]
            {
                self.looked += 1;
            }
            let frame = *pool.frame(id);
            if frame.holds == Holds::Nothing || pool.refs(id) != 0 {
                continue;
            }
            match frame.holds {
                Holds::Loading => return Some(id),
                Holds::Leaf if !frame.dirty => {
                    if frame.clock {
                        pool.frame_mut(id).clock = false;
                    } else {
                        self.map.remove(&frame.page);
                        self.evictions.add(1);
                        return Some(id);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Whether a write holds leaves.
    #[cfg(test)]
    pub(crate) fn writing(&self) -> bool {
        self.writing
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node;

    /// Caches leaf `page` in a new frame of `pages` pages, a large one if more than
    /// one, which nothing holds.
    fn cache_leaf(cache: &mut Cache, pool: &mut Pool, page: u64, pages: usize) -> FrameId {
        let frame = match pages {
            1 => cache.alloc(pool, Holds::Leaf, page),
            _ => cache.alloc_large(pool, Holds::Leaf, page, pages),
        };
        cache.insert(page, frame);
        pool.unref(frame);
        frame
    }

    /// Caches leaf `page` in a new frame, which the caller holds.
    fn pin_leaf(cache: &mut Cache, pool: &mut Pool, page: u64) -> FrameId {
        let frame = cache.alloc(pool, Holds::Leaf, page);
        cache.insert(page, frame);
        frame
    }

    #[test]
    fn making_room_for_a_value_leaf_passes_no_other_frame() {
        // Value leaves of two pages may take 256 of the 1024 pages.
        let (mut pool, mut cache) = (Pool::new(256), Cache::new(1024));
        for page in 0..700 {
            cache_leaf(&mut cache, &mut pool, page, 1);
        }
        let value_leaf = |i: u64| node::spanned(1000 + 2 * i, 2);
        for i in 0..128 {
            cache_leaf(&mut cache, &mut pool, value_leaf(i), 2);
        }
        assert_eq!(pool.large_pages(), 256);
        // Each value leaf read from now on evicts another. The hand looks at large
        // frames only, two rounds of them at most.
        for i in 128..512 {
            let looked = cache.looked;
            cache_leaf(&mut cache, &mut pool, value_leaf(i), 2);
            assert!(
                cache.looked - looked <= 2 * 128,
                "the hand passed other frames"
            );
            assert_eq!(pool.large_pages(), 256);
        }
        for page in 0..700 {
            assert!(cache.map.contains_key(&page), "leaf {page} evicted");
        }
    }

    #[test]
    fn a_write_holding_every_frame_looks_a_step_ahead_at_a_time() {
        let (mut pool, mut cache) = (Pool::new(256), Cache::new(128));
        cache.set_writing(true);
        // The write's leaves fill the pool, and the first sweep finds nothing to
        // evict, after two rounds. The pool grows from then on.
        let mut held: Vec<_> = (0..129)
            .map(|page| pin_leaf(&mut cache, &mut pool, page))
            .collect();
        assert_eq!(cache.looked, 2 * 128);
        for page in 129..1000 {
            let looked = cache.looked;
            held.push(pin_leaf(&mut cache, &mut pool, page));
            assert!(cache.looked - looked <= STEP, "the hand swept the pool");
        }
        // A leaf that a reader lets go of meanwhile is evicted once the steps reach
        // it: one round of them marks it cold, and the next evicts it.
        let frame = held.swap_remove(500);
        pool.unref(frame);
        let last = 1000 + (3 * pool.slots() / STEP) as u64;
        for page in 1000.. {
            assert!(page < last, "the leaf let go of was not evicted");
            held.push(pin_leaf(&mut cache, &mut pool, page));
            if held.last() == Some(&frame) {
                break;
            }
        }
        assert!(!cache.map.contains_key(&500));
        // Once the write lets go of its leaves, the next frame comes from a sweep of
        // the whole pool, which evicts a leaf rather than grow the pool.
        for frame in held {
            pool.unref(frame);
        }
        cache.set_writing(false);
        let (in_use, slots) = (pool.in_use(), pool.slots());
        cache_leaf(&mut cache, &mut pool, 2000, 1);
        assert_eq!((pool.in_use(), pool.slots()), (in_use, slots));
    }

    #[test]
    fn without_a_write_every_sweep_looks_at_the_whole_pool() {
        let (mut pool, mut cache) = (Pool::new(256), Cache::new(64));
        // Readers hold every leaf, so the sweep finds nothing to evict, and the
        // pool grows by a chunk. The hand is back where it started.
        let held: Vec<_> = (0..65)
            .map(|page| pin_leaf(&mut cache, &mut pool, page))
            .collect();
        assert_eq!((pool.slots(), cache.hand), (128, 0));
        // The leaf a reader lets go of lies further from the hand than a step, and
        // the next sweep still finds it.
        pool.unref(held[64]);
        pool.frame_mut(held[64]).clock = false;
        assert_eq!(pin_leaf(&mut cache, &mut pool, 65), held[64]);
        assert!(!cache.map.contains_key(&64));
    }
}
