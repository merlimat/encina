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

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::pool::{FrameId, Holds, Pool};

pub(crate) struct Cache {
    map: HashMap<u64, FrameId, BuildHasherDefault<PageHasher>>,
    hand: usize,
    /// Frames the pool may hold before leaves get evicted.
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

    /// A frame holding one reference, evicting a cold leaf first when the pool is
    /// at its budget. If every leaf is in use, the pool grows past the budget.
    pub(crate) fn alloc(&mut self, pool: &mut Pool, holds: Holds, page: u64) -> FrameId {
        if pool.in_use() >= self.budget
            && let Some(victim) = self.evict(pool)
        {
            pool.free(victim);
        }
        pool.alloc(holds, page)
    }

    /// Finds a frame to reuse: a clean, unreferenced leaf the clock hand finds
    /// cold, or one left over from a read whose future was dropped.
    fn evict(&mut self, pool: &mut Pool) -> Option<FrameId> {
        let n = pool.len();
        for _ in 0..2 * n {
            let id = (self.hand % n) as FrameId;
            self.hand = (self.hand + 1) % n;
            if pool.refs(id) != 0 {
                continue;
            }
            let frame = pool.frame_mut(id);
            match frame.holds {
                Holds::Loading => return Some(id),
                Holds::Leaf if !frame.dirty => {
                    if frame.clock {
                        frame.clock = false;
                    } else {
                        let page = frame.page;
                        self.map.remove(&page);
                        return Some(id);
                    }
                }
                _ => {}
            }
        }
        None
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.map.len()
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
