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
//!
//! A write takes its pages in order from runs of free pages, so that its commit
//! writes each run with one request. Free stretches shorter than [`MIN_RUN`] are
//! skipped, and the file grows instead: they wait until the pages around them are
//! freed too. Pages likely to be replaced soon and pages likely to stay take
//! separate runs, so that the pages replaced together are freed together, as
//! runs again.
//!
//! A few pages that stay can hold a whole stretch back. So a write moves the pages
//! it keeps out of windows that are mostly free, when that only costs their writes
//! (see [`Space::should_move`]), and once the end of the file is free, the file
//! ends earlier.

use std::collections::VecDeque;

/// The shortest stretch of free pages, short of the end of the file, that a
/// write allocates from: 64 KiB with 4 KiB pages.
const MIN_RUN: u64 = 16;

/// How many pages a run at the end of the file spans. Bounding it lets both
/// [`Life`]s grow the file, each in runs of its own.
const CHUNK: u64 = 256;

/// The most pages in use in a window of 64, the pages one word of the bitmap
/// tracks, for a write to move them out: each page moved frees at least three,
/// and the emptied window makes a run.
pub(crate) const MOVE_AT: u32 = 16;

/// How long a page a write allocates is likely to stay in use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Life {
    /// Likely replaced by one of the next writes: internal nodes, and leaves
    /// that may take more keys.
    Short = 0,
    /// Likely to stay: full leaves that a run of appends left behind.
    Long = 1,
}

/// A run of free pages that allocation takes pages from, in order: `next..end`.
#[derive(Clone, Copy, Default)]
struct Run {
    next: u64,
    end: u64,
}

pub(crate) struct Space {
    used: Vec<u64>,
    /// No page at or above this one is in use, or in the file's tree.
    high_water: u64,
    /// No page below this one is free.
    lowest_free: u64,
    /// The run each [`Life`] allocates from, while a write is in progress.
    runs: [Run; 2],
    /// Pages the write in progress allocated and freed again, as when two nodes
    /// merge: holes in its runs, which its next short-lived pages fill.
    holes: Vec<u64>,
    /// Pages the write in progress allocated, and how many of them were for nodes
    /// it moved.
    allocated: u64,
    moved: u64,
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
            runs: [Run::default(); 2],
            holes: Vec::new(),
            allocated: 0,
            moved: 0,
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

    /// Makes the next allocation start over from the lowest free page: each write
    /// starts with this, so that it reuses what earlier writes freed. The
    /// high-water mark comes down to the last page in use.
    pub(crate) fn restart(&mut self) {
        self.runs = [Run::default(); 2];
        self.holes.clear();
        (self.allocated, self.moved) = (0, 0);
        let last = self.used.iter().rposition(|&w| w != 0);
        self.high_water = last.map_or(0, |i| {
            (i as u64 + 1) * 64 - u64::from(self.used[i].leading_zeros())
        });
        self.used.truncate(self.high_water.div_ceil(64) as usize);
    }

    /// A free page, which becomes used: for a short-lived page, a hole the write
    /// left if any, or else the next free page of the run for `life`. When that
    /// run is used up, the next stretch of at least [`MIN_RUN`] free pages takes
    /// over, or else the end of the file.
    pub(crate) fn alloc(&mut self, life: Life) -> u64 {
        self.allocated += 1;
        // A run can take in holes, and a hole can be taken while in a run: both
        // ways skip the pages the other took.
        if life == Life::Short {
            while let Some(page) = self.holes.pop() {
                if !self.is_used(page) {
                    self.mark_used(page);
                    return page;
                }
            }
        }
        let (i, other) = (life as usize, self.runs[1 - life as usize]);
        let run = self.runs[i];
        let mut page = run.end;
        if run.next < run.end {
            page = self.next_free(run.next);
        }
        if page >= run.end {
            let (next, end) = self.find_run(run.end, other);
            (page, self.runs[i].end) = (next, end);
        }
        self.runs[i].next = page + 1;
        if page >= self.high_water {
            self.high_water = page + 1;
            self.used.resize(self.high_water.div_ceil(64) as usize, 0);
        }
        self.mark_used(page);
        page
    }

    /// Whether the write in progress should move a node it keeps, on `page`, to
    /// a page from [`alloc_moved`](Self::alloc_moved), freeing `page` once the
    /// write is published: whether `page` sits in a window of 64 pages, below the
    /// high-water mark, with at most [`MOVE_AT`] in use. A write moves at most as
    /// many nodes as it writes otherwise.
    pub(crate) fn should_move(&self, page: u64) -> bool {
        let word = page / 64;
        (word + 1) * 64 <= self.high_water
            && self.used[word as usize].count_ones() <= MOVE_AT
            && 2 * self.moved < self.allocated
    }

    /// A page for a node that the write in progress moves: the next short-lived
    /// one, so that the moved node shares a request with the nodes the write
    /// builds.
    pub(crate) fn alloc_moved(&mut self) -> u64 {
        self.moved += 1;
        self.alloc(Life::Short)
    }

    /// The first stretch of free pages at or after `from`, outside the `other`
    /// run, that is at least [`MIN_RUN`] long or reaches the end of the file,
    /// where it takes [`CHUNK`] pages.
    fn find_run(&mut self, from: u64, other: Run) -> (u64, u64) {
        let mut start = self.next_free(from.max(self.lowest_free));
        if from <= self.lowest_free {
            self.lowest_free = start;
        }
        loop {
            if (other.next..other.end).contains(&start) {
                start = self.next_free(other.end);
                continue;
            }
            let mut end = self.next_used(start).unwrap_or(start + CHUNK);
            if other.next < other.end && other.next > start {
                end = end.min(other.next);
            }
            if end - start >= MIN_RUN || end > self.high_water {
                return (start, end);
            }
            start = self.next_free(end);
        }
    }

    /// The first free page at or after `page`. Every page from the high-water
    /// mark up is free.
    fn next_free(&self, page: u64) -> u64 {
        self.find(page, false).unwrap_or(page.max(self.high_water))
    }

    /// The first used page at or after `page`, if any.
    fn next_used(&self, page: u64) -> Option<u64> {
        self.find(page, true)
    }

    /// The first page at or after `page`, below the high-water mark, that is used
    /// or free.
    fn find(&self, page: u64, used: bool) -> Option<u64> {
        let flip = if used { 0 } else { u64::MAX };
        let mut word = (page / 64) as usize;
        let mut bits = (self.used.get(word)? ^ flip) & (u64::MAX << (page % 64));
        while bits == 0 {
            word += 1;
            bits = self.used.get(word)? ^ flip;
        }
        Some(word as u64 * 64 + u64::from(bits.trailing_zeros())).filter(|&p| p < self.high_water)
    }

    /// Frees a page now. Only for pages no reader can reach and no durable
    /// version uses: those a write allocated but didn't commit.
    pub(crate) fn free(&mut self, page: u64) {
        self.unmark(page);
        self.holes.push(page);
    }

    fn unmark(&mut self, page: u64) {
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
            self.unmark(page);
            freed(page);
        }
    }

    /// Pages the last write moved.
    #[cfg(test)]
    pub(crate) fn moved(&self) -> u64 {
        self.moved
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

    /// Space for `high_water` pages, all used except `free`.
    fn space_with(high_water: u64, free: impl IntoIterator<Item = u64>) -> Space {
        let free: Vec<_> = free.into_iter().collect();
        let mut space = Space::new(high_water);
        for page in (0..high_water).filter(|page| !free.contains(page)) {
            space.mark_used(page);
        }
        space
    }

    fn alloc_n(space: &mut Space, life: Life, n: u64) -> Vec<u64> {
        (0..n).map(|_| space.alloc(life)).collect()
    }

    #[test]
    fn takes_runs_in_order_and_skips_short_holes() {
        let run = 30..30 + MIN_RUN;
        let mut space = space_with(100, (10..10 + MIN_RUN - 1).chain(run.clone()));
        assert_eq!(
            alloc_n(&mut space, Life::Short, MIN_RUN),
            Vec::from_iter(run)
        );
        // Then the file grows.
        assert_eq!(alloc_n(&mut space, Life::Short, 3), [100, 101, 102]);
        assert_eq!(space.high_water(), 103);
        // The short hole stays free.
        assert!(!space.is_used(10));
    }

    #[test]
    fn a_short_hole_at_the_end_of_the_file_is_used() {
        let mut space = space_with(10, 8..10);
        assert_eq!(alloc_n(&mut space, Life::Short, 4), [8, 9, 10, 11]);
    }

    #[test]
    fn each_life_takes_runs_of_its_own() {
        let mut space = space_with(2, []);
        assert_eq!(space.alloc(Life::Short), 2);
        assert_eq!(space.alloc(Life::Long), 2 + CHUNK);
        assert_eq!(space.alloc(Life::Short), 3);
        assert_eq!(space.alloc(Life::Long), 3 + CHUNK);
        // A used-up run at the end of the file moves past the other's.
        alloc_n(&mut space, Life::Short, CHUNK - 2);
        assert_eq!(space.alloc(Life::Short), 2 + 2 * CHUNK);
        // A run takes all of a free stretch: the other life looks further on.
        let mut space = space_with(1000, (100..100 + 2 * MIN_RUN).chain(500..500 + MIN_RUN));
        assert_eq!(space.alloc(Life::Short), 100);
        assert_eq!(space.alloc(Life::Long), 500);
        assert_eq!(space.alloc(Life::Short), 101);
    }

    #[test]
    fn short_lived_pages_fill_the_holes_a_write_leaves() {
        let mut space = space_with(2, []);
        assert_eq!(alloc_n(&mut space, Life::Short, 40), Vec::from_iter(2..42));
        for page in [3, 5, 7] {
            space.free(page);
        }
        let mut filled = alloc_n(&mut space, Life::Short, 3);
        filled.sort_unstable();
        assert_eq!(filled, [3, 5, 7]);
        assert_eq!(space.alloc(Life::Short), 42);
    }

    #[test]
    fn a_run_and_the_holes_never_hand_out_a_page_twice() {
        let mut space = space_with(2, []);
        alloc_n(&mut space, Life::Short, 40);
        for page in 10..10 + MIN_RUN + 1 {
            space.free(page);
        }
        // A long-lived run takes the holes in, and a short-lived page takes one
        // of them back from it.
        assert_eq!(space.alloc(Life::Long), 10);
        assert_eq!(space.alloc(Life::Short), 10 + MIN_RUN);
        let long = alloc_n(&mut space, Life::Long, MIN_RUN - 1);
        assert_eq!(long, Vec::from_iter(11..10 + MIN_RUN));
        // The other holes went to the run.
        assert_eq!(space.alloc(Life::Short), 42);
    }

    #[test]
    fn restarting_reuses_what_was_freed() {
        let mut space = space_with(2, []);
        let pages = alloc_n(&mut space, Life::Short, MIN_RUN);
        alloc_n(&mut space, Life::Long, 1);
        for page in pages {
            space.free(page);
        }
        space.restart();
        assert_eq!(space.alloc(Life::Long), 2);
        assert_eq!(space.alloc(Life::Short), 3 + CHUNK);
    }

    #[test]
    fn restarting_ends_the_file_after_the_last_page_in_use() {
        let mut space = space_with(200, 70..200);
        space.restart();
        assert_eq!(space.high_water(), 70);
        assert_eq!(alloc_n(&mut space, Life::Short, 3), [70, 71, 72]);
        assert_eq!(space.high_water(), 73);
    }

    #[test]
    fn kept_pages_move_out_of_mostly_free_windows() {
        // Of each window of 64 pages, the first has one page in use too many, the
        // second just few enough, the third is full, and the file ends in the
        // fourth. No free stretch but the end of the file makes a run.
        let few = u64::from(MOVE_AT);
        let mut space = Space::new(224);
        let used = (0..4 * few)
            .step_by(4)
            .chain([63])
            .chain((64..64 + 4 * few).step_by(4))
            .chain(128..193);
        for page in used {
            space.mark_used(page);
        }
        assert!(!space.should_move(68), "the write built nothing yet");
        assert_eq!(space.alloc(Life::Short), 193);
        assert!(space.should_move(68));
        assert!(!space.should_move(4));
        assert!(!space.should_move(130));
        assert!(!space.should_move(192), "the window runs past the end");
        // A write moves at most as many pages as it builds.
        assert_eq!(space.alloc_moved(), 194);
        assert!(!space.should_move(68));
        space.alloc(Life::Short);
        assert!(space.should_move(68));
        space.restart();
        assert!(!space.should_move(68));
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
        assert!(!space.is_used(7) && !space.is_used(8));
    }
}
