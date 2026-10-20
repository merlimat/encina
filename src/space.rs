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
//! use until no reader of an older version, which might still read it, is left,
//! and a write frees it.
//!
//! A write takes its pages in order from runs of free pages, so that its commit
//! writes each run with one request. Free stretches shorter than
//! [`Options::min_run_pages`](crate::Options::min_run_pages) are skipped, and the
//! file grows instead: they wait until the pages around them are freed too. Pages likely to be replaced soon and pages likely to stay take
//! separate runs, so that the pages replaced together are freed together, as
//! runs again.
//!
//! A few pages that stay can hold a whole stretch back. So a write moves the pages
//! it keeps out of windows that are mostly free, when that only costs their writes
//! (see [`Space::should_move`]), and once the end of the file is free, the file
//! ends earlier.

use std::collections::VecDeque;
use std::iter;

use crate::node;

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
    /// The shortest stretch of free pages, short of the end of the file, that a
    /// write allocates from.
    min_run: u64,
    /// No page at or above this one is in use, or in the file's tree.
    high_water: u64,
    /// No page below this one is free.
    lowest_free: u64,
    /// The run each [`Life`] allocates from, while a write is in progress.
    runs: [Run; 2],
    /// Pages the write in progress allocated and freed again: holes in its runs,
    /// which its next short-lived pages fill.
    holes: Vec<u64>,
    /// How many pages the write in progress freed again.
    #[cfg(test)]
    pub(crate) freed: usize,
    /// The nodes that published versions replaced, by page id, with the
    /// generation that replaced them, oldest first.
    dropped: VecDeque<(u64, u64)>,
    /// The generations of readers that may still read older versions.
    readers: Vec<u64>,
    /// How many pages are in use, and how many of them `dropped` holds.
    in_use: u64,
    held: u64,
}

impl Space {
    /// Space for a file whose pages up to `high_water` may be in use, allocated
    /// from stretches of at least `min_run` free pages. Mark the pages in use
    /// with [`mark_used`](Self::mark_used).
    pub(crate) fn new(high_water: u64, min_run: u64) -> Space {
        Space {
            used: vec![0; high_water.div_ceil(64) as usize],
            min_run,
            high_water,
            lowest_free: 0,
            runs: [Run::default(); 2],
            holes: Vec::new(),
            #[cfg(test)]
            freed: 0,
            dropped: VecDeque::new(),
            readers: Vec::new(),
            in_use: 0,
            held: 0,
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
        self.in_use += 1;
    }

    /// Makes the next allocation start over from the lowest free page: each write
    /// starts with this, so that it reuses what earlier writes freed. The
    /// high-water mark comes down to the last page in use.
    pub(crate) fn restart(&mut self) {
        self.runs = [Run::default(); 2];
        self.holes.clear();
        #[cfg(test)]
        {
            self.freed = 0;
        }
        let last = self.used.iter().rposition(|&w| w != 0);
        self.high_water = last.map_or(0, |i| {
            (i as u64 + 1) * 64 - u64::from(self.used[i].leading_zeros())
        });
        self.used.truncate(self.high_water.div_ceil(64) as usize);
    }

    /// A free page, which becomes used: for a short-lived page, a hole the write
    /// left if any, or else the next free page of the run for `life`. When that
    /// run is used up, the next stretch of at least `min_run` free pages takes
    /// over, or else the end of the file.
    pub(crate) fn alloc(&mut self, life: Life) -> u64 {
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
        self.take(page, 1);
        page
    }

    /// `n` adjacent free pages for a value leaf, which become used: from what is
    /// left of the run for long-lived pages if they fit there, or else from the
    /// first stretch of free pages long enough outside both runs, or the end of the
    /// file. Returns the first.
    pub(crate) fn alloc_span(&mut self, n: u64) -> u64 {
        let run = self.runs[Life::Long as usize];
        let start = self.next_free(run.next.min(run.end));
        if start + n <= run.end && self.next_used(start).is_none_or(|used| used >= start + n) {
            self.runs[Life::Long as usize].next = start + n;
            self.take(start, n);
            return start;
        }
        let mut start = self.next_free(self.lowest_free);
        loop {
            if let Some(run) = self.runs.iter().find(|r| (r.next..r.end).contains(&start)) {
                start = self.next_free(run.end);
                continue;
            }
            let mut end = self.next_used(start).unwrap_or(u64::MAX);
            for run in self
                .runs
                .iter()
                .filter(|r| r.next < r.end && r.next > start)
            {
                end = end.min(run.next);
            }
            if end - start >= n {
                self.take(start, n);
                return start;
            }
            start = self.next_free(end);
        }
    }

    /// Marks free pages `page..page + n` used, growing the file to hold them.
    fn take(&mut self, page: u64, n: u64) {
        if page + n > self.high_water {
            self.high_water = page + n;
            self.used.resize(self.high_water.div_ceil(64) as usize, 0);
        }
        for page in page..page + n {
            self.mark_used(page);
        }
    }

    /// Whether a node on `page` that the write in progress keeps should move to a
    /// new page, freeing `page` once the write is published: whether `page` sits
    /// in a window of 64 pages, below the high-water mark, with at most
    /// [`MOVE_AT`] in use.
    pub(crate) fn should_move(&self, page: u64) -> bool {
        let word = page / 64;
        (word + 1) * 64 <= self.high_water && self.used[word as usize].count_ones() <= MOVE_AT
    }

    /// The first stretch of free pages at or after `from`, outside the `other`
    /// run, that is at least `min_run` long or reaches the end of the file,
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
            if end - start >= self.min_run || end > self.high_water {
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
        #[cfg(test)]
        {
            self.freed += 1;
        }
    }

    fn unmark(&mut self, page: u64) {
        debug_assert!(self.is_used(page), "page {page} freed twice");
        self.used[(page / 64) as usize] &= !(1 << (page % 64));
        self.lowest_free = self.lowest_free.min(page);
        self.in_use -= 1;
    }

    /// Records that the version published as `generation` no longer uses the node
    /// with page id `id`, which takes [`node::pages`] pages.
    pub(crate) fn drop_node(&mut self, generation: u64, id: u64) {
        self.dropped.push_back((generation, id));
        self.held += node::pages(id);
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

    /// The dropped nodes that no reader can still need, by page id, oldest first:
    /// each node's pages are freed as it is yielded, and those not yielded stay
    /// dropped. A node dropped by generation `g` is needed by readers of earlier
    /// generations.
    pub(crate) fn reclaim(&mut self) -> impl Iterator<Item = u64> + '_ {
        let oldest = self.readers.iter().copied().min().unwrap_or(u64::MAX);
        iter::from_fn(move || {
            let &(generation, id) = self.dropped.front()?;
            if generation > oldest {
                return None;
            }
            self.dropped.pop_front();
            self.held -= node::pages(id);
            let first = node::first_page(id);
            for page in first..first + node::pages(id) {
                self.unmark(page);
            }
            Some(id)
        })
    }

    /// How many pages are in use, and how many of them were dropped but not
    /// reclaimed yet: readers of older versions hold them back.
    pub(crate) fn usage(&self) -> (u64, u64) {
        (self.in_use, self.held)
    }

    /// Pages dropped but not reclaimed yet.
    #[cfg(test)]
    pub(crate) fn dropped(&self) -> impl Iterator<Item = u64> + '_ {
        self.dropped.iter().flat_map(|&(_, id)| {
            let first = node::first_page(id);
            first..first + node::pages(id)
        })
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

    const MIN_RUN: u64 = 16;

    /// Space for `high_water` pages, all used except `free`.
    fn space_with(high_water: u64, free: impl IntoIterator<Item = u64>) -> Space {
        let free: Vec<_> = free.into_iter().collect();
        let mut space = Space::new(high_water, MIN_RUN);
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
    fn a_shorter_min_run_takes_shorter_holes() {
        let mut space = space_with(100, [10, 20, 21]);
        space.min_run = 2;
        assert_eq!(alloc_n(&mut space, Life::Short, 3), [20, 21, 100]);
        space.min_run = 1;
        space.restart();
        assert_eq!(space.alloc(Life::Short), 10);
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
        // fourth.
        let few = u64::from(MOVE_AT);
        let mut space = Space::new(224, MIN_RUN);
        let used = (0..4 * few)
            .step_by(4)
            .chain([63])
            .chain((64..64 + 4 * few).step_by(4))
            .chain(128..193);
        for page in used {
            space.mark_used(page);
        }
        assert!(!space.should_move(4));
        assert!(space.should_move(68));
        assert!(!space.should_move(130));
        assert!(!space.should_move(192), "the window runs past the end");
    }

    #[test]
    fn dropped_pages_wait_for_older_readers() {
        let mut space = Space::new(10, MIN_RUN);
        for page in 0..10 {
            space.mark_used(page);
        }
        space.add_reader(4);
        let value_leaf = node::spanned(2, 2);
        space.drop_node(5, 7);
        space.drop_node(6, value_leaf);
        space.drop_node(6, 8);
        let mut freed = Vec::new();
        freed.extend(space.reclaim());
        assert!(freed.is_empty());
        space.remove_reader(4);
        space.add_reader(5);
        freed.extend(space.reclaim());
        assert_eq!(freed, [7]);
        space.remove_reader(5);
        // Those not taken stay dropped.
        freed.extend(space.reclaim().take(1));
        assert_eq!(freed, [7, value_leaf]);
        assert!(space.is_used(8));
        freed.extend(space.reclaim());
        assert_eq!(freed, [7, value_leaf, 8]);
        assert!([2, 3, 7, 8].iter().all(|&page| !space.is_used(page)));
    }

    #[test]
    fn spans_take_the_long_lived_run_or_a_stretch_long_enough() {
        let mut space = space_with(200, (10..14).chain(50..50 + MIN_RUN).chain(100..120));
        // The long-lived run starts at the first stretch of `MIN_RUN` free pages.
        assert_eq!(space.alloc(Life::Long), 50);
        assert_eq!(space.alloc_span(5), 51);
        assert_eq!(space.alloc(Life::Long), 56);
        // Too long for what is left of the run: the first stretch that holds it,
        // past the short hole and the run.
        assert_eq!(space.alloc_span(MIN_RUN), 100);
        // What fits in the rest of the run goes there, then to the short hole.
        assert_eq!(space.alloc_span(4), 57);
        assert_eq!(space.alloc_span(5), 61);
        assert_eq!(space.alloc_span(4), 10);
        assert_eq!(space.alloc_span(4), 116);
        // Then the end of the file, past the run there.
        assert_eq!(space.alloc(Life::Short), 200);
        assert_eq!(space.alloc_span(300), 200 + CHUNK);
        assert_eq!(space.high_water(), 500 + CHUNK);
        assert_eq!(space.alloc(Life::Short), 201);
    }
}
