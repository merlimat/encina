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

//! Checks that reads and iteration over cached pages, and writes from a reused
//! batch, don't allocate, and that a new batch takes room for its points at once.
//! The simulated disk doesn't allocate either, once warm; compio allocates per
//! operation on a real one.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::pin::pin;

use crate::sim::{SimDb, SimDisk};
use crate::tests::{metered, now, poll};
use crate::tree::Budget;
use crate::{Comparison, Db, Options, WriteBatch};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn note_allocation() {
    if COUNTING.try_with(Cell::get).unwrap_or(false) {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
    }
}

// SAFETY: forwards everything to the system allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note_allocation();
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// The allocations `f` makes on this thread.
fn allocations(f: impl FnOnce()) -> usize {
    ALLOCATIONS.with(|n| n.set(0));
    COUNTING.with(|c| c.set(true));
    f();
    COUNTING.with(|c| c.set(false));
    ALLOCATIONS.with(Cell::get)
}

fn key(i: u64) -> [u8; 16] {
    let mut key = [0; 16];
    key[..8].copy_from_slice(&(i % 16).to_be_bytes());
    key[8..].copy_from_slice(&(i / 16).to_be_bytes());
    key
}

/// A location, starting with `round`.
fn value(i: u64, round: u8) -> [u8; 16] {
    let mut value = [round; 16];
    value[8..].copy_from_slice(&(i * 1024).to_be_bytes());
    value
}

const ENTRIES: u64 = 20_000;

fn fill(batch: &mut WriteBatch, round: u8) {
    fill_from(batch, 0, round);
}

fn fill_from(batch: &mut WriteBatch, first: u64, round: u8) {
    batch.clear();
    for i in first..first + ENTRIES {
        batch.put(&key(i), &value(i, round));
    }
}

#[test]
fn the_data_path_does_not_allocate() {
    data_path_does_not_allocate(Options::default());
}

/// Nor does recording metrics. The SDK reads the clock, which Miri doesn't allow.
#[test]
#[cfg_attr(miri, ignore)]
fn the_data_path_does_not_allocate_recording_metrics() {
    data_path_does_not_allocate(metered("alloc").page_size(4096));
}

fn data_path_does_not_allocate(opts: Options) {
    let db = now(Db::open_sim(SimDisk::new(), opts)).unwrap();
    let mut batch = WriteBatch::new();
    // The first writes grow the frame pool and the scratch space.
    for value in 0..3 {
        fill(&mut batch, value);
        now(db.write(&mut batch)).unwrap();
    }
    let read_all = || {
        for i in 0..ENTRIES {
            let found = now(db.get(&key(i), Comparison::Floor, |_, v| v[0])).unwrap();
            assert_eq!(found, Some(2));
        }
    };
    let scan_all = || {
        let mut iter = db.range(..);
        let mut n = 0;
        while now(iter.next()).unwrap().is_some() {
            n += 1;
        }
        assert_eq!(n, ENTRIES);
    };
    read_all();
    scan_all();
    assert_eq!(allocations(read_all), 0, "get allocated");
    assert_eq!(allocations(scan_all), 0, "iteration allocated");
    fill(&mut batch, 3);
    assert_eq!(
        allocations(|| now(db.write(&mut batch)).unwrap()),
        0,
        "write allocated"
    );
    // Nor does a write that pauses after every few units of work, normalizing its
    // batch, planning it, loading its leaves and merging it.
    db.set_merge_slice(Some(64));
    fill(&mut batch, 4);
    assert_eq!(
        allocations(|| now(db.write(&mut batch)).unwrap()),
        0,
        "a write that paused allocated"
    );
    // Nor one whose batch holds ranges, and puts and deletes in no order, once a
    // first one grew the space to sort them.
    fill_scattered(&mut batch, 5);
    now(db.write(&mut batch)).unwrap();
    fill_scattered(&mut batch, 6);
    assert_eq!(
        allocations(|| now(db.write(&mut batch)).unwrap()),
        0,
        "a write of ranges and scattered keys allocated"
    );
    // Nor does a reader that holds back the pages a write replaces, nor the next
    // write, which frees them, once a first round has grown the file, and with it
    // the simulated disk.
    let hold_back = |batch: &mut WriteBatch, round| {
        let reader = db.range(..);
        fill(batch, round);
        now(db.write(batch)).unwrap();
        drop(reader);
        fill(batch, round + 1);
        now(db.write(batch)).unwrap();
    };
    hold_back(&mut batch, 7);
    assert_eq!(
        allocations(|| hold_back(&mut batch, 9)),
        0,
        "freeing the pages a reader held back allocated"
    );
}

/// Fills `batch` with puts, deletes and ranges, in no order.
fn fill_scattered(batch: &mut WriteBatch, round: u8) {
    batch.clear();
    for n in 0..ENTRIES {
        let i = n * 7919 % ENTRIES;
        match n % 50 {
            0 => batch.delete(&key(i)),
            1 => batch.delete_range(&key(i), &key(i + 40)),
            _ => batch.put(&key(i), &value(i, round)),
        }
    }
}

#[test]
fn misses_do_not_allocate() {
    // Reopened with a cache of 16 pages, too small for the store, so that lookups
    // and scans keep reading leaves back, and reading ahead.
    let disk = SimDisk::new();
    let opts = Options::default().cache_size(16 * 4096);
    let db = now(Db::open_sim(disk.clone(), opts.clone())).unwrap();
    let mut batch = WriteBatch::new();
    fill(&mut batch, 0);
    now(db.write(&mut batch)).unwrap();
    drop(db);
    let disk = disk.restart();
    let db = now(Db::open_sim(disk.clone(), opts)).unwrap();
    let read_all = || {
        for i in (0..ENTRIES).step_by(7) {
            let found = now(db.get(&key(i), Comparison::Equal, |_, v| v[0])).unwrap();
            assert_eq!(found, Some(0));
        }
    };
    let scan_all = || {
        let mut iter = db.range(..);
        let mut n = 0;
        while now(iter.next()).unwrap().is_some() {
            n += 1;
        }
        assert_eq!(n, ENTRIES);
    };
    // The first misses grow the frame pool and the table of reads.
    read_all();
    scan_all();
    let ops = disk.ops();
    assert_eq!(allocations(read_all), 0, "lookups allocated");
    assert_eq!(allocations(scan_all), 0, "scans allocated");
    let leaves = db.tree().leaves().len() as u64;
    assert!(disk.ops() - ops > leaves, "the reads hit the cache");
}

#[test]
fn group_commits_do_not_allocate() {
    let disk = SimDisk::new();
    let db = now(Db::open_sim(disk.clone(), Options::default())).unwrap();
    let mut batches: [WriteBatch; 4] = Default::default();
    let fill_all = |batches: &mut [WriteBatch; 4], round: u8| {
        for (i, batch) in batches.iter_mut().enumerate() {
            fill_from(batch, i as u64 * ENTRIES, round);
        }
    };
    // The first write waits for its sync, and the others queue up behind it and
    // commit together.
    let write_group = |batches: &mut [WriteBatch; 4]| {
        let [a, b, c, d] = batches;
        let start = db.tree().current().generation;
        disk.hold_syncs();
        let mut a = pin!(db.write(a));
        let mut b = pin!(db.write(b));
        let mut c = pin!(db.write(c));
        let mut d = pin!(db.write(d));
        assert!(poll(a.as_mut()).is_pending());
        assert!(poll(b.as_mut()).is_pending());
        assert!(poll(c.as_mut()).is_pending());
        assert!(poll(d.as_mut()).is_pending());
        disk.release_syncs();
        for write in [a, b, c, d] {
            now(write).unwrap();
        }
        assert_eq!(db.tree().current().generation, start + 2);
    };
    // The first groups grow the frame pool, the scratch space, the groups'
    // batches, the table of writes waiting, and the page lists of write requests:
    // commits send more requests at once as the pages they free split up the free
    // space.
    for round in 0..10 {
        fill_all(&mut batches, round);
        write_group(&mut batches);
    }
    fill_all(&mut batches, 10);
    assert_eq!(
        allocations(|| write_group(&mut batches)),
        0,
        "group commit allocated"
    );
    db.check();
}

/// A store holding 10 entries too large for a leaf, of `len` bytes each.
fn value_leaves(opts: Options, len: usize) -> SimDb {
    let db = now(Db::open_sim(SimDisk::new(), opts)).unwrap();
    let mut batch = WriteBatch::new();
    for i in 0..10 {
        batch.put(&key(i), &vec![i as u8; len]);
    }
    now(db.write(&mut batch)).unwrap();
    db
}

fn scan_values(db: &SimDb, len: usize) {
    let mut iter = db.range(..);
    let mut n = 0;
    while let Some((_, v)) = now(iter.next()).unwrap() {
        assert_eq!(v.len(), len);
        n += 1;
    }
    assert_eq!(n, 10);
}

#[test]
fn cached_value_leaves_are_read_without_allocating() {
    let db = value_leaves(Options::default(), 10_000);
    let get_all = || {
        for i in 0..10 {
            let found = now(db.get(&key(i), Comparison::Equal, |_, v| v.len())).unwrap();
            assert_eq!(found, Some(10_000));
        }
    };
    // The first reads take a large frame for each, which the cache keeps, and the
    // first iterator its buffers.
    get_all();
    scan_values(&db, 10_000);
    assert_eq!(allocations(get_all), 0, "get allocated");
    assert_eq!(
        allocations(|| scan_values(&db, 10_000)),
        0,
        "iteration allocated"
    );
}

#[test]
fn reading_a_value_leaf_too_large_to_cache_allocates_its_buffer_only() {
    // A quarter of a 64 KiB cache is 4 pages, and these values take 5.
    let db = value_leaves(Options::default().cache_size(64 << 10), 20_000);
    let get = |i| now(db.get(&key(i), Comparison::Equal, |_, v| v.len())).unwrap();
    assert_eq!(get(0), Some(20_000));
    assert_eq!(allocations(|| assert_eq!(get(3), Some(20_000))), 1);
    // An iterator reuses its buffer for the next value leaf no larger.
    scan_values(&db, 20_000);
    assert_eq!(allocations(|| scan_values(&db, 20_000)), 1);
}

#[test]
fn a_new_batch_takes_room_for_its_points_at_once() {
    // Sorted, so that normalizing it needs no scratch buffer: room for the points,
    // and for the end of their one run.
    let mut batch = WriteBatch::new();
    for i in 0..100_000u64 {
        batch.put(&i.to_be_bytes(), &[1]);
    }
    let mut budget = Budget::fixed(usize::MAX);
    budget.start();
    assert_eq!(allocations(|| assert!(batch.prepare(&mut budget))), 2);
}
