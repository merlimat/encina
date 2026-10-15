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

//! Checks that reads, iteration, and writes from a reused batch don't allocate.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use crate::tests::now;
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

const ENTRIES: u64 = 5000;

fn fill(batch: &mut WriteBatch, value: u8) {
    batch.clear();
    for i in 0..ENTRIES {
        batch.put(&key(i), &[value; 16]);
    }
}

#[test]
fn the_data_path_does_not_allocate() {
    let db = Db::in_memory(Options::default()).unwrap();
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
}
