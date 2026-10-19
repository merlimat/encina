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

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::mem;

use crate::error::{Error, Result};
use crate::tree::Budget;

/// A group of writes that [`Db::write`](crate::Db::write) applies atomically.
///
/// Operations take effect in the order they were added. A batch keeps its memory
/// when cleared, so a reused batch doesn't allocate.
#[derive(Default)]
pub struct WriteBatch {
    bytes: Vec<u8>,
    ops: Vec<Op>,
    // Set by `prepare`: the surviving puts and deletes sorted by key, and the
    // ranges merged into disjoint intervals sorted by start.
    points: Vec<u32>,
    ranges: Vec<(u32, u32)>,
    by_start: Vec<u32>,
    // Scratch space for sorting them: the buffer that merge passes alternate with,
    // and where the runs they merge end. Then the ranges that may cover the key
    // `prepare` is at, the one added last on top, and how far it got.
    scratch: Vec<u32>,
    run_ends: Vec<u32>,
    covering: BinaryHeap<u32>,
    step: Step,
}

/// How far [`WriteBatch::prepare`] got.
#[derive(Clone, Copy, Default)]
enum Step {
    #[default]
    Start,
    /// Listing the puts and deletes, and the ranges that aren't empty, from
    /// operation `i` on.
    Collect(usize),
    /// Sorting the ranges by start, then the puts and deletes by key.
    SortRanges(Sort),
    SortPoints(Sort),
    /// Keeping the last operation on each key, from point `i` on: the first `kept`
    /// are kept. The first `started` ranges in `by_start` start at or before its
    /// key, and `covering` holds those of them that may cover it.
    Dedup {
        i: usize,
        kept: usize,
        started: usize,
    },
    /// Merging the ranges, from range `i` on.
    Merge(usize),
    Done,
}

/// A merge sort in progress: it finds the runs of items sorted already, then
/// merges them in pairs, pass after pass, until one is left.
#[derive(Clone, Copy, Default)]
struct Sort {
    /// Whether it found the runs, and whether the items are in the scratch buffer
    /// rather than their own.
    merging: bool,
    flipped: bool,
    /// Finding runs, the next item, and where the run it may extend starts.
    /// Merging, the next item of each run of pair `pair`.
    i: usize,
    j: usize,
    pair: usize,
}

#[derive(Clone, Copy)]
struct Op {
    kind: Kind,
    // Key and value for puts and deletes; start and end for ranges.
    a: u32,
    a_len: u32,
    b: u32,
    b_len: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Put,
    Delete,
    DeleteRange,
    /// An operation of a write that left the group whose batch holds it.
    Withdrawn,
}

impl WriteBatch {
    /// Creates an empty batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets `key` to `value`.
    ///
    /// `key` may not exceed [`Options::max_key_size`](crate::Options::max_key_size),
    /// a bit under a quarter of the page size. Otherwise
    /// [`Db::write`](crate::Db::write) rejects the whole batch with
    /// [`Error::KeyTooLarge`]. `value` may have any length, as long as the batch
    /// stays under 4 GiB. An entry too large for a leaf gets pages of its own.
    pub fn put(&mut self, key: &[u8], value: &[u8]) {
        self.push(Kind::Put, key, value);
    }

    /// Removes `key`, if present.
    pub fn delete(&mut self, key: &[u8]) {
        self.push(Kind::Delete, key, &[]);
    }

    /// Removes every key in `start..end`. If `start >= end`, it removes nothing.
    pub fn delete_range(&mut self, start: &[u8], end: &[u8]) {
        self.push(Kind::DeleteRange, start, end);
    }

    /// Empties the batch, keeping its memory for reuse.
    pub fn clear(&mut self) {
        self.bytes.clear();
        self.ops.clear();
    }

    fn push(&mut self, kind: Kind, a: &[u8], b: &[u8]) {
        assert!(
            self.bytes.len() + a.len() + b.len() <= u32::MAX as usize,
            "write batch larger than 4 GiB"
        );
        let a_at = self.bytes.len() as u32;
        self.bytes.extend_from_slice(a);
        let b_at = self.bytes.len() as u32;
        self.bytes.extend_from_slice(b);
        let (a_len, b_len) = (a.len() as u32, b.len() as u32);
        self.ops.push(Op {
            kind,
            a: a_at,
            a_len,
            b: b_at,
            b_len,
        });
    }

    /// Key of a put or delete.
    pub(crate) fn key(&self, op: u32) -> &[u8] {
        let op = &self.ops[op as usize];
        slice(&self.bytes, op.a, op.a_len)
    }

    /// Value of a put; `None` for a delete.
    pub(crate) fn value(&self, op: u32) -> Option<&[u8]> {
        let op = &self.ops[op as usize];
        (op.kind == Kind::Put).then(|| slice(&self.bytes, op.b, op.b_len))
    }

    pub(crate) fn range_start(&self, op: u32) -> &[u8] {
        self.key(op)
    }

    pub(crate) fn range_end(&self, op: u32) -> &[u8] {
        let op = &self.ops[op as usize];
        slice(&self.bytes, op.b, op.b_len)
    }

    /// The puts and deletes left by [`prepare`](Self::prepare), sorted by key.
    pub(crate) fn points(&self) -> &[u32] {
        &self.points
    }

    /// The ranges left by [`prepare`](Self::prepare): disjoint, sorted, each as the
    /// operation holding its start and the one holding its end.
    pub(crate) fn ranges(&self) -> &[(u32, u32)] {
        &self.ranges
    }

    /// Checks that no put's key exceeds `max_key` bytes.
    pub(crate) fn check(&self, max_key: usize) -> Result<()> {
        for op in &self.ops {
            let size = op.a_len as usize;
            if op.kind == Kind::Put && size > max_key {
                return Err(Error::KeyTooLarge { size, max: max_key });
            }
        }
        Ok(())
    }

    /// The memory its operations take.
    pub(crate) fn size(&self) -> usize {
        self.bytes.len() + self.ops.len() * mem::size_of::<Op>()
    }

    /// Adds the operations of `other` after its own, and returns the range of
    /// their indexes.
    pub(crate) fn append(&mut self, other: &WriteBatch) -> (u32, u32) {
        assert!(
            self.bytes.len() + other.bytes.len() <= u32::MAX as usize,
            "write batch larger than 4 GiB"
        );
        let shift = self.bytes.len() as u32;
        let first = self.ops.len() as u32;
        self.bytes.extend_from_slice(&other.bytes);
        self.ops.extend(other.ops.iter().map(|op| Op {
            a: op.a + shift,
            b: op.b + shift,
            ..*op
        }));
        (first, self.ops.len() as u32)
    }

    /// Takes the operations with indexes in `start..end` out of the batch.
    pub(crate) fn withdraw(&mut self, (start, end): (u32, u32)) {
        for op in &mut self.ops[start as usize..end as usize] {
            op.kind = Kind::Withdrawn;
        }
    }

    /// Makes the next [`prepare`](Self::prepare) start over: each write starts with
    /// this, since the batch may have changed since it was last prepared.
    pub(crate) fn restart(&mut self) {
        self.step = Step::Start;
    }

    /// Reduces the batch to an equivalent set of ranges to remove first and puts
    /// and deletes to apply after, at most one per key. It pauses once `budget` is
    /// spent, returning false: calling it again goes on where it left off, as long
    /// as the batch doesn't change meanwhile.
    pub(crate) fn prepare(&mut self, budget: &mut Budget) -> bool {
        let WriteBatch {
            bytes,
            ops,
            points,
            ranges,
            by_start,
            scratch,
            run_ends,
            covering,
            step,
        } = self;
        let a = |i: u32| {
            let op = &ops[i as usize];
            slice(bytes, op.a, op.a_len)
        };
        let b = |i: u32| {
            let op = &ops[i as usize];
            slice(bytes, op.b, op.b_len)
        };
        loop {
            match step {
                Step::Start => {
                    points.clear();
                    ranges.clear();
                    by_start.clear();
                    covering.clear();
                    *step = Step::Collect(0);
                }
                Step::Collect(i) => {
                    while *i < ops.len() {
                        if budget.spent() {
                            return false;
                        }
                        budget.work += 1;
                        let op = *i as u32;
                        match ops[*i].kind {
                            Kind::Put | Kind::Delete => points.push(op),
                            Kind::DeleteRange if a(op) < b(op) => by_start.push(op),
                            Kind::DeleteRange | Kind::Withdrawn => {}
                        }
                        *i += 1;
                    }
                    *step = Step::SortRanges(Sort::default());
                }
                // Both start in the order of their operations, and the sort keeps
                // the order of equal keys: the last operation on a key comes last.
                Step::SortRanges(s) => {
                    if !sort(by_start, scratch, run_ends, s, a, MIN_RUN, budget) {
                        return false;
                    }
                    *step = Step::SortPoints(Sort::default());
                }
                Step::SortPoints(s) => {
                    if !sort(points, scratch, run_ends, s, a, MIN_RUN, budget) {
                        return false;
                    }
                    *step = Step::Dedup {
                        i: 0,
                        kept: 0,
                        started: 0,
                    };
                }
                Step::Dedup { i, kept, started } => {
                    while *i < points.len() {
                        if budget.spent() {
                            return false;
                        }
                        budget.work += 1;
                        let op = points[*i];
                        let key = a(op);
                        if *i + 1 < points.len() && a(points[*i + 1]) == key {
                            *i += 1;
                            continue;
                        }
                        // The last operation on a key wins, unless a range added after
                        // it covers the key: then the range has the last word, and it
                        // is applied first anyway. The ranges that start at or before
                        // the key and end after it go on a heap, the one added last on
                        // top, which comes off while it ends at or before the key: it
                        // ends before the keys that follow too. Then the one on top
                        // covers the key, and none added after it does. A push or a
                        // pop counts a unit per level of the heap.
                        if let Some(&r) = by_start.get(*started)
                            && a(r) <= key
                        {
                            *started += 1;
                            if key < b(r) {
                                covering.push(r);
                                budget.work += covering.len().ilog2() as usize;
                            }
                            continue;
                        }
                        if let Some(&r) = covering.peek()
                            && b(r) <= key
                        {
                            budget.work += covering.len().ilog2() as usize;
                            covering.pop();
                            continue;
                        }
                        *i += 1;
                        if covering.peek().is_none_or(|&r| r < op) {
                            points[*kept] = op;
                            *kept += 1;
                        }
                    }
                    points.truncate(*kept);
                    *step = Step::Merge(0);
                }
                Step::Merge(i) => {
                    while *i < by_start.len() {
                        if budget.spent() {
                            return false;
                        }
                        budget.work += 1;
                        let r = by_start[*i];
                        *i += 1;
                        if let Some(last) = ranges.last_mut()
                            && a(r) <= b(last.1)
                        {
                            if b(r) > b(last.1) {
                                last.1 = r;
                            }
                            continue;
                        }
                        ranges.push((r, r));
                    }
                    *step = Step::Done;
                }
                Step::Done => return true,
            }
        }
    }
}

fn slice(bytes: &[u8], at: u32, len: u32) -> &[u8] {
    &bytes[at as usize..(at + len) as usize]
}

/// The shortest run of sorted items a sort merges. A shorter one, as unsorted
/// batches have, is extended by sorting this many items at once with the
/// standard library's sort, which takes 0.2 ms of random keys on an Apple M5 Pro.
/// That costs less than the passes of merging it saves: each step of a merge
/// waits on the comparison before it, which goes either way when keys are
/// unsorted.
const MIN_RUN: usize = 4096;

/// Sorts `items`, operations in the order they were added, by `key`, keeping that
/// order among equal keys, with `scratch` and `run_ends` as scratch space. It
/// finds the runs of items sorted already, extending short ones to `min_run`
/// items, then merges them in pairs, back and forth between `items` and
/// `scratch`, until one is left: sorted items cost one pass, and k sorted runs
/// log2(k) more. It pauses once `budget` is spent, returning false: `s` keeps how
/// far it got.
fn sort<'k>(
    items: &mut Vec<u32>,
    scratch: &mut Vec<u32>,
    run_ends: &mut Vec<u32>,
    s: &mut Sort,
    key: impl Fn(u32) -> &'k [u8],
    min_run: usize,
    budget: &mut Budget,
) -> bool {
    let n = items.len();
    if !s.merging {
        // The run being found starts at `j`, and `i` is the next item past it.
        if s.i == 0 {
            if n < 2 {
                return true;
            }
            run_ends.clear();
            (s.i, s.j) = (1, 0);
        }
        let mut prev = key(items[s.i - 1]);
        while s.i < n {
            if budget.spent() {
                return false;
            }
            budget.work += 1;
            let next = key(items[s.i]);
            if cmp_keys(next, prev).is_ge() {
                prev = next;
                s.i += 1;
                continue;
            }
            // The run ends here. A short one is extended, its equal keys sorted by
            // operation, as they were added.
            if s.i - s.j < min_run {
                s.i = (s.j + min_run).min(n);
                let run = &mut items[s.j..s.i];
                run.sort_unstable_by(|&x, &y| cmp_keys(key(x), key(y)).then(x.cmp(&y)));
                budget.work += run.len() * run.len().ilog2() as usize;
            }
            run_ends.push(s.i as u32);
            s.j = s.i;
            if s.i == n {
                break;
            }
            prev = key(items[s.i]);
            s.i += 1;
        }
        if s.j < n {
            run_ends.push(n as u32);
        }
        if run_ends.len() == 1 {
            return true;
        }
        scratch.resize(n, 0);
        (s.merging, s.i, s.j) = (true, 0, run_ends[0] as usize);
    }
    loop {
        let (src, dst) = if s.flipped {
            (&**scratch, &mut **items)
        } else {
            (&**items, &mut **scratch)
        };
        // Each pair of runs becomes one, and so does a last run without a pair:
        // pair `p` ends where its second run does, which `run_ends[p]` records.
        while 2 * s.pair < run_ends.len() {
            let mid = run_ends[2 * s.pair] as usize;
            let end = run_ends.get(2 * s.pair + 1).map_or(mid, |&e| e as usize);
            let (mut i, mut j) = (s.i, s.j);
            if i < mid && j < end {
                let (mut x, mut y) = (key(src[i]), key(src[j]));
                loop {
                    if budget.spent() {
                        (s.i, s.j) = (i, j);
                        return false;
                    }
                    budget.work += 1;
                    // The first run's keys go first among equal ones.
                    if cmp_keys(y, x).is_lt() {
                        dst[i + j - mid] = src[j];
                        j += 1;
                        if j == end {
                            break;
                        }
                        y = key(src[j]);
                    } else {
                        dst[i + j - mid] = src[i];
                        i += 1;
                        if i == mid {
                            break;
                        }
                        x = key(src[i]);
                    }
                }
            }
            // The rest of the run left over follows.
            let k = i + j - mid;
            dst[k..k + mid - i].copy_from_slice(&src[i..mid]);
            dst[k + mid - i..end].copy_from_slice(&src[j..end]);
            budget.work += end - k;
            run_ends[s.pair] = end as u32;
            s.pair += 1;
            s.i = end;
            s.j = run_ends.get(2 * s.pair).map_or(end, |&e| e as usize);
        }
        run_ends.truncate(s.pair);
        s.flipped = !s.flipped;
        if run_ends.len() == 1 {
            break;
        }
        (s.pair, s.i, s.j) = (0, 0, run_ends[0] as usize);
    }
    if s.flipped {
        mem::swap(items, scratch);
    }
    true
}

/// Compares two keys as byte strings, their first 16 bytes as one number:
/// BookKeeper's keys, which are 16 bytes, compare in one step, without a call to
/// `memcmp`. That makes merging about a fifth faster.
#[inline]
fn cmp_keys(a: &[u8], b: &[u8]) -> Ordering {
    if let (Some(x), Some(y)) = (a.first_chunk::<16>(), b.first_chunk::<16>()) {
        let (x, y) = (u128::from_be_bytes(*x), u128::from_be_bytes(*y));
        if x != y {
            return x.cmp(&y);
        }
        return a[16..].cmp(&b[16..]);
    }
    a.cmp(b)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    /// Prepares `batch` at once.
    fn prepare(batch: &mut WriteBatch) {
        prepare_in_slices(batch, usize::MAX);
    }

    /// Prepares `batch`, pausing after every `slice` units of work.
    fn prepare_in_slices(batch: &mut WriteBatch, slice: usize) {
        let mut budget = Budget::fixed(slice);
        batch.restart();
        loop {
            budget.start();
            if batch.prepare(&mut budget) {
                break;
            }
        }
    }

    /// The points and ranges of `batch` as `prepare` found them before it paused:
    /// at once, with a sort that doesn't keep the order of equal keys, and checking
    /// each key against every range that starts at or before it.
    fn prepared_at_once(batch: &WriteBatch) -> (Vec<u32>, Vec<(u32, u32)>) {
        let (bytes, ops) = (&batch.bytes, &batch.ops);
        let a = |i: u32| {
            let op = &ops[i as usize];
            slice(bytes, op.a, op.a_len)
        };
        let b = |i: u32| {
            let op = &ops[i as usize];
            slice(bytes, op.b, op.b_len)
        };

        let n = ops.len() as u32;
        let mut by_start: Vec<u32> = (0..n)
            .filter(|&i| ops[i as usize].kind == Kind::DeleteRange && a(i) < b(i))
            .collect();
        by_start.sort_unstable_by(|&x, &y| a(x).cmp(a(y)).then(x.cmp(&y)));

        let mut points: Vec<u32> = (0..n)
            .filter(|&i| matches!(ops[i as usize].kind, Kind::Put | Kind::Delete))
            .collect();
        points.sort_unstable_by(|&x, &y| a(x).cmp(a(y)).then(x.cmp(&y)));
        let covered_later = |op: u32| {
            let key = a(op);
            let candidates = by_start.partition_point(|&r| a(r) <= key);
            by_start[..candidates].iter().any(|&r| r > op && key < b(r))
        };
        let mut kept = 0;
        for i in 0..points.len() {
            let op = points[i];
            let last = i + 1 == points.len() || a(points[i + 1]) != a(op);
            if last && !covered_later(op) {
                points[kept] = op;
                kept += 1;
            }
        }
        points.truncate(kept);

        let mut ranges: Vec<(u32, u32)> = Vec::new();
        for &r in &by_start {
            if let Some(last) = ranges.last_mut()
                && a(r) <= b(last.1)
            {
                if b(r) > b(last.1) {
                    last.1 = r;
                }
                continue;
            }
            ranges.push((r, r));
        }
        (points, ranges)
    }

    #[derive(Clone, Debug)]
    enum TestOp {
        Put(Vec<u8>, u8),
        Delete(Vec<u8>),
        DeleteRange(Vec<u8>, Vec<u8>),
    }

    /// Short keys from a few bytes, so that operations share keys, keys are
    /// prefixes of others, and ranges overlap, touch, or are empty.
    fn test_key() -> impl Strategy<Value = Vec<u8>> {
        prop::collection::vec(0..4u8, 0..3)
    }

    fn test_op() -> impl Strategy<Value = TestOp> {
        prop_oneof![
            4 => (test_key(), any::<u8>()).prop_map(|(k, v)| TestOp::Put(k, v)),
            2 => test_key().prop_map(TestOp::Delete),
            1 => (test_key(), test_key()).prop_map(|(s, e)| TestOp::DeleteRange(s, e)),
        ]
    }

    /// Operations on two-byte keys, which few share: ranges span 100 keys.
    fn wide_op() -> impl Strategy<Value = TestOp> {
        (any::<u16>(), 0..16u8).prop_map(|(k, kind)| {
            let key = k.to_be_bytes().to_vec();
            match kind {
                0 => TestOp::Delete(key),
                1 => TestOp::DeleteRange(key, k.saturating_add(100).to_be_bytes().to_vec()),
                _ => TestOp::Put(key, kind),
            }
        })
    }

    /// Operations on 2,048 two-byte keys, a third of them ranges, from one key long
    /// to as long as all of them: short ones end under long ones added after them,
    /// and long ones outlast short ones added after them.
    fn range_op() -> impl Strategy<Value = TestOp> {
        (0..2048u16, 0..12u32, 0..3u8).prop_map(|(k, span, kind)| {
            let key = k.to_be_bytes().to_vec();
            match kind {
                0 => TestOp::DeleteRange(key, (k + (1 << span)).to_be_bytes().to_vec()),
                1 => TestOp::Delete(key),
                _ => TestOp::Put(key, kind),
            }
        })
    }

    /// Batches appended into one, as group commit does, the marked ones then
    /// withdrawn: some in key order, as appends to ledgers are.
    fn test_group() -> impl Strategy<Value = Vec<(Vec<TestOp>, bool, bool)>> {
        let batch = (
            prop::collection::vec(test_op(), 0..60),
            any::<bool>(),
            any::<bool>(),
        );
        prop::collection::vec(batch, 1..5)
    }

    fn group_of(batches: &[(Vec<TestOp>, bool, bool)]) -> WriteBatch {
        let mut group = WriteBatch::new();
        for (ops, sorted, withdrawn) in batches {
            let mut ops = ops.clone();
            if *sorted {
                ops.sort_by(|x, y| test_op_key(x).cmp(test_op_key(y)));
            }
            let mut batch = WriteBatch::new();
            for op in ops {
                match op {
                    TestOp::Put(k, v) => batch.put(&k, &[v]),
                    TestOp::Delete(k) => batch.delete(&k),
                    TestOp::DeleteRange(s, e) => batch.delete_range(&s, &e),
                }
            }
            let span = group.append(&batch);
            if *withdrawn {
                group.withdraw(span);
            }
        }
        group
    }

    fn test_op_key(op: &TestOp) -> &[u8] {
        match op {
            TestOp::Put(k, _) | TestOp::Delete(k) | TestOp::DeleteRange(k, _) => k,
        }
    }

    proptest! {
        // Every slice size pauses the sort, the dedup and the merge of ranges at
        // other points; they still come to what preparing at once by sorting did,
        // down to which operation of a key stays.
        #[test]
        fn prepare_matches_sorting_at_once_at_any_slice(
            batches in test_group(),
            slice in 1..64usize,
        ) {
            let mut group = group_of(&batches);
            let expected = prepared_at_once(&group);
            prepare_in_slices(&mut group, slice);
            prop_assert_eq!((group.points().to_vec(), group.ranges().to_vec()), expected.clone());
            // Prepared again, with the scratch space of the first time, it does the
            // same.
            prepare_in_slices(&mut group, slice + 1);
            prop_assert_eq!((group.points().to_vec(), group.ranges().to_vec()), expected);
        }

        // Large batches, whose sorts merge many runs over many passes.
        #[test]
        fn large_batches_prepare_as_at_once(
            ops in prop::collection::vec(wide_op(), 200..2000),
            slice in prop_oneof![1..16usize, Just(usize::MAX)],
        ) {
            let mut group = group_of(&[(ops, false, false)]);
            let expected = prepared_at_once(&group);
            prepare_in_slices(&mut group, slice);
            prop_assert_eq!((group.points().to_vec(), group.ranges().to_vec()), expected);
        }

        // Groups of many ranges among many points, some in key order: ranges pile
        // up on the heap under long ones, and come off once those end, and the
        // pushes and pops pause too.
        #[test]
        fn many_ranges_among_many_points_prepare_as_at_once(
            batches in prop::collection::vec(
                (prop::collection::vec(range_op(), 0..1000), any::<bool>(), any::<bool>()),
                1..4,
            ),
            slice in prop_oneof![1..16usize, Just(usize::MAX)],
        ) {
            let mut group = group_of(&batches);
            let expected = prepared_at_once(&group);
            prepare_in_slices(&mut group, slice);
            prop_assert_eq!((group.points().to_vec(), group.ranges().to_vec()), expected);
        }

        // Runs sorted already, of any length, and short runs extended to any
        // length: the sort orders them by key, then by operation.
        #[test]
        fn sort_orders_by_key_then_operation(
            runs in prop::collection::vec(
                (prop::collection::vec(prop::collection::vec(0..3u8, 0..20), 0..40), any::<bool>()),
                0..8,
            ),
            min_run in 1..12usize,
            slice in 1..20usize,
        ) {
            let mut batch = WriteBatch::new();
            for (mut keys, sorted) in runs {
                if sorted {
                    keys.sort();
                }
                for key in keys {
                    batch.put(&key, b"");
                }
            }
            let key = |i: u32| batch.key(i);
            let mut items: Vec<u32> = (0..batch.ops.len() as u32).collect();
            let mut expected = items.clone();
            expected.sort_by(|&x, &y| key(x).cmp(key(y)).then(x.cmp(&y)));
            let (mut scratch, mut run_ends, mut s) = (Vec::new(), Vec::new(), Sort::default());
            let mut budget = Budget::fixed(slice);
            loop {
                budget.start();
                if sort(&mut items, &mut scratch, &mut run_ends, &mut s, key, min_run, &mut budget) {
                    break;
                }
            }
            prop_assert_eq!(items, expected);
        }

        // Keys that share a prefix, as long as the part compared as a number or
        // longer, or shorter.
        #[test]
        fn keys_compare_as_byte_strings(
            prefix in prop::collection::vec(0..3u8, 0..20),
            a in prop::collection::vec(0..3u8, 0..6),
            b in prop::collection::vec(0..3u8, 0..6),
        ) {
            let (a, b) = ([&prefix[..], &a].concat(), [&prefix[..], &b].concat());
            prop_assert_eq!(cmp_keys(&a, &b), a.cmp(&b));
            prop_assert_eq!(cmp_keys(&a, &a), Ordering::Equal);
        }
    }

    #[test]
    fn sorted_batches_merge_as_at_once() {
        // Sorted batches longer than a run that the sort extends, appended out of
        // order, with operations out of order after them: their runs merge. The
        // keys are BookKeeper's, a ledger and an entry.
        let key = |i: u32| {
            let mut key = [0; 16];
            key[..8].copy_from_slice(&u64::from(i / 1000).to_be_bytes());
            key[8..].copy_from_slice(&u64::from(i % 1000).to_be_bytes());
            key
        };
        let mut group = WriteBatch::new();
        for first in [2, 0, 3, 1] {
            let mut batch = WriteBatch::new();
            for i in first * 5000..(first + 1) * 5000u32 {
                batch.put(&key(i), &[1]);
            }
            group.append(&batch);
        }
        for i in 0..3000u32 {
            let k = i.wrapping_mul(0x9e37_79b9) % 25_000;
            match i % 3 {
                0 => group.delete(&key(k)),
                1 => group.delete_range(&key(k), &key(k + 50)),
                _ => group.put(&key(k), &[2]),
            }
        }
        let expected = prepared_at_once(&group);
        for slice in [1, 7, 4096, usize::MAX] {
            prepare_in_slices(&mut group, slice);
            assert_eq!((group.points().to_vec(), group.ranges().to_vec()), expected);
        }
    }

    fn points(batch: &WriteBatch) -> Vec<(Vec<u8>, Option<Vec<u8>>)> {
        let point = |&op: &u32| (batch.key(op).to_vec(), batch.value(op).map(<[u8]>::to_vec));
        batch.points().iter().map(point).collect()
    }

    fn ranges(batch: &WriteBatch) -> Vec<(Vec<u8>, Vec<u8>)> {
        let range =
            |&(s, e): &(u32, u32)| (batch.range_start(s).to_vec(), batch.range_end(e).to_vec());
        batch.ranges().iter().map(range).collect()
    }

    #[test]
    fn last_operation_per_key_wins() {
        let mut batch = WriteBatch::new();
        batch.put(b"b", b"1");
        batch.put(b"a", b"1");
        batch.delete(b"b");
        batch.put(b"a", b"2");
        batch.delete(b"c");
        prepare(&mut batch);
        assert_eq!(
            points(&batch),
            vec![
                (b"a".to_vec(), Some(b"2".to_vec())),
                (b"b".to_vec(), None),
                (b"c".to_vec(), None)
            ]
        );
    }

    #[test]
    fn ranges_apply_in_order() {
        let mut batch = WriteBatch::new();
        batch.put(b"a", b"before");
        batch.put(b"b", b"before");
        batch.delete_range(b"a", b"c");
        batch.put(b"b", b"after");
        prepare(&mut batch);
        // "a" was put before the range removed it; "b" was put again after.
        assert_eq!(
            points(&batch),
            vec![(b"b".to_vec(), Some(b"after".to_vec()))]
        );
        assert_eq!(ranges(&batch), vec![(b"a".to_vec(), b"c".to_vec())]);
    }

    #[test]
    fn ranges_merge_and_empty_ones_vanish() {
        let mut batch = WriteBatch::new();
        batch.delete_range(b"m", b"p");
        batch.delete_range(b"a", b"c");
        batch.delete_range(b"b", b"d");
        batch.delete_range(b"d", b"e");
        batch.delete_range(b"z", b"y");
        batch.delete_range(b"n", b"o");
        prepare(&mut batch);
        assert_eq!(
            ranges(&batch),
            vec![
                (b"a".to_vec(), b"e".to_vec()),
                (b"m".to_vec(), b"p".to_vec())
            ]
        );
    }

    #[test]
    fn appended_batches_apply_in_order() {
        let mut first = WriteBatch::new();
        first.put(b"a", b"1");
        first.put(b"b", b"1");
        first.put(b"d", b"1");
        let mut second = WriteBatch::new();
        second.delete_range(b"a", b"c");
        second.put(b"b", b"2");
        let mut third = WriteBatch::new();
        third.put(b"d", b"3");
        third.delete_range(b"x", b"z");
        let mut group = WriteBatch::new();
        assert_eq!(group.append(&first), (0, 3));
        assert_eq!(group.append(&second), (3, 5));
        assert_eq!(group.append(&third), (5, 7));
        assert_eq!(group.size(), first.size() + second.size() + third.size());
        prepare(&mut group);
        // The second batch's range removes the first's "a" and "b", then puts "b"
        // back; the third's put of "d" replaces the first's.
        assert_eq!(
            points(&group),
            vec![
                (b"b".to_vec(), Some(b"2".to_vec())),
                (b"d".to_vec(), Some(b"3".to_vec()))
            ]
        );
        assert_eq!(
            ranges(&group),
            vec![
                (b"a".to_vec(), b"c".to_vec()),
                (b"x".to_vec(), b"z".to_vec())
            ]
        );
        // Without the second batch, the first's puts stand.
        group.withdraw((3, 5));
        prepare(&mut group);
        assert_eq!(
            points(&group),
            vec![
                (b"a".to_vec(), Some(b"1".to_vec())),
                (b"b".to_vec(), Some(b"1".to_vec())),
                (b"d".to_vec(), Some(b"3".to_vec()))
            ]
        );
        assert_eq!(ranges(&group), vec![(b"x".to_vec(), b"z".to_vec())]);
    }

    #[test]
    fn oversized_keys_are_rejected() {
        let mut batch = WriteBatch::new();
        batch.put(&[0; 13], b"");
        assert!(matches!(
            batch.check(12),
            Err(Error::KeyTooLarge { size: 13, max: 12 })
        ));
        // Values, and the keys of deletes and ranges, may be longer.
        batch.clear();
        batch.put(&[0; 12], &[0; 100]);
        batch.delete(&[0; 13]);
        batch.delete_range(&[0; 13], &[1; 13]);
        batch.check(12).unwrap();
    }
}
