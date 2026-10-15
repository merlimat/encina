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

//! Store tests: a `BTreeMap` model, BookKeeper-shaped scenarios, failures injected
//! into a simulated disk, and a real file.

use std::collections::BTreeMap;
use std::future::Future;
use std::ops::Bound;
use std::pin::pin;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

use proptest::prelude::*;

use crate::sim::{Fault, SimDisk};
use crate::{Comparison, Db, Error, Options, WriteBatch};

/// Runs a future that must finish without waiting, as everything does on the
/// simulated disk.
pub(crate) fn now<F: Future>(f: F) -> F::Output {
    match pin!(f).poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(output) => output,
        Poll::Pending => panic!("the future had to wait"),
    }
}

type Model = BTreeMap<Vec<u8>, Vec<u8>>;
type Entry = (Vec<u8>, Vec<u8>);
type Range<'a> = (Bound<&'a [u8]>, Bound<&'a [u8]>);

const ALL: Range<'static> = (Bound::Unbounded, Bound::Unbounded);

/// A store on a simulated disk.
struct Store {
    disk: Rc<SimDisk>,
    db: Db,
    opts: Options,
}

impl Store {
    fn new(opts: Options) -> Store {
        let disk = SimDisk::new();
        let db = now(Db::open_sim(disk.clone(), opts.clone())).unwrap();
        Store { disk, db, opts }
    }

    /// Restarts the machine, and opens the store again.
    fn reopen(&mut self) {
        self.disk = self.disk.restart();
        self.db = now(Db::open_sim(self.disk.clone(), self.opts.clone())).unwrap();
    }
}

#[derive(Clone, Debug)]
enum Op {
    Put(Vec<u8>, Vec<u8>),
    Delete(Vec<u8>),
    DeleteRange(Vec<u8>, Vec<u8>),
}

fn batch_of(ops: &[Op]) -> WriteBatch {
    let mut batch = WriteBatch::new();
    for op in ops {
        match op {
            Op::Put(k, v) => batch.put(k, v),
            Op::Delete(k) => batch.delete(k),
            Op::DeleteRange(start, end) => batch.delete_range(start, end),
        }
    }
    batch
}

fn write(db: &Db, ops: &[Op]) {
    now(db.write(&mut batch_of(ops))).unwrap();
}

fn apply(model: &mut Model, ops: &[Op]) {
    for op in ops {
        match op {
            Op::Put(k, v) => {
                model.insert(k.clone(), v.clone());
            }
            Op::Delete(k) => {
                model.remove(k);
            }
            Op::DeleteRange(start, end) => model.retain(|k, _| k < start || k >= end),
        }
    }
}

fn get(db: &Db, key: &[u8], cmp: Comparison) -> Option<Entry> {
    now(db.get(key, cmp, |k, v| (k.to_vec(), v.to_vec()))).unwrap()
}

fn expected(model: &Model, key: &[u8], cmp: Comparison) -> Option<Entry> {
    use Bound::{Excluded, Included, Unbounded};
    let found = match cmp {
        Comparison::Equal => model.get_key_value(key),
        Comparison::Floor => model
            .range::<[u8], _>((Unbounded, Included(key)))
            .next_back(),
        Comparison::Lower => model
            .range::<[u8], _>((Unbounded, Excluded(key)))
            .next_back(),
        Comparison::Ceiling => model.range::<[u8], _>((Included(key), Unbounded)).next(),
        Comparison::Higher => model.range::<[u8], _>((Excluded(key), Unbounded)).next(),
    };
    found.map(|(k, v)| (k.clone(), v.clone()))
}

fn scan(db: &Db, range: Range) -> Vec<Entry> {
    let mut iter = db.range(range);
    let mut entries = Vec::new();
    while let Some((k, v)) = now(iter.next()).unwrap() {
        entries.push((k.to_vec(), v.to_vec()));
    }
    entries
}

fn expected_scan(model: &Model, (lo, hi): Range) -> Vec<Entry> {
    let above = |k: &[u8]| match lo {
        Bound::Included(lo) => k >= lo,
        Bound::Excluded(lo) => k > lo,
        Bound::Unbounded => true,
    };
    let below = |k: &[u8]| match hi {
        Bound::Included(hi) => k <= hi,
        Bound::Excluded(hi) => k < hi,
        Bound::Unbounded => true,
    };
    model
        .iter()
        .filter(|(k, _)| above(k) && below(k))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

const COMPARISONS: [Comparison; 5] = [
    Comparison::Equal,
    Comparison::Floor,
    Comparison::Ceiling,
    Comparison::Lower,
    Comparison::Higher,
];

#[derive(Clone, Debug)]
enum Step {
    Write(Vec<Op>),
    /// Puts `count` consecutive keys after `prefix`, as when appending to a ledger.
    Load {
        prefix: Vec<u8>,
        first: u16,
        count: u16,
        value_len: usize,
    },
    Get(Vec<u8>, Comparison),
    Scan(Bound<Vec<u8>>, Bound<Vec<u8>>),
    /// Starts a full scan, writes, then finishes the scan, which mustn't see the write.
    WriteDuringScan(Vec<Op>),
    /// Restarts the machine and reopens the store.
    Reopen,
}

impl Step {
    fn ops(&self) -> Vec<Op> {
        match self {
            Step::Write(ops) | Step::WriteDuringScan(ops) => ops.clone(),
            Step::Load {
                prefix,
                first,
                count,
                value_len,
            } => (0..*count)
                .map(|i| {
                    let mut key = prefix.clone();
                    key.extend_from_slice(&first.wrapping_add(i).to_be_bytes());
                    Op::Put(key, vec![i as u8; *value_len])
                })
                .collect(),
            Step::Get(..) | Step::Scan(..) | Step::Reopen => Vec::new(),
        }
    }
}

/// Short keys over a small alphabet: lots of shared prefixes, keys that are
/// prefixes of others, zero bytes, and repeats.
fn key() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(prop::sample::select(&[0u8, 1, 2, 0x7f, 0xff][..]), 0..=8)
}

fn op(max_value: usize) -> impl Strategy<Value = Op> {
    prop_oneof![
        8 => (key(), prop::collection::vec(any::<u8>(), 0..=max_value))
            .prop_map(|(k, v)| Op::Put(k, v)),
        2 => key().prop_map(Op::Delete),
        // Wide ranges, which drop whole subtrees, and narrow ones.
        1 => (key(), key()).prop_map(|(start, end)| Op::DeleteRange(start, end)),
        1 => (key(), any::<u8>()).prop_map(|(start, last)| {
            let mut end = start.clone();
            end.push(last);
            Op::DeleteRange(start, end)
        }),
    ]
}

fn bound() -> impl Strategy<Value = Bound<Vec<u8>>> {
    prop_oneof![
        Just(Bound::Unbounded),
        key().prop_map(Bound::Included),
        key().prop_map(Bound::Excluded),
    ]
}

/// Steps whose values are at most `max_value` bytes.
fn step(max_value: usize) -> impl Strategy<Value = Step> {
    prop_oneof![
        4 => prop::collection::vec(op(max_value), 1..40).prop_map(Step::Write),
        1 => (key(), any::<u16>(), 1..300u16, 0..=max_value)
            .prop_map(|(prefix, first, count, value_len)| Step::Load { prefix, first, count, value_len }),
        3 => (key(), prop::sample::select(COMPARISONS.to_vec())).prop_map(|(k, cmp)| Step::Get(k, cmp)),
        1 => (bound(), bound()).prop_map(|(lo, hi)| Step::Scan(lo, hi)),
        1 => prop::collection::vec(op(max_value), 1..40).prop_map(Step::WriteDuringScan),
        1 => Just(Step::Reopen),
    ]
}

fn run(opts: Options, steps: &[Step]) {
    let mut store = Store::new(opts);
    let mut model = Model::new();
    for step in steps {
        match step {
            Step::Write(_) | Step::Load { .. } => {
                let ops = step.ops();
                write(&store.db, &ops);
                apply(&mut model, &ops);
                store.db.tree().check();
            }
            Step::Get(key, cmp) => {
                let got = get(&store.db, key, *cmp);
                assert_eq!(got, expected(&model, key, *cmp), "get({key:?}, {cmp:?})");
            }
            Step::Scan(lo, hi) => {
                let range = (
                    lo.as_ref().map(Vec::as_slice),
                    hi.as_ref().map(Vec::as_slice),
                );
                assert_eq!(
                    scan(&store.db, range),
                    expected_scan(&model, range),
                    "scan {range:?}"
                );
            }
            Step::WriteDuringScan(ops) => {
                let before = expected_scan(&model, ALL);
                let mut iter = store.db.range(..);
                let mut seen = Vec::new();
                if let Some((k, v)) = now(iter.next()).unwrap() {
                    seen.push((k.to_vec(), v.to_vec()));
                }
                write(&store.db, ops);
                apply(&mut model, ops);
                while let Some((k, v)) = now(iter.next()).unwrap() {
                    seen.push((k.to_vec(), v.to_vec()));
                }
                assert_eq!(seen, before, "the scan saw a later write");
                drop(iter);
                store.db.tree().check();
            }
            Step::Reopen => {
                store.reopen();
                store.db.tree().check();
            }
        }
    }
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    // Every comparison, on every key and on the gap just above it, which covers the
    // first and last keys of every leaf.
    for key in model.keys() {
        let mut above = key.clone();
        above.push(0);
        for probe in [key, &above] {
            for cmp in COMPARISONS {
                assert_eq!(
                    get(&store.db, probe, cmp),
                    expected(&model, probe, cmp),
                    "get({probe:?}, {cmp:?})"
                );
            }
        }
    }
    store.db.tree().check();
}

// Set PROPTEST_CASES to run more than the default 256 cases.
proptest! {
    // Small pages grow deep trees from few entries.
    #[test]
    fn matches_the_model_with_small_pages(steps in prop::collection::vec(step(12), 1..40)) {
        run(Options::new().page_size(256), &steps);
    }

    // A cache of 16 pages keeps evicting leaves and reading them back.
    #[test]
    fn matches_the_model_with_a_tiny_cache(steps in prop::collection::vec(step(12), 1..40)) {
        run(Options::new().page_size(256).cache_size(16 * 256), &steps);
    }

    // Large values fill default pages quickly enough to grow internal levels.
    #[test]
    fn matches_the_model_with_default_pages(steps in prop::collection::vec(step(200), 1..40)) {
        run(Options::new(), &steps);
    }
}

fn entry_key(ledger: u64, entry: u64) -> [u8; 16] {
    let mut key = [0; 16];
    key[..8].copy_from_slice(&ledger.to_be_bytes());
    key[8..].copy_from_slice(&entry.to_be_bytes());
    key
}

fn location(ledger: u64, entry: u64) -> [u8; 16] {
    let mut value = [0; 16];
    value[..8].copy_from_slice(&(ledger % 7).to_be_bytes());
    value[8..].copy_from_slice(&(entry * 1000).to_be_bytes());
    value
}

fn decode_key(key: &[u8]) -> (u64, u64) {
    let ledger = u64::from_be_bytes(key[..8].try_into().unwrap());
    (ledger, u64::from_be_bytes(key[8..].try_into().unwrap()))
}

#[test]
fn bookkeeper_workload() {
    let mut store = Store::new(Options::default());
    let mut batch = WriteBatch::new();
    let ledgers = 64;
    let mut next_entry = vec![0; ledgers as usize];
    for round in 0..30 {
        // Each batch appends to most ledgers.
        batch.clear();
        for ledger in (0..ledgers).filter(|l| (l + round) % 5 != 0) {
            for _ in 0..40 {
                let entry = &mut next_entry[ledger as usize];
                batch.put(&entry_key(1000 + ledger, *entry), &location(ledger, *entry));
                *entry += 1;
            }
        }
        now(store.db.write(&mut batch)).unwrap();
        store.db.tree().check();
    }
    let (leaves, used) = store.db.tree().leaf_usage();
    let fill = used as f64 / (leaves * store.db.tree().page_size()) as f64;
    assert!(fill > 0.8, "leaves are only {:.0}% full", fill * 100.0);

    // Deleting ledgers removes their entries and nothing else.
    batch.clear();
    for ledger in (0..ledgers).step_by(2) {
        batch.delete_range(&entry_key(1000 + ledger, 0), &entry_key(1001 + ledger, 0));
    }
    now(store.db.write(&mut batch)).unwrap();
    store.db.tree().check();
    store.reopen();
    store.db.tree().check();
    for ledger in 0..ledgers {
        let (id, entries) = (1000 + ledger, next_entry[ledger as usize]);
        let (start, end) = (entry_key(id, 0), entry_key(id + 1, 0));
        let found = scan(&store.db, (Bound::Included(&start), Bound::Excluded(&end))).len() as u64;
        let last = now(store
            .db
            .get(&entry_key(id, u64::MAX), Comparison::Floor, |k, _| {
                decode_key(k)
            }));
        if ledger % 2 == 0 {
            assert_eq!(found, 0);
            continue;
        }
        assert_eq!(found, entries);
        assert_eq!(last.unwrap(), Some((id, entries - 1)));
        for entry in (0..entries).step_by(7) {
            let found = now(store
                .db
                .get(&entry_key(id, entry), Comparison::Equal, |_, v| v.to_vec()));
            assert_eq!(found.unwrap(), Some(location(ledger, entry).to_vec()));
        }
    }
}

/// The benchmark's load, scaled down: each batch appends 100 entries to each of 100
/// ledgers, which close every 10 batches. Rebuilt leaves leave small pieces behind,
/// which merge with their neighbors before they are built.
#[test]
fn writes_build_no_node_they_discard() {
    let store = Store::new(Options::default());
    let mut batch = WriteBatch::new();
    for b in 0..20 {
        batch.clear();
        for slot in 0..100 {
            let ledger = b / 10 * 100 + slot;
            for entry in b % 10 * 100..(b % 10 + 1) * 100 {
                batch.put(&entry_key(ledger, entry), &location(ledger, entry));
            }
        }
        now(store.db.write(&mut batch)).unwrap();
        let freed = store.db.tree().pages.space.freed;
        assert_eq!(freed, 0, "batch {b} built {freed} pages it freed again");
        store.db.tree().check();
    }
}

#[test]
fn comparisons_at_the_edges() {
    use Comparison::{Ceiling, Equal, Floor, Higher, Lower};
    let store = Store::new(Options::default());
    for cmp in COMPARISONS {
        assert_eq!(get(&store.db, b"x", cmp), None);
    }
    write(
        &store.db,
        &[10u8, 20, 30].map(|k| Op::Put(vec![k], vec![k / 10])),
    );
    let e = |k: u8| Some((vec![k], vec![k / 10]));
    let cases = [
        (5, [None, None, e(10), None, e(10)]),
        (10, [e(10), e(10), e(10), None, e(20)]),
        (15, [None, e(10), e(20), e(10), e(20)]),
        (30, [e(30), e(30), e(30), e(20), None]),
        (35, [None, e(30), None, e(30), None]),
    ];
    for (key, want) in cases {
        for (cmp, want) in [Equal, Floor, Ceiling, Lower, Higher].into_iter().zip(want) {
            assert_eq!(get(&store.db, &[key], cmp), want, "get({key}, {cmp:?})");
        }
    }
}

#[test]
fn oversized_entries_leave_the_store_unchanged() {
    let store = Store::new(Options::default());
    let max = Options::default().max_entry_size();
    let mut batch = WriteBatch::new();
    batch.put(b"a", b"1");
    batch.put(b"big", &vec![0; max - 2]);
    assert!(matches!(
        now(store.db.write(&mut batch)),
        Err(Error::EntryTooLarge { .. })
    ));
    assert_eq!(get(&store.db, b"a", Comparison::Equal), None);
    batch.clear();
    batch.put(b"big", &vec![0; max - 3]);
    now(store.db.write(&mut batch)).unwrap();
    assert!(get(&store.db, b"big", Comparison::Equal).is_some());
}

#[test]
fn iterators_keep_their_version() {
    let store = Store::new(Options::new().page_size(256));
    let puts: Vec<_> = (0..200u16)
        .map(|i| Op::Put(i.to_be_bytes().to_vec(), vec![1]))
        .collect();
    write(&store.db, &puts);
    let mut iter = store.db.range(..);
    write(
        &store.db,
        &[
            Op::DeleteRange(vec![0, 50], vec![0, 150]),
            Op::Put(vec![0, 7], vec![2]),
        ],
    );
    let mut seen = 0;
    while let Some((_, value)) = now(iter.next()).unwrap() {
        assert_eq!(value, [1]);
        seen += 1;
    }
    assert_eq!(seen, 200);
    drop(iter);
    // Dropping the iterator freed the old version.
    store.db.tree().check();
    assert_eq!(scan(&store.db, ALL).len(), 100);
}

#[test]
fn deleting_everything_empties_the_tree() {
    let mut store = Store::new(Options::new().page_size(256));
    let puts: Vec<_> = (0..500u16)
        .map(|i| Op::Put(i.to_be_bytes().to_vec(), vec![0; 8]))
        .collect();
    write(&store.db, &puts);
    write(
        &store.db,
        &[Op::DeleteRange(vec![], vec![0xff, 0xff, 0xff])],
    );
    store.db.tree().check();
    assert!(store.db.tree().current().root.is_none());
    store.reopen();
    assert!(scan(&store.db, ALL).is_empty());
}

#[test]
fn reopening_with_a_cold_cache_reads_everything_back() {
    let mut store = Store::new(Options::new().page_size(256).cache_size(16 * 256));
    let mut model = Model::new();
    for round in 0..5u32 {
        let ops: Vec<_> = (0..400u32)
            .map(|i| {
                Op::Put(
                    (i * 7 + round).to_be_bytes().to_vec(),
                    round.to_be_bytes().to_vec(),
                )
            })
            .collect();
        write(&store.db, &ops);
        apply(&mut model, &ops);
    }
    store.reopen();
    store.db.tree().check();
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    for key in model.keys().step_by(13) {
        for cmp in COMPARISONS {
            assert_eq!(get(&store.db, key, cmp), expected(&model, key, cmp));
        }
    }
    store.db.tree().check();
}

#[test]
fn damaged_pages_are_reported() {
    let mut store = Store::new(Options::new().page_size(256).cache_size(16 * 256));
    let puts: Vec<_> = (0..300u32)
        .map(|i| Op::Put(i.to_be_bytes().to_vec(), vec![7; 8]))
        .collect();
    write(&store.db, &puts);
    let (root, leaf) = {
        let tree = store.db.tree();
        let root = tree.current().root.unwrap();
        assert!(root.level > 0);
        (
            root.page,
            tree.descend_first(root, &mut crate::tree::Cursor::new()),
        )
    };
    // A damaged leaf fails the reads that need it.
    store.disk.flip(leaf * 256 + 100);
    store.reopen();
    let mut iter = store.db.range(..);
    assert!(matches!(now(iter.next()), Err(Error::Corruption { page }) if page == leaf));
    drop(iter);
    // A damaged internal node fails the open.
    store.disk.flip(root * 256 + 100);
    let opened = now(Db::open_sim(store.disk.restart(), store.opts.clone()));
    assert!(matches!(opened, Err(Error::Corruption { page }) if page == root));
}

/// Batches that build a few levels on 256-byte pages, then rewrite and delete
/// parts of them.
fn failure_scenario() -> Vec<Vec<Op>> {
    let key = |i: u32| i.to_be_bytes().to_vec();
    vec![
        (0..300).map(|i| Op::Put(key(i), vec![1; 8])).collect(),
        (100..200).map(|i| Op::Put(key(i), vec![2; 8])).collect(),
        vec![
            Op::DeleteRange(key(50), key(250)),
            Op::Put(key(120), vec![3]),
        ],
        (0..300)
            .step_by(7)
            .map(|i| Op::Put(key(i), vec![4; 4]))
            .collect(),
    ]
}

/// Runs the scenario until an operation fails. Returns the store, if it opened,
/// and how many batches were written.
fn try_scenario(disk: &Rc<SimDisk>, opts: &Options, batches: &[Vec<Op>]) -> (Option<Db>, usize) {
    let Ok(db) = now(Db::open_sim(disk.clone(), opts.clone())) else {
        return (None, 0);
    };
    for (done, ops) in batches.iter().enumerate() {
        if now(db.write(&mut batch_of(ops))).is_err() {
            return (Some(db), done);
        }
        // Read too, so that reads meet failures as well.
        if now(db.get(&[0, 0, 0, 0x7f], Comparison::Ceiling, |_, _| ())).is_err() {
            return (Some(db), done + 1);
        }
    }
    (Some(db), batches.len())
}

/// Creating a store takes three operations: two meta slots and a sync.
const CREATE_OPS: u64 = 3;

#[test]
fn every_single_failure_leaves_a_recoverable_store() {
    // A small cache, so that writes read leaves back from the disk.
    let opts = Options::new().page_size(256).cache_size(16 * 256);
    let batches = failure_scenario();
    let mut states = vec![Vec::new()];
    let mut model = Model::new();
    for ops in &batches {
        apply(&mut model, ops);
        states.push(expected_scan(&model, ALL));
    }
    let clean = SimDisk::new();
    let (db, done) = try_scenario(&clean, &opts, &batches);
    assert_eq!(done, batches.len());
    drop(db);
    let total = clean.ops();
    for fault in [Fault::Error, Fault::Crash] {
        for op in 0..total {
            let disk = SimDisk::new();
            disk.inject(op, fault);
            let (db, done) = try_scenario(&disk, &opts, &batches);
            let failed = db.as_ref().is_some_and(Db::has_failed);
            if let Some(db) = &db {
                if failed {
                    assert!(matches!(
                        now(db.write(&mut WriteBatch::new())),
                        Err(Error::Failed)
                    ));
                } else {
                    // A write that failed before committing left the store as it was.
                    db.tree().check();
                }
            }
            drop(db);
            let reopened = match now(Db::open_sim(disk.restart(), opts.clone())) {
                Ok(db) => db,
                // A failure while creating the store may leave no store at all.
                Err(Error::InvalidFile(_)) if op < CREATE_OPS => continue,
                Err(e) => panic!("{fault:?} at operation {op}: reopening failed: {e}"),
            };
            reopened.tree().check();
            let state = scan(&reopened, ALL);
            // After a failure during a commit, either outcome is valid.
            let ok = state == states[done] || (failed && states.get(done + 1) == Some(&state));
            assert!(
                ok,
                "{fault:?} at operation {op}: {} entries after {done} batches",
                state.len()
            );
        }
    }
}

/// The write requests that writing `pages`, sorted, takes: one per run of adjacent
/// pages, split where a run outgrows a request.
fn requests_for(pages: &[u64], max_run: usize) -> u64 {
    let mut requests = 0;
    let mut len = 0;
    for (i, &page) in pages.iter().enumerate() {
        if i == 0 || page != pages[i - 1] + 1 || len == max_run {
            requests += 1;
            len = 0;
        }
        len += 1;
    }
    requests
}

#[test]
fn commits_write_each_run_of_pages_with_one_request() {
    let store = Store::new(Options::new().page_size(256));
    let max_run = crate::db::max_run(256);
    let key = |i: u32| i.to_be_bytes().to_vec();
    let batches: [Vec<Op>; 4] = [
        // More pages than one request takes.
        (0..30_000).map(|i| Op::Put(key(i), vec![1; 8])).collect(),
        // Scattered updates, into the space the first batch left.
        (0..30_000)
            .step_by(37)
            .map(|i| Op::Put(key(i), vec![2; 8]))
            .collect(),
        // Appends, and a range that frees a stretch of pages.
        (30_000..32_000)
            .map(|i| Op::Put(key(i), vec![3; 8]))
            .chain([Op::DeleteRange(key(1_000), key(9_000))])
            .collect(),
        // Updates everywhere, which reuse that stretch.
        (0..32_000)
            .step_by(5)
            .map(|i| Op::Put(key(i), vec![4; 8]))
            .collect(),
    ];
    for (n, ops) in batches.iter().enumerate() {
        let before = store.db.tree().pages();
        let ops_before = store.disk.ops();
        write(&store.db, ops);
        let written: Vec<_> = store
            .db
            .tree()
            .pages()
            .into_iter()
            .filter(|page| before.binary_search(page).is_err())
            .collect();
        if n == 0 {
            assert!(written.len() > max_run);
        }
        let requests = requests_for(&written, max_run);
        // The page writes, then a sync, the meta page and a sync.
        assert_eq!(store.disk.ops() - ops_before, requests + 3, "batch {n}");
        assert!(
            4 * requests < written.len() as u64,
            "batch {n}: {} pages took {requests} requests",
            written.len()
        );
        store.db.tree().check();
    }
}

#[test]
fn a_failed_page_write_leaves_its_pages_to_the_next_write() {
    // A store whose write fails once, and a twin that takes the same writes.
    let opts = Options::new().page_size(256);
    let (mut store, twin) = (Store::new(opts.clone()), Store::new(opts));
    let mut model = Model::new();
    let key = |i: u32| i.to_be_bytes().to_vec();
    for round in 0..6u32 {
        let ops: Vec<_> = (0..3_000)
            .map(|i| Op::Put(key(i * 3 + round % 3), vec![round as u8; 8]))
            .collect();
        let ops_before = twin.disk.ops();
        write(&twin.db, &ops);
        if round == 3 {
            // The commit's last page write fails, after the others succeeded.
            let writes = twin.disk.ops() - ops_before - 3;
            assert!(writes > 1, "the commit takes {writes} page writes");
            store
                .disk
                .inject(store.disk.ops() + writes - 1, Fault::Error);
            let res = now(store.db.write(&mut batch_of(&ops)));
            assert!(matches!(res, Err(Error::Io(_))));
            assert!(!store.db.has_failed());
            store.db.tree().check();
        }
        write(&store.db, &ops);
        apply(&mut model, &ops);
        // The retry used the very pages the failed write gave back.
        assert_eq!(store.db.tree().pages(), twin.db.tree().pages());
    }
    store.reopen();
    store.db.tree().check();
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
}

#[compio::test]
async fn a_file_store_survives_reopening() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("encina.db");
    let key = |i: u32| i.to_be_bytes();
    let db = Db::open(&path, Options::default()).await.unwrap();
    let mut batch = WriteBatch::new();
    for i in 0..20_000u32 {
        batch.put(&key(i), &key(i * 2));
    }
    db.write(&mut batch).await.unwrap();
    batch.clear();
    batch.delete_range(&key(5_000), &key(15_000));
    db.write(&mut batch).await.unwrap();
    db.close().await.unwrap();

    // The page size comes from the file. A small cache sends reads to the disk.
    let opts = Options::default().page_size(65536).cache_size(16 * 4096);
    let db = Db::open(&path, opts).await.unwrap();
    for i in (0..20_000u32).step_by(97) {
        let found = db
            .get(&key(i), Comparison::Equal, |_, v| v.to_vec())
            .await
            .unwrap();
        let expected = (!(5_000..15_000).contains(&i)).then(|| key(i * 2).to_vec());
        assert_eq!(found, expected, "key {i}");
    }
    let mut iter = db.range(..);
    let mut count = 0;
    while iter.next().await.unwrap().is_some() {
        count += 1;
    }
    assert_eq!(count, 10_000);
    drop(iter);
    db.close().await.unwrap();
}
