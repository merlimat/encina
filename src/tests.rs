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

use std::cell::Cell;
use std::collections::BTreeMap;
use std::future::Future;
use std::ops::Bound;
use std::pin::{Pin, pin};
use std::ptr;
use std::rc::Rc;
use std::sync::{Arc, LazyLock, Weak};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::time::Duration;

use opentelemetry::KeyValue;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, ResourceMetrics};
use opentelemetry_sdk::metrics::reader::MetricReader;
use opentelemetry_sdk::metrics::{
    InstrumentKind, ManualReader, Pipeline, SdkMeterProvider, Temporality,
};
use proptest::prelude::*;

use crate::meta::Meta;
use crate::node::{self, Node};
use crate::pool::Holds;
use crate::sim::{Fault, SimDisk};
use crate::{Comparison, Db, Error, Options, ReadOnlyDb, WriteBatch};

/// Runs a future that must finish without waiting, as everything does on the
/// simulated disk. It may yield, as a write does while it merges.
pub(crate) fn now<F: Future>(f: F) -> F::Output {
    match poll(pin!(f)) {
        Poll::Ready(output) => output,
        Poll::Pending => panic!("the future had to wait"),
    }
}

thread_local! {
    /// Whether [`woken_waker`] was woken.
    static WOKEN: Cell<bool> = const { Cell::new(false) };
}

/// A waker that notes on this thread that it was woken, without allocating.
fn woken_waker() -> Waker {
    const VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake, drop);
    fn clone(_: *const ()) -> RawWaker {
        RawWaker::new(ptr::null(), &VTABLE)
    }
    fn wake(_: *const ()) {
        WOKEN.set(true);
    }
    fn drop(_: *const ()) {}
    // SAFETY: the functions ignore the data pointer, and do what wakers may.
    unsafe { Waker::from_raw(RawWaker::new(ptr::null(), &VTABLE)) }
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
    /// Writes that start together: the first waits for its sync, and the others
    /// queue up behind it. Those marked are dropped while they wait.
    Group(Vec<(Vec<Op>, bool)>),
    /// A write polled at most this many times, each poll stopping where it yields,
    /// with a full scan after each, which mustn't see it. Then it is finished if
    /// marked, and dropped otherwise: if it paused, it applies nothing.
    Paused(Vec<Op>, u8, bool),
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
            Step::Get(..) | Step::Scan(..) | Step::Group(_) | Step::Paused(..) | Step::Reopen => {
                Vec::new()
            }
        }
    }
}

/// Short keys over a small alphabet: lots of shared prefixes, keys that are
/// prefixes of others, zero bytes, and repeats.
fn key() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(prop::sample::select(&[0u8, 1, 2, 0x7f, 0xff][..]), 0..=8)
}

fn op(value_len: impl Strategy<Value = usize>) -> impl Strategy<Value = Op> {
    let value = value_len.prop_flat_map(|n| prop::collection::vec(any::<u8>(), n));
    prop_oneof![
        8 => (key(), value).prop_map(|(k, v)| Op::Put(k, v)),
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

/// Steps whose values have lengths from `value_len`.
fn step(value_len: impl Strategy<Value = usize> + Clone) -> impl Strategy<Value = Step> {
    prop_oneof![
        4 => prop::collection::vec(op(value_len.clone()), 1..40).prop_map(Step::Write),
        1 => (key(), any::<u16>(), 1..300u16, value_len.clone())
            .prop_map(|(prefix, first, count, value_len)| Step::Load { prefix, first, count, value_len }),
        3 => (key(), prop::sample::select(COMPARISONS.to_vec())).prop_map(|(k, cmp)| Step::Get(k, cmp)),
        1 => (bound(), bound()).prop_map(|(lo, hi)| Step::Scan(lo, hi)),
        1 => prop::collection::vec(op(value_len.clone()), 1..40).prop_map(Step::WriteDuringScan),
        1 => prop::collection::vec(
            (prop::collection::vec(op(value_len.clone()), 1..20), prop::bool::weighted(0.25)),
            2..6,
        )
        .prop_map(Step::Group),
        1 => (prop::collection::vec(op(value_len), 1..40), 1..4u8, any::<bool>())
            .prop_map(|(ops, polls, finish)| Step::Paused(ops, polls, finish)),
        1 => Just(Step::Reopen),
    ]
}

/// Value lengths on both sides of what a leaf on 256-byte pages holds, so that
/// entries get value leaves of one page or several, among ordinary leaves.
fn value_len_across_the_limit() -> impl Strategy<Value = usize> + Clone {
    prop_oneof![3 => 0..=12usize, 1 => 20..=40usize, 2 => 41..=700usize]
}

/// Keys of one length, as dense nodes need.
fn dense_key() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(prop::sample::select(&[0u8, 1, 2, 0x7f, 0xff][..]), 4)
}

/// Puts and deletes of keys of one length, with values of one length that often
/// share a prefix, so that nodes are dense. Ranges have bounds of any length.
fn dense_op() -> impl Strategy<Value = Op> {
    let value = prop::collection::vec(prop::sample::select(&[0u8, 0, 1, 0xff][..]), 3);
    prop_oneof![
        8 => (dense_key(), value).prop_map(|(k, v)| Op::Put(k, v)),
        2 => dense_key().prop_map(Op::Delete),
        1 => (key(), key()).prop_map(|(start, end)| Op::DeleteRange(start, end)),
    ]
}

/// Steps that keep every key and every value at one length.
fn dense_step() -> impl Strategy<Value = Step> {
    let prefix = prop::collection::vec(prop::sample::select(&[0u8, 1, 2, 0x7f, 0xff][..]), 2);
    prop_oneof![
        4 => prop::collection::vec(dense_op(), 1..40).prop_map(Step::Write),
        // Two-byte prefixes and two-byte counters make four-byte keys.
        1 => (prefix, any::<u16>(), 1..300u16)
            .prop_map(|(prefix, first, count)| Step::Load { prefix, first, count, value_len: 3 }),
        3 => (key(), prop::sample::select(COMPARISONS.to_vec())).prop_map(|(k, cmp)| Step::Get(k, cmp)),
        1 => (bound(), bound()).prop_map(|(lo, hi)| Step::Scan(lo, hi)),
        1 => prop::collection::vec(dense_op(), 1..40).prop_map(Step::WriteDuringScan),
        1 => Just(Step::Reopen),
    ]
}

fn run(opts: Options, steps: &[Step]) {
    run_sliced(opts, None, steps);
}

/// Runs `steps` on a store whose writes pause after every `slice` of work, if set.
fn run_sliced(opts: Options, slice: Option<usize>, steps: &[Step]) {
    let mut store = Store::new(opts);
    store.db.set_merge_slice(slice);
    let mut model = Model::new();
    for step in steps {
        match step {
            Step::Write(_) | Step::Load { .. } => {
                let ops = step.ops();
                write(&store.db, &ops);
                apply(&mut model, &ops);
                check_tree(&store.db, &model);
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
                check_tree(&store.db, &model);
            }
            Step::Group(writes) => {
                let applied = write_group(&store, writes);
                for ((ops, _), applied) in writes.iter().zip(applied) {
                    if applied {
                        apply(&mut model, ops);
                    }
                }
                check_tree(&store.db, &model);
            }
            Step::Paused(ops, polls, finish) => {
                let before = expected_scan(&model, ALL);
                let mut after = model.clone();
                apply(&mut after, ops);
                let after = expected_scan(&after, ALL);
                let start = generation(&store.db);
                let mut batch = batch_of(ops);
                let mut write = Box::pin(store.db.write(&mut batch));
                let mut finished = false;
                for _ in 0..*polls {
                    if let Poll::Ready(result) = poll_once(write.as_mut()) {
                        result.unwrap();
                        finished = true;
                        break;
                    }
                    // Once published, the write pauses only to free the pages it
                    // replaced.
                    let published = generation(&store.db) > start;
                    assert_eq!(
                        &scan(&store.db, ALL),
                        if published { &after } else { &before },
                        "a scan saw a write in progress"
                    );
                }
                if *finish && !finished {
                    now(write).unwrap();
                    finished = true;
                } else {
                    drop(write);
                }
                if finished || generation(&store.db) > start {
                    apply(&mut model, ops);
                }
                check_tree(&store.db, &model);
            }
            Step::Reopen => {
                store.reopen();
                store.db.set_merge_slice(slice);
                check_tree(&store.db, &model);
            }
        }
    }
    check_reads(&store.db, &model);
}

/// Checks the tree, and that it counts the entries of `model`.
fn check_tree(db: &Db, model: &Model) {
    db.check();
    let bytes = model.iter().map(|(k, v)| (k.len() + v.len()) as u64).sum();
    let size = node::Size {
        entries: model.len() as u64,
        bytes,
    };
    assert_eq!(db.tree().size(), size, "entries miscounted");
}

/// Checks a full scan, and every comparison on every key and on the gap just above
/// it, which covers the first and last keys of every leaf, against `model`.
fn check_reads(db: &Db, model: &Model) {
    assert_eq!(scan(db, ALL), expected_scan(model, ALL));
    for key in model.keys() {
        let mut above = key.clone();
        above.push(0);
        for probe in [key, &above] {
            for cmp in COMPARISONS {
                assert_eq!(
                    get(db, probe, cmp),
                    expected(model, probe, cmp),
                    "get({probe:?}, {cmp:?})"
                );
            }
        }
    }
    db.check();
}

/// Starts `writes` together: the first to wait waits for its sync, and the others
/// queue up behind it. Drops those marked that queued, then lets the syncs go and
/// finishes the rest. Returns which writes applied.
fn write_group(store: &Store, writes: &[(Vec<Op>, bool)]) -> Vec<bool> {
    let mut batches: Vec<_> = writes.iter().map(|(ops, _)| batch_of(ops)).collect();
    store.disk.hold_syncs();
    let mut futures: Vec<_> = batches
        .iter_mut()
        .map(|batch| Some(Box::pin(store.db.write(batch))))
        .collect();
    let mut applied = vec![true; writes.len()];
    let mut committing = false;
    for (i, future) in futures.iter_mut().enumerate() {
        match poll(future.as_mut().unwrap().as_mut()) {
            // A batch that changes nothing commits nothing.
            Poll::Ready(result) => {
                result.unwrap();
                *future = None;
            }
            Poll::Pending if !committing => committing = true,
            Poll::Pending if writes[i].1 => {
                *future = None;
                applied[i] = false;
            }
            Poll::Pending => {}
        }
    }
    store.disk.release_syncs();
    for future in futures.into_iter().flatten() {
        now(future).unwrap();
    }
    applied
}

// Set PROPTEST_CASES to run more than the default 256 cases.
proptest! {
    // Small pages grow deep trees from few entries.
    #[test]
    fn matches_the_model_with_small_pages(steps in prop::collection::vec(step(0..=12usize), 1..40)) {
        run(Options::new().page_size(256), &steps);
    }

    // A cache of 16 pages keeps evicting leaves and reading them back.
    #[test]
    fn matches_the_model_with_a_tiny_cache(steps in prop::collection::vec(step(0..=12usize), 1..40)) {
        run(Options::new().page_size(256).cache_size(16 * 256), &steps);
    }

    // Large values fill default pages quickly enough to grow internal levels.
    #[test]
    fn matches_the_model_with_default_pages(steps in prop::collection::vec(step(0..=200usize), 1..40)) {
        run(Options::new(), &steps);
    }

    // Entries too large for a leaf get value leaves, which writes never read.
    #[test]
    fn matches_the_model_with_value_leaves(
        steps in prop::collection::vec(step(value_len_across_the_limit()), 1..40),
    ) {
        run(Options::new().page_size(256), &steps);
    }

    #[test]
    fn matches_the_model_with_value_leaves_and_a_tiny_cache(
        steps in prop::collection::vec(step(value_len_across_the_limit()), 1..40),
    ) {
        run(Options::new().page_size(256).cache_size(16 * 256), &steps);
    }

    // Keys and values of one length each make every node dense.
    #[test]
    fn matches_the_model_with_dense_nodes(steps in prop::collection::vec(dense_step(), 1..40)) {
        run(Options::new().page_size(256), &steps);
    }

    #[test]
    fn matches_the_model_with_dense_nodes_and_a_tiny_cache(
        steps in prop::collection::vec(dense_step(), 1..40),
    ) {
        run(Options::new().page_size(256).cache_size(16 * 256), &steps);
    }

    // Merges that pause after every few entries, and writes dropped while theirs
    // pauses, among entries with value leaves and without.
    #[test]
    fn matches_the_model_when_merges_pause(
        steps in prop::collection::vec(step(value_len_across_the_limit()), 1..40),
        slice in 1..100usize,
    ) {
        run_sliced(Options::new().page_size(256), Some(slice), &steps);
    }

    #[test]
    fn matches_the_model_when_merges_pause_with_a_tiny_cache(
        steps in prop::collection::vec(step(value_len_across_the_limit()), 1..40),
        slice in 1..100usize,
    ) {
        run_sliced(Options::new().page_size(256).cache_size(16 * 256), Some(slice), &steps);
    }

    #[test]
    fn merges_that_pause_build_the_same_trees(
        steps in prop::collection::vec(step(value_len_across_the_limit()), 1..40),
        slice in 1..100usize,
    ) {
        same_trees(Options::new().page_size(256), slice, &steps);
    }

    #[test]
    fn plans_that_pause_list_the_same_leaves(
        steps in prop::collection::vec(step(value_len_across_the_limit()), 1..40),
        ops in prop::collection::vec(op(0..=12usize), 1..60),
        slice in 1..100usize,
    ) {
        same_plans(&steps, &ops, slice);
    }

    #[test]
    fn merges_that_pause_build_the_same_dense_trees(
        steps in prop::collection::vec(dense_step(), 1..40),
        slice in 1..100usize,
    ) {
        same_trees(Options::new().page_size(256), slice, &steps);
    }
}

/// Writes `steps` to a store of four levels, then plans a write of `ops`, at once
/// and pausing after every `slice` of work: both list the leaves that planning
/// listed before it paused.
fn same_plans(steps: &[Step], ops: &[Op], slice: usize) {
    let store = Store::new(Options::new().page_size(256));
    let load: Vec<_> = (0..1500u32)
        .map(|i| Op::Put(i.to_be_bytes().to_vec(), vec![1; 40]))
        .collect();
    write(&store.db, &load);
    assert_eq!(store.db.tree().current().root.unwrap().level, 3);
    for ops in steps.iter().map(Step::ops).filter(|ops| !ops.is_empty()) {
        write(&store.db, &ops);
    }
    let mut tree = store.db.tree();
    let mut batch = batch_of(ops);
    for slice in [usize::MAX, slice] {
        tree.set_slice(Some(slice));
        while tree.plan(&mut batch).is_pending() {}
        let expected = tree.plan_at_once(&batch);
        assert_eq!(tree.touched, expected);
    }
}

/// Applies the writes of `steps` to two stores, one of whose merges pause after
/// every `slice` of work: both build the same trees, on the same pages.
fn same_trees(opts: Options, slice: usize, steps: &[Step]) {
    let (store, paused) = (Store::new(opts.clone()), Store::new(opts));
    paused.db.set_merge_slice(Some(slice));
    for ops in steps.iter().map(Step::ops).filter(|ops| !ops.is_empty()) {
        write(&store.db, &ops);
        write(&paused.db, &ops);
        assert_eq!(store.db.tree().leaves(), paused.db.tree().leaves());
        assert_eq!(store.db.tree().pages(), paused.db.tree().pages());
    }
    assert_eq!(scan(&store.db, ALL), scan(&paused.db, ALL));
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
    let ledgers = 16;
    let mut next_entry = vec![0; ledgers as usize];
    for round in 0..30 {
        // Each batch appends to most ledgers.
        batch.clear();
        for ledger in (0..ledgers).filter(|l| (l + round) % 5 != 0) {
            for _ in 0..160 {
                let entry = &mut next_entry[ledger as usize];
                batch.put(&entry_key(1000 + ledger, *entry), &location(ledger, *entry));
                *entry += 1;
            }
        }
        now(store.db.write(&mut batch)).unwrap();
        store.db.check();
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
    store.db.check();
    store.reopen();
    store.db.check();
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
fn ranges_accept_every_form_of_bounds() {
    let store = Store::new(Options::default());
    write(
        &store.db,
        &[1u8, 2, 3, 4].map(|k| Op::Put(vec![k], Vec::new())),
    );
    let keys = |mut iter: crate::Iter| {
        let mut keys = Vec::new();
        while let Some((k, _)) = now(iter.next()).unwrap() {
            keys.push(k[0]);
        }
        keys
    };
    let (db, two, three): (_, &[u8], &[u8]) = (&store.db, &[2], &[3]);
    assert_eq!(keys(db.range(two..three)), [2]);
    assert_eq!(keys(db.range(two..=three)), [2, 3]);
    assert_eq!(keys(db.range(two..)), [2, 3, 4]);
    assert_eq!(keys(db.range(..three)), [1, 2]);
    assert_eq!(keys(db.range(..=three)), [1, 2, 3]);
    assert_eq!(keys(db.range(..)), [1, 2, 3, 4]);
    assert_eq!(
        keys(db.range((Bound::Excluded(two), Bound::Unbounded))),
        [3, 4]
    );
}

#[test]
fn oversized_keys_leave_the_store_unchanged() {
    let store = Store::new(Options::default());
    let max = Options::default().max_key_size();
    let mut batch = WriteBatch::new();
    batch.put(b"a", b"1");
    batch.put(&vec![1; max + 1], b"");
    assert!(matches!(
        now(store.db.write(&mut batch)),
        Err(Error::KeyTooLarge { .. })
    ));
    assert_eq!(get(&store.db, b"a", Comparison::Equal), None);
    batch.clear();
    batch.put(&vec![1; max], b"");
    now(store.db.write(&mut batch)).unwrap();
    assert!(get(&store.db, &vec![1; max], Comparison::Equal).is_some());
}

/// The value leaves of the tree `db` reads, from a view: pages, entries and bytes
/// used, in key order.
fn value_leaves(db: &Store) -> Vec<(u64, usize, usize)> {
    let view = now(ReadOnlyDb::open_sim(db.disk.clone(), db.opts.clone())).unwrap();
    let mut found = Vec::new();
    now(view.pages(|info| found.push(info))).unwrap();
    found
        .iter()
        .filter(|info| info.value)
        .map(|info| (info.pages, info.entries, info.used))
        .collect()
}

#[test]
fn entries_too_large_for_a_leaf_get_value_leaves() {
    let max = node::max_entry(256);
    let mut store = Store::new(Options::new().page_size(256));
    let mut model = Model::new();
    // A store whose only entry is large keeps a node above its value leaf.
    let ops = [Op::Put(b"m".to_vec(), vec![7; 600])];
    write(&store.db, &ops);
    apply(&mut model, &ops);
    assert_eq!(store.db.tree().current().root.unwrap().level, 1);
    check_reads(&store.db, &model);
    // An entry just over the limit takes a page, a larger one several. Just under,
    // it stays in a leaf.
    let ops = [
        Op::Put(b"a".to_vec(), vec![1; 3]),
        Op::Put(b"c".to_vec(), vec![2; max]),
        Op::Put(b"e".to_vec(), vec![3; 2000]),
        Op::Put(b"g".to_vec(), vec![4; max - 1]),
    ];
    write(&store.db, &ops);
    apply(&mut model, &ops);
    check_reads(&store.db, &model);
    let used = |key: usize, value: usize| node::HEADER + key + value;
    let expected = [
        (1, 1, used(1, max)),
        (8, 1, used(1, 2000)),
        (3, 1, used(1, 600)),
    ];
    assert_eq!(value_leaves(&store), expected);
    store.reopen();
    check_reads(&store.db, &model);
    // Replacing one with a small entry, and a small entry with a large one.
    let ops = [
        Op::Put(b"e".to_vec(), vec![5]),
        Op::Put(b"g".to_vec(), vec![6; 300]),
    ];
    write(&store.db, &ops);
    apply(&mut model, &ops);
    check_reads(&store.db, &model);
    let expected = [
        (1, 1, used(1, max)),
        (2, 1, used(1, 300)),
        (3, 1, used(1, 600)),
    ];
    assert_eq!(value_leaves(&store), expected);
    // Deleting every value leaf leaves ordinary leaves only.
    let ops = [
        Op::Delete(b"c".to_vec()),
        Op::DeleteRange(b"f".to_vec(), b"n".to_vec()),
    ];
    write(&store.db, &ops);
    apply(&mut model, &ops);
    check_reads(&store.db, &model);
    assert_eq!(value_leaves(&store), []);
}

#[test]
fn writes_never_read_value_leaves() {
    let opts = Options::new().page_size(256).cache_size(16 * 256);
    let mut store = Store::new(opts);
    let mut model = Model::new();
    let key = |i: u32| i.to_be_bytes().to_vec();
    let ops: Vec<_> = (0..60)
        .map(|i| Op::Put(key(i), vec![i as u8; if i % 2 == 0 { 300 } else { 4 }]))
        .collect();
    write(&store.db, &ops);
    apply(&mut model, &ops);
    // Damage every value leaf: reading one fails now, but no write reads them.
    for (page, _) in store.db.tree().leaves() {
        if node::span(page) > 0 {
            store.disk.flip(node::first_page(page) * 256 + 100);
        }
    }
    store.reopen();
    let ops = [
        // Below the first value leaf's key, and just above another's.
        Op::Put(Vec::new(), vec![9]),
        Op::Put([key(40), vec![1]].concat(), vec![9; 3]),
        // A value leaf replaced by a small entry, and one by another value leaf.
        Op::Put(key(10), vec![9; 2]),
        Op::Put(key(12), vec![9; 500]),
        Op::Delete(key(14)),
        Op::DeleteRange(key(20), key(31)),
    ];
    write(&store.db, &ops);
    apply(&mut model, &ops);
    store.db.check();
    for (k, v) in &model {
        let found = now(store.db.get(k, Comparison::Equal, |_, v| v.to_vec()));
        if v.len() == 300 {
            assert!(matches!(found, Err(Error::Corruption { .. })), "{k:?}");
        } else {
            assert_eq!(found.unwrap().as_ref(), Some(v), "{k:?}");
        }
    }
}

#[test]
fn a_value_larger_than_a_request_is_written_in_several() {
    let mut store = Store::new(Options::new());
    let value: Vec<u8> = (0..17 << 20).map(|i| (i % 251) as u8).collect();
    let (before, _) = store.disk.written();
    write(&store.db, &[Op::Put(b"big".to_vec(), value.clone())]);
    // Requests of 1 MiB, 16 at a time, for 17 MiB and a header: 18 of them, then
    // the node above the value leaf and the meta page.
    assert_eq!(store.disk.written().0 - before, 18 + 2);
    let expected = Some((b"big".to_vec(), value));
    assert_eq!(get(&store.db, b"big", Comparison::Floor), expected);
    store.reopen();
    assert_eq!(get(&store.db, b"bigger", Comparison::Lower), expected);
    store.db.check();
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
    // Dropping the iterator lets go of the old version, whose pages the next write
    // frees.
    drop(iter);
    store.db.check();
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
    store.db.check();
    assert!(store.db.tree().current().root.is_none());
    store.reopen();
    assert!(scan(&store.db, ALL).is_empty());
}

/// A store of 2000 keys on 256-byte slotted pages, whose root is three levels above
/// the leaves.
fn deep_store() -> (Store, Model) {
    node::slotted_only();
    let store = Store::new(Options::new().page_size(256));
    let puts: Vec<_> = (0..2000u32)
        .map(|i| Op::Put(i.to_be_bytes().to_vec(), vec![i as u8; 8]))
        .collect();
    write(&store.db, &puts);
    let mut model = Model::new();
    apply(&mut model, &puts);
    assert_eq!(store.db.tree().current().root.unwrap().level, 3);
    (store, model)
}

/// A node on the way down to a key: its page, the keys of its subtree, `lo..hi`
/// (`hi` is `None` along the right edge), and how many children it has, if any.
struct Subtree {
    page: u64,
    lo: Vec<u8>,
    hi: Option<Vec<u8>>,
    children: usize,
}

/// The nodes from the root of the current version down to the leaf where `key`
/// belongs.
fn path_to(db: &Db, key: &[u8]) -> Vec<Subtree> {
    let tree = db.tree();
    let root = tree.current().root.unwrap();
    let mut path = vec![Subtree {
        page: root.page,
        lo: Vec::new(),
        hi: None,
        children: 0,
    }];
    let mut frame = root.frame;
    for _ in 0..root.level {
        let node = tree.node(frame);
        let n = node.count();
        let key_at = |i| {
            let mut key = Vec::new();
            node.key_into(i, &mut key);
            key
        };
        // The last child whose smallest key is at most `key`.
        let i = (1..n).filter(|&i| node.cmp_key(i, key).is_le()).count();
        let parent = path.last_mut().unwrap();
        parent.children = n;
        let hi = if i + 1 < n {
            Some(key_at(i + 1))
        } else {
            parent.hi.clone()
        };
        path.push(Subtree {
            page: node.child_page(i),
            lo: key_at(i),
            hi,
            children: 0,
        });
        frame = node.child(i);
    }
    path
}

/// Writes `ops`, which delete every key outside the subtree of the node at
/// `page`, `level` levels above the leaves, and checks that this node becomes the
/// root as it is: the commit writes no node, and the pages of the old version
/// that the new one doesn't use are free again. Then checks the store, before
/// and after reopening it.
fn collapse_onto(store: &mut Store, model: &mut Model, ops: &[Op], (page, level): (u64, u8)) {
    let before = store.db.tree().pages();
    let disk_ops = store.disk.ops();
    write(&store.db, ops);
    apply(model, ops);
    let root = store.db.tree().current().root.unwrap();
    assert_eq!((root.page, root.level), (page, level));
    // A sync, the meta page and a sync.
    assert_eq!(store.disk.ops() - disk_ops, 3, "the commit wrote nodes");
    let tree = store.db.tree();
    let after = tree.pages();
    for page in &before {
        let used = tree.pages.space.is_used(*page);
        assert_eq!(used, after.contains(page), "page {page}");
    }
    drop(tree);
    store.db.check();
    assert_eq!(scan(&store.db, ALL), expected_scan(model, ALL));
    store.reopen();
    store.db.check();
    assert_eq!(scan(&store.db, ALL), expected_scan(model, ALL));
}

#[test]
fn deleting_all_but_a_subtree_makes_its_node_the_root() {
    // The subtrees holding a key one, two and three levels below the root. The
    // root collapses onto the node of each, which the new version keeps as it is.
    for depth in 1..=3 {
        let (mut store, mut model) = deep_store();
        let path = path_to(&store.db, &1100u32.to_be_bytes());
        let node = &path[depth];
        let ops = [
            Op::DeleteRange(Vec::new(), node.lo.clone()),
            Op::DeleteRange(node.hi.clone().unwrap(), vec![0xff; 5]),
        ];
        let level = 3 - depth as u8;
        collapse_onto(&mut store, &mut model, &ops, (node.page, level));
    }
}

#[test]
fn a_root_collapses_through_a_node_left_with_one_child() {
    let (mut store, mut model) = deep_store();
    let path = path_to(&store.db, &1100u32.to_be_bytes());
    let (top, leaf) = (&path[1], &path[3]);
    // Deleting all but one leaf under a child of the root leaves the leaf's
    // parent with one child, and alone under its own parent, which merges into a
    // neighbor: the tree keeps a node with one child.
    let ops = [
        Op::DeleteRange(top.lo.clone(), leaf.lo.clone()),
        Op::DeleteRange(leaf.hi.clone().unwrap(), top.hi.clone().unwrap()),
    ];
    write(&store.db, &ops);
    apply(&mut model, &ops);
    store.db.check();
    let path = path_to(&store.db, &leaf.lo);
    assert_eq!((path.len(), path[2].children), (4, 1));
    // Deleting everything else collapses the root onto the leaf, through that
    // node, which the new version drops as it is.
    let ops = [
        Op::DeleteRange(Vec::new(), leaf.lo.clone()),
        Op::DeleteRange(leaf.hi.clone().unwrap(), vec![0xff; 5]),
    ];
    collapse_onto(&mut store, &mut model, &ops, (leaf.page, 0));
}

#[test]
fn a_batch_of_empty_ranges_changes_nothing() {
    let store = Store::new(Options::new().page_size(256));
    let key = |i: u32| i.to_be_bytes().to_vec();
    let puts: Vec<_> = (0..300).map(|i| Op::Put(key(i), vec![1; 8])).collect();
    write(&store.db, &puts);
    let mut model = Model::new();
    apply(&mut model, &puts);
    let generation = store.db.tree().current().generation;
    let disk_ops = store.disk.ops();
    // Ranges that end at or before their start hold no keys: a batch of them, like
    // an empty batch, commits nothing.
    write(
        &store.db,
        &[
            Op::DeleteRange(key(200), key(100)),
            Op::DeleteRange(key(50), key(50)),
        ],
    );
    now(store.db.write(&mut WriteBatch::new())).unwrap();
    assert_eq!(store.db.tree().current().generation, generation);
    assert_eq!(store.disk.ops(), disk_ops);
    store.db.check();
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
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
    store.db.check();
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    for key in model.keys().step_by(13) {
        for cmp in COMPARISONS {
            assert_eq!(get(&store.db, key, cmp), expected(&model, key, cmp));
        }
    }
    store.db.check();
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

#[test]
fn nodes_whose_entries_lie_outside_them_are_reported() {
    // A node's checksum matches, but its entries don't lie within it, as a bug or a
    // crafted file could leave it. Keys of two lengths make slotted nodes, with no
    // key prefix: two heads from byte 32, then slots of 6 bytes from 40, each
    // ending with its value's length.
    let current = |db: &Db| {
        let tree = db.tree();
        (tree.current().root.unwrap(), tree.current().generation)
    };
    // A root leaf whose second value runs on for 64 KiB fails the reads that need
    // it, rather than lend what lies past the page.
    let mut store = Store::new(Options::new().page_size(256));
    write(&store.db, &[Op::Put(b"a".to_vec(), b"x".to_vec())]);
    let (root, generation) = current(&store.db);
    let mut buf = node::EntryBuf::default();
    buf.push(b"a", b"x");
    buf.push(b"bb", b"y");
    let mut leaf = vec![0; 256];
    node::build(&mut leaf, 0, &buf, 0, 2);
    leaf[50..52].copy_from_slice(&u16::MAX.to_le_bytes());
    node::seal(&mut leaf, generation);
    store.disk.set_bytes(root.page * 256, &leaf);
    store.reopen();
    let found = now(store.db.get(b"bb", Comparison::Equal, |_, v| v.len()));
    assert!(
        matches!(found, Err(Error::Corruption { page }) if page == root.page),
        "{found:?}"
    );
    let mut iter = store.db.range(..);
    assert!(matches!(now(iter.next()), Err(Error::Corruption { page }) if page == root.page));
    drop(iter);
    // A root internal node whose first child reference is 3 bytes fails the open.
    let store = Store::new(Options::new().page_size(256));
    write(&store.db, &round_of(300, 0));
    let (root, generation) = current(&store.db);
    assert!(root.level > 0);
    let mut buf = node::EntryBuf::default();
    buf.push(b"a", &[0; 3]);
    buf.push(b"bb", &[0; node::CHILD_REF]);
    let mut internal = vec![0; 256];
    node::build(&mut internal, root.level, &buf, 0, 2);
    node::seal(&mut internal, generation);
    store.disk.set_bytes(root.page * 256, &internal);
    let opened = now(Db::open_sim(store.disk.restart(), store.opts.clone()));
    assert!(matches!(opened, Err(Error::Corruption { page }) if page == root.page));
}

/// Puts keys `0..n`, each with value `round`.
fn round_of(n: u32, round: u8) -> Vec<Op> {
    (0..n)
        .map(|i| Op::Put(i.to_be_bytes().to_vec(), vec![round; 8]))
        .collect()
}

fn view_get(view: &ReadOnlyDb, key: &[u8], cmp: Comparison) -> crate::Result<Option<Entry>> {
    now(view.get(key, cmp, |k, v| (k.to_vec(), v.to_vec())))
}

fn view_scan(view: &ReadOnlyDb) -> crate::Result<Vec<Entry>> {
    let mut iter = view.range(..);
    let mut entries = Vec::new();
    while let Some((k, v)) = now(iter.next())? {
        entries.push((k.to_vec(), v.to_vec()));
    }
    Ok(entries)
}

/// Checks that a read of a read-only view found `expected`, or, if `may_be_stale`,
/// failed with `Error::Stale`. Returns whether it failed.
fn is_stale<T: PartialEq + std::fmt::Debug>(
    found: crate::Result<T>,
    expected: T,
    may_be_stale: bool,
) -> bool {
    match found {
        Ok(found) => {
            assert_eq!(found, expected);
            false
        }
        Err(Error::Stale) if may_be_stale => true,
        Err(e) => panic!("{e}"),
    }
}

#[test]
fn a_read_only_view_reads_its_version_until_it_is_overwritten() {
    node::slotted_only();
    // Small caches, so that the view reads its leaves from the disk.
    let opts = Options::new().page_size(256).cache_size(16 * 256);
    let store = Store::new(opts.clone());
    let mut model = Model::new();
    let ops = round_of(300, 0);
    write(&store.db, &ops);
    apply(&mut model, &ops);
    let view = now(ReadOnlyDb::open_sim(store.disk.clone(), opts.clone())).unwrap();
    let mut stale = 0;
    for round in 1..10 {
        // Every round replaces every leaf. The owner reuses the pages a write
        // replaces from its next write on, so the first leaves the view intact.
        write(&store.db, &round_of(300, round));
        let may_be_stale = round > 1;
        stale += usize::from(is_stale(
            view_scan(&view),
            expected_scan(&model, ALL),
            may_be_stale,
        ));
        for i in (0..320u32).step_by(7) {
            let key = i.to_be_bytes();
            for cmp in COMPARISONS {
                let found = view_get(&view, &key, cmp);
                let expected = expected(&model, &key, cmp);
                stale += usize::from(is_stale(found, expected, may_be_stale));
            }
        }
    }
    assert!(stale > 0, "the owner never overwrote the view's version");
    // A new view reads the current version.
    let view = now(ReadOnlyDb::open_sim(store.disk.clone(), opts)).unwrap();
    assert_eq!(view_scan(&view).unwrap(), scan(&store.db, ALL));
}

#[test]
fn a_read_only_view_of_value_leaves_is_stale_once_they_are_overwritten() {
    let opts = Options::new().page_size(256);
    let store = Store::new(opts.clone());
    let key = |i: u32| i.to_be_bytes().to_vec();
    let round = |r: u8| {
        (0..20)
            .map(|i| Op::Put(key(i), vec![r; 300]))
            .collect::<Vec<_>>()
    };
    let mut model = Model::new();
    write(&store.db, &round(0));
    apply(&mut model, &round(0));
    // A cache that holds two of the value leaves, so that the view reads the
    // others from the disk.
    let view = now(ReadOnlyDb::open_sim(
        store.disk.clone(),
        opts.cache_size(16 * 256),
    ))
    .unwrap();
    let mut stale = 0;
    for r in 1..6 {
        // The owner reuses the pages of the value leaves a write replaces from its
        // next write on.
        write(&store.db, &round(r));
        let may_be_stale = r > 1;
        stale += usize::from(is_stale(
            view_scan(&view),
            expected_scan(&model, ALL),
            may_be_stale,
        ));
        for i in 0..20 {
            let found = view_get(&view, &key(i), Comparison::Equal);
            let expected = expected(&model, &key(i), Comparison::Equal);
            stale += usize::from(is_stale(found, expected, may_be_stale));
        }
    }
    assert!(
        stale > 0,
        "the owner never overwrote the view's value leaves"
    );
}

#[test]
fn a_read_only_view_lists_the_pages_of_its_tree() {
    let empty = Store::new(Options::new().page_size(256));
    let view = now(ReadOnlyDb::open_sim(empty.disk.clone(), empty.opts.clone())).unwrap();
    now(view.pages(|info| panic!("{info:?} in an empty store"))).unwrap();
    assert_eq!((view.entry_count(), view.entry_size()), (0, 0));

    let (store, model) = deep_store();
    // A cache too small for the leaves, so that most are read from the disk.
    let opts = Options::new().page_size(256).cache_size(16 * 256);
    let view = now(ReadOnlyDb::open_sim(store.disk.clone(), opts)).unwrap();
    assert_eq!(view.page_size(), 256);
    assert_eq!(view.entry_count(), model.len() as u64);
    let bytes = model.iter().map(|(k, v)| (k.len() + v.len()) as u64).sum();
    assert_eq!(view.entry_size(), bytes);
    let mut found = Vec::new();
    now(view.pages(|info| found.push(info))).unwrap();
    // Every page of the tree once: the internal nodes, then the leaves in key order.
    let mut pages: Vec<_> = found.iter().map(|info| info.page).collect();
    pages.sort_unstable();
    assert_eq!(pages, store.db.tree().pages());
    let leaves: Vec<_> = found
        .iter()
        .skip_while(|info| info.level > 0)
        .map(|info| info.page)
        .collect();
    let in_key_order: Vec<_> = store
        .db
        .tree()
        .leaves()
        .into_iter()
        .map(|(page, _)| page)
        .collect();
    assert_eq!(leaves, in_key_order);
    // The leaves hold every key, and the nodes of each level are the children of
    // the level above.
    let level = |level| found.iter().filter(move |info| info.level == level);
    assert_eq!(
        level(0).map(|info| info.entries).sum::<usize>(),
        model.len()
    );
    for l in 1..=3 {
        let children = level(l).map(|info| info.entries).sum::<usize>();
        assert_eq!(children, level(l - 1).count());
    }
    assert_eq!(level(3).count(), 1);
    assert!(
        found
            .iter()
            .all(|info| (1..=256).contains(&info.used) && !info.dense)
    );
}

#[test]
fn a_read_only_view_of_a_quiet_store_reports_damage() {
    let opts = Options::new().page_size(256).cache_size(16 * 256);
    let store = Store::new(opts.clone());
    write(&store.db, &round_of(300, 0));
    let (root, leaf) = {
        let tree = store.db.tree();
        let root = tree.current().root.unwrap();
        assert!(root.level > 0);
        (
            root.page,
            tree.descend_first(root, &mut crate::tree::Cursor::new()),
        )
    };
    // The owner has written nothing since the view opened: the page is corrupt.
    store.disk.flip(leaf * 256 + 100);
    let view = now(ReadOnlyDb::open_sim(store.disk.clone(), opts.clone())).unwrap();
    assert!(matches!(view_scan(&view), Err(Error::Corruption { page }) if page == leaf));
    store.disk.flip(root * 256 + 100);
    let opened = now(ReadOnlyDb::open_sim(store.disk.clone(), opts)).map(drop);
    assert!(matches!(opened, Err(Error::Corruption { page }) if page == root));
}

#[test]
fn a_read_only_view_is_stale_once_the_owner_cuts_its_pages() {
    node::slotted_only();
    // A small cache, so that the view reads its leaves from the disk.
    let opts = Options::new().page_size(256).cache_size(16 * 256);
    let store = Store::new(opts.clone());
    let batches = moving_scenario();
    for ops in &batches[..3] {
        write(&store.db, ops);
    }
    let view = now(ReadOnlyDb::open_sim(store.disk.clone(), opts)).unwrap();
    let leaves = store.db.tree().leaves();
    // The owner frees the end of the file, and cuts it.
    for ops in &batches[3..] {
        write(&store.db, ops);
    }
    let end = store.disk.len() / 256;
    let (_, key) = leaves
        .iter()
        .find(|&&(page, _)| page >= end)
        .expect("a leaf of the view past the end of the file");
    let found = view_get(&view, key, Comparison::Equal);
    assert!(matches!(found, Err(Error::Stale)), "{found:?}");
}

/// Opening a store reads three meta pages before its tree: the start of slot 0,
/// then both slots.
const RECOVER_OPS: u64 = 3;

#[test]
fn a_read_only_view_overwritten_while_it_opens_is_stale() {
    node::slotted_only();
    let opts = Options::new().page_size(256);
    let store = Store::new(opts.clone());
    for round in 0..2 {
        write(&store.db, &round_of(300, round));
    }
    // The view finds this version in the meta, then the owner's later writes,
    // which reuse its root page, where it reads the tree.
    let root = store.db.tree().current().root.unwrap().page;
    let disk = store.disk.restart();
    for round in 2..4 {
        write(&store.db, &round_of(300, round));
    }
    assert!(store.db.tree().pages().contains(&root));
    disk.overwrite_at(RECOVER_OPS, &store.disk);
    let opened = now(ReadOnlyDb::open_sim(disk, opts)).map(drop);
    assert!(matches!(opened, Err(Error::Stale)), "{opened:?}");
}

#[test]
fn a_bad_child_reference_fails_the_open() {
    let store = Store::new(Options::new().page_size(256));
    let puts: Vec<_> = (0..300u32)
        .map(|i| Op::Put(i.to_be_bytes().to_vec(), vec![7; 8]))
        .collect();
    write(&store.db, &puts);
    let (root, bytes) = {
        let tree = store.db.tree();
        let root = tree.current().root.unwrap();
        assert!(root.level > 0);
        (root.page, tree.pages.pool.page(root.frame).to_vec())
    };
    let first = Node::new(&bytes).child_page(0);
    // Point the first child past the end of the file, then the second at the
    // first's page. The root is sealed again, so that it passes its checksum.
    for (i, bad) in [(0, store.disk.len() / 256), (1, first)] {
        let mut page = bytes.clone();
        let (at, _) = Node::new(&page).value_span(i);
        let child = node::Child {
            page: bad,
            ..Node::new(&page).child_at(i)
        };
        page[at..at + node::CHILD_REF].copy_from_slice(&node::child_ref(child));
        let generation = Node::new(&page).generation();
        node::seal(&mut page, generation);
        let disk = store.disk.restart();
        disk.set_bytes(root * 256, &page);
        let opened = now(Db::open_sim(disk, store.opts.clone()));
        assert!(
            matches!(opened, Err(Error::Corruption { page }) if page == bad),
            "child {i}"
        );
    }
}

#[test]
fn bad_meta_pages_are_rejected() {
    let meta = Meta::empty(256);
    // The meta page in each slot of a two-page file, and why it isn't a store.
    let cases: [(&[Meta], &str); 7] = [
        // Zeros.
        (&[], "no intact meta page"),
        (
            &[Meta {
                page_size: 1000,
                ..meta
            }],
            "unsupported page size",
        ),
        (&[Meta { version: 2, ..meta }], "unsupported format version"),
        // The root is at the high-water mark, beyond the pages in use.
        (&[Meta { root: 2, ..meta }], "inconsistent meta page"),
        // The high-water mark leaves out meta page 1.
        (
            &[Meta {
                high_water: 1,
                ..meta
            }],
            "inconsistent meta page",
        ),
        // The tree is deeper than writes take.
        (
            &[Meta {
                root_level: crate::tree::MAX_HEIGHT as u8 - 1,
                ..meta
            }],
            "inconsistent meta page",
        ),
        // The newer slot has another page size.
        (
            &[
                meta,
                Meta {
                    page_size: 512,
                    generation: 1,
                    ..meta
                },
            ],
            "inconsistent meta page",
        ),
    ];
    for (metas, reason) in cases {
        let mut file = vec![0; 512];
        for (slot, meta) in file.chunks_mut(256).zip(metas) {
            meta.encode(slot);
        }
        let disk = SimDisk::new();
        disk.set_bytes(0, &file);
        let opened = now(Db::open_sim(disk, Options::new().page_size(256)));
        assert!(
            matches!(opened, Err(Error::InvalidFile(r)) if r == reason),
            "{reason}"
        );
    }
}

#[test]
fn the_file_bounds_the_high_water_mark() {
    let opts = Options::new().page_size(256);
    let open_both = |file: &[u8]| {
        let disk = SimDisk::new();
        disk.set_bytes(0, file);
        let viewed = now(ReadOnlyDb::open_sim(disk.clone(), opts.clone())).map(drop);
        (viewed, now(Db::open_sim(disk, opts.clone())))
    };
    // An empty store whose mark is as high as marks go: the open takes the end
    // of the file instead, and the store works.
    let mut file = vec![0; 512];
    Meta {
        high_water: u64::MAX,
        ..Meta::empty(256)
    }
    .encode(&mut file[..256]);
    open_both(&file).0.unwrap();
    let disk = SimDisk::new();
    disk.set_bytes(0, &file);
    let mut store = Store {
        db: now(Db::open_sim(disk.clone(), opts.clone())).unwrap(),
        disk,
        opts: opts.clone(),
    };
    assert_eq!(
        store.db.tree().pages.space.high_water(),
        crate::meta::META_PAGES
    );
    let ops = round_of(300, 1);
    write(&store.db, &ops);
    let mut model = Model::new();
    apply(&mut model, &ops);
    store.reopen();
    store.db.check();
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    // A three-page file whose root lists leaves on pages 3 to 5, within the mark:
    // no commit wrote them, so the tree is corrupt.
    let mut file = vec![0; 3 * 256];
    let mut buf = node::EntryBuf::default();
    for (i, key) in [b"a", b"b", b"c"].into_iter().enumerate() {
        let size = node::Size {
            entries: 1,
            bytes: 2,
        };
        buf.push(key, &node::child_ref(node::Child::leaf(3 + i as u64, size)));
    }
    node::build(&mut file[512..], 1, &buf, 0, 3);
    node::seal(&mut file[512..], 0);
    let meta = Meta {
        root: 2,
        root_level: 1,
        high_water: 10,
        entries: 3,
        entry_bytes: 6,
        ..Meta::empty(256)
    };
    meta.encode(&mut file[..256]);
    let (viewed, opened) = open_both(&file);
    assert!(
        matches!(viewed, Err(Error::Corruption { page: 3 })),
        "{viewed:?}"
    );
    assert!(matches!(opened, Err(Error::Corruption { page: 3 })));
    // So is a root leaf past the end of the file.
    Meta {
        root: 5,
        root_level: 0,
        ..meta
    }
    .encode(&mut file[..256]);
    let (viewed, opened) = open_both(&file[..512]);
    assert!(
        matches!(viewed, Err(Error::Corruption { page: 5 })),
        "{viewed:?}"
    );
    assert!(matches!(opened, Err(Error::Corruption { page: 5 })));
}

#[test]
fn a_view_reads_the_file_length_after_the_meta_page() {
    // The owner commits a larger version after the view read the file's length,
    // but before it read the meta page: the view opens that version, whose pages
    // reach past the length it read first.
    let opts = Options::new().page_size(256);
    let store = Store::new(opts.clone());
    let (first, second) = (round_of(10, 0), round_of(300, 1));
    write(&store.db, &first);
    let disk = store.disk.restart();
    write(&store.db, &second);
    assert!(store.disk.len() > disk.len());
    let mut model = Model::new();
    apply(&mut model, &first);
    apply(&mut model, &second);
    disk.overwrite_at(0, &store.disk);
    let view = now(ReadOnlyDb::open_sim(disk, opts)).unwrap();
    assert_eq!(view_scan(&view).unwrap(), expected_scan(&model, ALL));
}

/// Opens the store on `disk` with each read of the open failing in turn: that
/// open fails, and the next one works. Then opens it once more.
fn open_after_failed_reads(disk: &SimDisk, opts: &Options) -> Db {
    let clean = disk.restart();
    now(Db::open_sim(clean.clone(), opts.clone())).unwrap();
    for op in 0..clean.ops() {
        let disk = disk.restart();
        disk.inject(op, Fault::Error);
        let opened = now(Db::open_sim(disk.clone(), opts.clone()));
        assert!(
            matches!(opened, Err(Error::Io(_))),
            "read {op} failed, yet the open worked"
        );
        now(Db::open_sim(disk, opts.clone())).unwrap().check();
    }
    now(Db::open_sim(disk.restart(), opts.clone())).unwrap()
}

#[test]
fn a_failed_read_fails_only_that_open() {
    // The open reads both meta slots, then the internal nodes.
    let (store, model) = cold_store(2000, 64);
    assert!(store.db.tree().current().root.unwrap().level > 1);
    let db = open_after_failed_reads(&store.disk, &store.opts);
    assert_eq!(scan(&db, ALL), expected_scan(&model, ALL));
}

#[test]
fn a_damaged_meta_slot_leaves_the_other() {
    // Pages larger than the open's first read, which finds the page size in slot 0.
    let opts = Options::new().page_size(1024);
    let store = Store::new(opts.clone());
    write(&store.db, &[Op::Put(b"a".to_vec(), b"1".to_vec())]);
    let first = scan(&store.db, ALL);
    write(&store.db, &[Op::Put(b"b".to_vec(), b"2".to_vec())]);
    // The second write's meta, in slot 0, rots. The open looks for slot 1 where each
    // page size would put it, and finds the first write's.
    store.disk.flip(24);
    let db = open_after_failed_reads(&store.disk, &opts);
    assert_eq!(scan(&db, ALL), first);
}

#[test]
fn a_store_keeps_its_page_size() {
    let store = Store::new(Options::new().page_size(512));
    write(&store.db, &[Op::Put(b"a".to_vec(), b"1".to_vec())]);
    let expected = scan(&store.db, ALL);
    for other in [256, 1024] {
        let opts = Options::new().page_size(other);
        let opened = now(Db::open_sim(store.disk.restart(), opts.clone())).map(drop);
        assert!(
            matches!(opened, Err(Error::PageSizeMismatch { store: 512, options }) if options == other),
            "{opened:?}"
        );
        // A view writes nothing, so it reads the store anyway.
        let view = now(ReadOnlyDb::open_sim(store.disk.restart(), opts)).unwrap();
        assert_eq!(view_scan(&view).unwrap(), expected);
    }
    // Options without a page size open the store with its own, and its key limit.
    let db = now(Db::open_sim(store.disk.restart(), Options::new())).unwrap();
    assert_eq!(scan(&db, ALL), expected);
    let max = node::max_entry(512);
    let mut batch = WriteBatch::new();
    batch.put(&vec![1; max + 1], b"");
    let written = now(db.write(&mut batch));
    assert!(
        matches!(written, Err(Error::KeyTooLarge { size, max: m }) if size == max + 1 && m == max),
        "{written:?}"
    );
    batch.clear();
    batch.put(&vec![1; max], b"");
    now(db.write(&mut batch)).unwrap();
}

/// Batches that build a few levels on 256-byte pages, rewrite and delete parts of
/// them, then delete everything.
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
        vec![Op::DeleteRange(key(0), key(300))],
    ]
}

/// Batches on 256-byte pages that mix entries too large for a leaf, of one page,
/// several, and more than a write request carries, with small ones, then replace,
/// remove and add to them.
fn value_scenario() -> Vec<Vec<Op>> {
    let key = |i: u32| i.to_be_bytes().to_vec();
    let len = |i: u32| {
        if i.is_multiple_of(3) {
            40 + 20 * i as usize
        } else {
            8
        }
    };
    vec![
        (0..40).map(|i| Op::Put(key(i), vec![1; len(i)])).collect(),
        vec![Op::Put(key(17), vec![2; 300 << 10])],
        (0..40)
            .step_by(2)
            .map(|i| Op::Put(key(i), vec![3; 700]))
            .collect(),
        vec![
            Op::DeleteRange(key(5), key(25)),
            Op::Put(key(10), vec![4; 300]),
            Op::Put([key(30), vec![0]].concat(), vec![4]),
        ],
        (0..40).map(|i| Op::Delete(key(i))).collect(),
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
        // Read too, so that reads meet failures as well: a lookup, and a scan, which
        // reads ahead.
        if now(db.get(&[0, 0, 0, 0x7f], Comparison::Ceiling, |_, _| ())).is_err()
            || try_scan(&db).is_err()
        {
            return (Some(db), done + 1);
        }
    }
    (Some(db), batches.len())
}

fn try_scan(db: &Db) -> crate::Result<()> {
    let mut iter = db.range(..);
    while now(iter.next())?.is_some() {}
    Ok(())
}

/// Creating a store takes three operations: two meta slots and a sync.
const CREATE_OPS: u64 = 3;

#[test]
fn every_single_failure_leaves_a_recoverable_store() {
    // A small cache, so that writes read leaves back from the disk.
    let opts = Options::new().page_size(256).cache_size(16 * 256);
    check_every_single_failure(&opts, &failure_scenario());
}

#[test]
fn every_single_failure_with_value_leaves_leaves_a_recoverable_store() {
    let opts = Options::new().page_size(256).cache_size(16 * 256);
    check_every_single_failure(&opts, &value_scenario());
}

#[test]
fn every_single_failure_while_moving_or_cutting_leaves_a_recoverable_store() {
    check_every_single_failure(&Options::new().page_size(256), &moving_scenario());
}

/// Runs `batches` once per operation they take, with that operation failing, then
/// once more per operation, with the machine crashing there. Checks that every
/// time, the store is left usable, or failed, and that it reopens in the state
/// after the last write that succeeded.
fn check_every_single_failure(opts: &Options, batches: &[Vec<Op>]) {
    let mut states = vec![Vec::new()];
    let mut model = Model::new();
    for ops in batches {
        apply(&mut model, ops);
        states.push(expected_scan(&model, ALL));
    }
    let clean = SimDisk::new();
    let (db, done) = try_scenario(&clean, opts, batches);
    assert_eq!(done, batches.len());
    drop(db);
    let total = clean.ops();
    for fault in [Fault::Error, Fault::Crash] {
        for op in 0..total {
            let disk = SimDisk::new();
            disk.inject(op, fault);
            let (db, done) = try_scenario(&disk, opts, batches);
            let failed = db.as_ref().is_some_and(Db::has_failed);
            if let Some(db) = &db {
                if failed {
                    assert!(matches!(
                        now(db.write(&mut WriteBatch::new())),
                        Err(Error::Failed)
                    ));
                } else {
                    // A write that failed before committing left the store as it was.
                    db.check();
                }
            }
            drop(db);
            let reopened = match now(Db::open_sim(disk.restart(), opts.clone())) {
                Ok(db) => db,
                // A failure while creating the store may leave no store at all.
                Err(Error::InvalidFile(_)) if op < CREATE_OPS => continue,
                Err(e) => panic!("{fault:?} at operation {op}: reopening failed: {e}"),
            };
            reopened.check();
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

/// Rounds of batches to write as groups: in each, the first write waits for its
/// sync while the others queue up behind it, and they commit together. Batches
/// override parts of those before them in their group.
fn group_scenario() -> Vec<Vec<Vec<Op>>> {
    vec![
        vec![
            puts(0..300, 1),
            puts(100..200, 2),
            vec![range_op(50, 250), put_op(120, 3)],
            (0..300).step_by(7).map(|i| put_op(i, 4)).collect(),
        ],
        vec![
            vec![range_op(0, 100)],
            (0..300).step_by(3).map(|i| put_op(i, 5)).collect(),
            vec![range_op(200, 300), put_op(250, 6)],
        ],
    ]
}

/// What a write came to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Outcome {
    Applied,
    /// It failed, and left the store as it was.
    Unchanged,
    /// It failed during its commit, which left the store failed: it may have
    /// applied.
    InDoubt,
    /// The store had failed already.
    Rejected,
}

/// Runs the rounds until a write fails. Returns the store, if it opened, and the
/// outcome of each write that ran, in the order they queued.
fn try_group_scenario(
    disk: &Rc<SimDisk>,
    opts: &Options,
    rounds: &[Vec<Vec<Op>>],
) -> (Option<Db>, Vec<Outcome>) {
    let Ok(db) = now(Db::open_sim(disk.clone(), opts.clone())) else {
        return (None, Vec::new());
    };
    let outcome = |result: crate::Result<()>| match result {
        Ok(()) => Outcome::Applied,
        Err(Error::Failed) => Outcome::Rejected,
        Err(_) if db.has_failed() => Outcome::InDoubt,
        Err(_) => Outcome::Unchanged,
    };
    let mut outcomes = Vec::new();
    for round in rounds {
        let mut batches: Vec<_> = round.iter().map(|ops| batch_of(ops)).collect();
        let mut writes = Vec::new();
        disk.hold_syncs();
        for batch in &mut batches {
            let mut write: Write = Box::pin(db.write(batch));
            match poll(write.as_mut()) {
                Poll::Ready(result) => writes.push(Err(outcome(result))),
                Poll::Pending => writes.push(Ok(write)),
            }
        }
        disk.release_syncs();
        let start = outcomes.len();
        for write in writes {
            outcomes.push(write.map_or_else(|done| done, |write| outcome(now(write))));
        }
        if outcomes[start..].iter().any(|&o| o != Outcome::Applied) {
            break;
        }
    }
    (Some(db), outcomes)
}

#[test]
fn every_single_failure_in_a_group_leaves_a_recoverable_store() {
    // A small cache, so that groups read leaves back from the disk.
    let opts = Options::new().page_size(256).cache_size(16 * 256);
    let rounds = group_scenario();
    let batches: Vec<_> = rounds.iter().flatten().collect();
    let clean = SimDisk::new();
    let (db, outcomes) = try_group_scenario(&clean, &opts, &rounds);
    assert_eq!(outcomes, vec![Outcome::Applied; batches.len()]);
    drop(db);
    let total = clean.ops();
    for fault in [Fault::Error, Fault::Crash] {
        for op in 0..total {
            let disk = SimDisk::new();
            disk.inject(op, fault);
            let (db, outcomes) = try_group_scenario(&disk, &opts, &rounds);
            let failed = db.as_ref().is_some_and(Db::has_failed);
            if let Some(db) = &db {
                if failed {
                    assert!(matches!(
                        now(db.write(&mut WriteBatch::new())),
                        Err(Error::Failed)
                    ));
                } else {
                    db.check();
                }
            }
            drop(db);
            let reopened = match now(Db::open_sim(disk.restart(), opts.clone())) {
                Ok(db) => db,
                Err(Error::InvalidFile(_)) if op < CREATE_OPS => continue,
                Err(e) => panic!("{fault:?} at operation {op}: reopening failed: {e}"),
            };
            reopened.check();
            // The writes that succeeded applied, in order, and those that failed
            // with the store left usable didn't. Those whose commit failed the
            // store applied together, or not at all.
            let (mut applied, mut in_doubt) = (Model::new(), Model::new());
            for (ops, &outcome) in batches.iter().zip(&outcomes) {
                if outcome == Outcome::Applied {
                    apply(&mut applied, ops);
                }
                if matches!(outcome, Outcome::Applied | Outcome::InDoubt) {
                    apply(&mut in_doubt, ops);
                }
            }
            let state = scan(&reopened, ALL);
            assert!(
                state == expected_scan(&applied, ALL)
                    || (failed && state == expected_scan(&in_doubt, ALL)),
                "{fault:?} at operation {op}: {outcomes:?}"
            );
        }
    }
}

/// Batches on 256-byte pages: the first fills windows of 64 pages with leaves, the
/// second replaces all but one leaf in ten or so, leaving those windows mostly
/// free, and the third replaces a leaf under each parent, so that the leaves left
/// behind move. Then the upper half of the keys goes, freeing the end of the file,
/// which the next write cuts.
fn moving_scenario() -> Vec<Vec<Op>> {
    let key = |i: u32| i.to_be_bytes().to_vec();
    let put = |i: u32, value: u8| Op::Put(key(i), vec![value; 8]);
    vec![
        (0..2400).map(|i| put(i, 1)).collect(),
        (0..2400)
            .filter(|i| i % 240 >= 30)
            .map(|i| put(i, 2))
            .collect(),
        (0..2400).step_by(50).map(|i| put(i, 3)).collect(),
        vec![Op::DeleteRange(key(1200), key(2400))],
        vec![put(0, 4)],
    ]
}

/// The windows of 64 pages below the end of the file that hold some pages of the
/// current version, but few enough for writes to move them.
fn mostly_free_windows(store: &Store) -> usize {
    let tree = store.db.tree();
    let mut used = vec![0u32; tree.pages.space.high_water() as usize / 64];
    for page in tree.pages().into_iter().chain(0..crate::meta::META_PAGES) {
        if let Some(n) = used.get_mut(page as usize / 64) {
            *n += 1;
        }
    }
    let few = 1..=crate::space::MOVE_AT;
    used.iter().filter(|&&n| few.contains(&n)).count()
}

#[test]
fn writes_move_what_they_keep_out_of_mostly_free_windows() {
    node::slotted_only();
    let batches = moving_scenario();
    // Leaves that aren't cached don't move.
    for (cache, moves) in [(64 << 20, true), (16 * 256, false)] {
        let store = Store::new(Options::new().page_size(256).cache_size(cache));
        let mut model = Model::new();
        for ops in &batches[..2] {
            write(&store.db, ops);
            apply(&mut model, ops);
        }
        assert!(mostly_free_windows(&store) > 0);
        write(&store.db, &batches[2]);
        apply(&mut model, &batches[2]);
        assert_eq!(store.db.tree().moves().1 > 0, moves);
        if moves {
            assert_eq!(mostly_free_windows(&store), 0);
        }
        store.db.tree().check();
        assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    }
}

#[test]
fn writes_move_internal_nodes_too() {
    node::slotted_only();
    let mut store = Store::new(Options::new().page_size(256));
    let key = |i: u32| i.to_be_bytes().to_vec();
    let put = |i: u32, value: u8| Op::Put(key(i), vec![value; 8]);
    let mut model = Model::new();
    let batches = [
        // Five levels.
        (0..12_000).map(|i| put(i, 1)).collect(),
        // Replaces the nodes above all leaves but the first ones, leaving the
        // windows of the internal nodes that the first batch wrote mostly free.
        (1_500..12_000).map(|i| put(i, 2)).collect(),
        // Its nodes take a run in a window that is mostly free.
        vec![put(11_999, 3)],
        // Rebuilds the nodes above the first leaf, which move the internal nodes
        // they keep out of mostly free windows, one of them with internal children.
        vec![put(0, 4)],
    ];
    for ops in &batches {
        write(&store.db, ops);
        apply(&mut model, ops);
        store.db.tree().check();
    }
    assert!(store.db.tree().current().root.unwrap().level >= 3);
    assert!(store.db.tree().moves().1 > 0);
    store.reopen();
    store.db.tree().check();
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
}

#[test]
fn freeing_the_end_of_the_file_cuts_it() {
    node::slotted_only();
    let mut store = Store::new(Options::new().page_size(256));
    let batches = moving_scenario();
    let mut model = Model::new();
    for ops in &batches[..4] {
        write(&store.db, ops);
        apply(&mut model, ops);
    }
    let len = store.disk.len();
    // The next write ends the file at the last page in use.
    write(&store.db, &batches[4]);
    apply(&mut model, &batches[4]);
    let end = store.db.tree().pages.space.high_water() * 256;
    assert!(end < len * 7 / 8, "{end} of {len} bytes");
    assert_eq!(store.disk.len(), end);
    store.reopen();
    store.db.tree().check();
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    // And grows again from there.
    let ops: Vec<_> = (5000..7000u32)
        .map(|i| Op::Put(i.to_be_bytes().to_vec(), vec![5; 8]))
        .collect();
    write(&store.db, &ops);
    apply(&mut model, &ops);
    assert!(store.disk.len() > end);
    store.reopen();
    store.db.tree().check();
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
}

#[test]
fn a_shorter_min_run_reuses_shorter_holes() {
    // Updating one key moves its leaf to a new page and frees the old one: the
    // file grows past one free page, unless a run of one may reuse it.
    let peak = |opts: Options| {
        let store = Store::new(opts.page_size(256));
        let mut peak = 0;
        for i in 0..40u8 {
            write(&store.db, &[Op::Put(b"key".to_vec(), vec![i])]);
            peak = peak.max(store.disk.len() / 256);
        }
        peak
    };
    // The meta pages, a stretch of 16 free pages, and the leaf.
    assert_eq!(peak(Options::new()), 19);
    assert_eq!(peak(Options::new().min_run_pages(1)), 4);
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
    node::slotted_only();
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
        store.db.check();
    }
}

#[test]
fn adjacent_runs_of_pages_of_both_lives_take_one_request() {
    node::slotted_only();
    let store = Store::new(Options::new().page_size(256));
    let max_run = crate::db::max_run(256);
    // Ledgers 1,000 keys apart, so that appends to one go between its last entry
    // and the next ledger's first.
    let ledgers = |entries: std::ops::Range<u32>| -> Vec<Op> {
        (0..400)
            .flat_map(|ledger| entries.clone().map(move |e| ledger * 1000 + e))
            .map(|i| Op::Put(i.to_be_bytes().to_vec(), vec![1; 8]))
            .collect()
    };
    write(&store.db, &ledgers(0..1));
    // The appends fill long-lived leaves, and leave each ledger's last leaf
    // short-lived: the file grows by runs of pages of both lives, in turns.
    let before = store.db.tree().pages();
    let ops_before = store.disk.ops();
    write(&store.db, &ledgers(1..30));
    let written: Vec<_> = store
        .db
        .tree()
        .pages()
        .into_iter()
        .filter(|page| before.binary_search(page).is_err())
        .collect();
    let requests = requests_for(&written, max_run);
    assert_eq!(store.disk.ops() - ops_before, requests + 3);
    store.db.check();
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
            store.db.check();
        }
        write(&store.db, &ops);
        apply(&mut model, &ops);
        // The retry used the very pages the failed write gave back.
        assert_eq!(store.db.tree().pages(), twin.db.tree().pages());
    }
    store.reopen();
    store.db.check();
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
}

/// Polls a future until it finishes or waits: a future that yields, waking
/// itself before it returns `Pending`, is polled again.
pub(crate) fn poll<F: Future + ?Sized>(mut f: Pin<&mut F>) -> Poll<F::Output> {
    loop {
        WOKEN.set(false);
        match f.as_mut().poll(&mut Context::from_waker(&woken_waker())) {
            Poll::Pending if WOKEN.get() => {}
            polled => return polled,
        }
    }
}

/// Polls a future once: one that yields stops there.
fn poll_once<F: Future + ?Sized>(f: Pin<&mut F>) -> Poll<F::Output> {
    f.poll(&mut Context::from_waker(Waker::noop()))
}

/// A store of `n` entries on 256-byte pages, reopened with a cache of `frames`
/// pages, so that none of its leaves is cached.
fn cold_store(n: u32, frames: usize) -> (Store, Model) {
    let mut store = Store::new(Options::new().page_size(256).cache_size(frames * 256));
    let puts: Vec<_> = (0..n)
        .map(|i| Op::Put(i.to_be_bytes().to_vec(), vec![i as u8; 8]))
        .collect();
    write(&store.db, &puts);
    let mut model = Model::new();
    apply(&mut model, &puts);
    store.reopen();
    (store, model)
}

/// Frames that reads are filling, and frames that cancelled reads left behind.
fn loading_frames(db: &Db) -> (usize, usize) {
    let tree = db.tree();
    let pool = &tree.pages.pool;
    let loading = (0..pool.slots())
        .map(|i| pool.slot(i))
        .filter(|&f| pool.frame(f).holds == Holds::Loading);
    loading.fold((0, 0), |(busy, idle), f| match pool.refs(f) {
        0 => (busy, idle + 1),
        _ => (busy + 1, idle),
    })
}

/// How many keys of `model` sort below `key`.
fn below(model: &Model, key: &[u8]) -> usize {
    model
        .range::<[u8], _>((Bound::Unbounded, Bound::Excluded(key)))
        .count()
}

// The window a cold store with a 64-page cache reads ahead.
const WINDOW: u64 = 16;

#[test]
fn a_cold_scan_reads_ahead() {
    let (store, model) = cold_store(2000, 64);
    let leaves = store.db.tree().leaves().len() as u64;
    assert!(leaves > 4 * WINDOW);
    store.disk.hold();
    let ops = store.disk.ops();
    let mut iter = store.db.range(..);
    assert!(poll(pin!(iter.next())).is_pending());
    // The first leaf goes out together with the leaves after it.
    assert_eq!(store.disk.ops() - ops, 1 + WINDOW);
    store.disk.release();
    let mut seen = 0;
    while now(iter.next()).unwrap().is_some() {
        seen += 1;
    }
    assert_eq!(seen, model.len());
    // Each leaf was read once.
    assert_eq!(store.disk.ops() - ops, leaves);
    drop(iter);
    store.db.check();
}

#[test]
fn an_iterator_waits_for_the_leaf_read_ahead() {
    let (store, model) = cold_store(2000, 64);
    let leaves = store.db.tree().leaves();
    store.disk.hold();
    let ops = store.disk.ops();
    let mut iter = store.db.range(..);
    assert!(poll(pin!(iter.next())).is_pending());
    // Only the first leaf's read finishes.
    store.disk.release_through(ops);
    for _ in 0..below(&model, &leaves[1].1) {
        assert!(now(iter.next()).unwrap().is_some());
    }
    // The second leaf's read is in flight: the iterator waits for it, and reads
    // one more leaf ahead, rather than reading the second one again.
    assert!(poll(pin!(iter.next())).is_pending());
    assert_eq!(store.disk.ops() - ops, 1 + WINDOW + 1);
    store.disk.release();
    let mut seen = below(&model, &leaves[1].1);
    while now(iter.next()).unwrap().is_some() {
        seen += 1;
    }
    assert_eq!(seen, model.len());
    assert_eq!(store.disk.ops() - ops, leaves.len() as u64);
    drop(iter);
    store.db.check();
}

#[test]
fn concurrent_misses_share_one_read() {
    let (store, model) = cold_store(2000, 64);
    let key = 1234u32.to_be_bytes();
    store.disk.hold();
    let ops = store.disk.ops();
    let value = |_: &[u8], v: &[u8]| v.to_vec();
    let mut a = Box::pin(store.db.get(&key, Comparison::Equal, value));
    let mut b = pin!(store.db.get(&key, Comparison::Equal, value));
    let mut c = pin!(store.db.get(&key, Comparison::Equal, value));
    assert!(poll(a.as_mut()).is_pending());
    assert!(poll(b.as_mut()).is_pending());
    assert!(poll(c.as_mut()).is_pending());
    assert_eq!(store.disk.ops() - ops, 1, "a lookup read the leaf again");
    // A lookup dropped while it waits leaves the read to the others.
    drop(a);
    store.disk.release();
    let value = Some(model[&key[..]].clone());
    for get in [b, c] {
        assert_eq!(now(get).unwrap(), value);
    }
    assert_eq!(store.disk.ops() - ops, 1);
    store.db.check();
}

#[test]
fn a_failed_shared_read_is_read_again_by_the_others() {
    let (store, model) = cold_store(2000, 64);
    let key = 1234u32.to_be_bytes();
    store.disk.hold();
    let ops = store.disk.ops();
    store.disk.inject(ops, Fault::Error);
    let mut a = pin!(store.db.get(&key, Comparison::Equal, |_, v| v.to_vec()));
    let mut b = pin!(store.db.get(&key, Comparison::Equal, |_, v| v.to_vec()));
    assert!(poll(a.as_mut()).is_pending());
    assert!(poll(b.as_mut()).is_pending());
    store.disk.release();
    // The lookup that claims the failed read reports it; the other reads the leaf
    // itself.
    assert!(matches!(now(a), Err(Error::Io(_))));
    assert_eq!(now(b).unwrap(), Some(model[&key[..]].clone()));
    assert_eq!(store.disk.ops() - ops, 2);
    store.db.check();
}

#[test]
fn a_write_waiting_for_a_failed_commit_fails() {
    // A store of one leaf, which isn't cached.
    let (store, _) = cold_store(1, 64);
    store.disk.hold();
    let ops = store.disk.ops();
    // The first write takes the write lock, then waits for its leaf.
    let mut first = batch_of(&[Op::Put(vec![1], vec![1])]);
    let mut a = pin!(store.db.write(&mut first));
    assert!(poll(a.as_mut()).is_pending());
    // The second waits for the lock.
    let mut second = batch_of(&[Op::Put(vec![2], vec![2])]);
    let mut b = pin!(store.db.write(&mut second));
    assert!(poll(b.as_mut()).is_pending());
    // The first reads its leaf, writes the new one, and fails to sync it, which
    // leaves the store failed. The second then gets the lock, and fails too.
    store.disk.inject(ops + 2, Fault::Error);
    store.disk.release();
    assert!(matches!(now(a), Err(Error::Io(_))));
    assert!(matches!(now(b), Err(Error::Failed)));
}

#[test]
fn a_lookup_keeps_a_read_ahead_going() {
    let (store, model) = cold_store(2000, 64);
    let key = store.db.tree().leaves()[3].1.clone();
    store.disk.hold();
    let ops = store.disk.ops();
    let mut iter = store.db.range(..);
    assert!(poll(pin!(iter.next())).is_pending());
    let mut get = pin!(store.db.get(&key, Comparison::Equal, |_, v| v.to_vec()));
    assert!(poll(get.as_mut()).is_pending());
    // The lookup waits for the read of the leaf that the iterator reads ahead,
    // which goes on when the iterator is dropped.
    drop(iter);
    store.disk.release();
    assert_eq!(now(get).unwrap(), Some(model[&key].clone()));
    assert_eq!(store.disk.ops() - ops, 1 + WINDOW);
    store.db.check();
}

#[test]
fn reading_ahead_joins_the_reads_in_flight() {
    let (store, model) = cold_store(2000, 64);
    let leaves = store.db.tree().leaves();
    store.disk.hold();
    let ops = store.disk.ops();
    let key = leaves[3].1.clone();
    let mut get = Box::pin(store.db.get(&key, Comparison::Equal, |_, v| v.to_vec()));
    assert!(poll(get.as_mut()).is_pending());
    let mut iter = store.db.range(..);
    assert!(poll(pin!(iter.next())).is_pending());
    // The iterator reads ahead past the fourth leaf, whose read the lookup started.
    assert_eq!(store.disk.ops() - ops, 1 + WINDOW);
    // That read goes on without the lookup.
    drop(get);
    store.disk.release();
    let mut seen = 0;
    while now(iter.next()).unwrap().is_some() {
        seen += 1;
    }
    assert_eq!(seen, model.len());
    assert_eq!(store.disk.ops() - ops, leaves.len() as u64);
    drop(iter);
    store.db.check();
}

#[test]
fn reading_ahead_stops_at_the_end_of_the_range() {
    let (store, model) = cold_store(2000, 64);
    let leaves = store.db.tree().leaves();
    let (lo, hi) = (&leaves[5].1[..], &leaves[8].1[..]);
    let ops = store.disk.ops();
    let range = (Bound::Included(lo), Bound::Excluded(hi));
    assert_eq!(scan(&store.db, range), expected_scan(&model, range));
    // Leaves 5 to 7, and 8 to find where the range ends.
    assert_eq!(store.disk.ops() - ops, 4);
    let range = (Bound::Included(lo), Bound::Included(hi));
    assert_eq!(scan(&store.db, range), expected_scan(&model, range));
    store.db.check();
}

#[test]
fn reading_ahead_stays_within_the_cache_budget() {
    // Internal nodes alone fill the 8-page cache.
    let (store, model) = cold_store(6000, 8);
    assert!(store.db.tree().pages.pool.in_use() > 8);
    store.disk.hold();
    let ops = store.disk.ops();
    let mut iter = store.db.range(..);
    assert!(poll(pin!(iter.next())).is_pending());
    assert_eq!(store.disk.ops() - ops, 1, "read ahead past the budget");
    store.disk.release();
    let mut seen = 0;
    while now(iter.next()).unwrap().is_some() {
        seen += 1;
    }
    assert_eq!(seen, model.len());
    drop(iter);
    store.db.check();
}

/// A store on 256-byte pages of `n` entries too large for a leaf, two pages each,
/// reopened with a cache of `frames` pages, so that none of them is cached.
fn cold_value_store(n: u32, frames: usize) -> (Store, Model) {
    let mut store = Store::new(Options::new().page_size(256).cache_size(frames * 256));
    let puts: Vec<_> = (0..n)
        .map(|i| Op::Put(i.to_be_bytes().to_vec(), vec![i as u8; 300]))
        .collect();
    write(&store.db, &puts);
    let mut model = Model::new();
    apply(&mut model, &puts);
    store.reopen();
    (store, model)
}

#[test]
fn concurrent_misses_on_a_value_leaf_share_one_read() {
    let (store, model) = cold_value_store(40, 64);
    let key = 17u32.to_be_bytes();
    store.disk.hold();
    let ops = store.disk.ops();
    let value = |_: &[u8], v: &[u8]| v.to_vec();
    let mut a = Box::pin(store.db.get(&key, Comparison::Equal, value));
    let mut b = pin!(store.db.get(&key, Comparison::Equal, value));
    let mut c = pin!(store.db.get(&key, Comparison::Equal, value));
    assert!(poll(a.as_mut()).is_pending());
    assert!(poll(b.as_mut()).is_pending());
    assert!(poll(c.as_mut()).is_pending());
    assert_eq!(
        store.disk.ops() - ops,
        1,
        "a lookup read the value leaf again"
    );
    drop(a);
    store.disk.release();
    let value = Some(model[&key[..]].clone());
    for get in [b, c] {
        assert_eq!(now(get).unwrap(), value);
    }
    // The cache keeps it: reading it again reads nothing.
    let found = now(store.db.get(&key, Comparison::Floor, |_, v| v.to_vec())).unwrap();
    assert_eq!(found, value);
    assert_eq!(store.disk.ops() - ops, 1);
    store.db.check();
}

#[test]
fn a_cold_scan_reads_value_leaves_ahead() {
    let (store, model) = cold_value_store(40, 64);
    store.disk.hold();
    let ops = store.disk.ops();
    let mut iter = store.db.range(..);
    assert!(poll(pin!(iter.next())).is_pending());
    // Value leaves take at most a quarter of the 64-page cache: the one the
    // iterator waits for, and the 7 after it.
    assert_eq!(store.disk.ops() - ops, 8);
    assert_eq!(store.db.tree().pages.pool.large_pages(), 16);
    store.disk.release();
    assert_eq!(scan_rest(&mut iter), expected_scan(&model, ALL));
    // Each value leaf was read once, and the cache kept a quarter's worth.
    assert_eq!(store.disk.ops() - ops, 40);
    assert!(store.db.tree().pages.pool.large_pages() <= 16);
    drop(iter);
    store.db.check();
}

/// The entries left in `iter`.
fn scan_rest(iter: &mut crate::Iter) -> Vec<Entry> {
    let mut entries = Vec::new();
    while let Some((k, v)) = now(iter.next()).unwrap() {
        entries.push((k.to_vec(), v.to_vec()));
    }
    entries
}

#[test]
fn value_leaves_take_at_most_a_quarter_of_the_cache() {
    let (store, mut model) = cold_value_store(40, 64);
    for key in model.keys() {
        assert_eq!(
            get(&store.db, key, Comparison::Equal).unwrap().1,
            model[key]
        );
        assert!(store.db.tree().pages.pool.large_pages() <= 16);
    }
    // One larger than that quarter isn't cached: each read of it reads it.
    let ops = [Op::Put(vec![1], vec![1; 5000])];
    write(&store.db, &ops);
    apply(&mut model, &ops);
    for _ in 0..2 {
        let before = store.disk.ops();
        assert_eq!(
            get(&store.db, &[1], Comparison::Equal).unwrap().1,
            model[&vec![1]]
        );
        assert_eq!(store.disk.ops() - before, 1);
    }
    check_reads(&store.db, &model);
}

#[test]
fn a_write_larger_than_the_cache_looks_at_few_frames_for_each() {
    // A cache of 16 pages, and a write that pins hundreds of leaves, whose values
    // differ, and builds as many. Once it holds every frame, the clock hand looks a
    // step ahead for each frame the pool grows by, rather than at the whole pool.
    let store = Store::new(Options::new().page_size(256).cache_size(16 * 256));
    let puts = |round: u32| -> Vec<_> {
        (0..20_000u32)
            .map(|i| Op::Put(i.to_be_bytes().to_vec(), (i ^ round).to_le_bytes().to_vec()))
            .collect()
    };
    write(&store.db, &puts(0));
    let looked = store.db.tree().pages.cache.looked;
    write(&store.db, &puts(1));
    let tree = store.db.tree();
    let frames = tree.pages.pool.slots();
    assert!(frames > 500, "the write fit in the cache");
    assert!(tree.pages.cache.looked - looked < crate::cache::STEP * frames);
    drop(tree);
    store.db.check();
}

#[test]
fn a_cached_value_leaf_leaves_the_cache_with_its_pages() {
    let store = Store::new(Options::new().page_size(256));
    let key = b"k".to_vec();
    let mut pages = Vec::new();
    for round in 0..4u8 {
        // Each write frees the pages of the value leaf it replaces, and the next one
        // reuses them, under the same page id: no copy of the old one may remain.
        write(&store.db, &[Op::Put(key.clone(), vec![round; 300])]);
        assert_eq!(
            get(&store.db, &key, Comparison::Equal),
            Some((key.clone(), vec![round; 300]))
        );
        store.db.check();
        pages.push(store.db.tree().leaves()[0].0);
    }
    assert!(
        node::span(pages[0]) > 0 && pages[0] == pages[2],
        "{pages:?}"
    );
}

#[test]
fn a_failed_read_ahead_fails_at_its_leaf() {
    let (store, model) = cold_store(2000, 64);
    let leaves = store.db.tree().leaves();
    // The read of the fourth leaf, read ahead, fails.
    store.disk.inject(store.disk.ops() + 3, Fault::Error);
    let mut iter = store.db.range(..);
    let (mut seen, mut failed) = (Vec::new(), 0);
    loop {
        match now(iter.next()) {
            Ok(Some((k, _))) => seen.push(k.to_vec()),
            Ok(None) => break,
            Err(e) => {
                assert!(matches!(e, Error::Io(_)));
                assert_eq!(seen.len(), below(&model, &leaves[3].1));
                failed += 1;
            }
        }
    }
    // Calling again read the leaf again, and the scan went on.
    assert_eq!(failed, 1);
    assert!(seen.iter().eq(model.keys()));
    drop(iter);
    store.db.check();

    // A damaged leaf read ahead fails the scan when it gets there.
    store.disk.flip(leaves[3].0 * 256 + 100);
    let mut iter = store.db.range(..);
    for _ in 0..below(&model, &leaves[3].1) {
        assert!(now(iter.next()).unwrap().is_some());
    }
    assert!(matches!(now(iter.next()), Err(Error::Corruption { page }) if page == leaves[3].0));
    drop(iter);
    store.db.check();
}

#[test]
fn cancelled_reads_keep_their_frames_until_they_finish() {
    let (store, model) = cold_store(2000, 64);
    store.disk.hold();
    let mut iter = store.db.range(..);
    assert!(poll(pin!(iter.next())).is_pending());
    let mut get = Box::pin(
        store
            .db
            .get(&[0xff; 4], Comparison::Floor, |_, v| v.to_vec()),
    );
    assert!(poll(get.as_mut()).is_pending());
    assert_eq!(loading_frames(&store.db), (1 + WINDOW as usize + 1, 0));
    // Dropping the readers cancels their reads, but the disk still fills the
    // frames: they are only reclaimed once the reads finish.
    drop(iter);
    drop(get);
    assert_eq!(loading_frames(&store.db), (1 + WINDOW as usize + 1, 0));
    store.disk.release();
    assert_eq!(loading_frames(&store.db), (0, 1 + WINDOW as usize + 1));
    store.db.check();
    // The clock hand reuses them.
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    assert_eq!(loading_frames(&store.db), (0, 0));
    store.db.check();
}

#[test]
fn a_dropped_store_keeps_its_frames_until_its_reads_finish() {
    let (store, _) = cold_store(2000, 64);
    store.disk.hold();
    let mut iter = store.db.range(..);
    assert!(poll(pin!(iter.next())).is_pending());
    assert_eq!(loading_frames(&store.db), (1 + WINDOW as usize, 0));
    // The disk finishes the reads after the store is gone: their buffers must keep
    // the frame memory alive until then. Only a sanitizer or Miri sees if they don't.
    let Store { disk, db, .. } = store;
    drop(iter);
    drop(db);
    disk.release();
}

// Writes that queue up behind a commit share the next one.

fn put_op(i: u32, value: u8) -> Op {
    Op::Put(i.to_be_bytes().to_vec(), vec![value; 8])
}

fn range_op(start: u32, end: u32) -> Op {
    Op::DeleteRange(start.to_be_bytes().to_vec(), end.to_be_bytes().to_vec())
}

fn puts(keys: std::ops::Range<u32>, value: u8) -> Vec<Op> {
    keys.map(|i| put_op(i, value)).collect()
}

/// The generation of the store's current version: each commit adds one.
fn generation(db: &Db) -> u64 {
    db.tree().current().generation
}

type Write<'a> = Pin<Box<dyn Future<Output = crate::Result<()>> + 'a>>;

/// Starts writing each batch: polls each write once, which must wait.
fn start_writes<'a>(db: &'a Db, batches: &'a mut [WriteBatch]) -> Vec<Write<'a>> {
    let mut writes: Vec<Write> = Vec::new();
    for batch in batches {
        let mut write: Write = Box::pin(db.write(batch));
        assert!(poll(write.as_mut()).is_pending(), "the write didn't wait");
        writes.push(write);
    }
    writes
}

/// A store with `n` entries on 256-byte pages, and its model.
fn store_of(n: u32) -> (Store, Model) {
    let store = Store::new(Options::new().page_size(256));
    let mut model = Model::new();
    write(&store.db, &puts(0..n, 1));
    apply(&mut model, &puts(0..n, 1));
    (store, model)
}

#[test]
fn writes_queued_behind_a_commit_share_the_next_one() {
    let (mut store, mut model) = store_of(200);
    let first = puts(0..50, 2);
    // Each batch overrides parts of those before it: a range removes some of the
    // keys the first put, and a later batch puts some of them back.
    let queued = [
        puts(100..150, 3),
        vec![range_op(120, 180), put_op(130, 4)],
        vec![put_op(125, 5), put_op(130, 5), put_op(179, 5)],
    ];
    let start = generation(&store.db);
    store.disk.hold_syncs();
    let mut batch = batch_of(&first);
    let mut first_write = Box::pin(store.db.write(&mut batch));
    // The first write waits for its sync, and the others queue up behind it.
    assert!(poll(first_write.as_mut()).is_pending());
    let mut batches: Vec<_> = queued.iter().map(|ops| batch_of(ops)).collect();
    let writes = start_writes(&store.db, &mut batches);
    store.disk.release_syncs();
    now(first_write).unwrap();
    assert_eq!(generation(&store.db), start + 1);
    for write in writes {
        now(write).unwrap();
    }
    // One commit took all three, in order.
    assert_eq!(generation(&store.db), start + 2);
    for ops in [&first].into_iter().chain(&queued) {
        apply(&mut model, ops);
    }
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    store.db.check();
    // The batches are left as they were: writing one again applies it again.
    now(store.db.write(&mut batches[1])).unwrap();
    apply(&mut model, &queued[1]);
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    store.reopen();
    store.db.check();
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
}

#[test]
fn an_oversized_batch_is_rejected_on_its_own() {
    let (store, mut model) = store_of(200);
    let start = generation(&store.db);
    let too_large = vec![0; store.opts.max_key_size() + 1];
    store.disk.hold_syncs();
    let mut first = batch_of(&puts(0..10, 2));
    let mut first_write = pin!(store.db.write(&mut first));
    assert!(poll(first_write.as_mut()).is_pending());
    let mut second = batch_of(&puts(10..20, 3));
    let mut second_write = pin!(store.db.write(&mut second));
    assert!(poll(second_write.as_mut()).is_pending());
    // Rejected before it joins the others.
    let mut oversized = batch_of(&[put_op(20, 4), Op::Put(too_large, vec![1])]);
    assert!(matches!(
        poll(pin!(store.db.write(&mut oversized))),
        Poll::Ready(Err(Error::KeyTooLarge { .. }))
    ));
    let mut third = batch_of(&puts(30..40, 5));
    let mut third_write = pin!(store.db.write(&mut third));
    assert!(poll(third_write.as_mut()).is_pending());
    store.disk.release_syncs();
    now(first_write).unwrap();
    now(second_write).unwrap();
    now(third_write).unwrap();
    assert_eq!(generation(&store.db), start + 2);
    for ops in [puts(0..10, 2), puts(10..20, 3), puts(30..40, 5)] {
        apply(&mut model, &ops);
    }
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    store.db.check();
}

/// Starts a write of `first`, which waits for its sync, and queues writes of
/// `queued` behind it. Then lets the first finish.
fn queue_behind<'a>(
    store: &Store,
    first: &[Op],
    queued: &'a mut [WriteBatch],
    db: &'a Db,
) -> Vec<Write<'a>> {
    store.disk.hold_syncs();
    let mut batch = batch_of(first);
    let mut first_write = pin!(store.db.write(&mut batch));
    assert!(poll(first_write.as_mut()).is_pending());
    let writes = start_writes(db, queued);
    store.disk.release_syncs();
    now(first_write).unwrap();
    writes
}

#[test]
fn a_failure_before_the_commit_fails_the_whole_group() {
    let (store, mut model) = store_of(200);
    let first = puts(0..50, 2);
    let queued = [puts(50..100, 3), puts(100..150, 4)];
    let mut batches: Vec<_> = queued.iter().map(|ops| batch_of(ops)).collect();
    let db = store.db.clone();
    let writes = queue_behind(&store, &first, &mut batches, &db);
    apply(&mut model, &first);
    // The group's first operation, a page write, fails.
    store.disk.inject(store.disk.ops(), Fault::Error);
    for write in writes {
        assert!(matches!(now(write), Err(Error::Io(_))));
    }
    // The store is as it was, and takes writes again.
    assert!(!store.db.has_failed());
    store.db.check();
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    for ops in &queued {
        write(&store.db, ops);
        apply(&mut model, ops);
    }
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
}

#[test]
fn a_failure_during_the_commit_fails_the_store_for_the_whole_group() {
    let first = puts(0..50, 2);
    let queued = [puts(50..100, 3), vec![range_op(0, 75)]];
    // The group's first sync, its meta page and its second sync.
    for at in 0..3 {
        let (mut store, mut model) = store_of(200);
        let mut batches: Vec<_> = queued.iter().map(|ops| batch_of(ops)).collect();
        let db = store.db.clone();
        let mut writes = queue_behind(&store, &first, &mut batches, &db);
        apply(&mut model, &first);
        let before = expected_scan(&model, ALL);
        // The group's leader writes its pages, and waits for its sync.
        store.disk.hold_syncs();
        assert!(poll(writes[0].as_mut()).is_pending());
        store.disk.inject(store.disk.ops() + at, Fault::Error);
        store.disk.release_syncs();
        for write in writes {
            assert!(matches!(now(write), Err(Error::Io(_))), "failure at {at}");
        }
        assert!(store.db.has_failed());
        assert!(matches!(
            now(store.db.write(&mut WriteBatch::new())),
            Err(Error::Failed)
        ));
        drop(db);
        // Either every batch in the group committed, or none did.
        store.reopen();
        store.db.check();
        for ops in &queued {
            apply(&mut model, ops);
        }
        let state = scan(&store.db, ALL);
        assert!(
            state == before || state == expected_scan(&model, ALL),
            "failure at {at}"
        );
    }
}

#[test]
fn a_write_dropped_while_queued_is_withdrawn() {
    let (store, mut model) = store_of(200);
    let start = generation(&store.db);
    let first = puts(0..50, 2);
    let queued = [puts(50..100, 3), vec![range_op(0, 150)], puts(100..150, 4)];
    let mut batches: Vec<_> = queued.iter().map(|ops| batch_of(ops)).collect();
    store.disk.hold_syncs();
    let mut batch = batch_of(&first);
    let mut first_write = pin!(store.db.write(&mut batch));
    assert!(poll(first_write.as_mut()).is_pending());
    let mut writes = start_writes(&store.db, &mut batches);
    // The range goes before it commits.
    drop(writes.remove(1));
    store.disk.release_syncs();
    now(first_write).unwrap();
    for write in writes {
        now(write).unwrap();
    }
    assert_eq!(generation(&store.db), start + 2);
    for ops in [&first, &queued[0], &queued[2]] {
        apply(&mut model, ops);
    }
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    store.db.check();
}

#[test]
fn writes_dropped_while_queued_leave_nothing_behind() {
    let (store, mut model) = store_of(200);
    let first = puts(0..50, 2);
    let mut batches = [batch_of(&puts(50..100, 3)), batch_of(&[range_op(0, 200)])];
    // The second finds the group full, and waits for room.
    store.db.set_group_limit(batches[0].size());
    let db = store.db.clone();
    drop(queue_behind(&store, &first, &mut batches, &db));
    apply(&mut model, &first);
    store.db.check();
    // A write whose group committed goes before it learns so.
    store.db.set_group_limit(crate::writes::GROUP_SIZE);
    let queued = [puts(100..150, 4), puts(150..200, 5)];
    let mut batches: Vec<_> = queued.iter().map(|ops| batch_of(ops)).collect();
    let mut writes = queue_behind(&store, &first, &mut batches, &db);
    now(writes.remove(0)).unwrap();
    drop(writes);
    store.db.check();
    // The next write commits alone.
    let start = generation(&store.db);
    write(&store.db, &puts(200..250, 6));
    assert_eq!(generation(&store.db), start + 1);
    for ops in [&first, &queued[0], &queued[1], &puts(200..250, 6)] {
        apply(&mut model, ops);
    }
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
}

/// A cold store, whose leaves aren't cached, and three batches to write as a
/// group: the first puts a key that the second's range removes, so that the group
/// doesn't read the leaves the range covers, and the third puts elsewhere.
fn cold_group() -> (Store, Model, [Vec<Op>; 3]) {
    let (store, model) = cold_store(2000, 64);
    let queued = [
        vec![put_op(1000, 2)],
        vec![range_op(500, 1500)],
        vec![put_op(1800, 4)],
    ];
    (store, model, queued)
}

/// Queues writes of `queued` behind a first write, then has the first of them
/// lead its group until it waits for its leaves.
fn hold_group_building<'a>(
    store: &Store,
    model: &mut Model,
    queued: &'a mut [WriteBatch],
    db: &'a Db,
) -> Vec<Write<'a>> {
    let first = vec![put_op(10, 1)];
    let mut writes = queue_behind(store, &first, queued, db);
    apply(model, &first);
    store.disk.hold();
    assert!(poll(writes[0].as_mut()).is_pending());
    writes
}

#[test]
fn a_write_dropped_while_its_group_builds_is_withdrawn() {
    let (store, mut model, queued) = cold_group();
    let start = generation(&store.db);
    let mut batches: Vec<_> = queued.iter().map(|ops| batch_of(ops)).collect();
    let db = store.db.clone();
    let mut writes = hold_group_building(&store, &mut model, &mut batches, &db);
    drop(writes.remove(1));
    // Without the range, the group needs the leaf of the first put, which it
    // didn't read: it starts over.
    store.disk.release();
    for write in writes {
        now(write).unwrap();
    }
    assert_eq!(generation(&store.db), start + 2);
    apply(&mut model, &queued[0]);
    apply(&mut model, &queued[2]);
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    store.db.check();
}

#[test]
fn a_write_dropped_while_leading_its_group_leaves_it_to_the_others() {
    let (store, mut model, queued) = cold_group();
    let start = generation(&store.db);
    let mut batches: Vec<_> = queued.iter().map(|ops| batch_of(ops)).collect();
    let db = store.db.clone();
    let mut writes = hold_group_building(&store, &mut model, &mut batches, &db);
    // The leader goes before it writes anything; the next write takes over.
    drop(writes.remove(0));
    store.disk.release();
    for write in writes {
        now(write).unwrap();
    }
    assert!(!store.db.has_failed());
    assert_eq!(generation(&store.db), start + 2);
    apply(&mut model, &queued[1]);
    apply(&mut model, &queued[2]);
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    store.db.check();
}

#[test]
fn a_write_of_the_next_group_takes_over_a_group_left_without_its_leader() {
    let (store, mut model, queued) = cold_group();
    let start = generation(&store.db);
    let mut batches: Vec<_> = queued.iter().map(|ops| batch_of(ops)).collect();
    let db = store.db.clone();
    let mut writes = hold_group_building(&store, &mut model, &mut batches, &db);
    // Two writes queue up for the next group, then the leader goes.
    let later = [vec![put_op(1900, 5)], vec![put_op(1950, 6)]];
    let mut later_batches: Vec<_> = later.iter().map(|ops| batch_of(ops)).collect();
    let mut later_writes = start_writes(&store.db, &mut later_batches);
    drop(writes.remove(0));
    // Without waiting for the rest of the group, a later write takes it over, and
    // goes too while it waits for the group's leaves.
    let ops = store.disk.ops();
    assert!(poll(later_writes[0].as_mut()).is_pending());
    assert!(store.disk.ops() > ops, "it didn't read the group's leaves");
    drop(later_writes.remove(0));
    // The other takes over, then commits its own group.
    store.disk.release();
    now(later_writes.remove(0)).unwrap();
    assert_eq!(generation(&store.db), start + 3);
    for write in writes {
        now(write).unwrap();
    }
    assert!(!store.db.has_failed());
    for ops in [&queued[1], &queued[2], &later[1]] {
        apply(&mut model, ops);
    }
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    store.db.check();
}

#[test]
fn a_group_whose_writes_all_went_away_is_dropped() {
    let (store, mut model, queued) = cold_group();
    let mut batches: Vec<_> = queued.iter().map(|ops| batch_of(ops)).collect();
    let db = store.db.clone();
    let mut writes = hold_group_building(&store, &mut model, &mut batches, &db);
    // The leader goes, then the rest.
    drop(writes.remove(0));
    drop(writes);
    store.disk.release();
    store.db.check();
    let start = generation(&store.db);
    write(&store.db, &puts(0..10, 5));
    apply(&mut model, &puts(0..10, 5));
    assert_eq!(generation(&store.db), start + 1);
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
}

#[test]
fn a_write_dropped_while_its_group_commits_still_commits() {
    let (store, mut model) = store_of(200);
    let start = generation(&store.db);
    let first = puts(0..50, 2);
    let queued = [puts(50..100, 3), vec![range_op(0, 75)], puts(60..70, 4)];
    let mut batches: Vec<_> = queued.iter().map(|ops| batch_of(ops)).collect();
    let db = store.db.clone();
    let mut writes = queue_behind(&store, &first, &mut batches, &db);
    store.disk.hold_syncs();
    assert!(poll(writes[0].as_mut()).is_pending());
    // The group is writing: its batches go to disk together, or not at all.
    drop(writes.remove(1));
    store.disk.release_syncs();
    for write in writes {
        now(write).unwrap();
    }
    assert_eq!(generation(&store.db), start + 2);
    for ops in [&first].into_iter().chain(&queued) {
        apply(&mut model, ops);
    }
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    store.db.check();
}

#[test]
fn a_write_dropped_while_committing_its_group_fails_the_store() {
    let (mut store, mut model) = store_of(200);
    let first = puts(0..50, 2);
    let queued = [puts(50..100, 3), vec![range_op(0, 75)]];
    let mut batches: Vec<_> = queued.iter().map(|ops| batch_of(ops)).collect();
    let db = store.db.clone();
    let mut writes = queue_behind(&store, &first, &mut batches, &db);
    apply(&mut model, &first);
    let before = expected_scan(&model, ALL);
    store.disk.hold_syncs();
    assert!(poll(writes[0].as_mut()).is_pending());
    // A write queues up for the next group.
    let mut later = batch_of(&puts(150..160, 5));
    let mut later_write = Box::pin(store.db.write(&mut later));
    assert!(poll(later_write.as_mut()).is_pending());
    // The group's leader goes while it waits for its sync.
    drop(writes.remove(0));
    assert!(store.db.has_failed());
    store.disk.release_syncs();
    assert!(matches!(now(writes.remove(0)), Err(Error::Failed)));
    assert!(matches!(now(later_write), Err(Error::Failed)));
    drop(writes);
    drop(db);
    store.reopen();
    store.db.check();
    for ops in &queued {
        apply(&mut model, ops);
    }
    let state = scan(&store.db, ALL);
    assert!(state == before || state == expected_scan(&model, ALL));
}

#[test]
fn a_write_dropped_while_cutting_the_file_leaves_its_group_committed() {
    node::slotted_only();
    let (mut store, mut model) = store_of(0);
    let batches = moving_scenario();
    for ops in &batches[..3] {
        write(&store.db, ops);
        apply(&mut model, ops);
    }
    let len = store.disk.len();
    // Freeing the upper half of the keys lets the commit after it cut the file.
    let queued = [batches[4].clone(), puts(0..10, 5)];
    let mut group: Vec<_> = queued.iter().map(|ops| batch_of(ops)).collect();
    let db = store.db.clone();
    let mut writes = queue_behind(&store, &batches[3], &mut group, &db);
    apply(&mut model, &batches[3]);
    // The group's leader syncs, writes the meta page, syncs again, then cuts.
    store.disk.hold_syncs();
    assert!(poll(writes[0].as_mut()).is_pending());
    store.disk.allow_syncs(2);
    assert!(poll(writes[0].as_mut()).is_pending());
    // The group committed: its other write returns.
    let leader = writes.remove(0);
    now(writes.remove(0)).unwrap();
    // The leader goes while it cuts the file.
    drop(leader);
    assert!(store.db.has_failed());
    store.disk.release_syncs();
    assert_eq!(store.disk.len(), len);
    drop(writes);
    drop(db);
    store.reopen();
    store.db.check();
    for ops in &queued {
        apply(&mut model, ops);
    }
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
}

/// A batch that rewrites every entry of `store_of(200)`, and removes some.
fn rewrite_all() -> Vec<Op> {
    let mut ops = puts(0..200, 7);
    ops.push(range_op(50, 60));
    ops.push(Op::Delete(150u32.to_be_bytes().to_vec()));
    ops
}

#[test]
fn reads_see_the_last_version_while_a_write_merges() {
    let (store, mut model) = store_of(200);
    store.db.set_merge_slice(Some(4));
    let before = expected_scan(&model, ALL);
    let ops = rewrite_all();
    let mut batch = batch_of(&ops);
    let mut write = Box::pin(store.db.write(&mut batch));
    // The write pauses before it reads or writes anything, normalizing its batch.
    let io = store.disk.ops();
    assert!(poll_once(write.as_mut()).is_pending());
    assert_eq!(store.disk.ops(), io);
    for (k, v) in &before {
        assert_eq!(
            get(&store.db, k, Comparison::Equal),
            Some((k.clone(), v.clone()))
        );
    }
    assert_eq!(scan(&store.db, ALL), before);
    let mut iter = store.db.range(..);
    let mut seen = vec![];
    let (k, v) = now(iter.next()).unwrap().unwrap();
    seen.push((k.to_vec(), v.to_vec()));
    // It goes on a slice at a time.
    let mut pauses = 1;
    while poll_once(write.as_mut()).is_pending() {
        pauses += 1;
    }
    assert!(pauses > 10, "the write paused {pauses} times");
    // The iterator started before the write published: it doesn't see it.
    while let Some((k, v)) = now(iter.next()).unwrap() {
        seen.push((k.to_vec(), v.to_vec()));
    }
    assert_eq!(seen, before);
    drop(iter);
    apply(&mut model, &ops);
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    store.db.check();
}

#[test]
fn a_write_dropped_while_its_merge_pauses_changes_nothing() {
    let (mut store, model) = store_of(200);
    store.db.set_merge_slice(Some(4));
    let start = generation(&store.db);
    // The merge pauses 50 times, once normalizing and planning are done.
    for polls in [1, 10, 49] {
        let mut batch = batch_of(&rewrite_all());
        let mut write = Box::pin(store.db.write(&mut batch));
        pause_merging(&store.db, write.as_mut());
        for _ in 1..polls {
            assert!(poll_once(write.as_mut()).is_pending());
        }
        // The pages the merge built go back to the free set, and the leaves it
        // pinned are unpinned.
        drop(write);
        store.db.check();
        assert_eq!(generation(&store.db), start);
        assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    }
    store.reopen();
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    // The store takes writes again.
    let mut model = model;
    write(&store.db, &rewrite_all());
    apply(&mut model, &rewrite_all());
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    store.db.check();
}

#[test]
fn a_merge_counts_scattered_operations_as_more_work() {
    // The same updates, in key order, then put in an order that scatters each
    // leaf's operations in the batch. In slices of 32 units, the second merge
    // pauses after each operation.
    let n = 2000;
    let orders: [Vec<u32>; 2] = [(0..n).collect(), (0..n).map(|i| i * 7 % n).collect()];
    let mut pauses = [0; 2];
    for (order, pauses) in orders.iter().zip(&mut pauses) {
        let (store, mut model) = store_of(n);
        store.db.set_merge_slice(Some(32));
        let ops: Vec<_> = order.iter().map(|&i| put_op(i, 2)).collect();
        let mut batch = batch_of(&ops);
        let mut write = Box::pin(store.db.write(&mut batch));
        pause_merging(&store.db, write.as_mut());
        while store.db.tree().merging() {
            assert!(poll_once(write.as_mut()).is_pending());
            *pauses += 1;
        }
        while poll_once(write.as_mut()).is_pending() {}
        apply(&mut model, &ops);
        assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
        store.db.check();
    }
    let [sorted, scattered] = pauses;
    assert!(
        sorted < n / 4 && scattered >= n,
        "{sorted} and {scattered} pauses"
    );
}

#[test]
fn a_write_dropped_while_it_plans_changes_nothing() {
    // Three levels of 256-byte pages, whose planning pauses.
    let store = Store::new(Options::new().page_size(256));
    let mut model = Model::new();
    let load: Vec<_> = (0..2000).map(|i| scattered_put(i, 1)).collect();
    write(&store.db, &load);
    apply(&mut model, &load);
    assert_eq!(store.db.tree().current().root.unwrap().level, 2);
    let mut ops: Vec<_> = (0..2000).step_by(3).map(|i| scattered_put(i, 2)).collect();
    ops.push(range_op(500, 900));
    store.db.set_merge_slice(Some(4));
    let start = generation(&store.db);
    let mut batch = batch_of(&ops);
    // Dropped at its first pause, normalizing its batch, then once planning has
    // found leaves to load, before it loads them.
    for planning in [false, true] {
        let mut write = Box::pin(store.db.write(&mut batch));
        loop {
            assert!(poll_once(write.as_mut()).is_pending());
            let tree = store.db.tree();
            assert!(tree.pinned.is_empty(), "the write was done planning");
            if !planning || !tree.touched.is_empty() {
                break;
            }
        }
        drop(write);
        store.db.check();
        assert_eq!(generation(&store.db), start);
        assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    }
    // Their batch, reused, writes what it holds.
    now(store.db.write(&mut batch)).unwrap();
    apply(&mut model, &ops);
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    store.db.check();
}

#[test]
fn a_write_dropped_while_it_loads_its_leaves_changes_nothing() {
    // On 256-byte pages, the write changes more leaves than it loads at once.
    let store = Store::new(Options::new().page_size(256));
    let mut model = Model::new();
    let load: Vec<_> = (0..2000).map(|i| scattered_put(i, 1)).collect();
    write(&store.db, &load);
    apply(&mut model, &load);
    let ops: Vec<_> = (0..2000).step_by(3).map(|i| scattered_put(i, 2)).collect();
    store.db.set_merge_slice(Some(4));
    let start = generation(&store.db);
    let mut batch = batch_of(&ops);
    let mut write = Box::pin(store.db.write(&mut batch));
    // The leaves are cached, yet it pauses once it has pinned some of them.
    let io = store.disk.ops();
    loop {
        assert!(poll_once(write.as_mut()).is_pending());
        let tree = store.db.tree();
        if !tree.pinned.is_empty() {
            assert!(tree.pinned.len() < tree.touched.len());
            break;
        }
    }
    assert_eq!(store.disk.ops(), io);
    drop(write);
    store.db.check();
    assert_eq!(generation(&store.db), start);
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    now(store.db.write(&mut batch)).unwrap();
    apply(&mut model, &ops);
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    store.db.check();
}

#[test]
fn a_write_frees_the_pages_it_replaced_a_slice_at_a_time() {
    // On 256-byte pages, the write replaces hundreds of leaves.
    let store = Store::new(Options::new().page_size(256));
    let mut model = Model::new();
    let load: Vec<_> = (0..2000).map(|i| scattered_put(i, 1)).collect();
    write(&store.db, &load);
    apply(&mut model, &load);
    store.db.set_merge_slice(Some(4));
    let left = || store.db.tree().pages.space.dropped().count();
    // One write runs to its end, the next is dropped while it frees the pages, and
    // the one after frees the rest before it merges.
    for (round, drop_it) in [(2, false), (3, true), (4, false)] {
        let ops: Vec<_> = (0..2000)
            .step_by(3)
            .map(|i| scattered_put(i, round))
            .collect();
        let mut batch = batch_of(&ops);
        let mut write = Box::pin(store.db.write(&mut batch));
        if round == 4 {
            pause_merging(&store.db, write.as_mut());
            assert_eq!(left(), 0);
        }
        let start = generation(&store.db);
        while generation(&store.db) == start {
            assert!(poll_once(write.as_mut()).is_pending());
        }
        // Published, it pauses after each slice of the pages it frees.
        let before = left();
        assert!(poll_once(write.as_mut()).is_pending());
        assert!(0 < left() && left() < before, "{} of {before} left", left());
        // A reader that ends meanwhile leaves them to the write.
        let before = left();
        drop(store.db.range(..));
        assert_eq!(left(), before);
        if drop_it {
            drop(write);
            assert_eq!(left(), before);
        } else {
            now(write).unwrap();
            assert_eq!(left(), 0);
        }
        store.db.check();
        apply(&mut model, &ops);
        assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    }
}

#[test]
fn a_reader_leaves_the_pages_it_held_back_to_the_next_write() {
    // On 256-byte pages, each write replaces hundreds of leaves. A twin store takes
    // the same writes, with no reader to hold pages back.
    let opts = Options::new().page_size(256);
    let (store, twin) = (Store::new(opts.clone()), Store::new(opts));
    let load: Vec<_> = (0..2000).map(|i| scattered_put(i, 1)).collect();
    let update = |round| -> Vec<_> {
        (0..2000)
            .step_by(3)
            .map(|i| scattered_put(i, round))
            .collect()
    };
    for db in [&store.db, &twin.db] {
        write(db, &load);
        db.set_merge_slice(Some(4));
    }
    // A reader holds back the pages of the nodes the next write replaces.
    let reader = store.db.range(..);
    for db in [&store.db, &twin.db] {
        write(db, &update(2));
    }
    let left = || store.db.tree().pages.space.dropped().count();
    let held = left();
    assert!(held > 0);
    // Ending, it frees none of them.
    drop(reader);
    assert_eq!(left(), held);
    store.db.check();
    // The next write frees them before it merges, a slice at a time.
    let ops = update(3);
    let mut batch = batch_of(&ops);
    let mut next = Box::pin(store.db.write(&mut batch));
    while left() == held {
        assert!(poll_once(next.as_mut()).is_pending());
    }
    assert!(left() > 0, "freed at once");
    pause_merging(&store.db, next.as_mut());
    assert_eq!(left(), 0);
    now(next).unwrap();
    // So it takes the pages it would have if the reader had freed them.
    write(&twin.db, &ops);
    assert_eq!(store.db.tree().pages(), twin.db.tree().pages());
    store.db.check();
}

/// Polls `write`, each poll stopping where it yields, until it pauses merging,
/// which it does once it holds the leaves its batch changes.
fn pause_merging<F: Future + ?Sized>(db: &Db, mut write: Pin<&mut F>) {
    loop {
        assert!(
            poll_once(write.as_mut()).is_pending(),
            "the write didn't pause"
        );
        if db.tree().merging() {
            return;
        }
    }
}

/// A put of key `i`, written in `round`. Values that share no prefix keep leaves
/// from holding many entries.
fn scattered_put(i: u32, round: u32) -> Op {
    let value = (i.wrapping_mul(0x9e37_79b9) ^ round).to_be_bytes();
    Op::Put(i.to_be_bytes().to_vec(), value.repeat(2))
}

#[test]
fn a_write_dropped_at_any_pause_changes_nothing() {
    let put = scattered_put;
    // Four levels of 256-byte pages: the merge pauses with nodes of every level
    // built, and with unbuilt ones referring to nodes it built or kept.
    let store = Store::new(Options::new().page_size(256));
    let mut model = Model::new();
    for chunk in 0..4 {
        let load: Vec<_> = (chunk * 5000..(chunk + 1) * 5000)
            .map(|i| put(i, 1))
            .collect();
        write(&store.db, &load);
        apply(&mut model, &load);
    }
    assert!(store.db.tree().current().root.unwrap().level >= 3);
    // Rewrites some entries, appends, and removes ranges, some of whole subtrees:
    // nodes split and merge at every level.
    let mut ops: Vec<_> = (0..20_000).step_by(3).map(|i| put(i, 2)).collect();
    ops.extend((20_000..23_000).map(|i| put(i, 3)));
    ops.extend([range_op(5000, 9000), range_op(12_000, 12_100)]);
    store.db.set_merge_slice(Some(256));
    let start = generation(&store.db);
    let mut drops = 0;
    loop {
        let mut batch = batch_of(&ops);
        let mut write = Box::pin(store.db.write(&mut batch));
        // Poll one more time than last time, then drop it.
        let finished = (0..=drops).any(|_| match poll_once(write.as_mut()) {
            Poll::Ready(result) => {
                result.unwrap();
                true
            }
            Poll::Pending => false,
        });
        drop(write);
        // Once published, it pauses only to free the pages of the nodes it
        // replaced, and frees the rest at once when dropped.
        if finished || generation(&store.db) > start {
            break;
        }
        drops += 1;
        store.db.check();
        assert_eq!(generation(&store.db), start);
    }
    assert!(drops > 100, "dropped at {drops} pauses");
    assert_eq!(generation(&store.db), start + 1);
    apply(&mut model, &ops);
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    store.db.check();
}

/// Queues writes of `queued` behind a first write, then has the first of them
/// lead its group until it pauses: normalizing the group's batch, or once
/// `merging`, merging it.
fn pause_group<'a>(
    store: &Store,
    model: &mut Model,
    queued: &'a mut [WriteBatch],
    db: &'a Db,
    merging: bool,
) -> Vec<Write<'a>> {
    let first = puts(0..50, 2);
    let mut writes = queue_behind(store, &first, queued, db);
    apply(model, &first);
    db.set_merge_slice(Some(4));
    let io = store.disk.ops();
    if merging {
        pause_merging(db, writes[0].as_mut());
    } else {
        assert!(poll_once(writes[0].as_mut()).is_pending());
        assert!(
            db.tree().touched.is_empty(),
            "the group's write was planning"
        );
    }
    assert_eq!(store.disk.ops(), io, "the group's write didn't pause");
    writes
}

/// Drops the write leading a group while it pauses, `merging` or normalizing.
fn drop_group_leader(merging: bool) {
    let (store, mut model) = store_of(200);
    let start = generation(&store.db);
    let queued = [rewrite_all(), vec![range_op(100, 120)], puts(180..190, 4)];
    let mut batches: Vec<_> = queued.iter().map(|ops| batch_of(ops)).collect();
    let db = store.db.clone();
    let mut writes = pause_group(&store, &mut model, &mut batches, &db, merging);
    // The leader goes before it writes anything; the next write takes over.
    drop(writes.remove(0));
    for write in writes {
        now(write).unwrap();
    }
    assert!(!store.db.has_failed());
    assert_eq!(generation(&store.db), start + 2);
    apply(&mut model, &queued[1]);
    apply(&mut model, &queued[2]);
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    store.db.check();
}

/// Drops a write of a group while the write leading it pauses, `merging` or
/// normalizing.
fn drop_group_member(merging: bool) {
    let (store, mut model) = store_of(200);
    let start = generation(&store.db);
    let queued = [rewrite_all(), vec![range_op(100, 120)], puts(180..190, 4)];
    let mut batches: Vec<_> = queued.iter().map(|ops| batch_of(ops)).collect();
    let db = store.db.clone();
    let mut writes = pause_group(&store, &mut model, &mut batches, &db, merging);
    // The leader starts over without the range.
    drop(writes.remove(1));
    for write in writes {
        now(write).unwrap();
    }
    assert_eq!(generation(&store.db), start + 2);
    apply(&mut model, &queued[0]);
    apply(&mut model, &queued[2]);
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    store.db.check();
}

#[test]
fn a_write_dropped_while_its_merge_pauses_leaves_its_group_to_the_others() {
    drop_group_leader(true);
}

#[test]
fn a_write_dropped_while_it_normalizes_leaves_its_group_to_the_others() {
    drop_group_leader(false);
}

#[test]
fn a_write_dropped_while_its_group_merges_is_withdrawn() {
    drop_group_member(true);
}

#[test]
fn a_write_dropped_while_its_group_normalizes_is_withdrawn() {
    drop_group_member(false);
}

#[compio::test]
async fn other_tasks_run_while_a_write_merges() {
    let (store, _) = store_of(200);
    let db = store.db.clone();
    db.set_merge_slice(Some(4));
    let writing = Rc::new(Cell::new(true));
    let lookups = Rc::new(Cell::new(0));
    let reader = compio::runtime::spawn({
        let (db, writing, lookups) = (db.clone(), writing.clone(), lookups.clone());
        async move {
            let key = 100u32.to_be_bytes();
            let mut published = false;
            while writing.get() {
                let found = db.get(&key, Comparison::Equal, |_, v| v[0]).await;
                // The old value until the write publishes, then the new one, while
                // it frees the pages it replaced.
                match found.unwrap() {
                    Some(1) if !published => lookups.set(lookups.get() + 1),
                    Some(7) => published = true,
                    found => panic!("found {found:?}"),
                }
                yield_once().await;
            }
        }
    });
    db.write(&mut batch_of(&rewrite_all())).await.unwrap();
    writing.set(false);
    reader.await.unwrap();
    assert!(lookups.get() > 10, "{} lookups ran", lookups.get());
    let found = db
        .get(&100u32.to_be_bytes(), Comparison::Equal, |_, v| v[0])
        .await;
    assert_eq!(found.unwrap(), Some(7));
}

#[test]
fn a_group_holds_the_batches_that_fit() {
    let (store, mut model) = store_of(200);
    let first = puts(0..50, 2);
    let queued = [puts(50..60, 3), puts(55..65, 4), puts(60..70, 5)];
    let mut batches: Vec<_> = queued.iter().map(|ops| batch_of(ops)).collect();
    // Two of them fit in a group, not three.
    store.db.set_group_limit(2 * batches[0].size());
    let start = generation(&store.db);
    let db = store.db.clone();
    let writes = queue_behind(&store, &first, &mut batches, &db);
    // The last, which found no room, commits the full group before its own.
    for write in writes.into_iter().rev() {
        now(write).unwrap();
    }
    assert_eq!(generation(&store.db), start + 3);
    // A batch larger than the limit still joins an empty group.
    store.db.set_group_limit(1);
    let mut again: Vec<_> = queued.iter().map(|ops| batch_of(ops)).collect();
    for write in queue_behind(&store, &first, &mut again, &db) {
        now(write).unwrap();
    }
    assert_eq!(generation(&store.db), start + 7);
    for ops in [&first].into_iter().chain(&queued).cycle().take(8) {
        apply(&mut model, ops);
    }
    assert_eq!(scan(&store.db, ALL), expected_scan(&model, ALL));
    store.db.check();
}

/// A step of readers sharing held reads.
#[derive(Clone, Debug)]
enum Action {
    Scan(Bound<u32>, Bound<u32>),
    Get(u32, Comparison),
    /// Polls a reader once. A scan polls a new `next()` future, which it drops if
    /// it has to wait.
    Poll(usize),
    Drop(usize),
    /// Lets this many more held reads finish.
    Release(u8),
}

fn action() -> impl Strategy<Value = Action> {
    let key = || 0..2100u32;
    let bound = move || {
        prop_oneof![
            Just(Bound::Unbounded),
            key().prop_map(Bound::Included),
            key().prop_map(Bound::Excluded),
        ]
    };
    prop_oneof![
        1 => (bound(), bound()).prop_map(|(lo, hi)| Action::Scan(lo, hi)),
        1 => (key(), prop::sample::select(COMPARISONS.to_vec()))
            .prop_map(|(k, cmp)| Action::Get(k, cmp)),
        6 => any::<usize>().prop_map(Action::Poll),
        1 => any::<usize>().prop_map(Action::Drop),
        2 => (0..20u8).prop_map(Action::Release),
    ]
}

type Lookup<'a> = Pin<Box<dyn Future<Output = crate::Result<Option<Entry>>> + 'a>>;

enum Reader<'a> {
    Scan {
        iter: Box<crate::Iter>,
        range: (Bound<Vec<u8>>, Bound<Vec<u8>>),
        seen: Vec<Entry>,
        done: bool,
    },
    Get {
        lookup: Lookup<'a>,
        want: Option<Entry>,
        got: Option<Option<Entry>>,
    },
}

impl Reader<'_> {
    /// Polls once. Returns false if the reader is done.
    fn poll(&mut self) -> bool {
        match self {
            Reader::Scan { done: true, .. } | Reader::Get { got: Some(_), .. } => false,
            Reader::Scan {
                iter, seen, done, ..
            } => {
                if let Poll::Ready(next) = poll(pin!(iter.next())) {
                    match next.unwrap() {
                        Some((k, v)) => seen.push((k.to_vec(), v.to_vec())),
                        None => *done = true,
                    }
                }
                true
            }
            Reader::Get { lookup, got, .. } => {
                if let Poll::Ready(found) = poll(lookup.as_mut()) {
                    *got = Some(found.unwrap());
                }
                true
            }
        }
    }

    fn check(&self, model: &Model) {
        match self {
            Reader::Scan { range, seen, .. } => {
                let range = (
                    range.0.as_ref().map(Vec::as_slice),
                    range.1.as_ref().map(Vec::as_slice),
                );
                assert_eq!(seen, &expected_scan(model, range), "scan {range:?}");
            }
            Reader::Get { want, got, .. } => assert_eq!(got.as_ref(), Some(want)),
        }
    }
}

proptest! {
    // Scans and lookups share reads that the disk holds and releases bit by bit,
    // and are polled, cancelled and dropped in any order.
    #[test]
    fn readers_share_reads_in_any_order(
        frames in prop::sample::select(vec![8usize, 16, 64]),
        actions in prop::collection::vec(action(), 1..200),
    ) {
        let (store, model) = cold_store(2000, frames);
        store.disk.hold();
        let mut released = store.disk.ops();
        let key = |k: u32| k.to_be_bytes().to_vec();
        let value = |k: &[u8], v: &[u8]| (k.to_vec(), v.to_vec());
        let mut readers: Vec<Reader> = Vec::new();
        for action in &actions {
            match action {
                Action::Scan(lo, hi) => {
                    let range = (lo.map(key), hi.map(key));
                    let bounds = (range.0.as_ref().map(Vec::as_slice), range.1.as_ref().map(Vec::as_slice));
                    let iter = store.db.range(bounds);
                    readers.push(Reader::Scan { iter: Box::new(iter), range, seen: Vec::new(), done: false });
                }
                Action::Get(k, cmp) => {
                    let (k, cmp) = (key(*k), *cmp);
                    let want = expected(&model, &k, cmp);
                    let db = &store.db;
                    let lookup = Box::pin(async move { db.get(&k, cmp, value).await });
                    readers.push(Reader::Get { lookup, want, got: None });
                }
                Action::Poll(i) if !readers.is_empty() => {
                    let i = i % readers.len();
                    if !readers[i].poll() {
                        readers.swap_remove(i).check(&model);
                    }
                }
                Action::Drop(i) if !readers.is_empty() => {
                    readers.swap_remove(i % readers.len());
                }
                Action::Release(n) => {
                    released += u64::from(*n);
                    store.disk.release_through(released.saturating_sub(1));
                }
                Action::Poll(_) | Action::Drop(_) => {}
            }
        }
        store.disk.release();
        for mut reader in readers {
            while reader.poll() {}
            reader.check(&model);
        }
        store.db.check();
    }
}

// The tests on real files need compio's runtime, which Miri can't run.
#[compio::test]
#[cfg_attr(miri, ignore)]
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
    let opts = Options::default().cache_size(16 * 4096);
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

#[compio::test]
#[cfg_attr(miri, ignore)]
async fn a_file_store_has_one_owner_at_a_time() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("encina.db");
    let db = Db::open(&path, Options::default()).await.unwrap();
    let second = Db::open(&path, Options::default()).await.map(drop);
    assert!(matches!(second, Err(Error::Locked)), "{second:?}");

    // The failed open left the store alone.
    let mut batch = WriteBatch::new();
    batch.put(b"key", b"value");
    db.write(&mut batch).await.unwrap();

    // Closing the store lets go of it, and so does dropping it.
    db.close().await.unwrap();
    let db = Db::open(&path, Options::default()).await.unwrap();
    let found = db.get(b"key", Comparison::Equal, |_, v| v.to_vec());
    assert_eq!(found.await.unwrap(), Some(b"value".to_vec()));
    drop(db);
    Db::open(&path, Options::default()).await.unwrap();
}

#[compio::test]
#[cfg_attr(miri, ignore)]
async fn a_read_only_view_opens_beside_the_owner() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("encina.db");
    let db = Db::open(&path, Options::default()).await.unwrap();
    let mut batch = WriteBatch::new();
    batch.put(b"key", b"value");
    db.write(&mut batch).await.unwrap();

    // The view takes no lock, and leaves the owner's in place.
    let view = ReadOnlyDb::open(&path, Options::default()).await.unwrap();
    let second = Db::open(&path, Options::default()).await.map(drop);
    assert!(matches!(second, Err(Error::Locked)), "{second:?}");

    // It reads the version it opened, not later writes.
    batch.clear();
    batch.put(b"later", b"value");
    db.write(&mut batch).await.unwrap();
    let get = |key: &'static [u8]| view.get(key, Comparison::Equal, |_, v| v.to_vec());
    assert_eq!(get(b"key").await.unwrap(), Some(b"value".to_vec()));
    assert_eq!(get(b"later").await.unwrap(), None);

    // Nor does it keep another owner out.
    db.close().await.unwrap();
    Db::open(&path, Options::default()).await.unwrap();
    view.close().await.unwrap();
}

#[cfg(unix)]
#[compio::test]
#[cfg_attr(miri, ignore)]
async fn a_read_only_view_never_writes() {
    use std::os::unix::fs::PermissionsExt;
    use std::{fs, io};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("encina.db");

    // It doesn't create a store.
    let opened = ReadOnlyDb::open(&path, Options::default()).await.map(drop);
    assert!(
        matches!(&opened, Err(Error::Io(e)) if e.kind() == io::ErrorKind::NotFound),
        "{opened:?}"
    );
    assert!(!path.exists());
    fs::write(&path, b"").unwrap();
    let opened = ReadOnlyDb::open(&path, Options::default()).await.map(drop);
    assert!(matches!(opened, Err(Error::InvalidFile(_))), "{opened:?}");
    assert_eq!(fs::metadata(&path).unwrap().len(), 0);
    fs::remove_file(&path).unwrap();

    // It opens a store it may only read.
    let db = Db::open(&path, Options::default()).await.unwrap();
    let mut batch = WriteBatch::new();
    batch.put(b"key", b"value");
    db.write(&mut batch).await.unwrap();
    db.close().await.unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o400)).unwrap();
    let view = ReadOnlyDb::open(&path, Options::default()).await.unwrap();
    let found = view.get(b"key", Comparison::Equal, |_, v| v.to_vec());
    assert_eq!(found.await.unwrap(), Some(b"value".to_vec()));
    view.close().await.unwrap();
}

#[compio::test]
async fn a_store_closes_with_its_last_handle() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("encina.db");
    let db = Db::open(&path, Options::default()).await.unwrap();
    // Closing a clone, or the store while an iterator is alive, leaves it open.
    db.clone().close().await.unwrap();
    let iter = db.range(..);
    db.close().await.unwrap();
    let second = Db::open(&path, Options::default()).await.map(drop);
    assert!(matches!(second, Err(Error::Locked)), "{second:?}");
    // The iterator was the last handle.
    drop(iter);
    Db::open(&path, Options::default()).await.unwrap();
}

#[compio::test]
async fn a_write_dropped_mid_commit_fails_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("encina.db");
    let db = Db::open(&path, Options::default()).await.unwrap();
    let mut batch = WriteBatch::new();
    batch.put(b"key", b"value");
    // The first poll sends out the new leaf's write, and the write is dropped
    // before it finishes.
    assert!(poll(pin!(db.write(&mut batch))).is_pending());
    assert!(matches!(db.write(&mut batch).await, Err(Error::Failed)));
    // Closing waits for the leaf's write. The store reopens as it was, and takes
    // writes again.
    db.close().await.unwrap();
    let db = Db::open(&path, Options::default()).await.unwrap();
    let found = db.get(b"key", Comparison::Equal, |_, v| v.to_vec());
    assert_eq!(found.await.unwrap(), None);
    db.write(&mut batch).await.unwrap();
}

#[compio::test]
#[cfg_attr(miri, ignore)]
async fn concurrent_writes_to_a_file_share_commits() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("encina.db");
    let db = Db::open(&path, Options::default()).await.unwrap();
    let key = |i: u32| i.to_be_bytes();
    let mut batches: Vec<_> = (0..8u32)
        .map(|w| {
            let mut batch = WriteBatch::new();
            for i in 0..1000 {
                batch.put(&key(w * 1000 + i), &key(w));
            }
            batch
        })
        .collect();
    let start = generation(&db);
    let mut writes: Vec<_> = batches
        .iter_mut()
        .map(|batch| Some(Box::pin(db.write(batch))))
        .collect();
    std::future::poll_fn(|cx| {
        for slot in &mut writes {
            if let Some(write) = slot
                && let Poll::Ready(result) = write.as_mut().poll(cx)
            {
                result.unwrap();
                *slot = None;
            }
        }
        match writes.iter().all(Option::is_none) {
            true => Poll::Ready(()),
            false => Poll::Pending,
        }
    })
    .await;
    // The first commits alone, and the seven that queued up behind it together.
    assert_eq!(generation(&db), start + 2);
    drop(writes);
    db.close().await.unwrap();
    let db = Db::open(&path, Options::default()).await.unwrap();
    for i in (0..8000).step_by(97) {
        let found = db.get(&key(i), Comparison::Equal, |_, v| v.to_vec()).await;
        assert_eq!(found.unwrap(), Some(key(i / 1000).to_vec()));
    }
}

#[compio::test]
async fn bad_options_or_a_missing_directory_fail_the_open() {
    let dir = tempfile::tempdir().unwrap();
    let opts = Options::new().page_size(1000);
    let opened = Db::open(dir.path().join("encina.db"), opts).await.map(drop);
    assert!(
        matches!(opened, Err(Error::InvalidOptions(_))),
        "{opened:?}"
    );
    let path = dir.path().join("missing").join("encina.db");
    let opened = Db::open(path, Options::default()).await.map(drop);
    assert!(
        matches!(&opened, Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound),
        "{opened:?}"
    );
}

#[cfg(unix)]
#[compio::test]
#[cfg_attr(miri, ignore)]
async fn a_new_store_needs_its_directory_synced() {
    use std::os::unix::fs::PermissionsExt;
    use std::{fs, io};
    // SAFETY: geteuid has no preconditions.
    if unsafe { libc::geteuid() } == 0 {
        return; // Permissions don't stop root.
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("encina.db");
    let mode = |mode| fs::set_permissions(dir.path(), fs::Permissions::from_mode(mode)).unwrap();

    // Without read permission, a file can be created in the directory, but the
    // directory can't be opened to sync it.
    mode(0o300);
    let result = Db::open(&path, Options::default()).await.map(drop);
    mode(0o700);
    assert!(
        matches!(&result, Err(Error::Io(e)) if e.kind() == io::ErrorKind::PermissionDenied),
        "{result:?}"
    );

    // The file left behind is empty, so opening it again creates the store.
    assert_eq!(fs::metadata(&path).unwrap().len(), 0);
    let db = Db::open(&path, Options::default()).await.unwrap();
    let mut batch = WriteBatch::new();
    batch.put(b"key", b"value");
    db.write(&mut batch).await.unwrap();
    db.close().await.unwrap();
}

#[compio::test]
async fn a_file_store_shrinks_once_its_end_is_free() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("encina.db");
    let len = || std::fs::metadata(&path).unwrap().len();
    let key = |i: u32| i.to_be_bytes();
    let db = Db::open(&path, Options::default()).await.unwrap();
    let mut batch = WriteBatch::new();
    for i in 0..20_000u32 {
        batch.put(&key(i), &key(i * 2));
    }
    db.write(&mut batch).await.unwrap();
    let full = len();
    // Deleting the upper keys frees the end of the file, and the next write cuts it.
    batch.clear();
    batch.delete_range(&key(5_000), &key(20_000));
    db.write(&mut batch).await.unwrap();
    batch.clear();
    batch.put(&key(0), &key(1));
    db.write(&mut batch).await.unwrap();
    assert!(len() < full / 2, "{} of {full} bytes", len());
    db.close().await.unwrap();

    let db = Db::open(&path, Options::default()).await.unwrap();
    let mut iter = db.range(..);
    let mut count = 0u32;
    while let Some((k, v)) = iter.next().await.unwrap() {
        let expected = if count == 0 { key(1) } else { key(count * 2) };
        assert_eq!((k, v), (&key(count)[..], &expected[..]));
        count += 1;
    }
    assert_eq!(count, 5_000);
    drop(iter);
    db.close().await.unwrap();
}

/// xorshift64*: pseudo-random numbers for workloads.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

/// What a series of commits cost.
#[derive(Default)]
struct Usage {
    commits: u64,
    /// Page write requests, and pages written.
    requests: u64,
    pages: u64,
    /// The longest the file got, in pages.
    peak: u64,
}

impl Usage {
    /// Commits `n` batches that `fill` fills on `store`, adding up what they cost.
    fn commit(&mut self, store: &Store, n: u64, mut fill: impl FnMut(&mut WriteBatch)) {
        let page_size = store.db.tree().page_size() as u64;
        let mut batch = WriteBatch::new();
        for _ in 0..n {
            batch.clear();
            fill(&mut batch);
            let (requests, bytes) = store.disk.written();
            now(store.db.write(&mut batch)).unwrap();
            let (requests_after, bytes_after) = store.disk.written();
            // All but the meta page.
            self.requests += requests_after - requests - 1;
            self.pages += (bytes_after - bytes) / page_size - 1;
            self.commits += 1;
            self.peak = self.peak.max(store.disk.len() / page_size);
        }
    }

    /// Prints what the commits cost, and the file and live pages after them.
    fn report(&self, name: &str, store: &Store) {
        let page_size = store.db.tree().page_size() as u64;
        let live = store.db.tree().pages.space.used_count() as u64;
        let file = store.disk.len() / page_size;
        let per_commit = |n: u64| n as f64 / self.commits as f64;
        println!(
            "{name:<24} {live:>8} {file:>8} {:>7.2}x {:>7.2}x {:>10.1} {:>10.1}",
            file as f64 / live as f64,
            self.peak as f64 / live as f64,
            per_commit(self.requests),
            per_commit(self.pages),
        );
    }
}

/// Measures the file's size against the pages in use, and the write requests and
/// pages written per commit, on the simulated disk with 4 KiB pages and a cache
/// that holds everything. Run it with
/// `cargo test --release -- --ignored --nocapture file_size_and_requests`.
///
/// - append: the benchmark's load, BookKeeper-shaped (see bench/README.md).
/// - random: 1M random keys, loaded in batches of 20K, then random updates in
///   commits of 20K, 1K or 10, two million updates in all. Requests and pages
///   are per update commit; the peak counts the load too.
#[test]
#[ignore]
fn file_size_and_requests() {
    let opts = Options::new().cache_size(1 << 30);
    println!(
        "{:<24} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
        "workload", "live", "file", "ratio", "peak", "requests", "pages"
    );

    let store = Store::new(opts.clone());
    let mut usage = Usage::default();
    let (ledgers, entries, rollover) = (1000, 100, 10);
    let mut n = 0;
    usage.commit(&store, 100, |batch| {
        let first = n % rollover * entries;
        for slot in 0..ledgers {
            let ledger = n / rollover * ledgers + slot;
            for entry in first..first + entries {
                batch.put(&entry_key(ledger, entry), &location(ledger, entry));
            }
        }
        n += 1;
    });
    usage.report("append", &store);
    drop(store);

    for (name, size) in [
        ("random, 20K per commit", 20_000),
        ("random, 1K per commit", 1_000),
        ("random, 10 per commit", 10),
    ] {
        let store = Store::new(opts.clone());
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let keys: Vec<_> = (0..1_000_000)
            .map(|_| entry_key(rng.next(), rng.next()))
            .collect();
        let mut chunks = keys.chunks(20_000);
        let mut load = Usage::default();
        load.commit(&store, 50, |batch| {
            for key in chunks.next().unwrap() {
                batch.put(key, &[0; 16]);
            }
        });
        let mut usage = Usage {
            peak: load.peak,
            ..Usage::default()
        };
        let mut round = 0u64;
        usage.commit(&store, 2_000_000 / size, |batch| {
            round += 1;
            for _ in 0..size {
                let key = &keys[(rng.next() % keys.len() as u64) as usize];
                batch.put(key, &location(round, round));
            }
        });
        usage.report(name, &store);
    }
}

/// Measures writes larger than the cache, on the simulated disk with 4 KiB pages.
/// A write pins the leaves it changes, and the leaves it builds, until it
/// publishes, so the frame pool grows past the cache's budget. Run it with
/// `cargo test --release -- --ignored --nocapture writes_larger_than_the_cache`.
///
/// One write loads 1M random keys, and another updates 1M random keys of those,
/// with a cache of 16 MiB, and with one that holds everything. It reports how long
/// each write took, the frames the clock hand looked at in the update, and the
/// frames the pool grew to.
#[test]
#[ignore]
fn writes_larger_than_the_cache() {
    use std::time::Instant;

    println!(
        "{:<8} {:>9} {:>9} {:>9} {:>11} {:>9}",
        "cache", "leaves", "load", "update", "looked", "frames"
    );
    for cache in [16 << 20, 1 << 30] {
        let store = Store::new(Options::new().cache_size(cache));
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let keys: Vec<_> = (0..1_000_000)
            .map(|_| entry_key(rng.next(), rng.next()))
            .collect();
        let mut batch = WriteBatch::new();
        for key in &keys {
            batch.put(key, &[0; 16]);
        }
        let start = Instant::now();
        now(store.db.write(&mut batch)).unwrap();
        let load = start.elapsed();
        batch.clear();
        for round in 0..1_000_000 {
            let key = &keys[(rng.next() % keys.len() as u64) as usize];
            batch.put(key, &location(round, round));
        }
        let looked = store.db.tree().pages.cache.looked;
        let start = Instant::now();
        now(store.db.write(&mut batch)).unwrap();
        let update = start.elapsed();
        let tree = store.db.tree();
        let s = |d: std::time::Duration| format!("{:.3} s", d.as_secs_f64());
        println!(
            "{:<8} {:>9} {:>9} {:>9} {:>11} {:>9}",
            format!("{} MiB", cache >> 20),
            tree.leaves().len(),
            s(load),
            s(update),
            tree.pages.cache.looked - looked,
            tree.pages.pool.slots(),
        );
    }
}

/// Measures how long lookups wait while large writes merge, on the simulated disk
/// with 4 KiB pages and a cache that holds everything, so that lookups only ever
/// wait for the thread. The disk delays writes and syncs while the lookups run, as
/// a real disk's take time. Run it with
/// `cargo test --release -- --ignored --nocapture read_latency_during_a_load`.
///
/// A store of 2M BookKeeper-shaped entries takes writes of 1M entries each, while a
/// lookup of a random key is due every 20 µs: a lookup's latency runs from when it
/// was due until it returns. Writes that never pause show what a merge costs them.
/// It also counts the lookups that missed the cache, and reports the longest a
/// lookup ran without pausing (stretch), and the longest it ran in its last poll
/// (last), in which it lets go of the version it pinned, if it did.
///
/// - append: each write appends 1,000 entries to each of 1,000 ledgers.
/// - random: each write updates 1M random keys.
#[test]
#[ignore]
fn read_latency_during_a_load() {
    latency_header();
    for workload in ["append", "random"] {
        for pause in [false, true] {
            latency_during_writes(workload, pause, false);
        }
    }
}

/// Like [`read_latency_during_a_load`]'s random updates that pause, with lookups
/// that miss the cache. A lookup that misses pins the version it reads while it
/// waits for the disk, and the pages that a write replaces meanwhile stay in use
/// until it ends. Run it with
/// `cargo test --release -- --ignored --nocapture read_latency_with_misses`.
///
/// The lookups read 50K entries of their own, whose values of 2 KiB are too large
/// for a leaf: each has a value leaf, and the cache, of 256 MiB, keeps value leaves
/// in a quarter of it, so that seven lookups in ten miss. The disk's leaf reads
/// take 100 µs, and the next lookups don't wait for one that waits for the disk.
#[test]
#[ignore]
fn read_latency_with_misses_during_a_load() {
    latency_header();
    latency_during_writes("random", true, true);
}

fn latency_header() {
    println!(
        "{:<8} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "workload",
        "slice",
        "write",
        "lookups",
        "missed",
        "p50",
        "p99",
        "p99.9",
        "max",
        "stretch",
        "last"
    );
}

/// Runs [`read_latency_during_a_load`]'s `workload`, with writes that `pause`, and
/// lookups that read the ledgers' entries, or else, if they are to miss the
/// cache, their own.
fn latency_during_writes(workload: &str, pause: bool, misses: bool) {
    use std::cell::RefCell;
    use std::time::{Duration, Instant};

    let (ledgers, preload, writes, cold) = (1000u64, 2000u64, 3u64, 50_000u64);
    let cache = if misses { 256 << 20 } else { 4 << 30 };
    let store = Store::new(Options::new().cache_size(cache));
    store.disk.delay_writes();
    let db = store.db.clone();
    let mut batch = WriteBatch::new();
    for chunk in 0..preload / 100 {
        batch.clear();
        for ledger in 0..ledgers {
            for entry in chunk * 100..(chunk + 1) * 100 {
                batch.put(&entry_key(ledger, entry), &location(ledger, entry));
            }
        }
        now(db.write(&mut batch)).unwrap();
    }
    if misses {
        // After the ledgers' entries.
        batch.clear();
        for i in 0..cold {
            batch.put(&entry_key(ledgers, i), &[i as u8; 2048]);
        }
        now(db.write(&mut batch)).unwrap();
        store.disk.delay_reads(Duration::from_micros(100));
    }
    if !pause {
        db.set_merge_slice(Some(usize::MAX));
    }
    let runtime = compio::runtime::Runtime::new().unwrap();
    let done = Rc::new(Cell::new(false));
    let latencies = Rc::new(RefCell::new(Vec::new()));
    let reader = {
        let (db, done, latencies) = (db.clone(), done.clone(), latencies.clone());
        async move {
            type Lookup = Pin<Box<dyn Future<Output = ()>>>;
            let mut rng = Rng(0x2545_f491_4f6c_dd1d);
            let interval = Duration::from_micros(20);
            let mut due = Instant::now();
            // The lookups waiting for the disk, with when each was due.
            let mut waiting: Vec<(Instant, Lookup)> = Vec::new();
            let (mut missed, mut stretch, mut last) = (0u64, Duration::ZERO, Duration::ZERO);
            while !done.get() || !waiting.is_empty() {
                std::future::poll_fn(|cx| {
                    // Polls a lookup, and returns whether it waits.
                    let mut poll_lookup = |due: Instant, lookup: &mut Lookup| {
                        let start = Instant::now();
                        let polled = lookup.as_mut().poll(cx);
                        let ran = start.elapsed();
                        stretch = stretch.max(ran);
                        if polled.is_ready() {
                            last = last.max(ran);
                            latencies.borrow_mut().push(due.elapsed());
                        }
                        polled.is_pending()
                    };
                    waiting.retain_mut(|(due, lookup)| poll_lookup(*due, lookup));
                    while due <= Instant::now() {
                        let key = if misses {
                            entry_key(ledgers, rng.next() % cold)
                        } else {
                            entry_key(rng.next() % ledgers, rng.next() % preload)
                        };
                        let db = db.clone();
                        let mut lookup: Lookup = Box::pin(async move {
                            db.get(&key, Comparison::Equal, |_, _| ()).await.unwrap();
                        });
                        if poll_lookup(due, &mut lookup) {
                            missed += 1;
                            waiting.push((due, lookup));
                        }
                        due += interval;
                    }
                    Poll::Ready(())
                })
                .await;
                yield_once().await;
            }
            (missed, stretch, last)
        }
    };
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut batches: Vec<_> = (0..writes)
        .map(|round| {
            let mut batch = WriteBatch::new();
            match workload {
                "append" => {
                    let first = preload + round * 1000;
                    for ledger in 0..ledgers {
                        for entry in first..first + 1000 {
                            batch.put(&entry_key(ledger, entry), &location(ledger, entry));
                        }
                    }
                }
                _ => {
                    for _ in 0..1_000_000 {
                        let (ledger, entry) = (rng.next() % ledgers, rng.next() % preload);
                        batch.put(&entry_key(ledger, entry), &location(round, entry));
                    }
                }
            }
            batch
        })
        .collect();
    let (elapsed, (missed, stretch, last)) = runtime.block_on(async {
        let reader = compio::runtime::spawn(reader);
        let mut elapsed = Duration::ZERO;
        for batch in &mut batches {
            let start = Instant::now();
            db.write(batch).await.unwrap();
            elapsed += start.elapsed();
            // Let the lookups that were due catch up before the next write.
            yield_once().await;
        }
        done.set(true);
        (elapsed / writes as u32, reader.await.unwrap())
    });
    let mut latencies = latencies.take();
    latencies.sort_unstable();
    let at = |q: f64| latencies[((latencies.len() - 1) as f64 * q) as usize];
    let ms = |d: Duration| format!("{:.3} ms", d.as_secs_f64() * 1e3);
    let slice = if pause { "1 ms" } else { "never" };
    println!(
        "{:<8} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        workload,
        slice,
        ms(elapsed),
        latencies.len(),
        missed,
        ms(at(0.5)),
        ms(at(0.99)),
        ms(at(0.999)),
        ms(*latencies.last().unwrap()),
        ms(stretch),
        ms(last),
    );
}

/// Lets the runtime run the other tasks once.
async fn yield_once() {
    let mut yielded = false;
    std::future::poll_fn(|cx| {
        if yielded {
            return Poll::Ready(());
        }
        yielded = true;
        cx.waker().wake_by_ref();
        Poll::Pending
    })
    .await
}

// Metrics. nanometer's registry is process-wide, and registers a metric with the
// provider of the first store to create it: the tests share one provider, and tell
// their series apart by an attribute `test`. The SDK reads the clock as it
// collects, which Miri doesn't allow.

/// A `ManualReader` the tests keep a handle to: the provider takes the one it gets.
#[derive(Clone, Debug)]
struct MetricsReader(Arc<ManualReader>);

impl MetricReader for MetricsReader {
    fn register_pipeline(&self, pipeline: Weak<Pipeline>) {
        self.0.register_pipeline(pipeline)
    }
    fn collect(&self, rm: &mut ResourceMetrics) -> OTelSdkResult {
        self.0.collect(rm)
    }
    fn force_flush(&self) -> OTelSdkResult {
        self.0.force_flush()
    }
    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.0.shutdown_with_timeout(timeout)
    }
    fn temporality(&self, kind: InstrumentKind) -> Temporality {
        self.0.temporality(kind)
    }
}

static METRICS: LazyLock<(SdkMeterProvider, MetricsReader)> = LazyLock::new(|| {
    let reader = MetricsReader(Arc::new(ManualReader::builder().build()));
    let provider = SdkMeterProvider::builder()
        .with_reader(reader.clone())
        .build();
    (provider, reader)
});

/// Options with 256-byte pages that export metrics, with the attribute `test` set
/// to `test`.
pub(crate) fn metered(test: &'static str) -> Options {
    Options::new()
        .page_size(256)
        .meter_provider(&METRICS.0)
        .metric_attributes(&[KeyValue::new("test", test)])
}

/// The sum of the points of counter or up-down counter `metric` whose attributes
/// include `test` and `attrs`, or `None` if it has none.
fn sum(test: &str, metric: &str, attrs: &[(&str, &str)]) -> Option<i64> {
    let mut rm = ResourceMetrics::default();
    METRICS.1.collect(&mut rm).unwrap();
    let metric = rm
        .scope_metrics()
        .filter(|sm| sm.scope().name() == "encina")
        .flat_map(|sm| sm.metrics())
        .find(|m| m.name() == metric)?;
    let points: Vec<(Vec<KeyValue>, i64)> = match metric.data() {
        AggregatedMetrics::U64(MetricData::Sum(sum)) => sum
            .data_points()
            .map(|p| (p.attributes().cloned().collect(), p.value() as i64))
            .collect(),
        AggregatedMetrics::I64(MetricData::Sum(sum)) => sum
            .data_points()
            .map(|p| (p.attributes().cloned().collect(), p.value()))
            .collect(),
        data => panic!("{metric:?} is not a sum: {data:?}"),
    };
    let has = |attributes: &[KeyValue], key: &str, value: &str| {
        attributes
            .iter()
            .any(|kv| kv.key.as_str() == key && kv.value.as_str() == value)
    };
    points
        .iter()
        .filter(|(attributes, _)| {
            has(attributes, "test", test) && attrs.iter().all(|&(k, v)| has(attributes, k, v))
        })
        .map(|&(_, value)| value)
        .reduce(|a, b| a + b)
}

/// The cache's hits and misses that `test` counted.
fn lookups(test: &str) -> (i64, i64) {
    let lookups = |result| {
        sum(
            test,
            "encina.cache.lookups",
            &[("encina.cache.result", result)],
        )
        .unwrap()
    };
    (lookups("hit"), lookups("miss"))
}

/// How `test`'s store uses its file: live, held back, and free bytes.
fn file_usage(test: &str) -> [i64; 3] {
    ["live", "held", "free"]
        .map(|state| sum(test, "encina.file.usage", &[("encina.page.state", state)]).unwrap())
}

#[test]
#[cfg_attr(miri, ignore)]
fn metrics_count_writes_and_reads() {
    let test = "writes_and_reads";
    let store = Store::new(metered(test));
    write(
        &store.db,
        &[
            Op::Put(vec![1], vec![1]),
            Op::Put(vec![2], vec![2]),
            Op::Delete(vec![3]),
            Op::DeleteRange(vec![4], vec![5]),
        ],
    );
    let operations = |kind| {
        sum(
            test,
            "encina.write.operations",
            &[("encina.operation.type", kind)],
        )
    };
    assert_eq!(operations("put"), Some(2));
    assert_eq!(operations("delete"), Some(1));
    assert_eq!(operations("delete_range"), Some(1));
    assert_eq!(sum(test, "encina.write.pending", &[]), Some(0));
    // The new leaf and the meta page.
    let written = |metric| sum(test, metric, &[("disk.io.direction", "write")]);
    assert_eq!(written("encina.disk.io"), Some(2 * 256));
    assert_eq!(written("encina.disk.operations"), Some(2));

    assert_eq!(
        get(&store.db, &[1], Comparison::Equal),
        Some((vec![1], vec![1]))
    );
    assert_eq!(scan(&store.db, ALL).len(), 2);
    assert_eq!(sum(test, "encina.range.iterators", &[]), Some(1));
    assert_eq!(sum(test, "encina.range.entries", &[]), Some(2));
    assert_eq!(sum(test, "encina.reader.count", &[]), Some(0));
    // The lookup and the scan found the one leaf cached.
    assert_eq!(lookups(test), (2, 0));
}

#[test]
#[cfg_attr(miri, ignore)]
fn metrics_count_cache_hits_and_misses() {
    let test = "cache_lookups";
    let mut store = Store::new(metered(test).cache_size(1024 * 256));
    let puts: Vec<_> = (0..2000u32)
        .map(|i| Op::Put(i.to_be_bytes().to_vec(), vec![i as u8; 8]))
        .collect();
    write(&store.db, &puts);
    store.reopen();
    let leaves = store.db.tree().leaves().len() as i64;

    // A scan of a cold store misses every leaf, and the next one hits them all.
    let (hits, misses) = lookups(test);
    assert_eq!(scan(&store.db, ALL).len(), 2000);
    assert_eq!(lookups(test), (hits, misses + leaves));
    assert_eq!(scan(&store.db, ALL).len(), 2000);
    assert_eq!(lookups(test), (hits + leaves, misses + leaves));
    assert_eq!(sum(test, "encina.cache.evictions", &[]), Some(0));

    // A lookup that reads its leaf counts one miss, not a hit when it looks again.
    store.reopen();
    let (hits, misses) = lookups(test);
    assert!(get(&store.db, &7u32.to_be_bytes(), Comparison::Equal).is_some());
    assert_eq!(lookups(test), (hits, misses + 1));
    assert!(get(&store.db, &7u32.to_be_bytes(), Comparison::Equal).is_some());
    assert_eq!(lookups(test), (hits + 1, misses + 1));
}

#[test]
#[cfg_attr(miri, ignore)]
fn metrics_count_evictions() {
    let test = "evictions";
    let mut store = Store::new(metered(test).cache_size(16 * 256));
    let puts: Vec<_> = (0..2000u32)
        .map(|i| Op::Put(i.to_be_bytes().to_vec(), vec![i as u8; 8]))
        .collect();
    write(&store.db, &puts);
    store.reopen();
    assert_eq!(scan(&store.db, ALL).len(), 2000);
    assert!(sum(test, "encina.cache.evictions", &[]).unwrap() > 0);
}

#[test]
#[cfg_attr(miri, ignore)]
fn metrics_report_the_memory_and_the_file() {
    let test = "memory_and_file";
    let opts = metered(test).cache_size(64 * 256);
    let disk = SimDisk::new();
    let db = now(Db::open_sim(disk.clone(), opts)).unwrap();
    let puts: Vec<_> = (0..2000u32)
        .map(|i| Op::Put(i.to_be_bytes().to_vec(), vec![i as u8; 8]))
        .collect();
    write(&db, &puts);
    assert_eq!(sum(test, "encina.cache.limit", &[]), Some(64 * 256));
    let usage = |kind| sum(test, "encina.cache.usage", &[("encina.page.type", kind)]).unwrap();
    let in_use = db.tree().pages.pool.in_use() as i64 * 256;
    assert_eq!(usage("internal") + usage("leaf") + usage("value"), in_use);
    assert!(usage("internal") > 0);
    let [live, held, free] = file_usage(test);
    assert_eq!(held, 0);
    assert_eq!(live + held + free, disk.len() as i64);
    assert_eq!(sum(test, "encina.file.size", &[]), Some(disk.len() as i64));
    assert_eq!(sum(test, "encina.entry.count", &[]), Some(2000));
    assert_eq!(sum(test, "encina.entry.size", &[]), Some(2000 * 12));

    // A reader holds back the pages that a write replaces, until a write after it.
    let reader = db.range(ALL);
    assert_eq!(sum(test, "encina.reader.count", &[]), Some(1));
    let replaced: Vec<_> = (0..2000u32)
        .map(|i| Op::Put(i.to_be_bytes().to_vec(), vec![!i as u8; 8]))
        .collect();
    write(&db, &replaced);
    let [live, held, free] = file_usage(test);
    assert!(held > 0);
    assert_eq!(live + held + free, disk.len() as i64);
    drop(reader);
    assert_eq!(sum(test, "encina.reader.count", &[]), Some(0));
    write(&db, &[Op::Put(vec![0], vec![0])]);
    assert_eq!(file_usage(test)[1], 0);
    assert_eq!(sum(test, "encina.entry.count", &[]), Some(2001));
    assert_eq!(sum(test, "encina.entry.size", &[]), Some(2000 * 12 + 2));
    // A range removes leaves without reading them.
    write(&db, &[Op::DeleteRange(vec![0], vec![1])]);
    assert_eq!(sum(test, "encina.entry.count", &[]), Some(0));
    assert_eq!(sum(test, "encina.entry.size", &[]), Some(0));
    assert_eq!(sum(test, "encina.file.size", &[]), Some(disk.len() as i64));

    // A store takes its share back out when it closes.
    drop(db);
    assert_eq!(sum(test, "encina.cache.limit", &[]), Some(0));
    assert_eq!(usage("internal") + usage("leaf") + usage("value"), 0);
    assert_eq!(file_usage(test), [0; 3]);
    assert_eq!(sum(test, "encina.file.size", &[]), Some(0));
}

#[test]
#[cfg_attr(miri, ignore)]
fn metrics_count_errors_and_failures() {
    let test = "errors";
    let store = Store::new(metered(test));
    let errors = |kind| sum(test, "encina.write.errors", &[("error.type", kind)]);
    let mut batch = WriteBatch::new();
    batch.put(&[0; 1024], &[]);
    assert!(matches!(
        now(store.db.write(&mut batch)),
        Err(Error::KeyTooLarge { .. })
    ));
    assert_eq!(errors("key_too_large"), Some(1));
    assert_eq!(sum(test, "encina.store.failed", &[]), Some(0));

    // A write that fails to sync its leaf, the operation after writing it, leaves
    // the store failed.
    store.disk.inject(store.disk.ops() + 1, Fault::Error);
    let mut batch = batch_of(&[Op::Put(vec![1], vec![1])]);
    assert!(matches!(now(store.db.write(&mut batch)), Err(Error::Io(_))));
    assert_eq!(errors("io"), Some(1));
    assert_eq!(sum(test, "encina.store.failed", &[]), Some(1));
    assert!(matches!(
        now(store.db.write(&mut batch)),
        Err(Error::Failed)
    ));
    assert_eq!(errors("failed"), Some(1));
    assert_eq!(sum(test, "encina.write.pending", &[]), Some(0));
    drop(store);
    assert_eq!(sum(test, "encina.store.failed", &[]), Some(0));
}

#[test]
#[cfg_attr(miri, ignore)]
fn metrics_of_a_view_leave_out_writes_and_the_file() {
    let store = Store::new(Options::new().page_size(256));
    write(&store.db, &[Op::Put(vec![1], vec![1])]);
    let test = "view";
    let view = now(ReadOnlyDb::open_sim(store.disk.clone(), metered(test))).unwrap();
    assert!(
        now(view.get(&[1], Comparison::Equal, |_, v| v.to_vec()))
            .unwrap()
            .is_some()
    );
    assert!(sum(test, "encina.cache.limit", &[]).is_some());
    assert_eq!(lookups(test).0 + lookups(test).1, 1);
    for metric in [
        "encina.write.pending",
        "encina.entry.count",
        "encina.file.usage",
        "encina.store.failed",
    ] {
        assert_eq!(sum(test, metric, &[]), None, "{metric}");
    }
}

#[compio::test]
#[cfg_attr(miri, ignore)]
async fn metrics_name_the_store_by_its_path() {
    let dir = tempfile::tempdir().unwrap();
    let test = "store_path";
    let store = |path: &str, attr: &str| {
        sum(
            test,
            "encina.write.operations",
            &[("encina.store", path), ("encina.operation.type", attr)],
        )
    };
    // Files take pages of 4 KiB at least.
    let opts = metered(test).page_size(4096);
    let path = dir.path().join("a.db");
    let db = Db::open(&path, opts.clone()).await.unwrap();
    db.write(&mut batch_of(&[Op::Put(vec![1], vec![1])]))
        .await
        .unwrap();
    assert_eq!(store(path.to_str().unwrap(), "put"), Some(1));

    // The options can name it otherwise.
    let opts = opts.metric_attributes(&[KeyValue::new("encina.store", "ledgers")]);
    let db = Db::open(dir.path().join("b.db"), opts).await.unwrap();
    db.write(&mut batch_of(&[Op::Delete(vec![1])]))
        .await
        .unwrap();
    assert_eq!(store("ledgers", "delete"), Some(1));
}
