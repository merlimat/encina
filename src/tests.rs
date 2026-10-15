//! Store tests: a `BTreeMap` model, and BookKeeper-shaped scenarios.

use std::collections::BTreeMap;
use std::future::Future;
use std::ops::Bound;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

use proptest::prelude::*;

use crate::{Comparison, Db, Error, Options, WriteBatch};

/// Runs a future that must finish without waiting, as everything does while the
/// store lives only in memory.
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

#[derive(Clone, Debug)]
enum Op {
    Put(Vec<u8>, Vec<u8>),
    Delete(Vec<u8>),
    DeleteRange(Vec<u8>, Vec<u8>),
}

fn write(db: &Db, ops: &[Op]) {
    let mut batch = WriteBatch::new();
    for op in ops {
        match op {
            Op::Put(k, v) => batch.put(k, v),
            Op::Delete(k) => batch.delete(k),
            Op::DeleteRange(start, end) => batch.delete_range(start, end),
        }
    }
    now(db.write(&mut batch)).unwrap();
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
            Step::Get(..) | Step::Scan(..) => Vec::new(),
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
    let comparison = prop::sample::select(vec![
        Comparison::Equal,
        Comparison::Floor,
        Comparison::Ceiling,
        Comparison::Lower,
        Comparison::Higher,
    ]);
    prop_oneof![
        4 => prop::collection::vec(op(max_value), 1..40).prop_map(Step::Write),
        1 => (key(), any::<u16>(), 1..300u16, 0..=max_value)
            .prop_map(|(prefix, first, count, value_len)| Step::Load { prefix, first, count, value_len }),
        3 => (key(), comparison).prop_map(|(k, cmp)| Step::Get(k, cmp)),
        1 => (bound(), bound()).prop_map(|(lo, hi)| Step::Scan(lo, hi)),
        1 => prop::collection::vec(op(max_value), 1..40).prop_map(Step::WriteDuringScan),
    ]
}

fn run(page_size: usize, steps: &[Step]) {
    let db = Db::in_memory(Options::new().page_size(page_size)).unwrap();
    let mut model = Model::new();
    for step in steps {
        match step {
            Step::Write(_) | Step::Load { .. } => {
                let ops = step.ops();
                write(&db, &ops);
                apply(&mut model, &ops);
                db.tree().check();
            }
            Step::Get(key, cmp) => {
                let got = get(&db, key, *cmp);
                assert_eq!(got, expected(&model, key, *cmp), "get({key:?}, {cmp:?})");
            }
            Step::Scan(lo, hi) => {
                let range = (
                    lo.as_ref().map(Vec::as_slice),
                    hi.as_ref().map(Vec::as_slice),
                );
                assert_eq!(
                    scan(&db, range),
                    expected_scan(&model, range),
                    "scan {range:?}"
                );
            }
            Step::WriteDuringScan(ops) => {
                let before = expected_scan(&model, ALL);
                let mut iter = db.range(..);
                let mut seen = Vec::new();
                if let Some((k, v)) = now(iter.next()).unwrap() {
                    seen.push((k.to_vec(), v.to_vec()));
                }
                write(&db, ops);
                apply(&mut model, ops);
                while let Some((k, v)) = now(iter.next()).unwrap() {
                    seen.push((k.to_vec(), v.to_vec()));
                }
                assert_eq!(seen, before, "the scan saw a later write");
                drop(iter);
                db.tree().check();
            }
        }
    }
    assert_eq!(scan(&db, ALL), expected_scan(&model, ALL));
    // Every comparison, on every key and on the gap just above it, which covers the
    // first and last keys of every leaf.
    for key in model.keys() {
        let mut above = key.clone();
        above.push(0);
        for probe in [key, &above] {
            for cmp in [
                Comparison::Equal,
                Comparison::Floor,
                Comparison::Ceiling,
                Comparison::Lower,
                Comparison::Higher,
            ] {
                assert_eq!(
                    get(&db, probe, cmp),
                    expected(&model, probe, cmp),
                    "get({probe:?}, {cmp:?})"
                );
            }
        }
    }
}

// Set PROPTEST_CASES to run more cases than the default 256.
proptest! {
    // Small pages grow deep trees from few entries.
    #[test]
    fn matches_the_model_with_small_pages(steps in prop::collection::vec(step(12), 1..40)) {
        run(256, &steps);
    }

    // Large values fill default pages quickly enough to grow internal levels.
    #[test]
    fn matches_the_model_with_default_pages(steps in prop::collection::vec(step(200), 1..40)) {
        run(4096, &steps);
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
    let db = Db::in_memory(Options::default()).unwrap();
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
        now(db.write(&mut batch)).unwrap();
        db.tree().check();
    }
    for ledger in 0..ledgers {
        let (id, entries) = (1000 + ledger, next_entry[ledger as usize]);
        for entry in (0..entries).step_by(7) {
            let found = now(db.get(&entry_key(id, entry), Comparison::Equal, |_, v| v.to_vec()));
            assert_eq!(found.unwrap(), Some(location(ledger, entry).to_vec()));
        }
        let last = now(db.get(&entry_key(id, u64::MAX), Comparison::Floor, |k, _| {
            decode_key(k)
        }));
        assert_eq!(last.unwrap(), Some((id, entries - 1)));
    }
    let (leaves, used) = db.tree().leaf_usage();
    let fill = used as f64 / (leaves * db.tree().page_size()) as f64;
    assert!(fill > 0.8, "leaves are only {:.0}% full", fill * 100.0);

    // Deleting ledgers removes their entries and nothing else.
    batch.clear();
    for ledger in (0..ledgers).step_by(2) {
        batch.delete_range(&entry_key(1000 + ledger, 0), &entry_key(1001 + ledger, 0));
    }
    now(db.write(&mut batch)).unwrap();
    db.tree().check();
    for ledger in 0..ledgers {
        let (start, end) = (entry_key(1000 + ledger, 0), entry_key(1001 + ledger, 0));
        let found = scan(&db, (Bound::Included(&start), Bound::Excluded(&end))).len() as u64;
        assert_eq!(
            found,
            if ledger % 2 == 0 {
                0
            } else {
                next_entry[ledger as usize]
            }
        );
    }
}

#[test]
fn comparisons_at_the_edges() {
    use Comparison::{Ceiling, Equal, Floor, Higher, Lower};
    let db = Db::in_memory(Options::default()).unwrap();
    for cmp in [Equal, Floor, Ceiling, Lower, Higher] {
        assert_eq!(get(&db, b"x", cmp), None);
    }
    write(&db, &[10u8, 20, 30].map(|k| Op::Put(vec![k], vec![k / 10])));
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
            assert_eq!(get(&db, &[key], cmp), want, "get({key}, {cmp:?})");
        }
    }
}

#[test]
fn oversized_entries_leave_the_store_unchanged() {
    let db = Db::in_memory(Options::default()).unwrap();
    let max = Options::default().max_entry_size();
    let mut batch = WriteBatch::new();
    batch.put(b"a", b"1");
    batch.put(b"big", &vec![0; max - 2]);
    assert!(matches!(
        now(db.write(&mut batch)),
        Err(Error::EntryTooLarge { .. })
    ));
    assert_eq!(get(&db, b"a", Comparison::Equal), None);
    batch.clear();
    batch.put(b"big", &vec![0; max - 3]);
    now(db.write(&mut batch)).unwrap();
    assert!(get(&db, b"big", Comparison::Equal).is_some());
}

#[test]
fn iterators_keep_their_version() {
    let db = Db::in_memory(Options::new().page_size(256)).unwrap();
    write(
        &db,
        &(0..200u16)
            .map(|i| Op::Put(i.to_be_bytes().to_vec(), vec![1]))
            .collect::<Vec<_>>(),
    );
    let mut iter = db.range(..);
    write(
        &db,
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
    db.tree().check();
    assert_eq!(scan(&db, ALL).len(), 100);
}

#[test]
fn deleting_everything_empties_the_tree() {
    let db = Db::in_memory(Options::new().page_size(256)).unwrap();
    write(
        &db,
        &(0..500u16)
            .map(|i| Op::Put(i.to_be_bytes().to_vec(), vec![0; 8]))
            .collect::<Vec<_>>(),
    );
    write(&db, &[Op::DeleteRange(vec![], vec![0xff, 0xff, 0xff])]);
    db.tree().check();
    assert!(db.tree().root().is_none());
    assert!(scan(&db, ALL).is_empty());
}
