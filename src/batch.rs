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

use std::mem;

use crate::error::{Error, Result};

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
    /// `key.len() + value.len()` may not exceed
    /// [`Options::max_entry_size`](crate::Options::max_entry_size), a bit under a
    /// quarter of the page size. Otherwise [`Db::write`](crate::Db::write) rejects
    /// the whole batch with [`Error::EntryTooLarge`].
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

    /// Checks that no put's key plus value exceeds `max_entry` bytes.
    pub(crate) fn check(&self, max_entry: usize) -> Result<()> {
        for op in &self.ops {
            let size = (op.a_len + op.b_len) as usize;
            if op.kind == Kind::Put && size > max_entry {
                return Err(Error::EntryTooLarge {
                    size,
                    max: max_entry,
                });
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

    /// Reduces the batch to an equivalent set of ranges to remove first and puts
    /// and deletes to apply after, at most one per key.
    pub(crate) fn prepare(&mut self) {
        let (bytes, ops) = (&self.bytes, &self.ops);
        let a = |i: u32| {
            let op = &ops[i as usize];
            slice(bytes, op.a, op.a_len)
        };
        let b = |i: u32| {
            let op = &ops[i as usize];
            slice(bytes, op.b, op.b_len)
        };

        let n = ops.len() as u32;
        self.by_start.clear();
        self.by_start
            .extend((0..n).filter(|&i| ops[i as usize].kind == Kind::DeleteRange && a(i) < b(i)));
        self.by_start
            .sort_unstable_by(|&x, &y| a(x).cmp(a(y)).then(x.cmp(&y)));

        self.points.clear();
        self.points.extend(
            (0..n).filter(|&i| matches!(ops[i as usize].kind, Kind::Put | Kind::Delete)),
        );
        self.points
            .sort_unstable_by(|&x, &y| a(x).cmp(a(y)).then(x.cmp(&y)));
        // The last operation on a key wins, unless a range added after it covers the
        // key: then the range has the last word, and it is applied first anyway.
        let by_start = &self.by_start;
        let covered_later = |op: u32| {
            let key = a(op);
            let candidates = by_start.partition_point(|&r| a(r) <= key);
            by_start[..candidates].iter().any(|&r| r > op && key < b(r))
        };
        let mut kept = 0;
        for i in 0..self.points.len() {
            let op = self.points[i];
            let last = i + 1 == self.points.len() || a(self.points[i + 1]) != a(op);
            if last && !covered_later(op) {
                self.points[kept] = op;
                kept += 1;
            }
        }
        self.points.truncate(kept);

        self.ranges.clear();
        for &r in &self.by_start {
            if let Some(last) = self.ranges.last_mut()
                && a(r) <= b(last.1)
            {
                if b(r) > b(last.1) {
                    last.1 = r;
                }
                continue;
            }
            self.ranges.push((r, r));
        }
    }
}

fn slice(bytes: &[u8], at: u32, len: u32) -> &[u8] {
    &bytes[at as usize..(at + len) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

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
        batch.prepare();
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
        batch.prepare();
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
        batch.prepare();
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
        group.prepare();
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
        group.prepare();
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
    fn oversized_entries_are_rejected() {
        let mut batch = WriteBatch::new();
        batch.put(b"key", &[0; 10]);
        assert!(matches!(
            batch.check(12),
            Err(Error::EntryTooLarge { size: 13, max: 12 })
        ));
        batch.clear();
        batch.put(b"key", &[0; 9]);
        batch.check(12).unwrap();
    }
}
