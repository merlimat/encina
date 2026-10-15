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

//! The slotted node format: one node per page, searched in place.
//!
//! ```text
//! offset  size  field
//!      0     4  checksum (set when the page is written to disk; 0 in memory)
//!      4     1  format, always FORMAT_SLOTTED for now
//!      5     1  level: 0 for leaves
//!      6     2  entry count
//!      8     2  prefix length
//!     10     2  offset of the heads array
//!     12     2  offset of the slot array
//!     14     2  reserved
//!     16     4  bytes used
//!     20     8  generation (set when the page is written to disk)
//!     28     4  reserved
//!     32        key prefix | padding to 4 | heads | slots | heap
//! ```
//!
//! Every key in a node starts with the node's prefix: the longest prefix its first
//! and last keys share, stored once. The rest of a key is its suffix. The first four
//! suffix bytes, zero-padded and read as a big-endian `u32`, are the key's head. The
//! heads sit in one array, so a binary search touches nothing else. A slot holds the
//! entry's record offset (relative to the heap), suffix length and value length. The
//! record holds the suffix bytes past the head, then the value.
//!
//! Internal nodes use the same format. Their values are child references,
//! [`CHILD_REF`] bytes: the child's frame, then its page id once it has one.

use std::cmp::Ordering;

use crate::pool::FrameId;

pub(crate) const HEADER: usize = 32;
const HEAD: usize = 4;
const SLOT: usize = 6;
pub(crate) const CHILD_REF: usize = 12;
const FORMAT_SLOTTED: u8 = 1;

/// The largest key plus value that pages of `page_size` bytes accept: small enough
/// that any four such entries, or four child references with keys that long, fit
/// in one node.
pub(crate) fn max_entry(page_size: usize) -> usize {
    (page_size - HEADER) / 4 - HEAD - SLOT - CHILD_REF
}

/// The frame in a child reference to a leaf. Leaves can be evicted, so they are
/// found by page id, through the leaf cache.
pub(crate) const NO_FRAME: FrameId = FrameId::MAX;

/// Encodes a child reference for an internal node: the child's frame, or
/// [`NO_FRAME`] for a leaf, then its page id. Frames are only meaningful in
/// memory: loading an internal node rewrites them.
pub(crate) fn child_ref(frame: FrameId, page: u64) -> [u8; CHILD_REF] {
    let mut r = [0; CHILD_REF];
    r[..4].copy_from_slice(&frame.to_le_bytes());
    r[4..].copy_from_slice(&page.to_le_bytes());
    r
}

/// The frame a child reference points to.
pub(crate) fn child_frame(child_ref: &[u8]) -> FrameId {
    u32::from_le_bytes(child_ref[..4].try_into().unwrap())
}

/// The page a child reference points to.
pub(crate) fn child_page(child_ref: &[u8]) -> u64 {
    u64::from_le_bytes(child_ref[4..CHILD_REF].try_into().unwrap())
}

/// Stamps `generation` and the checksum into the header of the node in `page`,
/// ready to be written.
pub(crate) fn seal(page: &mut [u8], generation: u64) {
    page[20..28].copy_from_slice(&generation.to_le_bytes());
    let crc = crc32c::crc32c(&page[4..]);
    page[..4].copy_from_slice(&crc.to_le_bytes());
}

/// Whether `page`, as read from disk, holds an intact node: its checksum matches,
/// and its layout stays within the page.
pub(crate) fn verify(page: &[u8]) -> bool {
    let crc = u32::from_le_bytes(page[..4].try_into().unwrap());
    if page[4] != FORMAT_SLOTTED || crc32c::crc32c(&page[4..]) != crc {
        return false;
    }
    let node = Node::new(page);
    node.count > 0 && node.heap_at <= node.used() && node.used() <= page.len()
}

/// Points entry `i` of the internal node in `page` at `frame`.
pub(crate) fn set_child_frame(page: &mut [u8], i: usize, frame: FrameId) {
    let (at, _) = Node::new(page).value_span(i);
    page[at..at + 4].copy_from_slice(&frame.to_le_bytes());
}

/// A sorted run of entries to build nodes from.
pub(crate) trait Entries {
    fn key(&self, i: usize) -> &[u8];
    fn value(&self, i: usize) -> &[u8];
}

/// Bytes a node holding entries `a..b` of `src` takes.
pub(crate) fn size(src: &impl Entries, a: usize, b: usize) -> usize {
    let p = lcp(src.key(a), src.key(b - 1));
    let mut size = align4(HEADER + p) + (b - a) * (HEAD + SLOT);
    for i in a..b {
        size += rest_len(src.key(i).len() - p) + src.value(i).len();
    }
    size
}

/// The largest `e` in `a + 1..=b` such that entries `a..e` fit in `limit` bytes,
/// or `a + 1` if not even one entry does.
pub(crate) fn fit(src: &impl Entries, a: usize, b: usize, limit: usize) -> usize {
    let first = src.key(a);
    let mut prefix = first.len();
    // Record bytes (suffix past the head, plus value) of entries a..e.
    let mut records = src.value(a).len();
    let mut e = a + 1;
    while e < b {
        let key = src.key(e);
        let p = prefix.min(lcp(first, key));
        let mut r = records;
        if p != prefix {
            // A shorter prefix lengthens every suffix: recount.
            r = (a..e)
                .map(|i| rest_len(src.key(i).len() - p) + src.value(i).len())
                .sum();
        }
        r += rest_len(key.len() - p) + src.value(e).len();
        if align4(HEADER + p) + (e + 1 - a) * (HEAD + SLOT) + r > limit {
            break;
        }
        prefix = p;
        records = r;
        e += 1;
    }
    e
}

/// Writes entries `a..b` of `src`, which must be sorted and fit in `page`, as a node
/// at `level`. Returns the bytes used.
pub(crate) fn build(page: &mut [u8], level: u8, src: &impl Entries, a: usize, b: usize) -> usize {
    let n = b - a;
    let first = src.key(a);
    let p = lcp(first, src.key(b - 1));
    let heads_at = align4(HEADER + p);
    let slots_at = heads_at + n * HEAD;
    let heap_at = slots_at + n * SLOT;
    page[..HEADER].fill(0);
    page[4] = FORMAT_SLOTTED;
    page[5] = level;
    put_u16(page, 6, n);
    put_u16(page, 8, p);
    put_u16(page, 10, heads_at);
    put_u16(page, 12, slots_at);
    page[HEADER..HEADER + p].copy_from_slice(&first[..p]);
    page[HEADER + p..heads_at].fill(0);
    let mut at = heap_at;
    for (j, i) in (a..b).enumerate() {
        let suffix = &src.key(i)[p..];
        let value = src.value(i);
        page[heads_at + j * HEAD..][..HEAD].copy_from_slice(&head_of(suffix).to_le_bytes());
        let slot = slots_at + j * SLOT;
        put_u16(page, slot, at - heap_at);
        put_u16(page, slot + 2, suffix.len());
        put_u16(page, slot + 4, value.len());
        let rest = &suffix[suffix.len().min(HEAD)..];
        page[at..at + rest.len()].copy_from_slice(rest);
        at += rest.len();
        page[at..at + value.len()].copy_from_slice(value);
        at += value.len();
    }
    page[16..20].copy_from_slice(&u32::try_from(at).unwrap().to_le_bytes());
    at
}

/// A read-only view of a node in a page.
#[derive(Clone, Copy)]
pub(crate) struct Node<'a> {
    page: &'a [u8],
    count: usize,
    prefix_len: usize,
    heads_at: usize,
    slots_at: usize,
    heap_at: usize,
}

impl<'a> Node<'a> {
    pub(crate) fn new(page: &'a [u8]) -> Self {
        debug_assert_eq!(page[4], FORMAT_SLOTTED);
        let count = u16_at(page, 6);
        let slots_at = u16_at(page, 12);
        Node {
            page,
            count,
            prefix_len: u16_at(page, 8),
            heads_at: u16_at(page, 10),
            slots_at,
            heap_at: slots_at + count * SLOT,
        }
    }

    pub(crate) fn level(&self) -> u8 {
        self.page[5]
    }

    pub(crate) fn count(&self) -> usize {
        self.count
    }

    pub(crate) fn used(&self) -> usize {
        u32::from_le_bytes(self.page[16..20].try_into().unwrap()) as usize
    }

    fn prefix(&self) -> &'a [u8] {
        &self.page[HEADER..HEADER + self.prefix_len]
    }

    fn head(&self, i: usize) -> u32 {
        u32::from_le_bytes(
            self.page[self.heads_at + i * HEAD..][..HEAD]
                .try_into()
                .unwrap(),
        )
    }

    /// Absolute record offset, suffix length and value length of entry `i`.
    fn slot(&self, i: usize) -> (usize, usize, usize) {
        let s = self.slots_at + i * SLOT;
        let p = self.page;
        (
            self.heap_at + u16_at(p, s),
            u16_at(p, s + 2),
            u16_at(p, s + 4),
        )
    }

    /// The suffix bytes of entry `i` past its head, and its full suffix length.
    fn rest(&self, i: usize) -> (&'a [u8], usize) {
        let (at, slen, _) = self.slot(i);
        (&self.page[at..at + rest_len(slen)], slen)
    }

    /// Offset and length of entry `i`'s value within the page.
    pub(crate) fn value_span(&self, i: usize) -> (usize, usize) {
        let (at, slen, vlen) = self.slot(i);
        (at + rest_len(slen), vlen)
    }

    pub(crate) fn value(&self, i: usize) -> &'a [u8] {
        let (at, len) = self.value_span(i);
        &self.page[at..at + len]
    }

    /// The child an internal node's entry `i` refers to.
    pub(crate) fn child(&self, i: usize) -> FrameId {
        child_frame(self.value(i))
    }

    /// The page of the child an internal node's entry `i` refers to.
    pub(crate) fn child_page(&self, i: usize) -> u64 {
        child_page(self.value(i))
    }

    /// The generation that wrote this node, once it has been sealed.
    pub(crate) fn generation(&self) -> u64 {
        u64::from_le_bytes(self.page[20..28].try_into().unwrap())
    }

    /// Appends entry `i`'s key to `out`.
    pub(crate) fn append_key(&self, i: usize, out: &mut Vec<u8>) {
        let (rest, slen) = self.rest(i);
        out.extend_from_slice(self.prefix());
        out.extend_from_slice(&self.head(i).to_be_bytes()[..slen.min(HEAD)]);
        out.extend_from_slice(rest);
    }

    /// Replaces the contents of `out` with entry `i`'s key.
    pub(crate) fn key_into(&self, i: usize, out: &mut Vec<u8>) {
        out.clear();
        self.append_key(i, out);
    }

    /// Compares entry `i`'s key with `key`.
    pub(crate) fn cmp_key(&self, i: usize, key: &[u8]) -> Ordering {
        let p = self.prefix_len;
        let m = key.len().min(p);
        match self.prefix()[..m].cmp(&key[..m]) {
            Ordering::Equal if key.len() < p => return Ordering::Greater,
            Ordering::Equal => {}
            other => return other,
        }
        let key = &key[p..];
        let (rest, slen) = self.rest(i);
        let head = self.head(i).to_be_bytes();
        let head = &head[..slen.min(HEAD)];
        let m = head.len().min(key.len());
        match head[..m].cmp(&key[..m]) {
            Ordering::Equal if key.len() < head.len() => Ordering::Greater,
            Ordering::Equal => rest.cmp(&key[head.len()..]),
            other => other,
        }
    }

    /// The index of the first entry whose key is `>= key`, and whether it is equal.
    pub(crate) fn search(&self, key: &[u8]) -> (usize, bool) {
        let p = self.prefix_len;
        let m = key.len().min(p);
        match key[..m].cmp(&self.prefix()[..m]) {
            // A key shorter than the prefix, but matching it, sorts before every entry.
            Ordering::Equal if key.len() < p => return (0, false),
            Ordering::Equal => {}
            Ordering::Less => return (0, false),
            Ordering::Greater => return (self.count, false),
        }
        let suffix = &key[p..];
        let head = head_of(suffix);
        let target = (&suffix[suffix.len().min(HEAD)..], suffix.len());
        let mut i = lower_bound(self.count, |i| self.head(i) < head);
        // Keys sharing a head are ordered by the rest of the suffix, then length.
        while i < self.count && self.head(i) == head {
            match self.rest(i).cmp(&target) {
                Ordering::Less => i += 1,
                Ordering::Equal => return (i, true),
                Ordering::Greater => break,
            }
        }
        (i, false)
    }
}

/// Entries copied into one buffer, for rebuilding nodes.
#[derive(Default)]
pub(crate) struct EntryBuf {
    bytes: Vec<u8>,
    // Key offset, key length and value length; the value follows the key.
    items: Vec<[u32; 3]>,
}

impl EntryBuf {
    pub(crate) fn clear(&mut self) {
        self.bytes.clear();
        self.items.clear();
    }

    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }

    pub(crate) fn push(&mut self, key: &[u8], value: &[u8]) {
        let at = self.bytes.len();
        self.bytes.extend_from_slice(key);
        self.bytes.extend_from_slice(value);
        self.items
            .push([at as u32, key.len() as u32, value.len() as u32]);
    }

    /// Copies entry `i` of `node`.
    pub(crate) fn push_from(&mut self, node: &Node, i: usize) {
        let at = self.bytes.len();
        node.append_key(i, &mut self.bytes);
        let key_len = self.bytes.len() - at;
        let value = node.value(i);
        self.bytes.extend_from_slice(value);
        self.items
            .push([at as u32, key_len as u32, value.len() as u32]);
    }
}

impl Entries for EntryBuf {
    fn key(&self, i: usize) -> &[u8] {
        let [at, len, _] = self.items[i];
        &self.bytes[at as usize..(at + len) as usize]
    }

    fn value(&self, i: usize) -> &[u8] {
        let [at, key_len, len] = self.items[i];
        let at = (at + key_len) as usize;
        &self.bytes[at..at + len as usize]
    }
}

/// Branch-free binary search: the first index in `0..n` for which `less` is false.
fn lower_bound(n: usize, mut less: impl FnMut(usize) -> bool) -> usize {
    if n == 0 {
        return 0;
    }
    let (mut base, mut size) = (0, n);
    while size > 1 {
        let half = size / 2;
        if less(base + half) {
            base += half;
        }
        size -= half;
    }
    base + usize::from(less(base))
}

fn lcp(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn align4(n: usize) -> usize {
    (n + 3) & !3
}

fn head_of(suffix: &[u8]) -> u32 {
    let mut h = [0; HEAD];
    let n = suffix.len().min(HEAD);
    h[..n].copy_from_slice(&suffix[..n]);
    u32::from_be_bytes(h)
}

fn rest_len(suffix_len: usize) -> usize {
    suffix_len.saturating_sub(HEAD)
}

fn u16_at(page: &[u8], at: usize) -> usize {
    u16::from_le_bytes([page[at], page[at + 1]]) as usize
}

fn put_u16(page: &mut [u8], at: usize, v: usize) {
    page[at..at + 2].copy_from_slice(&u16::try_from(v).unwrap().to_le_bytes());
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use proptest::prelude::*;

    use super::*;

    fn keys() -> impl Strategy<Value = Vec<u8>> {
        prop::collection::vec(prop::sample::select(&[0u8, 1, 2, 0x7f, 0xff][..]), 0..=10)
    }

    proptest! {
        #[test]
        fn search_and_access_match_a_sorted_vec(
            map in prop::collection::btree_map(keys(), prop::collection::vec(any::<u8>(), 0..8), 1..40),
            probes in prop::collection::vec(keys(), 1..20),
        ) {
            let entries: Vec<(Vec<u8>, Vec<u8>)> = map.into_iter().collect();
            let mut buf = EntryBuf::default();
            for (k, v) in &entries {
                buf.push(k, v);
            }
            let mut page = vec![0xaa; 4096];
            let used = build(&mut page, 0, &buf, 0, buf.len());
            prop_assert_eq!(used, size(&buf, 0, buf.len()));
            let node = Node::new(&page);
            prop_assert_eq!(node.count(), entries.len());
            prop_assert_eq!(node.used(), used);
            let mut key = Vec::new();
            for (i, (k, v)) in entries.iter().enumerate() {
                node.key_into(i, &mut key);
                prop_assert_eq!(&key, k);
                prop_assert_eq!(node.value(i), &v[..]);
                prop_assert_eq!(node.search(k), (i, true));
            }
            for probe in &probes {
                let lb = entries.partition_point(|(k, _)| k < probe);
                let eq = lb < entries.len() && &entries[lb].0 == probe;
                prop_assert_eq!(node.search(probe), (lb, eq));
                for (i, (k, _)) in entries.iter().enumerate() {
                    prop_assert_eq!(node.cmp_key(i, probe), k.as_slice().cmp(probe.as_slice()));
                }
            }
        }

        #[test]
        fn fit_finds_the_largest_run_that_fits(
            map in prop::collection::btree_map(keys(), prop::collection::vec(any::<u8>(), 0..8), 1..60),
            limit in 40usize..400,
        ) {
            let mut buf = EntryBuf::default();
            for (k, v) in &map {
                buf.push(k, v);
            }
            let e = fit(&buf, 0, buf.len(), limit);
            prop_assert!(e >= 1 && e <= buf.len());
            prop_assert!(e == 1 || size(&buf, 0, e) <= limit);
            if e < buf.len() {
                prop_assert!(size(&buf, 0, e + 1) > limit);
            }
        }
    }

    #[test]
    fn sealed_pages_verify_and_damage_is_detected() {
        let mut buf = EntryBuf::default();
        buf.push(b"key", b"value");
        let mut page = vec![0; 4096];
        build(&mut page, 0, &buf, 0, 1);
        seal(&mut page, 42);
        assert!(verify(&page));
        assert_eq!(Node::new(&page).generation(), 42);
        for at in [0, 5, 100, 4095] {
            let mut damaged = page.clone();
            damaged[at] ^= 1;
            assert!(!verify(&damaged), "damage at byte {at} went unnoticed");
        }
    }

    #[test]
    fn child_refs_round_trip() {
        let r = child_ref(7, u64::MAX - 1);
        assert_eq!((child_frame(&r), child_page(&r)), (7, u64::MAX - 1));
    }

    #[test]
    fn max_entries_fit_four_to_a_node() {
        for page_size in [256, 4096, 65536] {
            let max = max_entry(page_size);
            // Four leaf entries with no shared prefix.
            let mut leaf = EntryBuf::default();
            for b in 0..4u8 {
                leaf.push(&[b], &vec![0; max - 1]);
            }
            assert!(size(&leaf, 0, 4) <= page_size);
            // Four internal entries with maximal keys.
            let mut internal = BTreeMap::new();
            for b in 0..4u8 {
                internal.insert(vec![b; max], child_ref(7, 9));
            }
            let mut buf = EntryBuf::default();
            for (k, v) in &internal {
                buf.push(k, v);
            }
            assert!(size(&buf, 0, 4) <= page_size);
        }
    }
}
