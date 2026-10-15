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

//! The node formats: one node per page, searched in place.
//!
//! ```text
//! offset  size  field
//!      0     4  checksum (set when the page is written to disk; 0 in memory)
//!      4     1  format: FORMAT_SLOTTED or FORMAT_DENSE
//!      5     1  level: 0 for leaves
//!      6     2  entry count
//!      8     2  key prefix length
//!     10     2  slotted: offset of the heads array; dense: key suffix length
//!     12     2  slotted: offset of the slot array; dense: value prefix length
//!     14     2  slotted: reserved; dense: value suffix length
//!     16     4  bytes used
//!     20     8  generation (set when the page is written to disk)
//!     28     4  reserved
//!     32        slotted: key prefix | padding to 4 | heads | slots | heap
//!               dense: key prefix | value prefix | key suffixes | value suffixes
//! ```
//!
//! Every key in a node starts with the node's prefix: the longest prefix its first
//! and last keys share, stored once. The rest of a key is its suffix.
//!
//! **Slotted** nodes hold keys and values of any lengths. The first four suffix
//! bytes, zero-padded and read as a big-endian `u32`, are the key's head. The heads
//! sit in one array, so a binary search touches nothing else. A slot holds the
//! entry's record offset (relative to the heap), suffix length and value length. The
//! record holds the suffix bytes past the head, then the value.
//!
//! **Dense** nodes hold entries whose keys all have one length, and whose values all
//! have one length. A leaf also stores the prefix all its values share once. The key
//! suffixes sit in one array at a fixed stride, and so do the value suffixes, so an
//! entry costs only its suffixes: there are no heads or slots. A node gets the dense
//! format whenever its entries allow it, since it is then never larger.
//!
//! Internal nodes use the same formats. Their values are child references,
//! [`CHILD_REF`] bytes: the child's frame, then its page id once it has one. Loading
//! an internal node rewrites the frames, so its values are always stored whole.

use std::cmp::Ordering;

use crate::pool::FrameId;

pub(crate) const HEADER: usize = 32;
const HEAD: usize = 4;
const SLOT: usize = 6;
pub(crate) const CHILD_REF: usize = 12;
const FORMAT_SLOTTED: u8 = 1;
const FORMAT_DENSE: u8 = 2;

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
    if !matches!(page[4], FORMAT_SLOTTED | FORMAT_DENSE) || crc32c::crc32c(&page[4..]) != crc {
        return false;
    }
    let node = Node::new(page);
    if node.count == 0 || node.used() > page.len() {
        return false;
    }
    if !node.dense {
        return node.heap_at <= node.used();
    }
    // An internal node's values are whole child references.
    node.used() == node.values_at + node.count * node.value_suffix
        && (node.level() == 0 || (node.value_prefix == 0 && node.value_suffix == CHILD_REF))
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

/// The layout of a dense node: the lengths of its key prefix, key suffixes, value
/// prefix and value suffixes.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Dense {
    key_prefix: usize,
    key_suffix: usize,
    value_prefix: usize,
    value_suffix: usize,
}

impl Dense {
    /// Bytes a dense node with this layout and `n` entries takes.
    fn size(&self, n: usize) -> usize {
        HEADER + self.key_prefix + self.value_prefix + n * (self.key_suffix + self.value_suffix)
    }
}

#[cfg(test)]
thread_local! {
    static SLOTTED_ONLY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Builds every node on this thread slotted, for tests whose scenarios rely on how
/// slotted nodes fill pages.
#[cfg(test)]
pub(crate) fn slotted_only() {
    SLOTTED_ONLY.set(true);
}

/// Whether nodes may be dense.
fn dense_allowed() -> bool {
    #[cfg(test)]
    if SLOTTED_ONLY.get() {
        return false;
    }
    true
}

/// The dense layout of entries `a..b` of `src` in a node at `level`, if their keys
/// all have one length and their values all have one length. Only leaves share a
/// value prefix.
fn dense(src: &impl Entries, a: usize, b: usize, level: u8) -> Option<Dense> {
    if !dense_allowed() {
        return None;
    }
    let (first_key, first_value) = (src.key(a), src.value(a));
    let mut value_prefix = if level == 0 { first_value.len() } else { 0 };
    for i in a + 1..b {
        let (key, value) = (src.key(i), src.value(i));
        if key.len() != first_key.len() || value.len() != first_value.len() {
            return None;
        }
        value_prefix = lcp(&first_value[..value_prefix], &value[..value_prefix]);
    }
    let key_prefix = lcp(first_key, src.key(b - 1));
    Some(Dense {
        key_prefix,
        key_suffix: first_key.len() - key_prefix,
        value_prefix,
        value_suffix: first_value.len() - value_prefix,
    })
}

/// Bytes a node at `level` holding entries `a..b` of `src` takes, in the format
/// [`build`] gives it.
pub(crate) fn size(src: &impl Entries, a: usize, b: usize, level: u8) -> usize {
    if let Some(dense) = dense(src, a, b, level) {
        return dense.size(b - a);
    }
    let p = lcp(src.key(a), src.key(b - 1));
    let mut size = align4(HEADER + p) + (b - a) * (HEAD + SLOT);
    for i in a..b {
        size += rest_len(src.key(i).len() - p) + src.value(i).len();
    }
    size
}

/// The largest `e` in `a + 1..=b` such that a node at `level` holding entries `a..e`
/// fits in `limit` bytes, or `a + 1` if not even one entry does, and the bytes those
/// entries take.
pub(crate) fn fit(
    src: &impl Entries,
    a: usize,
    b: usize,
    limit: usize,
    level: u8,
) -> (usize, usize) {
    let (first, first_value) = (src.key(a), src.value(a));
    let mut prefix = first.len();
    // Record bytes (suffix past the head, plus value) of entries a..e, slotted.
    let mut records = first_value.len();
    // While entries a..e can be dense: the length of the prefix their values share.
    let mut value_prefix =
        dense_allowed().then_some(if level == 0 { first_value.len() } else { 0 });
    let size =
        |n: usize, prefix: usize, records: usize, value_prefix: Option<usize>| match value_prefix {
            Some(value_prefix) => Dense {
                key_prefix: prefix,
                key_suffix: first.len() - prefix,
                value_prefix,
                value_suffix: first_value.len() - value_prefix,
            }
            .size(n),
            None => align4(HEADER + prefix) + n * (HEAD + SLOT) + records,
        };
    let mut e = a + 1;
    while e < b {
        let (key, value) = (src.key(e), src.value(e));
        let p = prefix.min(lcp(first, key));
        let mut r = records;
        if p != prefix {
            // A shorter prefix lengthens every suffix: recount.
            r = (a..e)
                .map(|i| rest_len(src.key(i).len() - p) + src.value(i).len())
                .sum();
        }
        r += rest_len(key.len() - p) + value.len();
        let vp = value_prefix
            .filter(|_| key.len() == first.len() && value.len() == first_value.len())
            .map(|vp| lcp(&first_value[..vp], &value[..vp]));
        if size(e + 1 - a, p, r, vp) > limit {
            break;
        }
        prefix = p;
        records = r;
        value_prefix = vp;
        e += 1;
    }
    (e, size(e - a, prefix, records, value_prefix))
}

/// At most [`Node::entry_bytes`] of a node that would hold entries `a..` of `src` in
/// `used` bytes: its header and prefixes take at most a header and its first entry.
pub(crate) fn entry_bytes_at_least(src: &impl Entries, a: usize, used: usize) -> usize {
    used.saturating_sub(align4(HEADER + src.key(a).len()) + src.value(a).len())
}

/// Writes entries `a..b` of `src`, which must be sorted and fit in `page`, as a node
/// at `level`. Returns the bytes used.
pub(crate) fn build(page: &mut [u8], level: u8, src: &impl Entries, a: usize, b: usize) -> usize {
    if let Some(dense) = dense(src, a, b, level) {
        return build_dense(page, level, src, a, b, dense);
    }
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

/// Writes entries `a..b` of `src` as a dense node at `level`, with layout `dense`.
/// Returns the bytes used.
fn build_dense(
    page: &mut [u8],
    level: u8,
    src: &impl Entries,
    a: usize,
    b: usize,
    dense: Dense,
) -> usize {
    let n = b - a;
    let Dense {
        key_prefix: kp,
        key_suffix: ks,
        value_prefix: vp,
        value_suffix: vs,
    } = dense;
    page[..HEADER].fill(0);
    page[4] = FORMAT_DENSE;
    page[5] = level;
    put_u16(page, 6, n);
    put_u16(page, 8, kp);
    put_u16(page, 10, ks);
    put_u16(page, 12, vp);
    put_u16(page, 14, vs);
    let keys_at = HEADER + kp + vp;
    let values_at = keys_at + n * ks;
    page[HEADER..HEADER + kp].copy_from_slice(&src.key(a)[..kp]);
    page[HEADER + kp..keys_at].copy_from_slice(&src.value(a)[..vp]);
    for (j, i) in (a..b).enumerate() {
        page[keys_at + j * ks..][..ks].copy_from_slice(&src.key(i)[kp..]);
        page[values_at + j * vs..][..vs].copy_from_slice(&src.value(i)[vp..]);
    }
    let used = values_at + n * vs;
    page[16..20].copy_from_slice(&u32::try_from(used).unwrap().to_le_bytes());
    used
}

/// A read-only view of a node in a page.
#[derive(Clone, Copy)]
pub(crate) struct Node<'a> {
    page: &'a [u8],
    dense: bool,
    count: usize,
    prefix_len: usize,
    // Slotted nodes: where the heads, slots and heap start.
    heads_at: usize,
    slots_at: usize,
    heap_at: usize,
    // Dense nodes: the suffix and prefix lengths, and where the key and value
    // suffixes start.
    key_suffix: usize,
    value_prefix: usize,
    value_suffix: usize,
    keys_at: usize,
    values_at: usize,
}

impl<'a> Node<'a> {
    pub(crate) fn new(page: &'a [u8]) -> Self {
        let count = u16_at(page, 6);
        let prefix_len = u16_at(page, 8);
        if page[4] == FORMAT_DENSE {
            let key_suffix = u16_at(page, 10);
            let value_prefix = u16_at(page, 12);
            let keys_at = HEADER + prefix_len + value_prefix;
            return Node {
                page,
                dense: true,
                count,
                prefix_len,
                heads_at: 0,
                slots_at: 0,
                heap_at: 0,
                key_suffix,
                value_prefix,
                value_suffix: u16_at(page, 14),
                keys_at,
                values_at: keys_at + count * key_suffix,
            };
        }
        let slots_at = u16_at(page, 12);
        Node {
            page,
            dense: false,
            count,
            prefix_len,
            heads_at: u16_at(page, 10),
            slots_at,
            heap_at: slots_at + count * SLOT,
            key_suffix: 0,
            value_prefix: 0,
            value_suffix: 0,
            keys_at: 0,
            values_at: 0,
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

    /// The bytes the node spends on its entries: all but its header and prefixes. A
    /// node holding these entries and others spends at least as many on them, since
    /// its prefixes can only be shorter, and a slotted node takes more per entry than
    /// a dense one.
    pub(crate) fn entry_bytes(&self) -> usize {
        let fixed = if self.dense {
            HEADER + self.prefix_len + self.value_prefix
        } else {
            self.heads_at
        };
        self.used() - fixed
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

    /// The key suffix of entry `i` of a dense node.
    fn suffix(&self, i: usize) -> &'a [u8] {
        &self.page[self.keys_at + i * self.key_suffix..][..self.key_suffix]
    }

    /// Offset and length of the bytes of entry `i`'s value that the entry itself
    /// stores: all of them, except in a dense leaf, where the value prefix comes
    /// first.
    fn stored_value(&self, i: usize) -> (usize, usize) {
        if self.dense {
            return (self.values_at + i * self.value_suffix, self.value_suffix);
        }
        let (at, slen, vlen) = self.slot(i);
        (at + rest_len(slen), vlen)
    }

    /// Offset and length of entry `i`'s value within the page, if the page holds it
    /// whole: always, except in a dense leaf whose values share a prefix.
    pub(crate) fn whole_value(&self, i: usize) -> Option<(usize, usize)> {
        (self.value_prefix == 0).then(|| self.stored_value(i))
    }

    /// Offset and length of entry `i`'s value within the page, which must hold it
    /// whole, as internal nodes do.
    pub(crate) fn value_span(&self, i: usize) -> (usize, usize) {
        debug_assert_eq!(self.value_prefix, 0);
        self.stored_value(i)
    }

    /// Entry `i`'s value, which the page must hold whole, as internal nodes do.
    pub(crate) fn value(&self, i: usize) -> &'a [u8] {
        let (at, len) = self.value_span(i);
        &self.page[at..at + len]
    }

    /// Appends entry `i`'s value to `out`.
    pub(crate) fn append_value(&self, i: usize, out: &mut Vec<u8>) {
        let prefix_at = HEADER + self.prefix_len;
        out.extend_from_slice(&self.page[prefix_at..prefix_at + self.value_prefix]);
        let (at, len) = self.stored_value(i);
        out.extend_from_slice(&self.page[at..at + len]);
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
        out.extend_from_slice(self.prefix());
        if self.dense {
            out.extend_from_slice(self.suffix(i));
            return;
        }
        let (rest, slen) = self.rest(i);
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
        if self.dense {
            return self.suffix(i).cmp(key);
        }
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
        if self.dense {
            return self.search_dense(suffix);
        }
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

    /// As [`search`](Self::search) in a dense node, for a key past the node's prefix.
    fn search_dense(&self, suffix: &[u8]) -> (usize, bool) {
        let i = if suffix.len() == self.key_suffix && suffix.len() <= 8 {
            // Byte strings of one length up to 8 compare as big-endian integers.
            let target = be_uint(suffix);
            lower_bound(self.count, |i| be_uint(self.suffix(i)) < target)
        } else {
            lower_bound(self.count, |i| self.suffix(i) < suffix)
        };
        (i, i < self.count && self.suffix(i) == suffix)
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
    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }

    /// Drops the entries from `len` on.
    pub(crate) fn truncate(&mut self, len: usize) {
        if let Some(&[at, _, _]) = self.items.get(len) {
            self.bytes.truncate(at as usize);
            self.items.truncate(len);
        }
    }

    pub(crate) fn push(&mut self, key: &[u8], value: &[u8]) {
        let at = self.bytes.len();
        self.bytes.extend_from_slice(key);
        self.bytes.extend_from_slice(value);
        self.items
            .push([at as u32, key.len() as u32, value.len() as u32]);
    }

    /// Appends a copy of entries `a..b`.
    pub(crate) fn extend_from_within(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        // The entries' bytes are contiguous.
        let start = self.items[a][0];
        let [at, key_len, len] = self.items[b - 1];
        let shift = self.bytes.len() as u32 - start;
        self.bytes
            .extend_from_within(start as usize..(at + key_len + len) as usize);
        for i in a..b {
            let [at, key_len, len] = self.items[i];
            self.items.push([at + shift, key_len, len]);
        }
    }

    /// Copies entry `i` of `node`.
    pub(crate) fn push_from(&mut self, node: &Node, i: usize) {
        let at = self.bytes.len();
        node.append_key(i, &mut self.bytes);
        let key_len = self.bytes.len() - at;
        node.append_value(i, &mut self.bytes);
        let value_len = self.bytes.len() - at - key_len;
        self.items
            .push([at as u32, key_len as u32, value_len as u32]);
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

/// The length of the longest prefix `a` and `b` share, compared eight bytes at a
/// time: building a dense node takes one of these per entry.
fn lcp(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let word = |s: &[u8], at: usize| u64::from_le_bytes(s[at..at + 8].try_into().unwrap());
    if n < 8 {
        return a.iter().zip(b).take_while(|(x, y)| x == y).count();
    }
    let mut at = 0;
    while at + 8 < n {
        let diff = word(a, at) ^ word(b, at);
        if diff != 0 {
            // Read little-endian, the first differing byte is the lowest set one.
            return at + diff.trailing_zeros() as usize / 8;
        }
        at += 8;
    }
    // The last eight bytes, which may overlap ones already compared.
    let diff = word(a, n - 8) ^ word(b, n - 8);
    if diff == 0 {
        return n;
    }
    n - 8 + diff.trailing_zeros() as usize / 8
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

/// Up to 8 bytes read as a big-endian integer.
fn be_uint(bytes: &[u8]) -> u64 {
    let mut b = [0; 8];
    b[8 - bytes.len()..].copy_from_slice(bytes);
    u64::from_be_bytes(b)
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

    const BYTES: &[u8] = &[0, 1, 2, 0x7f, 0xff];

    /// Sorted entries, and keys to probe them with.
    type Case = (Vec<(Vec<u8>, Vec<u8>)>, Vec<Vec<u8>>);

    fn keys() -> impl Strategy<Value = Vec<u8>> {
        prop::collection::vec(prop::sample::select(BYTES), 0..=10)
    }

    /// Entries whose keys all have one length, and whose values all have one length
    /// and often share a prefix, with probes of that key length and of others.
    fn uniform() -> impl Strategy<Value = Case> {
        (0usize..=9, 0usize..=6).prop_flat_map(|(k, v)| {
            let key = prop::collection::vec(prop::sample::select(BYTES), k);
            let value = prop::collection::vec(prop::sample::select(&[0u8, 0, 1, 0xff][..]), v);
            (
                prop::collection::btree_map(key.clone(), value, 1..60)
                    .prop_map(|map| map.into_iter().collect()),
                prop::collection::vec(prop_oneof![key, keys()], 1..20),
            )
        })
    }

    /// Entries of any lengths, without probes.
    fn mixed() -> impl Strategy<Value = Case> {
        prop::collection::btree_map(keys(), prop::collection::vec(any::<u8>(), 0..8), 1..60)
            .prop_map(|map| (map.into_iter().collect(), Vec::new()))
    }

    fn entry_buf(entries: &[(Vec<u8>, Vec<u8>)]) -> EntryBuf {
        let mut buf = EntryBuf::default();
        for (k, v) in entries {
            buf.push(k, v);
        }
        buf
    }

    /// Builds `entries` into a node at `level`, and checks the node against them.
    fn check_node(
        entries: &[(Vec<u8>, Vec<u8>)],
        probes: &[Vec<u8>],
        level: u8,
    ) -> Result<(), TestCaseError> {
        let buf = entry_buf(entries);
        let mut page = vec![0xaa; 4096];
        let used = build(&mut page, level, &buf, 0, buf.len());
        prop_assert_eq!(used, size(&buf, 0, buf.len(), level));
        let node = Node::new(&page);
        let (k0, v0) = &entries[0];
        let uniform = entries
            .iter()
            .all(|(k, v)| k.len() == k0.len() && v.len() == v0.len());
        prop_assert_eq!(node.dense, uniform);
        prop_assert_eq!(node.level(), level);
        prop_assert_eq!(node.count(), entries.len());
        prop_assert_eq!(node.used(), used);
        let (mut key, mut value) = (Vec::new(), Vec::new());
        for (i, (k, v)) in entries.iter().enumerate() {
            node.key_into(i, &mut key);
            prop_assert_eq!(&key, k);
            value.clear();
            node.append_value(i, &mut value);
            prop_assert_eq!(&value, v);
            match node.whole_value(i) {
                Some((at, len)) => prop_assert_eq!(&page[at..at + len], &v[..]),
                // Only dense leaves split values, when they share a prefix.
                None => prop_assert!(level == 0 && uniform && !v.is_empty()),
            }
            if level > 0 {
                prop_assert_eq!(node.value(i), &v[..]);
            }
            prop_assert_eq!(node.search(k), (i, true));
        }
        for probe in probes {
            let lb = entries.partition_point(|(k, _)| k < probe);
            let eq = lb < entries.len() && &entries[lb].0 == probe;
            prop_assert_eq!(node.search(probe), (lb, eq));
            for (i, (k, _)) in entries.iter().enumerate() {
                prop_assert_eq!(node.cmp_key(i, probe), k.as_slice().cmp(probe.as_slice()));
            }
        }
        Ok(())
    }

    /// Checks [`fit`] against [`size`] on `entries`.
    fn check_fit(
        entries: &[(Vec<u8>, Vec<u8>)],
        limit: usize,
        level: u8,
    ) -> Result<(), TestCaseError> {
        let buf = entry_buf(entries);
        let (e, used) = fit(&buf, 0, buf.len(), limit, level);
        prop_assert!(e >= 1 && e <= buf.len());
        prop_assert_eq!(used, size(&buf, 0, e, level));
        prop_assert!(e == 1 || used <= limit);
        if e < buf.len() {
            prop_assert!(size(&buf, 0, e + 1, level) > limit);
        }
        Ok(())
    }

    proptest! {
        #[test]
        fn lcp_matches_a_byte_at_a_time_count(
            shared in prop::collection::vec(any::<u8>(), 0..20),
            a in prop::collection::vec(any::<u8>(), 0..20),
            b in prop::collection::vec(any::<u8>(), 0..20),
        ) {
            let (a, b) = ([&shared[..], &a].concat(), [&shared[..], &b].concat());
            let expected = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
            prop_assert_eq!(lcp(&a, &b), expected);
        }

        #[test]
        fn search_and_access_match_a_sorted_vec(
            map in prop::collection::btree_map(keys(), prop::collection::vec(any::<u8>(), 0..8), 1..40),
            probes in prop::collection::vec(keys(), 1..20),
            level in 0u8..2,
        ) {
            let entries: Vec<_> = map.into_iter().collect();
            check_node(&entries, &probes, level)?;
        }

        #[test]
        fn dense_nodes_match_a_sorted_vec((entries, probes) in uniform(), level in 0u8..2) {
            check_node(&entries, &probes, level)?;
        }

        #[test]
        fn fit_finds_the_largest_run_that_fits(
            map in prop::collection::btree_map(keys(), prop::collection::vec(any::<u8>(), 0..8), 1..60),
            limit in 40usize..400,
            level in 0u8..2,
        ) {
            let entries: Vec<_> = map.into_iter().collect();
            check_fit(&entries, limit, level)?;
        }

        #[test]
        fn merged_nodes_spend_at_least_what_both_spend_on_entries(
            (entries, _) in prop_oneof![uniform(), mixed()],
            cut in any::<prop::sample::Index>(),
            level in 0u8..2,
        ) {
            let n = entries.len();
            prop_assume!(n >= 2);
            let cut = 1 + cut.index(n - 1);
            let buf = entry_buf(&entries);
            let mut page = vec![0; 4096];
            let mut least = HEADER;
            for (a, b) in [(0, cut), (cut, n)] {
                let used = build(&mut page, level, &buf, a, b);
                let exact = Node::new(&page).entry_bytes();
                prop_assert!(entry_bytes_at_least(&buf, a, used) <= exact);
                least += exact;
            }
            prop_assert!(size(&buf, 0, n, level) >= least);
        }

        #[test]
        fn fit_finds_the_largest_dense_run_that_fits(
            (mut entries, _) in uniform(),
            odd in prop::option::of((any::<prop::sample::Index>(), prop::collection::vec(any::<u8>(), 0..8))),
            limit in 40usize..400,
            level in 0u8..2,
        ) {
            // One entry with another value length ends the dense run there.
            if let Some((at, value)) = odd {
                let at = at.index(entries.len());
                entries[at].1 = value;
            }
            check_fit(&entries, limit, level)?;
        }
    }

    #[test]
    fn sealed_pages_verify_and_damage_is_detected() {
        // One entry makes a dense node; keys of two lengths, a slotted one.
        for entries in [
            &[(&b"key"[..], &b"value"[..])][..],
            &[(b"a", b"x"), (b"bb", b"y")],
        ] {
            let entries: Vec<_> = entries
                .iter()
                .map(|&(k, v)| (k.to_vec(), v.to_vec()))
                .collect();
            let buf = entry_buf(&entries);
            let mut page = vec![0; 4096];
            build(&mut page, 0, &buf, 0, buf.len());
            assert_eq!(Node::new(&page).dense, entries.len() == 1);
            seal(&mut page, 42);
            assert!(verify(&page));
            assert_eq!(Node::new(&page).generation(), 42);
            for at in [0, 4, 5, 100, 4095] {
                let mut damaged = page.clone();
                damaged[at] ^= 1;
                assert!(!verify(&damaged), "damage at byte {at} went unnoticed");
            }
        }
    }

    #[test]
    fn dense_internal_nodes_need_whole_child_refs() {
        let mut buf = EntryBuf::default();
        buf.push(b"a", &child_ref(1, 2));
        buf.push(b"b", &child_ref(1, 3));
        let mut page = vec![0; 4096];
        build(&mut page, 1, &buf, 0, 2);
        seal(&mut page, 1);
        assert!(Node::new(&page).dense);
        assert!(verify(&page));
        // The same entries with shorter values, sealed so that the checksum matches.
        let mut buf = EntryBuf::default();
        buf.push(b"a", b"12345");
        buf.push(b"b", b"12346");
        build(&mut page, 1, &buf, 0, 2);
        seal(&mut page, 1);
        assert!(!verify(&page));
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
            // Four leaf entries with no shared prefix, and keys of different
            // lengths, so that the node is slotted, the larger format.
            let mut leaf = EntryBuf::default();
            for b in 0..4u8 {
                let key = vec![b; 1 + usize::from(b)];
                leaf.push(&key, &vec![0; max - key.len()]);
            }
            assert!(!dense(&leaf, 0, 4, 0).is_some());
            assert!(size(&leaf, 0, 4, 0) <= page_size);
            // Four internal entries with maximal keys.
            let mut internal = BTreeMap::new();
            for b in 0..4u8 {
                internal.insert(vec![b; max - usize::from(b)], child_ref(7, 9));
            }
            let mut buf = EntryBuf::default();
            for (k, v) in &internal {
                buf.push(k, v);
            }
            assert!(size(&buf, 0, 4, 1) <= page_size);
        }
    }
}
