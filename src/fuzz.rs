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

//! Entry points for the fuzz targets in `fuzz/`, which `cargo fuzz` builds with
//! `--cfg fuzzing`. Each takes the fuzzer's bytes and panics only on a bug: what
//! encina reads back from disk must come out as an error, or as entries that lie
//! within the bytes it read, whatever those bytes are; and what an encoder writes
//! must decode back to what it was given. DESIGN.md §11 lists the targets.
//!
//! A checksum would stop the fuzzer at the first check, so the targets that read
//! stamp checksums into what they read, as a write does: the checks after it then
//! see the fuzzer's bytes.

use std::collections::BTreeMap;
use std::future::Future;
use std::hint::black_box;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

use crate::meta::{MAGIC, META_PAGES, Meta};
use crate::node::{
    self, Child, EntryBuf, FORMAT_DENSE, FORMAT_SLOTTED, FORMAT_VALUE, NO_FRAME, Node, Size,
    ValueLeaf,
};
use crate::options::MIN_PAGE_SIZE;
use crate::sim::SimDisk;
use crate::{Comparison, Db, Iter, Options, ReadOnlyDb};

/// Checks a page that [`node::verify`] accepts as lookups, scans and opens read
/// it: every key and value lies within the page, and so does every child
/// reference of an internal node. The page is `data`, zero-padded to a power of
/// two from 256 bytes to 64 KiB.
pub fn node(data: &[u8]) {
    let mut page = data.to_vec();
    page.resize(data.len().next_power_of_two().clamp(256, 65536), 0);
    stamp(&mut page);
    if !node::verify(&page) {
        return;
    }
    let node = Node::new(&page);
    black_box((node.level(), node.is_dense(), node.entry_bytes()));
    // Loading an internal node points its references at frames.
    let mut loaded = page.clone();
    let (mut key, mut value) = (Vec::new(), Vec::new());
    for i in 0..node.count() {
        // Lookups and scans lend a value that the page holds whole straight from
        // its frame, unchecked.
        if let Some((at, len)) = node.whole_value(i) {
            assert!(
                at + len <= page.len(),
                "entry {i}'s value ends past the page"
            );
        }
        node.key_into(i, &mut key);
        value.clear();
        node.append_value(i, &mut value);
        // Probes at the key, before it and after it. The one before may be
        // shorter than the prefix the node's keys share.
        let mut after = key.clone();
        after.push(0);
        for probe in [&key[..], &key[..key.len() / 2], &after[..]] {
            node.search(probe);
            node.cmp_key(i, probe);
        }
        if node.level() > 0 {
            node.child_at(i);
            node::set_child_frame(&mut loaded, i, 0);
        }
    }
}

/// Builds nodes of sorted entries a page at a time, as a write does, seals them,
/// and reads them back as a load does: [`node::verify`] accepts each, and each
/// entry comes back as it went in, where a search finds it. Internal nodes hold
/// child references, which decode back to the children they refer to.
pub fn node_round_trip(data: &[u8]) {
    let mut input = Input(data);
    let page_size = input.page_size();
    let level = input.u8() % 4;
    let generation = input.u64();
    let max = node::max_entry(page_size);
    // Keys of one length, and values of another, make dense nodes.
    let uniform = input.u8() % 2 == 1;
    let (mut key_len, mut value_len) = (input.u16(), input.u16());
    let mut entries = BTreeMap::new();
    while !input.is_empty() {
        let left = input.0.len();
        if !uniform {
            (key_len, value_len) = (input.u16(), input.u16());
        }
        let key = input.bytes(key_len % (max + 1)).to_vec();
        let value = match level {
            // A larger entry would get a value leaf.
            0 => input.bytes(value_len % (max - key.len() + 1)).to_vec(),
            1 => reference(
                Child::leaf(
                    input.u64(),
                    Size {
                        entries: input.u32().into(),
                        bytes: input.u32().into(),
                    },
                ),
                0,
            ),
            2 => reference(Child::level_1(input.u64()), 1),
            _ => reference(
                Child::internal(input.u32().min(NO_FRAME - 1), input.u64()),
                2,
            ),
        };
        entries.insert(key, value);
        // Uniform entries of no bytes would take none of the input.
        if input.0.len() == left {
            break;
        }
    }
    let entries: Vec<_> = entries.into_iter().collect();
    let mut buf = EntryBuf::default();
    for (key, value) in &entries {
        buf.push(key, value);
    }
    let mut page = vec![0; page_size];
    let mut start = 0;
    while start < entries.len() {
        let (end, used) = node::fit(&buf, start, entries.len(), page_size, level);
        // Over a page that held another node.
        page.fill(0xaa);
        assert_eq!(node::build(&mut page, level, &buf, start, end), used);
        node::seal(&mut page, generation);
        assert!(node::verify(&page), "a node as built fails its checks");
        let node = Node::new(&page);
        assert_eq!(
            (node.level(), node.count(), node.generation()),
            (level, end - start, generation)
        );
        let built = &entries[start..end];
        let (mut key, mut value) = (Vec::new(), Vec::new());
        for (i, (k, v)) in built.iter().enumerate() {
            node.key_into(i, &mut key);
            value.clear();
            node.append_value(i, &mut value);
            assert_eq!((&key, &value), (k, v));
            if let Some((at, len)) = node.whole_value(i) {
                assert_eq!(&page[at..at + len], v);
            }
            if level > 0 {
                assert_eq!(node.child_at(i), node::decode_child(v, level - 1));
            }
            // The key, and probes just after it and before it.
            let mut after = k.clone();
            after.push(0);
            for probe in [&k[..], &after[..], &k[..k.len() / 2]] {
                let at = built.partition_point(|(key, _)| key.as_slice() < probe);
                let found = built.get(at).is_some_and(|(key, _)| key == probe);
                assert_eq!(node.search(probe), (at, found));
                assert_eq!(node.cmp_key(i, probe), k.as_slice().cmp(probe));
            }
        }
        start = end;
    }
}

/// The reference to `child`, at `level`, which decodes back to it.
fn reference(child: Child, level: u8) -> Vec<u8> {
    let r = node::child_ref(child);
    assert_eq!(node::decode_child(&r, level), child);
    r.to_vec()
}

/// Checks a value leaf that [`node::verify_value_leaf`] accepts as reads of one
/// do: its key and value lie within its span. The span is `data`, zero-padded to
/// whole pages of 256 bytes, and is read as a value leaf on pages of each size
/// that divides it.
pub fn value_leaf(data: &[u8]) {
    let mut span = data.to_vec();
    span.resize(data.len().max(1).next_multiple_of(256), 0);
    stamp(&mut span);
    for page_size in (8..=16).map(|bits| 1 << bits) {
        if span.len().is_multiple_of(page_size) && node::verify_value_leaf(&span, page_size) {
            let leaf = ValueLeaf::new(&span);
            assert!(leaf.used() <= span.len(), "the value ends past the span");
            black_box((leaf.key(), leaf.value(), leaf.generation()));
        }
    }
}

/// Writes a value leaf as a commit does, its header first, then a page at a time
/// into frames that held something else, and reads it back:
/// [`node::verify_value_leaf`] accepts it, with the key, value and generation it
/// was written with.
pub fn value_leaf_round_trip(data: &[u8]) {
    let mut input = Input(data);
    let page_size = input.page_size();
    let generation = input.u64();
    let key_len = input.u16() % (node::max_entry(page_size) + 1);
    let key = input.bytes(key_len).to_vec();
    // The rest of the input is the value, repeated to span more pages.
    let repeat = 1 + usize::from(input.u8() % 8);
    let value = input.bytes(usize::MAX).repeat(repeat);
    let span_len = node::value_leaf_len(key.len(), value.len()).next_multiple_of(page_size);
    let header = node::value_header(&key, &value, generation, span_len);
    let mut span = vec![0xaa; span_len];
    for (i, page) in span.chunks_mut(page_size).enumerate() {
        node::copy_value_leaf(page, i * page_size, &header, &key, &value);
    }
    assert!(
        node::verify_value_leaf(&span, page_size),
        "a value leaf as written fails its checks"
    );
    let leaf = ValueLeaf::new(&span);
    assert_eq!(
        (leaf.key(), leaf.value(), leaf.generation()),
        (&key[..], &value[..], generation)
    );
}

/// Encodes a meta page with any fields into a slot that held something else, and
/// decodes it back: [`Meta::decode`] returns the same fields.
pub fn meta_round_trip(data: &[u8]) {
    let mut input = Input(data);
    let meta = Meta {
        version: input.u32(),
        page_size: input.u32(),
        root_level: input.u8(),
        generation: input.u64(),
        root: input.u64(),
        high_water: input.u64(),
        entries: input.u64(),
        entry_bytes: input.u64(),
    };
    // A slot is a page, at least the 64 bytes the meta takes.
    let mut slot = vec![0xaa; 64 + input.u16()];
    meta.encode(&mut slot);
    assert_eq!(Meta::decode(&slot), Some(meta));
}

/// Opens a store whose file is `data`, as a read-only view and as its owner, and
/// reads it as each would: a scan, then lookups with every comparison, and short
/// scans, from some of the keys the scan found and keys just after them. The view
/// then lists its pages, and the owner scans its level-1 nodes, as it does once it
/// opens. Each must return entries or an error, never panic.
///
/// The meta slots and every node get their checksums, with pages of the size slot
/// 0 records, or else slot 1 does where recovery looks for it.
pub fn open(data: &[u8]) {
    let mut file = data.to_vec();
    stamp_file(&mut file);
    let disk = SimDisk::new();
    disk.set_bytes(0, &file);
    run(async {
        // The smallest cache, so that the view's reads miss and read ahead.
        let small = Options::new().cache_size(0);
        if let Ok(view) = ReadOnlyDb::open_sim(disk.clone(), small).await {
            read(&view).await;
            let _ = view
                .pages(|page| {
                    black_box(page);
                })
                .await;
        }
        if let Ok(db) = Db::open_sim(disk, Options::new()).await {
            read(&db).await;
            let _ = db.scan_level_1(usize::MAX).await;
        }
    });
}

/// Scans a store, then looks up the first key the scan found and ever sparser
/// ones after it, and keys just after them, with every comparison, and scans a
/// few entries from each.
async fn read(store: &impl Store) {
    let mut keys = Vec::new();
    let mut iter = store.scan(&[]);
    // A page of a crafted store can hold 65,535 entries.
    for n in 1..=10_000usize {
        let Ok(Some((key, value))) = iter.next().await else {
            break;
        };
        touch(key, value);
        if n.is_power_of_two() {
            keys.push(key.to_vec());
        }
    }
    drop(iter);
    let mut probes = vec![Vec::new()];
    for key in &keys {
        let mut after = key.clone();
        after.push(0);
        probes.extend([key.clone(), after]);
    }
    for probe in &probes {
        for cmp in [
            Comparison::Equal,
            Comparison::Floor,
            Comparison::Ceiling,
            Comparison::Lower,
            Comparison::Higher,
        ] {
            store.lookup(probe, cmp).await;
        }
        let mut iter = store.scan(probe);
        for _ in 0..3 {
            let Ok(Some((key, value))) = iter.next().await else {
                break;
            };
            touch(key, value);
        }
    }
}

/// The reads [`read`] makes of a store: a [`Db`], or a [`ReadOnlyDb`].
trait Store {
    async fn lookup(&self, key: &[u8], cmp: Comparison);
    fn scan(&self, start: &[u8]) -> Iter;
}

impl Store for Db {
    async fn lookup(&self, key: &[u8], cmp: Comparison) {
        let _ = self.get(key, cmp, touch).await;
    }

    fn scan(&self, start: &[u8]) -> Iter {
        self.range(start..)
    }
}

impl Store for ReadOnlyDb {
    async fn lookup(&self, key: &[u8], cmp: Comparison) {
        let _ = self.get(key, cmp, touch).await;
    }

    fn scan(&self, start: &[u8]) -> Iter {
        self.range(start..)
    }
}

/// Reads every byte of an entry that a store lends.
fn touch(key: &[u8], value: &[u8]) {
    black_box(key.iter().chain(value).fold(0u8, |sum, &b| sum ^ b));
}

/// Stamps the checksums into a store's file: its meta slots, and every node in it,
/// with pages of the size slot 0 records, or else slot 1 does where recovery looks
/// for it. Pages go last to first, so that a value leaf's checksum covers the
/// pages of its span as they end up.
fn stamp_file(file: &mut [u8]) {
    let u32_at = |file: &[u8], at: usize| {
        let bytes = file.get(at..at + 4)?;
        Some(u32::from_le_bytes(bytes.try_into().unwrap()) as usize)
    };
    let in_slot1 = |size: usize| {
        file.get(size + 4..size + 12) == Some(&MAGIC[..]) && u32_at(file, size + 16) == Some(size)
    };
    let page_size = u32_at(file, 16)
        .filter(|&size| size.is_power_of_two() && (MIN_PAGE_SIZE..=65536).contains(&size))
        .or_else(|| (8..=16).map(|bits| 1 << bits).find(|&size| in_slot1(size)));
    let Some(page_size) = page_size else {
        return;
    };
    for slot in [0, page_size] {
        if let Some(meta) = file.get_mut(slot..slot + 64) {
            stamp(meta);
        }
    }
    for page in (META_PAGES as usize..file.len() / page_size).rev() {
        let at = page * page_size;
        let len = match file[at + 4] {
            FORMAT_SLOTTED | FORMAT_DENSE => page_size,
            FORMAT_VALUE => ValueLeaf::new(&file[at..])
                .used()
                .next_multiple_of(page_size),
            _ => continue,
        };
        if let Some(node) = file.get_mut(at..at + len) {
            stamp(node);
        }
    }
}

/// Stamps the checksum of bytes 4 on into the first four, as a write does: of a
/// meta slot's 64 bytes, a node's page, or a value leaf's span.
fn stamp(bytes: &mut [u8]) {
    let crc = crc32c::crc32c(&bytes[4..]);
    bytes[..4].copy_from_slice(&crc.to_le_bytes());
}

/// Polls `f` until it is done. On the simulated disk, nothing waits for more than
/// the next poll.
fn run<F: Future>(f: F) -> F::Output {
    let mut f = pin!(f);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(output) = f.as_mut().poll(&mut cx) {
            return output;
        }
    }
}

/// The fuzzer's bytes, read front to back. Reads past the end get zeros.
struct Input<'a>(&'a [u8]);

impl<'a> Input<'a> {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The next `n` bytes, or as many as are left.
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let (bytes, rest) = self.0.split_at(n.min(self.0.len()));
        self.0 = rest;
        bytes
    }

    fn array<const N: usize>(&mut self) -> [u8; N] {
        let mut array = [0; N];
        let bytes = self.bytes(N);
        array[..bytes.len()].copy_from_slice(bytes);
        array
    }

    fn u8(&mut self) -> u8 {
        self.array::<1>()[0]
    }

    fn u16(&mut self) -> usize {
        u16::from_le_bytes(self.array()).into()
    }

    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.array())
    }

    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.array())
    }

    /// A page size from 256 bytes to 64 KiB.
    fn page_size(&mut self) -> usize {
        256 << (self.u8() % 9)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::*;
    use crate::{PageInfo, WriteBatch};

    type Target = fn(&[u8]);

    /// The fuzz targets, by name.
    const TARGETS: [(&str, Target); 6] = [
        ("meta_round_trip", meta_round_trip),
        ("node", node),
        ("node_round_trip", node_round_trip),
        ("open", open),
        ("value_leaf", value_leaf),
        ("value_leaf_round_trip", value_leaf_round_trip),
    ];

    #[test]
    fn targets_take_their_seeds_and_other_bytes() {
        for (target, name, seed) in seeds() {
            let (_, run) = TARGETS.iter().find(|&&(t, _)| t == target).unwrap();
            println!("{target}: {name}");
            run(&seed);
        }
        let bytes: Vec<u8> = (0..=255).collect();
        for (_, run) in TARGETS {
            for input in [&[][..], &[0xff; 64][..], &bytes[..]] {
                run(input);
            }
        }
    }

    /// Writes the seeds to fuzz/seeds/, a directory per target, for cargo fuzz to
    /// start from. Run it again once the file format changes:
    /// `cargo test write_fuzz_seeds -- --ignored`.
    #[test]
    #[ignore]
    fn write_fuzz_seeds() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/seeds");
        let seeds = seeds();
        for (target, _, _) in &seeds {
            let _ = fs::remove_dir_all(dir.join(target));
        }
        for (target, name, seed) in seeds {
            fs::create_dir_all(dir.join(target)).unwrap();
            fs::write(dir.join(target).join(name), seed).unwrap();
        }
    }

    /// Seeds for the targets that read: stores of several shapes for `open`, and
    /// the nodes and value leaves in them for `node` and `value_leaf`. Returns the
    /// target, a name and the bytes of each.
    fn seeds() -> Vec<(&'static str, String, Vec<u8>)> {
        let mut seeds = Vec::new();
        for (name, file) in stores() {
            let (page_size, nodes) = nodes(&file);
            for info in nodes {
                let at = info.page as usize * page_size;
                let seed = file[at..at + info.pages as usize * page_size].to_vec();
                let target = if info.value { "value_leaf" } else { "node" };
                seeds.push((target, format!("{name}-{}", info.page), seed));
            }
            seeds.push(("open", name.to_string(), file));
        }
        seeds
    }

    /// The page size of the store in `file`, and its nodes, in page order.
    fn nodes(file: &[u8]) -> (usize, Vec<PageInfo>) {
        let disk = SimDisk::new();
        disk.set_bytes(0, file);
        run(async {
            let view = ReadOnlyDb::open_sim(disk, Options::new()).await.unwrap();
            let mut nodes = Vec::new();
            view.pages(|info| nodes.push(info)).await.unwrap();
            nodes.sort_by_key(|info| info.page);
            (view.page_size(), nodes)
        })
    }

    /// The entries of the store in `file`.
    fn entries(file: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
        let disk = SimDisk::new();
        disk.set_bytes(0, file);
        run(async {
            let view = ReadOnlyDb::open_sim(disk, Options::new()).await.unwrap();
            let mut iter = view.range(..);
            let mut entries = Vec::new();
            while let Some((key, value)) = iter.next().await.unwrap() {
                entries.push((key.to_vec(), value.to_vec()));
            }
            entries
        })
    }

    /// `file` with its store's nodes moved down, in order, to follow the meta
    /// slots, and its free pages left out. A write takes pages in runs that start
    /// hundreds of pages apart, so a few nodes can spread over a large file.
    fn compact(file: &[u8]) -> Vec<u8> {
        let (page_size, nodes) = nodes(file);
        let mut moved = BTreeMap::new();
        let mut end = META_PAGES;
        for info in &nodes {
            moved.insert(info.page, end);
            end += info.pages;
        }
        let metas = META_PAGES as usize * page_size;
        let mut out = vec![0; end as usize * page_size];
        out[..metas].copy_from_slice(&file[..metas]);
        for info in &nodes {
            let len = info.pages as usize * page_size;
            let from = info.page as usize * page_size;
            let to = moved[&info.page] as usize * page_size;
            let node = &mut out[to..to + len];
            node.copy_from_slice(&file[from..from + len]);
            if info.level > 0 {
                for i in 0..info.entries {
                    let (at, _) = Node::new(node).value_span(i);
                    // The span of a value leaf stays in the bits above its page.
                    let id = node::child_page(&node[at..]);
                    let first = node::first_page(id);
                    let id = moved[&first] | (id ^ first);
                    node[at + 4..at + 12].copy_from_slice(&id.to_le_bytes());
                }
            }
            stamp(node);
        }
        // The current meta slot, which now points at where the root went.
        let slots = [0, 1].map(|slot| Meta::decode(&file[slot * page_size..][..page_size]));
        let mut meta = slots
            .into_iter()
            .flatten()
            .max_by_key(|meta| meta.generation)
            .unwrap();
        if meta.root != 0 {
            meta.root = moved[&meta.root];
        }
        meta.high_water = end;
        let at = meta.slot() as usize * page_size;
        meta.encode(&mut out[at..at + page_size]);
        out
    }

    /// Stores of several shapes, by name: the bytes of their files.
    fn stores() -> Vec<(&'static str, Vec<u8>)> {
        // BookKeeper's entries: a ledger id and an entry id, big-endian, map to a
        // location, which the store's dense nodes strip of shared prefixes.
        let ledgers = |ledgers: u64, entries: u64| {
            puts((0..ledgers).flat_map(|l| {
                (0..entries).map(move |e| {
                    let key = [l.to_be_bytes(), e.to_be_bytes()].concat();
                    (key, ((l << 32) | (e * 64)).to_be_bytes().to_vec())
                })
            }))
        };
        // Keys and values of many lengths, which slotted nodes hold.
        let mixed = |n: u32| {
            puts((0..n).map(|i| {
                let key = i.wrapping_mul(2_654_435_761).to_be_bytes();
                (
                    key[..1 + i as usize % 4].to_vec(),
                    vec![i as u8; i as usize % 7],
                )
            }))
        };
        // Values too large for a leaf of 256-byte pages, between small ones.
        let values = puts([
            (b"a".to_vec(), vec![1; 100]),
            (b"b".to_vec(), vec![2; 10]),
            (b"c".to_vec(), vec![3; 600]),
            (b"d".to_vec(), vec![4; 5]),
        ]);
        let mut ranges = WriteBatch::new();
        ranges.delete_range(&[0, 0, 0, 0, 0, 0, 0, 1], &[0, 0, 0, 0, 0, 0, 0, 3]);
        vec![
            ("empty", store(256, [])),
            ("leaf", store(256, [mixed(5)])),
            ("ledgers", store(256, [ledgers(4, 50)])),
            ("mixed", store(256, [mixed(300)])),
            ("values", store(256, [values])),
            (
                "rewritten",
                store(256, [ledgers(4, 50), ranges, ledgers(1, 20)]),
            ),
            ("4k-pages", store(4096, [mixed(2000)])),
        ]
    }

    /// The file of a store with pages of `page_size` bytes, once `batches` are
    /// written to it, compacted.
    fn store(page_size: usize, batches: impl IntoIterator<Item = WriteBatch>) -> Vec<u8> {
        let disk = SimDisk::new();
        run(async {
            let opts = Options::new().page_size(page_size);
            let db = Db::open_sim(disk.clone(), opts).await.unwrap();
            for mut batch in batches {
                db.write(&mut batch).await.unwrap();
            }
        });
        let file = disk.contents();
        let compacted = compact(&file);
        assert_eq!(entries(&compacted), entries(&file));
        compacted
    }

    fn puts(entries: impl IntoIterator<Item = (Vec<u8>, Vec<u8>)>) -> WriteBatch {
        let mut batch = WriteBatch::new();
        for (key, value) in entries {
            batch.put(&key, &value);
        }
        batch
    }
}
