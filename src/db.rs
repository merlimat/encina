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

use std::array;
use std::cell::{Cell, RefCell, RefMut};
use std::future::{Future, poll_fn};
use std::io;
use std::iter;
use std::mem;
use std::ops::{Bound, RangeBounds};
use std::path::Path;
use std::pin::pin;
use std::rc::{Rc, Weak};
use std::slice;
use std::task::{Context, Poll, ready};

use compio::runtime::JoinHandle;

use crate::batch::WriteBatch;
use crate::cache::Cache;
use crate::error::{Error, Result};
use crate::meta::{self, FORMAT_VERSION, Meta};
use crate::metrics::{self, Metrics};
use crate::node::{self, NO_FRAME, Node, Size, ValueLeaf};
use crate::options::{DEFAULT_PAGE_SIZE, MIN_PAGE_SIZE, Options};
use crate::pool::{FrameBuf, FrameId, Holds, Pool, ValueBuf};
use crate::reads::{Claim, Reads};
use crate::scan::{self, Scan};
use crate::space::Space;
use crate::storage::{PageRead, Storage};
use crate::tree::{Cursor, Find, MAX_HEIGHT, Pages, Root, Tree, Version};
use crate::writes::{Turn, Writes};

/// How many reads or writes a write keeps in flight, and how many leaves an
/// iterator reads ahead.
const IO_WINDOW: usize = 16;

/// The most bytes one write request carries: a commit writes each run of adjacent
/// pages with as few requests as this allows.
const MAX_WRITE: usize = 1 << 20;

/// The most buffers one vectored write takes: `IOV_MAX` on Linux and macOS.
const MAX_IOVECS: usize = 1024;

/// A node of the tree, as [`ReadOnlyDb::pages`] finds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PageInfo {
    /// The page the node is on: its offset in the file, in pages.
    pub page: u64,
    /// 0 for a leaf, and one more than its children's for an internal node.
    pub level: u8,
    /// A leaf's entries, or an internal node's children.
    pub entries: usize,
    /// The bytes of its pages the node uses, its header included.
    pub used: usize,
    /// Whether the node is dense rather than slotted: a node is dense when its
    /// keys share one length and its values share another.
    pub dense: bool,
    /// Whether the node is a value leaf, which holds one entry too large for the
    /// other formats, on as many pages as it needs.
    pub value: bool,
    /// The pages the node takes, from `page` on: one, unless it is a value leaf.
    pub pages: u64,
}

impl PageInfo {
    fn of(page: u64, node: &Node) -> PageInfo {
        PageInfo {
            page,
            level: node.level(),
            entries: node.count(),
            used: node.used(),
            dense: node.is_dense(),
            value: false,
            pages: 1,
        }
    }

    /// The value leaf with page id `page`.
    fn of_value(page: u64, leaf: &ValueLeaf) -> PageInfo {
        PageInfo {
            page: node::first_page(page),
            level: 0,
            entries: 1,
            used: leaf.used(),
            dense: false,
            value: true,
            pages: node::span(page),
        }
    }
}

/// How [`Db::get`] matches its key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Comparison {
    /// The entry with exactly this key.
    Equal,
    /// The entry with the largest key less than or equal to it.
    Floor,
    /// The entry with the smallest key greater than or equal to it.
    Ceiling,
    /// The entry with the largest key less than it.
    Lower,
    /// The entry with the smallest key greater than it.
    Higher,
}

/// An embedded, ordered key-value store.
///
/// `Db` is a cheap handle: clones share the same store. It is neither `Send` nor
/// `Sync`: each thread that runs a compio runtime owns its own stores, and awaits
/// their futures on that runtime.
#[derive(Clone)]
pub struct Db {
    inner: Rc<Inner>,
}

struct Inner {
    tree: RefCell<Tree>,
    /// Leaf reads in flight, which readers needing the same page share.
    reads: RefCell<Reads>,
    storage: Storage,
    /// The writes committing or waiting to, which share commits.
    writes: RefCell<Writes>,
    /// Set when an I/O error left it unknown whether a write reached the disk.
    failed: Cell<bool>,
    metrics: Metrics,
    /// Pages the file may hold: it has grown to this length at most.
    file_end: Cell<u64>,
    /// Opened by a [`ReadOnlyDb`], whose version the store's owner may overwrite.
    read_only: bool,
    page_size: usize,
    max_key: usize,
    /// Key buffers for reads and iterators, reused so that reads don't allocate.
    buffers: RefCell<Vec<Vec<u8>>>,
    /// Buffer lists for the page runs a commit writes, reused likewise.
    runs: RefCell<Vec<Vec<FrameBuf>>>,
    /// The scan of level 1 that the open leaves to finish, until it ends.
    scan: RefCell<Option<Scan>>,
    /// The task that runs it on a file, which dropping the store cancels.
    scanner: Cell<Option<JoinHandle<()>>>,
    /// In tests, how many level-1 nodes the scan may have left for a write to let
    /// it end first, if not its window.
    #[cfg(test)]
    scan_wait: Cell<Option<usize>>,
}

impl Inner {
    fn take_buffer(&self) -> Vec<u8> {
        self.buffers
            .borrow_mut()
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.max_key))
    }

    fn put_buffer(&self, mut buffer: Vec<u8>) {
        buffer.clear();
        self.buffers.borrow_mut().push(buffer);
    }

    fn offset(&self, page: u64) -> u64 {
        page * self.page_size as u64
    }

    /// Makes writes fail until the store is reopened: an I/O error left it unknown
    /// whether a write reached the disk.
    fn fail(&self) {
        self.failed.set(true);
        self.metrics.failed.set(1);
    }

    /// Fails with the error that keeps the store from writing, if any: an earlier
    /// I/O error in a commit, or the error that stopped the scan of level 1, with
    /// which the open would have failed before.
    fn writable(&self) -> Result<()> {
        if self.failed.get() {
            return Err(Error::Failed);
        }
        match self.scan.borrow().as_ref().and_then(|s| s.error.as_ref()) {
            Some(e) => Err(e.duplicate()),
            None => Ok(()),
        }
    }

    /// Reports the current version's entries, the file's size, and how its pages
    /// are used: by the current version, held back by older versions' readers, or
    /// free. The last are known once the scan of level 1 has ended.
    fn report_usage(&self) {
        let tree = self.tree.borrow();
        let size = tree.size();
        self.metrics.entry_count.set(size.entries as i64);
        self.metrics.entry_size.set(size.bytes as i64);
        let page_size = self.page_size as u64;
        let end = self.file_end.get();
        self.metrics.file_size.set((end * page_size) as i64);
        if self.scan.borrow().is_some() {
            return;
        }
        let (used, held) = tree.pages.space.usage();
        let pages = [used - held, held, end - used];
        for (amount, pages) in self.metrics.file_usage.iter().zip(pages) {
            amount.set((pages * page_size) as i64);
        }
    }

    /// The error to report for `e`, which a read hit. In a read-only store it may
    /// mean the store is stale, as [`stale_or`] decides.
    async fn explain(&self, e: Error) -> Error {
        if !self.read_only {
            return e;
        }
        let generation = self.tree.borrow().current().generation;
        stale_or(&self.storage, self.page_size, generation, e).await
    }

    /// Drives the scan of level 1, if one runs: keeps a window of its reads in
    /// flight, and marks the leaves of the level-1 nodes they bring in, in key
    /// order. Returns once it has ended, with the error that stopped it, if any, or
    /// once it has marked the leaves of `quota` more nodes, which it counts down.
    /// After a window of nodes, it lets the other tasks run.
    fn poll_scan(&self, cx: &mut Context<'_>, quota: &mut usize) -> Poll<Result<()>> {
        let mut running = self.scan.borrow_mut();
        let Some(scan) = running.as_mut() else {
            return Poll::Ready(Ok(()));
        };
        if let Some(e) = &scan.error {
            return Poll::Ready(Err(e.duplicate()));
        }
        let mut tree = self.tree.borrow_mut();
        let mut reads = self.reads.borrow_mut();
        let window = scan_window(&tree.pages.cache);
        let mut marked = 0;
        loop {
            self.fill_scan(scan, &mut tree, &mut reads, window);
            if scan.len() == 0 {
                break;
            }
            if *quota == 0 {
                return Poll::Ready(Ok(()));
            }
            if marked == window {
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }
            let (page, held) = *scan.at(0);
            let frame = if held.is_some() {
                match poll_read(&mut tree, &mut reads, cx, page, 1, &mut scan.at(0).1) {
                    Poll::Pending => return Poll::Pending,
                    // Another reader claimed the read: the node is cached, unless the
                    // read failed.
                    Poll::Ready(None) => continue,
                    Poll::Ready(Some(Ok(frame))) => {
                        // Read for the scan alone: the cache lets go of it first.
                        tree.pages.pool.frame_mut(frame).clock = false;
                        frame
                    }
                    Poll::Ready(Some(Err(e))) => {
                        let e = scan.fail(e, &mut reads, &mut tree.pages.pool);
                        return Poll::Ready(Err(e));
                    }
                }
            } else if let Some(frame) = tree.pages.cache.peek(page) {
                tree.pages.pool.retain(frame);
                frame
            } else {
                // Like a lookup's, this read grows the pool past the budget if it
                // has to, so the scan always goes on.
                scan.at(0).1 = Some(read_leaf(self, &mut tree, &mut reads, page));
                continue;
            };
            let Pages { pool, space, .. } = &mut tree.pages;
            let checked = scan.mark(space, &Node::new(pool.page(frame)));
            pool.unref(frame);
            if let Err(e) = checked {
                let e = scan.fail(e, &mut reads, &mut tree.pages.pool);
                return Poll::Ready(Err(e));
            }
            scan.pop();
            *quota -= 1;
            marked += 1;
        }
        // Every page in use is marked.
        running.take().unwrap().end(&mut tree);
        drop((running, tree, reads));
        self.report_usage();
        Poll::Ready(Ok(()))
    }

    /// Tops up the window of the scan of level 1 with the next level-1 nodes, and
    /// starts reading those past the first that aren't cached, within the budget.
    fn fill_scan(&self, scan: &mut Scan, tree: &mut Tree, reads: &mut Reads, window: usize) {
        while scan.len() < window
            && let Some(page) = scan.next_page(&tree.pages.pool)
        {
            scan.push(page);
        }
        for k in 1..scan.len() {
            let (page, held) = *scan.at(k);
            if held.is_some() || tree.pages.cache.peek(page).is_some() {
                continue;
            }
            match read_ahead(self, tree, reads, page) {
                Some((_, slot)) => scan.at(k).1 = Some(slot),
                // No frame is free within the budget: the reads in flight free some.
                None => break,
            }
        }
    }

    /// Whether a write should let the scan of level 1 end first: when it has no
    /// more than a window of level-1 nodes left, reading them costs the write about
    /// one read, and saves it from taking pages past the end of the file, as a
    /// small store's first writes would.
    fn scan_nearly_done(&self) -> bool {
        let window = scan_window(&self.tree.borrow().pages.cache);
        #[cfg(test)]
        let window = self.scan_wait.get().unwrap_or(window);
        self.scan
            .borrow()
            .as_ref()
            .is_some_and(|s| s.left() <= window)
    }
}

/// How many level-1 nodes the scan of level 1 reads at once: at most a quarter
/// of the cache budget, as an iterator reads ahead.
fn scan_window(cache: &Cache) -> usize {
    scan::WINDOW.min(cache.budget() / 4).max(1)
}

/// Runs the scan of level 1 in a task of its own, which holds the store only while
/// it is polled: dropping the store cancels it.
async fn scan_level_1(inner: Weak<Inner>) {
    let mut all = usize::MAX;
    poll_fn(|cx| match inner.upgrade() {
        Some(inner) => inner.poll_scan(cx, &mut all).map(drop),
        None => Poll::Ready(()),
    })
    .await;
}

impl Db {
    /// Opens the store at `path`, creating it if needed.
    ///
    /// A new store takes its page size from `opts`, 4 KiB if it sets none. An
    /// existing one keeps the page size it was created with: this fails with
    /// [`Error::PageSizeMismatch`] if `opts` sets another.
    ///
    /// A store has one owner at a time: this fails with [`Error::Locked`] while
    /// another `Db`, in this process or another, has it open. That `Db` lets go
    /// once it, its clones and its iterators are closed or dropped, and any I/O it
    /// started has finished. A [`ReadOnlyDb`] can read the store meanwhile.
    ///
    /// Opening reads the internal nodes above the leaves' parents, a fraction of a
    /// percent of the internal nodes, and returns: the store serves reads and
    /// writes at once. A task that the store spawns on the current compio runtime
    /// then reads the leaves' parents, to find the pages free for reuse. Until it
    /// has, writes take new pages at the end of the file.
    pub async fn open(path: impl AsRef<Path>, opts: Options) -> Result<Db> {
        opts.validate()?;
        let (storage, len) = Storage::open(path.as_ref()).await?;
        Db::open_on(storage, len, opts, false, Some(path.as_ref())).await
    }

    /// Opens a store on a simulated disk.
    #[cfg(any(test, fuzzing))]
    pub(crate) async fn open_sim(disk: Rc<crate::sim::SimDisk>, opts: Options) -> Result<Db> {
        opts.validate()?;
        let len = disk.len();
        Db::open_on(Storage::Sim(disk), len, opts, false, None).await
    }

    /// Opens the store in `storage`, `len` bytes long, at `path` if it has one.
    async fn open_on(
        storage: Storage,
        len: u64,
        opts: Options,
        read_only: bool,
        path: Option<&Path>,
    ) -> Result<Db> {
        let meta = if len == 0 && !read_only {
            create(&storage, &opts).await?
        } else {
            recover(&storage, len).await?
        };
        let page_size = meta.page_size as usize;
        // A view writes nothing, so it reads the store whatever page size `opts` sets.
        if let Some(options) = opts.page_size
            && options != page_size
            && !read_only
        {
            return Err(Error::PageSizeMismatch {
                store: page_size,
                options,
            });
        }
        let opts = Options {
            page_size: Some(page_size),
            ..opts
        };
        // No page past the end of the file is in use: a commit writes its tree
        // before its meta page. So the file bounds the pages the open marks, and the
        // memory that takes, and a page the tree lists past its end is corrupt. A
        // view reads the length again, after the meta page: the owner may have
        // committed past the length the view read first.
        let len = if read_only { storage.len().await? } else { len };
        let file_end = len.div_ceil(page_size as u64).max(meta::META_PAGES);
        let meter = metrics::Meter::new(&opts, path);
        let mut pages = Pages {
            pool: Pool::new(page_size),
            cache: Cache::new(opts.cache_frames()),
            space: Space::new(meta.high_water.min(file_end), opts.min_run_pages as u64),
        };
        pages.pool.usage = meter.cache_usage();
        [pages.cache.hits, pages.cache.misses] = meter.cache_lookups();
        pages.cache.evictions = meter.cache_evictions();
        let mut reads = Reads::default();
        reads.duration = meter.disk_read_duration();
        let metrics = Metrics::new(&meter, read_only);
        metrics
            .cache_limit
            .set((pages.cache.budget() * page_size) as i64);
        for page in 0..meta::META_PAGES {
            pages.space.mark_used(page);
        }
        let root = match load_internal(&storage, &mut pages, &meta, &metrics).await {
            Err(e) if read_only => {
                return Err(stale_or(&storage, page_size, meta.generation, e).await);
            }
            root => root?,
        };
        let mut tree = Tree::new(
            pages,
            page_size,
            root,
            meta.generation,
            Size {
                entries: meta.entries,
                bytes: meta.entry_bytes,
            },
        );
        #[cfg(test)]
        if let Storage::Sim(disk) = &storage {
            tree.disk = Some(disk.clone());
        }
        // The scan of level 1 marks the pages of the leaves while the store serves
        // reads and writes. A view writes nothing, so it needs no free pages.
        let scan = match read_only {
            false => Scan::new(&mut tree),
            true => None,
        };
        let inner = Inner {
            tree: RefCell::new(tree),
            reads: RefCell::new(reads),
            storage,
            writes: RefCell::new(Writes::new()),
            failed: Cell::new(false),
            metrics,
            file_end: Cell::new(file_end),
            read_only,
            page_size,
            max_key: opts.max_key_size(),
            buffers: RefCell::new(Vec::new()),
            runs: RefCell::new(Vec::new()),
            scan: RefCell::new(scan),
            scanner: Cell::new(None),
            #[cfg(test)]
            scan_wait: Cell::new(None),
        };
        if !read_only {
            inner.report_usage();
        }
        let inner = Rc::new(inner);
        // On a file, the scan runs in a task of its own. On the simulated disk, the
        // tests drive it.
        let on_file = match &inner.storage {
            Storage::File(_) => true,
            #[cfg(any(test, fuzzing))]
            Storage::Sim(_) => false,
        };
        if on_file && inner.scan.borrow().is_some() {
            let task = compio::runtime::spawn(scan_level_1(Rc::downgrade(&inner)));
            inner.scanner.set(Some(task));
        }
        Ok(Db { inner })
    }

    /// Applies `batch` atomically and durably: once this returns `Ok`, the batch is
    /// on disk and visible to reads.
    ///
    /// Writes that arrive while another commits share the next commit: their
    /// batches apply in the order the writes arrived, each atomically, and one
    /// commit makes them all durable.
    ///
    /// Fails with [`Error::KeyTooLarge`], applying nothing, if a put's key exceeds
    /// [`Options::max_key_size`]. An I/O error before the commit also leaves the
    /// store unchanged. One during the commit leaves it unknown whether the batch
    /// is durable, and makes this and later writes fail with [`Error::Failed`]
    /// until the store is reopened. Writes that share a commit share its outcome.
    /// If the reads that follow the [`open`](Self::open) find a page corrupt, or
    /// fail, writes fail with that error until the store is reopened: the store
    /// can't know which of its pages are free.
    ///
    /// Dropping the future withdraws the batch while it waits, and until its
    /// commit starts writing pages. After that, the batch commits with the rest,
    /// and if this write was the one committing them, the store fails as above.
    /// The batch is left as it was, ready to be cleared and reused.
    pub async fn write(&self, batch: &mut WriteBatch) -> Result<()> {
        let metrics = &self.inner.metrics;
        let _pending = metrics.write_pending.hold();
        let timing = metrics.write.start();
        let result = self.write_batch(batch).await;
        match &result {
            Ok(()) => {
                timing.stop();
                for (counter, n) in metrics.write_operations.iter().zip(batch.counts()) {
                    counter.add(n as u64);
                }
                metrics.write_size.record(batch.data_len() as u64);
            }
            Err(e) => metrics.write_errors.add(e),
        }
        result
    }

    async fn write_batch(&self, batch: &mut WriteBatch) -> Result<()> {
        let inner = &*self.inner;
        inner.writable()?;
        batch.check(inner.max_key)?;
        // Wait for the commit in progress, if any. Then this write commits, alone
        // or with the writes that queued alongside it, or another of them does.
        let mut queued = Queued { inner, slot: None };
        loop {
            let (group, slot) = match poll_fn(|cx| queued.poll(cx, batch)).await {
                Turn::Alone => (false, None),
                Turn::Lead(slot) => (true, slot),
                Turn::Done(result) => return result,
            };
            let mut guard = WriteGuard {
                inner,
                stage: Stage::Building,
                group,
                slot,
                delivered: false,
            };
            let result = commit(inner, &mut guard, batch).await;
            guard.deliver(&result);
            // A write that took over a group without its batch waits for its own.
            if !group || slot.is_some() {
                return result;
            }
        }
    }

    /// Finds the entry matching `key` under `cmp` and calls `f` with its key and
    /// value, returning whatever `f` returns. The slices are only valid inside `f`,
    /// which is how `get` avoids allocating: decode or copy what you need there.
    ///
    /// Here keys are a ledger id and an entry id, big-endian, and values are the
    /// entries' locations:
    ///
    /// ```
    /// # use encina::{Comparison, Db, Options, WriteBatch};
    /// fn key(ledger: u64, entry: u64) -> [u8; 16] {
    ///     let mut key = [0; 16];
    ///     key[..8].copy_from_slice(&ledger.to_be_bytes());
    ///     key[8..].copy_from_slice(&entry.to_be_bytes());
    ///     key
    /// }
    ///
    /// fn decode_key(key: &[u8]) -> (u64, u64) {
    ///     let decode = |bytes: &[u8]| u64::from_be_bytes(bytes.try_into().unwrap());
    ///     (decode(&key[..8]), decode(&key[8..]))
    /// }
    /// # compio::runtime::Runtime::new().unwrap().block_on(async {
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let db = Db::open(dir.path().join("data.db"), Options::default()).await?;
    /// # let mut batch = WriteBatch::new();
    /// # for entry in 0..3u64 {
    /// #     batch.put(&key(7, entry), &(entry * 1024).to_be_bytes());
    /// # }
    /// # db.write(&mut batch).await?;
    /// # let (l, e) = (7, 1);
    ///
    /// // Location of entry `e` in ledger `l`.
    /// let loc: Option<u64> = db
    ///     .get(&key(l, e), Comparison::Equal, |_k, v| u64::from_be_bytes(v.try_into().unwrap()))
    ///     .await?;
    /// # assert_eq!(loc, Some(1024));
    ///
    /// // Last entry of ledger `l`: the largest key <= (l, u64::MAX), if it is in `l`.
    /// let last: Option<u64> = db
    ///     .get(&key(l, u64::MAX), Comparison::Floor, |k, _v| decode_key(k))
    ///     .await?
    ///     .and_then(|(ledger, entry)| (ledger == l).then_some(entry));
    /// # assert_eq!(last, Some(2));
    /// # Ok::<_, encina::Error>(())
    /// # }).unwrap();
    /// ```
    pub async fn get<R>(
        &self,
        key: &[u8],
        cmp: Comparison,
        f: impl FnOnce(&[u8], &[u8]) -> R,
    ) -> Result<Option<R>> {
        let metrics = &self.inner.metrics;
        let timing = metrics.get.start();
        let result = self.lookup(key, cmp, f).await;
        match &result {
            Ok(_) => timing.stop(),
            Err(e) => metrics.get_errors.add(e),
        }
        result
    }

    async fn lookup<R>(
        &self,
        key: &[u8],
        cmp: Comparison,
        f: impl FnOnce(&[u8], &[u8]) -> R,
    ) -> Result<Option<R>> {
        let inner = &*self.inner;
        let mut read = Read {
            inner,
            key: inner.take_buffer(),
            version: None,
            nodes: [NO_FRAME; 4],
            loaded: 0,
        };
        loop {
            let found = {
                let mut tree = inner.tree.borrow_mut();
                let root = read.version.map_or(tree.current().root, |v| v.root);
                // Only the first try counts its cache hits: the lookup then pins a
                // version, and later tries find the leaves it read, counted as misses.
                tree.find(root, key, cmp, &mut read.key, read.version.is_none())
            };
            match found {
                Find::Found(value, len) => {
                    // SAFETY: the leaf stays cached while `f`, a synchronous closure, runs:
                    // nothing that could evict it, or reuse its page, can happen meanwhile.
                    let value = unsafe { slice::from_raw_parts(value, len) };
                    return Ok(Some(f(&read.key, value)));
                }
                Find::Copied(key_len) => {
                    let (key, value) = read.key.split_at(key_len);
                    return Ok(Some(f(key, value)));
                }
                Find::NotFound => return Ok(None),
                Find::Missing(page, level) => {
                    if read.version.is_none() {
                        // Waiting on the disk lets writes run: pin this version first.
                        read.version = Some(inner.tree.borrow_mut().pin());
                        inner.metrics.readers.add(1);
                        continue;
                    }
                    // A lookup reads at most two leaves, and the level-1 nodes above
                    // them, which stay pinned until it ends.
                    assert!(
                        read.loaded < read.nodes.len(),
                        "lookup needs too many nodes"
                    );
                    read.nodes[read.loaded] = match load_node(inner, page, level).await {
                        Ok(frame) => frame,
                        Err(e) => return Err(inner.explain(e).await),
                    };
                    read.loaded += 1;
                }
                Find::Value(page) => {
                    if read.version.is_none() {
                        read.version = Some(inner.tree.borrow_mut().pin());
                        inner.metrics.readers.add(1);
                        continue;
                    }
                    let buf = match read_value_leaf(inner, page, None).await {
                        Ok(buf) => buf,
                        Err(e) => return Err(inner.explain(e).await),
                    };
                    let leaf = ValueLeaf::new(buf.bytes());
                    return Ok(Some(f(leaf.key(), leaf.value())));
                }
            }
        }
    }

    /// Iterates over the entries with keys in `range`, in key order, as of now:
    /// later writes don't affect it. Creating and advancing it never allocates,
    /// but for the frames of the leaves it reads into the cache, and to read a value
    /// too large to cache, if larger than the ones it read before.
    /// When it reaches a leaf that isn't cached, it also starts reading the next
    /// leaves in the range, up to 16 of them.
    ///
    /// The bounds are byte slices, as in `db.range(&start[..]..&end[..])` or any
    /// other range form. `db.range(..)` covers every key, and a pair of [`Bound`]s
    /// can also leave out the start.
    ///
    /// An iterator keeps its version's internal nodes above level 1 in memory, and
    /// the pages later writes replace allocated, until it is dropped, so keep
    /// iterators short-lived.
    pub fn range<'k>(&self, range: impl RangeBounds<&'k [u8]>) -> Iter {
        let mut bounds = self.inner.take_buffer();
        let start = copy_bound(range.start_bound().cloned(), &mut bounds);
        let start_len = bounds.len();
        let end = copy_bound(range.end_bound().cloned(), &mut bounds);
        let version = self.inner.tree.borrow_mut().pin();
        let metrics = &self.inner.metrics;
        metrics.range_iterators.add(1);
        metrics.readers.add(1);
        Iter {
            inner: self.inner.clone(),
            version,
            cursor: Cursor::new(),
            level_1: None,
            leaf: None,
            on_value: false,
            value: None,
            at: 0,
            state: State::Unstarted,
            start,
            end,
            start_len,
            bounds,
            key: self.inner.take_buffer(),
            waiting: None,
            ahead: Ahead::new(),
            missed: false,
            returned: 0,
        }
    }

    /// Closes the store if this is its last handle, freeing its file for the next
    /// [`open`](Self::open). Otherwise the store stays open until its last handle,
    /// a clone or an iterator, is dropped. Every successful [`write`](Self::write)
    /// is already durable.
    pub async fn close(self) -> Result<()> {
        if let Ok(inner) = Rc::try_unwrap(self.inner) {
            // The reads of the scan of level 1 keep the file open until it lets go
            // of them.
            if let Some(scan) = inner.scan.borrow_mut().as_mut() {
                let mut tree = inner.tree.borrow_mut();
                scan.release(&mut inner.reads.borrow_mut(), &mut tree.pages.pool);
            }
            match inner.storage {
                Storage::File(file) => file.close().await?,
                #[cfg(any(test, fuzzing))]
                Storage::Sim(_) => {}
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn tree(&self) -> std::cell::RefMut<'_, Tree> {
        self.inner.tree.borrow_mut()
    }

    #[cfg(test)]
    pub(crate) fn has_failed(&self) -> bool {
        self.inner.failed.get()
    }

    /// Sets the memory the batches of a group of writes may take.
    #[cfg(test)]
    pub(crate) fn set_group_limit(&self, limit: usize) {
        self.inner.writes.borrow_mut().set_limit(limit);
    }

    /// Sets the work a write does before it pauses, normalizing, planning or
    /// merging its batch, instead of the time it runs, or with `None`, back to the
    /// time.
    #[cfg(test)]
    pub(crate) fn set_merge_slice(&self, slice: Option<usize>) {
        self.tree().set_slice(slice);
    }

    /// Checks the tree, and that no read is held but the scan's, and no write
    /// waits: only valid while no reader or write is in progress.
    #[cfg(test)]
    pub(crate) fn check(&self) {
        let scan = self.inner.scan.borrow();
        let reads = self.inner.reads.borrow();
        let mut tree = self.tree();
        let slots: Vec<usize> = scan.iter().flat_map(Scan::slots).collect();
        let mut reading = Vec::new();
        for (slot, frame) in reads.held() {
            assert!(slots.contains(&slot), "reads still held");
            reading.extend(frame);
        }
        match scan.as_ref() {
            Some(scan) => {
                let level_1 = |page| tree.node_bytes(NO_FRAME, page, 1);
                let unmarked = scan.unmarked(&tree.pages.pool, level_1);
                tree.check_scanning(scan.version.root, &unmarked, &reading);
            }
            None => tree.check(),
        }
        assert!(self.inner.writes.borrow().is_idle(), "writes still waiting");
    }

    /// Drives the scan of level 1 until it has marked the leaves of `nodes` more
    /// level-1 nodes, or has ended. Fails with the error that stopped it, if any.
    #[cfg(any(test, fuzzing))]
    pub(crate) async fn scan_level_1(&self, nodes: usize) -> Result<()> {
        let mut quota = nodes;
        poll_fn(|cx| self.inner.poll_scan(cx, &mut quota)).await
    }

    /// Sets how many level-1 nodes the scan of level 1 may have left for a write
    /// to let it end first, or with `None`, back to the scan's window.
    #[cfg(test)]
    pub(crate) fn set_scan_wait(&self, nodes: Option<usize>) {
        self.inner.scan_wait.set(nodes);
    }

    /// The level-1 nodes whose leaves the scan of level 1 has left to mark, while
    /// it runs, or after it stopped at an error.
    #[cfg(test)]
    pub(crate) fn scan_left(&self) -> Option<usize> {
        self.inner.scan.borrow().as_ref().map(Scan::left)
    }
}

/// A read-only view of a store. It takes no lock, so it opens even while a [`Db`],
/// in this process or another, owns the store, and it never writes to the file.
///
/// It reads the version that was current when it opened, and doesn't see later
/// writes. The owner reuses the pages its writes replace, so once it writes again,
/// reads that miss the view's cache can fail with [`Error::Stale`]: open a new view
/// to read the current version. A view never returns entries of another version.
///
/// Like a `Db`, it is a cheap handle: clones share the same view.
#[derive(Clone)]
pub struct ReadOnlyDb {
    db: Db,
}

impl ReadOnlyDb {
    /// Opens a view of the existing store at `path`, as of its last write. `opts`
    /// sets the cache size; the page size comes from the file, whatever `opts` sets.
    ///
    /// Fails with [`Error::Stale`] if the store's owner overwrote that version
    /// while it was being opened: opening again finds a newer one.
    pub async fn open(path: impl AsRef<Path>, opts: Options) -> Result<ReadOnlyDb> {
        opts.validate()?;
        let (storage, len) = Storage::open_read_only(path.as_ref()).await?;
        let db = Db::open_on(storage, len, opts, true, Some(path.as_ref())).await?;
        Ok(ReadOnlyDb { db })
    }

    /// Opens a view of a store on a simulated disk.
    #[cfg(any(test, fuzzing))]
    pub(crate) async fn open_sim(
        disk: Rc<crate::sim::SimDisk>,
        opts: Options,
    ) -> Result<ReadOnlyDb> {
        opts.validate()?;
        let len = disk.len();
        let db = Db::open_on(Storage::Sim(disk), len, opts, true, None).await?;
        Ok(ReadOnlyDb { db })
    }

    /// As [`Db::get`], in the version this view reads.
    pub async fn get<R>(
        &self,
        key: &[u8],
        cmp: Comparison,
        f: impl FnOnce(&[u8], &[u8]) -> R,
    ) -> Result<Option<R>> {
        self.db.get(key, cmp, f).await
    }

    /// As [`Db::range`], in the version this view reads.
    pub fn range<'k>(&self, range: impl RangeBounds<&'k [u8]>) -> Iter {
        self.db.range(range)
    }

    /// The store's page size.
    pub fn page_size(&self) -> usize {
        self.db.inner.page_size
    }

    /// The entries in the version this view reads.
    pub fn entry_count(&self) -> u64 {
        self.db.inner.tree.borrow().size().entries
    }

    /// The bytes of the keys and values of the entries in the version this view
    /// reads: what the app wrote, not what the store takes on disk.
    pub fn entry_size(&self) -> u64 {
        self.db.inner.tree.borrow().size().bytes
    }

    /// Calls `f` on each node of the tree in the version this view reads: the
    /// internal nodes first, then the leaves in key order. Nodes that aren't
    /// cached are read one at a time, so this reads most of the file.
    ///
    /// The file's other pages are the two meta pages at its start, and free pages.
    pub async fn pages(&self, mut f: impl FnMut(PageInfo)) -> Result<()> {
        let inner = &*self.db.inner;
        // A view never writes: its tree stays as it is while nodes are read.
        let Some(root) = inner.tree.borrow().current().root else {
            return Ok(());
        };
        if root.level == 0 {
            f(leaf_info(inner, root.page).await?);
            return Ok(());
        }
        // The nodes above level 1 are in memory. They list the level-1 nodes, in key
        // order, which list the leaves.
        let mut level_1 = Vec::new();
        let mut stack = Vec::new();
        match root.level {
            1 => level_1.push(root.page),
            _ => stack.push((root.frame, root.page)),
        }
        while let Some((frame, page)) = stack.pop() {
            let info = {
                let tree = inner.tree.borrow();
                let node = tree.node(frame);
                let children = (0..node.count()).rev();
                match node.level() {
                    2 => level_1.extend(children.rev().map(|i| node.child_page(i))),
                    _ => stack.extend(children.map(|i| (node.child(i), node.child_page(i)))),
                }
                PageInfo::of(page, &node)
            };
            f(info);
        }
        for &page in &level_1 {
            let frame = load_level_1(inner, page).await?;
            let tree = inner.tree.borrow();
            let info = PageInfo::of(page, &tree.node(frame));
            tree.pages.pool.unref(frame);
            drop(tree);
            f(info);
        }
        for &page in &level_1 {
            let frame = load_level_1(inner, page).await?;
            let count = inner.tree.borrow().node(frame).count();
            for i in 0..count {
                let leaf = inner.tree.borrow().node(frame).child_page(i);
                match leaf_info(inner, leaf).await {
                    Ok(info) => f(info),
                    Err(e) => {
                        inner.tree.borrow().pages.pool.unref(frame);
                        return Err(e);
                    }
                }
            }
            inner.tree.borrow().pages.pool.unref(frame);
        }
        Ok(())
    }

    /// Closes the view, once this is its last handle.
    pub async fn close(self) -> Result<()> {
        self.db.close().await
    }
}

/// Pins the level-1 node on `page`, for [`ReadOnlyDb::pages`].
async fn load_level_1(inner: &Inner, page: u64) -> Result<FrameId> {
    match load_node(inner, page, 1).await {
        Ok(frame) => Ok(frame),
        Err(e) => Err(inner.explain(e).await),
    }
}

/// Reads the leaf on `page`, if it isn't cached, and describes it, for
/// [`ReadOnlyDb::pages`].
async fn leaf_info(inner: &Inner, page: u64) -> Result<PageInfo> {
    if node::span(page) > 0 {
        return match read_value_leaf(inner, page, None).await {
            Ok(buf) => Ok(PageInfo::of_value(page, &ValueLeaf::new(buf.bytes()))),
            Err(e) => Err(inner.explain(e).await),
        };
    }
    let frame = match load_node(inner, page, 0).await {
        Ok(frame) => frame,
        Err(e) => return Err(inner.explain(e).await),
    };
    let tree = inner.tree.borrow();
    let info = PageInfo::of(page, &tree.node(frame));
    tree.pages.pool.unref(frame);
    Ok(info)
}

/// Writes the meta page and the empty second slot of a new store.
async fn create(storage: &Storage, opts: &Options) -> Result<Meta> {
    let page_size = opts.page_size.unwrap_or(DEFAULT_PAGE_SIZE);
    let meta = Meta::empty(page_size);
    let mut pool = Pool::new(page_size);
    let frame = pool.alloc(Holds::Loading, 0);
    meta.encode(pool.page_mut(frame));
    let (res, buf) = storage.write(pool.buf(frame), 0).await;
    drop(buf);
    res?;
    pool.page_mut(frame).fill(0);
    let (res, buf) = storage.write(pool.buf(frame), page_size as u64).await;
    drop(buf);
    res?;
    storage.sync().await?;
    Ok(meta)
}

/// Reads the meta of an existing store: the newer of the two intact slots.
async fn recover(storage: &Storage, len: u64) -> Result<Meta> {
    // Probe with the smallest read the storage supports: no page is smaller.
    let probe = storage.min_read();
    if len < probe as u64 {
        return Err(Error::InvalidFile("no intact meta page"));
    }
    // Slot 0 gives the page size. If it was torn, look for slot 1 wherever each
    // page size would put it.
    let mut page_size = read_meta(storage, probe, 0)
        .await?
        .map(|m| m.page_size as usize);
    let mut size = probe;
    while page_size.is_none() && size <= 65536 && (size + probe) as u64 <= len {
        let meta = read_meta(storage, probe, size as u64).await?;
        page_size = meta.filter(|m| m.page_size as usize == size).map(|_| size);
        size *= 2;
    }
    let page_size = page_size.ok_or(Error::InvalidFile("no intact meta page"))?;
    if !page_size.is_power_of_two() || !(MIN_PAGE_SIZE..=65536).contains(&page_size) {
        return Err(Error::InvalidFile("unsupported page size"));
    }
    // Both slots, in full.
    let slot0 = read_meta(storage, page_size, 0).await?;
    let slot1 = if 2 * page_size as u64 <= len {
        read_meta(storage, page_size, page_size as u64).await?
    } else {
        None
    };
    let meta = match (slot0, slot1) {
        (Some(a), Some(b)) => {
            if a.generation > b.generation {
                a
            } else {
                b
            }
        }
        (Some(m), None) | (None, Some(m)) => m,
        (None, None) => return Err(Error::InvalidFile("no intact meta page")),
    };
    if meta.version != FORMAT_VERSION {
        return Err(Error::InvalidFile("unsupported format version"));
    }
    // The high-water mark leaves the meta pages in, and the tree is no deeper than
    // writes take.
    let root_ok = meta.root == 0 || (meta::META_PAGES..meta.high_water).contains(&meta.root);
    if meta.page_size as usize != page_size
        || meta.high_water < meta::META_PAGES
        || usize::from(meta.root_level) + 1 >= MAX_HEIGHT
        || !root_ok
    {
        return Err(Error::InvalidFile("inconsistent meta page"));
    }
    Ok(meta)
}

/// Reports error `e`, which a read-only store hit reading the version it opened as
/// of `generation`. The store's owner reuses the pages its later versions replaced,
/// and cuts the file once its end is free, so once it has committed another
/// version, a page that fails its checks, or lies past the end of the file, was
/// most likely overwritten or cut: then the store is [stale](Error::Stale) rather
/// than corrupt.
async fn stale_or(storage: &Storage, page_size: usize, generation: u64, e: Error) -> Error {
    let gone = match &e {
        Error::Corruption { .. } => true,
        Error::Io(e) => e.kind() == io::ErrorKind::UnexpectedEof,
        _ => false,
    };
    if !gone {
        return e;
    }
    for slot in 0..meta::META_PAGES {
        let meta = read_meta(storage, page_size, slot * page_size as u64).await;
        if let Ok(Some(meta)) = meta
            && meta.generation > generation
        {
            return Error::Stale;
        }
    }
    e
}

/// Reads `len` bytes at `offset`, and decodes them as a meta slot.
async fn read_meta(storage: &Storage, len: usize, offset: u64) -> Result<Option<Meta>> {
    let mut pool = Pool::new(len);
    let frame = pool.alloc(Holds::Loading, 0);
    let (res, buf) = storage.read(pool.buf(frame), offset).await;
    drop(buf);
    res?;
    Ok(Meta::decode(pool.page(frame)))
}

/// Loads the internal nodes above level 1 of the tree `meta` describes, which stay
/// in memory, each parent pointing at its children's frames, and marks their
/// pages, and those of the level-1 nodes, used. The scan of level 1 marks those
/// of the leaves. Returns the root.
async fn load_internal(
    storage: &Storage,
    pages: &mut Pages,
    meta: &Meta,
    metrics: &Metrics,
) -> Result<Option<Root>> {
    if meta.root == 0 {
        return Ok(None);
    }
    // The tree's pages lie below the high-water mark, which the file bounds.
    let high_water = pages.space.high_water();
    if meta.root >= high_water {
        return Err(Error::Corruption { page: meta.root });
    }
    pages.space.mark_used(meta.root);
    if meta.root_level <= 1 {
        return Ok(Some(Root {
            page: meta.root,
            frame: NO_FRAME,
            level: meta.root_level,
        }));
    }
    let page_size = meta.page_size as u64;
    let root = pages.pool.alloc(Holds::Internal, meta.root);
    let mut level_nodes = vec![(root, meta.root)];
    for level in (2..=meta.root_level).rev() {
        let mut next = Vec::new();
        // Read this level, then mark the pages of the level below, which it lists.
        for chunk in level_nodes.chunks(scan::WINDOW) {
            let reads: [_; scan::WINDOW] = array::from_fn(|i| {
                let &(frame, page) = chunk.get(i)?;
                metrics.read.add(page_size as usize);
                Some(storage.start_read(pages.pool.buf(frame), page * page_size))
            });
            let results = join(reads).await.into_iter().flatten();
            for (&(frame, page), (res, buf)) in chunk.iter().zip(results) {
                drop(buf);
                res?;
                let bytes = pages.pool.page(frame);
                let node = Node::new(bytes);
                let intact = node::verify(bytes)
                    && node.level() == level
                    && node.count() > 0
                    && node.generation() <= meta.generation;
                if !intact {
                    return Err(Error::Corruption { page });
                }
            }
            for &(parent, _) in chunk {
                let count = Node::new(pages.pool.page(parent)).count();
                for i in 0..count {
                    let page = Node::new(pages.pool.page(parent)).child_page(i);
                    // Only leaves are value leaves.
                    if node::span(page) > 0
                        || page < meta::META_PAGES
                        || page >= high_water
                        || pages.space.is_used(page)
                    {
                        return Err(Error::Corruption {
                            page: node::first_page(page),
                        });
                    }
                    pages.space.mark_used(page);
                    // Level-1 nodes are found by page id, through the cache: a
                    // level-2 node's references hold zeros where the frames go.
                    if level > 2 {
                        let frame = pages.pool.alloc(Holds::Internal, page);
                        node::set_child_frame(pages.pool.page_mut(parent), i, frame);
                        next.push((frame, page));
                    }
                }
            }
        }
        level_nodes = next;
    }
    Ok(Some(Root {
        page: meta.root,
        frame: root,
        level: meta.root_level,
    }))
}

/// Commits the batch of a write that commits alone, or the group of writes that
/// `guard` leads.
async fn commit(inner: &Inner, guard: &mut WriteGuard<'_>, batch: &mut WriteBatch) -> Result<()> {
    inner.writable()?;
    if inner.scan_nearly_done() {
        let mut all = usize::MAX;
        poll_fn(|cx| inner.poll_scan(cx, &mut all)).await?;
    }
    let timing = inner.metrics.commit.start();
    let group = guard.group;
    let changed = loop {
        // Normalize the batch and list the nodes it changes, a slice at a time, as
        // the merge goes below: first the level-1 nodes, then, once they are
        // loaded, the leaves under them.
        let planned = 'plan: loop {
            let listed = loop {
                let planned = {
                    let mut writes = inner.writes.borrow_mut();
                    // A write that left the group meanwhile took its batch along.
                    if writes.take_changed() {
                        break 'plan false;
                    }
                    let batch = if group { writes.group() } else { &mut *batch };
                    inner.tree.borrow_mut().plan(batch)
                };
                match planned {
                    Poll::Ready(listed) => break listed,
                    Poll::Pending => yield_now().await,
                }
            };
            // Load the nodes listed, keeping them in the cache. Cached ones load
            // without waiting, so this too pauses once it has run for a slice.
            let level = u8::from(listed);
            let touched = inner.tree.borrow().touched.len();
            for start in (0..touched).step_by(IO_WINDOW) {
                let loads: [_; IO_WINDOW] = array::from_fn(|i| {
                    let page = inner.tree.borrow().touched.get(start + i).copied();
                    page.map(|page| load_node(inner, page, level))
                });
                let mut error = None;
                for loaded in join(loads).await.into_iter().flatten() {
                    match loaded {
                        Ok(frame) => inner.tree.borrow_mut().pinned.push(frame),
                        Err(e) => error = error.or(Some(e)),
                    }
                }
                if let Some(e) = error {
                    return Err(e);
                }
                let n = (touched - start).min(IO_WINDOW);
                if inner.tree.borrow_mut().loaded(n) {
                    yield_now().await;
                    inner.tree.borrow_mut().resume();
                }
            }
            if !listed {
                break true;
            }
        };
        if !planned {
            // Start over without that batch.
            inner.tree.borrow_mut().discard();
            continue;
        }
        // Free the pages that readers held back and have let go of since, a slice
        // at a time: a reader that ends doesn't pause, and leaves them to the
        // writes. The merge then takes the pages it would have, had they freed
        // them.
        while inner.tree.borrow_mut().reclaim().is_pending() {
            yield_now().await;
            inner.tree.borrow_mut().resume();
        }
        // Build the new version, a slice of the merge at a time, letting the other
        // tasks run in between.
        let built = loop {
            let applied = {
                let mut writes = inner.writes.borrow_mut();
                // A write that left the group meanwhile took its batch along.
                if writes.take_changed() {
                    break None;
                }
                let batch = if group { writes.group() } else { &mut *batch };
                let applied = inner.tree.borrow_mut().apply(batch);
                // Once it is built, the group's writes can't leave it.
                if group && applied.is_ready() {
                    writes.committing();
                }
                applied
            };
            match applied {
                Poll::Ready(changed) => break Some(changed),
                Poll::Pending => yield_now().await,
            }
        };
        match built {
            Some(changed) => break changed,
            // Start over without that batch.
            None => inner.tree.borrow_mut().discard(),
        }
    };
    if !changed {
        // Planning may have freed pages that readers held back.
        inner.report_usage();
        return Ok(());
    }
    let generation = inner.tree.borrow().current().generation + 1;

    // Commit: write the new nodes, sync, then switch the meta page and sync.
    let meta = inner.tree.borrow().pending_meta(generation);
    inner
        .file_end
        .set(inner.file_end.get().max(meta.high_water));
    guard.stage = Stage::Writing;
    let mut written = write_dirty(inner).await;
    if written.is_ok() {
        written = write_values(inner, generation, group, batch).await;
    }
    guard.stage = Stage::Building;
    written?;
    guard.stage = Stage::Committing;
    sync(inner).await?;
    write_meta(inner, &meta).await?;
    sync(inner).await?;
    inner.tree.borrow_mut().publish(generation);
    timing.stop();
    let writes = if group {
        inner.writes.borrow().group_writes()
    } else {
        1
    };
    inner.metrics.commit_writes.record(writes as u64);
    // The batches are durable and visible: their writes can return.
    guard.deliver(&Ok(()));
    guard.stage = Stage::Cutting;
    shrink(inner, meta.high_water).await;
    // Free the pages of the nodes the commit replaced, unless older readers still
    // need them, a slice at a time, as the merge goes. If the write is dropped
    // meanwhile, the next one frees the rest.
    guard.stage = Stage::Done;
    while inner.tree.borrow_mut().reclaim().is_pending() {
        yield_now().await;
        inner.tree.borrow_mut().resume();
    }
    inner.report_usage();
    Ok(())
}

/// Makes every completed write durable.
async fn sync(inner: &Inner) -> Result<()> {
    let timing = inner.metrics.sync_duration.start();
    inner.storage.sync().await?;
    timing.stop();
    Ok(())
}

/// A write waiting for its turn to commit, with the slot it holds among the
/// writes waiting, which it lets go of when dropped.
struct Queued<'a> {
    inner: &'a Inner,
    slot: Option<usize>,
}

impl Queued<'_> {
    fn poll(&mut self, cx: &mut Context<'_>, batch: &WriteBatch) -> Poll<Turn> {
        self.inner
            .writes
            .borrow_mut()
            .poll(&mut self.slot, batch, cx)
    }
}

impl Drop for Queued<'_> {
    fn drop(&mut self) {
        if let Some(slot) = self.slot
            && let Some(mut writes) = for_drop(&self.inner.writes)
        {
            writes.leave(slot);
        }
    }
}

/// Reads the value leaf with page id `page`, which the cache doesn't keep, into a
/// buffer, `old` if it is large enough, and checks it. No value leaf a reader can
/// reach was written after the current generation, or lies past the high-water
/// mark: the span its parent gives is checked before the buffer is taken, since
/// the open no longer checks it (see [`Scan`]).
async fn read_value_leaf(inner: &Inner, page: u64, old: Option<ValueBuf>) -> Result<ValueBuf> {
    let first = node::first_page(page);
    if first + node::span(page) > inner.tree.borrow().pages.space.high_water() {
        return Err(Error::Corruption { page: first });
    }
    let len = node::span(page) as usize * inner.page_size;
    let buf = ValueBuf::new(old, len);
    inner.metrics.read.add(len);
    let timing = inner.metrics.read_duration.start();
    let (res, buf) = inner.storage.read_value(buf, inner.offset(first)).await;
    timing.stop();
    if res.is_err() {
        return Err(read_error(res, page));
    }
    let generation = inner.tree.borrow().current().generation;
    if !node::verify_value_leaf(buf.bytes(), inner.page_size)
        || ValueLeaf::new(buf.bytes()).generation() > generation
    {
        return Err(Error::Corruption { page: first });
    }
    Ok(buf)
}

/// Pins the frame holding the page of a leaf or a level-1 node, at `level`,
/// reading the page in if it isn't cached. A read of the page already in flight is
/// waited for, not repeated.
async fn load_node(inner: &Inner, page: u64, level: u8) -> Result<FrameId> {
    let mut load = Load {
        inner,
        slot: None,
        missed: false,
    };
    poll_fn(|cx| load.poll(cx, page, level)).await
}

/// A load in progress, and the read it holds, if any, which it lets go of when
/// dropped. It counts a cache miss once it has to read the node.
struct Load<'a> {
    inner: &'a Inner,
    slot: Option<usize>,
    missed: bool,
}

impl Load<'_> {
    fn poll(&mut self, cx: &mut Context<'_>, page: u64, level: u8) -> Poll<Result<FrameId>> {
        let inner = self.inner;
        let mut tree = inner.tree.borrow_mut();
        let mut reads = inner.reads.borrow_mut();
        loop {
            if self.slot.is_none() {
                if let Some(frame) = tree.pages.cached(page, level) {
                    if !self.missed {
                        tree.pages.cache.hits.add(1);
                    }
                    tree.pages.pool.retain(frame);
                    return Poll::Ready(Ok(frame));
                }
                if !mem::replace(&mut self.missed, true) {
                    tree.pages.cache.misses.add(1);
                }
                self.slot = Some(read_leaf(inner, &mut tree, &mut reads, page));
            }
            let slot = &mut self.slot;
            if let Some(loaded) = ready!(poll_read(&mut tree, &mut reads, cx, page, level, slot)) {
                return Poll::Ready(loaded);
            }
        }
    }
}

impl Drop for Load<'_> {
    fn drop(&mut self) {
        if let Some(slot) = self.slot
            && let Some(mut tree) = for_drop(&self.inner.tree)
            && let Some(mut reads) = for_drop(&self.inner.reads)
        {
            reads.leave(slot, &mut tree.pages.pool);
        }
    }
}

/// The slot of a read of the leaf or level-1 node on `page`, which the caller then
/// holds: the read in flight, or a new one.
fn read_leaf(inner: &Inner, tree: &mut Tree, reads: &mut Reads, page: u64) -> usize {
    if let Some(slot) = reads.find(page) {
        reads.join(slot);
        return slot;
    }
    let pages = &mut tree.pages;
    let frame = match node::span(page) {
        0 => pages.cache.alloc(&mut pages.pool, Holds::Loading, page),
        span => {
            let pool = &mut pages.pool;
            pages
                .cache
                .alloc_large(pool, Holds::Loading, page, span as usize)
        }
    };
    reads.start(page, frame, start_read(inner, &pages.pool, frame, page))
}

/// Starts reading the leaf with page id `page` into `frame`: a large frame for a
/// value leaf.
fn start_read(inner: &Inner, pool: &Pool, frame: FrameId, page: u64) -> PageRead {
    // The buffer takes the frame's only reference: if the read is cancelled, the
    // frame is reclaimed once the read completes.
    let offset = inner.offset(node::first_page(page));
    inner
        .metrics
        .read
        .add(node::pages(page) as usize * inner.page_size);
    inner.storage.start_read(pool.adopt_buf(frame), offset)
}

/// Waits for the read of `page`, a leaf or a level-1 node at `level`, held in
/// `slot` to finish, and lets go of it. If this reader claims the read, returns the
/// pinned frame, or the read's error. Returns `None` if another reader claimed it:
/// the page is cached then, unless the read failed.
fn poll_read(
    tree: &mut Tree,
    reads: &mut Reads,
    cx: &mut Context<'_>,
    page: u64,
    level: u8,
    slot: &mut Option<usize>,
) -> Poll<Option<Result<FrameId>>> {
    let held = slot.expect("no read held");
    let claimed = ready!(reads.poll(held, cx));
    *slot = None;
    reads.leave(held, &mut tree.pages.pool);
    let generation = tree.current().generation;
    Poll::Ready(claimed.map(|claim| finish(&mut tree.pages, page, level, claim, generation)))
}

/// Checks and caches the page a claimed read brought in, a leaf or a level-1 node
/// at `level`, pinning its frame. No node a reader can reach was written after
/// `generation`, the current one, or lies past the high-water mark.
fn finish(
    pages: &mut Pages,
    page: u64,
    level: u8,
    claim: Claim,
    generation: u64,
) -> Result<FrameId> {
    let Claim { frame, res, buf } = claim;
    pages.pool.retain(frame);
    drop(buf);
    let bytes = pages.pool.page(frame);
    let within = node::first_page(page) + node::pages(page) <= pages.space.high_water();
    let intact = within
        && if node::span(page) > 0 {
            node::verify_value_leaf(bytes, pages.pool.page_size())
                && ValueLeaf::new(bytes).generation() <= generation
        } else {
            node::verify(bytes)
                && Node::new(bytes).level() == level
                && Node::new(bytes).generation() <= generation
        };
    if res.is_err() || !intact {
        pages.pool.unref(frame);
        pages.pool.free(frame);
        return Err(read_error(res, page));
    }
    pages.adopt_leaf(frame, page);
    Ok(frame)
}

/// The error of a read of the node with page id `page`, which brought in bytes
/// that failed their checks if `res` is `Ok`. No node lies past the end of the
/// file, so a tree that lists one there is corrupt too.
fn read_error(res: io::Result<()>, page: u64) -> Error {
    match res {
        Err(e) if e.kind() != io::ErrorKind::UnexpectedEof => e.into(),
        _ => Error::Corruption {
            page: node::first_page(page),
        },
    }
}

/// Writes the pending version's dirty nodes, in page order: each run of adjacent
/// pages with one vectored write, up to `MAX_WRITE` bytes, `IO_WINDOW` writes at a
/// time. Every write completes before this returns, even when one fails.
async fn write_dirty(inner: &Inner) -> Result<()> {
    let max_run = max_run(inner.page_size);
    let mut error = None;
    // How many of the nodes of each life the writes took.
    let mut next = [0; 2];
    while inner.tree.borrow().next_dirty(next).is_some() {
        let writes: [_; IO_WINDOW] = {
            let tree = inner.tree.borrow();
            let mut runs = inner.runs.borrow_mut();
            array::from_fn(|_| {
                let (_, _, first) = tree.next_dirty(next)?;
                let mut bufs = runs.pop().unwrap_or_else(|| Vec::with_capacity(max_run));
                while let Some((life, frame, page)) = tree.next_dirty(next)
                    && page == first + bufs.len() as u64
                    && bufs.len() < max_run
                {
                    bufs.push(tree.pages.pool.buf(frame));
                    next[life] += 1;
                }
                inner.metrics.written.add(bufs.len() * inner.page_size);
                Some(inner.storage.write_run(bufs, inner.offset(first)))
            })
        };
        for (res, mut bufs) in join(writes).await.into_iter().flatten() {
            bufs.clear();
            inner.runs.borrow_mut().push(bufs);
            if let Err(e) = res {
                error = error.or(Some(e));
            }
        }
    }
    error.map_or(Ok(()), |e| Err(e.into()))
}

/// Writes the pending version's value leaves, whose entries are in the group's
/// batch or else in `batch`: each in requests of up to `MAX_WRITE` bytes,
/// `IO_WINDOW` at a time, copied from the batch into frames that are freed once
/// written. Every write completes before this returns, even when one fails.
async fn write_values(
    inner: &Inner,
    generation: u64,
    group: bool,
    batch: &WriteBatch,
) -> Result<()> {
    let (page_size, max_run) = (inner.page_size, max_run(inner.page_size));
    let total = inner.tree.borrow().values.len();
    let mut error = None;
    // The value leaf being written, its header, and the next of its pages to write.
    let (mut leaf, mut header, mut next) = (0, [0; node::HEADER], 0);
    while leaf < total {
        let writes: [_; IO_WINDOW] = {
            let mut writes = inner.writes.borrow_mut();
            let batch: &WriteBatch = if group { writes.group() } else { batch };
            let mut tree = inner.tree.borrow_mut();
            let mut runs = inner.runs.borrow_mut();
            array::from_fn(|_| {
                let &(page, op) = tree.values.get(leaf)?;
                let (key, value) = (batch.key(op), batch.value(op).unwrap());
                let span = node::span(page) as usize;
                if next == 0 {
                    header = node::value_header(key, value, generation, span * page_size);
                }
                let first = next;
                let mut bufs = runs.pop().unwrap_or_else(|| Vec::with_capacity(max_run));
                while next < span && bufs.len() < max_run {
                    let pages = &mut tree.pages;
                    let frame = pages.cache.alloc(&mut pages.pool, Holds::Loading, 0);
                    let out = pages.pool.page_mut(frame);
                    node::copy_value_leaf(out, next * page_size, &header, key, value);
                    // The buffer takes the frame's only reference.
                    bufs.push(pages.pool.adopt_buf(frame));
                    next += 1;
                }
                if next == span {
                    (leaf, next) = (leaf + 1, 0);
                }
                let at = inner.offset(node::first_page(page) + first as u64);
                inner.metrics.written.add(bufs.len() * page_size);
                Some(inner.storage.write_run(bufs, at))
            })
        };
        for (res, mut bufs) in join(writes).await.into_iter().flatten() {
            let mut tree = inner.tree.borrow_mut();
            for buf in bufs.drain(..) {
                let frame = buf.id();
                drop(buf);
                tree.pages.pool.free(frame);
            }
            inner.runs.borrow_mut().push(bufs);
            if let Err(e) = res {
                error = error.or(Some(e));
            }
        }
    }
    error.map_or(Ok(()), |e| Err(e.into()))
}

/// The most pages one write request carries.
pub(crate) fn max_run(page_size: usize) -> usize {
    (MAX_WRITE / page_size).clamp(1, MAX_IOVECS)
}

/// Cuts the file at `high_water`, past the pages that the version just committed
/// and the older ones still read can use, once that frees an eighth of it. Only
/// the newest meta page, which is durable, still matters. If this fails, the file
/// stays longer than it needs to be until a later commit cuts it.
async fn shrink(inner: &Inner, high_water: u64) {
    let end = inner.file_end.get();
    if 8 * (end - high_water) >= end
        && inner
            .storage
            .set_len(inner.offset(high_water))
            .await
            .is_ok()
    {
        inner.file_end.set(high_water);
    }
}

/// Writes `meta` to its slot.
async fn write_meta(inner: &Inner, meta: &Meta) -> Result<()> {
    let (frame, buf) = {
        let mut tree = inner.tree.borrow_mut();
        let pages = &mut tree.pages;
        let frame = pages.cache.alloc(&mut pages.pool, Holds::Loading, 0);
        meta.encode(pages.pool.page_mut(frame));
        // The buffer takes the frame's only reference, as in `load_leaf`.
        (frame, pages.pool.adopt_buf(frame))
    };
    inner.metrics.written.add(inner.page_size);
    let (res, buf) = inner.storage.write(buf, inner.offset(meta.slot())).await;
    drop(buf);
    inner.tree.borrow_mut().pages.pool.free(frame);
    Ok(res?)
}

/// Lets the runtime run the other tasks that are ready, and poll for I/O, before
/// this one goes on.
async fn yield_now() {
    let mut yielded = false;
    poll_fn(|cx| {
        if yielded {
            return Poll::Ready(());
        }
        yielded = true;
        cx.waker().wake_by_ref();
        Poll::Pending
    })
    .await
}

/// Runs the futures in `slots` concurrently, returning their outputs in order.
/// Unlike `join_all`, it keeps everything on the stack.
async fn join<F: Future, const N: usize>(slots: [Option<F>; N]) -> [Option<F::Output>; N] {
    let mut outputs: [Option<F::Output>; N] = array::from_fn(|_| None);
    let mut slots = pin!(slots);
    poll_fn(|cx| {
        let mut pending = false;
        for (i, output) in outputs.iter_mut().enumerate() {
            // SAFETY: the futures stay pinned in the array, and are only dropped in
            // place.
            let mut slot = unsafe { slots.as_mut().map_unchecked_mut(|s| &mut s[i]) };
            if let Some(future) = slot.as_mut().as_pin_mut() {
                match future.poll(cx) {
                    Poll::Ready(value) => {
                        *output = Some(value);
                        slot.set(None);
                    }
                    Poll::Pending => pending = true,
                }
            }
        }
        if pending {
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    })
    .await;
    outputs
}

/// How far a write got, which decides what an early exit has to undo.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// Nothing written: the pending version, if any, is thrown away.
    Building,
    /// Node writes may be in flight.
    Writing,
    /// The nodes are written: whether the commit is durable is unknown.
    Committing,
    /// The commit is published, and the file may be getting cut: a cut still in
    /// flight could cut what a later write adds.
    Cutting,
    Done,
}

/// A write committing, alone or for a group of writes.
struct WriteGuard<'a> {
    inner: &'a Inner,
    stage: Stage,
    /// Whether it commits a group of writes, and its slot among them, if the group
    /// holds its batch.
    group: bool,
    slot: Option<usize>,
    /// Whether the group's writes have their outcome.
    delivered: bool,
}

impl WriteGuard<'_> {
    /// Gives the other writes in the group `result`, the first time.
    fn deliver(&mut self, result: &Result<()>) {
        if self.group && !self.delivered {
            self.inner.writes.borrow_mut().deliver(self.slot, result);
        }
        self.delivered = true;
    }
}

/// Borrows the tree or the reads for a destructor. While a panic unwinds, they may
/// still be borrowed by the code that panicked: then the cleanup is skipped.
fn for_drop<T>(cell: &RefCell<T>) -> Option<RefMut<'_, T>> {
    match cell.try_borrow_mut() {
        Ok(value) => Some(value),
        Err(_) if std::thread::panicking() => None,
        Err(e) => panic!("borrowed during cleanup: {e}"),
    }
}

impl Drop for WriteGuard<'_> {
    fn drop(&mut self) {
        let (Some(mut tree), Some(mut writes)) =
            (for_drop(&self.inner.tree), for_drop(&self.inner.writes))
        else {
            return;
        };
        match self.stage {
            Stage::Done => {}
            Stage::Building => tree.discard(),
            // In-flight writes still hold the pending version's frames: leave them.
            Stage::Writing => {
                self.inner.fail();
                tree.unpin_touched();
            }
            Stage::Committing => {
                self.inner.fail();
                tree.discard();
            }
            Stage::Cutting => self.inner.fail(),
        }
        // Dropped before the group's writes had their outcome.
        if self.group && !self.delivered {
            if self.stage == Stage::Building {
                // Nothing was written: the others go ahead without this write.
                writes.abandon(self.slot);
            } else {
                writes.deliver(self.slot, &Err(Error::Failed));
            }
        }
        writes.release();
    }
}

/// What a lookup holds while it runs: a key buffer, and the version and nodes it
/// pinned if it had to wait for the disk.
struct Read<'a> {
    inner: &'a Inner,
    key: Vec<u8>,
    version: Option<Version>,
    nodes: [FrameId; 4],
    loaded: usize,
}

impl Drop for Read<'_> {
    fn drop(&mut self) {
        if let Some(mut tree) = for_drop(&self.inner.tree) {
            for &frame in &self.nodes[..self.loaded] {
                tree.pages.pool.unref(frame);
            }
            if let Some(version) = self.version {
                tree.unpin(version);
                self.inner.metrics.readers.add(-1);
            }
        }
        self.inner.put_buffer(mem::take(&mut self.key));
    }
}

fn copy_bound(bound: Bound<&[u8]>, out: &mut Vec<u8>) -> Bound<()> {
    match bound {
        Bound::Included(key) => {
            out.extend_from_slice(key);
            Bound::Included(())
        }
        Bound::Excluded(key) => {
            out.extend_from_slice(key);
            Bound::Excluded(())
        }
        Bound::Unbounded => Bound::Unbounded,
    }
}

/// A point-in-time iterator over a range of entries, from [`Db::range`] or
/// [`ReadOnlyDb::range`].
pub struct Iter {
    inner: Rc<Inner>,
    version: Version,
    cursor: Cursor,
    /// The level-1 node `cursor` is at, which the iterator pins.
    level_1: Option<FrameId>,
    /// The pinned leaf the iterator is on: a cached value leaf if `on_value`.
    leaf: Option<FrameId>,
    /// Whether it is on a value leaf: the one `leaf` pins, or else, for one too
    /// large to cache, the one read into `value`.
    on_value: bool,
    /// The last value leaf too large to cache that it read, whose buffer the next
    /// one reuses.
    value: Option<ValueBuf>,
    at: usize,
    state: State,
    start: Bound<()>,
    end: Bound<()>,
    /// `bounds` holds the start key, then the end key.
    start_len: usize,
    bounds: Vec<u8>,
    key: Vec<u8>,
    /// The read of the leaf being entered, while the iterator waits for it.
    waiting: Option<usize>,
    ahead: Ahead,
    /// Whether entering the leaf counted a cache miss, having to read it.
    missed: bool,
    /// The entries returned, counted when it is dropped.
    returned: u64,
}

#[derive(Clone, Copy)]
enum State {
    Unstarted,
    /// Next, enter this leaf: at its first entry, or where the start bound falls.
    Enter {
        page: u64,
        first: bool,
    },
    /// Next, read the level-1 node on page `node`, which the next leaf is under,
    /// then move to that leaf.
    Cross {
        node: u64,
    },
    /// The entry at `at` is next.
    Ready,
    /// The entry at `at` was returned.
    Returned,
    Done,
}

impl Iter {
    /// Lends the next entry's key and value, valid until the next call.
    ///
    /// If reading a leaf fails, this returns the error, and the next call reads it
    /// again.
    pub async fn next(&mut self) -> Result<Option<(&[u8], &[u8])>> {
        let Some(root) = self.version.root else {
            return Ok(None);
        };
        loop {
            match self.state {
                State::Done => return Ok(None),
                State::Ready => break,
                State::Unstarted => {
                    let found = {
                        let mut tree = self.inner.tree.borrow_mut();
                        let start = &self.bounds[..self.start_len];
                        let found = match self.start {
                            Bound::Unbounded => tree.descend_first(root, &mut self.cursor),
                            _ => tree.descend(root, start, &mut self.cursor),
                        };
                        if found.is_ok() {
                            hold_level_1(&tree.pages.pool, &mut self.level_1, &self.cursor);
                        }
                        found
                    };
                    match found {
                        Ok(page) => {
                            let first = matches!(self.start, Bound::Unbounded);
                            self.state = State::Enter { page, first };
                        }
                        Err(node) => self.load_level_1(node).await?,
                    }
                }
                State::Cross { node } => {
                    self.load_level_1(node).await?;
                    self.state = self.next_leaf();
                }
                State::Enter { page, first } if node::span(page) > 0 => {
                    // Its key, its parent's key for it, tells whether it is in the
                    // range before it is read.
                    let (before, past) = self.bounds_of_value_leaf(first);
                    if past {
                        self.state = State::Done;
                        continue;
                    }
                    if before {
                        self.state = self.next_leaf();
                        continue;
                    }
                    let cached = self
                        .inner
                        .tree
                        .borrow()
                        .pages
                        .cache
                        .caches(node::span(page));
                    let frame = if cached {
                        match poll_fn(|cx| self.poll_enter(cx, page)).await {
                            Ok(frame) => Some(frame),
                            Err(e) => return Err(self.inner.explain(e).await),
                        }
                    } else {
                        match read_value_leaf(&self.inner, page, self.value.take()).await {
                            Ok(buf) => self.value = Some(buf),
                            Err(e) => return Err(self.inner.explain(e).await),
                        }
                        None
                    };
                    if let Some(old) = mem::replace(&mut self.leaf, frame) {
                        self.inner.tree.borrow().pages.pool.unref(old);
                    }
                    (self.on_value, self.at) = (true, 0);
                    self.state = State::Ready;
                }
                State::Enter { page, first } => {
                    let frame = match poll_fn(|cx| self.poll_enter(cx, page)).await {
                        Ok(frame) => frame,
                        Err(e) => return Err(self.inner.explain(e).await),
                    };
                    let tree = self.inner.tree.borrow();
                    if let Some(old) = self.leaf.replace(frame) {
                        tree.pages.pool.unref(old);
                    }
                    self.on_value = false;
                    let node = tree.node(frame);
                    self.at = if first {
                        0
                    } else {
                        let (i, eq) = node.search(&self.bounds[..self.start_len]);
                        i + usize::from(eq && matches!(self.start, Bound::Excluded(())))
                    };
                    let count = node.count();
                    drop(tree);
                    self.state = if self.at < count {
                        State::Ready
                    } else {
                        self.next_leaf()
                    };
                }
                State::Returned => {
                    let count = match self.leaf {
                        Some(leaf) if !self.on_value => self.inner.tree.borrow().node(leaf).count(),
                        _ => 1,
                    };
                    self.at += 1;
                    self.state = if self.at < count {
                        State::Ready
                    } else {
                        self.next_leaf()
                    };
                }
            }
        }
        if self.on_value {
            // In the range: entering it checked.
            self.state = State::Returned;
            self.returned += 1;
            let bytes = match self.leaf {
                // SAFETY: the iterator pins this leaf, and frames are never modified
                // while referenced.
                Some(frame) => unsafe { self.inner.tree.borrow().pages.pool.page_unbounded(frame) },
                None => self.value.as_ref().unwrap().bytes(),
            };
            let leaf = ValueLeaf::new(bytes);
            return Ok(Some((leaf.key(), leaf.value())));
        }
        let frame = self.leaf.unwrap();
        let tree = self.inner.tree.borrow();
        let node = tree.node(frame);
        node.key_into(self.at, &mut self.key);
        let end = &self.bounds[self.start_len..];
        let in_range = match self.end {
            Bound::Included(()) => self.key.as_slice() <= end,
            Bound::Excluded(()) => self.key.as_slice() < end,
            Bound::Unbounded => true,
        };
        if !in_range {
            self.state = State::Done;
            return Ok(None);
        }
        self.state = State::Returned;
        self.returned += 1;
        let Some((offset, len)) = node.whole_value(self.at) else {
            // A dense leaf: the value goes after the key.
            let key_len = self.key.len();
            node.append_value(self.at, &mut self.key);
            return Ok(Some(self.key.split_at(key_len)));
        };
        let value = tree.addr(frame, offset);
        drop(tree);
        // SAFETY: the iterator pins this leaf, and frames are never modified while
        // referenced.
        Ok(Some((&self.key, unsafe {
            slice::from_raw_parts(value, len)
        })))
    }

    /// Reads the level-1 node on `page` into the cache, where the step the
    /// iterator takes next finds it: nothing can evict it before then, since the
    /// iterator doesn't wait meanwhile.
    async fn load_level_1(&mut self, page: u64) -> Result<()> {
        match poll_fn(|cx| self.poll_level_1(cx, page)).await {
            Ok(()) => Ok(()),
            Err(e) => Err(self.inner.explain(e).await),
        }
    }

    /// Waits for the level-1 node on `page` to be cached, reading it if it isn't.
    /// As in [`poll_enter`](Self::poll_enter), the read stays held across calls.
    fn poll_level_1(&mut self, cx: &mut Context<'_>, page: u64) -> Poll<Result<()>> {
        let inner = &*self.inner;
        let mut tree = inner.tree.borrow_mut();
        let mut reads = inner.reads.borrow_mut();
        loop {
            if self.waiting.is_none() {
                if tree.pages.cached(page, 1).is_some() {
                    return Poll::Ready(Ok(()));
                }
                tree.pages.cache.misses.add(1);
                self.waiting = Some(read_leaf(inner, &mut tree, &mut reads, page));
            }
            let polled = poll_read(&mut tree, &mut reads, cx, page, 1, &mut self.waiting);
            if let Some(loaded) = ready!(polled) {
                return Poll::Ready(loaded.map(|frame| {
                    tree.pages.pool.unref(frame);
                }));
            }
        }
    }

    /// Pins leaf `page`, at `self.cursor`. If it isn't cached, waits for a read of
    /// it, and reads the leaves after it meanwhile. The read stays held across
    /// calls, in case the caller drops the future and calls again.
    fn poll_enter(&mut self, cx: &mut Context<'_>, page: u64) -> Poll<Result<FrameId>> {
        let inner = &*self.inner;
        let mut tree = inner.tree.borrow_mut();
        let mut reads = inner.reads.borrow_mut();
        let end = match self.end {
            Bound::Included(()) => Bound::Included(&self.bounds[self.start_len..]),
            Bound::Excluded(()) => Bound::Excluded(&self.bounds[self.start_len..]),
            Bound::Unbounded => Bound::Unbounded,
        };
        loop {
            if self.waiting.is_none() {
                if self.ahead.front() == Some(page) {
                    self.waiting = self.ahead.pop();
                } else if let Some(frame) = tree.pages.cached(page, 0) {
                    if !mem::take(&mut self.missed) {
                        tree.pages.cache.hits.add(1);
                    }
                    tree.pages.pool.retain(frame);
                    return Poll::Ready(Ok(frame));
                } else {
                    self.waiting = Some(read_leaf(inner, &mut tree, &mut reads, page));
                }
                if !mem::replace(&mut self.missed, true) {
                    tree.pages.cache.misses.add(1);
                }
            }
            // Polling first sends this leaf's read out ahead of the others.
            let polled = poll_read(&mut tree, &mut reads, cx, page, 0, &mut self.waiting);
            self.ahead
                .fill(inner, &mut tree, &mut reads, &self.cursor, end);
            if let Some(loaded) = ready!(polled) {
                self.missed = false;
                return Poll::Ready(loaded);
            }
        }
    }

    /// Whether the value leaf at `self.cursor` falls before the start of the range,
    /// unless it is entered `first`, and whether it falls past the end.
    fn bounds_of_value_leaf(&self, first: bool) -> (bool, bool) {
        let tree = self.inner.tree.borrow();
        let (start, end) = self.bounds.split_at(self.start_len);
        let before = !first
            && match self.start {
                Bound::Included(()) => tree.cmp_leaf_key(&self.cursor, start).is_lt(),
                Bound::Excluded(()) => tree.cmp_leaf_key(&self.cursor, start).is_le(),
                Bound::Unbounded => false,
            };
        let past = match self.end {
            Bound::Included(()) => tree.cmp_leaf_key(&self.cursor, end).is_gt(),
            Bound::Excluded(()) => tree.cmp_leaf_key(&self.cursor, end).is_ge(),
            Bound::Unbounded => false,
        };
        (before, past)
    }

    fn next_leaf(&mut self) -> State {
        let mut tree = self.inner.tree.borrow_mut();
        match tree.next_leaf(&mut self.cursor, true) {
            Ok(Some(page)) => {
                hold_level_1(&tree.pages.pool, &mut self.level_1, &self.cursor);
                self.ahead.passed(window_pages(&tree.pages.cache, page));
                State::Enter { page, first: true }
            }
            Ok(None) => State::Done,
            Err(node) => State::Cross { node },
        }
    }
}

/// Moves the pin `held` to the level-1 node `cur` is at.
fn hold_level_1(pool: &Pool, held: &mut Option<FrameId>, cur: &Cursor) {
    let at = cur.level_1();
    if *held == at {
        return;
    }
    if let Some(frame) = at {
        pool.retain(frame);
    }
    if let Some(old) = mem::replace(held, at) {
        pool.unref(old);
    }
}

impl Drop for Iter {
    fn drop(&mut self) {
        if let Some(mut tree) = for_drop(&self.inner.tree) {
            if let Some(mut reads) = for_drop(&self.inner.reads) {
                let held = self.waiting.take();
                let node = self.ahead.node.take().map(|(_, slot)| slot);
                let ahead = iter::from_fn(|| self.ahead.pop());
                for slot in held.into_iter().chain(node).chain(ahead) {
                    reads.leave(slot, &mut tree.pages.pool);
                }
            }
            let pinned = [
                self.leaf.take(),
                self.level_1.take(),
                self.ahead.level_1.take(),
            ];
            for frame in pinned.into_iter().flatten() {
                tree.pages.pool.unref(frame);
            }
            tree.unpin(self.version);
            self.inner.metrics.readers.add(-1);
        }
        self.inner.metrics.range_entries.add(self.returned);
        self.inner.put_buffer(mem::take(&mut self.bounds));
        self.inner.put_buffer(mem::take(&mut self.key));
    }
}

/// The pages the leaf with page id `page` takes in the window an iterator reads
/// ahead: a value leaf too large to cache isn't read ahead, and takes none.
fn window_pages(cache: &Cache, page: u64) -> usize {
    match node::span(page) {
        0 => 1,
        span if cache.caches(span) => span as usize,
        _ => 0,
    }
}

/// The leaves an iterator reads ahead of the one it is on.
struct Ahead {
    /// The last leaf considered, `by` leaves past the current one, which take
    /// `pages` pages. When `by` is 0 it may be behind, and the next fill starts
    /// over from the current leaf. Its level-1 node stays pinned.
    cursor: Cursor,
    level_1: Option<FrameId>,
    by: usize,
    pages: usize,
    /// No leaf after `cursor` is in the range.
    done: bool,
    /// The reads held for the leaves considered that weren't cached, in key
    /// order: a ring of (page, slot), `len` of them from `first`.
    reads: [(u64, usize); IO_WINDOW],
    first: usize,
    len: usize,
    /// The read held for the level-1 node the next leaf is under, if it isn't
    /// cached, until someone claims it: (page, slot).
    node: Option<(u64, usize)>,
}

impl Ahead {
    fn new() -> Ahead {
        Ahead {
            cursor: Cursor::new(),
            level_1: None,
            by: 0,
            pages: 0,
            done: false,
            reads: [(0, 0); IO_WINDOW],
            first: 0,
            len: 0,
            node: None,
        }
    }

    /// The page of the next leaf read ahead.
    fn front(&self) -> Option<u64> {
        (self.len > 0).then(|| self.reads[self.first].0)
    }

    /// Takes the read of the next leaf read ahead.
    fn pop(&mut self) -> Option<usize> {
        if self.len == 0 {
            return None;
        }
        let (_, slot) = self.reads[self.first];
        self.first = (self.first + 1) % IO_WINDOW;
        self.len -= 1;
        Some(slot)
    }

    /// Moves past the first leaf considered, which takes `pages` pages: the
    /// iterator enters it.
    fn passed(&mut self, pages: usize) {
        if self.by > 0 {
            self.by -= 1;
            self.pages -= pages;
        }
    }

    /// Holds the read of the last leaf considered, at `page`.
    fn push(&mut self, page: u64, slot: usize) {
        // Each read held is for a different leaf of the `by` ahead.
        debug_assert!(self.len < self.by, "too many reads ahead");
        self.reads[(self.first + self.len) % IO_WINDOW] = (page, slot);
        self.len += 1;
    }

    /// Starts reading the leaves after the current one, at `current`, that aren't
    /// cached and fall in the range ending at `end`: at most `IO_WINDOW` leaves
    /// ahead, which take at most a quarter of the cache budget, so that reading
    /// ahead never crowds out the leaves other readers use. Reaching a level-1 node
    /// that isn't cached, it reads that, and goes on once someone has claimed the
    /// read.
    fn fill(
        &mut self,
        inner: &Inner,
        tree: &mut Tree,
        reads: &mut Reads,
        current: &Cursor,
        end: Bound<&[u8]>,
    ) {
        let window = tree.pages.cache.budget() / 4;
        if self.by == 0 {
            self.cursor.clone_from(current);
            hold_level_1(&tree.pages.pool, &mut self.level_1, &self.cursor);
            self.pages = 0;
        }
        if let Some((page, slot)) = self.node
            && reads.find(page) != Some(slot)
        {
            reads.leave(slot, &mut tree.pages.pool);
            self.node = None;
        }
        while !self.done && self.by < IO_WINDOW {
            let mut next = self.cursor.clone();
            let page = match tree.next_leaf(&mut next, true) {
                Ok(Some(page)) => page,
                Ok(None) => {
                    self.done = true;
                    break;
                }
                Err(node) => {
                    if self.node.is_none() {
                        self.node = read_ahead(inner, tree, reads, node);
                    }
                    break;
                }
            };
            let past = match end {
                Bound::Included(end) => tree.cmp_leaf_key(&next, end).is_gt(),
                Bound::Excluded(end) => tree.cmp_leaf_key(&next, end).is_ge(),
                Bound::Unbounded => false,
            };
            if past {
                self.done = true;
                break;
            }
            let pages = window_pages(&tree.pages.cache, page);
            if self.pages + pages > window {
                break;
            }
            // Finding a frame for the read may evict cold nodes: the level-1 node
            // `next` is at stays pinned meanwhile.
            let mut pinned = None;
            hold_level_1(&tree.pages.pool, &mut pinned, &next);
            let read = if pages == 0 || tree.pages.cached(page, 0).is_some() {
                Ok(None)
            } else {
                read_ahead(inner, tree, reads, page).map(Some).ok_or(())
            };
            if read.is_ok() {
                self.cursor = next;
                hold_level_1(&tree.pages.pool, &mut self.level_1, &self.cursor);
            }
            if let Some(frame) = pinned {
                tree.pages.pool.unref(frame);
            }
            let Ok(read) = read else {
                break;
            };
            self.by += 1;
            self.pages += pages;
            if let Some((page, slot)) = read {
                self.push(page, slot);
            }
        }
    }
}

/// Holds a read of the page with id `page`, which isn't cached: the read in
/// flight, or a new one, if a frame within the budget is free for it, which reading
/// ahead never grows the pool past. Returns the page and the slot.
fn read_ahead(
    inner: &Inner,
    tree: &mut Tree,
    reads: &mut Reads,
    page: u64,
) -> Option<(u64, usize)> {
    if let Some(slot) = reads.find(page) {
        reads.join(slot);
        return Some((page, slot));
    }
    let (pool, cache) = (&mut tree.pages.pool, &mut tree.pages.cache);
    let frame = match node::span(page) {
        0 => cache.try_alloc(pool, Holds::Loading, page)?,
        span => cache.try_alloc_large(pool, Holds::Loading, page, span as usize)?,
    };
    let slot = reads.start(page, frame, start_read(inner, pool, frame, page));
    reads.submit(slot);
    Some((page, slot))
}
