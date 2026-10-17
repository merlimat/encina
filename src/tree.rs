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

//! The copy-on-write B+tree.
//!
//! Nodes are immutable once published. A write builds replacement nodes along the
//! paths it touches and shares everything else with the previous version; once
//! they are committed, it swaps the root.
//!
//! Internal nodes stay in memory and refer to internal children by frame. Those
//! frames are reference counted: every node holds a reference to each internal
//! child, the current root holds one, and so does each pinned reader. Leaves are
//! found by page id through the leaf cache, so they can be evicted. A page a write
//! replaces stays allocated until no reader of an older version is left.
//!
//! Each internal key equals the smallest key in its child's subtree.
//!
//! An entry too large for a leaf gets a value leaf of its own, which no write ever
//! reads or rewrites: its key is its parent's key for it, and a write that replaces
//! or removes its entry drops it whole. A value leaf always has a parent, so the
//! root is never one.

use std::cmp::Ordering;
use std::mem;

use crate::Comparison;
use crate::batch::WriteBatch;
use crate::cache::Cache;
use crate::meta::Meta;
use crate::node::{self, CHILD_REF, Entries, EntryBuf, NO_FRAME, Node};
use crate::pool::{FrameId, Holds, Pool};
use crate::space::{Life, Space};

/// The deepest tree supported. Every split leaves internal nodes with at least two
/// children, so real trees stay far shallower.
pub(crate) const MAX_HEIGHT: usize = 32;

/// A version's root node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Root {
    pub(crate) page: u64,
    /// The root's frame, or [`NO_FRAME`] when it is a leaf.
    pub(crate) frame: FrameId,
    pub(crate) level: u8,
}

/// A version a reader pinned: its root, and the generation that published it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Version {
    pub(crate) root: Option<Root>,
    pub(crate) generation: u64,
}

/// The frames, the leaf cache and the page allocator.
pub(crate) struct Pages {
    pub(crate) pool: Pool,
    pub(crate) cache: Cache,
    pub(crate) space: Space,
}

impl Pages {
    /// Builds entries `a..b` of `src` into a new node at `level`, on a new page. The
    /// node is dirty until its write publishes. A leaf goes into the cache, holding
    /// a reference until then; an internal node's one reference goes to the caller.
    /// Returns the node's frame ([`NO_FRAME`] for a leaf), page, and bytes used.
    fn build(
        &mut self,
        level: u8,
        src: &impl Entries,
        a: usize,
        b: usize,
        life: Life,
    ) -> (FrameId, u64, usize) {
        let page = self.space.alloc(life);
        let holds = if level == 0 {
            Holds::Leaf
        } else {
            Holds::Internal
        };
        let frame = self.cache.alloc(&mut self.pool, holds, page);
        let used = node::build(self.pool.page_mut(frame), level, src, a, b);
        self.pool.frame_mut(frame).dirty = true;
        if level > 0 {
            return (frame, page, used);
        }
        self.cache.insert(page, frame);
        (NO_FRAME, page, used)
    }

    /// The cached frame of a leaf page.
    pub(crate) fn cached(&mut self, page: u64) -> Option<FrameId> {
        self.cache.get(&mut self.pool, page)
    }

    /// Caches the leaf page just read into `frame`. Reads of a page are shared, so
    /// it can't be cached already.
    pub(crate) fn adopt_leaf(&mut self, frame: FrameId, page: u64) {
        let info = self.pool.frame_mut(frame);
        info.holds = Holds::Leaf;
        info.page = page;
        self.cache.insert(page, frame);
    }
}

pub(crate) struct Tree {
    pub(crate) pages: Pages,
    root: Option<Root>,
    generation: u64,
    page_size: usize,
    // Scratch space, reused so that writes don't allocate:
    /// Replacement nodes, per level.
    lists: Vec<NodeList>,
    /// Separator keys, two per level.
    bounds: Vec<Vec<u8>>,
    /// Where each run of puts ends among the entries of the leaf being rebuilt.
    run_ends: Vec<u32>,
    /// Where each value leaf goes among those entries, and the put that holds it.
    value_puts: Vec<(u32, u32)>,
    /// Frames waiting to be released.
    stack: Vec<FrameId>,
    /// The leaf pages the write in progress changes, found before it loads them.
    pub(crate) touched: Vec<u64>,
    /// Leaves the write in progress keeps in the cache while it works.
    pub(crate) pinned: Vec<FrameId>,
    /// Pages the version the write in progress builds no longer uses.
    dropped: Vec<u64>,
    /// The nodes of that version that need writing: frame and page.
    pub(crate) dirty: Vec<(FrameId, u64)>,
    /// The value leaves of that version that need writing: page id, and the put
    /// in the batch that holds the entry.
    pub(crate) values: Vec<(u64, u32)>,
    /// That version, until it is published or discarded.
    pending: Option<Option<Root>>,
    /// The nodes the write in progress added for its changes, and those it moved.
    added: usize,
    moved: usize,
}

enum Outcome {
    Unchanged,
    Replaced,
}

/// The result of looking an entry up using cached leaves only.
pub(crate) enum Find {
    /// Found: the value's address and length. The key was copied out.
    Found(*const u8, usize),
    /// Found, and both the key and the value were copied out, the value after the
    /// key: the key's length. A dense leaf stores no value whole when its values
    /// share a prefix.
    Copied(usize),
    NotFound,
    /// The lookup needs this leaf page, which isn't cached.
    Missing(u64),
    /// Found in the value leaf with this page id, which has to be read. The key
    /// wasn't copied.
    Value(u64),
}

impl Tree {
    pub(crate) fn new(pages: Pages, page_size: usize, root: Option<Root>, generation: u64) -> Tree {
        Tree {
            pages,
            root,
            generation,
            page_size,
            lists: Vec::new(),
            bounds: Vec::new(),
            run_ends: Vec::new(),
            value_puts: Vec::new(),
            stack: Vec::new(),
            touched: Vec::new(),
            pinned: Vec::new(),
            dropped: Vec::new(),
            dirty: Vec::new(),
            values: Vec::new(),
            pending: None,
            added: 0,
            moved: 0,
        }
    }

    pub(crate) fn node(&self, frame: FrameId) -> Node<'_> {
        Node::new(self.pages.pool.page(frame))
    }

    /// Address of byte `offset` of `frame`.
    pub(crate) fn addr(&self, frame: FrameId, offset: usize) -> *const u8 {
        self.pages.pool.addr(frame, offset)
    }

    pub(crate) fn current(&self) -> Version {
        Version {
            root: self.root,
            generation: self.generation,
        }
    }

    /// Pins the current version: its internal nodes stay in memory, and its pages
    /// stay allocated, until it is [unpinned](Self::unpin).
    pub(crate) fn pin(&mut self) -> Version {
        if let Some(root) = self.root.filter(|r| r.frame != NO_FRAME) {
            self.pages.pool.retain(root.frame);
        }
        self.pages.space.add_reader(self.generation);
        self.current()
    }

    pub(crate) fn unpin(&mut self, version: Version) {
        if let Some(root) = version.root.filter(|r| r.frame != NO_FRAME) {
            self.release(root.frame);
        }
        self.pages.space.remove_reader(version.generation);
        self.reclaim();
    }

    /// Drops a reference to an internal node's frame. A frame that loses its last
    /// reference is freed, and releases its internal children in turn. A node built
    /// by the write in progress frees its page too: it was never committed.
    pub(crate) fn release(&mut self, frame: FrameId) {
        let mut stack = mem::take(&mut self.stack);
        stack.push(frame);
        while let Some(frame) = stack.pop() {
            let pool = &mut self.pages.pool;
            if !pool.unref(frame) {
                continue;
            }
            let node = Node::new(pool.page(frame));
            if node.level() > 1 {
                stack.extend((0..node.count()).map(|i| node.child(i)));
            }
            let info = *pool.frame(frame);
            debug_assert_eq!(info.holds, Holds::Internal);
            if info.dirty {
                self.pages.space.free(info.page);
            }
            pool.free(frame);
        }
        self.stack = stack;
    }

    /// Frees the pages replaced by versions no reader needs any more, evicting
    /// any copies cached.
    fn reclaim(&mut self) {
        let Pages { pool, cache, space } = &mut self.pages;
        space.reclaim(|page| {
            if let Some(frame) = cache.remove(page) {
                debug_assert_eq!(pool.refs(frame), 0, "reclaimed leaf {page} in use");
                pool.free(frame);
            }
        });
    }

    // Reads.

    /// Descends from `root` through the internal nodes towards `key`, recording the
    /// path in `cur`, and returns the page of the leaf where `key` belongs.
    pub(crate) fn descend(&self, root: Root, key: &[u8], cur: &mut Cursor) -> u64 {
        cur.depth = 0;
        if root.level == 0 {
            return root.page;
        }
        let mut frame = root.frame;
        loop {
            let node = self.node(frame);
            let (i, eq) = node.search(key);
            // The child holding `key`: the last one whose smallest key is <= it.
            let child = if eq { i } else { i.saturating_sub(1) };
            cur.push(frame, child);
            if node.level() == 1 {
                return node.child_page(child);
            }
            frame = node.child(child);
        }
    }

    /// Like [`descend`](Self::descend), to the first leaf.
    pub(crate) fn descend_first(&self, root: Root, cur: &mut Cursor) -> u64 {
        cur.depth = 0;
        if root.level == 0 {
            return root.page;
        }
        let mut frame = root.frame;
        loop {
            let node = self.node(frame);
            cur.push(frame, 0);
            if node.level() == 1 {
                return node.child_page(0);
            }
            frame = node.child(0);
        }
    }

    /// Moves `cur` to the next leaf, or the previous one, and returns its page;
    /// `None`, leaving `cur` as it was, if there is none.
    pub(crate) fn next_leaf(&self, cur: &mut Cursor, forward: bool) -> Option<u64> {
        // Climb to the nearest level that can move that way, and move there.
        let mut depth = cur.depth;
        loop {
            if depth == 0 {
                return None;
            }
            let (frame, i) = cur.path[depth - 1];
            let i = usize::from(i);
            if forward && i + 1 < self.node(frame).count() {
                cur.path[depth - 1].1 = (i + 1) as u16;
                break;
            }
            if !forward && i > 0 {
                cur.path[depth - 1].1 = (i - 1) as u16;
                break;
            }
            depth -= 1;
        }
        // Then descend along the left edge going forward, the right edge going back.
        while depth < cur.depth {
            let (frame, i) = cur.path[depth - 1];
            let child = self.node(frame).child(usize::from(i));
            let at = if forward {
                0
            } else {
                self.node(child).count() - 1
            };
            cur.path[depth] = (child, at as u16);
            depth += 1;
        }
        let (frame, i) = cur.path[cur.depth - 1];
        Some(self.node(frame).child_page(usize::from(i)))
    }

    /// Compares the smallest key of the leaf `cur` is at, below the root, with
    /// `key`.
    pub(crate) fn cmp_leaf_key(&self, cur: &Cursor, key: &[u8]) -> Ordering {
        let (frame, i) = cur.path[cur.depth - 1];
        self.node(frame).cmp_key(usize::from(i), key)
    }

    /// Looks up the entry matching `key` under `cmp` in the version rooted at
    /// `root`, using cached leaves only. Copies the key it finds into `out`.
    pub(crate) fn find(
        &mut self,
        root: Option<Root>,
        key: &[u8],
        cmp: Comparison,
        out: &mut Vec<u8>,
    ) -> Find {
        let Some(root) = root else {
            return Find::NotFound;
        };
        let mut cur = Cursor::new();
        let page = self.descend(root, key, &mut cur);
        let (frame, i, eq, n) = if node::span(page) > 0 {
            // A value leaf holds one entry, whose key is its parent's key for it.
            let ord = self.cmp_leaf_key(&cur, key);
            (NO_FRAME, usize::from(ord.is_lt()), ord.is_eq(), 1)
        } else {
            let Some(frame) = self.pages.cached(page) else {
                return Find::Missing(page);
            };
            let node = self.node(frame);
            let (i, eq) = node.search(key);
            (frame, i, eq, node.count())
        };
        let at = match cmp {
            Comparison::Equal => eq.then_some(i),
            Comparison::Floor if eq => Some(i),
            Comparison::Floor | Comparison::Lower => i.checked_sub(1),
            Comparison::Ceiling => Some(i).filter(|&i| i < n),
            Comparison::Higher => Some(i + usize::from(eq)).filter(|&i| i < n),
        };
        let (page, frame, at) = match at {
            Some(at) => (page, frame, at),
            None if cmp == Comparison::Equal => return Find::NotFound,
            // The match, if any, is in a neighboring leaf.
            None => {
                let forward = matches!(cmp, Comparison::Ceiling | Comparison::Higher);
                let Some(page) = self.next_leaf(&mut cur, forward) else {
                    return Find::NotFound;
                };
                if node::span(page) > 0 {
                    return Find::Value(page);
                }
                let Some(frame) = self.pages.cached(page) else {
                    return Find::Missing(page);
                };
                let at = if forward {
                    0
                } else {
                    self.node(frame).count() - 1
                };
                (page, frame, at)
            }
        };
        if frame == NO_FRAME {
            return Find::Value(page);
        }
        let node = self.node(frame);
        node.key_into(at, out);
        match node.whole_value(at) {
            Some((offset, len)) => Find::Found(self.addr(frame, offset), len),
            None => {
                let key_len = out.len();
                node.append_value(at, out);
                Find::Copied(key_len)
            }
        }
    }

    // Writes.

    /// Lists in `touched` the leaf pages applying `batch` changes, which the write
    /// must load before [`apply`](Self::apply). Leaves that ranges remove whole
    /// aren't listed: the write drops them without reading them.
    pub(crate) fn plan(&mut self, batch: &WriteBatch) {
        self.touched.clear();
        let (ops, ranges) = (batch.points(), batch.ranges());
        if ops.is_empty() && ranges.is_empty() {
            return;
        }
        match self.root {
            None => {}
            Some(root) if root.level == 0 => self.touched.push(root.page),
            Some(root) => {
                let level = usize::from(root.level);
                self.ensure_level(level);
                self.plan_internal(root.frame, level, ops, ranges, None, batch);
            }
        }
    }

    fn plan_internal(
        &mut self,
        frame: FrameId,
        level: usize,
        mut ops: &[u32],
        mut ranges: &[(u32, u32)],
        hi: Option<&[u8]>,
        batch: &WriteBatch,
    ) {
        // SAFETY: internal nodes of the current version stay put until it changes.
        let node = unsafe { Node::new(self.pages.pool.page_unbounded(frame)) };
        let mut key = mem::take(&mut self.bounds[2 * level]);
        let mut next = mem::take(&mut self.bounds[2 * level + 1]);
        node.key_into(0, &mut key);
        for i in 0..node.count() {
            let child_hi = if i + 1 < node.count() {
                node.key_into(i + 1, &mut next);
                Some(&next[..])
            } else {
                hi
            };
            let (child_ops, child_ranges, covered) =
                split(batch, &mut ops, &mut ranges, &key, child_hi);
            if !covered && !(child_ops.is_empty() && child_ranges.is_empty()) {
                if level == 1 {
                    // A write never reads a value leaf.
                    if node::span(node.child_page(i)) == 0 {
                        self.touched.push(node.child_page(i));
                    }
                } else {
                    self.plan_internal(
                        node.child(i),
                        level - 1,
                        child_ops,
                        child_ranges,
                        child_hi,
                        batch,
                    );
                }
            }
            mem::swap(&mut key, &mut next);
        }
        self.bounds[2 * level] = key;
        self.bounds[2 * level + 1] = next;
    }

    /// Applies a prepared batch, whose touched leaves must be cached, building the
    /// next version. Returns false if the batch changes nothing. Otherwise the new
    /// version is pending until [`publish`](Self::publish) or
    /// [`discard`](Self::discard).
    pub(crate) fn apply(&mut self, batch: &WriteBatch) -> bool {
        debug_assert!(self.pending.is_none());
        let (ops, ranges) = (batch.points(), batch.ranges());
        if ops.is_empty() && ranges.is_empty() {
            return false;
        }
        self.dropped.clear();
        self.pages.space.restart();
        (self.added, self.moved) = (0, 0);
        let (level, outcome) = match self.root {
            Some(root) => {
                let level = usize::from(root.level);
                self.ensure_level(level);
                (
                    level,
                    self.merge((root.frame, root.page), level, ops, ranges, None, batch),
                )
            }
            None => {
                self.ensure_level(0);
                (0, self.merge_leaf(None, ops, &[], batch))
            }
        };
        if let Outcome::Unchanged = outcome {
            return false;
        }
        if let Some(root) = self.root {
            self.dropped.push(root.page);
        }
        self.pending = Some(self.grow(level));
        true
    }

    /// The meta page that makes the pending version current as `generation`.
    pub(crate) fn pending_meta(&self, generation: u64) -> Meta {
        let root = self.pending.expect("no pending version");
        Meta {
            generation,
            root: root.map_or(0, |r| r.page),
            root_level: root.map_or(0, |r| r.level),
            high_water: self.pages.space.high_water(),
            ..Meta::empty(self.page_size)
        }
    }

    /// Lists in `dirty` the nodes of the pending version that need writing, and
    /// seals them as written by `generation`.
    pub(crate) fn seal_pending(&mut self, generation: u64) {
        self.collect_pending();
        for i in 0..self.dirty.len() {
            let frame = self.dirty[i].0;
            node::seal(self.pages.pool.page_mut(frame), generation);
        }
    }

    /// Lists in `dirty` the nodes the pending version doesn't share with the
    /// current one.
    fn collect_pending(&mut self) {
        self.dirty.clear();
        let Some(Some(root)) = self.pending else {
            return;
        };
        // Leaves this write built are cached until it ends, so an uncached leaf is
        // one it kept.
        if root.level == 0 {
            if let Some(frame) = self.pages.cached(root.page)
                && self.pages.pool.frame(frame).dirty
            {
                self.dirty.push((frame, root.page));
            }
        } else if self.pages.pool.frame(root.frame).dirty {
            self.collect_dirty(root.frame, root.page);
        }
    }

    fn collect_dirty(&mut self, frame: FrameId, page: u64) {
        self.dirty.push((frame, page));
        // SAFETY: the pending version's nodes stay put until it is published.
        let node = unsafe { Node::new(self.pages.pool.page_unbounded(frame)) };
        for i in 0..node.count() {
            let page = node.child_page(i);
            if node.level() == 1 {
                if let Some(leaf) = self.pages.cached(page)
                    && self.pages.pool.frame(leaf).dirty
                {
                    self.dirty.push((leaf, page));
                }
            } else if self.pages.pool.frame(node.child(i)).dirty {
                self.collect_dirty(node.child(i), page);
            }
        }
    }

    /// Makes the pending version current, as `generation`, once committed.
    pub(crate) fn publish(&mut self, generation: u64) {
        self.unpin_touched();
        let root = self.pending.take().expect("no pending version");
        let pool = &mut self.pages.pool;
        for &(frame, _) in &self.dirty {
            let info = pool.frame_mut(frame);
            info.dirty = false;
            if info.holds == Holds::Leaf {
                // Committed leaves are ordinary cached leaves, free to evict.
                pool.unref(frame);
            }
        }
        self.dirty.clear();
        self.values.clear();
        if let Some(old) = mem::replace(&mut self.root, root)
            && old.frame != NO_FRAME
        {
            self.release(old.frame);
        }
        for page in self.dropped.drain(..) {
            let (first, n) = (node::first_page(page), node::pages(page));
            self.pages.space.drop_pages(generation, first, n);
        }
        self.generation = generation;
        self.reclaim();
    }

    /// Throws the pending version away: the write failed before committing it.
    pub(crate) fn discard(&mut self) {
        self.unpin_touched();
        self.dropped.clear();
        self.collect_pending();
        let Some(root) = self.pending.take() else {
            return;
        };
        let Pages { pool, cache, space } = &mut self.pages;
        for &(frame, page) in &self.dirty {
            if pool.frame(frame).holds == Holds::Leaf {
                cache.remove(page);
                pool.unref(frame);
                pool.free(frame);
                space.free(page);
            }
        }
        self.dirty.clear();
        for (page, _) in self.values.drain(..) {
            let first = node::first_page(page);
            for page in first..first + node::span(page) {
                space.free(page);
            }
        }
        if let Some(root) = root.filter(|r| r.frame != NO_FRAME) {
            self.release(root.frame);
        }
    }

    /// Releases the leaves the write in progress pinned.
    pub(crate) fn unpin_touched(&mut self) {
        for frame in self.pinned.drain(..) {
            self.pages.pool.unref(frame);
        }
    }

    /// Makes room in the scratch space for a write reaching `level`.
    fn ensure_level(&mut self, level: usize) {
        assert!(level + 1 < MAX_HEIGHT, "tree too tall");
        if self.lists.len() < level + 2 {
            self.lists.resize_with(level + 2, NodeList::default);
        }
        if self.bounds.len() < 2 * (level + 1) {
            self.bounds.resize_with(2 * (level + 1), Vec::new);
        }
    }

    /// Applies `ops` and `ranges`, which all fall below `hi`, to the subtree at
    /// `node` (frame and page). Its replacement nodes, if any, go to `lists[level]`.
    fn merge(
        &mut self,
        node: (FrameId, u64),
        level: usize,
        ops: &[u32],
        ranges: &[(u32, u32)],
        hi: Option<&[u8]>,
        batch: &WriteBatch,
    ) -> Outcome {
        if level == 0 {
            self.merge_leaf(Some(node.1), ops, ranges, batch)
        } else {
            self.merge_internal(node.0, level, ops, ranges, hi, batch)
        }
    }

    fn merge_internal(
        &mut self,
        frame: FrameId,
        level: usize,
        mut ops: &[u32],
        mut ranges: &[(u32, u32)],
        hi: Option<&[u8]>,
        batch: &WriteBatch,
    ) -> Outcome {
        // SAFETY: published frames are never modified, and this one stays referenced
        // by the current root until the write publishes its new root.
        let node = unsafe { Node::new(self.pages.pool.page_unbounded(frame)) };
        let below = level - 1;
        let mark = self.lists[below].mark();
        let mut key = mem::take(&mut self.bounds[2 * level]);
        let mut next = mem::take(&mut self.bounds[2 * level + 1]);
        let mut changed = false;
        node.key_into(0, &mut key);
        for i in 0..node.count() {
            let child = (node.child(i), node.child_page(i));
            let child_hi = if i + 1 < node.count() {
                node.key_into(i + 1, &mut next);
                Some(&next[..])
            } else {
                hi
            };
            let (child_ops, child_ranges, covered) =
                split(batch, &mut ops, &mut ranges, &key, child_hi);
            if child_ops.is_empty() && child_ranges.is_empty() {
                self.keep(below, &key, child);
            } else if covered {
                // The whole subtree goes, without reading its leaves.
                self.drop_subtree(child, below);
                changed = true;
            } else if node::span(child.1) > 0 {
                if self.merge_value_leaf(child.1, &key, child_ops, child_ranges, batch) {
                    changed = true;
                } else {
                    self.keep(below, &key, child);
                }
            } else {
                match self.merge(child, below, child_ops, child_ranges, child_hi, batch) {
                    Outcome::Unchanged => self.keep(below, &key, child),
                    Outcome::Replaced => {
                        self.dropped.push(child.1);
                        changed = true;
                    }
                }
            }
            mem::swap(&mut key, &mut next);
        }
        self.bounds[2 * level] = key;
        self.bounds[2 * level + 1] = next;
        if !changed {
            self.discard_list(below, mark);
            return Outcome::Unchanged;
        }
        self.move_kept(below, mark.items);
        self.fix_underfull(below, mark.items);
        self.build_parents(below, mark);
        Outcome::Replaced
    }

    /// Applies `ops` and `ranges` to the value leaf on `page`, whose key is `key`,
    /// without reading it. A put or delete of `key`, or a range covering it, drops
    /// it; the puts of other keys go to new leaves beside it. Returns false if
    /// nothing changes: the caller keeps it then.
    fn merge_value_leaf(
        &mut self,
        page: u64,
        key: &[u8],
        ops: &[u32],
        ranges: &[(u32, u32)],
        batch: &WriteBatch,
    ) -> bool {
        let below = ops.partition_point(|&op| batch.key(op) < key);
        let replaced = ops.get(below).is_some_and(|&op| batch.key(op) == key)
            || ranges
                .iter()
                .any(|&(s, e)| batch.range_start(s) <= key && key < batch.range_end(e));
        if !replaced && ops.iter().all(|&op| batch.value(op).is_none()) {
            return false;
        }
        // Only the tree's first leaf takes keys below its own.
        self.merge_leaf(None, &ops[..below], &[], batch);
        if replaced {
            self.dropped.push(page);
        } else {
            self.lists[0].push(key, NO_FRAME, page);
        }
        self.merge_leaf(None, &ops[below..], &[], batch);
        true
    }

    fn merge_leaf(
        &mut self,
        page: Option<u64>,
        ops: &[u32],
        ranges: &[(u32, u32)],
        batch: &WriteBatch,
    ) -> Outcome {
        let node = page.map(|page| {
            let frame = self
                .pages
                .cached(page)
                .expect("the write loads the leaves it changes");
            // SAFETY: the write pinned this leaf, and published frames are never
            // modified.
            unsafe { Node::new(self.pages.pool.page_unbounded(frame)) }
        });
        let old = node.map_or(0, |n| n.count());
        let max_entry = node::max_entry(self.page_size);
        // The rebuilt leaf's entries go after those of its rebuilt siblings, until
        // the parent builds them all.
        let (leaf, run_ends, values) = (
            &mut self.lists[0].entries,
            &mut self.run_ends,
            &mut self.value_puts,
        );
        let base = leaf.len();
        run_ends.clear();
        values.clear();
        let (mut i, mut j, mut r) = (0, 0, 0);
        let mut changed = false;
        let mut in_run = false;
        loop {
            let ord = match (i < old, j < ops.len()) {
                (false, false) => break,
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                (true, true) => node.unwrap().cmp_key(i, batch.key(ops[j])),
            };
            if ord == Ordering::Less {
                let n = node.unwrap();
                while r < ranges.len() && n.cmp_key(i, batch.range_end(ranges[r].1)).is_ge() {
                    r += 1;
                }
                if r < ranges.len() && n.cmp_key(i, batch.range_start(ranges[r].0)).is_ge() {
                    changed = true;
                    i += 1;
                    continue;
                }
                // The entries below the next put or delete, and below the next range,
                // all stay: copy them without comparing each.
                let mut end = old;
                if j < ops.len() {
                    end = end.min(n.search(batch.key(ops[j])).0);
                }
                if r < ranges.len() {
                    end = end.min(n.search(batch.range_start(ranges[r].0)).0);
                }
                debug_assert!(end > i);
                if in_run {
                    run_ends.push(leaf.len() as u32);
                    in_run = false;
                }
                for k in i..end {
                    leaf.push_from(&n, k);
                }
                i = end;
                continue;
            }
            let op = ops[j];
            j += 1;
            if ord == Ordering::Equal {
                // The existing entry is replaced or deleted.
                i += 1;
                changed = true;
            }
            if let Some(value) = batch.value(op) {
                let key = batch.key(op);
                if key.len() + value.len() <= max_entry {
                    leaf.push(key, value);
                    in_run = true;
                } else {
                    // A value leaf of its own, between the leaves of the entries
                    // before and after it.
                    if in_run {
                        run_ends.push(leaf.len() as u32);
                        in_run = false;
                    }
                    values.push((leaf.len() as u32, op));
                }
                changed = true;
            }
        }
        if in_run {
            run_ends.push(leaf.len() as u32);
        }
        if !changed {
            leaf.truncate(base);
            return Outcome::Unchanged;
        }
        let (out, page) = (&mut self.lists[0], self.page_size);
        let items = out.len();
        let mut s = base;
        for &(at, op) in &self.value_puts {
            emit_leaves(out, &self.run_ends, s, at as usize, page);
            let (key, value) = (batch.key(op), batch.value(op).unwrap());
            out.emit_value(key, op, key.len() + value.len());
            s = at as usize;
        }
        emit_leaves(out, &self.run_ends, s, out.entries.len(), page);
        self.added += out.len() - items;
        Outcome::Replaced
    }

    /// Adds an unchanged node to `lists[level]`.
    fn keep(&mut self, level: usize, key: &[u8], (frame, page): (FrameId, u64)) {
        if frame != NO_FRAME {
            self.pages.pool.retain(frame);
        }
        self.lists[level].push(key, frame, page);
    }

    /// Records every page of the subtree at `node` as dropped.
    fn drop_subtree(&mut self, (frame, page): (FrameId, u64), level: usize) {
        self.dropped.push(page);
        if level == 0 {
            return;
        }
        // SAFETY: as in `merge_internal`.
        let node = unsafe { Node::new(self.pages.pool.page_unbounded(frame)) };
        for i in 0..node.count() {
            self.drop_subtree((node.child(i), node.child_page(i)), level - 1);
        }
    }

    /// Drops the nodes added to `lists[level]` since `mark`, all unchanged ones.
    fn discard_list(&mut self, level: usize, mark: Mark) {
        for i in mark.items..self.lists[level].len() {
            let frame = self.lists[level].items[i].frame;
            if frame != NO_FRAME {
                self.release(frame);
            }
        }
        self.lists[level].truncate(mark);
    }

    /// Builds the nodes this write added to `lists[level]` past `mark`, then adds
    /// nodes one level up that take them all over to `lists[level + 1]`.
    fn build_parents(&mut self, level: usize, mark: Mark) {
        for i in mark.items..self.lists[level].len() {
            if self.lists[level].items[i].fresh {
                self.build_node(level, i);
            }
        }
        let (lower, upper) = self.lists.split_at_mut(level + 1);
        let (children, parents) = (&mut lower[level], &mut upper[0]);
        let (a, items) = (parents.entries.len(), parents.len());
        for i in mark.items..children.len() {
            parents.entries.push(children.key(i), children.value(i));
        }
        parents.emit_even(a, parents.entries.len(), self.page_size, level as u8 + 1);
        self.added += parents.len() - items;
        children.truncate(mark);
    }

    /// Builds node `i` of `lists[level]`, which this write added, into a page of its
    /// own. An internal node takes over the references its entries hold.
    fn build_node(&mut self, level: usize, i: usize) {
        let list = &mut self.lists[level];
        let item = list.items[i];
        if let Some(op) = item.value {
            // The commit writes the value leaf straight from the batch.
            let len = node::value_leaf_len(0, item.used as usize);
            let span = len.div_ceil(self.page_size) as u64;
            let page = node::spanned(self.pages.space.alloc_span(span), span);
            self.values.push((page, op));
            list.built(i, NO_FRAME, page);
            return;
        }
        let (a, b) = (item.start as usize, item.end as usize);
        let (frame, page, used) = self
            .pages
            .build(level as u8, &list.entries, a, b, item.life);
        debug_assert_eq!(used, item.used as usize);
        list.built(i, frame, page);
    }

    /// Moves the nodes of `lists[level]` past `start` that the write keeps, but
    /// whose pages the allocator wants freed: they become nodes the write adds,
    /// with the same entries, built next to the others. Their parent is rebuilt
    /// anyway, so a move costs a page write and nothing else: leaves move only if
    /// they are cached. A write moves at most as many nodes as it adds.
    fn move_kept(&mut self, level: usize, start: usize) {
        for k in start..self.lists[level].len() {
            let item = self.lists[level].items[k];
            if item.fresh
                || item.is_value_leaf()
                || self.moved >= self.added
                || !self.pages.space.should_move(item.page)
            {
                continue;
            }
            let frame = if level > 0 {
                item.frame
            } else if let Some(frame) = self.pages.cached(item.page) {
                frame
            } else {
                continue;
            };
            let list = &mut self.lists[level];
            let a = list.entries.len();
            let node = Node::new(self.pages.pool.page(frame));
            for i in 0..node.count() {
                list.entries.push_from(&node, i);
                if level > 1 {
                    // The new node becomes another parent of each internal child.
                    self.pages.pool.retain(node.child(i));
                }
            }
            let b = list.entries.len();
            list.restage(k, a, b, node::size(&list.entries, a, b, level as u8));
            self.dropped.push(item.page);
            if level > 0 {
                self.release(frame);
            }
            self.moved += 1;
        }
    }

    /// Merges each node this write added that is under a quarter full into a
    /// neighbor, among the nodes of `lists[level]` past `start`, if the two fit in a
    /// page and the neighbor is in memory. The nodes this write added aren't built
    /// yet, so no build goes to waste.
    fn fix_underfull(&mut self, level: usize, start: usize) {
        let mut k = start;
        'next: while k < self.lists[level].len() {
            let len = self.lists[level].len();
            let item = self.lists[level].items[k];
            if item.fresh
                && !item.is_value_leaf()
                && (item.used as usize) < self.page_size / 4
                && len - start > 1
            {
                let left = k.checked_sub(1).filter(|&a| a >= start);
                let right = (k + 1 < len).then_some(k);
                for a in left.into_iter().chain(right) {
                    if self.merge_pair(level, a) {
                        k = a;
                        continue 'next;
                    }
                }
            }
            k += 1;
        }
    }

    /// Replaces nodes `a` and `a + 1` of `lists[level]` with a node holding the
    /// entries of both, if they fit and are in memory.
    fn merge_pair(&mut self, level: usize, a: usize) -> bool {
        let list = &mut self.lists[level];
        let (x, y) = (list.items[a], list.items[a + 1]);
        if x.is_value_leaf() || y.is_value_leaf() {
            return false;
        }
        // Find the frames of the unchanged nodes. A merged node spends at least the
        // bytes both nodes spend on their entries, so skip a pair that can't fit
        // before copying them.
        let mut frames = [NO_FRAME; 2];
        let mut least = node::HEADER;
        for (item, frame) in [x, y].iter().zip(&mut frames) {
            if item.fresh {
                let (start, used) = (item.start as usize, item.used as usize);
                least += node::entry_bytes_at_least(&list.entries, start, used);
                continue;
            }
            *frame = if level > 0 {
                item.frame
            } else if let Some(frame) = self.pages.cached(item.page) {
                frame
            } else {
                return false;
            };
            least += Node::new(self.pages.pool.page(*frame)).entry_bytes();
        }
        if least > self.page_size {
            return false;
        }
        let len = list.entries.len();
        let (start, end) = if x.fresh && y.fresh && x.end == y.start {
            // Both are entries next to each other already.
            (x.start as usize, y.end as usize)
        } else {
            for (item, frame) in [x, y].into_iter().zip(frames) {
                if item.fresh {
                    list.entries
                        .extend_from_within(item.start as usize, item.end as usize);
                    continue;
                }
                let node = Node::new(self.pages.pool.page(frame));
                for i in 0..node.count() {
                    list.entries.push_from(&node, i);
                }
            }
            (len, list.entries.len())
        };
        let used = node::size(&list.entries, start, end, level as u8);
        debug_assert!(used >= least);
        if used > self.page_size {
            list.entries.truncate(len);
            return false;
        }
        list.replace_pair(a, start, end, used);
        // The new version won't use the unchanged ones, which keep their pages until
        // older readers are gone. The new node becomes another parent of each of
        // their internal children.
        for item in [x, y].into_iter().filter(|item| !item.fresh) {
            if level > 1 {
                let node = Node::new(self.pages.pool.page(item.frame));
                for i in 0..node.count() {
                    self.pages.pool.retain(node.child(i));
                }
            }
            self.dropped.push(item.page);
            if item.frame != NO_FRAME {
                self.release(item.frame);
            }
        }
        true
    }

    /// Stacks levels on the replacement nodes for the root, in `lists[level]`,
    /// until one node remains, and returns it as the new root.
    fn grow(&mut self, mut level: usize) -> Option<Root> {
        loop {
            if self.lists[level].len() > 1 {
                self.fix_underfull(level, 0);
            }
            match self.lists[level].len() {
                0 => return None,
                // A value leaf always has a parent.
                1 if level > 0 || !self.lists[0].items[0].is_value_leaf() => break,
                _ => {
                    self.ensure_level(level + 1);
                    self.build_parents(level, Mark::default());
                    level += 1;
                }
            }
        }
        let item = self.lists[level].items[0];
        let only_child = (item.fresh && level > 0 && item.end - item.start == 1)
            .then(|| self.lists[level].entries.value(item.start as usize))
            .filter(|child| node::span(node::child_page(child)) == 0);
        let mut root = if let Some(child) = only_child {
            // A root with a single child is redundant: the child becomes the root
            // instead, with the entry's reference to it.
            Root {
                page: node::child_page(child),
                frame: node::child_frame(child),
                level: level as u8 - 1,
            }
        } else {
            if item.fresh {
                self.build_node(level, 0);
            }
            // The list's reference becomes the root's.
            let item = self.lists[level].items[0];
            Root {
                page: item.page,
                frame: item.frame,
                level: level as u8,
            }
        };
        self.lists[level].truncate(Mark::default());
        // A root with a single child is redundant.
        while root.level > 0 {
            let node = self.node(root.frame);
            if node.count() > 1 || node::span(node.child_page(0)) > 0 {
                break;
            }
            let child = Root {
                page: node.child_page(0),
                frame: node.child(0),
                level: root.level - 1,
            };
            if child.frame != NO_FRAME {
                self.pages.pool.retain(child.frame);
            }
            if !self.pages.pool.frame(root.frame).dirty {
                self.dropped.push(root.page);
            }
            self.release(root.frame);
            root = child;
        }
        Some(root)
    }
}

/// Adds leaves holding entries `a..b` of `out` to it, on pages of `page` bytes: one
/// if they fit.
fn emit_leaves(out: &mut NodeList, run_ends: &[u32], a: usize, b: usize, page: usize) {
    if a == b {
        return;
    }
    let used = node::size(&out.entries, a, b, 0);
    if used <= page {
        out.emit(a, b, used, Life::Short);
        return;
    }
    // Split after each run of puts that fills at least a quarter page, as when
    // appending to ledgers: pack the run full, and leave the entries after it to
    // other leaves, so that the next appends don't rewrite them. Smaller runs,
    // like scattered inserts, aren't worth leaves of their own.
    let mut s = a;
    for &end in run_ends {
        let end = end as usize;
        if end <= a || end > b || node::size(&out.entries, s, end, 0) < page / 4 {
            continue;
        }
        while s < end {
            let (e, used) = node::fit(&out.entries, s, end, page, 0);
            // The run's full leaves are likely to stay; appends go on in its last.
            let life = if e < end { Life::Long } else { Life::Short };
            out.emit(s, e, used, life);
            s = e;
        }
    }
    out.emit_even(s, b, page, 0);
}

/// Takes the operations and ranges falling in one child, whose keys are
/// `key..hi`, off the front of `ops` and `ranges`, which are sorted and fall at or
/// after `key`. Also returns whether the ranges remove the child whole.
fn split<'o, 'r>(
    batch: &WriteBatch,
    ops: &mut &'o [u32],
    ranges: &mut &'r [(u32, u32)],
    key: &[u8],
    hi: Option<&[u8]>,
) -> (&'o [u32], &'r [(u32, u32)], bool) {
    let n = hi.map_or(ops.len(), |hi| ops.partition_point(|&o| batch.key(o) < hi));
    let (child_ops, rest) = ops.split_at(n);
    *ops = rest;
    // Ranges ending at or before this child's smallest key can't touch it, or any
    // later child.
    *ranges = &ranges[ranges.partition_point(|&(_, e)| batch.range_end(e) <= key)..];
    let n = hi.map_or(ranges.len(), |hi| {
        ranges.partition_point(|&(s, _)| batch.range_start(s) < hi)
    });
    let child_ranges = &ranges[..n];
    // Only the first range can contain `key`; the child goes if it spans it all.
    let covered = child_ops.is_empty()
        && match (child_ranges.first(), hi) {
            (Some(&(s, e)), Some(hi)) => batch.range_start(s) <= key && batch.range_end(e) >= hi,
            _ => false,
        };
    (child_ops, child_ranges, covered)
}

/// Nodes replacing one node's children, in key order. Each internal node holds a
/// reference; each leaf is found through the cache.
///
/// The nodes this write adds aren't built at first: they are entries, kept here,
/// until no merge can change them any more.
#[derive(Default)]
struct NodeList {
    /// Each node's smallest key, followed by a child reference to it.
    bytes: Vec<u8>,
    items: Vec<Item>,
    /// The entries of the nodes this write adds.
    entries: EntryBuf,
}

#[derive(Clone, Copy)]
struct Item {
    at: u32,
    key_len: u32,
    /// The node's frame and page, which a node this write adds only has once built.
    frame: FrameId,
    page: u64,
    /// Added by this write, rather than kept from the previous version. Until it
    /// is built, such a node is entries `start..end` of the list's `entries`, which
    /// take `used` bytes; `life` is how long its page is likely to stay in use.
    fresh: bool,
    start: u32,
    end: u32,
    used: u32,
    life: Life,
    /// For a value leaf this write adds, the put in the batch that holds its
    /// entry, whose key plus value take `used` bytes.
    value: Option<u32>,
}

impl Item {
    fn is_value_leaf(&self) -> bool {
        self.value.is_some() || node::span(self.page) > 0
    }
}

#[derive(Clone, Copy, Default)]
struct Mark {
    items: usize,
    bytes: usize,
    entries: usize,
}

impl NodeList {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn mark(&self) -> Mark {
        Mark {
            items: self.items.len(),
            bytes: self.bytes.len(),
            entries: self.entries.len(),
        }
    }

    fn truncate(&mut self, mark: Mark) {
        self.items.truncate(mark.items);
        self.bytes.truncate(mark.bytes);
        self.entries.truncate(mark.entries);
    }

    /// Adds a node kept from the previous version.
    fn push(&mut self, key: &[u8], frame: FrameId, page: u64) {
        let (at, key_len) = append_ref(&mut self.bytes, key, frame, page);
        self.items.push(Item {
            at,
            key_len,
            frame,
            page,
            fresh: false,
            start: 0,
            end: 0,
            used: 0,
            life: Life::Short,
            value: None,
        });
    }

    /// Adds a node holding `entries[a..b]`, which take `used` bytes, to be built
    /// on a page with `life`.
    fn emit(&mut self, a: usize, b: usize, used: usize, life: Life) {
        // The child reference is filled in once the node is built.
        let (at, key_len) = append_ref(&mut self.bytes, self.entries.key(a), NO_FRAME, 0);
        self.items.push(Item {
            at,
            key_len,
            frame: NO_FRAME,
            page: 0,
            fresh: true,
            start: a as u32,
            end: b as u32,
            used: used as u32,
            life,
            value: None,
        });
    }

    /// Adds a value leaf for the put `op`, whose key is `key`, and whose key plus
    /// value take `len` bytes.
    fn emit_value(&mut self, key: &[u8], op: u32, len: usize) {
        let (at, key_len) = append_ref(&mut self.bytes, key, NO_FRAME, 0);
        self.items.push(Item {
            at,
            key_len,
            frame: NO_FRAME,
            page: 0,
            fresh: true,
            start: 0,
            end: 0,
            used: len as u32,
            life: Life::Long,
            value: Some(op),
        });
    }

    /// Adds nodes at `level` holding `entries[a..b]`: as few as fit in pages of
    /// `page_size` bytes, of similar sizes.
    fn emit_even(&mut self, a: usize, b: usize, page_size: usize, level: u8) {
        let mut s = a;
        while s < b {
            let total = node::size(&self.entries, s, b, level);
            if total <= page_size {
                self.emit(s, b, total, Life::Short);
                return;
            }
            let (e, used) = node::fit(
                &self.entries,
                s,
                b,
                total / total.div_ceil(page_size),
                level,
            );
            self.emit(s, e, used, Life::Short);
            s = e;
        }
    }

    /// Turns node `i`, kept from the previous version, into one this write adds,
    /// holding `entries[a..b]`, which take `used` bytes.
    fn restage(&mut self, i: usize, a: usize, b: usize, used: usize) {
        let item = &mut self.items[i];
        (item.frame, item.page, item.fresh) = (NO_FRAME, 0, true);
        (item.start, item.end, item.used) = (a as u32, b as u32, used as u32);
        item.life = Life::Short;
        let at = (item.at + item.key_len) as usize;
        self.bytes[at..at + CHILD_REF].copy_from_slice(&node::child_ref(NO_FRAME, 0));
    }

    /// Records that node `i`, which this write added, was built in `frame` and
    /// `page`.
    fn built(&mut self, i: usize, frame: FrameId, page: u64) {
        let item = &mut self.items[i];
        (item.frame, item.page) = (frame, page);
        let at = (item.at + item.key_len) as usize;
        self.bytes[at..at + CHILD_REF].copy_from_slice(&node::child_ref(frame, page));
    }

    /// Replaces nodes `i` and `i + 1` with a node holding `entries[a..b]`, which
    /// take `used` bytes.
    fn replace_pair(&mut self, i: usize, a: usize, b: usize, used: usize) {
        self.emit(a, b, used, Life::Short);
        self.items[i] = self.items.pop().unwrap();
        self.items.remove(i + 1);
    }
}

/// Appends `key` and a child reference to `bytes`, and returns where the key
/// starts and its length.
fn append_ref(bytes: &mut Vec<u8>, key: &[u8], frame: FrameId, page: u64) -> (u32, u32) {
    let at = bytes.len() as u32;
    bytes.extend_from_slice(key);
    bytes.extend_from_slice(&node::child_ref(frame, page));
    (at, key.len() as u32)
}

impl Entries for NodeList {
    fn key(&self, i: usize) -> &[u8] {
        let item = &self.items[i];
        &self.bytes[item.at as usize..][..item.key_len as usize]
    }

    fn value(&self, i: usize) -> &[u8] {
        let item = &self.items[i];
        &self.bytes[(item.at + item.key_len) as usize..][..CHILD_REF]
    }
}

/// A reader's path through the internal nodes: the node and child index at each
/// level, root first.
#[derive(Clone)]
pub(crate) struct Cursor {
    path: [(FrameId, u16); MAX_HEIGHT],
    depth: usize,
}

impl Cursor {
    pub(crate) fn new() -> Cursor {
        Cursor {
            path: [(0, 0); MAX_HEIGHT],
            depth: 0,
        }
    }

    fn push(&mut self, frame: FrameId, i: usize) {
        self.path[self.depth] = (frame, i as u16);
        self.depth += 1;
    }
}

#[cfg(test)]
impl Tree {
    /// Checks the tree's structure and accounting. Panics on a violation. Only
    /// valid while no reader is pinned and no write is in progress.
    ///
    /// Leaves are checked when cached; contents are left to the callers' scans.
    pub(crate) fn check(&mut self) {
        let mut stats = Stats::default();
        if let Some(root) = self.root {
            self.check_node((root.frame, root.page), root.level, None, None, &mut stats);
        }
        let Pages { pool, cache, space } = &self.pages;
        assert_eq!(space.dropped().count(), 0, "replaced pages not reclaimed");
        assert_eq!(
            space.used_count() as u64,
            crate::meta::META_PAGES + stats.pages,
            "pages leaked"
        );
        // Frames that cancelled reads left behind wait for the clock hand.
        let cancelled = (0..pool.len() as FrameId)
            .filter(|&f| pool.frame(f).holds == Holds::Loading && pool.refs(f) == 0)
            .count();
        assert_eq!(
            pool.in_use(),
            stats.internal + cache.len() + cancelled,
            "frames leaked"
        );
        for (page, frame) in cache.entries() {
            let info = pool.frame(frame);
            assert_eq!(
                (info.holds, info.page, info.dirty),
                (Holds::Leaf, page, false)
            );
            assert_eq!(pool.refs(frame), 0, "leaf {page} still pinned");
            assert!(space.is_used(page), "cached leaf {page} on a free page");
        }
        assert!(self.pinned.is_empty() && self.dirty.is_empty() && self.pending.is_none());
        assert!(self.values.is_empty());
    }

    /// Checks the subtree at `node`, whose keys fall in `lo..hi`, and returns its
    /// smallest key if known.
    fn check_node(
        &mut self,
        (frame, page): (FrameId, u64),
        level: u8,
        lo: Option<&[u8]>,
        hi: Option<&[u8]>,
        stats: &mut Stats,
    ) -> Option<Vec<u8>> {
        stats.pages += node::pages(page);
        let first = node::first_page(page);
        for page in first..first + node::pages(page) {
            assert!(
                self.pages.space.is_used(page),
                "page {page} in the tree but free"
            );
        }
        let frame = if level == 0 {
            assert_eq!(frame, NO_FRAME);
            if node::span(page) > 0 {
                // Its key is its parent's key for it, which the caller checks.
                assert!(lo.is_some(), "a value leaf at the root");
                return None;
            }
            self.pages.cached(page)?
        } else {
            stats.internal += 1;
            let info = *self.pages.pool.frame(frame);
            assert_eq!(
                (info.holds, info.page, info.dirty),
                (Holds::Internal, page, false)
            );
            assert_eq!(
                self.pages.pool.refs(frame),
                1,
                "frame {frame} has extra references"
            );
            frame
        };
        let node = self.node(frame);
        assert_eq!(node.level(), level, "uneven depth");
        assert!(node.generation() <= self.generation, "node from the future");
        let n = node.count();
        assert!(n > 0, "empty node");
        let mut entries = EntryBuf::default();
        for i in 0..n {
            entries.push_from(&node, i);
        }
        assert_eq!(
            node.used(),
            node::size(&entries, 0, n, level),
            "corrupt layout"
        );
        assert!(node.used() <= self.page_size, "node overflows its page");
        for i in 1..n {
            assert!(entries.key(i - 1) < entries.key(i), "keys out of order");
        }
        assert!(
            lo.is_none_or(|lo| entries.key(0) >= lo),
            "key below its subtree"
        );
        assert!(
            hi.is_none_or(|hi| entries.key(n - 1) < hi),
            "key above its subtree"
        );
        if level > 0 {
            for i in 0..n {
                let child_hi = if i + 1 < n {
                    Some(entries.key(i + 1))
                } else {
                    hi
                };
                let child = (
                    node::child_frame(entries.value(i)),
                    node::child_page(entries.value(i)),
                );
                let min = self.check_node(child, level - 1, Some(entries.key(i)), child_hi, stats);
                if let Some(min) = min {
                    assert_eq!(
                        min,
                        entries.key(i),
                        "internal key isn't its child's smallest"
                    );
                }
            }
        }
        Some(entries.key(0).to_vec())
    }

    /// The pages of the current version, sorted.
    pub(crate) fn pages(&self) -> Vec<u64> {
        let mut pages = Vec::new();
        let mut stack = Vec::new();
        if let Some(root) = self.root {
            pages.push(root.page);
            if root.level > 0 {
                stack.push(root.frame);
            }
        }
        while let Some(frame) = stack.pop() {
            let node = self.node(frame);
            for i in 0..node.count() {
                let first = node::first_page(node.child_page(i));
                pages.extend(first..first + node::pages(node.child_page(i)));
                if node.level() > 1 {
                    stack.push(node.child(i));
                }
            }
        }
        pages.sort_unstable();
        pages
    }

    /// Each leaf's page and smallest key, in key order, found from the internal
    /// nodes. The key is empty for a root leaf.
    pub(crate) fn leaves(&self) -> Vec<(u64, Vec<u8>)> {
        let Some(root) = self.root else {
            return Vec::new();
        };
        let mut cur = Cursor::new();
        let mut page = Some(self.descend_first(root, &mut cur));
        let mut leaves = Vec::new();
        while let Some(leaf) = page {
            let mut key = Vec::new();
            if cur.depth > 0 {
                let (frame, i) = cur.path[cur.depth - 1];
                self.node(frame).key_into(usize::from(i), &mut key);
            }
            leaves.push((leaf, key));
            page = self.next_leaf(&mut cur, true);
        }
        leaves
    }

    /// Number of cached leaves, and the bytes they use.
    pub(crate) fn leaf_usage(&self) -> (usize, usize) {
        let (pool, cache) = (&self.pages.pool, &self.pages.cache);
        cache.entries().fold((0, 0), |(n, used), (_, frame)| {
            (n + 1, used + Node::new(pool.page(frame)).used())
        })
    }

    pub(crate) fn page_size(&self) -> usize {
        self.page_size
    }

    /// The nodes the last write added for its changes, and those it moved.
    pub(crate) fn moves(&self) -> (usize, usize) {
        (self.added, self.moved)
    }
}

#[cfg(test)]
#[derive(Default)]
struct Stats {
    pages: u64,
    internal: usize,
}
