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
use std::task::Poll;
use std::time::{Duration, Instant};

use crate::Comparison;
use crate::batch::WriteBatch;
use crate::cache::Cache;
use crate::meta::Meta;
use crate::node::{self, CHILD_REF, Entries, EntryBuf, NO_FRAME, Node, ValueLeaf};
use crate::pool::{FrameId, Holds, Pool};
use crate::space::{Life, Space};

/// The deepest tree supported. Every split leaves internal nodes with at least two
/// children, so real trees stay far shallower.
pub(crate) const MAX_HEIGHT: usize = 32;

/// How long a write normalizes, plans or merges its batch, or loads the leaves it
/// changes, before it pauses, so that it can let the other tasks run. A pause
/// costs a trip through the runtime, about 15 µs on an Apple M5 Pro, so much
/// shorter slices would slow writes down.
const SLICE: Duration = Duration::from_millis(1);

/// The work a write does between looks at the clock: operations its normalizing
/// lists, compares or moves, nodes its planning visits, leaves it pins, and
/// entries its merge copies, merges and builds, and nodes it visits. On an Apple
/// M5 Pro, 4,096 units of merging take about 25 µs of a sorted load, and 180 µs of
/// random updates, whose operations lie scattered in the batch's memory.
const CHECK: usize = 4096;

/// The work planning counts for a node it visits, so that it looks at the clock
/// every 64 visits. A visit searches the operations and ranges left for the node,
/// whose keys lie scattered in the memory of an unsorted batch: on an Apple M5
/// Pro, it takes up to 2 µs of random updates.
const VISIT: usize = 64;

/// The work a write counts for a leaf it pins once it has planned, so that it
/// looks at the clock every 256 leaves: about 20 µs on an Apple M5 Pro, when they
/// are cached.
const PIN: usize = 16;

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
    /// Builds entries `a..b` of `src` into a new node at `level`, on a new page,
    /// sealed as written by `generation`. The node is dirty until its write
    /// publishes. A leaf goes into the cache, holding a reference until then; an
    /// internal node's one reference goes to the caller. Returns the node's frame
    /// ([`NO_FRAME`] for a leaf), page, and bytes used.
    fn build(
        &mut self,
        level: u8,
        src: &impl Entries,
        a: usize,
        b: usize,
        life: Life,
        generation: u64,
    ) -> (FrameId, u64, usize) {
        let page = self.space.alloc(life);
        let holds = if level == 0 {
            Holds::Leaf
        } else {
            Holds::Internal
        };
        let frame = self.cache.alloc(&mut self.pool, holds, page);
        let used = node::build(self.pool.page_mut(frame), level, src, a, b);
        node::seal(self.pool.page_mut(frame), generation);
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
    /// The write in progress, which pauses after each slice: how far it got
    /// planning or merging, what it keeps of each internal node on its path and of
    /// the leaf it rebuilds, and when it pauses next.
    phase: Phase,
    visits: [Visit; MAX_HEIGHT],
    leaf: LeafMerge,
    budget: Budget,
}

enum Outcome {
    Unchanged,
    Replaced,
    /// The merge did a slice of its work, and paused.
    Paused,
}

/// When the write in progress pauses its work on the batch: once it has run for
/// [`SLICE`], or, in tests, once it has done a set amount of work.
pub(crate) struct Budget {
    /// The work done since the write last paused, and the work after which it next
    /// looks at the clock, or pauses in tests.
    pub(crate) work: usize,
    check: usize,
    /// When it last started or resumed.
    start: Instant,
    /// The work it does before pausing, in tests.
    fixed: Option<usize>,
}

impl Budget {
    fn new() -> Budget {
        Budget {
            work: 0,
            check: CHECK,
            start: Instant::now(),
            fixed: None,
        }
    }

    /// A budget that pauses after `work` units of work.
    #[cfg(test)]
    pub(crate) fn fixed(work: usize) -> Budget {
        Budget {
            fixed: Some(work),
            ..Budget::new()
        }
    }

    /// Starts a slice.
    pub(crate) fn start(&mut self) {
        self.work = 0;
        self.check = self.fixed.unwrap_or(CHECK);
        self.start = Instant::now();
    }

    /// Whether the slice is over.
    pub(crate) fn spent(&mut self) -> bool {
        if self.work < self.check {
            return false;
        }
        if self.fixed.is_some() || self.start.elapsed() >= SLICE {
            return true;
        }
        self.check = self.work + CHECK;
        false
    }
}

/// How far the write in progress got.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    /// Normalizing the batch, which keeps how far it got.
    Normalizing,
    /// Listing the leaves the batch changes, from the root down.
    Planning,
    /// Merging the batch into the tree, from the root down.
    Merging,
    /// Stacking levels on the nodes that replace the root: at `level`, whose
    /// nodes under a quarter full are merged once `fixed`, then built from item
    /// `k` on.
    Growing {
        level: usize,
        fixed: bool,
        k: usize,
    },
}

/// What a merge keeps of an internal node it merges while it pauses. Planning,
/// which ends before the merge starts, keeps the same of a node it plans, up to
/// `inside`. The node is found again from its parent.
#[derive(Clone, Copy, Default)]
struct Visit {
    /// The merge, or planning, is at this level.
    active: bool,
    /// The operations and ranges that fall in the children it hasn't reached.
    ops: Span,
    ranges: Span,
    /// The child it is at, and whether that child's merge paused, with how far it
    /// got if the child is a value leaf.
    child: usize,
    inside: bool,
    value: ValueMerge,
    changed: bool,
    /// Where the nodes replacing its children start in the list below, and, once
    /// past its children, how far it got finishing them: at `stage`, from item
    /// `k` on.
    mark: Mark,
    stage: Stage,
    k: usize,
}

/// How far a merge got applying operations to a value leaf: whether they replace
/// it, and the operations below its key and above it, which go to new leaves
/// beside it, `second` once those below have.
#[derive(Clone, Copy, Default)]
struct ValueMerge {
    started: bool,
    replaced: bool,
    below: Span,
    above: Span,
    second: bool,
}

/// The steps that finish the nodes replacing an internal node's children.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum Stage {
    #[default]
    Move,
    Fix,
    Build,
}

/// What a merge keeps of the leaf it rebuilds while it pauses.
#[derive(Clone, Copy, Default)]
struct LeafMerge {
    /// The merge is at this leaf.
    active: bool,
    /// The leaf's frame, which the write pinned, or [`NO_FRAME`] in an empty tree.
    frame: FrameId,
    ops: Span,
    ranges: Span,
    /// Where the rebuilt entries start in `lists[0]`, and how far merging them
    /// got: the leaf's entry `i`, operation `j` and range `r`.
    base: usize,
    i: usize,
    j: usize,
    r: usize,
    changed: bool,
    in_run: bool,
    /// Once merged, the nodes they go to, from item `items` of `lists[0]` on: the
    /// value leaves before value put `vp` are added, and the entries from `a` up to
    /// it go to leaves. When those need more than one (`splitting`), they split
    /// after run `run`, from entry `s` on, while `packing` the run's entries into
    /// full nodes.
    merged: bool,
    items: usize,
    vp: usize,
    a: usize,
    splitting: bool,
    run: usize,
    s: usize,
    packing: bool,
}

/// Operations or ranges of a prepared batch: indexes `start..end` into its
/// points or ranges.
#[derive(Clone, Copy, Default)]
struct Span {
    start: usize,
    end: usize,
}

impl Span {
    fn all<T>(items: &[T]) -> Span {
        Span {
            start: 0,
            end: items.len(),
        }
    }

    fn of<T>(self, items: &[T]) -> &[T] {
        &items[self.start..self.end]
    }

    /// Its last `len` items.
    fn last(self, len: usize) -> Span {
        Span {
            start: self.end - len,
            end: self.end,
        }
    }
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
    /// Found in the value leaf with this page id, too large for the cache to keep,
    /// which has to be read. The key wasn't copied.
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
            phase: Phase::Idle,
            visits: [Visit::default(); MAX_HEIGHT],
            leaf: LeafMerge::default(),
            budget: Budget::new(),
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
    /// any copies cached, value leaves' included.
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
                    return self.find_value(page, out);
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
            return self.find_value(page, out);
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

    /// The lookup's match: the entry of the value leaf with page id `page`. Copies
    /// its key into `out` if the leaf is cached.
    fn find_value(&mut self, page: u64, out: &mut Vec<u8>) -> Find {
        if !self.pages.cache.caches(node::span(page)) {
            return Find::Value(page);
        }
        let Some(frame) = self.pages.cached(page) else {
            return Find::Missing(page);
        };
        let leaf = ValueLeaf::new(self.pages.pool.page(frame));
        out.clear();
        out.extend_from_slice(leaf.key());
        Find::Found(leaf.value().as_ptr(), leaf.value().len())
    }

    // Writes.

    /// Normalizes `batch`, then lists in `touched` the leaf pages applying it
    /// changes, which the write must load before [`apply`](Self::apply). Leaves
    /// that ranges remove whole aren't listed: the write drops them without reading
    /// them. Like `apply`, it pauses after each slice of work, returning `Pending`:
    /// calling this again with the same batch goes on where it left off, unless
    /// [`discard`](Self::discard) threw away how far it got.
    pub(crate) fn plan(&mut self, batch: &mut WriteBatch) -> Poll<()> {
        self.budget.start();
        loop {
            match self.phase {
                Phase::Idle => {
                    batch.restart();
                    self.touched.clear();
                    self.phase = Phase::Normalizing;
                }
                Phase::Normalizing => {
                    if !batch.prepare(&mut self.budget) {
                        return Poll::Pending;
                    }
                    self.ensure_level(self.root.map_or(0, |r| usize::from(r.level)));
                    self.phase = Phase::Planning;
                }
                Phase::Planning => {
                    let (ops, ranges) = (batch.points(), batch.ranges());
                    if let Some(root) = self.root
                        && !(ops.is_empty() && ranges.is_empty())
                    {
                        if root.level == 0 {
                            self.touched.push(root.page);
                        } else {
                            let (level, ops, ranges) =
                                (usize::from(root.level), Span::all(ops), Span::all(ranges));
                            if !self.plan_internal(root.frame, level, ops, ranges, None, batch) {
                                return Poll::Pending;
                            }
                        }
                    }
                    self.phase = Phase::Idle;
                    return Poll::Ready(());
                }
                Phase::Merging | Phase::Growing { .. } => unreachable!("planning during a merge"),
            }
        }
    }

    /// Lists the leaves under the internal node on `frame`, at `level`, that `ops`
    /// and `ranges`, which all fall below `hi`, change. Returns false if it paused:
    /// calling this again with the same node and `hi` goes on where it left off,
    /// with the operations and ranges it kept, as a merge does.
    fn plan_internal(
        &mut self,
        frame: FrameId,
        level: usize,
        ops: Span,
        ranges: Span,
        hi: Option<&[u8]>,
        batch: &WriteBatch,
    ) -> bool {
        // SAFETY: internal nodes of the current version stay put until it changes.
        let node = unsafe { Node::new(self.pages.pool.page_unbounded(frame)) };
        let below = level - 1;
        let mut v = self.visits[level];
        if !v.active {
            v = Visit {
                active: true,
                ops,
                ranges,
                ..Visit::default()
            };
            node.key_into(0, &mut self.bounds[2 * level]);
        }
        let mut key = mem::take(&mut self.bounds[2 * level]);
        let mut next = mem::take(&mut self.bounds[2 * level + 1]);
        // The operations and ranges for the children it hasn't reached, as in
        // `merge_internal`.
        let mut ops = v.ops.of(batch.points());
        let mut ranges = v.ranges.of(batch.ranges());
        while v.child < node.count() {
            let i = v.child;
            let child_hi = if i + 1 < node.count() {
                node.key_into(i + 1, &mut next);
                Some(&next[..])
            } else {
                hi
            };
            // Whether the child is planned.
            let planned = if v.inside {
                // The child's planning paused: it goes on with the operations it kept.
                let kept = Span::default();
                self.plan_internal(node.child(i), below, kept, kept, child_hi, batch)
            } else if self.budget.spent() {
                false
            } else {
                self.budget.work += VISIT;
                let (child_ops, child_ranges, covered) =
                    split(batch, &mut ops, &mut ranges, &key, child_hi);
                if covered || (child_ops.is_empty() && child_ranges.is_empty()) {
                    true
                } else if level == 1 {
                    // A write never reads a value leaf.
                    if node::span(node.child_page(i)) == 0 {
                        self.touched.push(node.child_page(i));
                    }
                    true
                } else {
                    (v.ops, v.ranges) = (v.ops.last(ops.len()), v.ranges.last(ranges.len()));
                    let child_ops = Span {
                        start: v.ops.start - child_ops.len(),
                        end: v.ops.start,
                    };
                    let child_ranges = Span {
                        start: v.ranges.start,
                        end: v.ranges.start + child_ranges.len(),
                    };
                    v.inside = true;
                    let child = node.child(i);
                    self.plan_internal(child, below, child_ops, child_ranges, child_hi, batch)
                }
            };
            if !planned {
                break;
            }
            v.inside = false;
            v.child += 1;
            mem::swap(&mut key, &mut next);
        }
        self.bounds[2 * level] = key;
        self.bounds[2 * level + 1] = next;
        if v.child < node.count() {
            (v.ops, v.ranges) = (v.ops.last(ops.len()), v.ranges.last(ranges.len()));
            self.visits[level] = v;
            return false;
        }
        self.visits[level].active = false;
        true
    }

    /// Applies a prepared batch, whose touched leaves must be cached, building the
    /// next version. The merge pauses after each slice of work, returning
    /// `Pending`, so that the caller can let other tasks run: calling this again
    /// with the same batch goes on where it left off. Its first slice is what is
    /// left of the one in which planning, and loading the leaves, ended. Returns
    /// false if the batch changes nothing. Otherwise the new version is pending
    /// until [`publish`](Self::publish) or [`discard`](Self::discard), which also
    /// throws away a merge that paused.
    pub(crate) fn apply(&mut self, batch: &WriteBatch) -> Poll<bool> {
        if self.phase != Phase::Idle {
            self.budget.start();
        }
        loop {
            match self.phase {
                Phase::Idle => {
                    debug_assert!(self.pending.is_none());
                    if batch.points().is_empty() && batch.ranges().is_empty() {
                        return Poll::Ready(false);
                    }
                    self.dropped.clear();
                    self.pages.space.restart();
                    (self.added, self.moved) = (0, 0);
                    self.ensure_level(self.root.map_or(0, |r| usize::from(r.level)));
                    self.phase = Phase::Merging;
                }
                Phase::Merging => {
                    let (ops, ranges) = (Span::all(batch.points()), Span::all(batch.ranges()));
                    let (level, outcome) = match self.root {
                        Some(root) => {
                            let level = usize::from(root.level);
                            let node = (root.frame, root.page);
                            (level, self.merge(node, level, ops, ranges, None, batch))
                        }
                        None => (0, self.merge_leaf(None, ops, Span::default(), batch)),
                    };
                    match outcome {
                        Outcome::Paused => return Poll::Pending,
                        Outcome::Unchanged => {
                            self.phase = Phase::Idle;
                            return Poll::Ready(false);
                        }
                        Outcome::Replaced => {
                            if let Some(root) = self.root {
                                self.dropped.push(root.page);
                            }
                            self.phase = Phase::Growing {
                                level,
                                fixed: false,
                                k: 0,
                            };
                        }
                    }
                }
                Phase::Growing { level, fixed, k } => {
                    let Some(root) = self.grow(level, fixed, k) else {
                        return Poll::Pending;
                    };
                    self.pending = Some(root);
                    self.phase = Phase::Idle;
                    return Poll::Ready(true);
                }
                Phase::Normalizing | Phase::Planning => unreachable!("merging an unplanned batch"),
            }
        }
    }

    /// Counts `n` leaves that the write in progress pinned, once planned, and
    /// returns whether its slice is over. The write then pauses, and
    /// [`resume`](Self::resume)s once the other tasks have run.
    pub(crate) fn loaded(&mut self, n: usize) -> bool {
        self.budget.work += n * PIN;
        self.budget.spent()
    }

    /// Starts the next slice of the write in progress, which paused loading its
    /// leaves.
    pub(crate) fn resume(&mut self) {
        self.budget.start();
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

    /// Lists in `dirty` the nodes of the pending version that need writing: those
    /// it doesn't share with the current one. They are sealed already.
    pub(crate) fn collect_pending(&mut self) {
        self.dirty.clear();
        if let Some(Some(root)) = self.pending {
            self.collect_built(usize::from(root.level), root.frame, root.page);
        }
    }

    /// Lists in `dirty` the node at `level`, on `frame` and `page`, if the write in
    /// progress built it, and the nodes it built below it.
    fn collect_built(&mut self, level: usize, frame: FrameId, page: u64) {
        if level > 0 {
            if self.pages.pool.frame(frame).dirty {
                self.collect_dirty(frame, page);
            }
        } else if let Some(leaf) = self.pages.cached(page)
            && self.pages.pool.frame(leaf).dirty
        {
            // Leaves this write built are cached until it ends, so an uncached leaf
            // is one it kept.
            self.dirty.push((leaf, page));
        }
    }

    fn collect_dirty(&mut self, frame: FrameId, page: u64) {
        self.dirty.push((frame, page));
        // SAFETY: the nodes this write built stay put until it ends.
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
            self.pages.space.drop_node(generation, page);
        }
        self.generation = generation;
        self.reclaim();
    }

    /// Throws away the version the write in progress builds, whether pending or
    /// halfway through its merge: the write failed, or went away, before committing
    /// it.
    pub(crate) fn discard(&mut self) {
        self.abort_merge();
        self.unpin_touched();
        self.dropped.clear();
        self.collect_pending();
        let Some(root) = self.pending.take() else {
            return;
        };
        self.free_built_leaves();
        if let Some(root) = root.filter(|r| r.frame != NO_FRAME) {
            self.release(root.frame);
        }
    }

    /// Throws away a merge that paused, if any. The nodes it built are in its lists,
    /// or below the nodes there, and so are its references to internal nodes.
    fn abort_merge(&mut self) {
        if self.phase == Phase::Idle {
            return;
        }
        // Collect the nodes it built.
        self.dirty.clear();
        for level in 0..self.lists.len() {
            for k in 0..self.lists[level].len() {
                let item = self.lists[level].items[k];
                if item.is_built() {
                    self.collect_built(level, item.frame, item.page);
                } else if item.fresh && level > 0 {
                    for e in item.start as usize..item.end as usize {
                        let child = self.lists[level].entries.value(e);
                        let (frame, page) = (node::child_frame(child), node::child_page(child));
                        self.collect_built(level - 1, frame, page);
                    }
                }
            }
        }
        // Free the leaves among them. Letting go of the references frees the
        // internal nodes.
        self.free_built_leaves();
        for level in 1..self.lists.len() {
            for k in 0..self.lists[level].len() {
                let item = self.lists[level].items[k];
                if !item.fresh || item.is_built() {
                    self.release(item.frame);
                } else if level > 1 {
                    for e in item.start as usize..item.end as usize {
                        self.release(node::child_frame(self.lists[level].entries.value(e)));
                    }
                }
            }
        }
        for list in &mut self.lists {
            list.truncate(Mark::default());
        }
        self.visits = [Visit::default(); MAX_HEIGHT];
        self.leaf = LeafMerge::default();
        self.phase = Phase::Idle;
    }

    /// Frees the leaves the write in progress built: those listed in `dirty`, and
    /// its value leaves. Empties both lists.
    fn free_built_leaves(&mut self) {
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
    /// If the merge pauses, calling this again with the same `node` and `hi` goes
    /// on where it left off, with the operations and ranges it kept.
    fn merge(
        &mut self,
        node: (FrameId, u64),
        level: usize,
        ops: Span,
        ranges: Span,
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
        ops: Span,
        ranges: Span,
        hi: Option<&[u8]>,
        batch: &WriteBatch,
    ) -> Outcome {
        // SAFETY: published frames are never modified, and this one stays referenced
        // by the current root until the write publishes its new root.
        let node = unsafe { Node::new(self.pages.pool.page_unbounded(frame)) };
        let below = level - 1;
        let mut v = self.visits[level];
        if !v.active {
            let mark = self.lists[below].mark();
            v = Visit {
                active: true,
                ops,
                ranges,
                mark,
                k: mark.items,
                ..Visit::default()
            };
            node.key_into(0, &mut self.bounds[2 * level]);
        }
        let mut key = mem::take(&mut self.bounds[2 * level]);
        let mut next = mem::take(&mut self.bounds[2 * level + 1]);
        // The operations and ranges for the children it hasn't reached, which
        // `split` narrows: what it leaves of each is their spans' last items.
        let mut ops = v.ops.of(batch.points());
        let mut ranges = v.ranges.of(batch.ranges());
        while v.child < node.count() {
            let i = v.child;
            let child = (node.child(i), node.child_page(i));
            let child_hi = if i + 1 < node.count() {
                node.key_into(i + 1, &mut next);
                Some(&next[..])
            } else {
                hi
            };
            // `None` once the child changed with nothing more to do: dropped whole, or
            // a value leaf that stays beside new leaves.
            let outcome = if v.inside {
                // The child's merge paused: it goes on with the operations it kept.
                let kept = Span::default();
                if node::span(child.1) > 0 {
                    self.merge_value_leaf(&mut v, child.1, &key, kept, kept, batch)
                } else {
                    Some(self.merge(child, below, kept, kept, child_hi, batch))
                }
            } else if self.budget.spent() {
                Some(Outcome::Paused)
            } else {
                self.budget.work += 1;
                let (child_ops, child_ranges, covered) =
                    split(batch, &mut ops, &mut ranges, &key, child_hi);
                if child_ops.is_empty() && child_ranges.is_empty() {
                    Some(Outcome::Unchanged)
                } else if covered {
                    // The whole subtree goes, without reading its leaves.
                    self.drop_subtree(child, below);
                    None
                } else {
                    // The child's operations come just before what `split` left,
                    // and its ranges first in what it left.
                    (v.ops, v.ranges) = (v.ops.last(ops.len()), v.ranges.last(ranges.len()));
                    let child_ops = Span {
                        start: v.ops.start - child_ops.len(),
                        end: v.ops.start,
                    };
                    let child_ranges = Span {
                        start: v.ranges.start,
                        end: v.ranges.start + child_ranges.len(),
                    };
                    v.inside = true;
                    if node::span(child.1) > 0 {
                        self.merge_value_leaf(&mut v, child.1, &key, child_ops, child_ranges, batch)
                    } else {
                        Some(self.merge(child, below, child_ops, child_ranges, child_hi, batch))
                    }
                }
            };
            match outcome {
                Some(Outcome::Paused) => break,
                Some(Outcome::Unchanged) => self.keep(below, &key, child),
                Some(Outcome::Replaced) => {
                    self.dropped.push(child.1);
                    v.changed = true;
                }
                None => v.changed = true,
            }
            v.inside = false;
            v.child += 1;
            mem::swap(&mut key, &mut next);
        }
        self.bounds[2 * level] = key;
        self.bounds[2 * level + 1] = next;
        (v.ops, v.ranges) = (v.ops.last(ops.len()), v.ranges.last(ranges.len()));
        // Paused at a child, or finishing them.
        if v.child < node.count() || (v.changed && !self.finish(below, &mut v)) {
            self.visits[level] = v;
            return Outcome::Paused;
        }
        self.visits[level].active = false;
        if !v.changed {
            self.discard_list(below, v.mark);
            return Outcome::Unchanged;
        }
        self.add_parents(below, v.mark);
        Outcome::Replaced
    }

    /// Finishes the nodes past `v.mark` in `lists[level]`, which replace the
    /// children of the node `v` merges: moves the kept nodes that should move,
    /// merges those under a quarter full, then builds them. Returns false if it
    /// paused: `v` keeps how far it got.
    fn finish(&mut self, level: usize, v: &mut Visit) -> bool {
        let start = v.mark.items;
        if v.stage == Stage::Move {
            if !self.move_kept(level, &mut v.k) {
                return false;
            }
            (v.stage, v.k) = (Stage::Fix, start);
        }
        if v.stage == Stage::Fix {
            if !self.fix_underfull(level, start, &mut v.k) {
                return false;
            }
            (v.stage, v.k) = (Stage::Build, start);
        }
        self.build_nodes(level, &mut v.k)
    }

    /// Applies `ops` and `ranges` to the value leaf on `page`, whose key is `key`,
    /// without reading it. A put or delete of `key`, or a range covering it, drops
    /// it; the puts of other keys go to new leaves beside it. Returns `Unchanged`
    /// if nothing changes, and the caller keeps it then, `Replaced` if it is
    /// dropped, and `None` if it stays beside new leaves. If the merge pauses,
    /// calling this again goes on where it left off, as `v` keeps.
    fn merge_value_leaf(
        &mut self,
        v: &mut Visit,
        page: u64,
        key: &[u8],
        ops: Span,
        ranges: Span,
        batch: &WriteBatch,
    ) -> Option<Outcome> {
        let mut value = v.value;
        if !value.started {
            let points = ops.of(batch.points());
            let below = points.partition_point(|&op| batch.key(op) < key);
            let replaced = points.get(below).is_some_and(|&op| batch.key(op) == key)
                || ranges
                    .of(batch.ranges())
                    .iter()
                    .any(|&(s, e)| batch.range_start(s) <= key && key < batch.range_end(e));
            if !replaced && points.iter().all(|&op| batch.value(op).is_none()) {
                return Some(Outcome::Unchanged);
            }
            let split = ops.start + below;
            value = ValueMerge {
                started: true,
                replaced,
                below: Span {
                    start: ops.start,
                    end: split,
                },
                above: Span {
                    start: split,
                    end: ops.end,
                },
                second: false,
            };
        }
        // Only the tree's first leaf takes keys below its own.
        let none = Span::default();
        if !value.second {
            if let Outcome::Paused = self.merge_leaf(None, value.below, none, batch) {
                v.value = value;
                return Some(Outcome::Paused);
            }
            if !value.replaced {
                self.lists[0].push(key, NO_FRAME, page);
            }
            value.second = true;
        }
        if let Outcome::Paused = self.merge_leaf(None, value.above, none, batch) {
            v.value = value;
            return Some(Outcome::Paused);
        }
        v.value = ValueMerge::default();
        if value.replaced {
            Some(Outcome::Replaced)
        } else {
            None
        }
    }

    fn merge_leaf(
        &mut self,
        page: Option<u64>,
        ops: Span,
        ranges: Span,
        batch: &WriteBatch,
    ) -> Outcome {
        let mut m = self.leaf;
        if !m.active {
            let frame = page.map_or(NO_FRAME, |page| {
                self.pages
                    .cached(page)
                    .expect("the write loads the leaves it changes")
            });
            m = LeafMerge {
                active: true,
                frame,
                ops,
                ranges,
                base: self.lists[0].entries.len(),
                ..LeafMerge::default()
            };
            self.run_ends.clear();
            self.value_puts.clear();
        }
        // SAFETY: the write pinned this leaf, and published frames are never
        // modified.
        let node = (m.frame != NO_FRAME)
            .then(|| unsafe { Node::new(self.pages.pool.page_unbounded(m.frame)) });
        if !m.merged {
            if !self.merge_entries(&mut m, node, batch) {
                self.leaf = m;
                return Outcome::Paused;
            }
            if !m.changed {
                self.lists[0].entries.truncate(m.base);
                self.leaf.active = false;
                return Outcome::Unchanged;
            }
            (m.merged, m.items, m.a) = (true, self.lists[0].len(), m.base);
        }
        let outcome = self.emit_leaves(&mut m, batch);
        self.leaf = m;
        self.leaf.active = matches!(outcome, Outcome::Paused);
        outcome
    }

    /// Merges the entries of the leaf `m` rebuilds, `node`, with its operations and
    /// ranges into `lists[0]`. Returns false if it paused: `m` keeps how far it
    /// got.
    fn merge_entries(&mut self, m: &mut LeafMerge, node: Option<Node>, batch: &WriteBatch) -> bool {
        let old = node.map_or(0, |n| n.count());
        let (ops, ranges) = (m.ops.of(batch.points()), m.ranges.of(batch.ranges()));
        let max_entry = node::max_entry(self.page_size);
        // The rebuilt leaf's entries go after those of its rebuilt siblings, until
        // the parent builds them all.
        let (leaf, run_ends, values) = (
            &mut self.lists[0].entries,
            &mut self.run_ends,
            &mut self.value_puts,
        );
        let LeafMerge {
            i,
            j,
            r,
            changed,
            in_run,
            ..
        } = m;
        loop {
            if self.budget.spent() {
                return false;
            }
            self.budget.work += 1;
            let ord = match (*i < old, *j < ops.len()) {
                (false, false) => break,
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                (true, true) => node.unwrap().cmp_key(*i, batch.key(ops[*j])),
            };
            if ord == Ordering::Less {
                let n = node.unwrap();
                while *r < ranges.len() && n.cmp_key(*i, batch.range_end(ranges[*r].1)).is_ge() {
                    *r += 1;
                }
                if *r < ranges.len() && n.cmp_key(*i, batch.range_start(ranges[*r].0)).is_ge() {
                    *changed = true;
                    *i += 1;
                    continue;
                }
                // The entries below the next put or delete, and below the next range,
                // all stay: copy them without comparing each.
                let mut end = old;
                if *j < ops.len() {
                    end = end.min(n.search(batch.key(ops[*j])).0);
                }
                if *r < ranges.len() {
                    end = end.min(n.search(batch.range_start(ranges[*r].0)).0);
                }
                debug_assert!(end > *i);
                if *in_run {
                    run_ends.push(leaf.len() as u32);
                    *in_run = false;
                }
                for k in *i..end {
                    leaf.push_from(&n, k);
                }
                self.budget.work += end - *i;
                *i = end;
                continue;
            }
            let op = ops[*j];
            *j += 1;
            if ord == Ordering::Equal {
                // The existing entry is replaced or deleted.
                *i += 1;
                *changed = true;
            }
            if let Some(value) = batch.value(op) {
                let key = batch.key(op);
                if key.len() + value.len() <= max_entry {
                    leaf.push(key, value);
                    *in_run = true;
                } else {
                    // A value leaf of its own, between the leaves of the entries
                    // before and after it.
                    if *in_run {
                        run_ends.push(leaf.len() as u32);
                        *in_run = false;
                    }
                    values.push((leaf.len() as u32, op));
                }
                *changed = true;
            }
        }
        if *in_run {
            run_ends.push(leaf.len() as u32);
            *in_run = false;
        }
        true
    }

    /// Adds to `lists[0]` the nodes, unbuilt, that the entries of the leaf `m`
    /// rebuilds go to, those past `m.base`, with the value leaves among them.
    /// Returns `Paused` if it paused: `m` keeps how far it got.
    fn emit_leaves(&mut self, m: &mut LeafMerge, batch: &WriteBatch) -> Outcome {
        loop {
            // The entries up to the next value leaf, or all those left.
            let b = match self.value_puts.get(m.vp) {
                Some(&(at, _)) => at as usize,
                None => self.lists[0].entries.len(),
            };
            if !self.emit_segment(m, b) {
                return Outcome::Paused;
            }
            let Some(&(at, op)) = self.value_puts.get(m.vp) else {
                break;
            };
            let (key, value) = (batch.key(op), batch.value(op).unwrap());
            self.lists[0].emit_value(key, op, key.len() + value.len());
            (m.vp, m.a) = (m.vp + 1, at as usize);
        }
        self.added += self.lists[0].len() - m.items;
        Outcome::Replaced
    }

    /// Adds leaves holding entries `m.a..b` of `lists[0]`: one if they fit.
    /// Returns false if it paused: `m` keeps how far it got.
    fn emit_segment(&mut self, m: &mut LeafMerge, b: usize) -> bool {
        let (out, page) = (&mut self.lists[0], self.page_size);
        if !m.splitting {
            if m.a == b {
                return true;
            }
            // Every entry takes a byte at least: only as many as a page has bytes
            // are worth measuring.
            let used = match b - m.a <= page {
                true => node::size(&out.entries, m.a, b, 0),
                false => usize::MAX,
            };
            if used <= page {
                out.emit(m.a, b, used, Life::Short);
                return true;
            }
            (m.splitting, m.run, m.s) = (true, 0, m.a);
        }
        // Split after each run of puts that fills at least a quarter page, as when
        // appending to ledgers: pack the run full, and leave the entries after it to
        // other leaves, so that the next appends don't rewrite them. Smaller runs,
        // like scattered inserts, aren't worth leaves of their own.
        while m.run < self.run_ends.len() {
            let end = self.run_ends[m.run] as usize;
            if !m.packing {
                // Whether entries `s..end` take less than a quarter page: as many
                // entries as a quarter page has bytes take more.
                let small =
                    |s: usize| end - s < page / 4 && node::size(&out.entries, s, end, 0) < page / 4;
                if end <= m.a || end > b || small(m.s) {
                    m.run += 1;
                    continue;
                }
                m.packing = true;
            }
            while m.s < end {
                if self.budget.spent() {
                    return false;
                }
                let (e, used) = node::fit(&out.entries, m.s, end, page, 0);
                // The run's full leaves are likely to stay; appends go on in its last.
                let life = if e < end { Life::Long } else { Life::Short };
                out.emit(m.s, e, used, life);
                self.budget.work += e - m.s;
                m.s = e;
            }
            m.packing = false;
            m.run += 1;
        }
        out.emit_even(m.s, b, page, 0);
        m.splitting = false;
        true
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
        self.budget.work += 1;
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

    /// Builds the nodes this write added to `lists[level]`, from item `*k` on.
    /// Returns false if it paused: `*k` is where it goes on.
    fn build_nodes(&mut self, level: usize, k: &mut usize) -> bool {
        while *k < self.lists[level].len() {
            if self.budget.spent() {
                return false;
            }
            self.budget.work += 1;
            if self.lists[level].items[*k].fresh {
                self.build_node(level, *k);
            }
            *k += 1;
        }
        true
    }

    /// Adds nodes one level up that take over the nodes of `lists[level]` past
    /// `mark`, all built, to `lists[level + 1]`.
    fn add_parents(&mut self, level: usize, mark: Mark) {
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
        let generation = self.generation + 1;
        let (frame, page, used) =
            self.pages
                .build(level as u8, &list.entries, a, b, item.life, generation);
        debug_assert_eq!(used, item.used as usize);
        list.built(i, frame, page);
        self.budget.work += b - a;
    }

    /// Moves the nodes of `lists[level]`, from item `*k` on, that the write keeps,
    /// but whose pages the allocator wants freed: they become nodes the write adds,
    /// with the same entries, built next to the others. Their parent is rebuilt
    /// anyway, so a move costs a page write and nothing else: leaves move only if
    /// they are cached. A write moves at most as many nodes as it adds. Returns
    /// false if it paused: `*k` is where it goes on.
    fn move_kept(&mut self, level: usize, k: &mut usize) -> bool {
        while *k < self.lists[level].len() {
            if self.budget.spent() {
                return false;
            }
            self.budget.work += 1;
            let i = *k;
            *k += 1;
            let item = self.lists[level].items[i];
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
            list.restage(i, a, b, node::size(&list.entries, a, b, level as u8));
            self.budget.work += b - a;
            self.dropped.push(item.page);
            if level > 0 {
                self.release(frame);
            }
            self.moved += 1;
        }
        true
    }

    /// Merges each node this write added that is under a quarter full into a
    /// neighbor, among the nodes of `lists[level]` past `start`, if the two fit in a
    /// page and the neighbor is in memory, going through them from item `*k` on.
    /// The nodes this write added aren't built yet, so no build goes to waste.
    /// Returns false if it paused: `*k` is where it goes on.
    fn fix_underfull(&mut self, level: usize, start: usize, k: &mut usize) -> bool {
        'next: while *k < self.lists[level].len() {
            if self.budget.spent() {
                return false;
            }
            self.budget.work += 1;
            let len = self.lists[level].len();
            let item = self.lists[level].items[*k];
            if item.fresh
                && !item.is_value_leaf()
                && (item.used as usize) < self.page_size / 4
                && len - start > 1
            {
                let left = k.checked_sub(1).filter(|&a| a >= start);
                let right = (*k + 1 < len).then_some(*k);
                for a in left.into_iter().chain(right) {
                    if self.merge_pair(level, a) {
                        *k = a;
                        continue 'next;
                    }
                }
            }
            *k += 1;
        }
        true
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
        self.budget.work += end - start;
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

    /// Stacks levels on the nodes that replace the root, from `lists[level]`, until
    /// one node remains, and returns it as the new root. Returns `None` if it
    /// paused, keeping how far it got in [`Phase::Growing`].
    fn grow(&mut self, mut level: usize, mut fixed: bool, mut k: usize) -> Option<Option<Root>> {
        loop {
            if !fixed {
                if self.lists[level].len() > 1 && !self.fix_underfull(level, 0, &mut k) {
                    self.phase = Phase::Growing { level, fixed, k };
                    return None;
                }
                (fixed, k) = (true, 0);
            }
            match self.lists[level].len() {
                0 => return Some(None),
                // A value leaf always has a parent.
                1 if level > 0 || !self.lists[0].items[0].is_value_leaf() => break,
                _ => {
                    self.ensure_level(level + 1);
                    if !self.build_nodes(level, &mut k) {
                        self.phase = Phase::Growing { level, fixed, k };
                        return None;
                    }
                    self.add_parents(level, Mark::default());
                    (level, fixed, k) = (level + 1, false, 0);
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
        Some(Some(root))
    }
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

    /// Whether this write added the node and built it.
    fn is_built(&self) -> bool {
        // No node is on page 0, a meta page.
        self.fresh && self.page != 0
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
        // The pages each frame takes: a large frame holds a value leaf.
        let pages = |frame| pool.frame(frame).pages as usize;
        let cancelled: usize = (0..pool.slots())
            .map(|i| pool.slot(i))
            .filter(|&f| pool.frame(f).holds == Holds::Loading && pool.refs(f) == 0)
            .map(pages)
            .sum();
        let cached: usize = cache.entries().map(|(_, frame)| pages(frame)).sum();
        assert_eq!(
            pool.in_use(),
            stats.internal + cached + cancelled,
            "frames leaked"
        );
        for (page, frame) in cache.entries() {
            let info = pool.frame(frame);
            assert_eq!(
                (info.holds, info.page, info.dirty),
                (Holds::Leaf, page, false)
            );
            assert_eq!(
                info.pages as u64,
                node::pages(page),
                "leaf {page} misframed"
            );
            assert_eq!(pool.refs(frame), 0, "leaf {page} still pinned");
            let first = node::first_page(page);
            assert!(space.is_used(first), "cached leaf {page} on a free page");
        }
        assert!(self.pinned.is_empty() && self.dirty.is_empty() && self.pending.is_none());
        assert!(self.phase == Phase::Idle, "a merge was left halfway");
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
                // Its key is its parent's key for it.
                let lo = lo.expect("a value leaf at the root");
                if let Some(frame) = self.pages.cached(page) {
                    let leaf = ValueLeaf::new(self.pages.pool.page(frame));
                    assert_eq!(leaf.key(), lo, "value leaf under another key");
                }
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

    pub(crate) fn set_slice(&mut self, slice: Option<usize>) {
        self.budget.fixed = slice;
    }

    /// Whether the write in progress is merging its batch.
    pub(crate) fn merging(&self) -> bool {
        matches!(self.phase, Phase::Merging | Phase::Growing { .. })
    }

    /// The leaf pages that planning a prepared `batch` lists, found as planning did
    /// before it paused: at once, recursively.
    pub(crate) fn plan_at_once(&self, batch: &WriteBatch) -> Vec<u64> {
        let mut touched = Vec::new();
        let (ops, ranges) = (batch.points(), batch.ranges());
        if ops.is_empty() && ranges.is_empty() {
            return touched;
        }
        match self.root {
            None => {}
            Some(root) if root.level == 0 => touched.push(root.page),
            Some(root) => self.plan_subtree(root.frame, ops, ranges, None, batch, &mut touched),
        }
        touched
    }

    fn plan_subtree(
        &self,
        frame: FrameId,
        mut ops: &[u32],
        mut ranges: &[(u32, u32)],
        hi: Option<&[u8]>,
        batch: &WriteBatch,
        touched: &mut Vec<u64>,
    ) {
        let node = self.node(frame);
        let keys: Vec<Vec<u8>> = (0..node.count())
            .map(|i| {
                let mut key = Vec::new();
                node.key_into(i, &mut key);
                key
            })
            .collect();
        for i in 0..node.count() {
            let child_hi = keys.get(i + 1).map(Vec::as_slice).or(hi);
            let (child_ops, child_ranges, covered) =
                split(batch, &mut ops, &mut ranges, &keys[i], child_hi);
            if covered || (child_ops.is_empty() && child_ranges.is_empty()) {
                continue;
            }
            if node.level() > 1 {
                let child = node.child(i);
                self.plan_subtree(child, child_ops, child_ranges, child_hi, batch, touched);
            } else if node::span(node.child_page(i)) == 0 {
                touched.push(node.child_page(i));
            }
        }
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
