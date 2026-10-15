//! The copy-on-write B+tree.
//!
//! Nodes are immutable once published. A write builds replacement nodes along the
//! paths it touches, shares everything else with the previous version, then swaps
//! the root. Frames are reference counted: every node holds a reference to each of
//! its children, the current root holds one, and so does each pinned reader. When a
//! version's last reference goes, exactly the nodes it doesn't share are freed.
//!
//! Each internal key equals the smallest key in its child's subtree.

use std::cmp::Ordering;
use std::mem;

use crate::Comparison;
use crate::batch::WriteBatch;
use crate::node::{self, CHILD_REF, Entries, EntryBuf, Node};
use crate::pool::{FrameId, Pool};

/// The deepest tree supported. Every split leaves internal nodes with at least two
/// children, so real trees stay far shallower.
pub(crate) const MAX_HEIGHT: usize = 32;

pub(crate) struct Tree {
    pool: Pool,
    root: Option<FrameId>,
    page_size: usize,
    // Scratch space, reused so that writes don't allocate:
    /// Replacement nodes, per level.
    lists: Vec<NodeList>,
    /// Separator keys, two per level.
    bounds: Vec<Vec<u8>>,
    /// Entries of the leaf being rebuilt.
    leaf: EntryBuf,
    /// Where each run of puts ends in `leaf`.
    run_ends: Vec<u32>,
    /// Entries of two nodes being merged.
    pair: EntryBuf,
    /// Frames waiting to be released.
    stack: Vec<FrameId>,
}

enum Outcome {
    Unchanged,
    Replaced,
}

impl Tree {
    pub(crate) fn new(page_size: usize) -> Tree {
        Tree {
            pool: Pool::new(page_size),
            root: None,
            page_size,
            lists: Vec::new(),
            bounds: Vec::new(),
            leaf: EntryBuf::default(),
            run_ends: Vec::new(),
            pair: EntryBuf::default(),
            stack: Vec::new(),
        }
    }

    pub(crate) fn node(&self, frame: FrameId) -> Node<'_> {
        Node::new(self.pool.page(frame))
    }

    /// Address of byte `offset` of `frame`.
    pub(crate) fn addr(&self, frame: FrameId, offset: usize) -> *const u8 {
        self.pool.addr(frame, offset)
    }

    pub(crate) fn root(&self) -> Option<FrameId> {
        self.root
    }

    /// Takes a reference to the current root, keeping the current version alive
    /// until it is [released](Self::release).
    pub(crate) fn pin(&mut self) -> Option<FrameId> {
        let root = self.root?;
        self.pool.retain(root);
        Some(root)
    }

    /// Drops a reference to `frame`. Frames that lose their last reference are
    /// freed, and release their children in turn.
    pub(crate) fn release(&mut self, frame: FrameId) {
        let mut stack = mem::take(&mut self.stack);
        stack.push(frame);
        while let Some(frame) = stack.pop() {
            if !self.pool.unref(frame) {
                continue;
            }
            let node = Node::new(self.pool.page(frame));
            if node.level() > 0 {
                stack.extend((0..node.count()).map(|i| node.child(i)));
            }
            self.pool.free(frame);
        }
        self.stack = stack;
    }

    /// Positions `cur` on the entry matching `key` under `cmp`, in the version rooted
    /// at `root`. Returns false if there is no such entry.
    pub(crate) fn seek(
        &self,
        root: FrameId,
        key: &[u8],
        cmp: Comparison,
        cur: &mut Cursor,
    ) -> bool {
        cur.depth = 0;
        let mut frame = root;
        loop {
            let node = self.node(frame);
            let (i, eq) = node.search(key);
            if node.level() > 0 {
                // The child holding `key`: the last one whose smallest key is <= it.
                let child = if eq { i } else { i.saturating_sub(1) };
                cur.push(frame, child);
                frame = node.child(child);
                continue;
            }
            let n = node.count();
            let at = match cmp {
                Comparison::Equal => eq.then_some(i),
                Comparison::Floor if eq => Some(i),
                Comparison::Floor | Comparison::Lower => i.checked_sub(1),
                Comparison::Ceiling => Some(i).filter(|&i| i < n),
                Comparison::Higher => Some(i + usize::from(eq)).filter(|&i| i < n),
            };
            if let Some(at) = at {
                cur.push(frame, at);
                return true;
            }
            // The match, if any, is in a neighboring leaf.
            return match cmp {
                Comparison::Equal => false,
                Comparison::Floor | Comparison::Lower => {
                    cur.push(frame, 0);
                    self.step(cur, false)
                }
                Comparison::Ceiling | Comparison::Higher => {
                    cur.push(frame, n - 1);
                    self.step(cur, true)
                }
            };
        }
    }

    /// Positions `cur` on the first entry of the version rooted at `root`.
    pub(crate) fn seek_first(&self, root: FrameId, cur: &mut Cursor) {
        cur.depth = 0;
        let mut frame = root;
        loop {
            let node = self.node(frame);
            cur.push(frame, 0);
            if node.level() == 0 {
                return;
            }
            frame = node.child(0);
        }
    }

    /// Moves `cur` to the next entry, or the previous one. Returns false, leaving
    /// `cur` where it was, if there is none.
    pub(crate) fn step(&self, cur: &mut Cursor, forward: bool) -> bool {
        // Climb to the nearest level that can move that way, and move there.
        let mut depth = cur.depth;
        loop {
            if depth == 0 {
                return false;
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
        true
    }

    /// Applies a prepared batch and publishes the result as the current version.
    pub(crate) fn apply(&mut self, batch: &WriteBatch) {
        let (ops, ranges) = (batch.points(), batch.ranges());
        let (level, outcome) = match self.root {
            Some(root) => {
                let level = usize::from(self.node(root).level());
                self.ensure_level(level);
                (level, self.merge(root, level, ops, ranges, None, batch))
            }
            None => {
                self.ensure_level(0);
                (0, self.merge_leaf(None, ops, &[], batch))
            }
        };
        if let Outcome::Unchanged = outcome {
            return;
        }
        let root = self.grow(level);
        if let Some(old) = mem::replace(&mut self.root, root) {
            self.release(old);
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
    /// `frame`. Its replacement nodes, if any, go to `lists[level]`.
    fn merge(
        &mut self,
        frame: FrameId,
        level: usize,
        ops: &[u32],
        ranges: &[(u32, u32)],
        hi: Option<&[u8]>,
        batch: &WriteBatch,
    ) -> Outcome {
        if level == 0 {
            self.merge_leaf(Some(frame), ops, ranges, batch)
        } else {
            self.merge_internal(frame, level, ops, ranges, hi, batch)
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
        let node = unsafe { Node::new(self.pool.page_unbounded(frame)) };
        let below = level - 1;
        let mark = self.lists[below].mark();
        let mut key = mem::take(&mut self.bounds[2 * level]);
        let mut next = mem::take(&mut self.bounds[2 * level + 1]);
        let mut changed = false;
        node.key_into(0, &mut key);
        for i in 0..node.count() {
            let child = node.child(i);
            let child_hi = if i + 1 < node.count() {
                node.key_into(i + 1, &mut next);
                Some(&next[..])
            } else {
                hi
            };
            let n = child_hi.map_or(ops.len(), |hi| ops.partition_point(|&o| batch.key(o) < hi));
            let (child_ops, rest) = ops.split_at(n);
            ops = rest;
            // Ranges ending at or before this child's smallest key can't touch it, or
            // any later child.
            ranges = &ranges[ranges.partition_point(|&(_, e)| batch.range_end(e) <= &key[..])..];
            let n = child_hi.map_or(ranges.len(), |hi| {
                ranges.partition_point(|&(s, _)| batch.range_start(s) < hi)
            });
            let child_ranges = &ranges[..n];
            if child_ops.is_empty() && child_ranges.is_empty() {
                self.keep(below, &key, child);
            } else if child_ops.is_empty() && covers(batch, child_ranges, &key, child_hi) {
                // The whole subtree goes, without visiting it.
                changed = true;
            } else {
                match self.merge(child, below, child_ops, child_ranges, child_hi, batch) {
                    Outcome::Unchanged => self.keep(below, &key, child),
                    Outcome::Replaced => changed = true,
                }
            }
            mem::swap(&mut key, &mut next);
        }
        self.bounds[2 * level] = key;
        self.bounds[2 * level + 1] = next;
        if !changed {
            self.discard(below, mark);
            return Outcome::Unchanged;
        }
        self.fix_underfull(below, mark.items);
        self.build_parents(below, mark);
        Outcome::Replaced
    }

    fn merge_leaf(
        &mut self,
        frame: Option<FrameId>,
        ops: &[u32],
        ranges: &[(u32, u32)],
        batch: &WriteBatch,
    ) -> Outcome {
        // SAFETY: as in `merge_internal`.
        let node = frame.map(|f| unsafe { Node::new(self.pool.page_unbounded(f)) });
        let old = node.map_or(0, |n| n.count());
        let (leaf, run_ends) = (&mut self.leaf, &mut self.run_ends);
        leaf.clear();
        run_ends.clear();
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
                } else {
                    if in_run {
                        run_ends.push(leaf.len() as u32);
                        in_run = false;
                    }
                    leaf.push_from(&n, i);
                }
                i += 1;
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
                leaf.push(batch.key(op), value);
                in_run = true;
                changed = true;
            }
        }
        if in_run {
            run_ends.push(leaf.len() as u32);
        }
        if !changed {
            return Outcome::Unchanged;
        }
        let (pool, out, leaf, page) = (
            &mut self.pool,
            &mut self.lists[0],
            &self.leaf,
            self.page_size,
        );
        let n = leaf.len();
        if n == 0 {
            return Outcome::Replaced;
        }
        if node::size(leaf, 0, n) <= page {
            emit(pool, out, 0, leaf, 0, n);
            return Outcome::Replaced;
        }
        // Split after each run of puts that fills at least a quarter page, as when
        // appending to ledgers: pack the run full, and leave the entries after it to
        // other leaves, so that the next appends don't rewrite them. Smaller runs,
        // like scattered inserts, aren't worth leaves of their own.
        let mut s = 0;
        for &end in &self.run_ends {
            let end = end as usize;
            if node::size(leaf, s, end) < page / 4 {
                continue;
            }
            while s < end {
                let e = node::fit(leaf, s, end, page);
                emit(pool, out, 0, leaf, s, e);
                s = e;
            }
        }
        emit_even(pool, out, 0, leaf, s, n, page);
        Outcome::Replaced
    }

    /// Adds an unchanged node to `lists[level]`.
    fn keep(&mut self, level: usize, key: &[u8], frame: FrameId) {
        self.pool.retain(frame);
        let used = self.node(frame).used();
        self.lists[level].push(key, frame, used, false);
    }

    /// Drops the nodes added to `lists[level]` since `mark`.
    fn discard(&mut self, level: usize, mark: Mark) {
        for i in mark.items..self.lists[level].len() {
            let frame = self.lists[level].items[i].frame;
            self.release(frame);
        }
        self.lists[level].truncate(mark);
    }

    /// Builds nodes one level up from the nodes in `lists[level]` past `mark`, which
    /// they take over, and appends them to `lists[level + 1]`.
    fn build_parents(&mut self, level: usize, mark: Mark) {
        let (lower, upper) = self.lists.split_at_mut(level + 1);
        let children = &mut lower[level];
        let (a, b) = (mark.items, children.len());
        emit_even(
            &mut self.pool,
            &mut upper[0],
            level as u8 + 1,
            &*children,
            a,
            b,
            self.page_size,
        );
        children.truncate(mark);
    }

    /// Merges each node this write built that is under a quarter full into a
    /// neighbor, among the nodes of `lists[level]` past `start`, if the two fit in a
    /// page.
    fn fix_underfull(&mut self, level: usize, start: usize) {
        let mut k = start;
        'next: while k < self.lists[level].len() {
            let len = self.lists[level].len();
            let item = self.lists[level].items[k];
            if item.fresh && (item.used as usize) < self.page_size / 4 && len - start > 1 {
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

    /// Rebuilds nodes `a` and `a + 1` of `lists[level]` as one node, if they fit.
    fn merge_pair(&mut self, level: usize, a: usize) -> bool {
        let (x, y) = {
            let items = &self.lists[level].items;
            (items[a].frame, items[a + 1].frame)
        };
        self.pair.clear();
        for frame in [x, y] {
            let node = Node::new(self.pool.page(frame));
            for i in 0..node.count() {
                self.pair.push_from(&node, i);
            }
        }
        let n = self.pair.len();
        if node::size(&self.pair, 0, n) > self.page_size {
            return false;
        }
        let frame = self.pool.alloc();
        let used = node::build(self.pool.page_mut(frame), level as u8, &self.pair, 0, n);
        if level > 0 {
            // The new node becomes another parent of every child of the two.
            for i in 0..n {
                self.pool.retain(node::child_frame(self.pair.value(i)));
            }
        }
        self.lists[level].replace_pair(a, self.pair.key(0), frame, used);
        self.release(x);
        self.release(y);
        true
    }

    /// Stacks levels on the replacement nodes for the root, in `lists[level]`,
    /// until one node remains, and returns it as the new root.
    fn grow(&mut self, mut level: usize) -> Option<FrameId> {
        loop {
            if self.lists[level].len() > 1 {
                self.fix_underfull(level, 0);
            }
            match self.lists[level].len() {
                0 => return None,
                1 => break,
                _ => {
                    self.ensure_level(level + 1);
                    self.build_parents(level, Mark::default());
                    level += 1;
                }
            }
        }
        // The list's reference becomes the root's.
        let mut root = self.lists[level].items[0].frame;
        self.lists[level].truncate(Mark::default());
        // A root with a single child is redundant.
        loop {
            let node = self.node(root);
            if node.level() == 0 || node.count() > 1 {
                return Some(root);
            }
            let child = node.child(0);
            self.pool.retain(child);
            self.release(root);
            root = child;
        }
    }
}

/// Whether `ranges` remove every key in `lo..hi`, which holds when the first of
/// them, the only one that can contain `lo`, spans it all.
fn covers(batch: &WriteBatch, ranges: &[(u32, u32)], lo: &[u8], hi: Option<&[u8]>) -> bool {
    match (ranges.first(), hi) {
        (Some(&(s, e)), Some(hi)) => batch.range_start(s) <= lo && batch.range_end(e) >= hi,
        _ => false,
    }
}

/// Builds entries `a..b` of `src` into a new node at `level`, appended to `out`.
fn emit(pool: &mut Pool, out: &mut NodeList, level: u8, src: &impl Entries, a: usize, b: usize) {
    let frame = pool.alloc();
    let used = node::build(pool.page_mut(frame), level, src, a, b);
    out.push(src.key(a), frame, used, true);
}

/// Builds entries `a..b` of `src` into as few nodes as fit, of similar sizes.
fn emit_even(
    pool: &mut Pool,
    out: &mut NodeList,
    level: u8,
    src: &impl Entries,
    a: usize,
    b: usize,
    page_size: usize,
) {
    let mut s = a;
    while s < b {
        let total = node::size(src, s, b);
        if total <= page_size {
            emit(pool, out, level, src, s, b);
            return;
        }
        let e = node::fit(src, s, b, total / total.div_ceil(page_size));
        emit(pool, out, level, src, s, e);
        s = e;
    }
}

/// Nodes replacing one node's children, in key order, each holding a reference.
#[derive(Default)]
struct NodeList {
    /// Each node's smallest key, followed by a child reference to it.
    bytes: Vec<u8>,
    items: Vec<Item>,
}

#[derive(Clone, Copy)]
struct Item {
    at: u32,
    key_len: u32,
    frame: FrameId,
    used: u32,
    /// Built by this write, rather than kept from the previous version.
    fresh: bool,
}

#[derive(Clone, Copy, Default)]
struct Mark {
    items: usize,
    bytes: usize,
}

impl NodeList {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn mark(&self) -> Mark {
        Mark {
            items: self.items.len(),
            bytes: self.bytes.len(),
        }
    }

    fn truncate(&mut self, mark: Mark) {
        self.items.truncate(mark.items);
        self.bytes.truncate(mark.bytes);
    }

    fn push(&mut self, key: &[u8], frame: FrameId, used: usize, fresh: bool) {
        let at = self.bytes.len() as u32;
        self.bytes.extend_from_slice(key);
        self.bytes.extend_from_slice(&node::child_ref(frame));
        let (key_len, used) = (key.len() as u32, used as u32);
        self.items.push(Item {
            at,
            key_len,
            frame,
            used,
            fresh,
        });
    }

    /// Replaces nodes `i` and `i + 1` with a fresh node starting at `key`.
    fn replace_pair(&mut self, i: usize, key: &[u8], frame: FrameId, used: usize) {
        self.push(key, frame, used, true);
        self.items[i] = self.items.pop().unwrap();
        self.items.remove(i + 1);
    }
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

/// A reader's position: the node and entry index at each level, root first.
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

    /// The leaf and entry index the cursor is on.
    pub(crate) fn leaf(&self) -> (FrameId, usize) {
        let (frame, i) = self.path[self.depth - 1];
        (frame, usize::from(i))
    }
}

#[cfg(test)]
impl Tree {
    /// Checks the tree's structure, and that it holds the only reference to every
    /// allocated frame. Panics on a violation. Only valid while no reader is pinned.
    pub(crate) fn check(&self) {
        let mut nodes = 0;
        if let Some(root) = self.root {
            let level = self.node(root).level();
            self.check_node(root, level, None, None, &mut nodes);
        }
        assert_eq!(self.pool.in_use(), nodes, "frames leaked");
    }

    /// Checks the subtree at `frame`, whose keys must fall in `lo..hi`, and returns
    /// its smallest key.
    fn check_node(
        &self,
        frame: FrameId,
        level: u8,
        lo: Option<&[u8]>,
        hi: Option<&[u8]>,
        nodes: &mut usize,
    ) -> Vec<u8> {
        *nodes += 1;
        assert_eq!(
            self.pool.refs(frame),
            1,
            "frame {frame} has extra references"
        );
        let node = self.node(frame);
        assert_eq!(node.level(), level, "uneven depth");
        let n = node.count();
        assert!(n > 0, "empty node");
        let mut entries = EntryBuf::default();
        for i in 0..n {
            entries.push_from(&node, i);
        }
        assert_eq!(node.used(), node::size(&entries, 0, n), "corrupt layout");
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
                let child = node.child(i);
                let min = self.check_node(child, level - 1, Some(entries.key(i)), child_hi, nodes);
                assert_eq!(
                    min,
                    entries.key(i),
                    "internal key isn't its child's smallest"
                );
            }
        }
        entries.key(0).to_vec()
    }

    /// Number of leaves, and the bytes they use.
    pub(crate) fn leaf_usage(&self) -> (usize, usize) {
        let mut usage = (0, 0);
        let mut stack: Vec<FrameId> = self.root.into_iter().collect();
        while let Some(frame) = stack.pop() {
            let node = self.node(frame);
            if node.level() == 0 {
                usage.0 += 1;
                usage.1 += node.used();
            } else {
                stack.extend((0..node.count()).map(|i| node.child(i)));
            }
        }
        usage
    }

    pub(crate) fn page_size(&self) -> usize {
        self.page_size
    }
}
