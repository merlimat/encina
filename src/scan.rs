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

//! The scan that finishes opening a store: it reads the level-1 nodes of the
//! version the store opened, and marks the pages of their leaves used.
//!
//! The open reads only the internal nodes above level 1, which stay in memory,
//! and marks their pages and those of the level-1 nodes. Level 1 holds nearly all
//! the internal nodes, so reading it took most of the open, and the scan reads it
//! instead while the store serves reads and writes. Until the scan ends, writes
//! take pages past where the file ended (see [`Space::start_scan`]). The scan
//! holds the version it reads, as an iterator would, so no write frees a page of
//! it before the scan has marked it, and no write overwrites a node it is yet to
//! read.
//!
//! The scan checks what it reads as the open checks the nodes above: a leaf page
//! listed twice, or past the end of the file, stops it with
//! [`Error::Corruption`], as would an I/O error with [`Error::Io`]. Writes then
//! fail with that error, since the store can't know which of its pages are free.

use crate::error::{Error, Result};
use crate::meta::META_PAGES;
use crate::node::{self, Node};
use crate::pool::{FrameId, Pool};
use crate::reads::Reads;
use crate::space::Space;
use crate::tree::{MAX_HEIGHT, Tree, Version};

/// The most level-1 nodes the scan reads at once. NVMe drives read faster with
/// more requests in flight than the 16 that writes and iterators keep.
pub(crate) const WINDOW: usize = 64;

pub(crate) struct Scan {
    /// The version the store opened, which the scan holds.
    pub(crate) version: Version,
    /// Where the file ended when the store opened: every page of `version` lies
    /// below it.
    high_water: u64,
    /// The root of `version`, if it is a level-1 node the scan hasn't reached.
    lone: Option<u64>,
    /// The path to the next level-1 node, through the version's nodes above
    /// level 1, root first: each node's frame, and the child it is at. Past the
    /// last level-1 node, `depth` is 0.
    path: [(FrameId, u16); MAX_HEIGHT],
    depth: usize,
    /// The level-1 nodes in the window, in key order, with the read held for
    /// each, if any: a ring of `len` from `first`.
    window: [(u64, Option<usize>); WINDOW],
    first: usize,
    len: usize,
    /// The level-1 nodes whose leaves it hasn't marked.
    left: usize,
    /// The error that stopped it, if any.
    pub(crate) error: Option<Error>,
}

impl Scan {
    /// A scan of the level-1 nodes of the tree's current version, which it pins,
    /// in a file whose pages up to the high-water mark may be in use: allocation
    /// takes none of them until the scan ends. `None` if the tree has no level 1.
    pub(crate) fn new(tree: &mut Tree) -> Option<Scan> {
        let root = tree.current().root.filter(|r| r.level > 0)?;
        let version = tree.pin();
        let space = &mut tree.pages.space;
        space.start_scan();
        let mut scan = Scan {
            version,
            high_water: space.high_water(),
            lone: None,
            path: [(0, 0); MAX_HEIGHT],
            depth: 0,
            window: [(0, None); WINDOW],
            first: 0,
            len: 0,
            left: 1,
            error: None,
        };
        if root.level == 1 {
            scan.lone = Some(root.page);
        } else {
            scan.path[0] = (root.frame, 0);
            scan.depth = 1;
            scan.left = level_1_under(&tree.pages.pool, root.frame);
        }
        Some(scan)
    }

    /// Ends the scan, every leaf marked: allocation may take any free page, and
    /// the version it held may go.
    pub(crate) fn end(self, tree: &mut Tree) {
        debug_assert!(self.left == 0 && self.error.is_none());
        tree.pages.space.end_scan();
        tree.unpin(self.version);
    }

    /// Stops the scan at error `e`, letting go of the reads it holds. It keeps the
    /// version it holds: writes fail from now on, so none replaces it. Returns a
    /// copy of the error.
    pub(crate) fn fail<R>(&mut self, e: Error, reads: &mut Reads<R>, pool: &mut Pool) -> Error {
        self.release(reads, pool);
        let copy = e.duplicate();
        self.error = Some(e);
        copy
    }

    /// Lets go of the reads it holds.
    pub(crate) fn release<R>(&mut self, reads: &mut Reads<R>, pool: &mut Pool) {
        for k in 0..self.len {
            if let Some(slot) = self.at(k).1.take() {
                reads.leave(slot, pool);
            }
        }
    }

    /// The level-1 nodes whose leaves it hasn't marked.
    pub(crate) fn left(&self) -> usize {
        self.left
    }

    /// Moves to the next level-1 node, and returns its page: `None` past the last.
    pub(crate) fn next_page(&mut self, pool: &Pool) -> Option<u64> {
        if let Some(page) = self.lone.take() {
            return Some(page);
        }
        while self.depth > 0 {
            let (frame, i) = self.path[self.depth - 1];
            let node = Node::new(pool.page(frame));
            let i = usize::from(i);
            if i == node.count() {
                // Past this node's children: on to its parent's next child.
                self.depth -= 1;
                if let Some((_, at)) = self.path[..self.depth].last_mut() {
                    *at += 1;
                }
            } else if node.level() == 2 {
                self.path[self.depth - 1].1 += 1;
                return Some(node.child_page(i));
            } else {
                self.path[self.depth] = (node.child(i), 0);
                self.depth += 1;
            }
        }
        None
    }

    /// The number of level-1 nodes in the window.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The `k`th level-1 node in the window: its page, and the read held for it.
    pub(crate) fn at(&mut self, k: usize) -> &mut (u64, Option<usize>) {
        debug_assert!(k < self.len);
        &mut self.window[(self.first + k) % WINDOW]
    }

    /// Adds the level-1 node on `page` to the end of the window.
    pub(crate) fn push(&mut self, page: u64) {
        debug_assert!(self.len < WINDOW, "the window is full");
        self.window[(self.first + self.len) % WINDOW] = (page, None);
        self.len += 1;
    }

    /// Takes the first level-1 node out of the window, once its leaves are marked.
    pub(crate) fn pop(&mut self) {
        debug_assert!(self.window[self.first].1.is_none(), "a read left held");
        self.first = (self.first + 1) % WINDOW;
        self.len -= 1;
        self.left -= 1;
    }

    /// Marks in `space` the pages of the leaves that `node`, the first level-1
    /// node in the window, lists. Fails with [`Error::Corruption`], marking none of
    /// them, if the node holds no children or was written after the version, or if
    /// a leaf lies among the meta pages, past the end of the file, or on a page in
    /// use already.
    pub(crate) fn mark(&self, space: &mut Space, node: &Node) -> Result<()> {
        let page = self.window[self.first].0;
        if node.level() != 1 || node.count() == 0 || node.generation() > self.version.generation {
            return Err(Error::Corruption { page });
        }
        for i in 0..node.count() {
            let leaf = node.child_page(i);
            let (first, n) = (node::first_page(leaf), node::pages(leaf));
            let bad = first < META_PAGES
                || first + n > self.high_water
                || (first..first + n).any(|page| space.is_used(page));
            if bad {
                for leaf in (0..i).map(|j| node.child_page(j)) {
                    let first = node::first_page(leaf);
                    for page in first..first + node::pages(leaf) {
                        space.unmark(page);
                    }
                }
                return Err(Error::Corruption { page: first });
            }
            for page in first..first + n {
                space.mark_used(page);
            }
        }
        Ok(())
    }

    /// The leaf pages it hasn't marked: those of the level-1 nodes in the window,
    /// and past it, which `level_1` gives the bytes of.
    #[cfg(test)]
    pub(crate) fn unmarked(
        &self,
        pool: &Pool,
        level_1: impl Fn(u64) -> Vec<u8>,
    ) -> std::collections::BTreeSet<u64> {
        let mut rest = Scan {
            error: None,
            ..*self
        };
        let window = (0..self.len).map(|k| self.window[(self.first + k) % WINDOW].0);
        let pages: Vec<u64> = window
            .chain(std::iter::from_fn(|| rest.next_page(pool)))
            .collect();
        assert_eq!(pages.len(), self.left, "level-1 nodes miscounted");
        let mut leaves = std::collections::BTreeSet::new();
        for page in pages {
            let bytes = level_1(page);
            let node = Node::new(&bytes);
            for i in 0..node.count() {
                let leaf = node.child_page(i);
                let first = node::first_page(leaf);
                leaves.extend(first..first + node::pages(leaf));
            }
        }
        leaves
    }

    /// The reads it holds.
    #[cfg(test)]
    pub(crate) fn slots(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len).filter_map(|k| self.window[(self.first + k) % WINDOW].1)
    }
}

/// The level-1 nodes under the node above level 1 in `frame`.
fn level_1_under(pool: &Pool, frame: FrameId) -> usize {
    let node = Node::new(pool.page(frame));
    match node.level() {
        2 => node.count(),
        _ => (0..node.count())
            .map(|i| level_1_under(pool, node.child(i)))
            .sum(),
    }
}
