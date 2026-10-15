use std::cell::RefCell;
use std::mem;
use std::ops::{Bound, RangeBounds};
use std::rc::Rc;
use std::slice;

use crate::batch::WriteBatch;
use crate::error::Result;
use crate::pool::FrameId;
use crate::tree::{Cursor, Tree};

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
/// `Sync`; each compio runtime thread owns its own stores.
#[derive(Clone)]
pub struct Db {
    inner: Rc<Inner>,
}

struct Inner {
    tree: RefCell<Tree>,
    max_entry: usize,
    /// Key buffers for reads and iterators, reused so that reads don't allocate.
    buffers: RefCell<Vec<Vec<u8>>>,
}

impl Inner {
    fn take_buffer(&self) -> Vec<u8> {
        self.buffers
            .borrow_mut()
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.max_entry))
    }

    fn put_buffer(&self, mut buffer: Vec<u8>) {
        buffer.clear();
        self.buffers.borrow_mut().push(buffer);
    }
}

impl Db {
    /// An empty store kept only in memory. Persistence comes in the next milestone.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Db::open replaces it with persistence")
    )]
    pub(crate) fn in_memory(opts: crate::Options) -> Result<Db> {
        opts.validate()?;
        let inner = Inner {
            tree: RefCell::new(Tree::new(opts.page_size)),
            max_entry: opts.max_entry_size(),
            buffers: RefCell::new(Vec::new()),
        };
        Ok(Db {
            inner: Rc::new(inner),
        })
    }

    /// Applies `batch` atomically: once this returns `Ok`, reads see all of it.
    ///
    /// Fails with [`Error::EntryTooLarge`](crate::Error::EntryTooLarge), applying
    /// nothing, if a put's key plus value exceeds
    /// [`Options::max_entry_size`](crate::Options::max_entry_size). The batch is
    /// left as it was, ready to be cleared and reused.
    pub async fn write(&self, batch: &mut WriteBatch) -> Result<()> {
        batch.prepare(self.inner.max_entry)?;
        self.inner.tree.borrow_mut().apply(batch);
        Ok(())
    }

    /// Finds the entry matching `key` under `cmp` and calls `f` with its key and
    /// value, returning whatever `f` returns. The slices are only valid inside `f`,
    /// which is how `get` avoids allocating: decode or copy what you need there.
    ///
    /// ```ignore
    /// // Location of entry `e` in ledger `l`.
    /// let loc: Option<Location> = db
    ///     .get(&key(l, e), Comparison::Equal, |_k, v| Location::decode(v))
    ///     .await?;
    ///
    /// // Last entry of ledger `l`: the largest key <= (l, u64::MAX), if it is in `l`.
    /// let last: Option<u64> = db
    ///     .get(&key(l, u64::MAX), Comparison::Floor, |k, _v| decode_key(k))
    ///     .await?
    ///     .and_then(|(ledger, entry)| (ledger == l).then_some(entry));
    /// ```
    pub async fn get<R>(
        &self,
        key: &[u8],
        cmp: Comparison,
        f: impl FnOnce(&[u8], &[u8]) -> R,
    ) -> Result<Option<R>> {
        let mut found_key = self.inner.take_buffer();
        let value = {
            let tree = self.inner.tree.borrow();
            let mut cursor = Cursor::new();
            match tree.root() {
                Some(root) if tree.seek(root, key, cmp, &mut cursor) => {
                    let (frame, i) = cursor.leaf();
                    let node = tree.node(frame);
                    node.key_into(i, &mut found_key);
                    let (offset, len) = node.value_span(i);
                    Some((tree.addr(frame, offset), len))
                }
                _ => None,
            }
        };
        // SAFETY: the frame belongs to the current version, which nothing can replace
        // while `f`, a synchronous closure, runs.
        let result =
            value.map(|(ptr, len)| f(&found_key, unsafe { slice::from_raw_parts(ptr, len) }));
        self.inner.put_buffer(found_key);
        Ok(result)
    }

    /// Iterates over the entries with keys in `range`, in key order, as of now:
    /// later writes don't affect it. Creating and advancing it never allocates.
    ///
    /// An iterator keeps its version, including entries later writes replace or
    /// delete, in memory until it is dropped, so keep iterators short-lived.
    pub fn range(&self, range: impl RangeBounds<[u8]>) -> Iter {
        let mut bounds = self.inner.take_buffer();
        let start = copy_bound(range.start_bound(), &mut bounds);
        let start_len = bounds.len();
        let end = copy_bound(range.end_bound(), &mut bounds);
        let root = self.inner.tree.borrow_mut().pin();
        Iter {
            inner: self.inner.clone(),
            root,
            cursor: Cursor::new(),
            state: State::Unstarted,
            start,
            end,
            start_len,
            bounds,
            key: self.inner.take_buffer(),
        }
    }

    /// Closes the store.
    pub async fn close(self) -> Result<()> {
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn tree(&self) -> std::cell::Ref<'_, Tree> {
        self.inner.tree.borrow()
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

/// A point-in-time iterator over a range of entries, from [`Db::range`].
pub struct Iter {
    inner: Rc<Inner>,
    /// The pinned root of the version being iterated.
    root: Option<FrameId>,
    cursor: Cursor,
    state: State,
    start: Bound<()>,
    end: Bound<()>,
    /// `bounds` holds the start key, then the end key.
    start_len: usize,
    bounds: Vec<u8>,
    key: Vec<u8>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Unstarted,
    Active,
    Done,
}

impl Iter {
    /// Lends the next entry's key and value, valid until the next call.
    pub async fn next(&mut self) -> Result<Option<(&[u8], &[u8])>> {
        let Some(root) = self.root else {
            return Ok(None);
        };
        let tree = self.inner.tree.borrow();
        let found = match self.state {
            State::Done => false,
            State::Active => tree.step(&mut self.cursor, true),
            State::Unstarted => {
                let start = &self.bounds[..self.start_len];
                match self.start {
                    Bound::Included(()) => {
                        tree.seek(root, start, crate::Comparison::Ceiling, &mut self.cursor)
                    }
                    Bound::Excluded(()) => {
                        tree.seek(root, start, crate::Comparison::Higher, &mut self.cursor)
                    }
                    Bound::Unbounded => {
                        tree.seek_first(root, &mut self.cursor);
                        true
                    }
                }
            }
        };
        if !found {
            self.state = State::Done;
            return Ok(None);
        }
        let (frame, i) = self.cursor.leaf();
        let node = tree.node(frame);
        node.key_into(i, &mut self.key);
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
        self.state = State::Active;
        let (offset, len) = node.value_span(i);
        let value = tree.addr(frame, offset);
        drop(tree);
        // SAFETY: the pinned root keeps this version's frames allocated, and frames
        // are never modified while referenced.
        Ok(Some((&self.key, unsafe {
            slice::from_raw_parts(value, len)
        })))
    }
}

impl Drop for Iter {
    fn drop(&mut self) {
        if let Some(root) = self.root.take() {
            self.inner.tree.borrow_mut().release(root);
        }
        self.inner.put_buffer(mem::take(&mut self.bounds));
        self.inner.put_buffer(mem::take(&mut self.key));
    }
}
