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

//! Page frames: page-sized, 4 KiB-aligned buffers with reference counts.
//!
//! Frames are allocated in chunks that never move. The chunks belong to a shared
//! `Memory` that every in-flight I/O buffer keeps alive, so an operation that
//! outlives its future, like a cancelled read, never touches freed memory. The
//! reference counts live there too, so a buffer can drop its reference wherever
//! the runtime drops it. Frames are the only memory the tree allocates.
//!
//! A value leaf that the cache keeps takes a large frame: one of several pages, in
//! memory of its own, which `Memory` owns as well. Freeing a large frame gives its
//! memory back.

use std::alloc::{self, Layout};
use std::cell::{Cell, RefCell};
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::rc::Rc;
use std::slice;

use compio::buf::{IoBuf, IoBufMut, SetLen};

use crate::metrics::Amount;

pub(crate) type FrameId = u32;

const CHUNK_FRAMES: usize = 64;
const ALIGN: usize = 4096;

/// The bit that marks the ids of large frames.
const LARGE: FrameId = 1 << 31;

/// What a frame holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Holds {
    /// Nothing: the frame is on the free list.
    Nothing,
    /// A page being read, not yet in the cache.
    Loading,
    /// An internal node, which stays resident.
    Internal,
    /// A leaf, in the cache.
    Leaf,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Frame {
    pub(crate) holds: Holds,
    /// The page the frame's node lives at.
    pub(crate) page: u64,
    /// Built by the write in progress and not yet committed.
    pub(crate) dirty: bool,
    /// Used since the cache's clock hand last passed it.
    pub(crate) clock: bool,
    /// The pages it spans: 1, unless it is a large frame.
    pub(crate) pages: u32,
}

struct Chunk {
    data: NonNull<u8>,
    /// Owns the reference counts, which `Pool` and `FrameBuf` reach by pointer.
    _refs: Box<[Cell<u32>]>,
}

/// A large frame's memory.
struct Large {
    data: NonNull<u8>,
    layout: Layout,
    /// Owns the reference count, which `Pool` and `FrameBuf` reach by pointer.
    _refs: Box<Cell<u32>>,
}

impl Drop for Large {
    fn drop(&mut self) {
        // SAFETY: allocated in `Pool::alloc_large` with this layout.
        unsafe { alloc::dealloc(self.data.as_ptr(), self.layout) };
    }
}

/// The frames' memory, shared by the pool and in-flight I/O buffers.
struct Memory {
    layout: Layout,
    chunks: RefCell<Vec<Chunk>>,
    /// The large frames, by index; `None` once freed.
    large: RefCell<Vec<Option<Large>>>,
}

impl Drop for Memory {
    fn drop(&mut self) {
        for chunk in self.chunks.get_mut() {
            // SAFETY: allocated in `Pool::grow` with this layout.
            unsafe { alloc::dealloc(chunk.data.as_ptr(), self.layout) };
        }
    }
}

pub(crate) struct Pool {
    mem: Rc<Memory>,
    page_size: usize,
    /// Each chunk's data and reference counts, copied out of `mem`.
    chunks: Vec<(NonNull<u8>, NonNull<Cell<u32>>)>,
    frames: Vec<Frame>,
    free: Vec<FrameId>,
    /// Each large frame's data and reference count, copied out of `mem`, and the
    /// frame; its indexes free for reuse; and the pages the large frames take.
    large: Vec<(NonNull<u8>, NonNull<Cell<u32>>, Frame)>,
    large_free: Vec<usize>,
    large_pages: usize,
    /// The memory of the frames in use, as [`usage`] sorts them, if the store
    /// records metrics.
    pub(crate) usage: [Amount; 3],
}

impl Pool {
    pub(crate) fn new(page_size: usize) -> Pool {
        let layout = Layout::from_size_align(page_size * CHUNK_FRAMES, ALIGN).unwrap();
        Pool {
            mem: Rc::new(Memory {
                layout,
                chunks: RefCell::new(Vec::new()),
                large: RefCell::new(Vec::new()),
            }),
            page_size,
            chunks: Vec::new(),
            frames: Vec::new(),
            free: Vec::new(),
            large: Vec::new(),
            large_free: Vec::new(),
            large_pages: 0,
            usage: Default::default(),
        }
    }

    pub(crate) fn page_size(&self) -> usize {
        self.page_size
    }

    /// Pages in use: the page frames in use, and the large frames' pages.
    pub(crate) fn in_use(&self) -> usize {
        self.frames.len() - self.free.len() + self.large_pages
    }

    /// The pages the large frames take.
    pub(crate) fn large_pages(&self) -> usize {
        self.large_pages
    }

    /// How many frame ids [`slot`](Self::slot) maps, the free ones included.
    pub(crate) fn slots(&self) -> usize {
        self.frames.len() + self.large.len()
    }

    /// The `i`th frame id: the page frames', then the large frames'.
    pub(crate) fn slot(&self, i: usize) -> FrameId {
        match i.checked_sub(self.frames.len()) {
            None => i as FrameId,
            Some(i) => self.large_slot(i),
        }
    }

    /// How many large frame ids [`large_slot`](Self::large_slot) maps, the free ones
    /// included.
    pub(crate) fn large_slots(&self) -> usize {
        self.large.len()
    }

    /// The `i`th large frame id.
    pub(crate) fn large_slot(&self, i: usize) -> FrameId {
        LARGE | i as FrameId
    }

    /// A free frame for `holds`, holding one reference.
    pub(crate) fn alloc(&mut self, holds: Holds, page: u64) -> FrameId {
        if self.free.is_empty() {
            self.grow();
        }
        let id = self.free.pop().unwrap();
        debug_assert_eq!(self.refs(id), 0);
        self.refs_cell(id).set(1);
        self.usage[usage(holds, false)].add(self.page_size as i64);
        self.frames[id as usize] = Frame {
            holds,
            page,
            dirty: false,
            clock: true,
            pages: 1,
        };
        id
    }

    /// A large frame of `pages` pages for `holds`, holding one reference.
    pub(crate) fn alloc_large(&mut self, holds: Holds, page: u64, pages: usize) -> FrameId {
        let layout = Layout::from_size_align(pages * self.page_size, ALIGN).unwrap();
        // SAFETY: the layout has a non-zero size.
        let data = unsafe { alloc::alloc_zeroed(layout) };
        let data = NonNull::new(data).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        let refs = Box::new(Cell::new(1));
        let frame = Frame {
            holds,
            page,
            dirty: false,
            clock: true,
            pages: pages as u32,
        };
        let mut large = self.mem.large.borrow_mut();
        let i = self.large_free.pop().unwrap_or_else(|| {
            large.push(None);
            self.large.push((data, NonNull::dangling(), frame));
            large.len() - 1
        });
        // The pointer to the count is taken once its box has moved into place.
        let refs = NonNull::from(
            &*large[i]
                .insert(Large {
                    data,
                    layout,
                    _refs: refs,
                })
                ._refs,
        );
        self.large[i] = (data, refs, frame);
        self.large_pages += pages;
        self.usage[usage(holds, true)].add((pages * self.page_size) as i64);
        LARGE | i as FrameId
    }

    fn grow(&mut self) {
        let layout = self.mem.layout;
        // SAFETY: the layout has a non-zero size.
        let data = unsafe { alloc::alloc_zeroed(layout) };
        let data = NonNull::new(data).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        let refs: Box<[Cell<u32>]> = (0..CHUNK_FRAMES).map(|_| Cell::new(0)).collect();
        let mut chunks = self.mem.chunks.borrow_mut();
        chunks.push(Chunk { data, _refs: refs });
        // A pointer to all the counts, taken once the box has moved into place:
        // moving a box invalidates the pointers taken from it.
        let refs = NonNull::from(&chunks.last().unwrap()._refs[..]).cast();
        drop(chunks);
        self.chunks.push((data, refs));
        let first = self.frames.len();
        let blank = Frame {
            holds: Holds::Nothing,
            page: 0,
            dirty: false,
            clock: false,
            pages: 1,
        };
        self.frames.resize(first + CHUNK_FRAMES, blank);
        // Room for every frame, so that freeing never allocates.
        self.free.reserve(self.frames.len() - self.free.len());
        self.free
            .extend((first..first + CHUNK_FRAMES).rev().map(|f| f as FrameId));
    }

    pub(crate) fn frame(&self, id: FrameId) -> &Frame {
        match large_index(id) {
            Some(i) => &self.large[i].2,
            None => &self.frames[id as usize],
        }
    }

    pub(crate) fn frame_mut(&mut self, id: FrameId) -> &mut Frame {
        match large_index(id) {
            Some(i) => &mut self.large[i].2,
            None => &mut self.frames[id as usize],
        }
    }

    fn refs_cell(&self, id: FrameId) -> &Cell<u32> {
        if let Some(i) = large_index(id) {
            debug_assert_ne!(
                self.large[i].2.holds,
                Holds::Nothing,
                "large frame {id} freed"
            );
            // SAFETY: the count lives as long as the frame, which is allocated.
            return unsafe { self.large[i].1.as_ref() };
        }
        let id = id as usize;
        // SAFETY: each chunk has CHUNK_FRAMES counts, alive as long as `mem`.
        unsafe {
            self.chunks[id / CHUNK_FRAMES]
                .1
                .add(id % CHUNK_FRAMES)
                .as_ref()
        }
    }

    pub(crate) fn refs(&self, id: FrameId) -> u32 {
        self.refs_cell(id).get()
    }

    pub(crate) fn retain(&self, id: FrameId) {
        let refs = self.refs_cell(id);
        refs.set(refs.get() + 1);
    }

    /// Drops one reference, returning whether it was the last.
    pub(crate) fn unref(&self, id: FrameId) -> bool {
        let refs = self.refs_cell(id);
        debug_assert!(refs.get() > 0, "frame {id} released too often");
        refs.set(refs.get() - 1);
        refs.get() == 0
    }

    /// Returns an unreferenced frame to the free list, or a large frame's memory
    /// to the allocator.
    pub(crate) fn free(&mut self, id: FrameId) {
        debug_assert_eq!(self.refs(id), 0, "freeing a referenced frame");
        if let Some(i) = large_index(id) {
            let frame = &mut self.large[i].2;
            let len = frame.pages as usize * self.page_size;
            self.usage[usage(frame.holds, true)].add(-(len as i64));
            frame.holds = Holds::Nothing;
            self.large_pages -= frame.pages as usize;
            self.mem.large.borrow_mut()[i] = None;
            self.large_free.push(i);
            return;
        }
        let frame = &mut self.frames[id as usize];
        self.usage[usage(frame.holds, false)].add(-(self.page_size as i64));
        frame.holds = Holds::Nothing;
        self.free.push(id);
    }

    fn ptr(&self, id: FrameId) -> *mut u8 {
        if let Some(i) = large_index(id) {
            return self.large[i].0.as_ptr();
        }
        let id = id as usize;
        // SAFETY: frames are carved out of their chunk at `page_size` strides.
        unsafe {
            self.chunks[id / CHUNK_FRAMES]
                .0
                .as_ptr()
                .add((id % CHUNK_FRAMES) * self.page_size)
        }
    }

    /// The bytes a frame spans: a page, or a large frame's pages.
    fn len_of(&self, id: FrameId) -> usize {
        self.frame(id).pages as usize * self.page_size
    }

    /// Address of byte `offset` of a frame.
    pub(crate) fn addr(&self, id: FrameId, offset: usize) -> *const u8 {
        debug_assert!(offset <= self.len_of(id));
        self.ptr(id).wrapping_add(offset)
    }

    /// The frame's page, or a large frame's pages.
    pub(crate) fn page(&self, id: FrameId) -> &[u8] {
        // SAFETY: frames are only written through `page_mut`, which needs `&mut self`,
        // or by I/O into frames nothing else is looking at.
        unsafe { slice::from_raw_parts(self.ptr(id), self.len_of(id)) }
    }

    /// Like [`page`](Self::page), without tying the slice to a borrow of the pool.
    ///
    /// # Safety
    ///
    /// The frame must stay allocated and unmodified for `'a`.
    pub(crate) unsafe fn page_unbounded<'a>(&self, id: FrameId) -> &'a [u8] {
        // SAFETY: guaranteed by the caller.
        unsafe { slice::from_raw_parts(self.ptr(id), self.len_of(id)) }
    }

    /// The page of a frame no reader can see: one this write just built, or one
    /// still loading.
    pub(crate) fn page_mut(&mut self, id: FrameId) -> &mut [u8] {
        // SAFETY: `&mut self` excludes bounded readers, and the frame is unpublished.
        unsafe { slice::from_raw_parts_mut(self.ptr(id), self.len_of(id)) }
    }

    /// An I/O buffer over a frame, which holds a reference to it until dropped.
    pub(crate) fn buf(&self, id: FrameId) -> FrameBuf {
        self.retain(id);
        self.adopt_buf(id)
    }

    /// An I/O buffer over a frame that takes over a reference the caller holds.
    pub(crate) fn adopt_buf(&self, id: FrameId) -> FrameBuf {
        FrameBuf {
            _mem: self.mem.clone(),
            id,
            data: NonNull::new(self.ptr(id)).unwrap(),
            len: self.len_of(id),
            refs: NonNull::from(self.refs_cell(id)),
        }
    }
}

/// Whether `id` is a large frame's.
fn is_large(id: FrameId) -> bool {
    id & LARGE != 0
}

/// The index of the large frame `id`, if it is one.
fn large_index(id: FrameId) -> Option<usize> {
    is_large(id).then_some((id & !LARGE) as usize)
}

/// Which of [`Pool::usage`] counts a frame that holds `holds`: internal nodes,
/// leaves and the pages being read or written, or the large frames of value leaves.
fn usage(holds: Holds, large: bool) -> usize {
    match holds {
        _ if large => 2,
        Holds::Internal => 0,
        _ => 1,
    }
}

/// An I/O buffer over one frame. It keeps the frame memory alive, and holds a
/// reference to the frame, until the runtime drops it, which may be after the
/// future that started the operation is gone.
pub(crate) struct FrameBuf {
    _mem: Rc<Memory>,
    id: FrameId,
    data: NonNull<u8>,
    len: usize,
    refs: NonNull<Cell<u32>>,
}

impl FrameBuf {
    /// The frame it is over.
    pub(crate) fn id(&self) -> FrameId {
        self.id
    }

    #[cfg(any(test, fuzzing))]
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: frame memory is initialized (zeroed when allocated).
        unsafe { slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }
}

impl Drop for FrameBuf {
    fn drop(&mut self) {
        // SAFETY: `_mem` keeps the count alive.
        let refs = unsafe { self.refs.as_ref() };
        refs.set(refs.get() - 1);
    }
}

impl IoBuf for FrameBuf {
    fn as_init(&self) -> &[u8] {
        // SAFETY: frame memory is initialized, and `_mem` keeps it alive.
        unsafe { slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }
}

impl SetLen for FrameBuf {
    unsafe fn set_len(&mut self, _len: usize) {
        // A frame is always full: a page, or a large frame's pages.
    }
}

impl IoBufMut for FrameBuf {
    fn as_uninit(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY: as in `as_init`.
        unsafe { slice::from_raw_parts_mut(self.data.as_ptr().cast(), self.len) }
    }
}

/// A 4 KiB-aligned buffer that a value leaf is read into. Value leaves aren't
/// cached, so their buffers aren't frames. A buffer can be reused for a value leaf
/// no larger.
pub(crate) struct ValueBuf {
    data: NonNull<u8>,
    len: usize,
    cap: usize,
}

impl ValueBuf {
    /// A buffer of `len` bytes, `old` if it holds that many.
    pub(crate) fn new(old: Option<ValueBuf>, len: usize) -> ValueBuf {
        if let Some(mut buf) = old.filter(|buf| buf.cap >= len) {
            buf.len = len;
            return buf;
        }
        let cap = len.max(1);
        let layout = Layout::from_size_align(cap, ALIGN).unwrap();
        // SAFETY: the layout has a non-zero size.
        let data = unsafe { alloc::alloc_zeroed(layout) };
        let data = NonNull::new(data).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        ValueBuf { data, len, cap }
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        // SAFETY: the memory is initialized (zeroed when allocated), and owned.
        unsafe { slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }

    #[cfg(any(test, fuzzing))]
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as in `bytes`.
        unsafe { slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }
}

impl Drop for ValueBuf {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.cap, ALIGN).unwrap();
        // SAFETY: allocated in `new` with this layout.
        unsafe { alloc::dealloc(self.data.as_ptr(), layout) };
    }
}

impl IoBuf for ValueBuf {
    fn as_init(&self) -> &[u8] {
        self.bytes()
    }
}

impl SetLen for ValueBuf {
    unsafe fn set_len(&mut self, _len: usize) {
        // Reads fill the whole buffer, which is initialized anyway.
    }
}

impl IoBufMut for ValueBuf {
    fn as_uninit(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY: as in `bytes`.
        unsafe { slice::from_raw_parts_mut(self.data.as_ptr().cast(), self.len) }
    }
}
