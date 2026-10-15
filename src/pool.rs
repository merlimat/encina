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

use std::alloc::{self, Layout};
use std::cell::{Cell, RefCell};
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::rc::Rc;
use std::slice;

use compio::buf::{IoBuf, IoBufMut, SetLen};

pub(crate) type FrameId = u32;

const CHUNK_FRAMES: usize = 64;
const ALIGN: usize = 4096;

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
}

struct Chunk {
    data: NonNull<u8>,
    /// Owns the reference counts, which `Pool` and `FrameBuf` reach by pointer.
    _refs: Box<[Cell<u32>]>,
}

/// The frames' memory, shared by the pool and in-flight I/O buffers.
struct Memory {
    layout: Layout,
    chunks: RefCell<Vec<Chunk>>,
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
}

impl Pool {
    pub(crate) fn new(page_size: usize) -> Pool {
        let layout = Layout::from_size_align(page_size * CHUNK_FRAMES, ALIGN).unwrap();
        Pool {
            mem: Rc::new(Memory {
                layout,
                chunks: RefCell::new(Vec::new()),
            }),
            page_size,
            chunks: Vec::new(),
            frames: Vec::new(),
            free: Vec::new(),
        }
    }

    /// Frames allocated so far, in use or free.
    pub(crate) fn len(&self) -> usize {
        self.frames.len()
    }

    /// Frames in use.
    pub(crate) fn in_use(&self) -> usize {
        self.frames.len() - self.free.len()
    }

    /// A free frame for `holds`, holding one reference.
    pub(crate) fn alloc(&mut self, holds: Holds, page: u64) -> FrameId {
        if self.free.is_empty() {
            self.grow();
        }
        let id = self.free.pop().unwrap();
        debug_assert_eq!(self.refs(id), 0);
        self.refs_cell(id).set(1);
        self.frames[id as usize] = Frame {
            holds,
            page,
            dirty: false,
            clock: true,
        };
        id
    }

    fn grow(&mut self) {
        let layout = self.mem.layout;
        // SAFETY: the layout has a non-zero size.
        let data = unsafe { alloc::alloc_zeroed(layout) };
        let data = NonNull::new(data).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        let refs: Box<[Cell<u32>]> = (0..CHUNK_FRAMES).map(|_| Cell::new(0)).collect();
        let refs_ptr = NonNull::from(&refs[0]);
        self.mem
            .chunks
            .borrow_mut()
            .push(Chunk { data, _refs: refs });
        self.chunks.push((data, refs_ptr));
        let first = self.frames.len();
        let blank = Frame {
            holds: Holds::Nothing,
            page: 0,
            dirty: false,
            clock: false,
        };
        self.frames.resize(first + CHUNK_FRAMES, blank);
        // Room for every frame, so that freeing never allocates.
        self.free.reserve(self.frames.len() - self.free.len());
        self.free
            .extend((first..first + CHUNK_FRAMES).rev().map(|f| f as FrameId));
    }

    pub(crate) fn frame(&self, id: FrameId) -> &Frame {
        &self.frames[id as usize]
    }

    pub(crate) fn frame_mut(&mut self, id: FrameId) -> &mut Frame {
        &mut self.frames[id as usize]
    }

    fn refs_cell(&self, id: FrameId) -> &Cell<u32> {
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

    /// Returns an unreferenced frame to the free list.
    pub(crate) fn free(&mut self, id: FrameId) {
        debug_assert_eq!(self.refs(id), 0, "freeing a referenced frame");
        self.frames[id as usize].holds = Holds::Nothing;
        self.free.push(id);
    }

    fn ptr(&self, id: FrameId) -> *mut u8 {
        let id = id as usize;
        // SAFETY: frames are carved out of their chunk at `page_size` strides.
        unsafe {
            self.chunks[id / CHUNK_FRAMES]
                .0
                .as_ptr()
                .add((id % CHUNK_FRAMES) * self.page_size)
        }
    }

    /// Address of byte `offset` of a frame.
    pub(crate) fn addr(&self, id: FrameId, offset: usize) -> *const u8 {
        debug_assert!(offset <= self.page_size);
        self.ptr(id).wrapping_add(offset)
    }

    pub(crate) fn page(&self, id: FrameId) -> &[u8] {
        // SAFETY: frames are only written through `page_mut`, which needs `&mut self`,
        // or by I/O into frames nothing else is looking at.
        unsafe { slice::from_raw_parts(self.ptr(id), self.page_size) }
    }

    /// Like [`page`](Self::page), without tying the slice to a borrow of the pool.
    ///
    /// # Safety
    ///
    /// The frame must stay allocated and unmodified for `'a`.
    pub(crate) unsafe fn page_unbounded<'a>(&self, id: FrameId) -> &'a [u8] {
        // SAFETY: guaranteed by the caller.
        unsafe { slice::from_raw_parts(self.ptr(id), self.page_size) }
    }

    /// The page of a frame no reader can see: one this write just built, or one
    /// still loading.
    pub(crate) fn page_mut(&mut self, id: FrameId) -> &mut [u8] {
        // SAFETY: `&mut self` excludes bounded readers, and the frame is unpublished.
        unsafe { slice::from_raw_parts_mut(self.ptr(id), self.page_size) }
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
            data: NonNull::new(self.ptr(id)).unwrap(),
            len: self.page_size,
            refs: NonNull::from(self.refs_cell(id)),
        }
    }
}

/// An I/O buffer over one frame. It keeps the frame memory alive, and holds a
/// reference to the frame, until the runtime drops it, which may be after the
/// future that started the operation is gone.
pub(crate) struct FrameBuf {
    _mem: Rc<Memory>,
    data: NonNull<u8>,
    len: usize,
    refs: NonNull<Cell<u32>>,
}

impl FrameBuf {
    #[cfg(test)]
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
        // A frame is always a full page.
    }
}

impl IoBufMut for FrameBuf {
    fn as_uninit(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY: as in `as_init`.
        unsafe { slice::from_raw_parts_mut(self.data.as_ptr().cast(), self.len) }
    }
}
