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
//! Frames are allocated in chunks that never move, so a frame's address is stable
//! for the pool's lifetime. This is the only memory the tree allocates.

use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::slice;

pub(crate) type FrameId = u32;

const CHUNK_FRAMES: usize = 64;
const ALIGN: usize = 4096;

pub(crate) struct Pool {
    page_size: usize,
    chunks: Vec<NonNull<u8>>,
    refs: Vec<u32>,
    free: Vec<FrameId>,
}

impl Pool {
    pub(crate) fn new(page_size: usize) -> Pool {
        Pool {
            page_size,
            chunks: Vec::new(),
            refs: Vec::new(),
            free: Vec::new(),
        }
    }

    fn chunk_layout(&self) -> Layout {
        Layout::from_size_align(self.page_size * CHUNK_FRAMES, ALIGN).unwrap()
    }

    /// A free frame, holding one reference.
    pub(crate) fn alloc(&mut self) -> FrameId {
        if self.free.is_empty() {
            self.grow();
        }
        let frame = self.free.pop().unwrap();
        debug_assert_eq!(self.refs[frame as usize], 0);
        self.refs[frame as usize] = 1;
        frame
    }

    fn grow(&mut self) {
        let layout = self.chunk_layout();
        // SAFETY: the layout has a non-zero size.
        let chunk = unsafe { alloc::alloc_zeroed(layout) };
        let chunk = NonNull::new(chunk).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        let first = self.refs.len();
        self.chunks.push(chunk);
        self.refs.resize(first + CHUNK_FRAMES, 0);
        // Room for every frame, so freeing never allocates.
        self.free.reserve(self.refs.len() - self.free.len());
        self.free
            .extend((first..first + CHUNK_FRAMES).rev().map(|f| f as FrameId));
    }

    pub(crate) fn retain(&mut self, frame: FrameId) {
        self.refs[frame as usize] += 1;
    }

    /// Drops one reference. Returns true if it was the last, in which case the caller
    /// must release the frame's children, then [`free`](Self::free) it.
    pub(crate) fn unref(&mut self, frame: FrameId) -> bool {
        let refs = &mut self.refs[frame as usize];
        debug_assert!(*refs > 0, "frame {frame} released too often");
        *refs -= 1;
        *refs == 0
    }

    pub(crate) fn free(&mut self, frame: FrameId) {
        debug_assert_eq!(self.refs[frame as usize], 0);
        self.free.push(frame);
    }

    fn ptr(&self, frame: FrameId) -> *mut u8 {
        let f = frame as usize;
        // SAFETY: frames are carved out of their chunk at `page_size` strides.
        unsafe {
            self.chunks[f / CHUNK_FRAMES]
                .as_ptr()
                .add((f % CHUNK_FRAMES) * self.page_size)
        }
    }

    /// Address of byte `offset` in `frame`.
    pub(crate) fn addr(&self, frame: FrameId, offset: usize) -> *const u8 {
        debug_assert!(offset <= self.page_size);
        self.ptr(frame).wrapping_add(offset)
    }

    pub(crate) fn page(&self, frame: FrameId) -> &[u8] {
        // SAFETY: frames are only written through `page_mut`, which needs `&mut self`.
        unsafe { slice::from_raw_parts(self.ptr(frame), self.page_size) }
    }

    /// Like [`page`](Self::page), without tying the slice to a borrow of the pool.
    ///
    /// # Safety
    ///
    /// The frame must stay allocated and unmodified for `'a`.
    pub(crate) unsafe fn page_unbounded<'a>(&self, frame: FrameId) -> &'a [u8] {
        // SAFETY: guaranteed by the caller.
        unsafe { slice::from_raw_parts(self.ptr(frame), self.page_size) }
    }

    /// The page of a frame this write just allocated. Published frames, which
    /// readers may be looking at, must never be modified.
    pub(crate) fn page_mut(&mut self, frame: FrameId) -> &mut [u8] {
        // SAFETY: `&mut self` excludes bounded readers; the frame is unpublished.
        unsafe { slice::from_raw_parts_mut(self.ptr(frame), self.page_size) }
    }

    #[cfg(test)]
    pub(crate) fn refs(&self, frame: FrameId) -> u32 {
        self.refs[frame as usize]
    }

    /// Frames currently holding a reference.
    #[cfg(test)]
    pub(crate) fn in_use(&self) -> usize {
        self.refs.len() - self.free.len()
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        let layout = self.chunk_layout();
        for chunk in &self.chunks {
            // SAFETY: allocated in `grow` with this layout.
            unsafe { alloc::dealloc(chunk.as_ptr(), layout) };
        }
    }
}
