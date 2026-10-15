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

//! A simulated disk for tests. It can fail any single operation, or crash during
//! one: then only what was synced survives, plus whichever pages written since
//! happen to have landed, with the write in progress torn. It can also hold leaf
//! reads in flight until a test releases them.

use std::cell::RefCell;
use std::io;
use std::rc::Rc;
use std::slice;
use std::task::{Context, Poll, Waker};

use compio::buf::IoBuf;

use crate::pool::FrameBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fault {
    /// The operation fails. Later ones work.
    Error,
    /// The machine crashes during the operation. It and every later operation
    /// fail; only a restart sees the disk again.
    Crash,
}

pub(crate) struct SimDisk(RefCell<Disk>);

#[derive(Default)]
struct Disk {
    /// What reads see.
    current: Vec<u8>,
    /// What survives a crash.
    durable: Vec<u8>,
    /// Pages written since the last sync: offset and length.
    unsynced: Vec<(usize, usize)>,
    ops: u64,
    /// Write operations, and bytes written.
    writes: u64,
    written: u64,
    fault: Option<(u64, Fault)>,
    /// The disk as a crash left it.
    crashed: Option<Vec<u8>>,
    /// While holding, reads numbered `released` or later wait for a release.
    holding: bool,
    released: u64,
    /// Reads waiting for a release.
    waiting: Vec<Waker>,
    /// The buffers of held reads that were cancelled, with their operation
    /// numbers. Like a kernel, the disk keeps them until the read finishes.
    cancelled: Vec<(u64, FrameBuf)>,
    /// What the disk holds from an operation number on, as if another process
    /// wrote it meanwhile.
    overwrite: Option<(u64, Vec<u8>)>,
}

impl SimDisk {
    pub(crate) fn new() -> Rc<SimDisk> {
        Rc::new(SimDisk(RefCell::new(Disk::default())))
    }

    pub(crate) fn len(&self) -> u64 {
        self.0.borrow().current.len() as u64
    }

    /// Operations started so far.
    pub(crate) fn ops(&self) -> u64 {
        self.0.borrow().ops
    }

    /// Write operations that succeeded so far, and the bytes they wrote.
    pub(crate) fn written(&self) -> (u64, u64) {
        let disk = self.0.borrow();
        (disk.writes, disk.written)
    }

    /// Flips a bit at `offset`, as bit rot would.
    pub(crate) fn flip(&self, offset: u64) {
        let mut disk = self.0.borrow_mut();
        let at = offset as usize;
        disk.current[at] ^= 1;
        disk.durable[at] ^= 1;
    }

    /// Makes operation number `op`, counting from 0, suffer `fault`.
    pub(crate) fn inject(&self, op: u64, fault: Fault) {
        self.0.borrow_mut().fault = Some((op, fault));
    }

    /// Makes the disk hold what `other` holds now from operation number `op` on.
    pub(crate) fn overwrite_at(&self, op: u64, other: &SimDisk) {
        let image = other.0.borrow().current.clone();
        self.0.borrow_mut().overwrite = Some((op, image));
    }

    /// The disk as a restarted machine finds it: after a crash, what the crash
    /// left; otherwise everything written.
    pub(crate) fn restart(&self) -> Rc<SimDisk> {
        let disk = self.0.borrow();
        let image = disk.crashed.clone().unwrap_or_else(|| disk.current.clone());
        let disk = Disk {
            current: image.clone(),
            durable: image,
            ..Disk::default()
        };
        Rc::new(SimDisk(RefCell::new(disk)))
    }

    /// Counts an operation and applies a fault planned for it. `write` is where
    /// the operation writes and what, if it does, for tearing.
    fn begin(&self, write: Option<(usize, &[FrameBuf])>) -> io::Result<()> {
        let mut disk = self.0.borrow_mut();
        if disk.crashed.is_some() {
            return Err(io::Error::other("the simulated machine crashed"));
        }
        let op = disk.ops;
        disk.ops += 1;
        if disk.overwrite.as_ref().is_some_and(|&(at, _)| at == op) {
            let (_, image) = disk.overwrite.take().unwrap();
            disk.durable.clone_from(&image);
            disk.current = image;
        }
        match disk.fault {
            Some((at, fault)) if at == op => {
                if fault == Fault::Crash {
                    disk.crash(op, write);
                }
                Err(io::Error::other(format!(
                    "simulated {fault:?} at operation {op}"
                )))
            }
            _ => Ok(()),
        }
    }

    pub(crate) fn read(&self, mut buf: FrameBuf, offset: u64) -> (io::Result<()>, FrameBuf) {
        let res = self.begin(None).and_then(|()| self.fill(&mut buf, offset));
        (res, buf)
    }

    /// Copies what the disk holds at `offset` into `buf`.
    fn fill(&self, buf: &mut FrameBuf, offset: u64) -> io::Result<()> {
        let disk = self.0.borrow();
        let (at, out) = (offset as usize, buf.bytes_mut());
        let data = disk.current.get(at..at + out.len());
        out.copy_from_slice(data.ok_or(io::ErrorKind::UnexpectedEof)?);
        Ok(())
    }

    /// Makes the leaf reads started from now on wait until released.
    pub(crate) fn hold(&self) {
        let mut disk = self.0.borrow_mut();
        disk.holding = true;
        disk.released = disk.ops;
    }

    /// Finishes the held reads numbered up to `op`.
    pub(crate) fn release_through(&self, op: u64) {
        let mut disk = self.0.borrow_mut();
        disk.released = disk.released.max(op + 1);
        disk.finish_held();
    }

    /// Finishes every held read, and stops holding.
    pub(crate) fn release(&self) {
        let mut disk = self.0.borrow_mut();
        disk.holding = false;
        disk.finish_held();
    }

    /// Starts a leaf read: returns its operation number and outcome.
    fn start_read(&self) -> (u64, io::Result<()>) {
        let op = self.ops();
        (op, self.begin(None))
    }

    /// Whether read `op` is held. If so, `waker` is woken on its release.
    fn holds(&self, op: u64, waker: &Waker) -> bool {
        let mut disk = self.0.borrow_mut();
        let held = disk.is_held(op);
        if held {
            disk.waiting.push(waker.clone());
        }
        held
    }

    /// Takes the buffer of read `op`, cancelled: it is dropped once the read
    /// would have finished.
    fn cancel(&self, op: u64, buf: FrameBuf) {
        let mut disk = self.0.borrow_mut();
        if disk.is_held(op) {
            disk.cancelled.push((op, buf));
        }
    }

    pub(crate) fn write(&self, buf: FrameBuf, offset: u64) -> (io::Result<()>, FrameBuf) {
        let res = self.write_pages(slice::from_ref(&buf), offset);
        (res, buf)
    }

    /// Writes the pages in `bufs` back to back from `offset`, in one operation.
    /// Until a sync, each page lands, or not, on its own.
    pub(crate) fn write_run(
        &self,
        bufs: Vec<FrameBuf>,
        offset: u64,
    ) -> (io::Result<()>, Vec<FrameBuf>) {
        let res = self.write_pages(&bufs, offset);
        (res, bufs)
    }

    fn write_pages(&self, bufs: &[FrameBuf], offset: u64) -> io::Result<()> {
        let mut at = offset as usize;
        self.begin(Some((at, bufs)))?;
        let mut disk = self.0.borrow_mut();
        disk.writes += 1;
        for buf in bufs {
            let data = buf.as_init();
            land(&mut disk.current, at, data);
            disk.unsynced.push((at, data.len()));
            disk.written += data.len() as u64;
            at += data.len();
        }
        Ok(())
    }

    /// Cuts the disk at `len` bytes. Until a sync, a crash may undo it.
    pub(crate) fn set_len(&self, len: u64) -> io::Result<()> {
        self.begin(None)?;
        self.0.borrow_mut().current.resize(len as usize, 0);
        Ok(())
    }

    pub(crate) fn sync(&self) -> io::Result<()> {
        self.begin(None)?;
        let mut disk = self.0.borrow_mut();
        let Disk {
            current,
            durable,
            unsynced,
            ..
        } = &mut *disk;
        durable.resize(current.len(), 0);
        for (at, len) in unsynced.drain(..) {
            durable[at..at + len].copy_from_slice(&current[at..at + len]);
        }
        Ok(())
    }
}

/// A leaf read on the simulated disk. It finishes when first polled, unless the
/// disk holds it.
pub(crate) struct SimRead {
    disk: Rc<SimDisk>,
    offset: u64,
    buf: Option<FrameBuf>,
    /// The read's operation number and outcome, once started.
    started: Option<(u64, io::Result<()>)>,
}

impl SimRead {
    pub(crate) fn new(disk: Rc<SimDisk>, buf: FrameBuf, offset: u64) -> SimRead {
        SimRead {
            disk,
            offset,
            buf: Some(buf),
            started: None,
        }
    }

    pub(crate) fn poll(&mut self, cx: &mut Context<'_>) -> Poll<(io::Result<()>, FrameBuf)> {
        let disk = &self.disk;
        let (op, _) = self.started.get_or_insert_with(|| disk.start_read());
        if disk.holds(*op, cx.waker()) {
            return Poll::Pending;
        }
        let (_, res) = self.started.take().unwrap();
        let mut buf = self.buf.take().unwrap();
        let res = res.and_then(|()| self.disk.fill(&mut buf, self.offset));
        Poll::Ready((res, buf))
    }
}

impl Drop for SimRead {
    fn drop(&mut self) {
        if let (Some((op, _)), Some(buf)) = (&self.started, self.buf.take()) {
            self.disk.cancel(*op, buf);
        }
    }
}

impl Disk {
    fn is_held(&self, op: u64) -> bool {
        self.holding && op >= self.released
    }

    /// Lets the reads no longer held finish: wakes those waiting, and drops the
    /// buffers of the cancelled ones.
    fn finish_held(&mut self) {
        let (holding, released) = (self.holding, self.released);
        self.cancelled.retain(|&(op, _)| holding && op >= released);
        for waker in self.waiting.drain(..) {
            waker.wake();
        }
    }

    /// Records what a crash during operation `op` leaves on disk: what was synced,
    /// plus a pseudo-random subset of the pages written since, plus the first half
    /// of the write in progress, if any.
    fn crash(&mut self, op: u64, write: Option<(usize, &[FrameBuf])>) {
        let mut image = self.durable.clone();
        let mut rng = op.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        for &(at, len) in &self.unsynced {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            if rng & 1 == 1 {
                land(&mut image, at, &self.current[at..at + len]);
            }
        }
        if let Some((mut at, bufs)) = write {
            let mut left = bufs.iter().map(|b| b.as_init().len()).sum::<usize>() / 2;
            for buf in bufs.iter().map(FrameBuf::as_init) {
                if left == 0 {
                    break;
                }
                let landed = left.min(buf.len());
                land(&mut image, at, &buf[..landed]);
                (left, at) = (left - landed, at + buf.len());
            }
        }
        self.crashed = Some(image);
    }
}

/// Copies `data` into `image` at `at`, growing it if needed.
fn land(image: &mut Vec<u8>, at: usize, data: &[u8]) {
    if image.len() < at + data.len() {
        image.resize(at + data.len(), 0);
    }
    image[at..at + data.len()].copy_from_slice(data);
}
