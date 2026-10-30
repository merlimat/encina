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

//! A simulated disk for tests, which implements [`Disk`] as a caller's disk does.
//! It can fail any single operation, or crash during one: then only what was synced
//! survives, plus whichever pages written since happen to have landed, with the
//! write in progress torn. It can also hold reads in flight, or syncs and cuts,
//! until a test releases them, and delay writes, syncs and reads while the
//! runtime runs other tasks, as a real disk's take time.
//!
//! It runs no task: the tests drive the scan that follows an open themselves.

use std::cell::RefCell;
use std::future::{Future, poll_fn};
use std::io;
use std::pin::Pin;
use std::rc::Rc;
use std::slice;
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use crate::disk::{Disk, ReadBuf, WriteBuf};
use crate::options::MIN_PAGE_SIZE;
use crate::pool::{Holds, Pool};
use crate::{Db, ReadOnlyDb};

/// A store on a simulated disk, and a view of one.
pub(crate) type SimDb = Db<Rc<SimDisk>>;
pub(crate) type SimView = ReadOnlyDb<Rc<SimDisk>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fault {
    /// The operation fails. Later ones work.
    Error,
    /// The machine crashes during the operation. It and every later operation
    /// fail; only a restart sees the disk again.
    Crash,
    /// The operation works, but returns other buffers than it was given, and keeps
    /// those until [`let_go`](SimDisk::let_go): a read returns one it fills, a
    /// write one of its own, or none. A cut or a sync has none to swap.
    Swap,
}

pub(crate) struct SimDisk(RefCell<State>);

#[derive(Default)]
struct State {
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
    /// While holding syncs and cuts, how many more may start before the rest wait.
    syncs_allowed: Option<u64>,
    /// Reads, syncs and cuts waiting for a release.
    waiting: Vec<Waker>,
    /// The buffers of held reads that were cancelled, with their operation
    /// numbers. Like a kernel, the disk keeps them until the read finishes.
    cancelled: Vec<(u64, ReadBuf)>,
    /// What the disk holds from an operation number on, as if another process
    /// wrote it meanwhile.
    overwrite: Option<(u64, Vec<u8>)>,
    /// The buffers that operations which swapped them kept.
    kept: (Vec<ReadBuf>, Vec<WriteBuf>),
    /// Whether writes and syncs first let the runtime run other tasks, and how many
    /// have finished. Syncs then keep nothing for a crash.
    delay: bool,
    finished: u64,
    /// How long each leaf read takes, if the disk delays reads.
    read_time: Option<Duration>,
}

impl SimDisk {
    pub(crate) fn new() -> Rc<SimDisk> {
        Rc::new(SimDisk(RefCell::new(State::default())))
    }

    pub(crate) fn len(&self) -> u64 {
        self.0.borrow().current.len() as u64
    }

    /// The `len` bytes at `offset` the disk holds now, read without counting an
    /// operation.
    pub(crate) fn peek(&self, offset: u64, len: usize) -> Vec<u8> {
        let disk = self.0.borrow();
        let at = offset as usize;
        disk.current[at..at + len].to_vec()
    }

    /// What reads see, byte for byte.
    pub(crate) fn contents(&self) -> Vec<u8> {
        self.0.borrow().current.clone()
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

    /// Sets the bytes at `offset` to `data`, growing the disk if needed, as if
    /// they had been written and synced: for files a store wouldn't write.
    pub(crate) fn set_bytes(&self, offset: u64, data: &[u8]) {
        let mut disk = self.0.borrow_mut();
        land(&mut disk.current, offset as usize, data);
        land(&mut disk.durable, offset as usize, data);
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
        let disk = State {
            current: image.clone(),
            durable: image,
            ..State::default()
        };
        Rc::new(SimDisk(RefCell::new(disk)))
    }

    /// Lets go of the buffers that operations which swapped them kept.
    pub(crate) fn let_go(&self) {
        self.0.borrow_mut().kept = Default::default();
    }

    /// Counts an operation and applies a fault planned for it. `write` is where
    /// the operation writes and what, if it does, for tearing. Returns whether the
    /// operation swaps its buffers.
    fn begin(&self, write: Option<(usize, &[WriteBuf])>) -> io::Result<bool> {
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
            Some((at, Fault::Swap)) if at == op => Ok(true),
            Some((at, fault)) if at == op => {
                if fault == Fault::Crash {
                    disk.crash(op, write);
                }
                Err(io::Error::other(format!(
                    "simulated {fault:?} at operation {op}"
                )))
            }
            _ => Ok(false),
        }
    }

    /// Copies what the disk holds at `offset` into `out`.
    fn fill(&self, out: &mut [u8], offset: u64) -> io::Result<()> {
        let disk = self.0.borrow();
        let at = offset as usize;
        let data = disk.current.get(at..at + out.len());
        out.copy_from_slice(data.ok_or(io::ErrorKind::UnexpectedEof)?);
        Ok(())
    }

    /// Makes the reads started from now on wait until released.
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

    /// Makes syncs and cuts wait before they start, until released.
    pub(crate) fn hold_syncs(&self) {
        self.0.borrow_mut().syncs_allowed = Some(0);
    }

    /// Lets the next `n` syncs or cuts held start, and keeps holding the rest.
    pub(crate) fn allow_syncs(&self, n: u64) {
        let mut disk = self.0.borrow_mut();
        disk.syncs_allowed = disk.syncs_allowed.map(|allowed| allowed + n);
        for waker in disk.waiting.drain(..) {
            waker.wake();
        }
    }

    /// Lets every sync and cut start, and stops holding them.
    pub(crate) fn release_syncs(&self) {
        let mut disk = self.0.borrow_mut();
        disk.syncs_allowed = None;
        for waker in disk.waiting.drain(..) {
            waker.wake();
        }
    }

    /// Makes each write and sync from now on first let the runtime run other
    /// tasks, as the thread would while a real disk worked, so that a test can time
    /// what the thread does. Syncs then keep no copy of what survives a crash: the
    /// disk mustn't crash.
    pub(crate) fn delay_writes(&self) {
        self.0.borrow_mut().delay = true;
    }

    /// Makes each read from now on take `time`, while the runtime runs other
    /// tasks, as the thread would while a real disk read the page.
    pub(crate) fn delay_reads(&self, time: Duration) {
        self.0.borrow_mut().read_time = Some(time);
    }

    /// Waits for the operation's turn, if writes and syncs are delayed: until the
    /// runtime has run other tasks, with no other one finishing since it last
    /// looked. One finishes per turn, as a disk works through its queue, so that the
    /// simulated disk copies the pages of one request at a time, which takes the
    /// thread only briefly.
    async fn delayed(&self) {
        if !self.0.borrow().delay {
            return;
        }
        // How many had finished when it last looked.
        let mut seen = None;
        poll_fn(|cx| {
            let mut disk = self.0.borrow_mut();
            if seen == Some(disk.finished) {
                disk.finished += 1;
                return Poll::Ready(());
            }
            seen = Some(disk.finished);
            cx.waker().wake_by_ref();
            Poll::Pending
        })
        .await;
    }

    /// Waits while syncs and cuts are held.
    async fn unheld(&self) {
        poll_fn(|cx| {
            let mut disk = self.0.borrow_mut();
            match disk.syncs_allowed {
                Some(0) => {
                    disk.waiting.push(cx.waker().clone());
                    Poll::Pending
                }
                Some(allowed) => {
                    disk.syncs_allowed = Some(allowed - 1);
                    Poll::Ready(())
                }
                None => Poll::Ready(()),
            }
        })
        .await;
    }

    /// Starts a read: returns its operation number and outcome.
    fn start_read(&self) -> (u64, io::Result<bool>) {
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
    fn cancel(&self, op: u64, buf: ReadBuf) {
        let mut disk = self.0.borrow_mut();
        if disk.is_held(op) {
            disk.cancelled.push((op, buf));
        }
    }

    /// Writes `bufs` back to back from `offset`. Returns whether the write swaps
    /// its buffers.
    fn write_pages(&self, bufs: &[WriteBuf], offset: u64) -> io::Result<bool> {
        let mut at = offset as usize;
        let swap = self.begin(Some((at, bufs)))?;
        let mut disk = self.0.borrow_mut();
        disk.writes += 1;
        for buf in bufs {
            land(&mut disk.current, at, buf);
            disk.unsynced.push((at, buf.len()));
            disk.written += buf.len() as u64;
            at += buf.len();
        }
        Ok(swap)
    }
}

impl Disk for Rc<SimDisk> {
    type Class = ();
    type Read = SimRead;

    async fn len(&self) -> io::Result<u64> {
        Ok(SimDisk::len(self))
    }

    fn min_read(&self) -> usize {
        MIN_PAGE_SIZE
    }

    fn read(&self, buf: ReadBuf, offset: u64, _: ()) -> SimRead {
        SimRead::new(self.clone(), buf, offset)
    }

    async fn write(&self, buf: WriteBuf, offset: u64, _: ()) -> (io::Result<()>, WriteBuf) {
        self.delayed().await;
        match self.write_pages(slice::from_ref(&buf), offset) {
            Ok(true) => {
                // Another buffer as long, in memory of its own.
                let mut pool = Pool::new(buf.len());
                let frame = pool.alloc(Holds::Loading, 0);
                let other = WriteBuf::new(pool.buf(frame));
                self.0.borrow_mut().kept.1.push(buf);
                (Ok(()), other)
            }
            res => (res.map(drop), buf),
        }
    }

    /// Writes the pages in `bufs` back to back from `offset`, in one operation.
    /// Until a sync, each page lands, or not, on its own.
    async fn write_vectored(
        &self,
        bufs: Vec<WriteBuf>,
        offset: u64,
        _: (),
    ) -> (io::Result<()>, Vec<WriteBuf>) {
        self.delayed().await;
        match self.write_pages(&bufs, offset) {
            Ok(true) => {
                self.0.borrow_mut().kept.1.extend(bufs);
                (Ok(()), Vec::new())
            }
            res => (res.map(drop), bufs),
        }
    }

    /// Cuts the disk at `len` bytes. Until a sync, a crash may undo it.
    async fn set_len(&self, len: u64, _: ()) -> io::Result<()> {
        self.unheld().await;
        self.begin(None)?;
        self.0.borrow_mut().current.resize(len as usize, 0);
        Ok(())
    }

    async fn sync(&self, _: ()) -> io::Result<()> {
        self.unheld().await;
        self.delayed().await;
        self.begin(None)?;
        let mut disk = self.0.borrow_mut();
        let State {
            current,
            durable,
            unsynced,
            delay,
            ..
        } = &mut *disk;
        if *delay {
            unsynced.clear();
            return Ok(());
        }
        durable.resize(current.len(), 0);
        for (at, len) in unsynced.drain(..) {
            durable[at..at + len].copy_from_slice(&current[at..at + len]);
        }
        Ok(())
    }

    /// Drops `task`: the tests drive the scan that follows an open.
    fn spawn(&self, task: impl Future<Output = ()> + 'static) {
        drop(task);
    }
}

/// A read on the simulated disk. It finishes when first polled, unless the disk
/// holds or delays it.
pub(crate) struct SimRead {
    disk: Rc<SimDisk>,
    offset: u64,
    buf: Option<ReadBuf>,
    /// The read's operation number and outcome, once started: whether it swaps
    /// its buffer, if it works.
    started: Option<(u64, io::Result<bool>)>,
    /// When it finishes, if the disk delays it.
    done: Option<Instant>,
}

impl SimRead {
    fn new(disk: Rc<SimDisk>, buf: ReadBuf, offset: u64) -> SimRead {
        SimRead {
            disk,
            offset,
            buf: Some(buf),
            started: None,
            done: None,
        }
    }
}

impl Future for SimRead {
    type Output = (io::Result<()>, ReadBuf);

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = &mut *self;
        let disk = &this.disk;
        let (op, _) = this.started.get_or_insert_with(|| disk.start_read());
        if disk.holds(*op, cx.waker()) {
            return Poll::Pending;
        }
        if let Some(time) = disk.0.borrow().read_time {
            let done = *this.done.get_or_insert_with(|| Instant::now() + time);
            if Instant::now() < done {
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }
        }
        let (_, res) = this.started.take().unwrap();
        let mut buf = this.buf.take().unwrap();
        if let Ok(true) = res {
            // Another buffer as long, a frame of a pool of its own.
            let mut pool = Pool::new(MIN_PAGE_SIZE);
            let frame = pool.alloc_large(Holds::Loading, 0, buf.len() / MIN_PAGE_SIZE);
            let mut other = ReadBuf::frame(pool.buf(frame));
            let res = this.disk.fill(&mut other, this.offset);
            this.disk.0.borrow_mut().kept.0.push(buf);
            return Poll::Ready((res, other));
        }
        let res = res.and_then(|_| this.disk.fill(&mut buf, this.offset));
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

impl State {
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
    fn crash(&mut self, op: u64, write: Option<(usize, &[WriteBuf])>) {
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
            let mut left = bufs.iter().map(|b| b.len()).sum::<usize>() / 2;
            for buf in bufs {
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
