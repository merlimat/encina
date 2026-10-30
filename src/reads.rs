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

//! Leaf reads in flight, shared by every reader that needs their pages.
//!
//! Each read sits in a slot that counts the readers holding it. Whichever of them
//! is polled drives the read, and the first to find it finished claims it: it
//! checks and caches the page. The others find the page cached, or, if the read
//! failed, read it again themselves. A read whose last reader lets go of it is
//! cancelled. Slots are reused, so once the table has grown to the most reads
//! ever in flight at once, starting a read allocates nothing.

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::time::Instant;

use crate::disk::ReadBuf;
use crate::metrics::Latency;
use crate::pool::{FrameId, Pool};

/// The reads in flight, each a disk's read future `R`.
pub(crate) struct Reads<R> {
    slots: Vec<Slot<R>>,
    /// Times the reads, if the store records metrics.
    pub(crate) duration: Latency,
}

struct Slot<R> {
    page: u64,
    /// The frame being read into.
    frame: FrameId,
    state: State<R>,
    /// When the read started, if it is timed.
    started: Option<Instant>,
    /// Readers holding the slot. Each claims the read or lets go of it.
    users: u32,
    waiters: Arc<Waiters>,
    /// Wakes `waiters`. The read is polled with it rather than with the waker of
    /// the reader polling it, so that the read finishing wakes every reader
    /// waiting for it, whichever polled it last.
    waker: Waker,
}

enum State<R> {
    Free,
    /// In flight, or finished without anyone having polled it since.
    Reading(R),
    /// Finished, and not claimed yet.
    Finished(io::Result<()>, ReadBuf),
    /// Claimed by a reader, which took the frame.
    Claimed,
}

/// A finished read, for the reader that claimed it: the frame, the outcome, and
/// the buffer the disk returned, which holds a reference to the frame, unless the
/// disk returned another.
pub(crate) struct Claim {
    pub(crate) frame: FrameId,
    pub(crate) res: io::Result<()>,
    pub(crate) buf: ReadBuf,
}

impl<R> Default for Reads<R> {
    fn default() -> Self {
        Reads {
            slots: Vec::new(),
            duration: Latency::default(),
        }
    }
}

impl<R> Reads<R> {
    /// The slot of the read of `page`, if one is in flight or unclaimed.
    pub(crate) fn find(&self, page: u64) -> Option<usize> {
        self.slots.iter().position(|s| {
            s.page == page && matches!(s.state, State::Reading(_) | State::Finished(..))
        })
    }

    /// Adds `read`, which fills `frame` with `page`, held by the caller. Returns
    /// its slot.
    pub(crate) fn start(&mut self, page: u64, frame: FrameId, read: R) -> usize {
        let i = match self
            .slots
            .iter()
            .position(|s| matches!(s.state, State::Free))
        {
            Some(i) => i,
            None => {
                let waiters = Arc::new(Waiters::default());
                let waker = Waker::from(waiters.clone());
                self.slots.push(Slot {
                    page: 0,
                    frame: 0,
                    state: State::Free,
                    started: None,
                    users: 0,
                    waiters,
                    waker,
                });
                self.slots.len() - 1
            }
        };
        let slot = &mut self.slots[i];
        slot.page = page;
        slot.frame = frame;
        slot.state = State::Reading(read);
        slot.started = self.duration.now();
        slot.users = 1;
        i
    }

    /// Makes the caller another holder of slot `i`.
    pub(crate) fn join(&mut self, i: usize) {
        self.slots[i].users += 1;
    }

    /// Lets go of slot `i`. If no reader holds it any more and its read wasn't
    /// claimed, the read is cancelled, and its frame freed once the read is done
    /// with it: now, or else by the cache's clock hand when it finishes.
    pub(crate) fn leave(&mut self, i: usize, pool: &mut Pool) {
        let slot = &mut self.slots[i];
        slot.users -= 1;
        if slot.users > 0 {
            return;
        }
        let claimed = matches!(slot.state, State::Claimed);
        slot.state = State::Free;
        slot.waiters.clear();
        if !claimed && pool.unused(slot.frame) {
            pool.free(slot.frame);
        }
    }

    /// The reads held: each one's slot, and the frame it fills, unless a reader
    /// claimed it already.
    #[cfg(test)]
    pub(crate) fn held(&self) -> impl Iterator<Item = (usize, Option<FrameId>)> + '_ {
        let held = self.slots.iter().enumerate();
        held.filter_map(|(i, s)| match s.state {
            State::Free => None,
            State::Claimed => Some((i, None)),
            State::Reading(_) | State::Finished(..) => Some((i, Some(s.frame))),
        })
    }
}

impl<R: Future<Output = (io::Result<()>, ReadBuf)> + Unpin> Reads<R> {
    /// Sends the read in slot `i` on its way, without claiming it.
    pub(crate) fn submit(&mut self, i: usize) {
        self.slots[i].advance(&self.duration);
    }

    /// Claims the read in slot `i` once it has finished. Returns `None` if
    /// another reader claimed it first.
    pub(crate) fn poll(&mut self, i: usize, cx: &mut Context<'_>) -> Poll<Option<Claim>> {
        let slot = &mut self.slots[i];
        if !slot.advance(&self.duration) {
            slot.waiters.add(cx.waker());
            return Poll::Pending;
        }
        match std::mem::replace(&mut slot.state, State::Claimed) {
            State::Finished(res, buf) => {
                // The other readers waiting look again.
                slot.waiters.wake_all();
                Poll::Ready(Some(Claim {
                    frame: slot.frame,
                    res,
                    buf,
                }))
            }
            State::Claimed => Poll::Ready(None),
            State::Free | State::Reading(_) => unreachable!("read slot {i} not held"),
        }
    }
}

impl<R: Future<Output = (io::Result<()>, ReadBuf)> + Unpin> Slot<R> {
    /// Polls the read, if it is in flight, recording its time in `duration` once it
    /// finishes. Returns whether it is over: finished or claimed.
    fn advance(&mut self, duration: &Latency) -> bool {
        if let State::Reading(read) = &mut self.state {
            match Pin::new(read).poll(&mut Context::from_waker(&self.waker)) {
                Poll::Ready((res, buf)) => {
                    duration.record_since(self.started);
                    self.state = State::Finished(res, buf);
                }
                Poll::Pending => return false,
            }
        }
        true
    }
}

/// The readers waiting for a read. Rust requires wakers to be `Send` and `Sync`,
/// hence the lock, which is never contended: a store's reads all run on its
/// thread.
#[derive(Default)]
struct Waiters(Mutex<Vec<Waker>>);

impl Waiters {
    fn add(&self, waker: &Waker) {
        let mut wakers = self.0.lock().unwrap();
        if !wakers.iter().any(|w| w.will_wake(waker)) {
            wakers.push(waker.clone());
        }
    }

    fn wake_all(&self) {
        for waker in self.0.lock().unwrap().drain(..) {
            waker.wake();
        }
    }

    fn clear(&self) {
        self.0.lock().unwrap().clear();
    }
}

impl Wake for Waiters {
    fn wake(self: Arc<Self>) {
        self.wake_all();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.wake_all();
    }
}
