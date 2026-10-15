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

//! The writes waiting for a commit (§6). A write that finds no commit running and
//! no write waiting commits its batch alone. Writes that arrive while a commit
//! runs join the next group: each appends its batch to the group's, in the order
//! they arrive. Once the commit ends, whichever of them runs first commits the
//! whole group, and the others get its outcome.

use std::mem;
use std::task::{Context, Poll, Waker};

use crate::batch::WriteBatch;
use crate::error::Result;

/// The memory the batches of a group may take together. A write whose batch would
/// take the group past it waits for the group after, unless it would be the
/// group's first.
pub(crate) const GROUP_SIZE: usize = 64 << 20;

/// What a write does once its turn comes.
pub(crate) enum Turn {
    /// It commits its own batch, alone.
    Alone,
    /// It commits the current group, which holds its batch, from this slot.
    Lead(usize),
    /// Another write committed its group, with this outcome.
    Done(Result<()>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    /// There is no current group.
    Idle,
    /// The current group waits for one of its writes to commit it: the write that
    /// was committing it went away before writing anything.
    Waiting,
    /// A write is building the group's new version. A write that leaves the group
    /// takes its batch with it.
    Building,
    /// A write is committing the group's new version: every batch in it commits,
    /// or none does.
    Committing,
}

/// A write waiting for a commit.
#[derive(Default)]
struct Member {
    /// The group it joined, if it has yet.
    group: Option<u64>,
    /// The indexes of its operations in that group's batch.
    ops: (u32, u32),
    /// Its outcome, once its group's commit is over.
    result: Option<Result<()>>,
    waker: Option<Waker>,
}

pub(crate) struct Writes {
    /// A write is committing. The others wait.
    busy: bool,
    /// The group being committed, or waiting to be: its writes' batches, appended
    /// in the order they joined, with how many writes it holds.
    current: WriteBatch,
    current_writes: usize,
    state: State,
    /// A write left the group since its leader last prepared its batch.
    changed: bool,
    /// The group that writes join now. Its number is one more than the current
    /// group's.
    next: WriteBatch,
    next_writes: usize,
    next_group: u64,
    /// The writes waiting, by slot, and the free slots. Slots are reused, so once
    /// warmed up, waiting allocates nothing; neither do the groups' batches.
    members: Vec<Member>,
    free: Vec<usize>,
    /// The memory a group's batches may take: `GROUP_SIZE`, or less in tests.
    limit: usize,
}

impl Writes {
    pub(crate) fn new() -> Writes {
        Writes {
            busy: false,
            current: WriteBatch::new(),
            current_writes: 0,
            state: State::Idle,
            changed: false,
            next: WriteBatch::new(),
            next_writes: 0,
            next_group: 0,
            members: Vec::new(),
            free: Vec::new(),
            limit: GROUP_SIZE,
        }
    }

    /// Moves along the write that holds `slot`, or none yet, with `batch`: it
    /// commits alone if nothing else is going on, otherwise it joins the next
    /// group. Then it waits until it can commit its group, or until another write
    /// has.
    pub(crate) fn poll(
        &mut self,
        slot: &mut Option<usize>,
        batch: &WriteBatch,
        cx: &mut Context<'_>,
    ) -> Poll<Turn> {
        let s = match *slot {
            Some(s) => s,
            None if !self.busy && self.state == State::Idle && self.next_writes == 0 => {
                self.busy = true;
                self.changed = false;
                return Poll::Ready(Turn::Alone);
            }
            None => *slot.insert(self.enter()),
        };
        if let Some(result) = self.members[s].result.take() {
            *slot = None;
            self.free_slot(s);
            return Poll::Ready(Turn::Done(result));
        }
        if self.members[s].group.is_none()
            && (self.next_writes == 0 || self.next.size() + batch.size() <= self.limit)
        {
            self.members[s].group = Some(self.next_group);
            self.members[s].ops = self.next.append(batch);
            self.next_writes += 1;
        }
        let first = match self.state {
            State::Idle => self.next_group,
            _ => self.next_group - 1,
        };
        if !self.busy && self.members[s].group == Some(first) {
            if self.state == State::Idle {
                self.seal();
            }
            self.state = State::Building;
            self.busy = true;
            self.changed = false;
            *slot = None;
            return Poll::Ready(Turn::Lead(s));
        }
        let waker = &mut self.members[s].waker;
        if !waker.as_ref().is_some_and(|w| w.will_wake(cx.waker())) {
            *waker = Some(cx.waker().clone());
        }
        Poll::Pending
    }

    /// The current group's batch.
    pub(crate) fn group(&mut self) -> &mut WriteBatch {
        &mut self.current
    }

    /// Whether a write left the group since the last call.
    pub(crate) fn take_changed(&mut self) -> bool {
        mem::take(&mut self.changed)
    }

    /// Marks the current group as committing: from now on, its writes can't leave.
    pub(crate) fn committing(&mut self) {
        self.state = State::Committing;
    }

    /// Ends the commit of the current group, which the write in `slot` led: the
    /// group's other writes get `result`.
    pub(crate) fn deliver(&mut self, slot: usize, result: &Result<()>) {
        let group = Some(self.next_group - 1);
        for (i, member) in self.members.iter_mut().enumerate() {
            if i != slot && member.group == group {
                member.result = Some(match result {
                    Ok(()) => Ok(()),
                    Err(e) => Err(e.duplicate()),
                });
            }
        }
        self.free_slot(slot);
        self.current.clear();
        self.current_writes = 0;
        self.state = State::Idle;
        self.wake_all();
    }

    /// Takes the batch of the write in `slot`, which went away while building the
    /// current group, out of the group. Another of the group's writes commits the
    /// rest.
    pub(crate) fn abandon(&mut self, slot: usize) {
        self.leave(slot);
        if self.state == State::Building {
            self.state = State::Waiting;
        }
    }

    /// Frees the slot of a write that went away. Its batch leaves its group, unless
    /// the group is already committing.
    pub(crate) fn leave(&mut self, slot: usize) {
        let member = mem::take(&mut self.members[slot]);
        self.free.push(slot);
        let Some(group) = member.group.filter(|_| member.result.is_none()) else {
            return;
        };
        if group == self.next_group {
            self.next.withdraw(member.ops);
            self.next_writes -= 1;
            if self.next_writes == 0 {
                self.next.clear();
            }
        } else if self.state != State::Committing {
            self.current.withdraw(member.ops);
            self.current_writes -= 1;
            self.changed = true;
            if self.current_writes == 0 {
                self.current.clear();
                self.state = State::Idle;
            }
        }
        self.wake_all();
    }

    /// Ends the commit in progress: another may start.
    pub(crate) fn release(&mut self) {
        self.busy = false;
        self.wake_all();
    }

    /// Makes the next group current, and opens a new one.
    fn seal(&mut self) {
        debug_assert_eq!(self.current.size(), 0, "a group is still current");
        mem::swap(&mut self.current, &mut self.next);
        self.current_writes = mem::take(&mut self.next_writes);
        self.next_group += 1;
        // Writes that found the group full can join the new one.
        self.wake_all();
    }

    fn enter(&mut self) -> usize {
        self.free.pop().unwrap_or_else(|| {
            self.members.push(Member::default());
            self.members.len() - 1
        })
    }

    fn free_slot(&mut self, slot: usize) {
        self.members[slot] = Member::default();
        self.free.push(slot);
    }

    fn wake_all(&mut self) {
        for member in &mut self.members {
            if let Some(waker) = member.waker.take() {
                waker.wake();
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn set_limit(&mut self, limit: usize) {
        self.limit = limit;
    }

    /// Whether no write is committing or waiting.
    #[cfg(test)]
    pub(crate) fn is_idle(&self) -> bool {
        !self.busy
            && self.state == State::Idle
            && self.next_writes == 0
            && self.free.len() == self.members.len()
    }
}
