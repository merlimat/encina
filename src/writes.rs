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
//! whole group, and the others get its outcome. No write depends on another being
//! polled, except on the one committing.
//!
//! Each write has a class (see [`Disk::Class`](crate::Disk::Class)). A commit's
//! requests take the greatest class among the writes it commits and those that
//! wait for it: a write that waits behind a commit raises its later requests.

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
    /// It commits the current group: from this slot if the group holds its batch.
    /// Otherwise the group lost the write committing it, and this write takes over
    /// before its own turn.
    Lead(Option<usize>),
    /// Another write committed its group, with this outcome.
    Done(Result<()>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    /// There is no current group.
    Idle,
    /// The current group waits for a write to take it over: the write that was
    /// committing it went away before writing anything.
    Waiting,
    /// A write is building the group's new version. A write that leaves the group
    /// takes its batch with it.
    Building,
    /// A write is committing the group's new version: every batch in it commits,
    /// or none does.
    Committing,
}

/// A write waiting for a commit, of class `C`.
#[derive(Default)]
struct Member<C> {
    /// Its class, while the slot is held.
    class: Option<C>,
    /// The group it joined, if it has yet.
    group: Option<u64>,
    /// The indexes of its operations in that group's batch.
    ops: (u32, u32),
    /// Its outcome, once its group's commit is over.
    result: Option<Result<()>>,
    waker: Option<Waker>,
}

pub(crate) struct Writes<C> {
    /// A write is committing. The others wait.
    busy: bool,
    /// The class of the commit's requests.
    class: C,
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
    members: Vec<Member<C>>,
    free: Vec<usize>,
    /// The memory a group's batches may take: `GROUP_SIZE`, or less in tests.
    limit: usize,
}

impl<C: Copy + Ord + Default> Writes<C> {
    pub(crate) fn new() -> Writes<C> {
        Writes {
            busy: false,
            class: C::default(),
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

    /// Moves along the write of `class` that holds `slot`, or none yet, with
    /// `batch`: it commits alone if nothing else is going on, otherwise it joins
    /// the next group. Then it waits until it can commit its group, or until
    /// another write has.
    pub(crate) fn poll(
        &mut self,
        slot: &mut Option<usize>,
        batch: &WriteBatch,
        class: C,
        cx: &mut Context<'_>,
    ) -> Poll<Turn> {
        let s = match *slot {
            Some(s) => s,
            None if !self.busy && self.state == State::Idle && self.next_writes == 0 => {
                self.busy = true;
                self.changed = false;
                self.class = class;
                return Poll::Ready(Turn::Alone);
            }
            None => *slot.insert(self.enter(class)),
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
        // With no commit running, this write starts one: of the current group, if
        // one waits for a write to take it over, or else of the next group.
        if !self.busy {
            if self.state == State::Idle {
                self.seal();
            }
            let own = self.members[s].group == Some(self.next_group - 1);
            self.state = State::Building;
            self.busy = true;
            self.changed = false;
            // The writes that waited, in its group or for the next one, and not
            // those whose outcome another group's commit gave them.
            let waited = self.members.iter().filter(|m| m.result.is_none());
            self.class = waited.filter_map(|m| m.class).fold(class, C::max);
            if !own {
                return Poll::Ready(Turn::Lead(None));
            }
            *slot = None;
            return Poll::Ready(Turn::Lead(Some(s)));
        }
        // It waits for the commit in progress, which takes its class if greater.
        self.class = self.class.max(class);
        let waker = &mut self.members[s].waker;
        if !waker.as_ref().is_some_and(|w| w.will_wake(cx.waker())) {
            *waker = Some(cx.waker().clone());
        }
        Poll::Pending
    }

    /// The class of the commit in progress.
    pub(crate) fn class(&self) -> C {
        self.class
    }

    /// The current group's batch.
    pub(crate) fn group(&mut self) -> &mut WriteBatch {
        &mut self.current
    }

    /// How many writes the current group holds.
    pub(crate) fn group_writes(&self) -> usize {
        self.current_writes
    }

    /// Whether a write left the group since the last call.
    pub(crate) fn take_changed(&mut self) -> bool {
        mem::take(&mut self.changed)
    }

    /// Marks the current group as committing: from now on, its writes can't leave.
    pub(crate) fn committing(&mut self) {
        self.state = State::Committing;
    }

    /// Ends the commit of the current group, led from `slot` if the leader was in
    /// the group: the group's other writes get `result`.
    pub(crate) fn deliver(&mut self, slot: Option<usize>, result: &Result<()>) {
        let group = Some(self.next_group - 1);
        for (i, member) in self.members.iter_mut().enumerate() {
            if Some(i) != slot && member.group == group {
                member.result = Some(match result {
                    Ok(()) => Ok(()),
                    Err(e) => Err(e.duplicate()),
                });
            }
        }
        if let Some(slot) = slot {
            self.free_slot(slot);
        }
        self.current.clear();
        self.current_writes = 0;
        self.state = State::Idle;
        self.wake_all();
    }

    /// Leaves the current group to another write: the one building it, from `slot`
    /// if it was in the group, went away, and takes its batch along.
    pub(crate) fn abandon(&mut self, slot: Option<usize>) {
        if let Some(slot) = slot {
            self.leave(slot);
        }
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

    fn enter(&mut self, class: C) -> usize {
        let slot = self.free.pop().unwrap_or_else(|| {
            self.members.push(Member::default());
            self.members.len() - 1
        });
        self.members[slot].class = Some(class);
        slot
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
