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
//! one: then only what was synced survives, plus whichever unsynced writes happen
//! to have landed, with the write in progress torn.

use std::cell::RefCell;
use std::io;
use std::rc::Rc;

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
    /// Writes since the last sync: offset and length.
    unsynced: Vec<(usize, usize)>,
    ops: u64,
    fault: Option<(u64, Fault)>,
    /// The disk as a crash left it.
    crashed: Option<Vec<u8>>,
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

    /// Counts an operation and applies a fault planned for it. `write` is the data
    /// the operation writes, if any, for tearing.
    fn begin(&self, write: Option<(usize, &[u8])>) -> io::Result<()> {
        let mut disk = self.0.borrow_mut();
        if disk.crashed.is_some() {
            return Err(io::Error::other("the simulated machine crashed"));
        }
        let op = disk.ops;
        disk.ops += 1;
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
        if let Err(e) = self.begin(None) {
            return (Err(e), buf);
        }
        let disk = self.0.borrow();
        let (at, out) = (offset as usize, buf.bytes_mut());
        let res = match disk.current.get(at..at + out.len()) {
            Some(data) => {
                out.copy_from_slice(data);
                Ok(())
            }
            None => Err(io::ErrorKind::UnexpectedEof.into()),
        };
        drop(disk);
        (res, buf)
    }

    pub(crate) fn write(&self, buf: FrameBuf, offset: u64) -> (io::Result<()>, FrameBuf) {
        let (at, data) = (offset as usize, buf.as_init());
        if let Err(e) = self.begin(Some((at, data))) {
            return (Err(e), buf);
        }
        let mut disk = self.0.borrow_mut();
        if disk.current.len() < at + data.len() {
            disk.current.resize(at + data.len(), 0);
        }
        disk.current[at..at + data.len()].copy_from_slice(data);
        disk.unsynced.push((at, data.len()));
        drop(disk);
        (Ok(()), buf)
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

impl Disk {
    /// Records what a crash during operation `op` leaves on disk: what was synced,
    /// plus a pseudo-random subset of the unsynced writes, plus the first half of
    /// the write in progress, if any.
    fn crash(&mut self, op: u64, write: Option<(usize, &[u8])>) {
        let mut image = self.durable.clone();
        let land = |image: &mut Vec<u8>, at: usize, data: &[u8]| {
            if image.len() < at + data.len() {
                image.resize(at + data.len(), 0);
            }
            image[at..at + data.len()].copy_from_slice(data);
        };
        let mut rng = op.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        for &(at, len) in &self.unsynced {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            if rng & 1 == 1 {
                land(&mut image, at, &self.current[at..at + len]);
            }
        }
        if let Some((at, data)) = write {
            land(&mut image, at, &data[..data.len() / 2]);
        }
        self.crashed = Some(image);
    }
}
