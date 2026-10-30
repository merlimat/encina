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

//! Stores in an env the test implements, through the public API alone: its disk
//! keeps the store in memory and notes the class of each request, and it runs the
//! task the store spawns on the test's own executor. No compio runtime runs, and
//! no clock decides what the store does.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::future::{Future, Ready, poll_fn, ready};
use std::io;
use std::mem;
use std::path::{Path, PathBuf};
use std::pin::{Pin, pin};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Wake, Waker};

use encina::{
    Access, Comparison, Db, Disk, Env, Error, Options, ReadBuf, ReadOnlyDb, WriteBatch, WriteBuf,
};

/// A request the disk took, with its class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Request {
    Read(u8),
    Write(u8),
    Cut(u8),
    Sync(u8),
}

#[derive(Default)]
struct State {
    bytes: RefCell<Vec<u8>>,
    requests: RefCell<Vec<Request>>,
    tasks: RefCell<Vec<Pin<Box<dyn Future<Output = ()>>>>>,
    /// While set, syncs wait, and these are the wakers of those waiting.
    hold_syncs: Cell<bool>,
    waiting: RefCell<Vec<Waker>>,
    /// Whether asking for its length fails.
    broken: Cell<bool>,
    /// The opens asked of the env, with their access.
    opens: RefCell<Vec<(PathBuf, Access)>>,
    /// While set, an owner's open fails, as with `Files` while another owner has
    /// the file.
    taken: Cell<bool>,
}

/// A disk in memory, whose requests finish at once.
#[derive(Clone, Default)]
struct MemDisk(Rc<State>);

impl MemDisk {
    fn note(&self, request: Request) {
        self.0.requests.borrow_mut().push(request);
    }

    /// The requests taken since the last call.
    fn requests(&self) -> Vec<Request> {
        mem::take(&mut self.0.requests.borrow_mut())
    }

    /// The opens asked since the last call.
    fn opens(&self) -> Vec<(PathBuf, Access)> {
        mem::take(&mut self.0.opens.borrow_mut())
    }

    fn land(&self, offset: u64, data: &[u8]) {
        let mut bytes = self.0.bytes.borrow_mut();
        let at = offset as usize;
        if bytes.len() < at + data.len() {
            bytes.resize(at + data.len(), 0);
        }
        bytes[at..at + data.len()].copy_from_slice(data);
    }

    fn hold_syncs(&self, hold: bool) {
        self.0.hold_syncs.set(hold);
        for waker in self.0.waiting.borrow_mut().drain(..) {
            waker.wake();
        }
    }

    /// The tasks spawned that haven't ended.
    fn tasks(&self) -> usize {
        self.0.tasks.borrow().len()
    }

    /// Polls each task spawned once, with `waker`, and drops those that end.
    /// Returns how many are left.
    fn poll_tasks(&self, waker: &Waker) -> usize {
        // A task polled may spawn another, or drop the store that spawned it.
        let mut tasks = mem::take(&mut *self.0.tasks.borrow_mut());
        let mut cx = Context::from_waker(waker);
        tasks.retain_mut(|task| task.as_mut().poll(&mut cx).is_pending());
        let mut spawned = self.0.tasks.borrow_mut();
        tasks.append(&mut spawned);
        *spawned = tasks;
        spawned.len()
    }
}

impl Disk for MemDisk {
    type Class = u8;
    type Read = Ready<(io::Result<()>, ReadBuf)>;

    async fn len(&self) -> io::Result<u64> {
        if self.0.broken.get() {
            return Err(io::Error::other("broken"));
        }
        Ok(self.0.bytes.borrow().len() as u64)
    }

    fn read(&self, mut buf: ReadBuf, offset: u64, class: u8) -> Self::Read {
        self.note(Request::Read(class));
        let at = offset as usize;
        let res = match self.0.bytes.borrow().get(at..at + buf.len()) {
            Some(data) => {
                buf.copy_from_slice(data);
                Ok(())
            }
            None => Err(io::ErrorKind::UnexpectedEof.into()),
        };
        ready((res, buf))
    }

    async fn write(&self, buf: WriteBuf, offset: u64, class: u8) -> (io::Result<()>, WriteBuf) {
        self.note(Request::Write(class));
        self.land(offset, &buf);
        (Ok(()), buf)
    }

    async fn write_vectored(
        &self,
        bufs: Vec<WriteBuf>,
        offset: u64,
        class: u8,
    ) -> (io::Result<()>, Vec<WriteBuf>) {
        self.note(Request::Write(class));
        let mut at = offset;
        for buf in &bufs {
            self.land(at, buf);
            at += buf.len() as u64;
        }
        (Ok(()), bufs)
    }

    async fn set_len(&self, len: u64, class: u8) -> io::Result<()> {
        self.note(Request::Cut(class));
        self.0.bytes.borrow_mut().resize(len as usize, 0);
        Ok(())
    }

    async fn sync(&self, class: u8) -> io::Result<()> {
        self.note(Request::Sync(class));
        poll_fn(|cx| {
            if !self.0.hold_syncs.get() {
                return Poll::Ready(());
            }
            self.0.waiting.borrow_mut().push(cx.waker().clone());
            Poll::Pending
        })
        .await;
        Ok(())
    }
}

/// The disk is its own env: it opens itself, whatever the path, and keeps the
/// tasks the store spawns for the test's executor.
impl Env for MemDisk {
    type Disk = MemDisk;

    async fn open(&self, path: &Path, access: Access) -> encina::Result<MemDisk> {
        self.0.opens.borrow_mut().push((path.to_owned(), access));
        if access == Access::ReadWrite && self.0.taken.get() {
            return Err(Error::Locked);
        }
        Ok(self.clone())
    }

    fn spawn(&self, task: impl Future<Output = ()> + 'static) {
        self.0.tasks.borrow_mut().push(Box::pin(task));
    }
}

/// Counts its wakes.
#[derive(Default)]
struct Wakes(AtomicUsize);

impl Wake for Wakes {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

/// Runs `f` to its end on this thread, and the tasks the env spawns meanwhile,
/// polling each in turn as long as `f` hasn't ended. Returns its output, and how
/// many times it paused.
fn run_counting<F: Future>(disk: &MemDisk, f: F) -> (F::Output, usize) {
    let mut f = pin!(f);
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut pauses = 0;
    loop {
        if let Poll::Ready(output) = f.as_mut().poll(&mut cx) {
            return (output, pauses);
        }
        pauses += 1;
        assert!(pauses < 100_000, "the future never ends");
        disk.poll_tasks(waker);
    }
}

fn run<F: Future>(disk: &MemDisk, f: F) -> F::Output {
    run_counting(disk, f).0
}

/// Runs the tasks the disk spawned until they all end.
fn finish_tasks(disk: &MemDisk) {
    for _ in 0..10_000 {
        if disk.poll_tasks(Waker::noop()) == 0 {
            return;
        }
    }
    panic!("the tasks never end");
}

fn key(i: u32) -> [u8; 4] {
    i.to_be_bytes()
}

/// Puts entries `range` with values of 100 bytes from `seed`, and every 997th a
/// value too large for a leaf, into `batch` and `model`.
fn puts(
    batch: &mut WriteBatch,
    model: &mut BTreeMap<Vec<u8>, Vec<u8>>,
    range: std::ops::Range<u32>,
    seed: u8,
) {
    for i in range {
        let len = if i % 997 == 0 { 70_000 } else { 100 };
        let value = vec![seed ^ i as u8; len];
        batch.put(&key(i), &value);
        model.insert(key(i).to_vec(), value);
    }
}

/// A store of a few level-1 nodes, on a new disk, closed: the next open leaves
/// the scan of level 1 to the task it spawns. Returns the disk and the entries.
fn store(opts: &Options) -> (MemDisk, BTreeMap<Vec<u8>, Vec<u8>>) {
    let disk = MemDisk::default();
    let db = run(&disk, Db::open(&disk, "store", opts.clone())).unwrap();
    // A new store has no level 1 to scan.
    assert_eq!(disk.tasks(), 0);
    let mut model = BTreeMap::new();
    for round in 0..4 {
        let mut batch = WriteBatch::new();
        puts(&mut batch, &mut model, round * 5000..(round + 1) * 5000, 1);
        run(&disk, db.write(&mut batch)).unwrap();
    }
    run(&disk, db.close()).unwrap();
    (disk, model)
}

fn entries(db: &Db<MemDisk>, disk: &MemDisk) -> Vec<(Vec<u8>, Vec<u8>)> {
    run(disk, async {
        let mut iter = db.range(..);
        let mut entries = Vec::new();
        while let Some((k, v)) = iter.next().await.unwrap() {
            entries.push((k.to_vec(), v.to_vec()));
        }
        entries
    })
}

fn expected(model: &BTreeMap<Vec<u8>, Vec<u8>>) -> Vec<(Vec<u8>, Vec<u8>)> {
    model.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// A cache of 64 pages, which keeps no value too large for a leaf.
fn opts() -> Options {
    Options::new().cache_size(64 * 4096)
}

#[test]
fn a_store_works_on_a_disk_of_its_callers() {
    let opts = opts();
    let (disk, mut model) = store(&opts);
    let db = run(&disk, Db::open(&disk, "store", opts.clone())).unwrap();
    // The store spawned a task for the scan of level 1, which lookups don't wait
    // for.
    assert_eq!(disk.tasks(), 1);
    for i in (0..20_000).step_by(331) {
        for cmp in [Comparison::Equal, Comparison::Floor, Comparison::Higher] {
            let found = run(&disk, db.get(&key(i), cmp, |k, v| (k.to_vec(), v.to_vec())));
            let want = match cmp {
                Comparison::Higher => model.range(key(i).to_vec()..).nth(1),
                _ => model.get_key_value(&key(i)[..]),
            };
            let want = want.map(|(k, v)| (k.clone(), v.clone()));
            assert_eq!(found.unwrap(), want, "{cmp:?} {i}");
        }
    }
    finish_tasks(&disk);
    assert_eq!(entries(&db, &disk), expected(&model));

    // Writes once the scan has ended reuse the pages earlier ones freed: three
    // that each rewrite every leaf don't double the disk.
    let len = disk.0.bytes.borrow().len();
    for round in 0..3 {
        let mut batch = WriteBatch::new();
        puts(&mut batch, &mut model, 0..20_000, round + 2);
        batch.delete_range(&key(5000), &key(6000));
        model.retain(|k, _| !(&key(5000)[..]..&key(6000)[..]).contains(&&k[..]));
        run(&disk, db.write(&mut batch)).unwrap();
    }
    assert!(disk.0.bytes.borrow().len() < 2 * len);
    assert_eq!(entries(&db, &disk), expected(&model));

    // A view reads it as of when it opens, and spawns nothing.
    let view = run(&disk, ReadOnlyDb::open(&disk, "store", opts.clone())).unwrap();
    assert_eq!(disk.tasks(), 0);
    // Each open asked the env for the store's path: the new store's and the
    // reopen's as its owner, and the view's to read only.
    let path = PathBuf::from("store");
    assert_eq!(
        disk.opens(),
        [
            (path.clone(), Access::ReadWrite),
            (path.clone(), Access::ReadWrite),
            (path, Access::ReadOnly),
        ]
    );
    let (mut nodes, mut leaves) = (0, 0);
    run(
        &disk,
        view.pages(|info| (nodes, leaves) = (nodes + 1, leaves + usize::from(info.level == 0))),
    )
    .unwrap();
    assert!(
        leaves > 100 && nodes > leaves,
        "{nodes} nodes, {leaves} leaves"
    );
    run(&disk, db.close()).unwrap();
    let view = view.clone();
    let found = run(
        &disk,
        view.get(&key(997), Comparison::Equal, |_, v| v.to_vec()),
    );
    assert_eq!(found.unwrap().as_ref(), model.get(&key(997)[..]));
    run(&disk, view.close()).unwrap();
}

#[test]
fn the_scan_task_ends_with_its_store() {
    // A cache of 8 pages scans two level-1 nodes at a time, and pauses in between.
    let opts = Options::new().cache_size(8 * 4096);
    let (disk, _) = store(&opts);
    let db = run(&disk, Db::open(&disk, "store", opts)).unwrap();
    let wakes = Arc::new(Wakes::default());
    let waker = Waker::from(wakes.clone());
    assert_eq!(disk.poll_tasks(&waker), 1, "the scan ended at once");
    let before = wakes.0.load(Ordering::Relaxed);
    drop(db);
    // The store's drop woke the task, which ends.
    assert!(wakes.0.load(Ordering::Relaxed) > before);
    assert_eq!(disk.poll_tasks(&waker), 0);
}

#[test]
fn each_request_takes_the_class_of_what_caused_it() {
    let opts = opts();
    let (disk, model) = store(&opts);
    disk.requests();
    let db = run(&disk, Db::open(&disk, "store", opts.clone())).unwrap();
    finish_tasks(&disk);
    // The open's reads, and the scan's, take the default class.
    let opened = disk.requests();
    assert!(!opened.is_empty() && opened.iter().all(|&r| r == Request::Read(0)));
    let all_are = |want: Request| {
        let requests = disk.requests();
        assert!(!requests.is_empty(), "no requests");
        assert!(
            requests.iter().all(|&r| r == want),
            "{requests:?} for {want:?}"
        );
    };

    let get =
        |db: &Db<MemDisk>, i: u32| run(&disk, db.get(&key(i), Comparison::Equal, |_, v| v.len()));
    assert_eq!(get(&db.with_class(3), 4321).unwrap(), Some(100));
    all_are(Request::Read(3));
    // A value too large for the cache is read whole each time.
    assert_eq!(get(&db.with_class(7), 997).unwrap(), Some(70_000));
    all_are(Request::Read(7));

    let reader = db.with_class(4);
    let mut iter = reader.range(&key(7000)[..]..);
    let first = run(&disk, iter.next()).unwrap().map(|(k, _)| k.to_vec());
    assert_eq!(first, Some(key(7000).to_vec()));
    drop(iter);
    all_are(Request::Read(4));

    let mut batch = WriteBatch::new();
    batch.put(&key(12_345), b"new");
    run(&disk, db.with_class(5).write(&mut batch)).unwrap();
    let requests = disk.requests();
    assert!(requests.contains(&Request::Sync(5)));
    assert!(
        requests.iter().all(|r| matches!(
            r,
            Request::Read(5) | Request::Write(5) | Request::Cut(5) | Request::Sync(5)
        )),
        "{requests:?}"
    );

    let view = run(&disk, ReadOnlyDb::open(&disk, "store", opts)).unwrap();
    disk.requests();
    let found = run(
        &disk,
        view.with_class(6)
            .get(&key(8888), Comparison::Equal, |_, v| v.to_vec()),
    );
    assert_eq!(found.unwrap().as_ref(), model.get(&key(8888)[..]));
    all_are(Request::Read(6));
}

#[test]
fn a_commit_takes_the_class_of_the_writes_waiting_for_it() {
    let opts = opts();
    let (disk, _) = store(&opts);
    let db = run(&disk, Db::open(&disk, "store", opts)).unwrap();
    finish_tasks(&disk);
    disk.requests();
    let mut batches: Vec<_> = (0..3u32)
        .map(|i| {
            let mut batch = WriteBatch::new();
            batch.put(&key(i * 7000), b"new");
            batch
        })
        .collect();
    let [one, two, three] = &mut batches[..] else {
        unreachable!()
    };
    let (low, high, mid) = (db.with_class(1), db.with_class(9), db.with_class(4));
    let mut cx = Context::from_waker(Waker::noop());

    // The first write commits alone, and waits for its first sync.
    disk.hold_syncs(true);
    let mut first = pin!(low.write(one));
    assert!(first.as_mut().poll(&mut cx).is_pending());
    let alone = disk.requests();
    assert_eq!(alone.last(), Some(&Request::Sync(1)), "{alone:?}");
    assert!(
        alone
            .iter()
            .all(|r| matches!(r, Request::Read(1) | Request::Write(1) | Request::Sync(1))),
        "{alone:?}"
    );

    // Two more queue behind it, and raise its class to the greater of theirs.
    let mut second = pin!(high.write(two));
    let mut third = pin!(mid.write(three));
    assert!(second.as_mut().poll(&mut cx).is_pending());
    assert!(third.as_mut().poll(&mut cx).is_pending());
    assert_eq!(disk.requests(), []);
    disk.hold_syncs(false);
    run(&disk, first).unwrap();
    // Its meta page and second sync, and a cut if any.
    let raised = disk.requests();
    assert_eq!(raised[..2], [Request::Write(9), Request::Sync(9)]);
    assert!(
        raised[2..].iter().all(|&r| r == Request::Cut(9)),
        "{raised:?}"
    );

    // They commit together, with the greater class.
    run(&disk, async {
        second.await.unwrap();
        third.await.unwrap();
    });
    let group = disk.requests();
    assert!(group.contains(&Request::Sync(9)));
    assert!(
        group.iter().all(|r| matches!(
            r,
            Request::Read(9) | Request::Write(9) | Request::Cut(9) | Request::Sync(9)
        )),
        "{group:?}"
    );
}

#[test]
fn writes_pause_in_the_same_places_each_time() {
    // An unsorted batch, whose normalizing and merge pause many times, on a new
    // store each time.
    let write = || {
        let mut batch = WriteBatch::new();
        let mut i = 1u32;
        for _ in 0..30_000 {
            i = i.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            batch.put(&i.to_be_bytes(), &[i as u8; 16]);
        }
        let disk = MemDisk::default();
        let db = run(&disk, Db::open(&disk, "store", Options::new())).unwrap();
        let (written, pauses) = run_counting(&disk, db.write(&mut batch));
        written.unwrap();
        (pauses, disk.requests(), disk.0.bytes.borrow().clone())
    };
    let (pauses, requests, bytes) = write();
    assert!(pauses > 3, "{pauses} pauses");
    assert_eq!(write(), (pauses, requests, bytes));
}

#[test]
fn opens_on_a_disk_fail_as_on_a_file() {
    let disk = MemDisk::default();
    let opened = run(&disk, ReadOnlyDb::open(&disk, "store", Options::new()));
    assert!(matches!(opened, Err(Error::InvalidFile(_))));
    let opened = run(
        &disk,
        Db::open(&disk, "store", Options::new().page_size(1000)),
    );
    assert!(matches!(opened, Err(Error::InvalidOptions(_))));
    let opened = run(
        &disk,
        ReadOnlyDb::open(&disk, "store", Options::new().page_size(1000)),
    );
    assert!(matches!(opened, Err(Error::InvalidOptions(_))));
    // Options are checked before the env opens anything.
    assert_eq!(disk.opens(), [(PathBuf::from("store"), Access::ReadOnly)]);
    // The env's errors come back from the open: an owner's is refused, a view's
    // isn't.
    disk.0.taken.set(true);
    assert!(matches!(
        run(&disk, Db::open(&disk, "store", Options::new())),
        Err(Error::Locked)
    ));
    let opened = run(&disk, ReadOnlyDb::open(&disk, "store", Options::new()));
    assert!(matches!(opened, Err(Error::InvalidFile(_))));
    disk.0.taken.set(false);
    // So do the disk's.
    disk.0.broken.set(true);
    assert!(matches!(
        run(&disk, Db::open(&disk, "store", Options::new())),
        Err(Error::Io(_))
    ));
    let opened = run(&disk, ReadOnlyDb::open(&disk, "store", Options::new()));
    assert!(matches!(opened, Err(Error::Io(_))));
}
