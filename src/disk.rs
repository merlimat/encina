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

//! Where a store keeps its pages: a file, or a disk the caller implements.

use std::cell::Cell;
use std::future::Future;
use std::io;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::pin::Pin;
use std::task::{Context, Poll, ready};

use compio::BufResult;
use compio::buf::{IntoInner, IoBuf, IoBufMut, SetLen, Slice};
use compio::driver::op::ReadAt;
use compio::driver::{SharedFd, ToSharedFd};
use compio::fs::{File, OpenOptions};
use compio::io::AsyncWriteAtExt;
use compio::runtime::{JoinHandle, Submit};

use crate::error::Result;
use crate::pool::{FrameBuf, ValueBuf};

/// Direct I/O moves whole blocks: buffer, offset and length must all be multiples
/// of the device's block size, at most 4 KiB.
const BLOCK: u64 = 4096;

/// The disk a store keeps its pages on.
///
/// [`Db::open`](crate::Db::open) keeps a store in a file, which it reads and writes
/// with direct I/O on the current compio runtime. A caller that runs its own I/O,
/// such as a scheduler that orders the requests of several stores, or a simulated
/// disk in tests, implements this trait instead, and opens the store with
/// [`Db::open_on`](crate::Db::open_on) or
/// [`ReadOnlyDb::open_on`](crate::ReadOnlyDb::open_on).
///
/// The disk holds one store, from offset 0: two meta pages, then the tree's. The
/// store calls it from the thread it runs on, and needs nothing else from its
/// runtime: on such a disk it spawns no task but through [`spawn`](Self::spawn),
/// and no clock decides what it does (see [`Db::open_on`](crate::Db::open_on)).
///
/// Requests move owned buffers: [`ReadBuf`]s to read into, and [`WriteBuf`]s to
/// write from. Each spans whole pages, 4 KiB-aligned as direct I/O asks, and
/// offsets are multiples of the page size. A request returns the buffers it was
/// given with its outcome. A read whose future is dropped before it finishes is
/// cancelled: the disk keeps its buffer for as long as the device may still fill
/// it, as the kernel does with compio's.
///
/// The store checks the pages it reads, so a disk that loses, tears or garbles
/// them makes reads fail, as a failing device would. Nothing a disk does with its
/// buffers can make the store touch memory one still lends: a read that returns
/// another buffer than it was given fails with an I/O error, and the frames of
/// buffers a disk keeps aren't reused until it drops them.
#[allow(clippy::len_without_is_empty)]
pub trait Disk: Sized + 'static {
    /// What a request is for: the class of the operation that caused it, so that a
    /// disk that schedules its requests can order them. Of two classes, the
    /// greater is the more urgent. A disk that doesn't schedule takes `()`.
    ///
    /// The reads of a lookup or an iterator take the class of the handle it came
    /// from (see [`Db::with_class`](crate::Db::with_class)). The requests of a
    /// commit take the greatest class among the writes it commits, and the
    /// greatest among those that wait for it once it has started: a write that
    /// waits behind a commit raises the class of the commit's later requests. A
    /// read that several readers share keeps the class of the one that started it.
    /// The open's reads, and those of the scan that follows it, take the default.
    type Class: Copy + Ord + Default + 'static;

    /// A read in flight, which [`read`](Self::read) returns. A store keeps it in a
    /// table, and polls it from whichever of the readers waiting for its page is
    /// polled, each time with the same waker: it must be `Unpin`, so that it can
    /// move with the table.
    type Read: Future<Output = (io::Result<()>, ReadBuf)> + Unpin + 'static;

    /// The disk's length, in bytes.
    fn len(&self) -> impl Future<Output = io::Result<u64>>;

    /// The smallest read the disk supports: 4 KiB by default. No page is smaller,
    /// and an open reads this much at the start of the disk to find the page size.
    fn min_read(&self) -> usize {
        BLOCK as usize
    }

    /// Starts filling `buf` from `offset`. A read past the end of the disk fails
    /// with [`io::ErrorKind::UnexpectedEof`].
    fn read(&self, buf: ReadBuf, offset: u64, class: Self::Class) -> Self::Read;

    /// Writes all of `buf` at `offset`, growing the disk if needed.
    fn write(
        &self,
        buf: WriteBuf,
        offset: u64,
        class: Self::Class,
    ) -> impl Future<Output = (io::Result<()>, WriteBuf)>;

    /// Writes the buffers in `bufs` back to back from `offset`, in one request:
    /// at most 1,024 of them, and 1 MiB in all. Until a sync, each may land, or
    /// not, on its own.
    fn write_vectored(
        &self,
        bufs: Vec<WriteBuf>,
        offset: u64,
        class: Self::Class,
    ) -> impl Future<Output = (io::Result<()>, Vec<WriteBuf>)>;

    /// Cuts the disk at `len` bytes, which drops pages no version uses.
    fn set_len(&self, len: u64, class: Self::Class) -> impl Future<Output = io::Result<()>>;

    /// Makes every write that has finished durable, and the disk's length.
    fn sync(&self, class: Self::Class) -> impl Future<Output = io::Result<()>>;

    /// Runs `task` beside the store until it ends, as a task of the runtime the
    /// store runs on. A store that takes writes calls this as it opens, unless its
    /// tree is a single leaf: the task reads the leaves' parents to find the pages
    /// free for reuse, and until it has, writes take new pages at the end of the
    /// disk. It holds the store only while it is polled. Once the store is closed
    /// or dropped, which wakes it, it ends the next time it is polled.
    fn spawn(&self, task: impl Future<Output = ()> + 'static);

    /// Closes the disk, once the store's last handle is closed: nothing, by
    /// default. A store dropped rather than closed drops its disk.
    fn close(self) -> impl Future<Output = io::Result<()>> {
        async { Ok(()) }
    }
}

/// A buffer that a disk reads pages into, from [`Disk::read`].
///
/// It dereferences to its bytes, which the read fills, and implements compio's
/// buffer traits, so that a compio file can read into it. A read returns it with
/// its outcome. Its bytes are always initialized: as with compio's own buffers,
/// `IoBufMut::as_uninit` lends them as possibly uninitialized only so that a read
/// can fill them, and a disk must not write uninitialized bytes through it.
pub struct ReadBuf(Mem);

/// The memory a [`ReadBuf`] lends: a page frame of the store's pool, which the
/// cache keeps, or a buffer for a value leaf too large to cache.
enum Mem {
    Frame(FrameBuf),
    Value(ValueBuf),
}

impl ReadBuf {
    pub(crate) fn frame(buf: FrameBuf) -> ReadBuf {
        ReadBuf(Mem::Frame(buf))
    }

    pub(crate) fn value(buf: ValueBuf) -> ReadBuf {
        ReadBuf(Mem::Value(buf))
    }

    /// The value leaf's buffer this is, if it is one.
    pub(crate) fn into_value(self) -> Option<ValueBuf> {
        match self.0 {
            Mem::Value(buf) => Some(buf),
            Mem::Frame(_) => None,
        }
    }
}

impl Deref for ReadBuf {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match &self.0 {
            Mem::Frame(buf) => buf.bytes(),
            Mem::Value(buf) => buf.bytes(),
        }
    }
}

impl DerefMut for ReadBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        match &mut self.0 {
            Mem::Frame(buf) => buf.bytes_mut(),
            Mem::Value(buf) => buf.bytes_mut(),
        }
    }
}

impl IoBuf for ReadBuf {
    fn as_init(&self) -> &[u8] {
        self
    }
}

impl SetLen for ReadBuf {
    unsafe fn set_len(&mut self, _len: usize) {
        // A read fills the whole buffer, which is initialized anyway.
    }
}

impl IoBufMut for ReadBuf {
    fn as_uninit(&mut self) -> &mut [MaybeUninit<u8>] {
        let bytes: &mut [u8] = self;
        // SAFETY: `MaybeUninit<u8>` has the layout of `u8`, and only initialized
        // bytes can be written through it.
        unsafe { &mut *(bytes as *mut [u8] as *mut [MaybeUninit<u8>]) }
    }
}

/// Pages that a disk writes, from [`Disk::write`] and [`Disk::write_vectored`].
///
/// They dereference to their bytes, which the disk can read but not change, and
/// implement compio's buffer traits, so that a compio file can write them. A write
/// returns them with its outcome.
pub struct WriteBuf(FrameBuf);

impl WriteBuf {
    pub(crate) fn new(buf: FrameBuf) -> WriteBuf {
        WriteBuf(buf)
    }

    pub(crate) fn into_frame(self) -> FrameBuf {
        self.0
    }
}

impl Deref for WriteBuf {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.0.bytes()
    }
}

impl IoBuf for WriteBuf {
    fn as_init(&self) -> &[u8] {
        self
    }
}

/// The disk of a store in a file, which [`Db::open`](crate::Db::open) and
/// [`ReadOnlyDb::open`](crate::ReadOnlyDb::open) open. It reads and writes the file
/// with direct I/O on the current compio runtime, and runs the scan that follows
/// an open there.
pub struct FileDisk {
    file: File,
    /// The task that runs the scan that follows the open, which dropping the disk
    /// cancels.
    task: Cell<Option<JoinHandle<()>>>,
}

impl FileDisk {
    /// Opens the file at `path`, creating it if needed, and locks it until it
    /// closes. Returns it with its length.
    ///
    /// Fails with [`Error::Locked`](crate::Error::Locked) if another open of the
    /// file holds the lock, in this process or another.
    pub(crate) async fn open(path: &Path) -> Result<(FileDisk, u64)> {
        let file = open_uncached(path, true).await?;
        lock(&file)?;
        let len = file.metadata().await?.len();
        if len == 0 {
            // A new file's name is durable only once its directory is synced.
            sync_dir(path).await?;
        }
        Ok((FileDisk::new(file), len))
    }

    /// Opens the existing file at `path` for reading only, without locking it.
    /// Returns it with its length.
    pub(crate) async fn open_read_only(path: &Path) -> Result<(FileDisk, u64)> {
        let file = open_uncached(path, false).await?;
        let len = file.metadata().await?.len();
        Ok((FileDisk::new(file), len))
    }

    fn new(file: File) -> FileDisk {
        FileDisk {
            file,
            task: Cell::new(None),
        }
    }
}

impl Disk for FileDisk {
    type Class = ();
    type Read = FileRead;

    async fn len(&self) -> io::Result<u64> {
        Ok(self.file.metadata().await?.len())
    }

    fn read(&self, buf: ReadBuf, offset: u64, _: ()) -> FileRead {
        debug_assert_blocks(&buf, offset);
        FileRead::new(self.file.to_shared_fd(), buf, offset)
    }

    async fn write(&self, buf: WriteBuf, offset: u64, _: ()) -> (io::Result<()>, WriteBuf) {
        debug_assert_blocks(&buf, offset);
        let mut file = &self.file;
        let BufResult(res, buf) = file.write_all_at(buf, offset).await;
        (res, buf)
    }

    async fn write_vectored(
        &self,
        bufs: Vec<WriteBuf>,
        offset: u64,
        _: (),
    ) -> (io::Result<()>, Vec<WriteBuf>) {
        for buf in &bufs {
            debug_assert_blocks(buf, offset);
        }
        let mut file = &self.file;
        let BufResult(res, bufs) = file.write_vectored_all_at(bufs, offset).await;
        (res, bufs)
    }

    async fn set_len(&self, len: u64, _: ()) -> io::Result<()> {
        self.file.set_len(len).await
    }

    async fn sync(&self, _: ()) -> io::Result<()> {
        self.file.sync_data().await
    }

    fn spawn(&self, task: impl Future<Output = ()> + 'static) {
        self.task.set(Some(compio::runtime::spawn(task)));
    }

    async fn close(self) -> io::Result<()> {
        self.file.close().await
    }
}

/// A read from a [`FileDisk`]: it fills its buffer, reading the rest after a short
/// read, as `read_exact_at` does. Dropping it mid-read is safe: the runtime keeps
/// the buffer until the read finishes.
pub struct FileRead {
    fd: SharedFd<std::fs::File>,
    offset: u64,
    op: Submit<ReadAt<Slice<ReadBuf>, SharedFd<std::fs::File>>>,
}

impl FileRead {
    fn new(fd: SharedFd<std::fs::File>, buf: ReadBuf, offset: u64) -> FileRead {
        let op = compio::runtime::submit(ReadAt::new(fd.clone(), offset, buf.slice(..)));
        FileRead { fd, offset, op }
    }
}

impl Future for FileRead {
    type Output = (io::Result<()>, ReadBuf);

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            let BufResult(res, op) = ready!(Pin::new(&mut self.op).poll(cx));
            let mut rest = op.into_inner();
            let (read, len) = (rest.begin(), rest.as_inner().buf_len());
            match res {
                Ok(0) => {
                    let eof = io::Error::from(io::ErrorKind::UnexpectedEof);
                    return Poll::Ready((Err(eof), rest.into_inner()));
                }
                Ok(n) if read + n == len => return Poll::Ready((Ok(()), rest.into_inner())),
                Ok(n) => rest.set_begin(read + n),
                Err(e) => return Poll::Ready((Err(e), rest.into_inner())),
            }
            let at = self.offset + rest.begin() as u64;
            self.op = compio::runtime::submit(ReadAt::new(self.fd.clone(), at, rest));
        }
    }
}

/// Checks that file I/O is block-aligned, as direct I/O requires, even where it
/// isn't enforced, like on macOS.
fn debug_assert_blocks(buf: &[u8], offset: u64) {
    let (ptr, len) = (buf.as_ptr() as u64, buf.len() as u64);
    debug_assert!(
        ptr.is_multiple_of(BLOCK) && len.is_multiple_of(BLOCK) && offset.is_multiple_of(BLOCK),
        "unaligned file I/O: {len} bytes at offset {offset}"
    );
}

/// Opens a file for I/O that bypasses the OS page cache, which would otherwise keep
/// a second copy of every page: `O_DIRECT` on Linux, where the file system supports
/// it, and `F_NOCACHE` on macOS. A writable open creates the file if needed.
async fn open_uncached(path: &Path, writable: bool) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(writable).create(writable);
    #[cfg(target_os = "linux")]
    {
        let mut direct = options.clone();
        direct.custom_flags(libc::O_DIRECT);
        match direct.open(path).await {
            Ok(file) => return Ok(file),
            // Some file systems, like tmpfs, don't support O_DIRECT.
            Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {}
            Err(e) => return Err(e),
        }
    }
    let file = options.open(path).await?;
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: the descriptor is open, and F_NOCACHE takes an int.
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(file)
}

/// Takes an exclusive lock on `file`, which closing it releases. A `flock` lock
/// belongs to this open of the file rather than to the process, so it also keeps
/// out a second open by another thread of this process.
#[cfg(unix)]
fn lock(file: &File) -> Result<()> {
    use std::os::fd::AsRawFd;
    // SAFETY: the descriptor is open, and flock takes an int.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == -1 {
        let e = io::Error::last_os_error();
        return Err(match e.kind() {
            io::ErrorKind::WouldBlock => crate::Error::Locked,
            _ => e.into(),
        });
    }
    Ok(())
}

/// Syncs the directory holding `path`, which makes the names in it durable.
#[cfg(unix)]
async fn sync_dir(path: &Path) -> io::Result<()> {
    File::open(dir_of(path)).await?.sync_all().await
}

/// Elsewhere, encina neither locks its file nor syncs its directory.
#[cfg(not(unix))]
fn lock(_: &File) -> Result<()> {
    Ok(())
}

#[cfg(not(unix))]
async fn sync_dir(_: &Path) -> io::Result<()> {
    Ok(())
}

/// The directory holding `path`: a bare file name is in the working directory.
#[cfg(unix)]
fn dir_of(path: &Path) -> &Path {
    match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::pool::{Holds, Pool};

    #[test]
    fn a_file_name_alone_is_in_the_working_directory() {
        assert_eq!(dir_of(Path::new("encina.db")), Path::new("."));
        assert_eq!(dir_of(Path::new("data/encina.db")), Path::new("data"));
        assert_eq!(dir_of(Path::new("/encina.db")), Path::new("/"));
    }

    #[compio::test]
    #[cfg_attr(miri, ignore)] // Miri can't run compio's runtime.
    async fn reads_past_the_end_fail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file");
        // A block and a half.
        std::fs::write(&path, vec![7; BLOCK as usize * 3 / 2]).unwrap();
        let (disk, _) = FileDisk::open(&path).await.unwrap();
        let mut pool = Pool::new(BLOCK as usize);
        let frame = pool.alloc(Holds::Loading, 0);
        let read = |offset| disk.read(ReadBuf::frame(pool.buf(frame)), offset, ());
        let (res, buf) = read(0).await;
        res.unwrap();
        drop(buf);
        assert!(pool.page(frame).iter().all(|&b| b == 7));
        // The second block is cut short: its read gets the half there, then fails
        // to read the rest.
        let (res, buf) = read(BLOCK).await;
        assert!(res.is_err());
        drop(buf);
        let (res, _) = read(2 * BLOCK).await;
        assert_eq!(res.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    }
}
