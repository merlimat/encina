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

//! Where a store runs: the env that opens its disk and runs its task, and the disk
//! that keeps its pages. [`DefaultEnv`] keeps stores in files, on compio.

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
use compio::runtime::Submit;

use crate::error::Result;
use crate::pool::{FrameBuf, ValueBuf};

/// Direct I/O moves whole blocks: buffer, offset and length must all be multiples
/// of the device's block size, at most 4 KiB.
const BLOCK: u64 = 4096;

/// Where a store runs: what opens its file, runs its task, and lets it read the
/// clock, or not.
///
/// [`DefaultEnv`] keeps stores in files, on the current compio runtime. A caller
/// that runs its own I/O, such as a scheduler that orders the requests of several
/// stores, or a simulation that replays a run from its seed, implements this trait
/// and [`Disk`] instead. Either way, [`Db::open`](crate::Db::open) and
/// [`ReadOnlyDb::open`](crate::ReadOnlyDb::open) take the env, the store's path and
/// its options.
///
/// A store uses its env only while it opens, from the thread it runs on, and needs
/// nothing else from its runtime: it spawns no task but through
/// [`spawn`](Self::spawn), and reads no clock unless [`real_time`](Self::real_time)
/// lets it.
pub trait Env {
    /// The disk that [`open`](Self::open) opens.
    type Disk: Disk;

    /// Opens the store's file at `path`, which the env interprets: [`DefaultEnv`]
    /// opens it in the file system.
    ///
    /// For [`ReadWrite`](Access::ReadWrite), the store's owner, it creates the file
    /// if needed, empty, and keeps out other owners until the disk closes: two
    /// would corrupt the store. [`DefaultEnv`] fails a second owner's open with
    /// [`Error::Locked`](crate::Error::Locked). A file it creates must outlast a
    /// crash once a sync of the disk has finished. For
    /// [`ReadOnly`](Access::ReadOnly), a view, it opens the file as it is.
    fn open(&self, path: &Path, access: Access) -> impl Future<Output = Result<Self::Disk>>;

    /// Runs `task` beside the store until it ends. A store that takes writes calls
    /// this as it opens, unless its tree is a single leaf: the task reads the
    /// leaves' parents to find the pages free for reuse, and until it has, writes
    /// take new pages at the end of the disk. It holds the store only while it is
    /// polled. Once the store is closed or dropped, which wakes it, it ends the next
    /// time it is polled.
    fn spawn(&self, task: impl Future<Output = ()> + 'static);

    /// Whether the store may read the real clock, to pace its writes: false by
    /// default.
    ///
    /// A large write pauses now and then to let other tasks run: with the clock,
    /// about every millisecond; without it, after a set amount of work, about a
    /// millisecond of a sorted load's merge, and so in the same places each time.
    /// Without the clock, given the same requests and the same outcomes from its
    /// disk, a store does the same, as a simulation that replays a run needs.
    fn real_time(&self) -> bool {
        false
    }
}

/// What a store opens its file for, in [`Env::open`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// To read and write it, as the store's owner, a [`Db`](crate::Db).
    ReadWrite,
    /// To read it only, as a view, a [`ReadOnlyDb`](crate::ReadOnlyDb).
    ReadOnly,
}

/// The disk a store keeps its pages on, which its [`Env`] opens.
///
/// [`DefaultEnv`] opens a [`FileDisk`], which reads and writes a file with direct
/// I/O on the current compio runtime. A caller that runs its own I/O implements
/// this trait, and an [`Env`] that opens it.
///
/// The disk holds one store, from offset 0: two meta pages, then the tree's. The
/// store calls it from the thread it runs on.
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

/// The env to open a store in unless the caller runs its own I/O: it keeps stores in
/// files. It opens each with direct I/O, as a [`FileDisk`], runs the store's task on
/// the current compio runtime, and lets writes pause by the real clock.
///
/// An owner's open locks the file until its disk closes, so that a second owner, in
/// this process or another, fails with [`Error::Locked`](crate::Error::Locked); a
/// view's takes no lock. A new file's directory is synced, which makes its name
/// durable.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultEnv;

impl Env for DefaultEnv {
    type Disk = FileDisk;

    async fn open(&self, path: &Path, access: Access) -> Result<FileDisk> {
        match access {
            Access::ReadWrite => FileDisk::open(path).await,
            Access::ReadOnly => FileDisk::open_read_only(path).await,
        }
    }

    fn spawn(&self, task: impl Future<Output = ()> + 'static) {
        compio::runtime::spawn(task).detach();
    }

    fn real_time(&self) -> bool {
        true
    }
}

/// The disk of a store in a file, which [`DefaultEnv`] opens. It reads and writes
/// the file with direct I/O on the current compio runtime.
pub struct FileDisk {
    file: File,
}

impl FileDisk {
    /// Opens the file at `path`, creating it if needed, and locks it until it
    /// closes.
    ///
    /// Fails with [`Error::Locked`](crate::Error::Locked) if another open of the
    /// file holds the lock, in this process or another.
    async fn open(path: &Path) -> Result<FileDisk> {
        let file = open_uncached(path, true).await?;
        lock(&file)?;
        if file.metadata().await?.len() == 0 {
            // A new file's name is durable only once its directory is synced.
            sync_dir(path).await?;
        }
        Ok(FileDisk { file })
    }

    /// Opens the existing file at `path` for reading only, without locking it.
    async fn open_read_only(path: &Path) -> Result<FileDisk> {
        let file = open_uncached(path, false).await?;
        Ok(FileDisk { file })
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
        let disk = FileDisk::open(&path).await.unwrap();
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
