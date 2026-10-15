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

//! Where pages are stored: a file, or in tests a simulated disk.

use std::future::Future;
use std::io;
use std::path::Path;
use std::pin::Pin;
#[cfg(test)]
use std::rc::Rc;
use std::task::{Context, Poll, ready};

use compio::BufResult;
use compio::buf::{IntoInner, IoBuf, Slice};
use compio::driver::op::ReadAt;
use compio::driver::{SharedFd, ToSharedFd};
use compio::fs::{File, OpenOptions};
use compio::io::{AsyncReadAtExt, AsyncWriteAtExt};
use compio::runtime::Submit;

use crate::error::Result;
use crate::pool::FrameBuf;

/// Direct I/O moves whole blocks: buffer, offset and length must all be multiples
/// of the device's block size, at most 4 KiB.
const BLOCK: u64 = 4096;

pub(crate) enum Storage {
    File(File),
    #[cfg(test)]
    Sim(Rc<crate::sim::SimDisk>),
}

impl Storage {
    /// Opens the file at `path`, creating it if needed, and locks it until it
    /// closes. Returns it with its length.
    ///
    /// Fails with [`Error::Locked`](crate::Error::Locked) if another open of the
    /// file holds the lock, in this process or another.
    pub(crate) async fn open(path: &Path) -> Result<(Storage, u64)> {
        let file = open_uncached(path).await?;
        lock(&file)?;
        let len = file.metadata().await?.len();
        if len == 0 {
            // A new file's name is durable only once its directory is synced.
            sync_dir(path).await?;
        }
        Ok((Storage::File(file), len))
    }

    /// The smallest read it supports: a block for a file, whose direct I/O moves
    /// whole blocks, or the tiny pages tests use on the simulated disk.
    pub(crate) fn min_read(&self) -> usize {
        match self {
            Storage::File(_) => BLOCK as usize,
            #[cfg(test)]
            Storage::Sim(_) => crate::options::MIN_PAGE_SIZE,
        }
    }

    /// Fills `buf` from `offset`.
    pub(crate) async fn read(&self, buf: FrameBuf, offset: u64) -> (io::Result<()>, FrameBuf) {
        match self {
            Storage::File(file) => {
                debug_assert_blocks(&buf, offset);
                let BufResult(res, buf) = file.read_exact_at(buf, offset).await;
                (res, buf)
            }
            #[cfg(test)]
            Storage::Sim(disk) => disk.read(buf, offset),
        }
    }

    /// Starts filling `buf` from `offset`. Unlike [`read`](Self::read), the read
    /// is a future of a known type, so it can be kept and polled by whichever
    /// reader waits for it.
    pub(crate) fn start_read(&self, buf: FrameBuf, offset: u64) -> PageRead {
        match self {
            Storage::File(file) => {
                debug_assert_blocks(&buf, offset);
                PageRead::File(FileRead::new(file.to_shared_fd(), buf, offset))
            }
            #[cfg(test)]
            Storage::Sim(disk) => {
                PageRead::Sim(crate::sim::SimRead::new(disk.clone(), buf, offset))
            }
        }
    }

    /// Writes all of `buf` at `offset`.
    pub(crate) async fn write(&self, buf: FrameBuf, offset: u64) -> (io::Result<()>, FrameBuf) {
        match self {
            Storage::File(file) => {
                debug_assert_blocks(&buf, offset);
                let mut file = file;
                let BufResult(res, buf) = file.write_all_at(buf, offset).await;
                (res, buf)
            }
            #[cfg(test)]
            Storage::Sim(disk) => disk.write(buf, offset),
        }
    }

    /// Writes the pages in `bufs` back to back from `offset`, in one request.
    pub(crate) async fn write_run(
        &self,
        bufs: Vec<FrameBuf>,
        offset: u64,
    ) -> (io::Result<()>, Vec<FrameBuf>) {
        match self {
            Storage::File(file) => {
                for buf in &bufs {
                    debug_assert_blocks(buf, offset);
                }
                let mut file = file;
                let BufResult(res, bufs) = file.write_vectored_all_at(bufs, offset).await;
                (res, bufs)
            }
            #[cfg(test)]
            Storage::Sim(disk) => disk.write_run(bufs, offset),
        }
    }

    /// Makes every completed write durable.
    pub(crate) async fn sync(&self) -> io::Result<()> {
        match self {
            Storage::File(file) => file.sync_data().await,
            #[cfg(test)]
            Storage::Sim(disk) => disk.sync(),
        }
    }
}

/// A read from [`Storage::start_read`]. Dropping it mid-read is safe: the runtime
/// keeps the buffer until the read finishes.
pub(crate) enum PageRead {
    File(FileRead),
    #[cfg(test)]
    Sim(crate::sim::SimRead),
}

impl Future for PageRead {
    type Output = (io::Result<()>, FrameBuf);

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            PageRead::File(read) => read.poll(cx),
            #[cfg(test)]
            PageRead::Sim(read) => read.poll(cx),
        }
    }
}

/// Fills a buffer from a file, reading the rest after a short read, as
/// `read_exact_at` does.
pub(crate) struct FileRead {
    fd: SharedFd<std::fs::File>,
    offset: u64,
    op: Submit<ReadAt<Slice<FrameBuf>, SharedFd<std::fs::File>>>,
}

impl FileRead {
    fn new(fd: SharedFd<std::fs::File>, buf: FrameBuf, offset: u64) -> FileRead {
        let op = compio::runtime::submit(ReadAt::new(fd.clone(), offset, buf.slice(..)));
        FileRead { fd, offset, op }
    }

    fn poll(&mut self, cx: &mut Context<'_>) -> Poll<(io::Result<()>, FrameBuf)> {
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
fn debug_assert_blocks(buf: &FrameBuf, offset: u64) {
    use compio::buf::IoBuf;
    let (ptr, len) = (buf.as_init().as_ptr() as u64, buf.as_init().len() as u64);
    debug_assert!(
        ptr.is_multiple_of(BLOCK) && len.is_multiple_of(BLOCK) && offset.is_multiple_of(BLOCK),
        "unaligned file I/O: {len} bytes at offset {offset}"
    );
}

/// Opens a file for I/O that bypasses the OS page cache, which would otherwise keep
/// a second copy of every page: `O_DIRECT` on Linux, where the file system supports
/// it, and `F_NOCACHE` on macOS.
async fn open_uncached(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true);
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
    async fn reads_past_the_end_fail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file");
        // A block and a half.
        std::fs::write(&path, vec![7; BLOCK as usize * 3 / 2]).unwrap();
        let (storage, _) = Storage::open(&path).await.unwrap();
        let mut pool = Pool::new(BLOCK as usize);
        let frame = pool.alloc(Holds::Loading, 0);
        let (res, buf) = storage.start_read(pool.buf(frame), 0).await;
        res.unwrap();
        drop(buf);
        assert!(pool.page(frame).iter().all(|&b| b == 7));
        // The second block is cut short: its read gets the half there, then fails
        // to read the rest.
        let (res, buf) = storage.start_read(pool.buf(frame), BLOCK).await;
        assert!(res.is_err());
        drop(buf);
        let (res, _) = storage.start_read(pool.buf(frame), 2 * BLOCK).await;
        assert_eq!(res.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    }
}
