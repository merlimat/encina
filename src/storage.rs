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

use std::io;
use std::path::Path;
#[cfg(test)]
use std::rc::Rc;

use compio::BufResult;
use compio::fs::{File, OpenOptions};
use compio::io::{AsyncReadAtExt, AsyncWriteAtExt};

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

    #[test]
    fn a_file_name_alone_is_in_the_working_directory() {
        assert_eq!(dir_of(Path::new("encina.db")), Path::new("."));
        assert_eq!(dir_of(Path::new("data/encina.db")), Path::new("data"));
        assert_eq!(dir_of(Path::new("/encina.db")), Path::new("/"));
    }
}
