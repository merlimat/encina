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

use crate::pool::FrameBuf;

pub(crate) enum Storage {
    File(File),
    #[cfg(test)]
    Sim(Rc<crate::sim::SimDisk>),
}

impl Storage {
    /// Opens the file at `path`, creating it if needed. Returns it with its length.
    pub(crate) async fn open(path: &Path) -> io::Result<(Storage, u64)> {
        let file = open_uncached(path).await?;
        let len = file.metadata().await?.len();
        Ok((Storage::File(file), len))
    }

    /// Fills `buf` from `offset`.
    pub(crate) async fn read(&self, buf: FrameBuf, offset: u64) -> (io::Result<()>, FrameBuf) {
        match self {
            Storage::File(file) => {
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
                let mut file = file;
                let BufResult(res, buf) = file.write_all_at(buf, offset).await;
                (res, buf)
            }
            #[cfg(test)]
            Storage::Sim(disk) => disk.write(buf, offset),
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
