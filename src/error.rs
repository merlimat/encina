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

use std::fmt;

/// Errors returned by encina.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A put's key plus value exceeds [`Options::max_entry_size`](crate::Options::max_entry_size).
    /// The batch containing it was not applied.
    EntryTooLarge {
        /// The entry's key plus value, in bytes.
        size: usize,
        /// The largest key plus value the store accepts, in bytes.
        max: usize,
    },
    /// The options are invalid.
    InvalidOptions(&'static str),
    /// The file is not a store this version of encina can open.
    InvalidFile(&'static str),
    /// Another [`Db`](crate::Db), in this process or another, has the store open.
    Locked,
    /// A page failed its checksum, or doesn't hold the node expected there.
    Corruption {
        /// The page's number: its offset in the file divided by the page size.
        page: u64,
    },
    /// An I/O error.
    Io(std::io::Error),
    /// An earlier I/O error left it unknown whether a write reached the disk. The
    /// store rejects further writes; reopen it to recover.
    Failed,
}

/// The result of encina's operations, which fail with an [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::EntryTooLarge { size, max } => {
                write!(f, "entry of {size} bytes exceeds the maximum of {max}")
            }
            Error::InvalidOptions(reason) => write!(f, "invalid options: {reason}"),
            Error::InvalidFile(reason) => write!(f, "invalid store file: {reason}"),
            Error::Locked => write!(f, "the store is already open"),
            Error::Corruption { page } => write!(f, "page {page} is corrupt"),
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Failed => write!(f, "an earlier I/O error left the store failed; reopen it"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
