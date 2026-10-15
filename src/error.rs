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
use std::io;

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
    /// A [`ReadOnlyDb`](crate::ReadOnlyDb) can no longer read its version: the
    /// store's owner has since overwritten pages of it. Open a new view to read the
    /// current version.
    Stale,
}

/// The result of encina's operations, which fail with an [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    /// A copy of the error, for each write of a group that failed together.
    pub(crate) fn duplicate(&self) -> Error {
        match self {
            Error::EntryTooLarge { size, max } => Error::EntryTooLarge {
                size: *size,
                max: *max,
            },
            Error::InvalidOptions(reason) => Error::InvalidOptions(reason),
            Error::InvalidFile(reason) => Error::InvalidFile(reason),
            Error::Locked => Error::Locked,
            Error::Corruption { page } => Error::Corruption { page: *page },
            Error::Io(e) => Error::Io(match e.raw_os_error() {
                Some(code) => io::Error::from_raw_os_error(code),
                None => io::Error::new(e.kind(), e.to_string()),
            }),
            Error::Failed => Error::Failed,
            Error::Stale => Error::Stale,
        }
    }
}

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
            Error::Stale => write!(f, "the version this read-only view reads was overwritten"),
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

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use super::*;

    #[test]
    fn errors_describe_themselves() {
        let io = Error::from(std::io::Error::other("disk on fire"));
        assert_eq!(io.to_string(), "I/O error: disk on fire");
        assert_eq!(io.source().unwrap().to_string(), "disk on fire");
        let others = [
            (
                Error::EntryTooLarge {
                    size: 2000,
                    max: 1000,
                },
                "entry of 2000 bytes exceeds the maximum of 1000",
            ),
            (Error::InvalidOptions("why"), "invalid options: why"),
            (Error::InvalidFile("why"), "invalid store file: why"),
            (Error::Locked, "the store is already open"),
            (Error::Corruption { page: 7 }, "page 7 is corrupt"),
            (
                Error::Failed,
                "an earlier I/O error left the store failed; reopen it",
            ),
            (
                Error::Stale,
                "the version this read-only view reads was overwritten",
            ),
        ];
        for (e, message) in others {
            assert_eq!(e.to_string(), message);
            assert_eq!(e.duplicate().to_string(), message);
            assert!(e.source().is_none(), "{e}");
        }
    }

    #[test]
    fn io_errors_duplicate_their_kind_and_message() {
        let os = Error::from(std::io::Error::from_raw_os_error(28));
        let custom = Error::from(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "short read",
        ));
        let io = |e: &Error| {
            let e = e
                .source()
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .unwrap();
            (e.kind(), e.raw_os_error(), e.to_string())
        };
        for e in [os, custom] {
            assert_eq!(io(&e.duplicate()), io(&e));
        }
    }
}
