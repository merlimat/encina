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
    EntryTooLarge { size: usize, max: usize },
    /// The options are invalid.
    InvalidOptions(&'static str),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::EntryTooLarge { size, max } => {
                write!(f, "entry of {size} bytes exceeds the maximum of {max}")
            }
            Error::InvalidOptions(reason) => write!(f, "invalid options: {reason}"),
        }
    }
}

impl std::error::Error for Error {}
