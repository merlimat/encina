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

//! An embedded, ordered key-value store built on the [compio] runtime, designed for
//! many small key-value pairs.
//!
//! encina is a copy-on-write B+tree. It favors large batched writes, point and
//! ordered lookups, range scans, and deleting whole key ranges at once.
//!
//! - **Durable, atomic batches.** [`Db::write`] applies a [`WriteBatch`] in one step
//!   and returns once it is on disk.
//! - **Lookups without copies.** [`Db::get`] finds a key, or its nearest neighbor
//!   under a [`Comparison`], and lends the entry to a closure. [`Db::range`]
//!   iterates over a point-in-time view, in key order.
//! - **Bounded memory.** Internal nodes stay in memory, and leaves are cached within
//!   [`Options::cache_size`]. The store bypasses the OS page cache.
//! - **One owner per store.** A [`Db`] is neither `Send` nor `Sync`: each compio
//!   runtime thread owns its stores, and a store file has one `Db` at a time. A
//!   [`ReadOnlyDb`] can read it meanwhile, as of when the view opened.
//!
//! Keys sort bytewise, so encode integers big-endian. A key may not exceed
//! [`Options::max_key_size`], a bit under a quarter of the page size. Values may be
//! any size that fits in memory: an entry too large for a leaf gets pages of its own.
//!
//! # Example
//!
//! ```
//! use encina::{Comparison, Db, Options, WriteBatch};
//!
//! # compio::runtime::Runtime::new().unwrap().block_on(async {
//! # let dir = tempfile::tempdir().unwrap();
//! # let path = dir.path().join("data.db");
//! let db = Db::open(&path, Options::default()).await?;
//!
//! let mut batch = WriteBatch::new();
//! batch.put(b"user/0001", b"alice");
//! batch.put(b"user/0002", b"bob");
//! batch.put(b"session/42", b"token");
//! db.write(&mut batch).await?;
//!
//! // A cleared batch keeps its memory. This one removes every key starting with
//! // "session/".
//! batch.clear();
//! batch.delete_range(b"session/", b"session0");
//! db.write(&mut batch).await?;
//!
//! // The entry with the largest key at or below "user/9999". `get` lends the key
//! // and value to a closure instead of allocating copies.
//! let last = db
//!     .get(b"user/9999", Comparison::Floor, |k, _v| String::from_utf8_lossy(k).into_owned())
//!     .await?;
//! assert_eq!(last.as_deref(), Some("user/0002"));
//!
//! // A point-in-time scan of every key starting with "user/".
//! let mut names = Vec::new();
//! let mut iter = db.range(&b"user/"[..]..&b"user0"[..]);
//! while let Some((_k, v)) = iter.next().await? {
//!     names.push(String::from_utf8_lossy(v).into_owned());
//! }
//! assert_eq!(names, ["alice", "bob"]);
//! # Ok::<_, encina::Error>(())
//! # }).unwrap();
//! ```
//!
//! # Metrics
//!
//! With [`Options::meter_provider`], a store exports metrics through OpenTelemetry,
//! under the scope `encina`, recorded with [nanometer](https://docs.rs/nanometer)
//! in a few nanoseconds each. Each series has the attribute `encina.store`, the
//! path the store was opened at, and those of [`Options::metric_attributes`].
//!
//! Amounts a store holds are up-down counters, so that the stores of a process add
//! up. A store takes its share back out when it closes. A [`ReadOnlyDb`] reports
//! neither writes nor its file, which the store's owner reports.
//!
//! | Metric | Type | Unit | What it measures |
//! |---|---|---|---|
//! | `encina.write.duration` | histogram | `s` | Each successful [`Db::write`], waiting for its commit included |
//! | `encina.write.operations` | counter | `{operation}` | Operations written, by `encina.operation.type`: `put`, `delete`, `delete_range` |
//! | `encina.write.size` | histogram | `By` | The keys and values of each batch written |
//! | `encina.write.pending` | up-down counter | `{write}` | Writes waiting for their commit or committing |
//! | `encina.write.errors` | counter | `{error}` | Failed writes, by `error.type` |
//! | `encina.commit.duration` | histogram | `s` | Each commit, from planning its batches to publishing its version |
//! | `encina.commit.writes` | histogram | `{write}` | The writes each commit makes durable together |
//! | `encina.get.duration` | histogram | `s` | Each successful [`Db::get`] |
//! | `encina.get.errors` | counter | `{error}` | Failed lookups, by `error.type` |
//! | `encina.range.iterators` | counter | `{iterator}` | Iterators [`Db::range`] created |
//! | `encina.range.entries` | counter | `{entry}` | Entries iterators returned, counted when they are dropped |
//! | `encina.reader.count` | up-down counter | `{reader}` | Iterators, and lookups waiting for the disk: they pin a version |
//! | `encina.cache.lookups` | counter | `{lookup}` | Leaves that reads and writes needed, by `encina.cache.result`: `hit`, `miss` |
//! | `encina.cache.evictions` | counter | `{leaf}` | Leaves evicted to make room |
//! | `encina.cache.usage` | up-down counter | `By` | Page frames in use, by `encina.page.type`: `internal`, `leaf`, `value` |
//! | `encina.cache.limit` | up-down counter | `By` | The budget for page frames, [`Options::cache_size`] |
//! | `encina.disk.io` | counter | `By` | Bytes read and written, by `disk.io.direction`: `read`, `write` |
//! | `encina.disk.operations` | counter | `{operation}` | Read and write requests, by `disk.io.direction` |
//! | `encina.disk.read.duration` | histogram | `s` | Each read of a leaf |
//! | `encina.disk.sync.duration` | histogram | `s` | Each sync, two per commit |
//! | `encina.file.usage` | up-down counter | `By` | The file's pages, by `encina.page.state`: `live` in the current version, `held` back for older versions' readers, or `free` |
//! | `encina.store.failed` | up-down counter | `{store}` | 1 while the store rejects writes, after an I/O error in a commit |
//!
//! Histograms export only through nanometer's `Producer`, which the app registers
//! with its metric reader.
//!
//! # Platforms
//!
//! Linux with io_uring is the target, and there the store file uses direct I/O.
//! macOS works for development, but without a durability guarantee against power
//! loss.
//!
//! [DESIGN.md](https://github.com/merlimat/encina/blob/main/DESIGN.md) describes the
//! design.
#![warn(missing_docs)]

mod batch;
mod cache;
mod db;
mod error;
mod meta;
mod metrics;
mod node;
mod options;
mod pool;
mod reads;
#[cfg(test)]
mod sim;
mod space;
mod storage;
mod tree;
mod writes;

#[cfg(test)]
mod alloc_tests;
#[cfg(test)]
mod tests;

pub use batch::WriteBatch;
pub use db::{Comparison, Db, Iter, PageInfo, ReadOnlyDb};
pub use error::{Error, Result};
pub use options::Options;
