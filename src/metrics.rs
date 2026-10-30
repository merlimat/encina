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

//! The metrics a store exports through OpenTelemetry, recorded with [nanometer]
//! handles: a few nanoseconds a record, without allocating.
//!
//! A store creates its handles when it opens, with its attributes: `encina.store`,
//! its path, then those its options add, which can replace it. Without a meter
//! provider in its options it creates none, and recording does nothing.
//!
//! Amounts a store holds, such as its cache's memory, are up-down counters rather
//! than gauges, so that the stores of a process add up to one series per set of
//! attributes. Each store adds what changes, and takes its share back out when it
//! closes.

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::time::Instant;

use nanometer::{
    CounterDef, HistogramDef, LatencyDef, LocalCounter, LocalHistogram, LocalLatency,
    LocalUpDownCounter, Timer, UpDownCounterDef,
};
use opentelemetry::KeyValue;

use crate::error::Error;
use crate::options::Options;

/// The instrumentation scope of encina's metrics.
pub(crate) const SCOPE: &str = "encina";

const WRITE_DURATION: LatencyDef = LatencyDef::new("encina.write.duration")
    .description("Time a successful write takes, from the call until its batch is durable");
const WRITE_OPERATIONS: CounterDef = CounterDef::new("encina.write.operations")
    .unit("{operation}")
    .description("Operations in the batches written successfully");
const WRITE_SIZE: HistogramDef = HistogramDef::bytes("encina.write.size")
    .description("Bytes of keys and values in each batch written successfully");
const WRITE_PENDING: UpDownCounterDef = UpDownCounterDef::new("encina.write.pending")
    .unit("{write}")
    .description("Writes waiting for their commit or committing");
const WRITE_ERRORS: CounterDef = CounterDef::new("encina.write.errors")
    .unit("{error}")
    .description("Writes that failed");
const COMMIT_DURATION: LatencyDef = LatencyDef::new("encina.commit.duration")
    .description("Time a commit takes, from planning its batches to publishing its version");
const COMMIT_WRITES: HistogramDef = HistogramDef::counts("encina.commit.writes")
    .unit("{write}")
    .description("Writes whose batches each commit makes durable");
const GET_DURATION: LatencyDef =
    LatencyDef::new("encina.get.duration").description("Time a successful lookup takes");
const GET_ERRORS: CounterDef = CounterDef::new("encina.get.errors")
    .unit("{error}")
    .description("Lookups that failed");
const RANGE_ITERATORS: CounterDef = CounterDef::new("encina.range.iterators")
    .unit("{iterator}")
    .description("Range iterators created");
const RANGE_ENTRIES: CounterDef = CounterDef::new("encina.range.entries")
    .unit("{entry}")
    .description("Entries that range iterators returned");
const READER_COUNT: UpDownCounterDef = UpDownCounterDef::new("encina.reader.count")
    .unit("{reader}")
    .description("Reads that pin a version: iterators, and lookups waiting for the disk");
const CACHE_LOOKUPS: CounterDef = CounterDef::new("encina.cache.lookups")
    .unit("{lookup}")
    .description("Leaves that reads and writes looked for in the cache");
const CACHE_EVICTIONS: CounterDef = CounterDef::new("encina.cache.evictions")
    .unit("{leaf}")
    .description("Leaves evicted from the cache to make room");
const CACHE_USAGE: UpDownCounterDef = UpDownCounterDef::new("encina.cache.usage")
    .unit("By")
    .description("Memory that the page frames in use take");
const CACHE_LIMIT: UpDownCounterDef = UpDownCounterDef::new("encina.cache.limit")
    .unit("By")
    .description("The memory budget for page frames");
const DISK_IO: CounterDef = CounterDef::new("encina.disk.io")
    .unit("By")
    .description("Bytes read from and written to the store file");
const DISK_OPERATIONS: CounterDef = CounterDef::new("encina.disk.operations")
    .unit("{operation}")
    .description("Read and write requests to the store file");
const DISK_READ_DURATION: LatencyDef = LatencyDef::new("encina.disk.read.duration")
    .description("Time a read of a leaf or a value leaf takes");
const DISK_SYNC_DURATION: LatencyDef =
    LatencyDef::new("encina.disk.sync.duration").description("Time a sync of the store file takes");
const ENTRY_COUNT: UpDownCounterDef = UpDownCounterDef::new("encina.entry.count")
    .unit("{entry}")
    .description("Entries in the store's current version");
const ENTRY_SIZE: UpDownCounterDef = UpDownCounterDef::new("encina.entry.size")
    .unit("By")
    .description("Bytes of the keys and values of the entries in the store's current version");
const FILE_SIZE: UpDownCounterDef = UpDownCounterDef::new("encina.file.size")
    .unit("By")
    .description("The store file's size");
const FILE_USAGE: UpDownCounterDef = UpDownCounterDef::new("encina.file.usage")
    .unit("By")
    .description("The store file's pages: in the current version, held back by readers, or free");
const STORE_FAILED: UpDownCounterDef = UpDownCounterDef::new("encina.store.failed")
    .unit("{store}")
    .description("Stores that reject writes until reopened, after an I/O error in a commit");

const STORE: &str = "encina.store";
const OPERATION_TYPE: &str = "encina.operation.type";
const CACHE_RESULT: &str = "encina.cache.result";
const PAGE_TYPE: &str = "encina.page.type";
const PAGE_STATE: &str = "encina.page.state";
const DIRECTION: &str = "disk.io.direction";
const ERROR_TYPE: &str = "error.type";

/// Creates a store's metric handles, with its attributes.
#[derive(Clone, Default)]
pub(crate) struct Meter {
    meter: Option<nanometer::Meter>,
    attributes: Vec<KeyValue>,
}

impl Meter {
    /// The meter of a store opened at `path` with `opts`.
    pub(crate) fn new(opts: &Options, path: &Path) -> Meter {
        let Some(meter) = opts.meter.clone() else {
            return Meter::default();
        };
        let store = KeyValue::new(STORE, path.to_string_lossy().into_owned());
        // The options' attributes go last: of a key given twice, the last value wins.
        let attributes = std::iter::once(store)
            .chain(opts.metric_attributes.iter().cloned())
            .collect();
        Meter {
            meter: Some(meter),
            attributes,
        }
    }

    /// The store's attributes, then `extra`.
    fn attributes(&self, extra: &[KeyValue]) -> Vec<KeyValue> {
        let mut attributes = self.attributes.clone();
        attributes.extend_from_slice(extra);
        attributes
    }

    fn counter(&self, def: CounterDef, extra: &[KeyValue]) -> Counter {
        Counter(
            self.meter
                .as_ref()
                .map(|m| m.counter(def, &self.attributes(extra))),
        )
    }

    fn amount(&self, def: UpDownCounterDef, extra: &[KeyValue]) -> Amount {
        Amount {
            counter: self
                .meter
                .as_ref()
                .map(|m| m.up_down_counter(def, &self.attributes(extra))),
            value: Cell::new(0),
        }
    }

    fn histogram(&self, def: HistogramDef) -> Histogram {
        Histogram(
            self.meter
                .as_ref()
                .map(|m| m.histogram(def, &self.attributes)),
        )
    }

    fn latency(&self, def: LatencyDef) -> Latency {
        Latency(
            self.meter
                .as_ref()
                .map(|m| m.latency(def, &self.attributes)),
        )
    }

    /// The memory of the frames in use, by what they hold: internal nodes above
    /// level 1, leaves and level-1 nodes, and value leaves. Pages being read or
    /// written count as leaves.
    pub(crate) fn cache_usage(&self) -> [Amount; 3] {
        ["internal", "leaf", "value"]
            .map(|kind| self.amount(CACHE_USAGE, &[KeyValue::new(PAGE_TYPE, kind)]))
    }

    /// The cache's hits and misses.
    pub(crate) fn cache_lookups(&self) -> [Counter; 2] {
        ["hit", "miss"]
            .map(|result| self.counter(CACHE_LOOKUPS, &[KeyValue::new(CACHE_RESULT, result)]))
    }

    pub(crate) fn cache_evictions(&self) -> Counter {
        self.counter(CACHE_EVICTIONS, &[])
    }

    pub(crate) fn disk_read_duration(&self) -> Latency {
        self.latency(DISK_READ_DURATION)
    }
}

/// The handles a store records with, but for those of its pool, cache and reads.
pub(crate) struct Metrics {
    pub(crate) write: Latency,
    /// Puts, deletes and range deletes.
    pub(crate) write_operations: [Counter; 3],
    pub(crate) write_size: Histogram,
    pub(crate) write_pending: Amount,
    pub(crate) write_errors: Errors,
    pub(crate) commit: Latency,
    pub(crate) commit_writes: Histogram,
    pub(crate) get: Latency,
    pub(crate) get_errors: Errors,
    pub(crate) range_iterators: Counter,
    pub(crate) range_entries: Counter,
    pub(crate) readers: Amount,
    pub(crate) cache_limit: Amount,
    pub(crate) read: Io,
    pub(crate) written: Io,
    pub(crate) read_duration: Latency,
    pub(crate) sync_duration: Latency,
    pub(crate) entry_count: Amount,
    pub(crate) entry_size: Amount,
    pub(crate) file_size: Amount,
    /// The file's pages: live, held back by readers, and free.
    pub(crate) file_usage: [Amount; 3],
    pub(crate) failed: Amount,
}

impl Metrics {
    /// The handles of a store, from `meter`. A read-only view leaves out writes, the
    /// entries and the file, which the store's owner reports.
    pub(crate) fn new(meter: &Meter, read_only: bool) -> Metrics {
        let none = Meter::default();
        let owner = if read_only { &none } else { meter };
        Metrics {
            write: owner.latency(WRITE_DURATION),
            write_operations: ["put", "delete", "delete_range"].map(|kind| {
                owner.counter(WRITE_OPERATIONS, &[KeyValue::new(OPERATION_TYPE, kind)])
            }),
            write_size: owner.histogram(WRITE_SIZE),
            write_pending: owner.amount(WRITE_PENDING, &[]),
            write_errors: Errors::new(owner, WRITE_ERRORS),
            commit: owner.latency(COMMIT_DURATION),
            commit_writes: owner.histogram(COMMIT_WRITES),
            get: meter.latency(GET_DURATION),
            get_errors: Errors::new(meter, GET_ERRORS),
            range_iterators: meter.counter(RANGE_ITERATORS, &[]),
            range_entries: meter.counter(RANGE_ENTRIES, &[]),
            readers: meter.amount(READER_COUNT, &[]),
            cache_limit: meter.amount(CACHE_LIMIT, &[]),
            read: Io::new(meter, "read"),
            written: Io::new(owner, "write"),
            read_duration: meter.disk_read_duration(),
            sync_duration: owner.latency(DISK_SYNC_DURATION),
            entry_count: owner.amount(ENTRY_COUNT, &[]),
            entry_size: owner.amount(ENTRY_SIZE, &[]),
            file_size: owner.amount(FILE_SIZE, &[]),
            file_usage: ["live", "held", "free"]
                .map(|state| owner.amount(FILE_USAGE, &[KeyValue::new(PAGE_STATE, state)])),
            failed: owner.amount(STORE_FAILED, &[]),
        }
    }
}

/// The bytes and requests of one direction of I/O.
pub(crate) struct Io {
    bytes: Counter,
    operations: Counter,
}

impl Io {
    fn new(meter: &Meter, direction: &'static str) -> Io {
        let attributes = [KeyValue::new(DIRECTION, direction)];
        Io {
            bytes: meter.counter(DISK_IO, &attributes),
            operations: meter.counter(DISK_OPERATIONS, &attributes),
        }
    }

    /// Counts a request of `bytes` bytes.
    pub(crate) fn add(&self, bytes: usize) {
        self.bytes.add(bytes as u64);
        self.operations.add(1);
    }
}

/// A counter, or nothing for a store without metrics.
#[derive(Default)]
pub(crate) struct Counter(Option<LocalCounter>);

impl Counter {
    pub(crate) fn add(&self, n: u64) {
        if let Some(counter) = &self.0 {
            counter.add(n);
        }
    }
}

/// A store's share of an up-down counter: an amount the store holds, which it takes
/// back out when dropped.
#[derive(Default)]
pub(crate) struct Amount {
    counter: Option<LocalUpDownCounter>,
    value: Cell<i64>,
}

impl Amount {
    pub(crate) fn add(&self, n: i64) {
        if let Some(counter) = &self.counter {
            self.value.set(self.value.get() + n);
            counter.add(n);
        }
    }

    pub(crate) fn set(&self, value: i64) {
        self.add(value - self.value.get());
    }

    /// Adds one, until the guard it returns is dropped.
    pub(crate) fn hold(&self) -> Hold<'_> {
        self.add(1);
        Hold(self)
    }
}

impl Drop for Amount {
    fn drop(&mut self) {
        self.set(0);
    }
}

/// One added to an [`Amount`], until dropped.
pub(crate) struct Hold<'a>(&'a Amount);

impl Drop for Hold<'_> {
    fn drop(&mut self) {
        self.0.add(-1);
    }
}

/// A histogram, or nothing for a store without metrics.
#[derive(Default)]
pub(crate) struct Histogram(Option<LocalHistogram>);

impl Histogram {
    pub(crate) fn record(&self, value: u64) {
        if let Some(histogram) = &self.0 {
            histogram.record(value);
        }
    }
}

/// A latency histogram, or nothing for a store without metrics.
#[derive(Default)]
pub(crate) struct Latency(Option<LocalLatency>);

impl Latency {
    /// Starts timing an operation, which [`Timing::stop`] records.
    pub(crate) fn start(&self) -> Timing<'_> {
        Timing(self.0.as_ref().map(LocalLatency::start))
    }

    /// The time now, if this records anything: for an operation that can't hold a
    /// [`Timing`], which [`record_since`](Self::record_since) records.
    pub(crate) fn now(&self) -> Option<Instant> {
        self.0.as_ref().map(|_| Instant::now())
    }

    pub(crate) fn record_since(&self, start: Option<Instant>) {
        if let (Some(latency), Some(start)) = (&self.0, start) {
            latency.record(start.elapsed());
        }
    }
}

/// An operation being timed. Dropped without [`stop`](Self::stop), as when the
/// operation fails or its future is dropped, it records nothing.
pub(crate) struct Timing<'a>(Option<Timer<'a>>);

impl Timing<'_> {
    pub(crate) fn stop(mut self) {
        if let Some(timer) = self.0.take() {
            timer.stop();
        }
    }
}

impl Drop for Timing<'_> {
    fn drop(&mut self) {
        if let Some(timer) = self.0.take() {
            timer.discard();
        }
    }
}

/// Counts errors by `error.type`, with a counter for each type, created when it
/// counts its first error.
pub(crate) struct Errors {
    meter: Meter,
    def: CounterDef,
    counters: RefCell<Vec<(&'static str, LocalCounter)>>,
}

impl Errors {
    fn new(meter: &Meter, def: CounterDef) -> Errors {
        Errors {
            meter: meter.clone(),
            def,
            counters: RefCell::new(Vec::new()),
        }
    }

    pub(crate) fn add(&self, e: &Error) {
        let Some(meter) = &self.meter.meter else {
            return;
        };
        let kind = error_type(e);
        let mut counters = self.counters.borrow_mut();
        let i = match counters.iter().position(|&(k, _)| k == kind) {
            Some(i) => i,
            None => {
                let attributes = self.meter.attributes(&[KeyValue::new(ERROR_TYPE, kind)]);
                counters.push((kind, meter.counter(self.def, &attributes)));
                counters.len() - 1
            }
        };
        counters[i].1.add(1);
    }
}

/// The `error.type` of `e`.
fn error_type(e: &Error) -> &'static str {
    match e {
        Error::KeyTooLarge { .. } => "key_too_large",
        Error::InvalidOptions(_) => "invalid_options",
        Error::PageSizeMismatch { .. } => "page_size_mismatch",
        Error::InvalidFile(_) => "invalid_file",
        Error::Locked => "locked",
        Error::Corruption { .. } => "corruption",
        Error::Io(_) => "io",
        Error::Failed => "failed",
        Error::Stale => "stale",
    }
}
