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

//! Compares encina with RocksDB on a workload shaped like BookKeeper's entry
//! location index: 16-byte keys, a ledger id and an entry id, mapping to 16-byte
//! locations, written in large durable batches that append to many ledgers at once.
//!
//! A run benchmarks one engine, in a process of its own so that the CPU time, disk
//! traffic and memory it reports are that engine's alone. See README.md.

use std::cell::RefCell;
use std::fs;
use std::future::{Future, poll_fn};
use std::hint::black_box;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process;
use std::sync::Arc;
use std::task::Poll;
use std::time::{Duration, Instant};

use encina::Comparison;

const USAGE: &str = "\
usage: encina-bench --engine encina|rocksdb --dir DIR [options]

Creates the store at DIR/encina or DIR/rocksdb, which must not exist yet.

  --ledgers N    ledgers each batch appends to (1000)
  --entries N    entries each batch appends to each ledger (100)
  --batches N    batches to write (100)
  --rollover N   batches after which the ledgers close and new ones open (10)
  --writers N    writers, each appending to its share of the ledgers (1)
  --reads N      lookups in each get and floor phase (1000000)
  --cache-mb N   memory for cached pages or blocks, in MiB (512)";

fn main() {
    let cfg = Config::parse();
    let path = cfg.dir.join(cfg.engine.name());
    if path.exists() {
        fail(&format!("{} exists: remove it first", path.display()));
    }
    fs::create_dir_all(&cfg.dir).expect("create the directory");
    let runtime = compio::runtime::Runtime::new().expect("start the runtime");
    runtime.block_on(async {
        match cfg.engine {
            Engine::Encina => run(&cfg, Encina::open(&path, cfg.cache).await).await,
            Engine::RocksDb => run(&cfg, Rocks::open(&path, cfg.cache)).await,
        }
    });
}

async fn run(cfg: &Config, store: impl Store) {
    println!(
        "{}: {} entries in {} batches of {} ledgers x {} entries, new ledgers every {} batches, {} MiB cache{}",
        cfg.engine.name(),
        cfg.batches * cfg.ledgers * cfg.entries,
        cfg.batches,
        cfg.ledgers,
        cfg.entries,
        cfg.rollover,
        cfg.cache >> 20,
        match cfg.writers {
            1 => String::new(),
            n => format!(", {n} writers"),
        },
    );
    println!(
        "{:<7} {:>11} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7} {:>9} {:>9} {:>9}",
        "phase",
        "count",
        "time",
        "rate",
        "p50",
        "p99",
        "p99.9",
        "max",
        "cpu s",
        "read MiB",
        "write MiB",
        "size MiB",
    );
    let mut rng = Rng(0x5eed);
    load(cfg, &store).await;
    settle(&store).await;
    read(cfg, &store, 0, &mut rng).await;
    if let Some(first) = delete(cfg, &store).await {
        read(cfg, &store, first, &mut rng).await;
    }
    println!("peak memory: {:.1} MiB", mib(peak_rss()));
}

/// Writes the batches, committing each durably. Each appends to every open ledger,
/// in key order, as BookKeeper's flushes of its write cache do. With several
/// writers, each writes its share of every batch, as a batch of its own, at the
/// same time as the others.
async fn load(cfg: &Config, store: &impl Store) {
    let meter = RefCell::new(Meter::start(cfg.batches * cfg.writers));
    let writers = (0..cfg.writers).map(|w| {
        let mut writer = store.writer();
        let meter = &meter;
        async move {
            for batch in 0..cfg.batches {
                let start = Instant::now();
                let first = cfg.first_entry(batch);
                for slot in cfg.slots(w) {
                    let ledger = cfg.ledger(batch, slot);
                    for i in 0..cfg.entries {
                        writer.put(&key(ledger, first + i), &value(cfg.seq(batch, slot, i)));
                    }
                }
                writer.commit().await;
                meter.borrow_mut().record(start);
            }
        }
    });
    join_all(writers.collect()).await;
    let count = cfg.batches * cfg.ledgers * cfg.entries;
    meter.into_inner().finish("load", count, store);
}

/// Runs `futures` together on this task, until they all finish.
async fn join_all<F: Future<Output = ()>>(futures: Vec<F>) {
    let mut futures: Vec<Option<Pin<Box<F>>>> =
        futures.into_iter().map(|f| Some(Box::pin(f))).collect();
    poll_fn(|cx| {
        for slot in &mut futures {
            if let Some(future) = slot
                && future.as_mut().poll(cx).is_ready()
            {
                *slot = None;
            }
        }
        match futures.iter().all(Option::is_none) {
            true => Poll::Ready(()),
            false => Poll::Pending,
        }
    })
    .await;
}

/// Waits for the work the load left in the background, RocksDB's compactions, so
/// that it shows here rather than in the phases after.
async fn settle(store: &impl Store) {
    let meter = Meter::start(0);
    store.settle().await;
    meter.finish("settle", 0, store);
}

/// The read phases, over the entries the batches from `first` on wrote.
async fn read(cfg: &Config, store: &impl Store, first: u64, rng: &mut Rng) {
    scan(cfg, store, first).await;
    get(cfg, store, first, rng).await;
    floor(cfg, store, first, rng).await;
}

/// Reads every entry, in key order.
async fn scan(cfg: &Config, store: &impl Store, first: u64) {
    let meter = Meter::start(0);
    let count = store.scan().await;
    assert_eq!(count, (cfg.batches - first) * cfg.ledgers * cfg.entries);
    meter.finish("scan", count, store);
}

/// Looks up random keys, each of them present.
async fn get(cfg: &Config, store: &impl Store, first: u64, rng: &mut Rng) {
    let mut meter = Meter::start(cfg.reads);
    for _ in 0..cfg.reads {
        let batch = first + rng.below(cfg.batches - first);
        let (slot, i) = (rng.below(cfg.ledgers), rng.below(cfg.entries));
        let k = key(cfg.ledger(batch, slot), cfg.first_entry(batch) + i);
        let start = Instant::now();
        let found = store.get(&k).await;
        meter.record(start);
        assert_eq!(
            found,
            Some(cfg.seq(batch, slot, i)),
            "value at {:?}",
            unpair(&k)
        );
    }
    meter.finish("get", cfg.reads, store);
}

/// Finds the last entry of random ledgers, the largest key at or below
/// (ledger, u64::MAX), as BookKeeper does when it recovers a ledger.
async fn floor(cfg: &Config, store: &impl Store, first: u64, rng: &mut Rng) {
    let mut meter = Meter::start(cfg.reads);
    for _ in 0..cfg.reads {
        let batch = first + rng.below(cfg.batches - first);
        let ledger = cfg.ledger(batch, rng.below(cfg.ledgers));
        let k = key(ledger, u64::MAX);
        let start = Instant::now();
        let found = store.floor(&k).await;
        meter.record(start);
        assert_eq!(
            found,
            Some((ledger, cfg.last_entry(ledger))),
            "last of {ledger}"
        );
    }
    meter.finish("floor", cfg.reads, store);
}

/// Deletes the older half of the ledgers, a range per ledger, in one commit.
/// Returns the first batch whose entries remain, or `None` if there are too few
/// ledgers to delete any.
async fn delete(cfg: &Config, store: &impl Store) -> Option<u64> {
    let generations = cfg.batches.div_ceil(cfg.rollover) / 2;
    if generations == 0 {
        return None;
    }
    let ledgers = generations * cfg.ledgers;
    let mut writer = store.writer();
    let mut meter = Meter::start(1);
    let start = Instant::now();
    for ledger in 0..ledgers {
        writer.delete_range(&key(ledger, 0), &key(ledger + 1, 0));
    }
    writer.commit().await;
    meter.record(start);
    meter.finish("delete", ledgers, store);
    Some(generations * cfg.rollover)
}

#[derive(Clone, Copy)]
enum Engine {
    Encina,
    RocksDb,
}

impl Engine {
    fn name(self) -> &'static str {
        match self {
            Engine::Encina => "encina",
            Engine::RocksDb => "rocksdb",
        }
    }
}

/// The benchmark's settings. The workload has `ledgers` ledgers open at a time:
/// each batch appends `entries` entries to each, and every `rollover` batches they
/// close and as many new ones open.
struct Config {
    engine: Engine,
    dir: PathBuf,
    ledgers: u64,
    entries: u64,
    batches: u64,
    rollover: u64,
    writers: u64,
    reads: u64,
    cache: usize,
}

impl Config {
    fn parse() -> Config {
        let (mut engine, mut dir) = (None, None);
        let mut cfg = Config {
            engine: Engine::Encina,
            dir: PathBuf::new(),
            ledgers: 1000,
            entries: 100,
            batches: 100,
            rollover: 10,
            writers: 1,
            reads: 1_000_000,
            cache: 512 << 20,
        };
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let Some(value) = args.next() else {
                fail(USAGE)
            };
            match flag.as_str() {
                "--engine" => {
                    engine = Some(match value.as_str() {
                        "encina" => Engine::Encina,
                        "rocksdb" => Engine::RocksDb,
                        _ => fail(USAGE),
                    })
                }
                "--dir" => dir = Some(PathBuf::from(value)),
                "--ledgers" => cfg.ledgers = number(&value),
                "--entries" => cfg.entries = number(&value),
                "--batches" => cfg.batches = number(&value),
                "--rollover" => cfg.rollover = number(&value),
                "--writers" => cfg.writers = number(&value),
                "--reads" => cfg.reads = number(&value),
                "--cache-mb" => cfg.cache = (number(&value) as usize) << 20,
                _ => fail(USAGE),
            }
        }
        let (Some(engine), Some(dir)) = (engine, dir) else {
            fail(USAGE)
        };
        if cfg.writers > cfg.ledgers {
            fail("--writers may not exceed --ledgers");
        }
        Config { engine, dir, ..cfg }
    }

    /// The slots of the ledgers that writer `w` appends to.
    fn slots(&self, w: u64) -> Range<u64> {
        w * self.ledgers / self.writers..(w + 1) * self.ledgers / self.writers
    }

    /// The ledger that batch `batch` appends to in slot `slot`.
    fn ledger(&self, batch: u64, slot: u64) -> u64 {
        batch / self.rollover * self.ledgers + slot
    }

    /// The first entry batch `batch` appends to each ledger.
    fn first_entry(&self, batch: u64) -> u64 {
        batch % self.rollover * self.entries
    }

    /// The last entry of `ledger`.
    fn last_entry(&self, ledger: u64) -> u64 {
        let end = ((ledger / self.ledgers + 1) * self.rollover).min(self.batches);
        self.first_entry(end - 1) + self.entries - 1
    }

    /// The position in the write order of entry `i` of slot `slot` in batch
    /// `batch`: the entries of a batch arrive interleaved across its ledgers.
    fn seq(&self, batch: u64, slot: u64, i: u64) -> u64 {
        (batch * self.entries + i) * self.ledgers + slot
    }
}

fn number(s: &str) -> u64 {
    match s.parse() {
        Ok(n) if n > 0 => n,
        _ => fail(USAGE),
    }
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    process::exit(2)
}

/// A key: ledger id then entry id, so keys sort by ledger, then by entry.
fn key(ledger: u64, entry: u64) -> [u8; 16] {
    pair(ledger, entry)
}

/// Entries per entry log, which BookKeeper rolls over at 1 GiB.
const LOG_ENTRIES: u64 = 1 << 20;

/// The location of the `seq`-th entry written, with entries of 1 KiB: its entry
/// log and the offset in it.
fn value(seq: u64) -> [u8; 16] {
    pair(seq / LOG_ENTRIES, seq % LOG_ENTRIES * 1024)
}

fn decode_value(value: &[u8]) -> u64 {
    let (log, offset) = unpair(value);
    log * LOG_ENTRIES + offset / 1024
}

/// Two numbers, big-endian, so that bytewise order is numeric order.
fn pair(a: u64, b: u64) -> [u8; 16] {
    let mut bytes = [0; 16];
    bytes[..8].copy_from_slice(&a.to_be_bytes());
    bytes[8..].copy_from_slice(&b.to_be_bytes());
    bytes
}

fn unpair(bytes: &[u8]) -> (u64, u64) {
    let (a, b) = bytes.split_at(8);
    (
        u64::from_be_bytes(a.try_into().unwrap()),
        u64::from_be_bytes(b.try_into().unwrap()),
    )
}

/// What the benchmark does with a store.
trait Store {
    type Writer: Writer;

    /// A writer, with a batch of its own.
    fn writer(&self) -> Self::Writer;

    /// Waits for the store's background work to finish.
    async fn settle(&self);

    /// The value at `key`, decoded.
    async fn get(&self, key: &[u8]) -> Option<u64>;

    /// The largest key at or below `key`, decoded.
    async fn floor(&self, key: &[u8]) -> Option<(u64, u64)>;

    /// Reads every entry, in key order, and returns how many there are.
    async fn scan(&self) -> u64;

    /// The bytes the store takes on disk.
    fn size(&self) -> u64;
}

/// Writes to a store, a batch at a time.
trait Writer {
    /// Adds a put to the next commit.
    fn put(&mut self, key: &[u8], value: &[u8]);

    /// Adds the deletion of the keys in `start..end` to the next commit.
    fn delete_range(&mut self, start: &[u8], end: &[u8]);

    /// Applies what was added since the last commit, atomically and durably.
    async fn commit(&mut self);
}

struct Encina {
    db: encina::Db,
    path: PathBuf,
}

impl Encina {
    async fn open(path: &Path, cache: usize) -> Encina {
        let options = encina::Options::new().cache_size(cache);
        Encina {
            db: encina::Db::open(path, options).await.expect("open encina"),
            path: path.to_owned(),
        }
    }
}

struct EncinaWriter {
    db: encina::Db,
    batch: encina::WriteBatch,
}

impl Writer for EncinaWriter {
    fn put(&mut self, key: &[u8], value: &[u8]) {
        self.batch.put(key, value);
    }

    fn delete_range(&mut self, start: &[u8], end: &[u8]) {
        self.batch.delete_range(start, end);
    }

    async fn commit(&mut self) {
        self.db.write(&mut self.batch).await.expect("write");
        self.batch.clear();
    }
}

impl Store for Encina {
    type Writer = EncinaWriter;

    fn writer(&self) -> EncinaWriter {
        EncinaWriter {
            db: self.db.clone(),
            batch: encina::WriteBatch::new(),
        }
    }

    /// encina does no work in the background.
    async fn settle(&self) {}

    async fn get(&self, key: &[u8]) -> Option<u64> {
        let found = self.db.get(key, Comparison::Equal, |_, v| decode_value(v));
        found.await.expect("get")
    }

    async fn floor(&self, key: &[u8]) -> Option<(u64, u64)> {
        let found = self.db.get(key, Comparison::Floor, |k, _| unpair(k));
        found.await.expect("get")
    }

    async fn scan(&self) -> u64 {
        let mut iter = self.db.range(..);
        let (mut count, mut sum) = (0, 0u8);
        while let Some((k, v)) = iter.next().await.expect("scan") {
            count += 1;
            sum ^= k[15] ^ v[15];
        }
        black_box(sum);
        count
    }

    fn size(&self) -> u64 {
        fs::metadata(&self.path).expect("stat the file").len()
    }
}

struct Rocks {
    db: Arc<rocksdb::DB>,
    path: PathBuf,
}

impl Rocks {
    /// Opens RocksDB with the settings BookKeeper gives its entry location index,
    /// with two changes. It uses direct I/O: encina doesn't use the OS page cache,
    /// so RocksDB doesn't either, and each reads only what its own cache holds. And
    /// it partitions index and filter blocks, as RocksDB advises when they are
    /// cached: a large file's whole filter can outgrow a cache shard, and then
    /// every lookup reads it from disk again.
    fn open(path: &Path, cache: usize) -> Rocks {
        let mut table = rocksdb::BlockBasedOptions::default();
        table.set_block_size(64 << 10);
        table.set_block_cache(&rocksdb::Cache::new_lru_cache(cache));
        table.set_cache_index_and_filter_blocks(true);
        table.set_bloom_filter(10.0, false);
        table.set_index_type(rocksdb::BlockBasedIndexType::TwoLevelIndexSearch);
        table.set_partition_filters(true);
        table.set_pin_top_level_index_and_filter(true);

        let mut options = rocksdb::Options::default();
        options.create_if_missing(true);
        options.set_block_based_table_factory(&table);
        options.set_compression_type(rocksdb::DBCompressionType::Lz4);
        options.set_write_buffer_size(64 << 20);
        options.set_max_write_buffer_number(4);
        options.set_target_file_size_base(64 << 20);
        options.set_max_bytes_for_level_base(256 << 20);
        options.set_level_compaction_dynamic_level_bytes(true);
        options.set_use_direct_reads(true);
        options.set_use_direct_io_for_flush_and_compaction(true);

        Rocks {
            db: Arc::new(rocksdb::DB::open(&options, path).expect("open RocksDB")),
            path: path.to_owned(),
        }
    }
}

struct RocksWriter {
    db: Arc<rocksdb::DB>,
    batch: rocksdb::WriteBatch,
}

impl Writer for RocksWriter {
    fn put(&mut self, key: &[u8], value: &[u8]) {
        self.batch.put(key, value);
    }

    fn delete_range(&mut self, start: &[u8], end: &[u8]) {
        self.batch.delete_range(start, end);
    }

    /// Commits on a thread of its own, since RocksDB blocks: writers that commit
    /// at the same time then share RocksDB's group commit.
    async fn commit(&mut self) {
        let batch = std::mem::take(&mut self.batch);
        let db = self.db.clone();
        let write = compio::runtime::spawn_blocking(move || {
            let mut sync = rocksdb::WriteOptions::default();
            sync.set_sync(true);
            db.write_opt(batch, &sync).expect("write");
        });
        write.await.expect("the write's thread failed");
    }
}

impl Store for Rocks {
    type Writer = RocksWriter;

    fn writer(&self) -> RocksWriter {
        RocksWriter {
            db: self.db.clone(),
            batch: rocksdb::WriteBatch::default(),
        }
    }

    async fn settle(&self) {
        let options = rocksdb::WaitForCompactOptions::default();
        self.db
            .wait_for_compact(&options)
            .expect("wait for compactions");
    }

    async fn get(&self, key: &[u8]) -> Option<u64> {
        let found = self.db.get_pinned(key).expect("get");
        found.map(|v| decode_value(&v))
    }

    async fn floor(&self, key: &[u8]) -> Option<(u64, u64)> {
        // A new iterator for each lookup, as BookKeeper does.
        let mut iter = self.db.raw_iterator();
        iter.seek_for_prev(key);
        iter.status().expect("seek");
        iter.key().map(unpair)
    }

    async fn scan(&self) -> u64 {
        let mut iter = self.db.raw_iterator();
        let (mut count, mut sum) = (0, 0u8);
        iter.seek_to_first();
        while iter.valid() {
            count += 1;
            sum ^= iter.key().unwrap()[15] ^ iter.value().unwrap()[15];
            iter.next();
        }
        iter.status().expect("scan");
        black_box(sum);
        count
    }

    fn size(&self) -> u64 {
        let files = fs::read_dir(&self.path).expect("list the files");
        files
            .map(|f| f.and_then(|f| f.metadata()).expect("stat a file").len())
            .sum()
    }
}

/// Measures a phase: how long it took and each of its operations took, and the CPU
/// time and disk traffic of the whole process meanwhile, background threads included.
struct Meter {
    start: Instant,
    usage: Usage,
    latencies: Vec<u64>,
}

impl Meter {
    fn start(ops: u64) -> Meter {
        Meter {
            start: Instant::now(),
            usage: Usage::now(),
            latencies: Vec::with_capacity(ops as usize),
        }
    }

    /// Records an operation that began at `start`.
    fn record(&mut self, start: Instant) {
        self.latencies.push(start.elapsed().as_nanos() as u64);
    }

    /// Prints the phase's line. Its rate is of `count`, which for writes and scans
    /// counts entries rather than the operations timed, and none if it is zero.
    fn finish(mut self, phase: &str, count: u64, store: &impl Store) {
        let elapsed = self.start.elapsed();
        let usage = Usage::now();
        self.latencies.sort_unstable();
        let percentile = |p: f64| match self.latencies.len() {
            0 => "-".to_string(),
            n => duration(self.latencies[((n as f64 * p).ceil() as usize).clamp(1, n) - 1]),
        };
        println!(
            "{phase:<7} {count:>11} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7.2} {:>9.1} {:>9.1} {:>9.1}",
            duration(elapsed.as_nanos() as u64),
            match count {
                0 => "-".to_string(),
                _ => rate(count as f64 / elapsed.as_secs_f64()),
            },
            percentile(0.5),
            percentile(0.99),
            percentile(0.999),
            percentile(1.0),
            (usage.cpu - self.usage.cpu).as_secs_f64(),
            mib(usage.read - self.usage.read),
            mib(usage.written - self.usage.written),
            mib(store.size()),
        );
    }
}

/// What the process has used so far, background threads included.
#[derive(Clone, Copy)]
struct Usage {
    cpu: Duration,
    /// Bytes read from and written to storage, as Linux counts them.
    read: u64,
    written: u64,
}

impl Usage {
    fn now() -> Usage {
        let usage = rusage();
        let time = |t: libc::timeval| Duration::new(t.tv_sec as u64, t.tv_usec as u32 * 1000);
        // Counted on Linux only.
        let io = fs::read_to_string("/proc/self/io").unwrap_or_default();
        let field = |name: &str| {
            let line = io.lines().find_map(|line| line.strip_prefix(name));
            line.map_or(0, |n| n.trim().parse().expect("a number"))
        };
        Usage {
            cpu: time(usage.ru_utime) + time(usage.ru_stime),
            read: field("read_bytes:"),
            written: field("write_bytes:"),
        }
    }
}

fn rusage() -> libc::rusage {
    // SAFETY: getrusage fills in the struct it is given, which may start zeroed.
    unsafe {
        let mut usage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        usage
    }
}

/// The process's peak resident memory, in bytes.
fn peak_rss() -> u64 {
    let unit = if cfg!(target_os = "macos") { 1 } else { 1024 };
    rusage().ru_maxrss as u64 * unit
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1 << 20) as f64
}

/// Nanoseconds in a unit that keeps three significant digits.
fn duration(nanos: u64) -> String {
    let n = nanos as f64;
    if n < 1e3 {
        format!("{nanos}ns")
    } else if n < 1e6 {
        format!("{}us", digits(n / 1e3))
    } else if n < 1e9 {
        format!("{}ms", digits(n / 1e6))
    } else {
        format!("{}s", digits(n / 1e9))
    }
}

/// Operations per second, with three significant digits.
fn rate(per_sec: f64) -> String {
    if per_sec < 1e3 {
        format!("{}/s", digits(per_sec))
    } else if per_sec < 1e6 {
        format!("{}K/s", digits(per_sec / 1e3))
    } else {
        format!("{}M/s", digits(per_sec / 1e6))
    }
}

fn digits(v: f64) -> String {
    let decimals = if v >= 100.0 {
        0
    } else if v >= 10.0 {
        1
    } else {
        2
    };
    format!("{v:.decimals$}")
}

/// SplitMix64: fast, and random enough to pick keys.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A number below `n`.
    fn below(&mut self, n: u64) -> u64 {
        ((self.next() as u128 * n as u128) >> 64) as u64
    }
}
