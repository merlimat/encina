# encina benchmark

Compares encina with RocksDB on a workload shaped like BookKeeper's entry location
index: 16-byte keys, a ledger id and an entry id, mapping to 16-byte locations.

## Phases

- **load**: writes `--batches` batches and commits each durably. Every batch appends
  `--entries` consecutive entries to each of `--ledgers` open ledgers, in key order.
  Every `--rollover` batches, the open ledgers close and as many new ones open. With
  `--writers N`, N writers each append to their own share of the ledgers, so each
  batch becomes N smaller ones, which the writers commit at the same time. The
  writers are tasks on the benchmark's one thread.
- **settle**: waits for the work the load left in the background: RocksDB's
  compactions. encina has none.
- **scan**: reads every entry, in key order.
- **get**: looks up `--reads` random keys, all of them present.
- **floor**: finds the last entry of `--reads` random ledgers: the largest key at or
  below (ledger, `u64::MAX`).
- **delete**: deletes the older half of the ledgers, one range per ledger, in a
  single commit. Then scan, get and floor run again over the entries that remain.

The defaults write 10 million entries, 305 MiB of keys and values, into a 512 MiB
cache, so the store fits in memory. A smaller `--cache-mb` makes reads go to disk.

## Running

Each run benchmarks one engine. It runs in its own process, so the CPU time, disk
traffic and peak memory it reports belong to that engine alone:

```sh
cargo run --release -- --engine encina --dir /path/to/scratch
cargo run --release -- --engine rocksdb --dir /path/to/scratch
```

The store is created at `DIR/encina` or `DIR/rocksdb`, which must not exist yet.
Building RocksDB needs a C++ compiler and libclang. On Ubuntu, install
`build-essential clang libclang-dev`.

## Output

Each phase prints one line:

- **count**: entries for load and scan, lookups for get and floor, and ledgers for
  delete. **rate** is per second of this count.
- **p50 … max**: the latency of each operation. For load, an operation is a
  writer's batch, from the first put to the end of its commit; for delete, it is
  the single commit.
- **cpu s**: CPU time of the whole process, background threads included.
- **read MiB, write MiB**: storage traffic, from `/proc/self/io`, so Linux only.
- **size MiB**: the store's size on disk at the end of the phase.

The last line is the process's peak resident memory.

## Configuration

- Both engines get the same `--cache-mb`. For encina, that is its page cache. For
  RocksDB, it is the block cache, which also holds index and filter blocks.
  RocksDB's memtables come on top of that, up to 4 × 64 MiB; the peak memory line
  shows the total.
- Both bypass the OS page cache. encina always does, and RocksDB uses direct I/O
  for reads, flushes and compactions.
- Both commit each batch durably. RocksDB writes with `sync` set, from a thread of
  compio's blocking pool, so that writers committing at the same time share
  RocksDB's group commit, as they share encina's.
- Otherwise RocksDB uses BookKeeper's settings for this index: LZ4 compression,
  64 KiB blocks, 10-bit Bloom filters, 64 MiB write buffers and files, a 256 MiB
  first level, and dynamic level sizes. Like BookKeeper, it creates an iterator for
  each floor lookup.
- Unlike BookKeeper, RocksDB partitions its index and filter blocks, as RocksDB
  advises when the block cache holds them. Unpartitioned, a large file's filter
  outgrows a cache shard, so lookups keep reading it from disk. At this benchmark's
  default sizes, the get phase then read 5 TB in ten minutes without finishing.
