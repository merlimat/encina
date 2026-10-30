<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/merlimat/encina/main/assets/logo-dark.svg">
    <img src="https://raw.githubusercontent.com/merlimat/encina/main/assets/logo.svg" alt="encina logo" width="160">
  </picture>
</p>

# encina

An embedded, ordered key-value store for Rust, built on the
[compio](https://github.com/compio-rs/compio) completion-based runtime.

encina is a general-purpose, ordered key-value library, designed for many small
key-value pairs. It is a copy-on-write B+tree that favors large batched writes, point
and ordered lookups, range scans, and deleting whole key ranges at once.

## Highlights

- **Asynchronous throughout.** All I/O goes through compio (io_uring on Linux), with
  no `mmap` and nothing that blocks the event loop.
- **Fast reads.** Internal nodes stay in memory, the bottom level as the cache has
  room for it, so a lookup is a few binary searches plus at most one leaf access.
  Nodes are searched in place, without decoding.
- **Bounded memory.** The store keeps within its cache budget, however large it
  grows: the bottom level of internal nodes, nearly all of them, is cached like the
  leaves.
- **Cheap range deletes.** Subtrees inside a deleted range are dropped without
  reading their leaves, and leave nothing behind for later reads to skip.
- **No compaction.** Copy-on-write pages instead of an LSM tree: no background
  rewrites and no write stalls.
- **No allocation on the data path.** Reads, iteration, and writes that reuse a
  batch work in preallocated memory. Reading an entry too large for a leaf into
  the cache takes a frame of its own size.
- **Thread-per-core.** A `Db` is `!Send`: each compio thread owns its own stores,
  with no locks or atomics.

## Status

[DESIGN.md](DESIGN.md) covers the design and the milestones:

- [x] **M1, in-memory tree:** the node format, batched writes, `get` with five
  comparisons, range iterators, and range deletes.
- [x] **M2, persistence and caching:** atomic durable commits, recovery, the leaf
  cache, and direct I/O.
- [ ] **M3, performance:** group commit, a dense node format, yielding inside large
  merges, opening without reading the bottom level of internal nodes, and
  benchmarks against RocksDB are done. Writing a large batch's new leaves before
  it commits ([#25](https://github.com/merlimat/encina/issues/25)) remains.
- [x] **M4, large values:** an entry too large for a leaf gets pages of its own, so
  values may be any size that fits in memory. The cache keeps them, up to a quarter
  of its budget.

Until 1.0, a minor release may change the API and the file format. A store records
its format version, and encina refuses to open one whose version it doesn't know,
with `Error::InvalidFile`.

## Example

```rust
use encina::{Comparison, Db, DefaultEnv, Options, WriteBatch};

let db = Db::open(&DefaultEnv, "data.db", Options::default()).await?;

// Batches apply atomically. Keys sort bytewise, so encode integers big-endian.
let mut batch = WriteBatch::new();
batch.put(b"user/0001", b"alice");
batch.put(b"user/0002", b"bob");
batch.delete_range(b"session/", b"session0"); // every key starting with "session/"
db.write(&mut batch).await?;

// The entry with the largest key at or below "user/9999". `get` lends the key and
// value to a closure instead of allocating copies.
let last_user = db
    .get(b"user/9999", Comparison::Floor, |k, _v| String::from_utf8_lossy(k).into_owned())
    .await?;

// A point-in-time scan of every key starting with "user/".
let mut iter = db.range(&b"user/"[..]..&b"user0"[..]);
while let Some((k, v)) = iter.next().await? {
    // ...
}
```

## Performance

Measured with [bench/](bench/README.md) on a workload shaped like BookKeeper's entry
location index: 10 million 16-byte keys mapped to 16-byte values, appended in 100
durable batches, then scanned, looked up at random, and half deleted by range. Both
engines bypass the OS page cache and get the same cache size. The machine is an Ubuntu
26.04 VM with 6 aarch64 vCPUs and ext4 on virtio, under UTM on an Apple M5 Pro. Each
figure is from a single run of the version that added dense nodes, and repeat runs on
this VM vary by up to 40%.

| 512 MiB cache: the store fits in memory | encina | RocksDB |
|---|---:|---:|
| Load, entries/s | 9.88M | 6.65M |
| Load, p99 batch commit | 17.9 ms | 18.5 ms |
| Load, CPU seconds | 0.7 | 2.6, then 1.0 compacting |
| Load, MiB written | 425 | 493, then 102 compacting |
| Scan, entries/s | 47.4M | 11.9M |
| Get, lookups/s | 1.73M | 446K |
| Floor, lookups/s | 3.13M | 322K |
| Delete half the ledgers, one commit | 1.4 ms | 7.5 ms |
| Size on disk, MiB | 116 | 110 |
| Peak memory, MiB | 125 | 349 |

With `--cache-mb 64 --reads 200000`, reads go to disk:

| 64 MiB cache | encina | RocksDB |
|---|---:|---:|
| Load, entries/s | 9.87M | 6.31M |
| Scan, entries/s | 36.3M | 16.2M |
| Get, lookups/s | 67.7K | 11.5K |
| Get, KiB read per lookup | 0.9 | 19 |
| Floor, lookups/s | 899K | 15.3K |
| Peak memory, MiB | 87 | 183 |

Floor finds a ledger's last entry: the largest key at or below a given one. RocksDB
compresses its blocks with LZ4, which takes its files from 291 MiB to 110 MiB. encina
doesn't compress, but its dense nodes store what their keys and their values share
once, so an entry takes about 5 bytes: its pages in use take 87 MiB, and the file's
other 29 MiB is free pages that later writes reuse.

With `--writers N`, N writers append to their own shares of the ledgers at the same
time, each committing a batch of its own. In both engines, writes that queue up
behind a commit share the next one. With 4 or 16 writers, the load runs about as
fast as with one. With 10 entries per ledger in each batch instead of 100, there
are ten times as many commits, and the load runs at a third of the rate. Single
runs of the version that added group commit, with the 512 MiB cache:

| Load with concurrent writers | encina | RocksDB |
|---|---:|---:|
| 4 writers, entries/s | 8.84M | 6.29M |
| 4 writers, p99 batch commit | 20.6 ms | 33.7 ms |
| 16 writers, entries/s | 8.16M | 5.44M |
| 16 writers, p99 batch commit | 20.9 ms | 26.1 ms |
| 8 writers, 1,000 batches of 10 entries per ledger, entries/s | 2.90M | 2.38M |
| 8 writers, 1,000 batches of 10 entries per ledger, p99 batch commit | 5.7 ms | 6.7 ms |

## Command-line tool

The `encina` command, in [cli/](cli), puts, gets, scans and deletes entries, for
inspecting a store by hand:

```sh
cargo install encina-cli
encina -f data.db put user/0001 alice
encina -f data.db get user/0001            # alice
encina -f data.db scan -s user/ -e user0   # one key<TAB>value line per entry
encina -f data.db scan -x                  # keys and values as hexdump -C shows them
encina -f data.db delete user/0001
encina -f data.db delete-range -s user/ -e user0
encina -f data.db stats                    # entries, file size, pages level by level
encina -f data.db pages                    # a map of the file's pages, 64 to a line
```

`cargo install encina-cli` builds the command and copies it to `~/.cargo/bin`. From a
checkout, `cargo install --path cli` does the same, and `cargo build --release` at the
root builds it in place, at `target/release/encina`.

`get`, `scan`, `stats` and `pages` open the store read-only, so they work while
another process has it open. If that process overwrites the version they read, they
open the store again, and `scan` resumes after the last key it printed. Arguments
take `\xNN` for any byte and `\\` for a backslash, and output prints UTF-8 text as
is and escapes other bytes the same way. `encina --help` has the details.

`encina completions` prints a script that completes encina's commands and options in
bash, zsh, fish, elvish or PowerShell. For bash:

```sh
encina completions bash > ~/.local/share/bash-completion/completions/encina
```

## Metrics

A store exports metrics through OpenTelemetry when its options name a meter
provider: write and commit latency, cache hits and misses, disk traffic, the
entries stored and their size, how the file's pages are used, and more. Each
record takes a few nanoseconds, through
[nanometer](https://github.com/merlimat/nanometer), and never allocates.

```rust
let opts = Options::new().meter_provider(&provider);
```

[METRICS.md](METRICS.md) lists the metrics, their attributes, and what each one
measures.

## Building

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

The property tests run 256 cases each; set `PROPTEST_CASES` to run more.

Linux with io_uring is the target platform. macOS works for development.

## License

Copyright 2026 Matteo Merli. Licensed under the [Apache License, Version 2.0](LICENSE).
