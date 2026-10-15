<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="encina logo" width="160">
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
- **Fast reads.** Internal nodes stay in memory, so a lookup is a few binary
  searches plus at most one leaf access. Nodes are searched in place, without
  decoding.
- **Cheap range deletes.** Subtrees inside a deleted range are dropped without being
  read, and leave nothing behind for later reads to skip.
- **No compaction.** Copy-on-write pages instead of an LSM tree: no background
  rewrites and no write stalls.
- **No allocation on the data path.** Reads, iteration, and writes that reuse a
  batch work in preallocated memory.
- **Thread-per-core.** A `Db` is `!Send`: each compio thread owns its own stores,
  with no locks or atomics.

## Status

Early development. [DESIGN.md](DESIGN.md) covers the design and the milestones:

- [x] **M1, in-memory tree:** the node format, batched writes, `get` with five
  comparisons, range iterators, and range deletes.
- [x] **M2, persistence and caching:** atomic durable commits, recovery, the leaf
  cache, and direct I/O.
- [ ] **M3, performance:** group commit, a dense node format, and benchmarks against
  RocksDB.

## Example

```rust
use encina::{Comparison, Db, Options, WriteBatch};

let db = Db::open("data.db", Options::default()).await?;

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
figure is from a single run at commit 7a0e12b, and repeat runs on this VM vary by up to
40%.

| 512 MiB cache: the store fits in memory | encina | RocksDB |
|---|---:|---:|
| Load, entries/s | 6.41M | 6.26M |
| Load, p99 batch commit | 22.2 ms | 18.8 ms |
| Load, CPU seconds | 1.0 | 2.6, then 1.1 compacting |
| Load, MiB written | 706 | 493, then 102 compacting |
| Scan, entries/s | 37.4M | 12.4M |
| Get, lookups/s | 1.36M | 375K |
| Floor, lookups/s | 2.68M | 316K |
| Delete half the ledgers, one commit | 1.7 ms | 3.9 ms |
| Size on disk, MiB | 347 | 110 |
| Peak memory, MiB | 335 | 355 |

With `--cache-mb 64 --reads 200000`, reads go to disk:

| 64 MiB cache | encina | RocksDB |
|---|---:|---:|
| Load, entries/s | 6.34M | 6.37M |
| Scan, entries/s | 14.5M | 17.3M |
| Get, lookups/s | 21.1K | 9.75K |
| Get, KiB read per lookup | 3 | 19 |
| Floor, lookups/s | 312K | 14.9K |
| Peak memory, MiB | 89 | 184 |

Floor finds a ledger's last entry: the largest key at or below a given one. RocksDB
compresses its blocks with LZ4, which takes its files from 291 MiB to 110 MiB. encina
doesn't compress: its pages in use take 285 MiB, and the file's other 62 MiB is free
pages that later writes reuse.

## Command-line tool

The `encina` command, in [cli/](cli), puts, gets and scans entries, for inspecting a
store by hand:

```sh
cargo install --path cli
encina put data.db user/0001 alice
encina get data.db user/0001         # alice
encina scan data.db user/ user0      # one key<TAB>value line per entry
encina scan -x data.db               # keys and values as hexdump -C shows them
```

`get` and `scan` open the store read-only, so they work while another process has
it open. If that process overwrites the version they read, they open the store
again, and `scan` resumes after the last key it printed. Arguments take `\xNN` for
any byte and `\\` for a backslash, and output prints UTF-8 text as is and escapes
other bytes the same way. `encina --help` has the details.

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
