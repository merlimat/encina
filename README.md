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
