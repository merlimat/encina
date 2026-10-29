# encina — metrics

A store exports metrics through OpenTelemetry once its options name a meter
provider. They are recorded with [nanometer](https://github.com/merlimat/nanometer)
handles, which a store creates when it opens: a record takes a few nanoseconds and
never allocates. Without a provider, a store records nothing.

## Setup

```rust
use encina::{Db, Options};
use opentelemetry::KeyValue;

let opts = Options::new()
    .meter_provider(&provider)
    .metric_attributes(&[KeyValue::new("tenant", "a")]);
let db = Db::open("/data/0/index.db", opts).await?;
```

- The metrics come under the instrumentation scope `encina`.
- encina uses `opentelemetry` 0.33.1, the version nanometer pins, so the app's
  provider must come from that version too.
- Counters and up-down counters export through any provider. Histograms export
  only through nanometer's `Producer`, which the app registers with its metric
  reader. Until an OTel SDK release has `MetricProducer`, that takes a patched SDK:
  see nanometer's README, under
  [Exporting histograms](https://github.com/merlimat/nanometer#exporting-histograms).
- The first store with metrics in a process calibrates the clock that times
  operations, which blocks its open for 10 to 200 ms, once.

## Attributes

Every series has `encina.store`, the path the store was opened at, as passed to
`open`. The attributes of `Options::metric_attributes` come after it, and one with
the key `encina.store` replaces the path. Some metrics add attributes of their own,
listed with them below.

Amounts a store holds are up-down counters, not gauges, so stores whose attributes
are the same add up: two stores both named `ledgers` report their total. A store
takes its share back out when it closes, once its last handle, clone or iterator is
dropped. Counters and histograms keep counting across stores with the same
attributes, and across reopening the same path.

A `ReadOnlyDb` reports its lookups, iterators, readers, cache and disk reads, which
add to the owner's when both open the store by the same path. It doesn't report
writes, commits, syncs, the entries, the file or `encina.store.failed`, which the
store's owner reports.

## Writes and commits

| Metric | Type | Unit | Attributes |
|---|---|---|---|
| `encina.write.duration` | histogram | `s` | |
| `encina.write.operations` | counter | `{operation}` | `encina.operation.type`: `put`, `delete`, `delete_range` |
| `encina.write.size` | histogram | `By` | |
| `encina.write.pending` | up-down counter | `{write}` | |
| `encina.write.errors` | counter | `{error}` | `error.type` |
| `encina.commit.duration` | histogram | `s` | |
| `encina.commit.writes` | histogram | `{write}` | |

- **`encina.write.duration`**: how long each successful `Db::write` takes, from the
  call until its batch is durable. That includes waiting for the commit in progress.
  A write that fails or is dropped records nothing.
- **`encina.write.operations`**: the operations in the batches written successfully,
  as they were added to them: a batch that puts a key twice counts two puts.
- **`encina.write.size`**: the bytes of keys and values in each batch written
  successfully. A delete counts its key, and a range delete its two bounds. The
  histogram's sum is the bytes the app wrote.
- **`encina.write.pending`**: the writes in progress, waiting for their turn or
  committing.
- **`encina.write.errors`**: the writes that failed, by `error.type`:
  - `key_too_large`: a put's key exceeds `Options::max_key_size`.
  - `io`, `corruption`: an I/O error, or a leaf the write read failed its checks.
  - `failed`: an earlier I/O error left the store failed (see
    `encina.store.failed`).
- **`encina.commit.duration`**: how long each commit takes, from the write taking its
  turn until the new version is published. That covers:
  - planning the batches and loading the leaves they change
  - merging
  - writing the new pages
  - two syncs, with the meta page written between them

  A commit that changes nothing, or fails, records nothing.
- **`encina.commit.writes`**: the writes each commit makes durable. More than 1 means
  writes arrived while another commit ran, and shared the next one.

## Reads

| Metric | Type | Unit | Attributes |
|---|---|---|---|
| `encina.get.duration` | histogram | `s` | |
| `encina.get.errors` | counter | `{error}` | `error.type` |
| `encina.range.iterators` | counter | `{iterator}` | |
| `encina.range.entries` | counter | `{entry}` | |
| `encina.reader.count` | up-down counter | `{reader}` | |

- **`encina.get.duration`**: how long each successful `get` takes.
  - A lookup of cached leaves takes well under 10 µs, the lowest bucket bound, so
    those all land in the first bucket.
  - The buckets above show the lookups that waited for the disk.
- **`encina.get.errors`**: the lookups that failed, by `error.type`: `io`,
  `corruption`, or `stale`, for a `ReadOnlyDb` whose version the owner overwrote.
- **`encina.range.iterators`**: the iterators that `range` created.
- **`encina.range.entries`**: the entries the iterators returned. An iterator adds its
  count when it is dropped.
- **`encina.reader.count`**: the readers that pin a version: open iterators, and
  lookups waiting for the disk.
  - A pinned version keeps its internal nodes above level 1 in memory, and the pages
    later writes replace stay allocated: see `encina.file.usage`, state `held`.
  - A count that stays up points to iterators that aren't dropped.

## Cache

| Metric | Type | Unit | Attributes |
|---|---|---|---|
| `encina.cache.lookups` | counter | `{lookup}` | `encina.cache.result`: `hit`, `miss` |
| `encina.cache.evictions` | counter | `{leaf}` | |
| `encina.cache.usage` | up-down counter | `By` | `encina.page.type`: `internal`, `leaf`, `value` |
| `encina.cache.limit` | up-down counter | `By` | |

- **`encina.cache.lookups`**: each leaf that a lookup, an iterator or a write needed,
  and each level-1 node, the internal nodes right above the leaves, that one had to
  read, or that a write needed.
  - `hit`: the leaf was in the cache.
  - `miss`: it had to wait for a read, its own or one another reader started. That
    includes the leaves an iterator read ahead.
  - A lookup that misses counts once, not again when it finds the leaf it read.
  - Values too large to cache, over a quarter of the budget, are read without the
    cache and don't count.
  - The hit ratio is `hit / (hit + miss)`.
- **`encina.cache.evictions`**: the leaves and level-1 nodes evicted to make room
  for others.
- **`encina.cache.usage`**: the memory of the page frames in use, by what they hold:
  - `internal`: internal nodes above level 1, which stay in memory.
  - `leaf`: cached leaves and level-1 nodes, and pages being read or written.
  - `value`: value leaves, each in a frame of its own pages.

  A write holds the leaves it changes and builds until it commits, so a write
  larger than the cache takes the usage past the limit for a while.
- **`encina.cache.limit`**: the budget for page frames: `Options::cache_size`, rounded
  down to whole pages, and 8 pages at least.

## Disk

| Metric | Type | Unit | Attributes |
|---|---|---|---|
| `encina.disk.io` | counter | `By` | `disk.io.direction`: `read`, `write` |
| `encina.disk.operations` | counter | `{operation}` | `disk.io.direction` |
| `encina.disk.read.duration` | histogram | `s` | |
| `encina.disk.sync.duration` | histogram | `s` | |

- **`encina.disk.io`**: the bytes read from and written to the store file.
  - Reads: leaves, value leaves and level-1 nodes, plus the internal nodes the
    store reads when it opens.
  - Writes: each commit's new pages and its meta page.
  - Dividing the bytes written by the sum of `encina.write.size` gives the write
    amplification.
- **`encina.disk.operations`**: the read and write requests. A commit writes each run
  of adjacent pages, up to 1 MiB, with one request.
- **`encina.disk.read.duration`**: how long each read of a leaf, a value leaf or a
  level-1 node takes, until a reader sees it finish. The reads of internal nodes at
  open aren't timed.
- **`encina.disk.sync.duration`**: how long each sync of the store file takes. A
  commit syncs twice: once its pages are written, and once its meta page is.

## Entries, space and state

| Metric | Type | Unit | Attributes |
|---|---|---|---|
| `encina.entry.count` | up-down counter | `{entry}` | |
| `encina.entry.size` | up-down counter | `By` | |
| `encina.file.size` | up-down counter | `By` | |
| `encina.file.usage` | up-down counter | `By` | `encina.page.state`: `live`, `held`, `free` |
| `encina.store.failed` | up-down counter | `{store}` | |

- **`encina.entry.count`**: the entries in the store's current version.
- **`encina.entry.size`**: the bytes of their keys and values.
  - It counts what the app wrote, not what the store takes: leaves store the
    prefixes their keys and values share once, and value leaves round their
    entries up to whole pages.
  - `encina.file.usage`, state `live`, gives the bytes the entries take on disk.
- **`encina.file.size`**: the store file's size.
- **`encina.file.usage`**: the store file's pages, by state. The three add up to the
  file's size.
  - `live`: the current version's pages, the two meta pages included.
  - `held`: pages that commits replaced, kept for readers of older versions. The
    first write after those readers end frees them.
  - `free`: pages later writes can reuse. That includes stretches shorter than
    `Options::min_run_pages`, which writes skip until the pages around them are
    freed too, and the end of the file, which is cut only once that frees an eighth
    of it.

  These four are updated when the store opens and after each commit. The store
  keeps the entries' count and size in its meta page, so they cost no reads.
- **`encina.store.failed`**: 1 while the store rejects writes with `Error::Failed`. An
  I/O error during a commit left it unknown whether the commit is durable.
  Reopening the store recovers it.
