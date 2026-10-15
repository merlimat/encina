# encina — design

> Status: M1 and M2 are implemented: the tree, persistence and the leaf cache.
> M3, performance, is under way (§12): concurrent misses share one read, and
> iterators read ahead (§8).

encina is an embedded, ordered key-value store for Rust, built on the
[compio](https://github.com/compio-rs/compio) completion-based runtime. Its first
user is the job RocksDB does inside BookKeeper's `DbLedgerStorage`: mapping
`(ledgerId, entryId)` to the physical location of an entry.

## 1. Workload

| Aspect | Shape |
|---|---|
| Keys / values | In BookKeeper, 16 bytes → 16 bytes. Key = `ledgerId` (u64 big-endian) ‖ `entryId` (u64 big-endian). The store itself accepts any byte strings (§5). |
| Writes | Large batches, each made durable as a unit. A batch holds appends to many open ledgers at once, so new keys land at *A* append points (one per active ledger), not at one end of the key space. Entry-log compaction rewrites existing keys in batches. |
| Reads | Point lookups; "last entry in ledger" = floor lookup (largest key ≤ `(L, u64::MAX)`); occasional scans of one ledger. |
| Deletes | Whole ledgers: `delete_range((L, 0)..(L + 1, 0))`. |
| Size | Usually fits in memory, but must keep working when it doesn't. Highly compressible (shared ledger prefixes, sequential entry ids, increasing offsets). |
| Durability | The caller already has a journal. The index only needs to be durable at batch boundaries, and must never be torn. |

## 2. Why a B+tree rather than an LSM

The B+tree is the right call for this workload.

- **Reads.** Internal nodes are always in memory, so a point read is a few binary
  searches over in-memory nodes plus at most one leaf access. There are no bloom
  filters and no probing of several levels. A floor lookup is a single descent.
- **`delete_range` leaves nothing behind.** Covered subtrees are dropped wholesale
  without reading their leaves; only the two boundary leaves are rewritten. An LSM
  writes a range tombstone instead, and every later read in that range has to check
  it until compaction removes it.
- **No compaction.** No background rewriting, no write stalls, predictable latency.

The costs to accept:

- **Write amplification is per touched leaf.** Copy-on-write means every commit
  (each `write()`) rewrites each leaf that received at least one key. Touched
  leaves ≤ min(new entries, active ledgers *A*), so a commit writes at most about
  `max(new entry bytes, A × page size)` plus internal nodes. With *A* = 10,000 and
  4 KiB pages, that is at most 40 MiB per commit, however few entries each ledger
  got. Against the entry data written in the same interval (a write-cache flush is
  hundreds of MB), that is a few percent. **Big batches are the operating point**;
  small, frequent writes are not.
- **Random inserts are the worst case**, since every key touches a different leaf.
  That is not this workload.
- **Copy-on-write scatters leaves on disk**, so long scans are not physically
  sequential. That doesn't matter for cached leaves, and random 4 KiB reads are
  cheap on NVMe.

## 3. Decisions

| Topic | Decision |
|---|---|
| Durability | Each `write()` is an atomic, durable commit: when it returns, the batch is on disk, and readers see it from that moment on. There is no separate flush and no internal WAL: after a crash the store opens at the last completed write, and the caller's journal replays anything newer. |
| Threading | One `!Send` instance per compio thread, no locks or atomics, except in wakers, which Rust requires to be `Send` and `Sync` (§8). Scale by sharding (for example, by ledger id) above the library. |
| Memory | Bounded from day one and managed entirely by encina: internal nodes stay resident, and leaves live in encina's own leaf cache with a budget, read on demand (§8). No reliance on the OS page cache or `mmap`: the data file is opened for direct I/O (§9). |
| Allocation | No heap allocation on the data path. Page frames in the leaf cache are the only exception (§8); write batches, cursors and reader bookkeeping reuse their memory. compio itself allocates once per I/O operation, so cache misses and commits are not allocation-free (§9). |
| Keys and values | **Proposed:** arbitrary byte strings, starting with one node format, slotted, which handles any lengths. A dense format for fixed-size data can be added later as a per-node specialization if benchmarks justify it (§5). |

## 4. Architecture

```
write(batch) ─► new version ─► write new pages ─► fsync ─► meta ─► fsync ─► publish root
                (path copy)           │                                          │
                                      ▼                                          ▼
                 data file: [meta 0][meta 1][page]...        get / range ─► current root
                                      │                                          │
                                      └──────── miss: direct read ────────► leaf cache
                                                                      (frames, budget, CLOCK)
```

- **Pages live in frames.** All node memory is a pool of page-sized, 4 KiB-aligned
  frames owned by the leaf cache. A frame table records what each frame holds, its
  page id, reference count, dirty flag and CLOCK bit. Nodes refer to each other by
  page id or frame index, never by heap pointer, so building and dropping nodes
  never calls the allocator. In-flight I/O buffers share the frames' memory and
  reference counts, and keep them alive until the kernel is done with them: a
  cancelled read never touches freed memory.
- **Leaves are pages.** The node format (§5) is searched in place, so a leaf in
  memory is byte-for-byte its on-disk page. A cache miss needs no decoding, and the
  budget is counted in frames. Frames double as I/O buffers, which direct I/O needs
  aligned anyway.
- **Internal nodes are always resident**: an encoded key block (same format as
  leaves) plus child references. A reference to an internal child holds its frame
  and page id; a reference to a leaf holds only its page id, and the leaf is found
  through the cache, so it can be evicted whenever no one is using it.
- **Versions.** A version is a root reference. A node is never mutated after it is
  published. A write builds new nodes along the paths it touches and shares
  everything else, then swaps the root once its commit is durable. Readers holding
  the old root are unaffected, so an iterator is a consistent snapshot that costs
  one reference-count increment.
- **Dirty nodes** are the nodes a write has built but not yet written. They exist
  only while that write is in progress, and get their page as they are built. Path
  copying means the parent of a dirty node is always dirty, so the commit finds
  every dirty node by descending through dirty nodes only.
- **Single writer.** Writes are serialized by an async lock, held for the whole
  write, including its commit. Reads never take it.

## 5. Keys, values and node formats

Keys and values are arbitrary byte strings. Keys are ordered lexicographically, so
callers encode integers big-endian.

An entry (key + value) must fit in a quarter of a page (1 KiB with 4 KiB pages):
every leaf then holds at least four entries, and there are no overflow pages. This
limit is part of the public contract. It is documented on `WriteBatch::put` and
`Options::page_size`, and `write()` rejects a batch containing an oversized entry
with `Error::EntryTooLarge` before applying any of it.

### What generality costs

Measured with [experiments/leaf-layouts](experiments/leaf-layouts): random point
lookups of existing keys on BookKeeper-shaped data, with leaves packed into 4 KiB
pages and routed through an identical in-memory array. Apple M5 Pro; run-to-run
noise is about ±5%.

| Leaf layout | B/entry | 20M entries (DRAM-bound) | 100K entries (in L2) |
|---|---:|---:|---:|
| Specialized: native `u128` keys | 32.0 | 198 ns (1.00×) | 29 ns (1.00×) |
| Generic, dense, compared as big-endian integers | 32.0 | 212 ns (1.07×) | 30 ns (1.05×) |
| Generic, dense, compared with `memcmp` | 32.0 | 256 ns (1.30×) | 59 ns (2.03×) |
| Generic, dense, key prefix stripped | 17.8 | 200 ns (1.01×) | 41 ns (1.41×) |
| Generic, dense, key and value prefixes stripped | 5.1 | 200 ns (1.01×) | 42 ns (1.45×) |
| Generic, slotted, `memcmp` | 38.0 | 352 ns (1.78×) | 95 ns (3.29×) |
| Generic, slotted, prefix stripped, 4-byte heads | 26.1 | 228 ns (1.15×) | 51 ns (1.77×) |

Returning each value as a `Vec<u8>` adds about 15 ns when cached and 75 ns when
DRAM-bound, since allocating also limits how much consecutive lookups overlap.

What this shows:

- A byte-slice API costs nothing measurable by itself, as long as reads don't
  allocate.
- The node format is what matters. With prefixes stripped, dense nodes match the
  specialized layout once the index outgrows the CPU cache, and they are 6×
  smaller. With a bounded cache, 6× smaller means 6× more entries per GiB. When
  everything sits in L2, they cost about 13 ns more per lookup.
- Copy-on-write rebuilds a node on every change, so the format can be chosen per
  node when it is built. Specialization is then automatic and invisible in the API:
  every node of a BookKeeper index would be uniform and get the dense format.
- Caveats: the synthetic data is more regular than real locations (a bookie stores a
  striped subset of each ledger's entries, and compaction moves entries between
  logs), so expect less than 6×. Only the leaf level is measured, on one machine.

### Node format

v1 starts with one format, **slotted**, because it handles any key and value
lengths. The header holds the prefix that all keys in the node share. The body holds
a contiguous array of 4-byte key heads (the first bytes of each suffix, as a
big-endian `u32`), a slot array (offset, suffix length, value length), and a heap
with the rest of each suffix and the value. The node is searched in place without
decoding: the binary search reads only the heads, and ties fall back to the heap.
Internal nodes use the same format, with child page ids as fixed-size values.

The specialization path stays open. The page header records each node's format, so
a **dense** format (for nodes whose entries all share a key length and a value
length: shared key and value prefixes in the header, suffixes at a fixed stride,
compared as big-endian integers) can be added later, chosen per node when it is
built, without changing the API or invalidating existing files.

Reads never allocate. Results reach the caller as borrowed slices: the value comes
straight from the leaf page (the slotted heap stores it contiguously), and the key
is rebuilt from prefix and suffix in a buffer owned by the call.

## 6. Operations

### `write(batch)`

A `WriteBatch` holds `put`, `delete` and `delete_range` operations, encoded into an
arena that `clear()` empties without releasing, so a caller that reuses its batch
doesn't allocate. The operations are applied in order and atomically.

1. **Normalize.** Drop point operations made obsolete by a later range in the same
   batch. Stable-sort the rest by key (through an index array kept in the batch) and
   keep the last operation per key. Sorted input (BookKeeper's write cache is
   sorted) costs one linear check.
2. **Plan.** Descend the resident internal nodes to find the leaves the batch
   changes, and load the ones that aren't cached, keeping them pinned. Subtrees
   that ranges remove whole aren't read.
3. **Apply ranges** (see below).
4. **Bulk merge.** Descend recursively, partitioning the sorted operations by child
   key range. Each touched leaf is rebuilt once from its old entries plus the
   operations and re-encoded into one or more pages. Each parent rebuilds its child
   list, splitting if needed, up to the root.
5. **Commit** the new version (§7).
6. **Publish** the new root. Readers see the batch only now, once it is durable,
   and the pages of the nodes it replaced become free, subject to §8.

Readers keep running while a write waits for the disk. The merge itself doesn't
yield yet: it is CPU work bounded by the batch, and yielding inside it is planned
for M3. Doing so is safe: readers keep using the old root until the swap, and the
lock keeps other writers out.

**Split policy.** A rebuilt leaf that overflows splits right after each run of
inserted keys that fills at least a quarter page (the per-ledger append case): the
run's leaves are packed full, and the existing entries after it start a new leaf. A
ledger's settled entries then stop being rewritten by later appends, even when one
leaf holds the tails of several ledgers. Smaller runs, like scattered inserts, don't
get leaves of their own, and whatever follows the last split point is split evenly
by bytes. There is no strict 50% minimum fill: a rebuilt node below ~25% merges with
a neighbor, if the neighbor is in memory. A level's new nodes are only built once
its merges are decided, so a write builds no node that a merge then replaces.

### `delete_range(start..end)`

At each internal node, children that lie entirely inside the range are dropped.
Their page ids come from the resident internal nodes, so no leaf is read.
Partially covered children recurse, and the two boundary leaves are rebuilt. Empty
nodes are removed, and a root left with a single child collapses. Later reads pay
nothing.

### `get` / `range`

- `get(k, cmp)`: one descent finds the entry matching `k` under `cmp`: `Equal`,
  `Floor` (largest key ≤ k), `Ceiling` (smallest key ≥ k), `Lower` (largest key
  < k) or `Higher` (smallest key > k). Each internal key equals the smallest key in
  its child's subtree, so `Equal` and `Floor` always finish in the leaf the descent
  reaches; the other comparisons may step to the neighboring leaf. Copy-on-write
  rewrites the whole path anyway, so keeping internal keys exact costs nothing.
  BookKeeper's `getLastEntryInLedger(L)` is `get((L, u64::MAX), Floor)` followed by
  a check that the ledger id matches.
- `range(r)`: an iterator holding a version and a cursor stack of (node, index) per
  level, stored inline (the tree is shallow), so creating one doesn't allocate. It
  is a point-in-time snapshot. Holding one keeps replaced nodes in memory and holds
  back page reuse (§8), so iterators should be short-lived. When the leaf it moves
  to isn't cached, it also reads the next leaves ahead (§8).

## 7. Commit and on-disk format

### Commit protocol

Every `write()` ends with a commit that makes its batch durable in one atomic step.

1. Seal the new version's dirty nodes: stamp the generation and the checksum. Each
   already has its page, so they can be written in any order.
2. Sort them by page, and write each run of adjacent pages with one vectored write
   of up to 1 MiB, 16 writes at a time. Await every write, then `sync_data`.
   Allocation hands a write its pages in runs (see below), so a commit takes a few
   large requests rather than one per page.
3. Write the meta page into slot `(gen + 1) % 2`, then `sync_data`. The commit is
   durable.
4. Publish the new root. The written leaves become evictable, and the pages this
   write dropped become reusable, subject to §8.

A write keeps the leaves it changes and the ones it builds in memory until it
publishes, so a batch larger than the cache budget grows the pool past it. Writing
new leaves out as the merge produces them, which is safe since nothing reaches them
until the meta page does, is planned for M3.

**Invariant:** a commit never overwrites a page reachable from the last durable
commit. Whenever a crash happens, the newest valid meta slot describes an intact
tree. A torn meta write fails its checksum, and recovery uses the other slot.

**Failures.** If loading a leaf or writing a page fails, nothing is published: once
the rest of that I/O has finished, the batch is dropped, its pages return to the
free set, and `write()` returns the error, leaving the store as it was. If a
`sync_data` or the meta page write fails, what reached the disk is unknown, so the
instance enters a failed state: reads keep serving the last published version,
writes return `Error::Failed`, and the store must be reopened. Dropping a `write()`
future is harmless until it starts writing pages; after that, it also puts the
instance in the failed state.

### File layout (format v1)

One file with a fixed page size (default 4 KiB, chosen at creation, a power of two
from 4 to 64 KiB).

| Pages | Content |
|---|---|
| 0, 1 | Meta slots: checksum, magic, format version, page size, root level, generation, root page, high-water mark. |
| ≥ 2 | Tree pages, each with a header: crc32c, node format (slotted in v1), level, entry count, generation written. |

### Free space is derived, not stored

No free list is persisted. At open, the live set is the meta pages, every internal
page, and every leaf page id listed in the bottom internal level. Every other page
below the high-water mark is free, tracked in an in-memory bitmap (1 bit per page:
128 KiB covers 4 GiB). Nothing on disk can drift out of sync with the tree, and a
crash cannot leak space.

At runtime, a write takes its pages in order from runs of free pages: the first
free stretch of at least 16 pages (64 KiB with 4 KiB pages), then the next one,
and at the end of the file, 256 pages at a time. Shorter holes are skipped: each
would cost a request for a page or two, and they become usable once the pages
around them are freed too. Every write starts over from the lowest free page. The
pages it frees again before it ends, as when its new root loses more than one
level, are holes in its own runs, which its next short-lived pages fill (see below).

Pages replaced together should be freed together, as runs again, so a write
allocates from two runs at once: one for pages likely to be replaced soon
(internal nodes, and leaves that may take more keys), and one for the full leaves
that a run of appends leaves behind, which are likely to stay. With appends, as to
ledgers, each batch replaces the short-lived pages of the batch before as a whole,
and the batch after reuses their space as one run.

The cost is space. On the BookKeeper-shaped benchmark (10 million entries in 100
batches) the file ends up 21% larger than when allocation took the lowest free
page, and in a simulation of random updates over 22,000 leaves it settled at
2.8–3.3× the live pages instead of 2×. In exchange, that benchmark's load sent
4,540 write requests averaging 158 KiB instead of 169,500 of 4 KiB, and ran three
times as fast.

### Recovery

1. Read slot 0, which records the page size; if it is torn, look for slot 1 where
   each possible page size would put it. Then read both slots, and pick the intact
   one with the highest generation.
2. Load the internal pages level by level, verifying checksums, levels and
   generations, and point each parent at its children's frames. Leaves are read on
   demand, so open time scales with the internal pages, a fraction of a percent of
   the file.
3. Build the free bitmap from the pages the tree reaches.

Opening an empty file creates a store. It first syncs the file's directory, since a
new file's name is durable only once its directory is: without that, a crash could
lose a store whose every write had been synced. Then it writes both meta slots and
syncs them. A file too short to hold a meta page, or with no intact slot, isn't a
store (`Error::InvalidFile`): a crash while a store is being created can leave one.

There is no log to replay: the caller's journal covers anything newer than the last
completed write.

## 8. Memory and the leaf cache

- **Frames** are the only allocation on the data path. The cache allocates them
  (4 KiB-aligned) up to its budget and recycles them after that. Internal nodes,
  cached leaves and dirty leaves all live in frames.
- **Internal nodes** are always resident. With prefixes stripped, an internal page
  holds several hundred children, so internal nodes are well under 1% of the data.
- **Leaves** live in the cache, keyed by page id in a table sized at open, with CLOCK
  eviction. Only clean, unpinned leaves can be evicted: a write pins the leaves it
  changes and the ones it builds until it publishes, and a reader pins the leaf it
  is on. Exposing cache numbers is left to a later metrics design.
- **Misses** read the page straight into a free frame (direct I/O) and verify the
  checksum. Concurrent misses on the same page share one read, and iterators read
  ahead (below). A frame left behind by a cancelled read is reclaimed by the clock
  hand once the read completes.
- **Page reuse.** A page dropped by a write becomes reusable once that write is
  published and no reader that started before the publish is still alive (such a
  reader might still need to read it). Iterators always register as readers;
  point reads register only when they have to wait for I/O. Reclaiming a page
  evicts any stale cache entry for it.
- **Capacity.** At ~26 B/entry for slotted nodes on BookKeeper-shaped data, a 1 GiB
  cache holds roughly 40M entries. A dense format (~5 B/entry) would raise that to
  roughly 200M, which is the main reason to add it later.

### Reads in flight

Leaf reads go through a table of the reads in flight. A reader that misses looks
there before starting a read, so concurrent misses on a page share one read. Each
read sits in a slot that counts the readers holding it. Whichever of them is polled
drives the read, and the first to find it finished claims it: it verifies and
caches the page. The others then find the page cached, or, if the read failed,
read it again themselves: a failure goes to the reader that claimed it.

No reader depends on another being polled, not even the one that started the
read: the read is polled with a waker of its own, which wakes every reader waiting
for it. That waker keeps the waiting readers in a `Mutex`, because Rust requires
wakers to be `Send` and `Sync`. It is never contended: a store's reads all run on
its thread. A read whose last reader lets go of it is cancelled. Its frame is freed
at once if the read is done with it, and otherwise reclaimed by the clock hand once
the read finishes: the buffer the runtime holds keeps the frame alive until then.

Slots are reused: once the table has grown to the most reads ever in flight at
once, a miss allocates nothing but compio's operation (§9).

### Reading ahead

Leaves are scattered on disk (§2), and direct I/O leaves no OS read-ahead anyway,
so iterators read ahead themselves. Internal nodes are resident, so an iterator
knows the pages of the leaves it will visit. When the leaf it moves to isn't
cached, it waits for that leaf's read and meanwhile starts reads of the following
leaves that aren't cached either, so that a cold scan keeps many reads in flight
instead of one. Each such miss tops the window up again. The window spans
`IO_WINDOW` (16) leaves past the current one, and at most a quarter of the cache
budget. It ends at the last leaf, and at the end of the range: the internal key
above a leaf is its smallest key, so the iterator knows which leaves the range
reaches without reading them.

The reads ahead sit in the table like any other read: another reader needing one of
those pages waits for it, and so does the iterator when it gets there, rather than
reading the page again. Reading ahead only takes frames within the cache budget: it
evicts cold leaves but, unlike a read someone waits for, never grows the pool. A
read ahead that fails is reported when the iterator reaches its leaf, and calling
`next` again reads the leaf again. Dropping an iterator cancels the reads it
holds.

On a Linux VM (6 vCPUs, ext4 on virtio), reading ahead took a cold scan of 10M
BookKeeper-shaped entries through a 64 MiB cache from 4.2 s to 0.75 s, reading the
same 282 MiB.

## 9. Concurrency and I/O

- `Db` is `!Send` and `!Sync` (`Rc`, `RefCell`, `Cell`). It is a cheap `Clone`
  handle shared by the tasks of one compio runtime thread. BookKeeper already keeps
  one index per ledger directory; here, one instance per core or shard plays the
  same role. A `Send` facade (compio dispatcher or channels) can come later.
- **One owner per file.** Opening takes an exclusive `flock` on the file, which
  closing it releases, so a second open fails with `Error::Locked` instead of
  corrupting the store. A `flock` lock belongs to the open file rather than to the
  process, so it also stops a second open from another thread of the same process,
  the likely mistake with one instance per thread.
- **Read-only views.** A `ReadOnlyDb` opens the file read-only and takes no lock,
  so it can read a store that a `Db` owns, in this process or another. It reads
  the version that was current when it opened, with a cache of its own. The owner
  doesn't know about it, so it reuses that version's pages from its second write
  after the view opens. Every leaf read checks that the page holds a leaf no newer
  than the current generation, so a reused page is never taken for the one the
  tree points to. When a view's page fails its checks, the view rereads both meta
  slots: if the owner has committed since, the error is `Error::Stale`, and a new
  view reads the current version; otherwise it is `Error::Corruption`. A corrupt
  page reported as stale is reported as corrupt by the next view, if that view
  still uses it. Opening a view checks the internal nodes it loads the same way.
  Views suit inspection and tooling: one that should keep up with a busy owner has
  to be reopened, since nothing holds back the owner's page reuse for it.
- All file I/O goes through compio 0.19: `read_at` / `write_at` on `&File`, with
  frames as the owned buffers, and `sync_data`. A commit writes each run of pages
  with `write_vectored_at`, whose buffers are the frames themselves, so nothing is
  copied; a run is at most 1,024 frames (`IOV_MAX`).
- **No OS page cache, no `mmap`.** On Linux the data file is opened with `O_DIRECT`
  (`OpenOptions::custom_flags`), so pages move straight between frames and the
  device; on file systems that don't support it, like tmpfs, it falls back to
  buffered I/O. On macOS, for development only, `F_NOCACHE` is set instead.
  encina's cache is then the only cache, memory use matches the budget, and no page
  fault can block the event loop. `sync_data` is still required for durability.
- compio 0.19 heap-allocates a small control block for every submitted operation,
  and a vectored write also allocates its `iovec` array. The cache-hit path does no
  I/O and stays allocation-free; a miss pays one allocation per page read, and a
  commit two per write request. encina adds none: a write keeps its in-flight
  operations in a fixed-size array, and reuses their lists of frames; an iterator
  keeps the reads it holds inline; and the table of reads in flight reuses its
  slots (§8). If compio's show up in profiles, the fix belongs upstream (pooled
  operation slots).
- CPU-heavy steps, large merges in particular, don't yield to the executor yet;
  that is planned for M3.
- **Linux with io_uring is the target.** macOS works for development, but there
  compio runs file I/O on a blocking thread pool and `fsync` is a plain `fsync`
  (no `F_FULLFSYNC`), so macOS gives no durability guarantee against power loss.

## 10. API sketch

```rust
#[derive(Clone)]
pub struct Db { /* Rc<Inner> */ }

impl Db {
    pub async fn open(path: impl AsRef<Path>, opts: Options) -> Result<Db>;
    /// Applies `batch` atomically and durably: once this returns `Ok`, the batch is
    /// on disk and visible to reads. Fails with `Error::EntryTooLarge`, applying
    /// nothing, if an entry is too large (§5).
    pub async fn write(&self, batch: &mut WriteBatch) -> Result<()>;
    /// Finds the entry matching `key` under `cmp` and calls `f` with its key and
    /// value, returning whatever `f` returns. The slices are only valid inside `f`,
    /// which is how `get` avoids allocating: decode or copy what you need there.
    ///
    /// ```ignore
    /// // Location of entry `e` in ledger `l`.
    /// let loc: Option<Location> = db
    ///     .get(&key(l, e), Comparison::Equal, |_k, v| Location::decode(v))
    ///     .await?;
    ///
    /// // Last entry of ledger `l`: the largest key <= (l, u64::MAX), if it is in `l`.
    /// let last: Option<u64> = db
    ///     .get(&key(l, u64::MAX), Comparison::Floor, |k, _v| decode_key(k))
    ///     .await?
    ///     .and_then(|(ledger, entry)| (ledger == l).then_some(entry));
    /// ```
    pub async fn get<R>(
        &self,
        key: &[u8],
        cmp: Comparison,
        f: impl FnOnce(&[u8], &[u8]) -> R,
    ) -> Result<Option<R>>;
    /// Point-in-time iterator. Creating and advancing it never allocates.
    pub fn range(&self, range: impl RangeBounds<[u8]>) -> Iter;
    /// Closes the store. Every successful `write()` is already durable.
    pub async fn close(self) -> Result<()>;
}

/// A read-only view of a store, as of when it opened, which takes no lock (§9).
#[derive(Clone)]
pub struct ReadOnlyDb { /* Db */ }

impl ReadOnlyDb {
    /// Fails with `Error::Stale` if the owner overwrote the version meanwhile.
    pub async fn open(path: impl AsRef<Path>, opts: Options) -> Result<ReadOnlyDb>;
    /// As `Db::get` and `Db::range`. Once the owner overwrites pages they need,
    /// they fail with `Error::Stale`.
    pub async fn get<R>(
        &self,
        key: &[u8],
        cmp: Comparison,
        f: impl FnOnce(&[u8], &[u8]) -> R,
    ) -> Result<Option<R>>;
    pub fn range(&self, range: impl RangeBounds<[u8]>) -> Iter;
    pub async fn close(self) -> Result<()>;
}

pub enum Comparison { Equal, Floor, Ceiling, Lower, Higher }

impl Iter {
    /// Lends the next entry; the slices stay valid until the next call.
    pub async fn next(&mut self) -> Result<Option<(&[u8], &[u8])>>;
}

impl WriteBatch {
    pub fn put(&mut self, key: &[u8], value: &[u8]);
    pub fn delete(&mut self, key: &[u8]);
    pub fn delete_range(&mut self, start: &[u8], end: &[u8]);
    /// Empties the batch, keeping its memory for reuse.
    pub fn clear(&mut self);
}
```

Reads are `async` because a leaf may have to be read from disk; on a cache hit they
complete without suspending.

| BookKeeper `EntryLocationIndex` | encina |
|---|---|
| `addLocation` batch, then flush | `write(&mut batch)` |
| `getLocation(l, e)` | `get(&key(l, e), Equal, …)` |
| `getLastEntryInLedger(l)` | `get(&key(l, u64::MAX), Floor, …)`, then check the ledger id |
| `delete(l)` | `delete_range(&key(l, 0), &key(l + 1, 0))` |
| `updateLocations` (compaction) | `write(&mut batch)` with puts |

## 11. Testing

- **Format tests:** encode and search round trips for the node format, including
  mixed lengths, empty values, and keys that are prefixes of each other.
- **Model-based property tests:** random sequences of writes (put, delete,
  delete_range), get with every comparison, range and reopen, checked against a
  `BTreeMap`, with a tiny cache budget so eviction and reloads happen constantly.
- **Invariant checker** after every step: keys sorted, internal keys exact, levels
  consistent, and every page either live or free (never both, never neither).
- **Failure paths, aiming for ~100% coverage.** All I/O goes through one internal
  interface with two implementations: compio files, and a simulated disk that can
  fail or crash at any single operation. A crash keeps what was synced, a
  pseudo-random subset of the pages written since, and the first half of the write
  in progress, which may span several pages. The failure scenario first runs once
  to count its I/O operations, then reruns once per operation with an error
  injected there, and once more with a crash. After each run, a store left usable
  must pass the invariant checker, one left failed must reject writes, and
  reopening must yield the state after the last successful write (or after the
  failed one, when the failure hit a commit).
  This enumerates every failure point instead of sampling them; the scenario's
  reads include scans, so failures hit reads ahead too. CI measures coverage on
  every pull request (`cargo llvm-cov`), posts the totals as a status, and lists
  the lines and regions no test runs, such as the error branch of a `?`. An
  uncovered failure path needs a test or a written reason.
- **Reads in flight:** the simulated disk can also hold leaf reads until a test
  releases them, and keeps the buffer of a held read that is cancelled, as a kernel
  would. Tests use it to check that concurrent misses share one read, that an
  iterator reads ahead and waits for the leaf it read ahead rather than reading it
  again, that a read survives as long as some reader holds it, that a failed read
  reaches the reader that claims it, and that a cancelled read's frame is only
  reused once the read finishes.
- **Damage:** a flipped bit in a leaf fails the reads that need it, and one in an
  internal node fails the open, both with `Error::Corruption`.
- **Read-only views:** a view on the owner's simulated disk returns exactly its
  version while the owner rewrites every leaf, until its reads fail with
  `Error::Stale`, and never with other entries. Damage in a store the owner hasn't
  written since is still `Error::Corruption`. The simulated disk can also take
  another disk's contents at a given operation, as if the owner wrote meanwhile:
  a view whose internal nodes are overwritten that way while it opens fails with
  `Error::Stale`.
- **Real files** cover what the simulated disk doesn't model: reopening, the lock
  that keeps out a second open, syncing the directory of a new store, including
  a directory that can't be synced, and a read-only view, which opens beside the
  owner, creates nothing, and opens a file it may only read.
- **Allocation tests:** a counting global allocator asserts that `get` and
  iteration over cached pages, `get` and iteration that miss the cache (reading
  ahead included), and `write` with a reused batch, all on the simulated disk,
  perform no allocations. On a real disk, compio adds one per I/O operation.
- **Sanitizers:** CI runs every test under AddressSanitizer, std included, so a
  use of freed memory or a leak fails the run even where the test's own checks
  pass, real files' io_uring path included. One test drops a store while the
  simulated disk still holds its reads, which checks that their buffers keep the
  frame memory alive until the reads finish.
- **Benchmarks** against RocksDB on the BookKeeper workload: sorted batches over
  *A* active ledgers, random point reads, floor lookups, and `delete_range` of old
  ledgers.

## 12. Milestones

1. **M1, in-memory tree:** the slotted node format with in-place search, frames,
   versions, batch writes applied in memory, get with every comparison, range,
   delete_range, property and allocation tests. No I/O.
2. **M2, persistence and caching:** page I/O on compio with direct I/O, the commit
   at the end of `write()`, open and recovery, free bitmap, leaf cache with budget
   and eviction, snapshot-aware page reuse, failure-path tests.
3. **M3, performance:** group commit (writes that queue up during a commit share
   the next one), the dense node format if benchmarks justify it, yielding inside
   large merges, writing large batches' new leaves early, one shared read for
   concurrent misses and reading ahead in iterators (done, §8), coverage measured
   in CI (done, §11), and benchmarks against RocksDB.

## Non-goals (for now)

- Access to one instance from multiple threads.
- An internal WAL.
- Transactions beyond atomic batches.
- Entries larger than a quarter of a page (no overflow pages). The limit is part of
  the documented public API (§5).
- Custom comparators: keys compare as bytes.
