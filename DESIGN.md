# encina — design

> Status: M1 and M2 are implemented: the tree, persistence and the leaf cache.
> M3, performance, is under way (§12): writes that queue up share commits (§6),
> large writes pause to let other tasks run (§6), concurrent misses share one
> read, iterators read ahead (§8), nodes whose entries allow it are dense (§5),
> and bench/ compares encina with RocksDB (§11). Writing large batches' new
> leaves early remains. M4 is done: entries too large for a leaf get value
> leaves, which the cache keeps and iterators read ahead (§5).

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
  (each `write()`, or each group of them, §6) rewrites each leaf that received at
  least one key. Touched leaves ≤ min(new entries, active ledgers *A*), so a
  commit writes at most about `max(new entry bytes, A × page size)` plus internal
  nodes. With *A* = 10,000 and 4 KiB pages, that is at most 40 MiB per commit,
  however few entries each ledger got. Against the entry data written in the same
  interval (a write-cache flush is hundreds of MB), that is a few percent. **Big
  batches are the operating point**; small, frequent writes are not.
- **Random inserts are the worst case**, since every key touches a different leaf.
  That is not this workload.
- **Copy-on-write scatters leaves on disk**, so long scans are not physically
  sequential. That doesn't matter for cached leaves, and random 4 KiB reads are
  cheap on NVMe.

## 3. Decisions

| Topic | Decision |
|---|---|
| Durability | Each `write()` is atomic and durable: when it returns, the batch is on disk, and readers see it from that moment on. Writes that queue up behind a commit share the next one (§6). There is no separate flush and no internal WAL: after a crash the store opens at the last completed write, and the caller's journal replays anything newer. |
| Threading | One `!Send` instance per compio thread, no locks or atomics, except in wakers, which Rust requires to be `Send` and `Sync` (§8). Scale by sharding (for example, by ledger id) above the library. |
| Memory | Bounded from day one and managed entirely by encina: internal nodes stay resident, and leaves live in encina's own leaf cache with a budget, read on demand (§8). No reliance on the OS page cache or `mmap`: the data file is opened for direct I/O (§9). |
| Allocation | No heap allocation on the data path. Frames in the leaf cache are the only exception (§8), besides the buffer a value leaf too large to cache is read into (§5); write batches, the batches that groups of writes share, cursors and reader bookkeeping reuse their memory. compio itself allocates once per I/O operation, so cache misses and commits are not allocation-free (§9). |
| Keys and values | Arbitrary byte strings, keys up to a quarter of a page. Two node formats: slotted, which handles any lengths, and dense, for nodes whose keys share one length and whose values share another, chosen per node when it is built. An entry too large for either gets a value leaf of its own (§5). |

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
- **One commit at a time.** Writes that arrive while a commit runs queue up, and
  the next commit takes them all (§6). Reads never wait for writes.

## 5. Keys, values and node formats

Keys and values are arbitrary byte strings. Keys are ordered lexicographically, so
callers encode integers big-endian.

A key must fit in a quarter of a page (1 KiB with 4 KiB pages): every internal
node then holds at least four children. This limit is part of the public contract.
It is documented on `WriteBatch::put` and `Options::page_size`, and `write()`
rejects a batch containing an oversized key with `Error::KeyTooLarge` before
applying any of it. An entry whose key plus value exceeds a quarter of a page gets a
value leaf of its own (below), so every ordinary leaf holds at least four entries,
and there are no overflow pages.

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

### Node formats

Both formats store the prefix that all keys in the node share once, in the header,
and both are searched in place, without decoding. The page header records each
node's format, so copy-on-write picks one for every node it builds.

- **Slotted** handles any key and value lengths. The body holds a contiguous array
  of 4-byte key heads (the first bytes of each suffix, as a big-endian `u32`), a
  slot array (offset, suffix length, value length), and a heap with the rest of each
  suffix and the value. The binary search reads only the heads, and ties fall back
  to the heap.
- **Dense** holds nodes whose keys all have one length and whose values all have
  one length. A leaf also stores once the prefix its values share. The key suffixes
  sit in one array at a fixed stride, and the value suffixes in another, so an entry
  costs only its two suffixes. Key suffixes of up to 8 bytes compare as big-endian
  integers.

A node is dense whenever its entries allow it, since it is then never larger. On the
benchmark's BookKeeper-shaped data, a full 4 KiB leaf holds 807 entries, 5.1 bytes
each, against about 26 for slotted leaves, as the experiment above predicted.
Internal nodes use the same formats, with child references as fixed-size values. A
dense internal node never shares a value prefix, since loading it rewrites the
frame in each reference. §7 gives the formats byte by byte.

Reads never allocate. Results reach the caller as borrowed slices. The key is
rebuilt from prefix and suffix in a buffer owned by the call. The value comes
straight from the leaf page, except in a dense leaf whose values share a prefix,
where it is rebuilt in the same buffer, after the key.

### Value leaves

Value leaves lift the limit on value sizes: an entry too large for an ordinary leaf
gets a leaf of its own, which spans as many adjacent pages as it needs. Keys keep
the limit, since each leaf's smallest key is also a key in its parent, and an
internal node must fit four children with keys that long. Values are written and
read whole, so a value must fit in memory, and a batch holds at most 4 GiB.

**Format.** A value leaf holds one entry: a header like any node's, with a format of
its own, then the key, the value, and zeros to the end of its last page (§7). The
header's checksum covers every page of the span. An entry goes to a value leaf when
its key plus value exceeds the limit of ordinary leaves; its key alone may not.

**References.** A reference to a value leaf records its span with its page id. The
8-byte page field keeps the page id in its low 40 bits and the span in its high 24,
which are 0 for every other node, so references keep their size and a parent knows
a value leaf without reading it. The span then goes wherever the reference goes,
the pages a write drops included. A value leaf always has a parent: a store whose
only entry is large keeps an internal node above it, so the meta page's root is
never one. With 4 KiB pages, 40 bits address 4 PiB, and 24 bits give spans of up to
64 GiB, more than a batch holds.

**Free space.** The bottom internal level lists each value leaf with its span, so
the open still builds the free bitmap from internal nodes alone (§7), and
`delete_range` still drops value leaves without reading them. A dropped value leaf's
pages become reusable together, subject to §8. A value leaf needs adjacent pages: it
takes them from the write's run of long-lived pages when what is left of that run
holds it, and otherwise from the first free stretch long enough, or from the end of
the file. A write never moves a value leaf out of a mostly free window (§7): that
would cost reading it.

**Writes.** A write never reads a value leaf. A value leaf's key is its key in the
parent, so the descent knows which operations reach it: a put or delete of that key
replaces it, a range that covers the key drops it, and the other operations in its
child range go to a new leaf beside it, which may then merge with its other neighbor
like any underfull leaf. When a write rebuilds a leaf, a put too large for it
becomes a value leaf between the leaves that the entries before and after it fill,
and the split policy (§6) runs on each side. A value leaf is never underfull and
never merges with a neighbor. Its parent keeps the reference as it is, so later
writes never copy the value, while an entry in an ordinary leaf is copied each time
its leaf is rebuilt.

**Commit.** A value leaf is built only as the commit writes it. Direct I/O can't
write from the batch, whose bytes aren't aligned, and building the whole leaf first
would double the memory a large value takes. So the commit computes the checksum in
one pass over the value, then copies the value from the batch into frames one
request at a time, and writes its pages like any run (§7): requests of up to 1 MiB,
16 in flight. It frees the frames once they are written. The value is then in
memory once, in the batch, plus the requests in flight. A write that queues behind a
commit still copies its batch into the group's (§6), values included.

**Reads.** The descent finds a value leaf's key in its parent, so `get` resolves its
comparison without reading the value leaf. It then finds the value leaf in the
cache, or reads it, checks its checksum and generation, and lends the key and the
value to `f`, straight from the frame. Like a leaf read, it pins the version while
it waits. An iterator that reaches a value leaf tells from the key in the parent
whether it is in the range, and only then enters it. A read-only view treats a value
leaf that fails its checks, or reaches past the end of the file, as it treats a
leaf: it may be stale (§9).

**Caching.** The cache keeps a value leaf in a large frame: one frame of as many
pages as the leaf spans, in memory of its own, 4 KiB-aligned for direct I/O. Its
pages count against the budget like any frame's, and the clock hand evicts it like
any leaf. Value leaves take at most a quarter of the budget in all, so a large
value never evicts the whole cache, and one larger than that quarter isn't cached:
each read of it reads it into a buffer that the read allocates, or that an iterator
keeps for the next one. Keeping them within that quarter takes another value leaf's
frame, which a clock hand of their own finds: it goes over the large frames only.
With one hand over every frame, a lookup that missed on a value leaf walked it past
all the others each time it came round, which, with 16K value leaves cached among
40K other frames, took up to 0.15 ms (§6). Otherwise value leaves go through the
table of reads in flight like any leaf (§8): concurrent misses share one read,
iterators read value leaves ahead, and a cached one is read without allocating. The
page frames a large frame evicts stay in the pool, which never shrinks, so memory
can reach 1.25 times the budget when the pool has grown to it before value leaves
take their quarter. A large frame's memory goes back to the allocator once it is
evicted. A value leaf is cached under its page id, whose span bits set it apart from
any ordinary leaf, and reclaiming its pages evicts it (§8).

**Costs.** A value leaf rounds its entry up to whole pages: an entry just over the
limit fills a bit more than a quarter of its page, and on average half a page per
value leaf is padding. A larger page size raises the limit, and keeps more values in
ordinary leaves. In memory, a value leaf costs its parent one reference, as any leaf
does. Reading one into the cache allocates its large frame.

**API.** `WriteBatch::put` takes values of any length the batch holds, and keys up
to the limit, which `Options::max_key_size` reports; `write()` rejects a longer one
with `Error::KeyTooLarge`. `get` and `range` don't change: values still reach the
caller whole, as borrowed slices. `PageInfo` says whether a node is a value leaf,
and how many pages it takes, so `encina stats` and `encina pages` show value leaves
apart.

**Alternatives.**

- Overflow pages that leaves point to, as in LMDB and SQLite, pack leaves better.
  But leaves aren't resident: the open would have to read every leaf to build the
  free bitmap, and `delete_range` the leaves it drops, unless free space were stored
  rather than derived (§7).
- A value log, as in WiscKey, needs garbage collection: the compaction §2 avoids.
- Chunks of each value as ordinary entries, in a second tree keyed by the encoded
  key and a chunk number, keep every invariant without a new format. But a read
  gathers a value from many leaves and copies it, and every write and `delete_range`
  updates both trees.

**Testing.** Format tests round-trip value leaves a page at a time, and spans in
page ids. Cache tests check that concurrent misses on a value leaf share one read,
that a cold scan reads value leaves ahead within the window and the value leaves'
quarter of the budget, that making room within that quarter passes no other frame,
that a value leaf larger than that quarter is read each time, and that a value leaf
whose pages are reclaimed leaves the cache, though the next one on those pages has
the same page id. The model-based tests write values on both sides of the limit,
alone, in groups and during scans, overwrite small values with large ones and back,
cover value leaves with `delete_range`, and land every comparison on and beside
them; the invariant checker counts each span's pages as live or free. A store whose
value leaves are all damaged still takes writes around them, over them and across
them, which shows that writes don't read them. A value of 17 MiB takes 18 write
requests. The failure tests inject an error at each operation of a scenario with
value leaves, one of them larger than a request, and crash there. A read-only view
whose value leaves the owner reuses gets `Error::Stale`. The allocation tests check
that reading a cached value leaf allocates nothing, and that reading one too large
to cache allocates its buffer and nothing else, which an iterator then reuses.

## 6. Operations

### `write(batch)`

A `WriteBatch` holds `put`, `delete` and `delete_range` operations, encoded into an
arena that `clear()` empties without releasing, so a caller that reuses its batch
doesn't allocate. It also keeps the length of the longest key put, so that a write
checks the limit on keys (§5) without reading the operations. The operations are
applied in order and atomically.

1. **Normalize.** Drop point operations made obsolete by a later range in the same
   batch. Sort the rest by key, through an index array kept in the batch, keeping
   the order of equal keys, and keep the last operation per key. The sort is a
   natural merge sort: it finds the runs sorted already, then merges them in pairs,
   so sorted input (BookKeeper's write cache is sorted) costs one linear check, and
   a group of k sorted batches log2(k) passes. A run shorter than 4,096
   operations, as unsorted batches have, is first extended by sorting 4,096 at once
   with the standard library's sort. The ranges are sorted by start the same way,
   and the pass that keeps the last operation per key finds the later range that
   covers it, if any: the ranges that start at or before the key and end after it
   wait on a heap kept in the batch, the one added last on top, which comes off
   once it ends at or before the key. For a group that appends 1M entries to 1,000
   ledgers and deletes 500 older ones, normalizing takes 6 ms on an Apple M5 Pro,
   where checking each key against every range that starts before it took 0.8 s.
2. **Plan.** Descend the resident internal nodes to find the leaves the batch
   changes, and load the ones that aren't cached, keeping them pinned. Subtrees
   that ranges remove whole aren't read. Then free the pages that readers held
   back, and have let go of since (§8), so that the merge can take them.
3. **Apply ranges** (see below).
4. **Bulk merge.** Descend recursively, partitioning the sorted operations by child
   key range. Each touched leaf is rebuilt once from its old entries plus the
   operations and re-encoded into one or more pages: the old entries between two
   operations are found with a search, and copied without comparing each. Each
   parent rebuilds its child list, splitting if needed, up to the root.
5. **Commit** the new version (§7).
6. **Publish** the new root. Readers see the batch only now, once it is durable,
   and the pages of the nodes it replaced become free, subject to §8.

**Pausing.** Readers keep running while a write waits for the disk, and while it
normalizes and plans its batch, pins the leaves it changes, frees the pages that
readers let go of, merges it, and, once published, frees the pages of the nodes it
replaced. That is CPU work that grows with the batch, so once it has run for a
millisecond, it pauses and lets the other tasks run; the slice in which one step
ends goes on with the next. It looks at the clock every 4,096 units of work:
operations listed, compared or moved, nodes visited, leaves pinned, entries copied,
merged or built, and nodes freed. A node that planning visits counts as 64, since
searching an unsorted batch's operations for it takes up to 2 µs, and a leaf pinned
or a node freed as 16, so that it looks every 256 of them: about 20 µs of pinning
cached leaves, and 11 µs of freeing. An operation the merge applies counts as 32
when the first two operations of its leaf don't come one after the other in the
batch, as a sorted batch's do: an unsorted batch's operations lie scattered, and
reading each one misses the cache. Normalizing a batch larger than any it held
before takes room for all its points at once, rather than growing their array as
it lists them, and zeroes the sort's new scratch space a page at a time, counting
a unit per item. That is safe: readers keep using the old root
until the swap, only one commit runs at a time, and the pages a write hasn't freed
yet stay in use. Each step keeps its place in fixed state. Normalizing keeps the
step it is at and how far it got: in the merge sort, the pass, the pair of runs it
merges, and the operation it is at in each. Pinning keeps how many of the leaves it
pinned. Planning and the merge keep one record per level of the path they are on:
the child they are at, the operations left for the children after it, and how far
the merge got rebuilding a leaf or building nodes. Freeing keeps nothing: the nodes
left to free wait in the list of dropped ones (§8). So pausing allocates nothing,
and resuming descends again to where it left off. A merge that pauses builds the
same tree, on the same pages, as one that doesn't: readers that end meanwhile free
no pages (§8), and the next commit starts once the pages are free.

Measured with the ignored test `read_latency_during_a_load`, on an Apple M5 Pro: a
store of 2M BookKeeper-shaped entries on the simulated disk takes writes of 1M
entries, while a lookup is due every 20 µs. A lookup's latency runs from when it
was due until it returns. The simulated disk lets the lookups run before each write
request and sync, and finishes one request per turn of the runtime, as a real disk
works while the thread runs other tasks: the test times the thread alone.

| Write of 1M entries | Write | Lookup p50 | p99 | Max |
|---|---|---:|---:|---:|
| Appends to 1,000 ledgers | doesn't pause | 20.0 ms | 42.9 ms | 43.8 ms |
| | pauses | 0.50 ms | 1.02 ms | 1.08 ms |
| Random updates | doesn't pause | 183 ms | 377 ms | 383 ms |
| | pauses | 0.52 ms | 1.06 ms | 1.22 ms |

Normalizing, planning, pinning, merging and freeing then never run for much more
than 1.5 ms at a time, and a lookup waits for what is left of the slice in which it
comes due: half of one at the median, and about a whole one at the p99. Most slices
run 1.0 to 1.1 ms, the random updates' merge slices too, since their operations
count as 32 units each: counted as one, 4,096 units of their merge, between two
looks at the clock, took 180 µs on an idle machine and up to 0.5 ms on a busy one,
and their merge slices ran up to 1.3 and 1.5 ms. Over 48 alternating runs on a
machine kept busy by other work, counting them as 32 made the random updates' p99
1.07 ms at the median rather than 1.20, lower in 47 runs, and their max 1.47 ms
rather than 1.54, lower in 32. Their max then came from normalizing a new batch,
whose arrays grew by megabytes at once, in up to 0.65 ms. Over 40 alternating runs
against the build before both changes, normalizing in slices too made the random
updates' p99 1.06 ms rather than 1.18, and their max 1.22 ms rather than 1.49, lower
in 35 runs. Planning sets it now, looking at the clock every 150 to 200 µs of a busy
machine, along with the standard library's sorts of 4,096 operations, 0.2 ms each.
The simulated disk copies a request's pages in at most 0.25 ms. The rest of a
commit is short: the merge lists the nodes it builds as it goes, in page order, so
the commit needn't find them in the new version and sort them by page, which took
the random updates' 19,700 nodes another 0.7 ms after the merge's last slice; and
freeing the nodes the commit replaced, 0.8 ms of work, pauses. Over 35 alternating
runs on a machine kept busy by other work, doing both at once made the random
updates' max 1.57 ms at the median rather than 1.50, and 16 runs rather than 5
reached 1.6 to 2.5 ms. Checking a write's keys against the limit doesn't read its
batch: reading 1M operations would take 1 ms, right before the first slice of
normalizing, and make the appends' p99 1.5 ms rather than 1.0.

A lookup that misses the cache pins its version while it waits for the disk (§8),
so the lookups in flight when a commit publishes hold back the pages of the nodes
it replaced. The ignored test `read_latency_with_misses_during_a_load` runs the
random updates above with lookups that miss seven times in ten: they read entries
of their own, whose values of 2 KiB take value leaves, which the cache keeps in a
quarter of its 256 MiB, and the simulated disk's leaf reads take 100 µs. A few
lookups are in flight whenever a commit publishes. The last of them to end frees
none of those pages, which the next write frees before it merges, a slice at a
time: no lookup runs longer than 0.06 ms in the poll in which it lets go of its
version, 0.03 ms at the median of 25 runs. Freeing them then, which can't pause,
took it 0.41 ms at the median for the 17K to 20K nodes a write replaced, and up to
0.77 ms. The longest a lookup runs without pausing is 0.08 ms at the median of 25
runs, down from 0.15 ms when the clock hand that looks for a value leaf's frame to
evict went past the ordinary leaves' frames too (§5). What is left is the first time
that hand goes round the value leaves, once they fill their quarter: each of the 16K
is still marked as used from when it was read, so the hand clears them all before it
finds one, in 70 µs; and the allocator, which now and then takes as long to give a
large frame its memory. The lookups' latency is the same either way, 1.5 ms at the
median, 2.5 ms at the p99 and 2.8 to 2.9 ms at most: a lookup that misses waits for
what is left of the write's slice before its read, and for another slice after it.

A pause is a trip through the runtime, about 15 µs on an Apple M5 Pro, which is
why slices are measured in time: a fixed amount of work takes ten times longer in
some batches than in others, since the operations of an unsorted batch lie
scattered in its memory. Slices of 16K units, a tenth of a millisecond of the
appends' merge, made the benchmark's load (bench/) 8% slower. With slices of a
millisecond it is about 2% slower with one writer, and no slower with four, within
the runs' noise. Counting scattered operations as 32 costs it nothing: over 30
alternating runs, each round's time over that of the build before was 0.995 at the
median with one writer and 1.008 with four, as close to 1 as a second copy of the
same build came, 0.985 and 0.998.

Normalizing and planning in slices cost no more, since the merge sort is faster
than the standard library's sort on the batches the benchmark writes. Over 15
alternating runs of its load, against a build that never pauses, the median time
was 0.6% lower with one writer, 3.8% lower with four, whose groups of sorted
batches the merge sort merges rather than sorting them again, and 0.4% higher with
commits of 100 entries, within the runs' noise; pausing only the merge made it
1.7%, 0.9% and 1.2% higher in the same runs. On its own, the merge sort normalizes
1M sorted operations in 7.3 ms rather than the 8.5 ms the standard library's sort
took, four sorted batches of 250K in 11 ms rather than 50, and 1M random ones in
155 ms rather than 140.

Pinning in slices costs no more either: over 40 alternating runs of the load, the
median time was 0.7% higher with one writer and 0.5% higher with four than when
pinning never paused, and the median of each round's ratio 0.6% higher and 0.7%
lower, within the runs' noise.

**Split policy.** A rebuilt leaf that overflows splits right after each run of
inserted keys that fills at least a quarter page (the per-ledger append case): the
run's leaves are packed full, and the existing entries after it start a new leaf. A
ledger's settled entries then stop being rewritten by later appends, even when one
leaf holds the tails of several ledgers. Smaller runs, like scattered inserts, don't
get leaves of their own, and whatever follows the last split point is split evenly
by bytes. There is no strict 50% minimum fill: a rebuilt node below ~25% merges with
a neighbor, if the neighbor is in memory. A level's new nodes are only built once
its merges are decided, so a write builds no node that a merge then replaces.

### Group commit

One commit runs at a time. A write that finds none running, and no write waiting,
commits its own batch as above. Writes that arrive while a commit runs queue up for
the next one: each checks its entries' sizes, then appends its batch to the batch
of the next group, which the store keeps. Once the commit ends, whichever write of
the group runs first commits the whole group: it normalizes the group's batch,
which applies the batches in order exactly as if they had been written one after
another, then plans, merges and commits once. A group of N writes costs one merge,
one round of page writes and two `sync_data` calls instead of N of each, plus a
copy of each batch. The other writes return the group's outcome once it is
published.

- **Order.** Batches apply in the order their writes queued, each atomically: a
  group commits whole, or not at all.
- **Size.** A group takes batches until they hold 64 MiB, counting operations and
  their keys and values. A write that doesn't fit waits for the next group, unless
  its batch would be the group's first. The store keeps two group batches, the one
  committing and the one filling, and reuses their memory, so once warmed up,
  queueing allocates nothing.
- **Errors.** A batch with a key too long is rejected before it queues,
  alone. Any other error fails every write in the group: before the commit, it
  leaves the store unchanged; during it, the store fails (§7).
- **Cancellation.** The group holds a copy of each batch, never a reference to a
  caller's. Dropping a write withdraws its batch while the write waits, and while
  its group is being built, its normalizing, planning and merge included: the write
  building it then starts over without that batch, since leaves the batch's ranges
  covered may now be needed. Once the group
  writes pages, its batches commit together, whichever of its writes are dropped.
  Dropping the write committing them then fails the store, as it does for a single
  write, and its group's other writes return `Error::Failed`. If the write
  building a group is dropped before it writes anything, another write takes the
  group over: one of the group's, or one queued for the next group, which then
  commits its own next. So no write depends on another being polled, except the
  one committing.

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

Every `write()` ends with a commit, its own or its group's (§6), that makes its
batch durable in one atomic step.

1. Take the new version's dirty nodes, which the merge listed as it built them. It
   sealed each, stamping the generation and the checksum, and gave it its page, so
   they can be written in any order. It keeps one list for the pages likely to be
   replaced soon and one for those likely to stay (see below), each in the order
   it built them, which is page order: each kind takes its pages in order.
2. Write them in page order, merging the two lists: each run of adjacent pages with
   one vectored write of up to 1 MiB, 16 writes at a time. Await every write, then
   `sync_data`. Allocation hands a write its pages in runs (see below), so a commit
   takes a few large requests rather than one per page.
3. Write the meta page into slot `(gen + 1) % 2`, then `sync_data`. The commit is
   durable.
4. Publish the new root. The written leaves become evictable, and the pages this
   write dropped become reusable, subject to §8. The group's other writes return.
5. If an eighth of the file or more lies past the high-water mark that the meta
   page recorded, cut the file there (`set_len`). Only that meta page, which is
   durable, still matters, and no version a reader holds uses a page past the
   mark. If the cut fails, the file stays longer than it needs to be until a later
   commit cuts it.
6. Free the pages that no reader needs any more, a slice at a time (§6, §8). Then
   the write returns.

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
writes return `Error::Failed`, and the store must be reopened. Every write in a
group gets the error. Dropping the `write()` future that is committing is harmless
until it starts writing pages; after that, it also puts the instance in the failed
state, even while it only cuts the file: a cut still in flight could cut what a
later write adds. Dropping any other write is harmless (§6).

### File layout (format v1)

One file with a fixed page size (default 4 KiB, chosen at creation, a power of two
from 4 to 64 KiB). Opening it for writing with options that set another page size
fails; a read-only view, which writes nothing, takes the file's.

| Pages | Content |
|---|---|
| 0, 1 | The two meta slots. |
| ≥ 2 | Tree pages: one node per page, or a value leaf over adjacent pages (§5). |

Integers are little-endian unless said otherwise, and checksums are crc32c. Nothing
else is stored: free space is derived from the tree (below).

**Meta page.** Each slot fills a page, and the commit of generation `g` writes slot
`g % 2`. The bytes past the 48 below are zero.

```text
offset  size  field
     0     4  checksum of bytes 4..48
     4     8  magic: "encina\0\0"
    12     4  format version: 1
    16     4  page size, in bytes
    20     1  root level: 0 when the root is a leaf
    21     3  zero
    24     8  generation
    32     8  root's page id, 0 for an empty tree
    40     8  high-water mark: no page at or above it is in use
```

**Node header.** Every node starts with 32 bytes:

```text
offset  size  field
     0     4  checksum of bytes 4 to the end of the page
     4     1  format: 1 slotted, 2 dense, 3 value leaf
     5     1  level: 0 for leaves
     6     2  entry count
     8     2  key prefix length
    10     2  slotted: offset of the heads; dense: key suffix length
    12     2  slotted: offset of the slots; dense: value prefix length
    14     2  slotted: zero; dense: value suffix length
    16     4  bytes used, from the start of the page
    20     8  generation that wrote it
    28     4  zero
```

The checksum and generation are stamped when the page is written. The bytes past
the bytes used belong to no entry, though the checksum covers them too. Every key in
a node starts with the node's key prefix: the longest prefix its first and last keys
share, stored once, right after the header. The rest of a key is its suffix. Entries
are in key order.

**Slotted nodes** hold keys and values of any lengths:

```text
32               key prefix
                 zeros, to a multiple of 4
heads offset     heads: 4 bytes per entry
slots offset     slots: 6 bytes per entry
slots + 6 × n    heap: a record per entry
```

- A head is the first 4 suffix bytes, zero-padded and read as a big-endian `u32`,
  stored as a little-endian one. A binary search reads only the heads, and ties
  fall back to the records.
- A slot holds the record's offset from the start of the heap, the suffix length and
  the value length, 2 bytes each.
- A record holds the suffix bytes past the first 4, then the value.

**Dense nodes** hold entries whose keys all have one length, and whose values all
have one length:

```text
32                         key prefix
32 + kp                    value prefix
32 + kp + vp               key suffixes: ks bytes per entry
32 + kp + vp + n × ks      value suffixes: vs bytes per entry
```

The value prefix is the longest prefix every value shares, in leaves; internal nodes
have none. Key suffixes of up to 8 bytes compare as big-endian integers. A node is
built dense whenever its entries allow it, since it is then never larger.

**Internal nodes** use the same formats, at levels 1 and up. Entry `i`'s key is the
smallest key in child `i`'s subtree, and its value is a child reference:

```text
offset  size  field
     0     4  frame: the child's frame in memory, NO_FRAME (0xFFFFFFFF) for a leaf
     4     8  the child's page id
```

The frame only means something in memory: it is written to disk as it stands, and
loading the node rewrites it. So an internal node stores its references whole: a
dense one has no value prefix, and a value suffix of 12 bytes. Leaves are found by
page id, through the cache, so a level-1 node's frames are all `NO_FRAME`.

**Page ids.** The low 40 bits hold a node's first page, and the high 24 bits a value
leaf's span, in pages: 0 for every other node. With 4 KiB pages, 40 bits address
4 PiB, and 24 bits give spans of up to 64 GiB.

**Value leaves** hold one entry, over as many adjacent pages as it needs:

```text
offset  size  field
     0     4  checksum of bytes 4 to the end of the last page
     4     1  format: 3
     5     1  level: 0
     6     2  entry count: 1
     8     2  key length
    10     2  zero
    12     4  value length
    16     4  zero
    20     8  generation that wrote it
    28     4  zero
    32        the key, the value, then zeros to the end of the last page
```

**Checks.** A node read from disk must have a known format, a matching checksum,
entries, and a layout that stays within its page. Its level must be the one its
parent expects, and its generation no newer than the meta page's. A value leaf's
length must fill its span, give or take the last page's zeros.

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
around them are freed too. `Options::min_run_pages` sets the 16: a smaller value
trades requests for a smaller file. Every write starts over from the lowest free
page. The pages it frees again before it ends, as when its new root loses more
than one level, are holes in its own runs, which its next short-lived pages fill
(see below).

Pages replaced together should be freed together, as runs again, so a write
allocates from two runs at once: one for pages likely to be replaced soon
(internal nodes, and leaves that may take more keys), and one for the full leaves
that a run of appends leaves behind, which are likely to stay. With appends, as to
ledgers, each batch replaces the short-lived pages of the batch before as a whole,
and the batch after reuses their space as one run.

Updates spread over the keys don't free pages together: a few pages that stay
hold a whole stretch back, as short holes that no run takes, until the last of
them goes. So a write moves such pages when that costs only their writes. When it
rebuilds an internal node, it gives the node's unchanged children new pages, next
to the pages it builds, if they sit in a window of 64 pages (one word of the
bitmap) with at most 16 in use, and, for leaves, if they are cached. Their parent
is rebuilt anyway, so a move takes no read and no other write; each page moved
frees at least three, and the window, once empty, makes a run for a later write. A
write moves at most as many nodes as it builds for its changes.

The file gives back its end once that is free: every write starts by lowering the
high-water mark to the last page in use, which its meta page records, and the
commit cuts the file once an eighth of it lies past the mark (step 5 above).

Writing in runs costs space, in exchange for requests. The table shows the file
against the pages in use, and the write requests and pages per commit, measured
on the simulated disk with 4 KiB pages and a cache that holds everything (the
ignored test `file_size_and_requests`): the benchmark's load (bench/README.md: 10
million entries in 100 batches), and 1 million random keys, loaded in batches of
20,000, then updated at random, 2 million updates in all, in commits of the size
shown.

| Workload | Allocation | File | Requests | Pages |
|---|---|---:|---:|---:|
| Benchmark load | lowest free page | 1.02× | 1,790 | 1,790 |
| | runs | 1.21× | 25 | 1,790 |
| | runs, moving pages | 1.22× | 26 | 1,807 |
| Random, 20,000 per commit | lowest free page | 1.72× | 10,882 | 10,882 |
| | runs | 2.73× | 433 | 10,882 |
| | runs, moving pages | 2.63× | 316 | 11,319 |
| Random, 1,000 per commit | lowest free page | 1.70× | 1,158 | 1,158 |
| | runs | 3.00× | 57 | 1,158 |
| | runs, moving pages | 2.08× | 26 | 1,420 |
| Random, 10 per commit | lowest free page | 1.70× | 23 | 23 |
| | runs | 4.16× | 1.8 | 23 |
| | runs, moving pages | 2.87× | 1.9 | 29 |

Taking the lowest free page sends a request per page, and the random keys' load
sets its file size. Moving pages costs page writes where updates are small and
spread out, but takes fewer requests there, since the emptied windows make longer
runs; the benchmark's load barely changes. Big commits gain little: the windows a
commit fills still hold over a quarter of their pages after the next one, too many
to move, and the commit after replaces most of those anyway. On the benchmark VM,
writing in runs cut the load's requests from 169,500 of 4 KiB to 4,540 averaging
158 KiB, and made it three times as fast.

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
  (4 KiB-aligned) up to its budget, counted in pages, and recycles them after that.
  Internal nodes, cached leaves and dirty leaves all live in frames. A cached value
  leaf takes a large frame, of all its pages, which the cache allocates and frees
  (§5).
- **Internal nodes** are always resident. With prefixes stripped, an internal page
  holds several hundred children, so internal nodes are well under 1% of the data.
- **Leaves** live in the cache, keyed by page id in a table sized at open, with CLOCK
  eviction. Only clean, unpinned leaves can be evicted: a write pins the leaves it
  changes and the ones it builds until it publishes, and a reader pins the leaf it
  is on. Exposing cache numbers is left to a later metrics design. Value leaves
  take at most a quarter of the budget, and have a clock hand of their own (§5).
- **Writes larger than the cache** leave the clock hand nothing to evict, and the
  pool grows past the budget (§7). Once a sweep has found nothing while a write
  holds leaves, the hand looks at no more than 64 frames for each frame taken, until
  the write publishes or is discarded: then it sweeps the whole pool again. It
  still finds the leaves that readers let go of meanwhile, a step at a time.
  Sweeping the whole pool twice for every frame made such a write quadratic in its
  leaves. In the ignored test `writes_larger_than_the_cache`, a write of 1M random
  updates to a store of 1M keys, through a cache of 16 MiB, looked at 664M frames
  and took 2.50 s, against 0.33 s with a cache that holds everything. Now it looks
  at 1.4M and takes 0.34 s, at the median of 6 runs. Either way, the pool grows to
  26K frames: the 4K leaves the write changes, and the 22K it builds. With no write
  in progress, every sweep looks at the whole pool: one finds nothing then only when
  readers and internal nodes fill the cache, and stepping until the next write would
  let the pool grow while there are leaves to evict.
- **Misses** read the page straight into a free frame (direct I/O) and verify the
  checksum. Concurrent misses on the same page share one read, and iterators read
  ahead (below). A frame left behind by a cancelled read is reclaimed by the clock
  hand once the read completes.
- **Page reuse.** A page dropped by a write becomes reusable once that write is
  published and no reader that started before the publish is still alive (such a
  reader might still need to read it). Iterators always register as readers;
  point reads register only when they have to wait for I/O. Reclaiming a node's
  pages evicts any stale cache entry for it, under its page id. Only writes
  reclaim, a slice at a time, as they merge (§6): once published, a write reclaims
  what it can, and returns when it is done. A reader that ends can't pause, so it
  reclaims nothing: what it held back waits for the next write, which reclaims it
  before it merges, and its leaves may stay cached until then, as clean leaves the
  cache can evict. Only a merge takes pages, so it takes the ones it would have if
  the reader had reclaimed them as it ended, and readers that end while it pauses
  don't change them. Leaving them to the end of the next commit instead would keep
  them from its merge, which would take others, growing the file. A write dropped
  while it reclaims leaves the rest to the next one too.
- **Capacity.** On BookKeeper-shaped data, dense leaves take ~5 B/entry, so a 1 GiB
  cache holds roughly 200M entries. Slotted leaves, at ~26 B/entry, would hold
  roughly 40M.

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

Leaves are scattered on disk (§2), and direct I/O leaves no OS read-ahead anyway, so
iterators read ahead themselves. Internal nodes are resident, so an iterator knows
the pages of the leaves it will visit. When the leaf it moves to isn't cached, it
waits for that leaf's read and meanwhile starts reads of the following leaves that
aren't cached either, so that a cold scan keeps many reads in flight instead of one.
Each such miss tops the window up again. The window spans `IO_WINDOW` (16) leaves
past the current one, which take at most a quarter of the cache budget, a value leaf
all of its pages. A value leaf too large to cache isn't read ahead. The window ends
at the last leaf, and at the end of the range: the internal key above a leaf is its
smallest key, so the iterator knows which leaves the range reaches without reading
them.

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
- **Read-only views.** A `ReadOnlyDb` opens the file read-only and takes no lock, so
  it can read a store that a `Db` owns, in this process or another. It reads the
  version that was current when it opened, with a cache of its own. The owner
  doesn't know about it, so it reuses that version's pages from its second write
  after the view opens, and may cut them off the end of the file (§7). Every leaf
  read checks that the page holds a leaf no newer than the current generation, so a
  reused page is never taken for the one the tree points to. When a view's page
  fails its checks, or lies past the end of the file, the view rereads both meta
  slots: if the owner has committed since, the error is `Error::Stale`, and a new
  view reads the current version; otherwise it is the error the read met. A corrupt
  page reported as stale is reported as corrupt by the next view, if that view still
  uses it. Opening a view checks the internal nodes it loads the same way. Views
  suit inspection and tooling: one that should keep up with a busy owner has to be
  reopened, since nothing holds back the owner's page reuse for it.
- All file I/O goes through compio 0.19: `read_at` / `write_at` on `&File`, with
  frames as the owned buffers, `sync_data`, and `set_len` to cut the file. A commit
  writes each run of pages with `write_vectored_at`, whose buffers are the frames
  themselves, so nothing is copied; a run is at most 1,024 frames (`IOV_MAX`).
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
- A large write normalizes and plans its batch, pins the leaves it changes, and
  merges it, a slice at a time, and yields to the executor in between (§6).
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
    /// on disk and visible to reads. Writes that queue up behind a commit share the
    /// next one (§6). Fails with `Error::KeyTooLarge`, applying nothing, if a key
    /// is too long (§5).
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
    /// Point-in-time iterator. Creating and advancing it never allocates. The bounds
    /// are byte slices, in any range form: `&a[..]..&b[..]`, `..`, a pair of `Bound`s.
    pub fn range<'k>(&self, range: impl RangeBounds<&'k [u8]>) -> Iter;
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
    pub fn range<'k>(&self, range: impl RangeBounds<&'k [u8]>) -> Iter;
    pub fn page_size(&self) -> usize;
    /// Calls `f` on each node of the tree: the internal nodes, then the leaves in
    /// key order, which it reads. The `stats` and `pages` commands use it.
    pub async fn pages(&self, f: impl FnMut(PageInfo)) -> Result<()>;
    pub async fn close(self) -> Result<()>;
}

/// A node: its first page, level, entries (an internal node's are its children),
/// the bytes it uses, whether it is dense or a value leaf, and the pages it takes.
pub struct PageInfo {
    pub page: u64, pub level: u8, pub entries: usize, pub used: usize,
    pub dense: bool, pub value: bool, pub pages: u64,
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
  in progress, which may span several pages; a cut of the file lasts once synced.
  Each failure scenario first runs once to count its I/O operations, then reruns
  once per operation with an error injected there, and once more with a crash. One
  scenario reads leaves back through a tiny cache; another moves pages out of
  mostly free windows and cuts the file; a third writes in groups, so that
  failures hit groups before and during their commits. After each run, a store
  left usable must pass the invariant checker, one left failed must reject writes,
  and reopening must yield the state after the last successful write (or after the
  failed one, or its whole group, when the failure hit a commit). Tests also fail
  each read of an open in turn: that open fails with `Error::Io`, and the next one
  works.
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
  reused once the read finishes. They also hold a write's leaf read while a second
  write queues up: when the first then fails its commit, the second gets
  `Error::Failed`.
- **Leaf cache:** tests drive the clock hands over a pool directly. Making room
  for a value leaf passes no other frame. While a write holds every frame, the hand
  looks a step ahead for each frame taken, still finds a leaf that a reader lets go
  of, and sweeps the whole pool again once the write lets go; with no write in
  progress, every sweep looks at the whole pool. A write larger than a cache of 16
  pages looks at fewer than 64 frames for each frame of the pool, and the invariant
  checker asserts that no write holds leaves once writes are done.
- **Group commit:** the simulated disk can also hold syncs and cuts, so that writes
  queue up behind a commit. Tests check that a group commits once, applying its
  batches in order, a later batch's puts and ranges overriding an earlier one's;
  that a batch with a key too long is rejected alone; that an error before the
  commit fails the whole group and leaves the store as it was, and one during it
  fails the store, leaving all of the group or none after reopening; and that a
  group holds what fits. They drop writes at every stage: while queued or waiting
  for room, which withdraws them; while their group is built, which makes the
  write building it start over, with a leaf it hadn't needed before, or while it
  normalizes the group's batch or its merge pauses; the write building, whose group
  another write takes over, one queued for the next group included, and while it
  normalizes or its merge pauses; while the group commits, which commits them
  anyway; and the write committing, while it syncs or cuts the file. The model-based property tests also write groups, dropping some of
  their writes, and a test on a real file checks that eight concurrent writes take
  two commits.
- **Pausing writes:** the model-based property tests also run with writes that
  pause after every 1 to 100 units of work rather than after a millisecond, with a
  tiny cache too, scanning the store while a write pauses, then finishing or
  dropping it. Normalizing that pauses after any number of units gives the same
  operations and ranges, down to which of a key's operations is kept, as sorting
  the batch at once did: for random batches, groups some of whose writes left, and
  sorted batches merged out of order. The merge sort orders runs of any length,
  extended to any length, by key, then by operation, and its comparison of keys as
  16-byte numbers agrees with comparing their bytes. Planning that pauses lists the
  same leaves as planning at once, in trees of four levels. Two stores, one of whose
  merges pause, build the same trees on the same pages. A write dropped at each of
  its pauses in turn, in a tree of four levels, leaves the store as it was, every
  page and frame accounted for; so does one dropped while it normalizes, plans,
  or pins the leaves it changes, all of them cached, and its batch, reused, then
  writes what it holds. On a compio runtime, another task's lookups run while a
  write merges, and see the last version.
- **Damage:** a flipped bit in a leaf fails the reads that need it, and one in an
  internal node fails the open, both with `Error::Corruption`, as does a child
  reference past the end of the file or to a page already used. A flipped bit in
  meta slot 0 leaves slot 1, which the open finds where each page size would put
  it. A file of zeros, or with meta pages built from bad fields, isn't a store
  (`Error::InvalidFile`).
- **Read-only views:** a view on the owner's simulated disk returns exactly its
  version while the owner rewrites every leaf, until its reads fail with
  `Error::Stale`, and never with other entries; so does a read of a page the owner
  cut off the end of the file. Damage in a store the owner hasn't written since is
  still `Error::Corruption`. The simulated disk can also take another disk's
  contents at a given operation, as if the owner wrote meanwhile: a view whose
  internal nodes are overwritten that way while it opens fails with `Error::Stale`.
- **Real files** cover what the simulated disk doesn't model: reopening, the lock
  that keeps out a second open, syncing the directory of a new store, including
  a directory that can't be synced, and a read-only view, which opens beside the
  owner, creates nothing, and opens a file it may only read. A write dropped while
  its page write is in flight leaves the store failed until it reopens.
- **Allocation tests:** a counting global allocator asserts that `get` and iteration
  over cached pages, `get` and iteration that miss the cache (reading ahead
  included), and `write` with a reused batch, alone, pausing as it normalizes, plans
  and merges, with ranges and keys in no order, in a group, or freeing the pages a
  reader held back, all on the simulated disk, perform no allocations. On a real
  disk, compio adds one per I/O operation.
- **Sanitizers:** CI runs every test under AddressSanitizer, std included, so a
  use of freed memory or a leak fails the run even where the test's own checks
  pass, real files' io_uring path included. One test drops a store while the
  simulated disk still holds its reads, which checks that their buffers keep the
  frame memory alive until the reads finish.
- **Benchmarks** against RocksDB on the BookKeeper workload: sorted batches over
  *A* active ledgers, random point reads, floor lookups, and `delete_range` of old
  ledgers. On the simulated disk, an ignored test measures the file's size and the
  write requests per commit of that load and of random updates (§7), and another
  writes larger than the cache (§8).

## 12. Milestones

1. **M1, in-memory tree:** the slotted node format with in-place search, frames,
   versions, batch writes applied in memory, get with every comparison, range,
   delete_range, property and allocation tests. No I/O.
2. **M2, persistence and caching:** page I/O on compio with direct I/O, the commit
   at the end of `write()`, open and recovery, free bitmap, leaf cache with budget
   and eviction, snapshot-aware page reuse, failure-path tests.
3. **M3, performance:** group commit (writes that queue up during a commit share
   the next one, done, §6), the dense node format (done, §5), yielding inside
   large merges (done, §6), writing large batches' new leaves early
   ([#25](https://github.com/merlimat/encina/issues/25)), one shared read for
   concurrent misses and reading ahead in iterators (done, §8), coverage measured
   in CI (done, §11), and benchmarks against RocksDB (done, §11).
4. **M4, large values:** value leaves (§5), with values written and read whole,
   which the cache keeps and iterators read ahead (done).

## Non-goals (for now)

- Access to one instance from multiple threads.
- An internal WAL.
- Transactions beyond atomic batches.
- Keys larger than a quarter of a page: each leaf's smallest key is also in its
  parent (§5). The limit is part of the documented public API.
- Values that don't fit in memory: value leaves are written and read whole.
- Custom comparators: keys compare as bytes.
