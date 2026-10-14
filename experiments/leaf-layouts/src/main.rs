//! Leaf-layout micro-benchmark for encina: fixed 16 -> 16 layout vs generic
//! byte-slice layouts, on BookKeeper-shaped keys (ledgerId BE || entryId BE).
//!
//! Every layout packs sorted entries into leaves that fit a 4 KiB page, the
//! leaves are routed by an identical top-level array of min keys, and the
//! benchmark measures random point lookups of existing keys.

use std::hint::black_box;
use std::time::Instant;

const PAGE: usize = 4096;
const HDR: usize = 32;
const CAP: usize = PAGE - HDR;

type Key = [u8; 16];
type Val = [u8; 16];

fn key(ledger: u64, entry: u64) -> Key {
    let mut k = [0u8; 16];
    k[..8].copy_from_slice(&ledger.to_be_bytes());
    k[8..].copy_from_slice(&entry.to_be_bytes());
    k
}

fn ledger_id(i: u64) -> u64 {
    1_000 + i * 3
}

fn gen_data(ledgers: u64, per_ledger: u64) -> Vec<(Key, Val)> {
    let mut out = Vec::with_capacity((ledgers * per_ledger) as usize);
    let mut offset = 0u64;
    for l in 0..ledgers {
        for e in 0..per_ledger {
            let mut v = [0u8; 16];
            v[..8].copy_from_slice(&(l / 16).to_be_bytes()); // entry log id
            v[8..].copy_from_slice(&offset.to_be_bytes()); // offset in log
            offset += 1_000 + (e % 7) * 100;
            out.push((key(ledger_id(l), e), v));
        }
    }
    out
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn gen_queries(ledgers: u64, per_ledger: u64, n: usize, seed: u64) -> Vec<Key> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| key(ledger_id(rng.next() % ledgers), rng.next() % per_ledger))
        .collect()
}

/// Branch-free lower bound: the first index `i` in `0..n` for which `less(i)` is false.
#[inline(always)]
fn lower_bound(n: usize, mut less: impl FnMut(usize) -> bool) -> usize {
    if n == 0 {
        return 0;
    }
    let (mut base, mut size) = (0usize, n);
    while size > 1 {
        let half = size / 2;
        if less(base + half) {
            base += half;
        }
        size -= half;
    }
    base + less(base) as usize
}

fn lcp(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[inline(always)]
fn u16_at(buf: &[u8], at: usize) -> usize {
    u16::from_le_bytes(unsafe { *(buf.as_ptr().add(at) as *const [u8; 2]) }) as usize
}

trait Leaf: Sized {
    const NAME: &'static str;
    fn encoded_size(entries: &[(Key, Val)]) -> usize;
    fn build(entries: &[(Key, Val)]) -> Self;
    fn get(&self, k: &Key) -> Option<Val>;
    fn mem_bytes(&self) -> usize;
}

/// Specialized layout: keys as native u128 (one compare per probe), values as
/// 16 bytes. One allocation: [keys; n][values; n].
struct Fixed {
    n: usize,
    data: Box<[u128]>,
}

impl Leaf for Fixed {
    const NAME: &'static str = "fixed u128 (specialized)";

    fn encoded_size(e: &[(Key, Val)]) -> usize {
        e.len() * 32
    }

    fn build(e: &[(Key, Val)]) -> Self {
        let mut data = Vec::with_capacity(e.len() * 2);
        data.extend(e.iter().map(|(k, _)| u128::from_be_bytes(*k)));
        data.extend(e.iter().map(|(_, v)| u128::from_ne_bytes(*v)));
        Fixed { n: e.len(), data: data.into_boxed_slice() }
    }

    #[inline(always)]
    fn get(&self, k: &Key) -> Option<Val> {
        let t = u128::from_be_bytes(*k);
        let d = &self.data;
        let i = lower_bound(self.n, |i| unsafe { *d.get_unchecked(i) } < t);
        (i < self.n && d[i] == t).then(|| d[self.n + i].to_ne_bytes())
    }

    fn mem_bytes(&self) -> usize {
        self.data.len() * 16
    }
}

/// Generic API, uniform-size node: every entry in the node has the same key and
/// value length, so keys sit at a fixed stride. Keys compared as big-endian u128.
struct DenseU128 {
    n: usize,
    data: Box<[u8]>,
}

/// Same storage as `DenseU128`, keys compared with memcmp (slice Ord).
struct DenseMemcmp {
    n: usize,
    data: Box<[u8]>,
}

fn build_dense(e: &[(Key, Val)]) -> (usize, Box<[u8]>) {
    let mut data = Vec::with_capacity(e.len() * 32);
    for (k, _) in e {
        data.extend_from_slice(k);
    }
    for (_, v) in e {
        data.extend_from_slice(v);
    }
    (e.len(), data.into_boxed_slice())
}

#[inline(always)]
fn dense_key(data: &[u8], i: usize) -> &[u8; 16] {
    unsafe { &*(data.as_ptr().add(i * 16) as *const [u8; 16]) }
}

impl Leaf for DenseU128 {
    const NAME: &'static str = "generic, dense node (u128 cmp)";

    fn encoded_size(e: &[(Key, Val)]) -> usize {
        e.len() * 32
    }

    fn build(e: &[(Key, Val)]) -> Self {
        let (n, data) = build_dense(e);
        DenseU128 { n, data }
    }

    #[inline(always)]
    fn get(&self, k: &Key) -> Option<Val> {
        let t = u128::from_be_bytes(*k);
        let d = &self.data;
        let i = lower_bound(self.n, |i| u128::from_be_bytes(*dense_key(d, i)) < t);
        (i < self.n && dense_key(d, i) == k).then(|| *dense_key(d, self.n + i))
    }

    fn mem_bytes(&self) -> usize {
        self.data.len()
    }
}

impl Leaf for DenseMemcmp {
    const NAME: &'static str = "generic, dense node (memcmp)";

    fn encoded_size(e: &[(Key, Val)]) -> usize {
        e.len() * 32
    }

    fn build(e: &[(Key, Val)]) -> Self {
        let (n, data) = build_dense(e);
        DenseMemcmp { n, data }
    }

    #[inline(always)]
    fn get(&self, k: &Key) -> Option<Val> {
        let d = &self.data;
        let i = lower_bound(self.n, |i| dense_key(d, i)[..] < k[..]);
        (i < self.n && dense_key(d, i) == k).then(|| *dense_key(d, self.n + i))
    }

    fn mem_bytes(&self) -> usize {
        self.data.len()
    }
}

/// Generic slotted node: slot array of (offset, key len, value len) followed by
/// a heap of key and value bytes. Keys compared with memcmp through the slot.
struct Slotted {
    n: usize,
    buf: Box<[u8]>,
}

impl Slotted {
    #[inline(always)]
    fn slot(&self, i: usize) -> (usize, usize, usize) {
        let b = &self.buf;
        (u16_at(b, i * 6), u16_at(b, i * 6 + 2), u16_at(b, i * 6 + 4))
    }

    #[inline(always)]
    fn key(&self, i: usize) -> &[u8] {
        let (off, klen, _) = self.slot(i);
        unsafe { self.buf.get_unchecked(off..off + klen) }
    }
}

impl Leaf for Slotted {
    const NAME: &'static str = "generic, slotted (memcmp)";

    fn encoded_size(e: &[(Key, Val)]) -> usize {
        e.len() * (6 + 32)
    }

    fn build(e: &[(Key, Val)]) -> Self {
        let n = e.len();
        let heap_at = n * 6;
        let mut buf = vec![0u8; heap_at + n * 32];
        for (i, (k, v)) in e.iter().enumerate() {
            let off = heap_at + i * 32;
            buf[i * 6..i * 6 + 2].copy_from_slice(&(off as u16).to_le_bytes());
            buf[i * 6 + 2..i * 6 + 4].copy_from_slice(&16u16.to_le_bytes());
            buf[i * 6 + 4..i * 6 + 6].copy_from_slice(&16u16.to_le_bytes());
            buf[off..off + 16].copy_from_slice(k);
            buf[off + 16..off + 32].copy_from_slice(v);
        }
        Slotted { n, buf: buf.into_boxed_slice() }
    }

    #[inline(always)]
    fn get(&self, k: &Key) -> Option<Val> {
        let i = lower_bound(self.n, |i| self.key(i) < &k[..]);
        if i < self.n && self.key(i) == k {
            let (off, klen, _) = self.slot(i);
            Some(self.buf[off + klen..off + klen + 16].try_into().unwrap())
        } else {
            None
        }
    }

    fn mem_bytes(&self) -> usize {
        self.buf.len()
    }
}

/// Generic slotted node with prefix truncation: the common prefix of the node's
/// keys is stored once, each slot keeps the first 4 bytes of the key suffix as a
/// big-endian u32 "head" (heads are contiguous, so the binary search touches only
/// them), and only suffix bytes beyond the head go to the heap.
///
/// Layout: [prefix][pad][heads; n x u32][slots; n x (off, suffix len, value len)][heap]
struct Prefix {
    n: usize,
    plen: usize,
    heads_at: usize,
    slots_at: usize,
    buf: Box<[u8]>,
}

fn head(suffix: &[u8]) -> u32 {
    let mut h = [0u8; 4];
    let m = suffix.len().min(4);
    h[..m].copy_from_slice(&suffix[..m]);
    u32::from_be_bytes(h)
}

impl Prefix {
    #[inline(always)]
    fn head_at(&self, i: usize) -> u32 {
        u32::from_ne_bytes(unsafe { *(self.buf.as_ptr().add(self.heads_at + i * 4) as *const [u8; 4]) })
    }

    #[inline(always)]
    fn slot(&self, i: usize) -> (usize, usize, usize) {
        let at = self.slots_at + i * 6;
        (u16_at(&self.buf, at), u16_at(&self.buf, at + 2), u16_at(&self.buf, at + 4))
    }

    /// Suffix bytes beyond the head, and the full suffix length.
    #[inline(always)]
    fn rest(&self, i: usize) -> (&[u8], usize) {
        let (off, slen, _) = self.slot(i);
        let rlen = slen.saturating_sub(4);
        (unsafe { self.buf.get_unchecked(off..off + rlen) }, slen)
    }
}

impl Leaf for Prefix {
    const NAME: &'static str = "generic, slotted + prefix + head";

    fn encoded_size(e: &[(Key, Val)]) -> usize {
        let p = lcp(&e[0].0, &e[e.len() - 1].0);
        p + e.len() * (4 + 6 + 16 + (16 - p).saturating_sub(4))
    }

    fn build(e: &[(Key, Val)]) -> Self {
        let n = e.len();
        let p = lcp(&e[0].0, &e[n - 1].0);
        let rlen = (16 - p).saturating_sub(4);
        let heads_at = (p + 3) & !3;
        let slots_at = heads_at + n * 4;
        let heap_at = slots_at + n * 6;
        let mut buf = vec![0u8; heap_at + n * (rlen + 16)];
        buf[..p].copy_from_slice(&e[0].0[..p]);
        for (i, (k, v)) in e.iter().enumerate() {
            let suffix = &k[p..];
            buf[heads_at + i * 4..heads_at + i * 4 + 4].copy_from_slice(&head(suffix).to_ne_bytes());
            let off = heap_at + i * (rlen + 16);
            let s = slots_at + i * 6;
            buf[s..s + 2].copy_from_slice(&(off as u16).to_le_bytes());
            buf[s + 2..s + 4].copy_from_slice(&(suffix.len() as u16).to_le_bytes());
            buf[s + 4..s + 6].copy_from_slice(&16u16.to_le_bytes());
            buf[off..off + rlen].copy_from_slice(&suffix[suffix.len().min(4)..]);
            buf[off + rlen..off + rlen + 16].copy_from_slice(v);
        }
        Prefix { n, plen: p, heads_at, slots_at, buf: buf.into_boxed_slice() }
    }

    #[inline(always)]
    fn get(&self, k: &Key) -> Option<Val> {
        let p = self.plen;
        if k[..p] != self.buf[..p] {
            return None;
        }
        let suffix = &k[p..];
        let th = head(suffix);
        let target = (&suffix[suffix.len().min(4)..], suffix.len());
        let i = lower_bound(self.n, |i| {
            let h = self.head_at(i);
            if h != th { h < th } else { self.rest(i) < target }
        });
        if i < self.n && self.head_at(i) == th && self.rest(i) == target {
            let (off, slen, _) = self.slot(i);
            let at = off + slen.saturating_sub(4);
            Some(self.buf[at..at + 16].try_into().unwrap())
        } else {
            None
        }
    }

    fn mem_bytes(&self) -> usize {
        self.buf.len()
    }
}

/// Same node format as `Prefix`, with a tuned search: a branch-free binary search
/// over the u32 heads alone, then a linear scan over the (rare) run of equal heads.
/// The prefix is compared with inline 8-byte loads instead of a memcmp call.
struct PrefixFast(Prefix);

#[inline(always)]
fn prefix_eq(a: &[u8], b: &[u8]) -> bool {
    let n = b.len();
    let (mut i, mut eq) = (0, true);
    while i + 8 <= n {
        let x = u64::from_ne_bytes(unsafe { *(a.as_ptr().add(i) as *const [u8; 8]) });
        let y = u64::from_ne_bytes(unsafe { *(b.as_ptr().add(i) as *const [u8; 8]) });
        eq &= x == y;
        i += 8;
    }
    while i < n {
        eq &= a[i] == b[i];
        i += 1;
    }
    eq
}

#[inline(always)]
fn head_fast(suffix: &[u8]) -> u32 {
    if suffix.len() >= 4 {
        u32::from_be_bytes(suffix[..4].try_into().unwrap())
    } else {
        head(suffix)
    }
}

impl Leaf for PrefixFast {
    const NAME: &'static str = "generic, slotted + prefix + head, tuned";

    fn encoded_size(e: &[(Key, Val)]) -> usize {
        Prefix::encoded_size(e)
    }

    fn build(e: &[(Key, Val)]) -> Self {
        PrefixFast(Prefix::build(e))
    }

    #[inline(always)]
    fn get(&self, k: &Key) -> Option<Val> {
        let s = &self.0;
        let p = s.plen;
        if !prefix_eq(k, &s.buf[..p]) {
            return None;
        }
        let suffix = &k[p..];
        let th = head_fast(suffix);
        let mut i = lower_bound(s.n, |i| s.head_at(i) < th);
        let target = (&suffix[suffix.len().min(4)..], suffix.len());
        while i < s.n && s.head_at(i) == th {
            match s.rest(i).cmp(&target) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Equal => {
                    let (off, slen, _) = s.slot(i);
                    let at = off + slen.saturating_sub(4);
                    return Some(s.buf[at..at + 16].try_into().unwrap());
                }
                std::cmp::Ordering::Greater => return None,
            }
        }
        None
    }

    fn mem_bytes(&self) -> usize {
        self.0.mem_bytes()
    }
}

/// Generic API, uniform-size node with prefix truncation: the keys' common prefix
/// is stored once, and each key keeps only its suffix at a fixed stride, compared
/// as a big-endian integer. With `VALUES`, the values' common prefix is stripped
/// the same way and a value is rebuilt (prefix + suffix) on read; otherwise values
/// stay 16 bytes and could be returned without copying.
///
/// Layout: [key prefix][value prefix][key suffixes; n x stride][values; n x vstride][8 pad]
struct DensePrefix<const VALUES: bool> {
    n: usize,
    plen: usize,
    stride: usize,
    vplen: usize,
    vstride: usize,
    keys_at: usize,
    vals_at: usize,
    buf: Box<[u8]>,
}

fn value_prefix<const VALUES: bool>(e: &[(Key, Val)]) -> usize {
    if VALUES { e.iter().map(|(_, v)| lcp(v, &e[0].1)).min().unwrap() } else { 0 }
}

impl<const VALUES: bool> DensePrefix<VALUES> {
    /// The `stride`-byte key suffix of entry `i` as a big-endian integer (stride 1..=8).
    #[inline(always)]
    fn key64(&self, i: usize) -> u64 {
        let x = u64::from_be_bytes(unsafe { *(self.buf.as_ptr().add(self.keys_at + i * self.stride) as *const [u8; 8]) });
        x >> (64 - 8 * self.stride)
    }

    /// The `stride`-byte key suffix of entry `i` as a big-endian integer (stride 9..=16).
    #[inline(always)]
    fn key128(&self, i: usize) -> u128 {
        let x = u128::from_be_bytes(unsafe { *(self.buf.as_ptr().add(self.keys_at + i * self.stride) as *const [u8; 16]) });
        x >> (128 - 8 * self.stride)
    }

    #[inline(always)]
    fn value(&self, i: usize) -> Val {
        let mut v = [0u8; 16];
        let at = self.plen;
        v[..self.vplen].copy_from_slice(&self.buf[at..at + self.vplen]);
        let s = self.vals_at + i * self.vstride;
        v[self.vplen..].copy_from_slice(&self.buf[s..s + self.vstride]);
        v
    }
}

impl<const VALUES: bool> Leaf for DensePrefix<VALUES> {
    const NAME: &'static str = if VALUES {
        "generic, dense + key & value prefix"
    } else {
        "generic, dense + key prefix"
    };

    fn encoded_size(e: &[(Key, Val)]) -> usize {
        let p = lcp(&e[0].0, &e[e.len() - 1].0);
        let vp = value_prefix::<VALUES>(e);
        p + vp + e.len() * (16 - p + 16 - vp)
    }

    fn build(e: &[(Key, Val)]) -> Self {
        let n = e.len();
        let p = lcp(&e[0].0, &e[n - 1].0);
        let vp = value_prefix::<VALUES>(e);
        let (stride, vstride) = (16 - p, 16 - vp);
        let keys_at = p + vp;
        let vals_at = keys_at + n * stride;
        let mut buf = vec![0u8; vals_at + n * vstride + 16];
        buf[..p].copy_from_slice(&e[0].0[..p]);
        buf[p..p + vp].copy_from_slice(&e[0].1[..vp]);
        for (i, (k, v)) in e.iter().enumerate() {
            buf[keys_at + i * stride..keys_at + (i + 1) * stride].copy_from_slice(&k[p..]);
            buf[vals_at + i * vstride..vals_at + (i + 1) * vstride].copy_from_slice(&v[vp..]);
        }
        DensePrefix { n, plen: p, stride, vplen: vp, vstride, keys_at, vals_at, buf: buf.into_boxed_slice() }
    }

    #[inline(always)]
    fn get(&self, k: &Key) -> Option<Val> {
        let p = self.plen;
        if !prefix_eq(k, &self.buf[..p]) {
            return None;
        }
        let s = self.stride;
        if s == 0 {
            return Some(self.value(0)); // single-entry node: the prefix is the whole key
        }
        let i = if s <= 8 {
            let mut t = [0u8; 8];
            t[..s].copy_from_slice(&k[p..]);
            let t = u64::from_be_bytes(t) >> (64 - 8 * s);
            let i = lower_bound(self.n, |i| self.key64(i) < t);
            (i < self.n && self.key64(i) == t).then_some(i)
        } else {
            let mut t = [0u8; 16];
            t[..s].copy_from_slice(&k[p..]);
            let t = u128::from_be_bytes(t) >> (128 - 8 * s);
            let i = lower_bound(self.n, |i| self.key128(i) < t);
            (i < self.n && self.key128(i) == t).then_some(i)
        };
        i.map(|i| self.value(i))
    }

    fn mem_bytes(&self) -> usize {
        self.buf.len() - 16
    }
}

/// Packs `data` into leaves of layout `L`, then runs the queries three times and
/// keeps the best run. With `ALLOC`, each hit is also copied into a `Vec<u8>`, to
/// show the cost of an API that returns owned values.
fn bench<L: Leaf, const ALLOC: bool>(data: &[(Key, Val)], queries: &[Key], baseline: &mut Option<f64>) {
    let t0 = Instant::now();
    let mut mins = Vec::new();
    let mut leaves = Vec::new();
    let mut start = 0;
    while start < data.len() {
        // Largest `end` whose entries fit a page; size only grows with `end`, so
        // search exponentially, then binary-search the last step.
        let fits = |end: usize| L::encoded_size(&data[start..end]) <= CAP;
        let (mut good, mut step) = (start + 1, 64);
        let mut bad = None;
        loop {
            let cand = (good + step).min(data.len());
            if cand == good {
                break;
            }
            if fits(cand) {
                good = cand;
                step *= 2;
            } else {
                bad = Some(cand);
                break;
            }
        }
        if let Some(mut bad) = bad {
            while bad - good > 1 {
                let mid = (good + bad) / 2;
                if fits(mid) { good = mid } else { bad = mid }
            }
        }
        mins.push(u128::from_be_bytes(data[start].0));
        leaves.push(L::build(&data[start..good]));
        start = good;
    }
    let build_ns = t0.elapsed().as_nanos() as f64 / data.len() as f64;
    let mem: usize = leaves.iter().map(L::mem_bytes).sum();

    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        let mut sum = 0u64;
        let mut hits = 0usize;
        for q in queries {
            let qk = u128::from_be_bytes(*q);
            let li = lower_bound(mins.len(), |i| unsafe { *mins.get_unchecked(i) } <= qk) - 1;
            let leaf = unsafe { leaves.get_unchecked(li) };
            if ALLOC {
                if let Some(v) = leaf.get(black_box(q)).map(|v| v.to_vec()) {
                    sum = sum.wrapping_add(v[15] as u64);
                    hits += 1;
                    black_box(v);
                }
            } else if let Some(v) = leaf.get(black_box(q)) {
                sum = sum.wrapping_add(v[15] as u64);
                hits += 1;
            }
        }
        black_box(sum);
        assert_eq!(hits, queries.len(), "{}: lookups missed", L::NAME);
        best = best.min(t.elapsed().as_nanos() as f64 / queries.len() as f64);
    }
    let rel = match baseline {
        None => {
            *baseline = Some(best);
            1.0
        }
        Some(b) => best / *b,
    };
    let name = format!("{}{}", L::NAME, if ALLOC { " + Vec return" } else { "" });
    println!(
        "{:<46} {:>8.1} {:>8.1} {:>9.1} {:>8.1} {:>7.2}x",
        name,
        data.len() as f64 / leaves.len() as f64,
        mem as f64 / data.len() as f64,
        build_ns,
        best,
        rel
    );
}

fn main() {
    let sets = [
        (100u64, 1_000u64, "100K entries: fits in L2, CPU-bound"),
        (2_000, 10_000, "20M entries: DRAM-bound"),
    ];
    for (ledgers, per_ledger, label) in sets {
        let data = gen_data(ledgers, per_ledger);
        let queries = gen_queries(ledgers, per_ledger, 5_000_000, 42);
        println!("\n{label} ({ledgers} ledgers x {per_ledger} entries, 5M random hits)");
        println!(
            "{:<46} {:>8} {:>8} {:>9} {:>8} {:>8}",
            "layout", "ent/leaf", "B/entry", "build ns", "get ns", "vs fixed"
        );
        let mut base = None;
        bench::<Fixed, false>(&data, &queries, &mut base);
        bench::<DenseU128, false>(&data, &queries, &mut base);
        bench::<DenseMemcmp, false>(&data, &queries, &mut base);
        bench::<DensePrefix<false>, false>(&data, &queries, &mut base);
        bench::<DensePrefix<true>, false>(&data, &queries, &mut base);
        bench::<Slotted, false>(&data, &queries, &mut base);
        bench::<Prefix, false>(&data, &queries, &mut base);
        bench::<PrefixFast, false>(&data, &queries, &mut base);
        bench::<Prefix, true>(&data, &queries, &mut base);
    }
}
