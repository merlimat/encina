//! Times opening a store of 32-byte keys and 100-byte values, whose open used to
//! read all of level 1 (DESIGN.md §7). Pointing `encina` in Cargo.toml at another
//! checkout times that one's open.
//!
//! - `build PATH ENTRIES` writes ENTRIES entries, in sorted batches of 1M.
//! - `open PATH ENTRIES [CACHE_MB]` opens the store with a cache of CACHE_MB MiB,
//!   64 by default, and times it, then 200 lookups of random keys right after.
//!   Then it waits until the process has read nothing for half a second: the scan
//!   of level 1 has ended by then. It counts the bytes read from /proc/self/io, so
//!   on Linux only.

use std::env;
use std::fs;
use std::time::{Duration, Instant};

use encina::{Comparison, Db, Files, Options, WriteBatch};

const KEY: usize = 32;
const VALUE: usize = 100;

fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn fill(out: &mut [u8], seed: u64) {
    for (j, chunk) in out.chunks_mut(8).enumerate() {
        let bytes = mix(seed ^ (j as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)).to_le_bytes();
        chunk.copy_from_slice(&bytes[..chunk.len()]);
    }
}

fn key(i: u64, out: &mut [u8; KEY]) {
    out[..8].copy_from_slice(&i.to_be_bytes());
    fill(&mut out[8..], i);
}

fn value(i: u64, out: &mut [u8; VALUE]) {
    fill(out, !i);
}

/// The bytes this process has read from storage, 0 where /proc doesn't say.
fn read_bytes() -> u64 {
    let Ok(io) = fs::read_to_string("/proc/self/io") else {
        return 0;
    };
    io.lines()
        .find_map(|l| l.strip_prefix("read_bytes: "))
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1 << 20) as f64
}

#[compio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    let path = &args[2];
    let entries: u64 = args[3].parse().unwrap();
    match args[1].as_str() {
        "build" => build(path, entries).await,
        "open" => {
            let cache_mb = args.get(4).map_or(64, |s| s.parse().unwrap());
            open(path, entries, cache_mb).await
        }
        other => panic!("unknown command {other}"),
    }
}

async fn build(path: &str, entries: u64) {
    let db = Db::open(&Files, path, Options::default()).await.unwrap();
    let mut batch = WriteBatch::new();
    let (mut k, mut v) = ([0; KEY], [0; VALUE]);
    let start = Instant::now();
    for i in 0..entries {
        key(i, &mut k);
        value(i, &mut v);
        batch.put(&k, &v);
        if i % 1_000_000 == 999_999 || i + 1 == entries {
            db.write(&mut batch).await.unwrap();
            batch.clear();
        }
    }
    db.close().await.unwrap();
    println!(
        "built {entries} entries in {:.1} s",
        start.elapsed().as_secs_f64()
    );
}

async fn open(path: &str, entries: u64, cache_mb: usize) {
    let opts = Options::default().cache_size(cache_mb << 20);
    let io = read_bytes();
    let start = Instant::now();
    let db = Db::open(&Files, path, opts).await.unwrap();
    let opened = start.elapsed();
    let at_open = read_bytes() - io;
    // Cold lookups of random keys, right after the open.
    let (mut k, mut latencies) = ([0; KEY], Vec::new());
    for n in 0..200 {
        key(mix(n + 12345) % entries, &mut k);
        let t = Instant::now();
        let found = db.get(&k, Comparison::Equal, |_, v| v.len()).await.unwrap();
        latencies.push(t.elapsed());
        assert_eq!(found, Some(VALUE));
    }
    latencies.sort();
    // Wait until the reads stop for half a second.
    let mut last = read_bytes();
    let mut changed = Instant::now();
    loop {
        compio::time::sleep(Duration::from_millis(1)).await;
        let now = read_bytes();
        if now != last {
            (last, changed) = (now, Instant::now());
        } else if changed.elapsed() > Duration::from_millis(500) {
            break;
        }
    }
    println!(
        "open {:8.1} ms {:7.1} MiB | lookups p50 {:6.3} ms max {:6.3} ms | reads end {:8.1} ms {:7.1} MiB",
        ms(opened),
        mib(at_open),
        ms(latencies[latencies.len() / 2]),
        ms(*latencies.last().unwrap()),
        ms(changed - start),
        mib(last - io),
    );
    db.close().await.unwrap();
}
