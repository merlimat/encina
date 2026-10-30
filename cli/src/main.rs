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

//! `encina`: puts, gets, scans and deletes the entries of a store from the command
//! line.

use std::error::Error;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::ops::Bound;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use encina::{Comparison, Db, DefaultEnv, Options, PageInfo, ReadOnlyDb, WriteBatch};

/// How many times in a row a read opens a new view of the store, after finding
/// that the store's owner overwrote the version its view reads.
const MAX_STALE: u32 = 10;

/// The meta pages at the start of a store's file.
const META_PAGES: usize = 2;

const AFTER_HELP: &str = r"get, scan, stats and pages open the store read-only, so they work while another
process has it open for writing. If that process overwrites the version they read,
they open the store again: scan resumes after the last key it printed, and the
others start over on the current version.

Keys and values are bytes: in an argument, \xNN stands for the byte with hex value
NN, and \\ for a backslash. Output prints UTF-8 text as is, and escapes backslashes,
control characters and bytes that aren't UTF-8 the same way.

Exit status: 0 on success, 1 if get finds no such key, 2 on errors.";

/// Puts, gets, scans and deletes the entries of an encina store.
#[derive(Parser)]
#[command(version, after_help = AFTER_HELP)]
struct Cli {
    /// The store's file
    #[arg(short = 'f', long = "file", value_name = "PATH")]
    path: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Writes a value under a key, creating the store if needed
    Put {
        #[arg(value_parser = unescape)]
        key: Box<[u8]>,
        #[arg(value_parser = unescape)]
        value: Box<[u8]>,
    },
    /// Removes a key, if present
    Delete {
        #[arg(value_parser = unescape)]
        key: Box<[u8]>,
    },
    /// Removes the entries in a range of keys
    DeleteRange {
        /// The first key in the range; "" for the first key in the store
        #[arg(short = 's', long, value_name = "KEY", value_parser = unescape)]
        start: Box<[u8]>,
        /// The key the range stops before
        #[arg(short = 'e', long, value_name = "KEY", value_parser = unescape)]
        end: Box<[u8]>,
    },
    /// Prints the value under a key
    Get {
        #[arg(value_parser = unescape)]
        key: Box<[u8]>,
        /// Prints the value in the layout of `hexdump -C`
        #[arg(short = 'x', long)]
        hexdump: bool,
    },
    /// Prints the entries in a range of keys, one per line as key<TAB>value
    Scan {
        /// The first key in the range; by default, the first key in the store
        #[arg(short = 's', long, value_name = "KEY", value_parser = unescape)]
        start: Option<Box<[u8]>>,
        /// The key the range stops before; by default, the range has no end
        #[arg(short = 'e', long, value_name = "KEY", value_parser = unescape)]
        end: Option<Box<[u8]>>,
        /// Prints keys and values in the layout of `hexdump -C`
        #[arg(short = 'x', long)]
        hexdump: bool,
    },
    /// Prints how many pages the store's file holds, and what they hold
    ///
    /// The summary gives the store's entries, the bytes of their keys and values as
    /// written, and the file's size in bytes, then its pages.
    ///
    /// The table counts the pages at each level of the tree, from the root down to
    /// the leaves, at level 0: the dense nodes and the value leaves among them, their
    /// entries (an internal node's entries are its children), and how full their
    /// pages are. A value leaf holds an entry too large for a leaf, on as many pages
    /// as it needs.
    Stats,
    /// Prints a map of the store's pages, 64 to a line
    ///
    /// Each line starts with the number of its first page. M is a meta page, L a
    /// leaf, D a dense leaf, V a page of a value leaf, a digit an internal node at
    /// that level, and . a free page.
    Pages,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result: Result<ExitCode, Box<dyn Error>> = compio::runtime::Runtime::new()
        .map_err(Box::from)
        .and_then(|runtime| runtime.block_on(run(&cli.path, cli.command)));
    match result {
        Ok(code) => code,
        // The reader went away, as in `encina -f data.db scan | head`: stop quietly.
        Err(e)
            if e.downcast_ref::<io::Error>().map(io::Error::kind)
                == Some(io::ErrorKind::BrokenPipe) =>
        {
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("encina: {e}");
            ExitCode::from(2)
        }
    }
}

async fn run(path: &Path, command: Command) -> Result<ExitCode, Box<dyn Error>> {
    let mut out = BufWriter::new(io::stdout().lock());
    match command {
        Command::Put { key, value } => {
            let mut batch = WriteBatch::new();
            batch.put(&key, &value);
            write(path, &mut batch).await?;
        }
        Command::Delete { key } => {
            let mut batch = WriteBatch::new();
            batch.delete(&key);
            write_existing(path, &mut batch).await?;
        }
        Command::DeleteRange { start, end } => {
            let mut batch = WriteBatch::new();
            batch.delete_range(&start, &end);
            write_existing(path, &mut batch).await?;
        }
        Command::Get { key, hexdump } => {
            let Some(value) = get(path, &key).await.map_err(|e| at(path, e))? else {
                eprintln!("encina: key not found");
                return Ok(ExitCode::FAILURE);
            };
            if hexdump {
                write_hexdump(&mut out, "", &value)?;
            } else {
                write_text(&mut out, &value)?;
                out.write_all(b"\n")?;
            }
        }
        Command::Scan {
            start,
            end,
            hexdump,
        } => {
            let start = start.as_deref().map_or(Bound::Unbounded, Bound::Included);
            let end = end.as_deref().map_or(Bound::Unbounded, Bound::Excluded);
            scan(path, start, end, |key, value| {
                if hexdump {
                    write_hexdump(&mut out, "key   ", key)?;
                    write_hexdump(&mut out, "value ", value)
                } else {
                    write_text(&mut out, key)?;
                    out.write_all(b"\t")?;
                    write_text(&mut out, value)?;
                    out.write_all(b"\n")
                }
            })
            .await?;
        }
        Command::Stats => {
            let Store {
                page_size,
                file_size,
                entry_count,
                entry_size,
                pages,
            } = read_pages(path).await?;
            let nodes: Vec<_> = pages
                .iter()
                .filter_map(|page| match page {
                    Page::Node(info) => Some(info),
                    _ => None,
                })
                .collect();
            let free: Vec<_> = pages
                .chunk_by(|a, b| matches!(a, Page::Free) == matches!(b, Page::Free))
                .filter(|stretch| matches!(stretch[0], Page::Free))
                .map(<[Page]>::len)
                .collect();
            writeln!(out, "entries         {entry_count:>12}")?;
            writeln!(out, "entry size      {entry_size:>12}")?;
            writeln!(out, "page size       {page_size:>12}")?;
            writeln!(out, "file size       {file_size:>12}")?;
            writeln!(out, "file pages      {:>12}", pages.len())?;
            writeln!(out, "meta pages      {META_PAGES:>12}")?;
            writeln!(out, "tree pages      {:>12}", nodes.len())?;
            writeln!(out, "free pages      {:>12}", free.iter().sum::<usize>())?;
            writeln!(out, "free stretches  {:>12}", free.len())?;
            writeln!(
                out,
                "longest stretch {:>12}",
                free.iter().max().unwrap_or(&0)
            )?;
            if let Some(height) = nodes.iter().map(|info| info.level).max() {
                writeln!(
                    out,
                    "\nlevel      pages      dense     values      entries    full"
                )?;
                for level in (0..=height).rev() {
                    let (mut count, mut dense, mut values, mut entries, mut used) = (0, 0, 0, 0, 0);
                    // A value leaf is listed once for each of its pages.
                    for (i, info) in nodes.iter().enumerate() {
                        if info.level != level || (i > 0 && nodes[i - 1].page == info.page) {
                            continue;
                        }
                        count += info.pages as usize;
                        dense += usize::from(info.dense);
                        values += usize::from(info.value);
                        entries += info.entries;
                        used += info.used;
                    }
                    let full = 100.0 * used as f64 / (count * page_size) as f64;
                    writeln!(
                        out,
                        "{level:>5} {count:>10} {dense:>10} {values:>10} {entries:>12} {full:>6.1}%"
                    )?;
                }
            }
        }
        Command::Pages => {
            let Store { pages, .. } = read_pages(path).await?;
            let width = (pages.len() - 1).to_string().len();
            for (row, line) in pages.chunks(64).enumerate() {
                write!(out, "{:>width$} ", row * 64)?;
                for (i, page) in line.iter().enumerate() {
                    if i % 16 == 0 {
                        out.write_all(b" ")?;
                    }
                    let c = match page {
                        Page::Meta => 'M',
                        Page::Free => '.',
                        Page::Node(info) if info.level > 0 => {
                            char::from_digit(info.level.into(), 36).unwrap_or('+')
                        }
                        Page::Node(info) if info.dense => 'D',
                        Page::Node(info) if info.value => 'V',
                        Page::Node(_) => 'L',
                    };
                    write!(out, "{c}")?;
                }
                writeln!(out)?;
            }
        }
    }
    out.flush()?;
    Ok(ExitCode::SUCCESS)
}

/// Names the store that `e` is about.
fn at(path: &Path, e: encina::Error) -> Box<dyn Error> {
    format!("{}: {e}", path.display()).into()
}

/// Writes `batch` to the store at `path`, creating the store if needed.
async fn write(path: &Path, batch: &mut WriteBatch) -> Result<(), Box<dyn Error>> {
    // Each command is a write of its own, of a few pages: taking any free page
    // keeps the file small, at the cost of a request or two.
    let db = Db::open(&DefaultEnv, path, Options::default().min_run_pages(1))
        .await
        .map_err(|e| at(path, e))?;
    db.write(batch).await.map_err(|e| at(path, e))
}

/// Writes `batch` to the existing store at `path`: a delete has nothing to remove
/// from a store that isn't there, and doesn't create one.
async fn write_existing(path: &Path, batch: &mut WriteBatch) -> Result<(), Box<dyn Error>> {
    fs::metadata(path).map_err(|e| at(path, e.into()))?;
    write(path, batch).await
}

/// Reads the value under `key`. When the store's owner overwrites the version the
/// view reads, opens a new view and reads again.
async fn get(path: &Path, key: &[u8]) -> encina::Result<Option<Vec<u8>>> {
    let mut stale = 0;
    loop {
        let read = async {
            let db = ReadOnlyDb::open(&DefaultEnv, path, Options::default()).await?;
            db.get(key, Comparison::Equal, |_k, v| v.to_vec()).await
        };
        match read.await {
            Err(encina::Error::Stale) if stale < MAX_STALE => stale += 1,
            result => return result,
        }
    }
}

/// What a page of a store's file holds.
enum Page {
    Meta,
    Node(PageInfo),
    Free,
}

/// A version of a store, and its file, as [`read_pages`] finds them.
struct Store {
    page_size: usize,
    /// The file's length, in bytes.
    file_size: u64,
    /// The version's entries, and the bytes of their keys and values.
    entry_count: u64,
    entry_size: u64,
    /// What each page of the file holds.
    pages: Vec<Page>,
}

/// Reads the store's current version, and what each page of its file holds. When
/// the store's owner overwrites the version the view reads, opens a new view and
/// starts over.
async fn read_pages(path: &Path) -> Result<Store, Box<dyn Error>> {
    let mut stale = 0;
    let (page_size, entry_count, entry_size, nodes) = loop {
        let read = async {
            let db = ReadOnlyDb::open(&DefaultEnv, path, Options::default()).await?;
            let mut nodes = Vec::new();
            db.pages(|info| nodes.push(info)).await?;
            Ok((db.page_size(), db.entry_count(), db.entry_size(), nodes))
        };
        match read.await {
            Err(encina::Error::Stale) if stale < MAX_STALE => stale += 1,
            result => break result.map_err(|e| at(path, e))?,
        }
    };
    let file_size = fs::metadata(path)?.len();
    let len = file_size as usize;
    let end = nodes
        .iter()
        .map(|info| (info.page + info.pages) as usize)
        .max()
        .unwrap_or(0);
    let mut pages: Vec<_> = (0..end.max(len / page_size).max(META_PAGES))
        .map(|page| {
            if page < META_PAGES {
                Page::Meta
            } else {
                Page::Free
            }
        })
        .collect();
    for info in nodes {
        for page in info.page..info.page + info.pages {
            pages[page as usize] = Page::Node(info);
        }
    }
    Ok(Store {
        page_size,
        file_size,
        entry_count,
        entry_size,
        pages,
    })
}

/// Calls `print` on the entries with keys in the range, in key order. When the
/// store's owner overwrites the version the view reads, opens a new view, and
/// resumes after the last key printed.
async fn scan(
    path: &Path,
    start: Bound<&[u8]>,
    end: Bound<&[u8]>,
    mut print: impl FnMut(&[u8], &[u8]) -> io::Result<()>,
) -> Result<(), Box<dyn Error>> {
    let mut last: Option<Vec<u8>> = None;
    let mut stale = 0;
    loop {
        let db = match ReadOnlyDb::open(&DefaultEnv, path, Options::default()).await {
            Ok(db) => db,
            Err(encina::Error::Stale) if stale < MAX_STALE => {
                stale += 1;
                continue;
            }
            Err(e) => return Err(at(path, e)),
        };
        let start = last.as_deref().map_or(start, Bound::Excluded);
        let mut iter = db.range((start, end));
        loop {
            match iter.next().await {
                Ok(Some((key, value))) => {
                    print(key, value)?;
                    last = Some(key.to_vec());
                    stale = 0;
                }
                Ok(None) => return Ok(()),
                Err(encina::Error::Stale) if stale < MAX_STALE => {
                    stale += 1;
                    break;
                }
                Err(e) => return Err(at(path, e)),
            }
        }
    }
}

/// Decodes an argument, in which `\xNN` stands for the byte with hex value `NN`, and
/// `\\` for a backslash.
fn unescape(arg: &str) -> Result<Box<[u8]>, String> {
    let invalid = || r"use \xNN for a byte, or \\ for a backslash".to_string();
    let hex = |d: &u8| (*d as char).to_digit(16);
    let mut bytes = Vec::with_capacity(arg.len());
    let mut rest = arg.as_bytes();
    while let [b, tail @ ..] = rest {
        rest = match (b, tail) {
            (b'\\', [b'\\', tail @ ..]) => {
                bytes.push(b'\\');
                tail
            }
            (b'\\', [b'x', hi, lo, tail @ ..]) => match (hex(hi), hex(lo)) {
                (Some(hi), Some(lo)) => {
                    bytes.push((hi * 16 + lo) as u8);
                    tail
                }
                _ => return Err(invalid()),
            },
            (b'\\', _) => return Err(invalid()),
            (b, tail) => {
                bytes.push(*b);
                tail
            }
        };
    }
    Ok(bytes.into_boxed_slice())
}

/// Writes `bytes` as [`unescape`] reads them: UTF-8 text as is, and backslashes,
/// control characters and bytes that aren't UTF-8 escaped.
fn write_text(out: &mut impl Write, bytes: &[u8]) -> io::Result<()> {
    for chunk in bytes.utf8_chunks() {
        for c in chunk.valid().chars() {
            if c == '\\' {
                out.write_all(br"\\")?;
            } else if c.is_control() {
                for b in c.encode_utf8(&mut [0; 4]).bytes() {
                    write!(out, r"\x{b:02x}")?;
                }
            } else {
                write!(out, "{c}")?;
            }
        }
        for b in chunk.invalid() {
            write!(out, r"\x{b:02x}")?;
        }
    }
    Ok(())
}

/// Writes `bytes` in the layout of `hexdump -C`: lines of up to 16 bytes, each with
/// its offset, the bytes in hex, and the bytes as ASCII. `label` starts the first
/// line, and as many spaces the others.
fn write_hexdump(out: &mut impl Write, label: &str, bytes: &[u8]) -> io::Result<()> {
    let width = label.len();
    // No bytes still take a line.
    for offset in (0..bytes.len().max(1)).step_by(16) {
        let line = &bytes[offset..bytes.len().min(offset + 16)];
        let label = if offset == 0 { label } else { "" };
        write!(out, "{label:width$}{offset:08x} ")?;
        for i in 0..16 {
            if i % 8 == 0 {
                out.write_all(b" ")?;
            }
            match line.get(i) {
                Some(b) => write!(out, "{b:02x} ")?,
                None => out.write_all(b"   ")?,
            }
        }
        out.write_all(b" |")?;
        for &b in line {
            let shown = if b == b' ' || b.is_ascii_graphic() {
                b
            } else {
                b'.'
            };
            out.write_all(&[shown])?;
        }
        out.write_all(b"|\n")?;
    }
    Ok(())
}
