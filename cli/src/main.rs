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

//! `encina`: puts, gets and scans the entries of a store from the command line.

use std::error::Error;
use std::io::{self, BufWriter, Write};
use std::ops::Bound;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use encina::{Comparison, Db, Options, ReadOnlyDb, WriteBatch};

/// How many times in a row a read opens a new view of the store, after finding
/// that the store's owner overwrote the version its view reads.
const MAX_STALE: u32 = 10;

const AFTER_HELP: &str = r"get and scan open the store read-only, so they work while another process has it
open for writing. If that process overwrites the version they read, they open the
store again: get reads the current version, and scan resumes after the last key it
printed.

Keys and values are bytes: in an argument, \xNN stands for the byte with hex value
NN, and \\ for a backslash. Output prints UTF-8 text as is, and escapes backslashes,
control characters and bytes that aren't UTF-8 the same way.

Exit status: 0 on success, 1 if get finds no such key, 2 on errors.";

/// Puts, gets and scans the entries of an encina store.
#[derive(Parser)]
#[command(version, after_help = AFTER_HELP)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Writes a value under a key, creating the store if needed
    Put {
        /// The store's file
        path: PathBuf,
        #[arg(value_parser = unescape)]
        key: Box<[u8]>,
        #[arg(value_parser = unescape)]
        value: Box<[u8]>,
    },
    /// Prints the value under a key
    Get {
        /// The store's file
        path: PathBuf,
        #[arg(value_parser = unescape)]
        key: Box<[u8]>,
        /// Prints the value in the layout of `hexdump -C`
        #[arg(short = 'x', long)]
        hexdump: bool,
    },
    /// Prints the entries in a range of keys, one per line as key<TAB>value
    Scan {
        /// The store's file
        path: PathBuf,
        /// The first key in the range; by default, the first key in the store
        #[arg(value_parser = unescape)]
        start: Option<Box<[u8]>>,
        /// The key the range stops before; by default, the range has no end
        #[arg(value_parser = unescape)]
        end: Option<Box<[u8]>>,
        /// Prints keys and values in the layout of `hexdump -C`
        #[arg(short = 'x', long)]
        hexdump: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result: Result<ExitCode, Box<dyn Error>> = compio::runtime::Runtime::new()
        .map_err(Box::from)
        .and_then(|runtime| runtime.block_on(run(cli.command)));
    match result {
        Ok(code) => code,
        // The reader went away, as in `encina scan data.db | head`: stop quietly.
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

async fn run(command: Command) -> Result<ExitCode, Box<dyn Error>> {
    let mut out = BufWriter::new(io::stdout().lock());
    match command {
        Command::Put { path, key, value } => {
            let db = Db::open(&path, Options::default())
                .await
                .map_err(|e| at(&path, e))?;
            let mut batch = WriteBatch::new();
            batch.put(&key, &value);
            db.write(&mut batch).await.map_err(|e| at(&path, e))?;
        }
        Command::Get { path, key, hexdump } => {
            let Some(value) = get(&path, &key).await.map_err(|e| at(&path, e))? else {
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
            path,
            start,
            end,
            hexdump,
        } => {
            let start = start.as_deref().map_or(Bound::Unbounded, Bound::Included);
            let end = end.as_deref().map_or(Bound::Unbounded, Bound::Excluded);
            scan(&path, start, end, |key, value| {
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
    }
    out.flush()?;
    Ok(ExitCode::SUCCESS)
}

/// Names the store that `e` is about.
fn at(path: &Path, e: encina::Error) -> Box<dyn Error> {
    format!("{}: {e}", path.display()).into()
}

/// Reads the value under `key`. When the store's owner overwrites the version the
/// view reads, opens a new view and reads again.
async fn get(path: &Path, key: &[u8]) -> encina::Result<Option<Vec<u8>>> {
    let mut stale = 0;
    loop {
        let read = async {
            let db = ReadOnlyDb::open(path, Options::default()).await?;
            db.get(key, Comparison::Equal, |_k, v| v.to_vec()).await
        };
        match read.await {
            Err(encina::Error::Stale) if stale < MAX_STALE => stale += 1,
            result => return result,
        }
    }
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
        let db = match ReadOnlyDb::open(path, Options::default()).await {
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
