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

//! Runs the `encina` command on stores in temporary directories.

use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Command, Output, Stdio};

use encina::{Db, Options, WriteBatch};

fn encina(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_encina"))
        .args(args)
        .output()
        .unwrap()
}

/// Runs `encina` and returns its standard output, checking that it succeeded.
fn ok(args: &[&str]) -> String {
    let output = encina(args);
    assert!(output.status.success(), "{args:?} failed: {output:?}");
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn put_get() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.db");
    let path = path.to_str().unwrap();

    ok(&["-f", path, "put", "user/1", "alice"]);
    ok(&["-f", path, "put", "user/1", "bob"]);
    assert_eq!(ok(&["-f", path, "get", "user/1"]), "bob\n");

    let missing = encina(&["-f", path, "get", "user/2"]);
    assert_eq!(missing.status.code(), Some(1));
    assert!(missing.stdout.is_empty());
}

#[test]
fn puts_reuse_the_pages_they_free() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.db");
    let path = path.to_str().unwrap();

    // Each put moves the key's leaf to a new page and frees the one before, which
    // the next put takes back: the file holds the meta pages and two leaves.
    for i in 0..20 {
        ok(&["-f", path, "put", "key", &i.to_string()]);
        let len = std::fs::metadata(path).unwrap().len();
        assert!(len <= 4 * 4096, "{len} bytes after {} puts", i + 1);
    }
}

#[test]
fn scan_bounds() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.db");
    let path = path.to_str().unwrap();
    for key in ["a", "b", "c", "d"] {
        ok(&["-f", path, "put", key, &key.to_uppercase()]);
    }

    assert_eq!(ok(&["-f", path, "scan"]), "a\tA\nb\tB\nc\tC\nd\tD\n");
    assert_eq!(
        ok(&["-f", path, "scan", "--start", "b"]),
        "b\tB\nc\tC\nd\tD\n"
    );
    assert_eq!(
        ok(&["-f", path, "scan", "-s", "b", "-e", "d"]),
        "b\tB\nc\tC\n"
    );
    assert_eq!(ok(&["-f", path, "scan", "--end", "c"]), "a\tA\nb\tB\n");
    assert_eq!(ok(&["-f", path, "scan", "--start", "x"]), "");
}

#[test]
fn delete_and_delete_range() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.db");
    let path = path.to_str().unwrap();
    for key in ["a", "b", "c", "d", "e"] {
        ok(&["-f", path, "put", key, &key.to_uppercase()]);
    }

    ok(&["-f", path, "delete", "b"]);
    assert_eq!(encina(&["-f", path, "get", "b"]).status.code(), Some(1));
    // Deleting a key that isn't there succeeds, and changes nothing.
    ok(&["-f", path, "delete", "b"]);
    assert_eq!(ok(&["-f", path, "scan"]), "a\tA\nc\tC\nd\tD\ne\tE\n");

    ok(&["-f", path, "delete-range", "-s", "c", "-e", "e"]);
    assert_eq!(ok(&["-f", path, "scan"]), "a\tA\ne\tE\n");
    // A range that starts at or after its end removes nothing.
    ok(&["-f", path, "delete-range", "--start", "e", "--end", "a"]);
    assert_eq!(ok(&["-f", path, "scan"]), "a\tA\ne\tE\n");
    ok(&["-f", path, "delete-range", "--start", "", "--end", "z"]);
    assert_eq!(ok(&["-f", path, "scan"]), "");
}

#[test]
fn text_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.db");
    let path = path.to_str().unwrap();

    // UTF-8 text prints as is; backslashes, control characters and bytes that
    // aren't UTF-8 print escaped, as arguments take them.
    ok(&["-f", path, "put", r"\x00\xFF", r"café\x09\xc2\x85\\"]);
    let printed = ok(&["-f", path, "scan"]);
    assert_eq!(printed, "\\x00\\xff\tcafé\\x09\\xc2\\x85\\\\\n");
    let (key, value) = printed.trim_end().split_once('\t').unwrap();
    assert_eq!(ok(&["-f", path, "get", key]), format!("{value}\n"));
}

#[test]
fn hexdump() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.db");
    let path = path.to_str().unwrap();
    ok(&["-f", path, "put", r"k\x00", "hello world, this is long"]);
    ok(&["-f", path, "put", "z", ""]);

    assert_eq!(
        ok(&["-f", path, "get", "-x", r"k\x00"]),
        "00000000  68 65 6c 6c 6f 20 77 6f  72 6c 64 2c 20 74 68 69  |hello world, thi|\n\
         00000010  73 20 69 73 20 6c 6f 6e  67                       |s is long|\n"
    );
    assert_eq!(
        ok(&["-f", path, "scan", "--hexdump", "--start", "z"]),
        "key   00000000  7a                                                |z|\n\
         value 00000000                                                    ||\n"
    );
}

#[test]
fn errors_leave_no_store() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.db");
    let path = path.to_str().unwrap();

    // Reads and deletes never create the store, and a put with a bad argument
    // doesn't open it.
    assert_eq!(encina(&["-f", path, "get", "k"]).status.code(), Some(2));
    assert_eq!(encina(&["-f", path, "scan"]).status.code(), Some(2));
    assert_eq!(encina(&["-f", path, "delete", "k"]).status.code(), Some(2));
    assert_eq!(
        encina(&["-f", path, "delete-range", "--start", "a", "--end", "b"])
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        encina(&["-f", path, "put", r"k\q", "v"]).status.code(),
        Some(2)
    );
    assert_eq!(
        encina(&["-f", path, "put", r"k\x0", "v"]).status.code(),
        Some(2)
    );
    assert!(!Path::new(path).exists());

    assert_eq!(encina(&[]).status.code(), Some(2));
    // The store's file comes before the command, with -f.
    assert_eq!(encina(&["put", path, "k", "v"]).status.code(), Some(2));
    assert_eq!(encina(&["-f", path, "frob"]).status.code(), Some(2));
    assert_eq!(encina(&["-f", path, "get"]).status.code(), Some(2));
    // A range's keys come with --start and --end, and delete-range needs both.
    assert_eq!(encina(&["-f", path, "scan", "a"]).status.code(), Some(2));
    assert_eq!(
        encina(&["-f", path, "delete-range", "--start", "a"])
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        encina(&["-f", path, "delete-range", "--end", "b"])
            .status
            .code(),
        Some(2)
    );
    assert!(ok(&["--help"]).starts_with("Puts, gets, scans and deletes"));
}

#[compio::test]
async fn reads_while_another_process_writes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.db");
    let db = Db::open(&path, Options::default()).await.unwrap();
    let mut batch = WriteBatch::new();
    batch.put(b"k", b"v");
    db.write(&mut batch).await.unwrap();
    let path = path.to_str().unwrap();

    // `db` holds the store's lock: reads go through a read-only view, but a put
    // can't open it.
    assert_eq!(ok(&["-f", path, "get", "k"]), "v\n");
    assert_eq!(ok(&["-f", path, "scan"]), "k\tv\n");
    let put = encina(&["-f", path, "put", "k", "w"]);
    assert_eq!(put.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&put.stderr).contains("already open"));

    db.close().await.unwrap();
    ok(&["-f", path, "put", "k", "w"]);
    assert_eq!(ok(&["-f", path, "get", "k"]), "w\n");
}

#[compio::test]
async fn works_on_any_page_size() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.db");
    let db = Db::open(&path, Options::default().page_size(16384))
        .await
        .unwrap();
    let mut batch = WriteBatch::new();
    batch.put(b"k", b"v");
    db.write(&mut batch).await.unwrap();
    db.close().await.unwrap();
    let path = path.to_str().unwrap();

    // The command sets no page size, so it opens the store with its own.
    ok(&["-f", path, "put", "k", "w"]);
    assert_eq!(ok(&["-f", path, "get", "k"]), "w\n");
    assert_eq!(ok(&["-f", path, "scan"]), "k\tw\n");
}

/// The value of a line of `encina stats`, by its name.
fn stat(stats: &str, name: &str) -> usize {
    let line = stats.lines().find(|line| line.starts_with(name)).unwrap();
    line[name.len()..].trim().parse().unwrap()
}

/// The rows of the table of `encina stats`, but for how full the pages are: level,
/// pages, dense nodes, value leaves and entries.
fn levels(stats: &str) -> Vec<[usize; 5]> {
    let table = stats.lines().skip_while(|line| !line.starts_with("level"));
    table
        .skip(1)
        .map(|row| {
            let cells: Vec<usize> = row
                .split_whitespace()
                .take(5)
                .map(|c| c.parse().unwrap())
                .collect();
            cells.try_into().unwrap()
        })
        .collect()
}

#[compio::test]
async fn stats_and_pages() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.db");
    let db = Db::open(&path, Options::default()).await.unwrap();
    for value in [b"v", b"w"] {
        let mut batch = WriteBatch::new();
        batch.put(b"k", value);
        db.write(&mut batch).await.unwrap();
    }
    let path_str = path.to_str().unwrap();

    // The second write moved the leaf to a new page, and freed the first. Both
    // commands read the store while its owner has it open.
    assert_eq!(ok(&["-f", path_str, "pages"]), "0  MM.D\n");
    let stats = ok(&["-f", path_str, "stats"]);
    let summary = [
        ("entries", 1),
        ("entry size", 2),
        ("page size", 4096),
        ("file size", 4 * 4096),
        ("file pages", 4),
        ("meta pages", 2),
        ("tree pages", 1),
        ("free pages", 1),
        ("free stretches", 1),
        ("longest stretch", 1),
    ];
    for (name, value) in summary {
        assert_eq!(stat(&stats, name), value, "{name}");
    }
    assert_eq!(levels(&stats), [[0, 1, 1, 0, 1]]);

    // A tree of two levels.
    let mut batch = WriteBatch::new();
    for i in 0..20_000 {
        batch.put(format!("{i:06}").as_bytes(), &[b'v'; 100]);
    }
    db.write(&mut batch).await.unwrap();
    let stats = ok(&["-f", path_str, "stats"]);
    let [root, leaves] = levels(&stats)[..] else {
        panic!("{stats}");
    };
    assert_eq!(root[..2], [1, 1]);
    assert_eq!((leaves[0], leaves[4]), (0, 20_001), "k and the 20,000 keys");
    assert_eq!(root[4], leaves[1], "the root's entries are the leaves");
    assert_eq!(stat(&stats, "tree pages"), 1 + leaves[1]);
    let file = stat(&stats, "file pages");
    assert_eq!(file, 2 + 1 + leaves[1] + stat(&stats, "free pages"));
    assert_eq!(file as u64 * 4096, std::fs::metadata(&path).unwrap().len());
    assert_eq!(stat(&stats, "file size"), file * 4096);
    assert_eq!(stat(&stats, "entries"), 20_001);
    assert_eq!(stat(&stats, "entry size"), 2 + 20_000 * 106);

    // The map shows the same pages, 64 to a line.
    let map = ok(&["-f", path_str, "pages"]);
    assert_eq!(map.lines().count(), file.div_ceil(64));
    let shown = |c| {
        map.lines()
            .flat_map(|line| line.split_whitespace().skip(1))
            .flat_map(str::chars)
            .filter(|&s| s == c)
            .count()
    };
    assert_eq!(shown('M'), 2);
    assert_eq!(shown('1'), 1);
    assert_eq!(shown('L') + shown('D'), leaves[1]);

    // An entry too large for a leaf takes three pages of its own.
    let mut batch = WriteBatch::new();
    batch.put(b"big", &[b'v'; 10_000]);
    db.write(&mut batch).await.unwrap();
    let stats = ok(&["-f", path_str, "stats"]);
    let leaves = *levels(&stats).last().unwrap();
    assert_eq!((leaves[3], leaves[4]), (1, 20_002));
    assert_eq!(stat(&stats, "entry size"), 2 + 20_000 * 106 + 10_003);
    let map = ok(&["-f", path_str, "pages"]);
    let shown = |c| map.chars().filter(|&s| s == c).count();
    assert_eq!(shown('V'), 3);
    assert_eq!(shown('L') + shown('D') + shown('V'), leaves[1]);
    assert_eq!(
        ok(&["-f", path_str, "get", "big"]),
        format!("{}\n", "v".repeat(10_000))
    );
    db.close().await.unwrap();
}

/// A scan whose version the store's owner overwrites midway opens a new view, and
/// resumes after the last key it printed.
#[compio::test]
async fn scan_resumes_on_a_new_view() {
    const ENTRIES: usize = 20_000;
    async fn write_all(db: &Db, generation: char) {
        let value = generation.to_string().repeat(100);
        let mut batch = WriteBatch::new();
        for i in 0..ENTRIES {
            batch.put(format!("{i:06}").as_bytes(), value.as_bytes());
        }
        db.write(&mut batch).await.unwrap();
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.db");
    let db = Db::open(&path, Options::default()).await.unwrap();
    write_all(&db, '0').await;

    let mut scan = Command::new(env!("CARGO_BIN_EXE_encina"))
        .args(["-f", path.to_str().unwrap(), "scan"])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdout = BufReader::new(scan.stdout.take().unwrap());
    // Once the scan prints, its view is open. It stops soon after, when the pipe
    // fills, far from the end of the store.
    let mut printed = String::new();
    stdout.read_line(&mut printed).unwrap();
    // From its second write on, the owner reuses the pages of the view's version.
    write_all(&db, '1').await;
    write_all(&db, '2').await;
    stdout.read_to_string(&mut printed).unwrap();
    assert!(scan.wait().unwrap().success());

    let mut generations = Vec::new();
    for (i, line) in printed.lines().enumerate() {
        let (key, value) = line.split_once('\t').unwrap();
        assert_eq!(key, format!("{i:06}"));
        assert_eq!(value, value[..1].repeat(100));
        generations.push(value.as_bytes()[0]);
    }
    assert_eq!(generations.len(), ENTRIES);
    // It began in the first version, and finished in a later one.
    assert!(generations.is_sorted());
    assert_eq!(generations[0], b'0');
    assert_ne!(generations[ENTRIES - 1], b'0');
}
