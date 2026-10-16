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

    ok(&["put", path, "user/1", "alice"]);
    ok(&["put", path, "user/1", "bob"]);
    assert_eq!(ok(&["get", path, "user/1"]), "bob\n");

    let missing = encina(&["get", path, "user/2"]);
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
        ok(&["put", path, "key", &i.to_string()]);
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
        ok(&["put", path, key, &key.to_uppercase()]);
    }

    assert_eq!(ok(&["scan", path]), "a\tA\nb\tB\nc\tC\nd\tD\n");
    assert_eq!(ok(&["scan", path, "b"]), "b\tB\nc\tC\nd\tD\n");
    assert_eq!(ok(&["scan", path, "b", "d"]), "b\tB\nc\tC\n");
    assert_eq!(ok(&["scan", path, "", "c"]), "a\tA\nb\tB\n");
    assert_eq!(ok(&["scan", path, "x"]), "");
}

#[test]
fn text_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.db");
    let path = path.to_str().unwrap();

    // UTF-8 text prints as is; backslashes, control characters and bytes that
    // aren't UTF-8 print escaped, as arguments take them.
    ok(&["put", path, r"\x00\xFF", r"café\x09\xc2\x85\\"]);
    let printed = ok(&["scan", path]);
    assert_eq!(printed, "\\x00\\xff\tcafé\\x09\\xc2\\x85\\\\\n");
    let (key, value) = printed.trim_end().split_once('\t').unwrap();
    assert_eq!(ok(&["get", path, key]), format!("{value}\n"));
}

#[test]
fn hexdump() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.db");
    let path = path.to_str().unwrap();
    ok(&["put", path, r"k\x00", "hello world, this is long"]);
    ok(&["put", path, "z", ""]);

    assert_eq!(
        ok(&["get", "-x", path, r"k\x00"]),
        "00000000  68 65 6c 6c 6f 20 77 6f  72 6c 64 2c 20 74 68 69  |hello world, thi|\n\
         00000010  73 20 69 73 20 6c 6f 6e  67                       |s is long|\n"
    );
    assert_eq!(
        ok(&["scan", "--hexdump", path, "z"]),
        "key   00000000  7a                                                |z|\n\
         value 00000000                                                    ||\n"
    );
}

#[test]
fn errors_leave_no_store() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.db");
    let path = path.to_str().unwrap();

    // Reads never create the store, and a put with a bad argument doesn't open it.
    assert_eq!(encina(&["get", path, "k"]).status.code(), Some(2));
    assert_eq!(encina(&["scan", path]).status.code(), Some(2));
    assert_eq!(encina(&["put", path, r"k\q", "v"]).status.code(), Some(2));
    assert_eq!(encina(&["put", path, r"k\x0", "v"]).status.code(), Some(2));
    assert!(!Path::new(path).exists());

    assert_eq!(encina(&[]).status.code(), Some(2));
    assert_eq!(encina(&["frob", path]).status.code(), Some(2));
    assert_eq!(encina(&["get", path]).status.code(), Some(2));
    assert_eq!(
        encina(&["scan", path, "a", "b", "c"]).status.code(),
        Some(2)
    );
    assert!(ok(&["--help"]).starts_with("Puts, gets and scans"));
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
    assert_eq!(ok(&["get", path, "k"]), "v\n");
    assert_eq!(ok(&["scan", path]), "k\tv\n");
    let put = encina(&["put", path, "k", "w"]);
    assert_eq!(put.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&put.stderr).contains("already open"));

    db.close().await.unwrap();
    ok(&["put", path, "k", "w"]);
    assert_eq!(ok(&["get", path, "k"]), "w\n");
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
        .args(["scan", path.to_str().unwrap()])
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
