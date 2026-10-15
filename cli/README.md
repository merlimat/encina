# encina-cli

The `encina` command puts, gets and scans the entries of an
[encina](https://crates.io/crates/encina) store, for inspecting a store by hand.

```sh
cargo install encina-cli
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

## License

Copyright 2026 Matteo Merli. Licensed under the [Apache License, Version 2.0](../LICENSE).
