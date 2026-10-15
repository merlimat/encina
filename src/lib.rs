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

//! encina: an embedded, ordered key-value store built on the compio runtime.
//!
//! See `DESIGN.md` for the design.

mod batch;
mod db;
mod error;
mod node;
mod options;
mod pool;
mod tree;

#[cfg(test)]
mod alloc_tests;
#[cfg(test)]
mod tests;

pub use batch::WriteBatch;
pub use db::{Comparison, Db, Iter};
pub use error::{Error, Result};
pub use options::Options;
