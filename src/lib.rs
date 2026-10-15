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
