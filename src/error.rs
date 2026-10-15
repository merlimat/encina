use std::fmt;

/// Errors returned by encina.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A put's key plus value exceeds [`Options::max_entry_size`](crate::Options::max_entry_size).
    /// The batch containing it was not applied.
    EntryTooLarge { size: usize, max: usize },
    /// The options are invalid.
    InvalidOptions(&'static str),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::EntryTooLarge { size, max } => {
                write!(f, "entry of {size} bytes exceeds the maximum of {max}")
            }
            Error::InvalidOptions(reason) => write!(f, "invalid options: {reason}"),
        }
    }
}

impl std::error::Error for Error {}
