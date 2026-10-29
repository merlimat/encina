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

//! The meta page records which tree is current.
//!
//! There are two slots, pages 0 and 1, and commits alternate between them: a torn
//! write of one slot leaves the other intact.
//!
//! ```text
//! offset  size  field
//!      0     4  checksum (crc32c of bytes 4..64)
//!      4     8  magic
//!     12     4  format version
//!     16     4  page size
//!     20     1  root level
//!     24     8  generation
//!     32     8  root page, 0 for an empty tree
//!     40     8  high-water mark: no page at or above it is in use
//!     48     8  entries in the tree
//!     56     8  bytes of their keys and values
//! ```

/// Pages 0 and 1 hold the meta slots; nodes start at page 2.
pub(crate) const META_PAGES: u64 = 2;

pub(crate) const MAGIC: [u8; 8] = *b"encina\0\0";
pub(crate) const FORMAT_VERSION: u32 = 1;
const LEN: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Meta {
    pub(crate) version: u32,
    pub(crate) page_size: u32,
    pub(crate) generation: u64,
    /// Page of the root node, 0 if the tree is empty.
    pub(crate) root: u64,
    pub(crate) root_level: u8,
    pub(crate) high_water: u64,
    /// The tree's entries, and the bytes of their keys and values.
    pub(crate) entries: u64,
    pub(crate) entry_bytes: u64,
}

impl Meta {
    /// The meta of a new, empty store.
    pub(crate) fn empty(page_size: usize) -> Meta {
        Meta {
            version: FORMAT_VERSION,
            page_size: page_size as u32,
            generation: 0,
            root: 0,
            root_level: 0,
            high_water: META_PAGES,
            entries: 0,
            entry_bytes: 0,
        }
    }

    /// The page holding this generation's slot.
    pub(crate) fn slot(&self) -> u64 {
        self.generation % 2
    }

    pub(crate) fn encode(&self, page: &mut [u8]) {
        page.fill(0);
        page[4..12].copy_from_slice(&MAGIC);
        page[12..16].copy_from_slice(&self.version.to_le_bytes());
        page[16..20].copy_from_slice(&self.page_size.to_le_bytes());
        page[20] = self.root_level;
        page[24..32].copy_from_slice(&self.generation.to_le_bytes());
        page[32..40].copy_from_slice(&self.root.to_le_bytes());
        page[40..48].copy_from_slice(&self.high_water.to_le_bytes());
        page[48..56].copy_from_slice(&self.entries.to_le_bytes());
        page[56..64].copy_from_slice(&self.entry_bytes.to_le_bytes());
        let crc = crc32c::crc32c(&page[4..LEN]);
        page[..4].copy_from_slice(&crc.to_le_bytes());
    }

    /// Decodes a slot, or returns `None` if it doesn't hold an intact meta page:
    /// never written, torn, or not from encina.
    pub(crate) fn decode(page: &[u8]) -> Option<Meta> {
        if page.len() < LEN || page[4..12] != MAGIC {
            return None;
        }
        let crc = u32::from_le_bytes(page[..4].try_into().unwrap());
        if crc32c::crc32c(&page[4..LEN]) != crc {
            return None;
        }
        let u32_at = |at: usize| u32::from_le_bytes(page[at..at + 4].try_into().unwrap());
        let u64_at = |at: usize| u64::from_le_bytes(page[at..at + 8].try_into().unwrap());
        Some(Meta {
            version: u32_at(12),
            page_size: u32_at(16),
            root_level: page[20],
            generation: u64_at(24),
            root: u64_at(32),
            high_water: u64_at(40),
            entries: u64_at(48),
            entry_bytes: u64_at(56),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_rejects_damage() {
        let meta = Meta {
            version: FORMAT_VERSION,
            page_size: 4096,
            generation: 7,
            root: 1234,
            root_level: 3,
            high_water: 99,
            entries: 12_345,
            entry_bytes: 1 << 40,
        };
        let mut page = vec![0xff; 4096];
        meta.encode(&mut page);
        assert_eq!(Meta::decode(&page), Some(meta));
        assert_eq!(meta.slot(), 1);
        for at in [0, 4, 20, 47, 63] {
            let mut torn = page.clone();
            torn[at] ^= 0x10;
            assert_eq!(
                Meta::decode(&torn),
                None,
                "damage at byte {at} went unnoticed"
            );
        }
        assert_eq!(Meta::decode(&[0; 4096]), None);
    }
}
