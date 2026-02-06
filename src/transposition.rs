use crate::r#move::Move;
use std::sync::atomic::{AtomicU64, Ordering};

/// Transposition table entry
#[derive(Debug, Clone, Copy)]
pub struct TTEntry {
    /// Zobrist hash of the position
    pub hash: u64,
    /// Best move found from this position
    pub best_move: Move,
    /// Score from the search
    pub score: i32,
    /// Depth of the search
    pub depth: u8,
    /// Entry type (bounds)
    pub flag: TTFlag,
    /// Age of the entry (for replacement)
    pub age: u8,
}

/// Transposition table entry flag (bound type)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TTFlag {
    /// Exact score (PV node)
    Exact = 0,
    /// Lower bound (beta cutoff - score >= actual)
    Lower = 1,
    /// Upper bound (alpha cutoff - score <= actual)
    Upper = 2,
}

impl TTEntry {
    /// Create a new TT entry
    #[inline]
    pub fn new(hash: u64, best_move: Move, score: i32, depth: u8, flag: TTFlag, age: u8) -> Self {
        TTEntry {
            hash,
            best_move,
            score,
            depth,
            flag,
            age,
        }
    }

    /// Create an empty entry
    #[inline]
    pub const fn empty() -> Self {
        TTEntry {
            hash: 0,
            best_move: Move::null(),
            score: 0,
            depth: 0,
            flag: TTFlag::Exact,
            age: 0,
        }
    }

    /// Check if this entry is empty
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.hash == 0 && self.depth == 0
    }
}

/// Transposition table
pub struct TranspositionTable {
    /// Table entries (boxed slice for better memory locality)
    entries: Box<[TTEntry]>,
    /// Size of the table in bytes
    size_bytes: usize,
    /// Number of entries in the table
    num_entries: usize,
    /// Mask for indexing into the table (must be power of 2 - 1)
    index_mask: usize,
    /// Current generation/age
    current_age: AtomicU64,
    /// Statistics
    hits: AtomicU64,
    misses: AtomicU64,
    stores: AtomicU64,
    collisions: AtomicU64,
}

impl TranspositionTable {
    /// Create a new transposition table with the specified size in megabytes
    pub fn new(size_mb: usize) -> Self {
        let size_bytes = size_mb * 1024 * 1024;
        let entry_size = std::mem::size_of::<TTEntry>();

        // Calculate number of entries (must be power of 2)
        let mut num_entries = size_bytes / entry_size;
        // Round down to power of 2
        if num_entries > 1 {
            let leading_zeros = num_entries.leading_zeros() as usize;
            num_entries = 1 << (64 - leading_zeros);
            // If we rounded up, go back down
            if num_entries > (size_bytes / entry_size) {
                num_entries >>= 1;
            }
        } else {
            num_entries = 1;
        }

        let actual_size = num_entries * entry_size;

        // Create boxed slice with all entries empty
        let entries: Vec<TTEntry> = vec![TTEntry::empty(); num_entries];
        let entries = entries.into_boxed_slice();

        let index_mask = num_entries - 1;

        TranspositionTable {
            entries,
            size_bytes: actual_size,
            num_entries,
            index_mask,
            current_age: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            stores: AtomicU64::new(0),
            collisions: AtomicU64::new(0),
        }
    }

    /// Get the size of the transposition table in megabytes
    #[inline]
    pub fn size_mb(&self) -> usize {
        self.size_bytes / (1024 * 1024)
    }

    /// Get the index for a given hash
    #[inline]
    fn get_index(&self, hash: u64) -> usize {
        (hash as usize) & self.index_mask
    }

    /// Probe the transposition table for a given hash
    #[inline]
    pub fn probe(&self, hash: u64) -> Option<TTEntry> {
        let index = self.get_index(hash);
        let entry = self.entries[index];

        if entry.hash == hash && !entry.is_empty() {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry)
        } else if !entry.is_empty() {
            // Hash collision
            self.collisions.fetch_add(1, Ordering::Relaxed);
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Store an entry in the transposition table
    #[inline]
    pub fn store(&self, hash: u64, best_move: Move, score: i32, depth: u8, flag: TTFlag) {
        let index = self.get_index(hash);
        let entry = self.entries[index];

        // Replacement strategy: prefer deeper searches or newer entries
        let should_replace = entry.is_empty()
            || entry.hash != hash
            || depth >= entry.depth
            || (self.current_age.load(Ordering::Relaxed) as u8) > entry.age + 3;

        if should_replace {
            let new_entry = TTEntry::new(
                hash,
                best_move,
                score,
                depth,
                flag,
                self.current_age.load(Ordering::Relaxed) as u8,
            );

            // Safety: We're only replacing the entry at this index
            // This is safe because we're not holding any references to it
            unsafe {
                let ptr = self.entries.as_ptr().add(index) as *mut TTEntry;
                *ptr = new_entry;
            }

            self.stores.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Increment the age (generation) - should be called at the start of each search
    pub fn new_generation(&self) {
        self.current_age.fetch_add(1, Ordering::Relaxed);
    }

    /// Clear the transposition table
    pub fn clear(&self) {
        for i in 0..self.num_entries {
            unsafe {
                let ptr = self.entries.as_ptr().add(i) as *mut TTEntry;
                *ptr = TTEntry::empty();
            }
        }

        // Reset age
        self.current_age.store(0, Ordering::Relaxed);
    }

    /// Get the size of the table in bytes
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Get the number of entries
    pub fn num_entries(&self) -> usize {
        self.num_entries
    }

    /// Get the hashfull percentage (entries filled)
    pub fn hashfull(&self) -> u32 {
        let sample_size = 1000.min(self.num_entries);
        let mut filled = 0;

        for i in 0..sample_size {
            if !self.entries[i].is_empty() {
                filled += 1;
            }
        }

        (filled as u32 * 1000) / (sample_size as u32)
    }

    /// Get statistics about the transposition table
    pub fn stats(&self) -> TTStats {
        TTStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            stores: self.stores.load(Ordering::Relaxed),
            collisions: self.collisions.load(Ordering::Relaxed),
            hashfull: self.hashfull(),
        }
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.stores.store(0, Ordering::Relaxed);
        self.collisions.store(0, Ordering::Relaxed);
    }
}

/// Transposition table statistics
#[derive(Debug, Clone, Copy)]
pub struct TTStats {
    pub hits: u64,
    pub misses: u64,
    pub stores: u64,
    pub collisions: u64,
    pub hashfull: u32,
}

impl TTStats {
    /// Calculate the hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64) / (total as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tt_entry() {
        let entry = TTEntry::new(12345, Move::null(), 100, 5, TTFlag::Exact, 0);
        assert_eq!(entry.hash, 12345);
        assert_eq!(entry.score, 100);
        assert_eq!(entry.depth, 5);
        assert_eq!(entry.flag, TTFlag::Exact);
        assert!(!entry.is_empty());
    }

    #[test]
    fn test_tt_creation() {
        let tt = TranspositionTable::new(1); // 1 MB
        assert!(tt.num_entries() > 0);
        assert!(tt.size_bytes() <= 1024 * 1024);
    }

    #[test]
    fn test_tt_store_and_probe() {
        let tt = TranspositionTable::new(1);
        let hash = 0x123456789ABCDEF0;

        // Store an entry
        tt.store(hash, Move::null(), 100, 5, TTFlag::Exact);

        // Probe for it
        let entry = tt.probe(hash);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().score, 100);

        // Probe for different hash
        assert!(tt.probe(0xFEDCBA9876543210).is_none());
    }

    #[test]
    fn test_tt_replacement() {
        let tt = TranspositionTable::new(1);
        let hash1 = 0x1111111111111111;
        let hash2 = 0x2222222222222222;

        // Store entry with depth 5
        tt.store(hash1, Move::null(), 100, 5, TTFlag::Exact);
        assert_eq!(tt.probe(hash1).unwrap().depth, 5);

        // Store entry with higher depth at same index (if collision)
        tt.store(hash2, Move::null(), 200, 10, TTFlag::Exact);

        // Either hash1 is replaced or hash2 is stored
        // This test just verifies replacement doesn't crash
    }

    #[test]
    fn test_tt_stats() {
        let tt = TranspositionTable::new(1);
        let hash = 0x123456789ABCDEF0;

        tt.store(hash, Move::null(), 100, 5, TTFlag::Exact);
        tt.probe(hash);
        tt.probe(0xFFFFFFFFFFFFFFFF);

        let stats = tt.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.stores, 1);
        assert!(stats.hit_rate() > 0.0);
    }
}
