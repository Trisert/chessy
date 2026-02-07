use crate::r#move::Move;
use std::sync::atomic::{AtomicU64, Ordering};

/// Transposition table entry flag (bound type)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TTFlag {
    /// Exact score (PV node)
    Exact = 0,
    /// Lower bound (beta cutoff - score >= actual)
    Lower = 1,
    /// Upper bound (alpha cutoff - score <= actual)
    Upper = 2,
}

/// Compact transposition table entry (16 bytes)
///
/// Layout:
/// - hash64: u64 (full 64-bit hash for zero collisions)
/// - score: i32 (32-bit score for better range)
/// - move: Move (u16, 2 bytes)
/// - depth: u8
/// - flag: u8
/// - age: u8
/// - padding: u8
#[derive(Debug, Clone, Copy)]
pub struct TTEntry {
    /// Full 64-bit hash for perfect collision detection
    hash64: u64,
    /// Score from the search
    score: i32,
    /// Best move found
    best_move: Move,
    /// Depth of the search
    depth: u8,
    /// Entry type (bounds)
    flag: TTFlag,
    /// Age generation counter
    age: u8,
    _padding: u8,
}

impl TTEntry {
    /// Create a new TT entry
    #[inline]
    pub fn new(hash64: u64, best_move: Move, score: i32, depth: u8, flag: TTFlag, age: u8) -> Self {
        TTEntry {
            hash64,
            score,
            best_move,
            depth,
            flag,
            age,
            _padding: 0,
        }
    }

    /// Create an empty entry
    #[inline]
    pub const fn empty() -> Self {
        TTEntry {
            hash64: 0,
            score: 0,
            best_move: Move::null(),
            depth: 0,
            flag: TTFlag::Exact,
            age: 0,
            _padding: 0,
        }
    }

    /// Get depth
    #[inline]
    pub const fn depth(&self) -> u8 {
        self.depth
    }

    /// Get flag
    #[inline]
    pub const fn flag(&self) -> TTFlag {
        self.flag
    }

    /// Get the hash signature
    #[inline]
    pub const fn hash64(&self) -> u64 {
        self.hash64
    }

    /// Get the score
    #[inline]
    pub const fn score(&self) -> i32 {
        self.score
    }

    /// Get the best move
    #[inline]
    pub const fn best_move(&self) -> Move {
        self.best_move
    }

    /// Get the age
    #[inline]
    pub const fn get_age(&self) -> u8 {
        self.age
    }

    /// Check if this entry is empty
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.hash64 == 0
    }

    /// Check if this entry matches the given hash
    #[inline]
    pub fn matches(&self, hash: u64) -> bool {
        self.hash64 == hash
    }
}

/// Cluster of transposition table entries (3 entries = 48 bytes)
///
/// Clustering improves cache efficiency by storing multiple entries.
#[derive(Debug, Clone, Copy)]
pub struct TTCluster {
    entries: [TTEntry; 3],
}

impl TTCluster {
    /// Create an empty cluster
    #[inline]
    pub const fn new() -> Self {
        TTCluster {
            entries: [TTEntry::empty(), TTEntry::empty(), TTEntry::empty()],
        }
    }

    /// Find the best matching entry in this cluster
    #[inline]
    pub fn find(&self, hash: u64) -> Option<TTEntry> {
        for i in 0..3 {
            let entry = self.entries[i];
            if !entry.is_empty() && entry.hash64 == hash {
                return Some(entry);
            }
        }

        None
    }

    /// Get the index of the entry to replace
    /// Replacement strategy (in order of preference):
    /// 1. Empty entry
    /// 2. Entry from old generation
    /// 3. Shallower entry
    /// 4. First entry (as fallback)
    #[inline]
    pub fn replacement_index(&self, new_depth: u8, new_age: u8) -> usize {
        // First, try to find an empty entry
        for i in 0..3 {
            if self.entries[i].is_empty() {
                return i;
            }
        }

        // Find the entry with lowest score for replacement
        let mut best_idx = 0;
        let mut best_score = self.replacement_score(0, new_depth, new_age);

        for i in 1..3 {
            let score = self.replacement_score(i, new_depth, new_age);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Calculate replacement score for an entry (higher = better to replace)
    #[inline]
    fn replacement_score(&self, idx: usize, new_depth: u8, new_age: u8) -> i32 {
        let entry = self.entries[idx];
        let age_diff = (new_age as i16 - entry.get_age() as i16).abs();

        // Prefer replacing:
        // 1. Old entries (large age difference)
        // 2. Shallower entries
        // 3. Lower-depth entries
        age_diff as i32 * 100 + (new_depth as i32 - entry.depth() as i32) * 10
    }

    /// Store an entry in the cluster
    #[inline]
    pub fn store(&mut self, hash: u64, best_move: Move, score: i32, depth: u8, flag: TTFlag, age: u8) {
        let idx = self.replacement_index(depth, age);

        self.entries[idx] = TTEntry::new(hash, best_move, score, depth, flag, age);
    }
}

/// Transposition table with cluster-based storage
pub struct TranspositionTable {
    /// Table clusters (boxed slice for better memory locality)
    clusters: Box<[TTCluster]>,
    /// Size of the table in bytes
    size_bytes: usize,
    /// Requested size in megabytes (for accurate reporting)
    requested_size_mb: usize,
    /// Number of clusters in the table
    num_clusters: usize,
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
        let cluster_size = std::mem::size_of::<TTCluster>();

        // Calculate number of clusters (must be power of 2)
        let mut num_clusters = size_bytes / cluster_size;
        // Round down to power of 2
        if num_clusters > 1 {
            let leading_zeros = num_clusters.leading_zeros() as usize;
            num_clusters = 1 << (64 - leading_zeros);
            // If we rounded up, go back down
            if num_clusters > (size_bytes / cluster_size) {
                num_clusters >>= 1;
            }
        } else {
            num_clusters = 1;
        }

        let actual_size = num_clusters * cluster_size;

        // Create boxed slice with all clusters empty
        let clusters: Vec<TTCluster> = vec![TTCluster::new(); num_clusters];
        let clusters = clusters.into_boxed_slice();

        let index_mask = num_clusters - 1;

        TranspositionTable {
            clusters,
            size_bytes: actual_size,
            requested_size_mb: size_mb,
            num_clusters,
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
        self.requested_size_mb
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
        let cluster = self.clusters[index];

        if let Some(entry) = cluster.find(hash) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry)
        } else {
            // Check if cluster is occupied (collision)
            if !cluster.entries[0].is_empty()
                || !cluster.entries[1].is_empty()
                || !cluster.entries[2].is_empty() {
                self.collisions.fetch_add(1, Ordering::Relaxed);
            }
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Store an entry in the transposition table
    #[inline]
    pub fn store(&mut self, hash: u64, best_move: Move, score: i32, depth: u8, flag: TTFlag) {
        let index = self.get_index(hash);
        let age = self.current_age.load(Ordering::Relaxed) as u8;

        unsafe {
            let ptr = self.clusters.as_mut_ptr().add(index);
            (*ptr).store(hash, best_move, score, depth, flag, age);
        }

        self.stores.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the age (generation) - should be called at the start of each search
    pub fn new_generation(&self) {
        self.current_age.fetch_add(1, Ordering::Relaxed);
    }

    /// Clear the transposition table
    pub fn clear(&self) {
        for i in 0..self.num_clusters {
            unsafe {
                let ptr = self.clusters.as_ptr().add(i) as *mut TTCluster;
                *ptr = TTCluster::new();
            }
        }

        // Reset age
        self.current_age.store(0, Ordering::Relaxed);
    }

    /// Get the size of the table in bytes
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Get the number of clusters
    pub fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Get the hashfull percentage (entries filled)
    pub fn hashfull(&self) -> u32 {
        let sample_size = 1000.min(self.num_clusters);
        let mut filled = 0;

        for i in 0..sample_size {
            let cluster = self.clusters[i];
            for j in 0..3 {
                if !cluster.entries[j].is_empty() {
                    filled += 1;
                }
            }
        }

        // filled is out of 3 * sample_size entries
        (filled as u32 * 1000) / ((3 * sample_size) as u32)
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
        let entry = TTEntry::new(0x1234, Move::null(), 100, 5, TTFlag::Exact, 0);
        assert_eq!(entry.hash64(), 0x1234);
        assert_eq!(entry.score(), 100);
        assert_eq!(entry.depth(), 5);
        assert_eq!(entry.flag(), TTFlag::Exact);
        assert!(!entry.is_empty());
    }

    #[test]
    fn test_tt_cluster() {
        let mut cluster = TTCluster::new();

        // Store and find
        cluster.store(0x1234567890ABCDEF, Move::null(), 100, 5, TTFlag::Exact, 0);
        assert!(cluster.find(0x1234567890ABCDEF).is_some());
        assert!(cluster.find(0xFEDCBA0987654321).is_none());
    }

    #[test]
    fn test_tt_creation() {
        let tt = TranspositionTable::new(1); // 1 MB
        assert!(tt.num_clusters() > 0);
        assert!(tt.size_bytes() <= 1024 * 1024);
    }

    #[test]
    fn test_tt_store_and_probe() {
        let mut tt = TranspositionTable::new(1);
        let hash = 0x123456789ABCDEF0;

        // Store an entry
        tt.store(hash, Move::null(), 100, 5, TTFlag::Exact);

        // Probe for it
        let entry = tt.probe(hash);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().score(), 100);

        // Probe for different hash
        assert!(tt.probe(0xFEDCBA9876543210).is_none());
    }

    #[test]
    fn test_tt_cluster_replacement() {
        let mut cluster = TTCluster::new();

        // Fill cluster with 3 entries
        cluster.store(0x1111111111111111, Move::null(), 100, 5, TTFlag::Exact, 0);
        cluster.store(0x2222222222222222, Move::null(), 100, 5, TTFlag::Exact, 0);
        cluster.store(0x3333333333333333, Move::null(), 100, 5, TTFlag::Exact, 0);

        // All should be present
        assert!(cluster.find(0x1111111111111111).is_some());
        assert!(cluster.find(0x2222222222222222).is_some());
        assert!(cluster.find(0x3333333333333333).is_some());

        // Add 4th entry - should replace one based on depth/age
        cluster.store(0x4444444444444444, Move::null(), 100, 6, TTFlag::Exact, 0);
        assert!(cluster.find(0x4444444444444444).is_some());
    }

    #[test]
    fn test_tt_stats() {
        let mut tt = TranspositionTable::new(1);
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
