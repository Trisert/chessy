use crate::utils::{File, Rank, Square};

/// Bitboard type - wrapper around u64 for efficient bit operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bitboard(pub u64);

impl Bitboard {
    /// Empty bitboard
    pub const EMPTY: Bitboard = Bitboard(0);

    /// Full bitboard (all squares)
    pub const FULL: Bitboard = Bitboard(0xFFFFFFFFFFFFFFFF);

    /// Create a new bitboard from a u64
    #[inline]
    pub const fn new(bb: u64) -> Self {
        Bitboard(bb)
    }

    /// Create a bitboard from a square
    #[inline]
    pub const fn from_square(sq: Square) -> Self {
        Bitboard(1u64 << sq)
    }

    /// Get the underlying u64 value
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Set a bit (square)
    #[inline]
    pub fn set(&mut self, sq: Square) {
        self.0 |= 1u64 << sq;
    }

    /// Clear a bit (square)
    #[inline]
    pub fn clear(&mut self, sq: Square) {
        self.0 &= !(1u64 << sq);
    }

    /// Check if a bit (square) is set
    #[inline]
    pub const fn get(self, sq: Square) -> bool {
        (self.0 & (1u64 << sq)) != 0
    }

    /// Flip a bit (square)
    #[inline]
    pub fn flip(&mut self, sq: Square) {
        self.0 ^= 1u64 << sq;
    }

    /// Count the number of set bits (population count)
    #[inline]
    pub fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Get the index of the least significant set bit
    /// Returns None if the bitboard is empty
    #[inline]
    pub fn lsb(self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(self.0.trailing_zeros() as Square)
        }
    }

    /// Get the index of the most significant set bit
    /// Returns None if the bitboard is empty
    #[inline]
    pub fn msb(self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some((63 - self.0.leading_zeros()) as Square)
        }
    }

    /// Pop and return the least significant set bit
    /// Returns None if the bitboard is empty
    #[inline]
    pub fn pop_lsb(&mut self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            let lsb = self.lsb().unwrap();
            self.0 &= self.0 - 1; // Clear the least significant set bit
            Some(lsb)
        }
    }

    /// Pop and return the most significant set bit
    /// Returns None if the bitboard is empty
    #[inline]
    pub fn pop_msb(&mut self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            let msb = self.msb().unwrap();
            self.0 &= !(1u64 << msb);
            Some(msb)
        }
    }

    /// Check if the bitboard is empty
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Check if the bitboard has more than one bit set
    #[inline]
    pub fn multiple(self) -> bool {
        (self.0 & (self.0 - 1)) != 0
    }

    /// Shift the bitboard by a given number of squares
    /// Positive shifts up (towards higher ranks), negative shifts down
    #[inline]
    pub const fn shift(self, amount: i32) -> Self {
        if amount >= 0 {
            Bitboard(self.0 << amount)
        } else {
            Bitboard(self.0 >> (-amount))
        }
    }

    /// Get all squares in the bitboard as a vector
    #[inline]
    pub fn squares(self) -> Vec<Square> {
        let mut squares = Vec::new();
        let mut bb = self;
        while let Some(sq) = bb.pop_lsb() {
            squares.push(sq);
        }
        squares
    }

    /// Bitwise AND
    #[inline]
    pub const fn and(self, other: Self) -> Self {
        Bitboard(self.0 & other.0)
    }

    /// Bitwise OR
    #[inline]
    pub const fn or(self, other: Self) -> Self {
        Bitboard(self.0 | other.0)
    }

    /// Bitwise XOR
    #[inline]
    pub const fn xor(self, other: Self) -> Self {
        Bitboard(self.0 ^ other.0)
    }

    /// Bitwise NOT
    #[inline]
    pub const fn not(self) -> Self {
        Bitboard(!self.0)
    }

    /// Check if two bitboards have any overlapping bits
    #[inline]
    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    /// Rank mask for a given rank
    #[inline]
    pub const fn rank_mask(rank: Rank) -> Self {
        Bitboard(0xFF << (rank * 8))
    }

    /// File mask for a given file
    #[inline]
    pub const fn file_mask(file: File) -> Self {
        Bitboard(0x0101010101010101 << file)
    }

    /// Create a bitboard from a slice of squares
    #[inline]
    pub fn from_squares(squares: &[Square]) -> Self {
        let mut bb = Bitboard::EMPTY;
        for &sq in squares {
            bb.set(sq);
        }
        bb
    }

    /// Get a specific rank's bitboard
    #[inline]
    pub fn rank(self, rank: Rank) -> Self {
        self.and(Bitboard::rank_mask(rank))
    }

    /// Get a specific file's bitboard
    #[inline]
    pub fn file(self, file: File) -> Self {
        self.and(Bitboard::file_mask(file))
    }

    /// Convert to a string representation (for debugging)
    pub fn to_string(self) -> String {
        let mut result = String::new();
        for rank in (0..8).rev() {
            for file in 0..8 {
                let sq = rank * 8 + file;
                if self.get(sq) {
                    result.push('1');
                } else {
                    result.push('.');
                }
            }
            result.push('\n');
        }
        result
    }

    /// Convert to a visual representation with piece symbols
    pub fn to_visual(self, piece: char) -> String {
        let mut result = String::new();
        for rank in (0..8).rev() {
            for file in 0..8 {
                let sq = rank * 8 + file;
                if self.get(sq) {
                    result.push(piece);
                } else {
                    result.push('.');
                }
                result.push(' ');
            }
            result.push('\n');
        }
        result
    }
}

impl Default for Bitboard {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl std::ops::BitAnd for Bitboard {
    type Output = Self;

    fn bitand(self, other: Self) -> Self {
        self.and(other)
    }
}

impl std::ops::BitAndAssign for Bitboard {
    fn bitand_assign(&mut self, other: Self) {
        self.0 &= other.0;
    }
}

impl std::ops::BitOr for Bitboard {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        self.or(other)
    }
}

impl std::ops::BitOrAssign for Bitboard {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

impl std::ops::BitXor for Bitboard {
    type Output = Self;

    fn bitxor(self, other: Self) -> Self {
        self.xor(other)
    }
}

impl std::ops::BitXorAssign for Bitboard {
    fn bitxor_assign(&mut self, other: Self) {
        self.0 ^= other.0;
    }
}

impl std::ops::Not for Bitboard {
    type Output = Self;

    fn not(self) -> Self {
        self.not()
    }
}

impl std::ops::Shl<u32> for Bitboard {
    type Output = Self;

    fn shl(self, rhs: u32) -> Self {
        Bitboard(self.0 << rhs)
    }
}

impl std::ops::Shr<u32> for Bitboard {
    type Output = Self;

    fn shr(self, rhs: u32) -> Self {
        Bitboard(self.0 >> rhs)
    }
}

/// Direction masks for sliding pieces
pub const NORTH: u64 = 0x0101010101010100;
pub const EAST: u64 = 0x00000000000000FE;
pub const SOUTH: u64 = 0x0080808080808080;
pub const WEST: u64 = 0x000000000000007F;

pub const NORTH_EAST: u64 = 0x0102040810204000;
pub const SOUTH_EAST: u64 = 0x0040201008040200;
pub const SOUTH_WEST: u64 = 0x0002040810204080;
pub const NORTH_WEST: u64 = 0x0008040201000000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitboard_creation() {
        let bb = Bitboard::EMPTY;
        assert_eq!(bb.as_u64(), 0);

        let bb = Bitboard::FULL;
        assert_eq!(bb.as_u64(), 0xFFFFFFFFFFFFFFFF);

        let bb = Bitboard::from_square(0); // A1
        assert_eq!(bb.as_u64(), 1);

        let bb = Bitboard::from_square(63); // H8
        assert_eq!(bb.as_u64(), 0x8000000000000000);
    }

    #[test]
    fn test_bitboard_set_clear() {
        let mut bb = Bitboard::EMPTY;
        assert!(!bb.get(0));

        bb.set(0);
        assert!(bb.get(0));
        assert_eq!(bb.as_u64(), 1);

        bb.clear(0);
        assert!(!bb.get(0));
        assert_eq!(bb.as_u64(), 0);
    }

    #[test]
    fn test_bitboard_lsb_msb() {
        let bb = Bitboard::from_square(28) | Bitboard::from_square(35);

        assert_eq!(bb.lsb(), Some(28));
        assert_eq!(bb.msb(), Some(35));
    }

    #[test]
    fn test_bitboard_pop_lsb() {
        let mut bb = Bitboard::from_square(10) | Bitboard::from_square(20) | Bitboard::from_square(30);

        assert_eq!(bb.pop_lsb(), Some(10));
        assert_eq!(bb.pop_lsb(), Some(20));
        assert_eq!(bb.pop_lsb(), Some(30));
        assert_eq!(bb.pop_lsb(), None);
    }

    #[test]
    fn test_bitboard_count() {
        let bb = Bitboard::EMPTY;
        assert_eq!(bb.count(), 0);

        let bb = Bitboard::from_square(0) | Bitboard::from_square(1) | Bitboard::from_square(2);
        assert_eq!(bb.count(), 3);

        let bb = Bitboard::FULL;
        assert_eq!(bb.count(), 64);
    }

    #[test]
    fn test_bitboard_operations() {
        let bb1 = Bitboard::from_square(0) | Bitboard::from_square(1);
        let bb2 = Bitboard::from_square(1) | Bitboard::from_square(2);

        let and_bb = bb1 & bb2;
        assert_eq!(and_bb, Bitboard::from_square(1));

        let or_bb = bb1 | bb2;
        assert_eq!(or_bb, Bitboard::from_square(0) | Bitboard::from_square(1) | Bitboard::from_square(2));

        let xor_bb = bb1 ^ bb2;
        assert_eq!(xor_bb, Bitboard::from_square(0) | Bitboard::from_square(2));
    }

    #[test]
    fn test_bitboard_rank_file_mask() {
        let rank_1 = Bitboard::rank_mask(0);
        assert_eq!(rank_1.as_u64(), 0xFF);

        let rank_8 = Bitboard::rank_mask(7);
        assert_eq!(rank_8.as_u64(), 0xFF00000000000000);

        let file_a = Bitboard::file_mask(0);
        assert_eq!(file_a.as_u64(), 0x0101010101010101);

        let file_h = Bitboard::file_mask(7);
        assert_eq!(file_h.as_u64(), 0x8080808080808080);
    }

    #[test]
    fn test_bitboard_shift() {
        let bb = Bitboard::from_square(0); // A1

        let shifted = bb.shift(8); // Shift up one rank
        assert_eq!(shifted, Bitboard::from_square(8)); // A2

        let shifted = bb.shift(1); // Shift right one file
        assert_eq!(shifted, Bitboard::from_square(1)); // B1
    }

    #[test]
    fn test_bitboard_squares() {
        let bb = Bitboard::from_square(0) | Bitboard::from_square(28) | Bitboard::from_square(63);
        let squares = bb.squares();

        assert_eq!(squares.len(), 3);
        assert!(squares.contains(&0));
        assert!(squares.contains(&28));
        assert!(squares.contains(&63));
    }

    #[test]
    fn test_bitboard_to_string() {
        let bb = Bitboard::from_square(0);
        let s = bb.to_string();

        // Square 0 (A1) is on rank 0, which is the last row printed
        assert!(s.contains("1"));
        assert!(s.ends_with("1.......\n"));
    }
}
