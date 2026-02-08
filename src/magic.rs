use crate::bitboard::Bitboard;
use crate::utils::Square;

/// Magic number entry for a piece type (bishop or rook)
#[derive(Debug, Clone, Copy)]
pub struct Magic {
    /// Magic number for multiplication
    pub magic: u64,
    /// Attack mask (relevant occupancy bits, excluding edges)
    pub mask: u64,
    /// Right shift amount (64 - popcount(mask))
    pub shift: u32,
    /// Offset into the attack table
    pub offset: usize,
}

/// Known-good bishop magic numbers (from magic-bits library)
/// These are just the magic multipliers - masks/shifts/offsets are computed at init
const BISHOP_MAGIC_NUMBERS: [u64; 64] = [
    0x89a1121896040240, 0x2004844802002010, 0x2068080051921000, 0x62880a0220200808,
    0x0004042004000000, 0x0100822020200011, 0xc00444222012000a, 0x0028808801216001,
    0x0400492088408100, 0x0201c401040c0084, 0x00840800910a0010, 0x0000082080240060,
    0x2000840504006000, 0x30010c4108405004, 0x1008005410080802, 0x8144042209100900,
    0x0208081020014400, 0x004800201208ca00, 0x0f18140408012008, 0x1004002802102001,
    0x0841000820080811, 0x0040200200a42008, 0x0000800054042000, 0x88010400410c9000,
    0x0520040470104290, 0x1004040051500081, 0x2002081833080021, 0x000400c00c010142,
    0x941408200c002000, 0x0658810000806011, 0x0188071040440a00, 0x4800404002011c00,
    0x0104442040404200, 0x0511080202091021, 0x0004022401120400, 0x80c0040400080120,
    0x8040010040820802, 0x0480810700020090, 0x0102008e00040242, 0x0809005202050100,
    0x8002024220104080, 0x0431008804142000, 0x0019001802081400, 0x0200014208040080,
    0x3308082008200100, 0x041010500040c020, 0x4012020c04210308, 0x208220a202004080,
    0x0111040120082000, 0x6803040141280a00, 0x2101004202410000, 0x8200000041108022,
    0x0000021082088000, 0x0002410204010040, 0x0040100400809000, 0x0822088220820214,
    0x0040808090012004, 0x00910224040218c9, 0x0402814422015008, 0x0090014004842410,
    0x0001000042304105, 0x0010008830412a00, 0x2520081090008908, 0x40102000a0a60140,
];

/// Known-good rook magic numbers (from magic-bits library)
const ROOK_MAGIC_NUMBERS: [u64; 64] = [
    0x0a8002c000108020, 0x06c00049b0002001, 0x0100200010090040, 0x2480041000800801,
    0x0280028004000800, 0x0900410008040022, 0x0280020001001080, 0x2880002041000080,
    0x00a0800040400020, 0x0004500040401000, 0x0001008002001040, 0x0003100808010040,
    0x0010084020001000, 0x0040002004800080, 0x0001000200120004, 0x0000200040100044,
    0x0080008040002080, 0x4020404000800080, 0x0001080040001000, 0x0002040088000100,
    0x0000884001000108, 0x2000820004000080, 0x0040004002008010, 0x000000c020040108,
    0x5040002080400100, 0x0200200040100040, 0x0500200040104100, 0x0800080100080080,
    0x0080040080080080, 0x0022004800040080, 0x0800040010200100, 0x0000402100004104,
    0x4000800020801000, 0x0008804000802000, 0x0020008020100080, 0x4001184018001000,
    0x0002000808004002, 0x0002002004010010, 0x0800840001000200, 0x0800410044000081,
    0x0200410040004000, 0x0000820040402000, 0x0080020010080880, 0x0004010008020008,
    0x0001100800280008, 0x0010008002008400, 0x2800020001108204, 0x0040a0100801c084,
    0x0024008040002080, 0x0080010040200100, 0x0400080080800200, 0x0140100040010800,
    0x0800800040008080, 0x1000080100020400, 0x1200040020100200, 0x0010000400200041,
    0x0010820040012082, 0x1004010040008021, 0x0008120010100804, 0x0448011000082201,
    0x1000420002008102, 0x4000100401020001, 0x0400280410008804, 0x0200240800410082,
];

/// Bishop magics array - populated at init
static mut BISHOP_MAGICS: [Magic; 64] = [Magic { magic: 0, mask: 0, shift: 0, offset: 0 }; 64];

/// Rook magics array - populated at init
static mut ROOK_MAGICS: [Magic; 64] = [Magic { magic: 0, mask: 0, shift: 0, offset: 0 }; 64];

/// Attack table for bishops (variable size based on fancy approach)
const BISHOP_TABLE_SIZE: usize = 5248;

/// Attack table for rooks
const ROOK_TABLE_SIZE: usize = 102400;

/// Combined attack table size
const ATTACK_TABLE_SIZE: usize = BISHOP_TABLE_SIZE + ROOK_TABLE_SIZE;

/// Combined attack table for both bishops and rooks
static mut ATTACK_TABLE: [u64; ATTACK_TABLE_SIZE] = [0u64; ATTACK_TABLE_SIZE];

/// Get bishop attacks for a given square and occupancy
#[inline]
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    unsafe {
        let magic = &BISHOP_MAGICS[sq as usize];
        let index = magic.index(occupied);
        Bitboard::new(ATTACK_TABLE[magic.offset + index])
    }
}

/// Get rook attacks for a given square and occupancy
#[inline]
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    unsafe {
        let magic = &ROOK_MAGICS[sq as usize];
        let index = magic.index(occupied);
        Bitboard::new(ATTACK_TABLE[magic.offset + index])
    }
}

/// Get queen attacks (bishop | rook)
#[inline]
pub fn queen_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    bishop_attacks(sq, occupied) | rook_attacks(sq, occupied)
}

impl Magic {
    /// Calculate index into attack table using magic multiplication
    #[inline]
    fn index(&self, occupied: Bitboard) -> usize {
        let masked = occupied.as_u64() & self.mask;
        ((masked.wrapping_mul(self.magic)) >> self.shift) as usize
    }
}

/// Generate bishop mask for a square (excludes edges)
fn bishop_mask(sq: u8) -> u64 {
    let mut mask = 0u64;
    let rank = (sq / 8) as i32;
    let file = (sq % 8) as i32;

    // Northeast (excluding edges)
    for i in 1..7 {
        let r = rank + i;
        let f = file + i;
        if r >= 7 || f >= 7 { break; }
        mask |= 1u64 << (r * 8 + f);
    }

    // Northwest (excluding edges)
    for i in 1..7 {
        let r = rank + i;
        let f = file - i;
        if r >= 7 || f <= 0 { break; }
        mask |= 1u64 << (r * 8 + f);
    }

    // Southeast (excluding edges)
    for i in 1..7 {
        let r = rank - i;
        let f = file + i;
        if r <= 0 || f >= 7 { break; }
        mask |= 1u64 << (r * 8 + f);
    }

    // Southwest (excluding edges)
    for i in 1..7 {
        let r = rank - i;
        let f = file - i;
        if r <= 0 || f <= 0 { break; }
        mask |= 1u64 << (r * 8 + f);
    }

    mask
}

/// Generate rook mask for a square (excludes edges)
fn rook_mask(sq: u8) -> u64 {
    let mut mask = 0u64;
    let rank = (sq / 8) as i32;
    let file = (sq % 8) as i32;

    // North (excluding edge)
    for r in (rank + 1)..7 {
        mask |= 1u64 << (r * 8 + file);
    }

    // South (excluding edge)
    for r in 1..rank {
        mask |= 1u64 << (r * 8 + file);
    }

    // East (excluding edge)
    for f in (file + 1)..7 {
        mask |= 1u64 << (rank * 8 + f);
    }

    // West (excluding edge)
    for f in 1..file {
        mask |= 1u64 << (rank * 8 + f);
    }

    mask
}

/// Generate bishop attacks for a square with blocking pieces
fn generate_bishop_attacks(sq: u8, occupied: u64) -> u64 {
    let mut attacks = 0u64;
    let rank = (sq / 8) as i32;
    let file = (sq % 8) as i32;

    // Northeast
    for i in 1..8 {
        let r = rank + i;
        let f = file + i;
        if r > 7 || f > 7 { break; }
        let bit = 1u64 << (r * 8 + f);
        attacks |= bit;
        if occupied & bit != 0 { break; }
    }

    // Northwest
    for i in 1..8 {
        let r = rank + i;
        let f = file - i;
        if r > 7 || f < 0 { break; }
        let bit = 1u64 << (r * 8 + f);
        attacks |= bit;
        if occupied & bit != 0 { break; }
    }

    // Southeast
    for i in 1..8 {
        let r = rank - i;
        let f = file + i;
        if r < 0 || f > 7 { break; }
        let bit = 1u64 << (r * 8 + f);
        attacks |= bit;
        if occupied & bit != 0 { break; }
    }

    // Southwest
    for i in 1..8 {
        let r = rank - i;
        let f = file - i;
        if r < 0 || f < 0 { break; }
        let bit = 1u64 << (r * 8 + f);
        attacks |= bit;
        if occupied & bit != 0 { break; }
    }

    attacks
}

/// Generate rook attacks for a square with blocking pieces
fn generate_rook_attacks(sq: u8, occupied: u64) -> u64 {
    let mut attacks = 0u64;
    let rank = (sq / 8) as i32;
    let file = (sq % 8) as i32;

    // North
    for r in (rank + 1)..8 {
        let bit = 1u64 << (r * 8 + file);
        attacks |= bit;
        if occupied & bit != 0 { break; }
    }

    // South
    for r in (0..rank).rev() {
        let bit = 1u64 << (r * 8 + file);
        attacks |= bit;
        if occupied & bit != 0 { break; }
    }

    // East
    for f in (file + 1)..8 {
        let bit = 1u64 << (rank * 8 + f);
        attacks |= bit;
        if occupied & bit != 0 { break; }
    }

    // West
    for f in (0..file).rev() {
        let bit = 1u64 << (rank * 8 + f);
        attacks |= bit;
        if occupied & bit != 0 { break; }
    }

    attacks
}

/// Count set bits in a u64
fn popcount(n: u64) -> u32 {
    n.count_ones()
}

/// Initialize the magic bitboard attack tables
pub fn init_attack_table() {
    unsafe {
        let mut offset = 0usize;

        // Initialize bishop magics
        for sq in 0..64u8 {
            let mask = bishop_mask(sq);
            let shift = 64 - popcount(mask);
            let table_size = 1usize << popcount(mask);

            BISHOP_MAGICS[sq as usize] = Magic {
                magic: BISHOP_MAGIC_NUMBERS[sq as usize],
                mask,
                shift,
                offset,
            };

            // Enumerate all subsets of mask using carry-rippler
            let mut occ = 0u64;
            loop {
                let attacks = generate_bishop_attacks(sq, occ);
                let index = ((occ.wrapping_mul(BISHOP_MAGIC_NUMBERS[sq as usize])) >> shift) as usize;
                ATTACK_TABLE[offset + index] = attacks;

                occ = occ.wrapping_sub(mask) & mask;
                if occ == 0 { break; }
            }

            offset += table_size;
        }

        // Initialize rook magics
        for sq in 0..64u8 {
            let mask = rook_mask(sq);
            let shift = 64 - popcount(mask);
            let table_size = 1usize << popcount(mask);

            ROOK_MAGICS[sq as usize] = Magic {
                magic: ROOK_MAGIC_NUMBERS[sq as usize],
                mask,
                shift,
                offset,
            };

            // Enumerate all subsets of mask using carry-rippler
            let mut occ = 0u64;
            loop {
                let attacks = generate_rook_attacks(sq, occ);
                let index = ((occ.wrapping_mul(ROOK_MAGIC_NUMBERS[sq as usize])) >> shift) as usize;
                ATTACK_TABLE[offset + index] = attacks;

                occ = occ.wrapping_sub(mask) & mask;
                if occ == 0 { break; }
            }

            offset += table_size;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_magic() {
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            init_attack_table();
        });
    }

    #[test]
    fn test_bishop_attacks_center() {
        init_magic();

        let occupied = Bitboard::EMPTY;
        let attacks = bishop_attacks(28, occupied); // e4

        // e4 should have 13 squares on diagonals (from center)
        assert_eq!(attacks.count(), 13);
    }

    #[test]
    fn test_bishop_attacks_blocked() {
        init_magic();

        // Place a piece on f5 (blocks one diagonal)
        let occupied = Bitboard::from_square(37);
        let attacks = bishop_attacks(28, occupied); // e4

        // Should have fewer squares due to blocking
        assert!(attacks.count() < 13);
        // f5 should still be included (we can capture)
        assert!(attacks.get(37));
    }

    #[test]
    fn test_rook_attacks_center() {
        init_magic();

        let occupied = Bitboard::EMPTY;
        let attacks = rook_attacks(28, occupied); // e4

        // e4 should have 14 squares on ranks and files (from center)
        assert_eq!(attacks.count(), 14);
    }

    #[test]
    fn test_queen_attacks() {
        init_magic();

        let occupied = Bitboard::EMPTY;
        let attacks = queen_attacks(28, occupied); // e4

        // Queen attacks = bishop + rook
        assert_eq!(attacks.count(), 27); // 13 + 14
    }

    #[test]
    fn test_bishop_corner() {
        init_magic();

        let occupied = Bitboard::EMPTY;
        let attacks = bishop_attacks(0, occupied); // a1

        // a1 bishop should have 7 squares on diagonal
        assert_eq!(attacks.count(), 7);
    }

    #[test]
    fn test_rook_corner() {
        init_magic();

        let occupied = Bitboard::EMPTY;
        let attacks = rook_attacks(0, occupied); // a1

        // a1 rook should have 14 squares
        assert_eq!(attacks.count(), 14);
    }
}
