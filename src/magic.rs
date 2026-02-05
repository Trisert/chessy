use crate::bitboard::Bitboard;
use crate::utils::Square;

/// Magic number for a piece type (bishop or rook)
#[derive(Debug, Clone, Copy)]
pub struct Magic {
    /// Magic number
    pub magic: u64,
    /// Attack mask (relevant occupancy bits)
    pub mask: u64,
    /// Shift right amount
    pub shift: u32,
    /// Offset into the attack table
    pub offset: usize,
}

/// Bishop magic numbers (from chessprogramming.org)
const BISHOP_MAGICS: [Magic; 64] = [
    Magic {
        magic: 0x4004080808000,
        mask: 0x40201008040200,
        shift: 6,
        offset: 0,
    },
    Magic {
        magic: 0x4004088180000,
        mask: 0x40201008040200,
        shift: 6,
        offset: 64,
    },
    Magic {
        magic: 0x20002010408,
        mask: 0x40201008040000,
        shift: 5,
        offset: 128,
    },
    Magic {
        magic: 0x4180080804000,
        mask: 0x402010083C0000,
        shift: 6,
        offset: 192,
    },
    Magic {
        magic: 0x4080008080000,
        mask: 0x4020153C3C0000,
        shift: 6,
        offset: 256,
    },
    Magic {
        magic: 0x2000200020410,
        mask: 0x4020103C3C0000,
        shift: 6,
        offset: 320,
    },
    Magic {
        magic: 0x110402001000,
        mask: 0x40201008040000,
        shift: 5,
        offset: 384,
    },
    Magic {
        magic: 0x2000802080000,
        mask: 0x40201008040200,
        shift: 6,
        offset: 448,
    },
    Magic {
        magic: 0x2200008080000,
        mask: 0x40201008040200,
        shift: 6,
        offset: 512,
    },
    Magic {
        magic: 0x40200808000,
        mask: 0x142002010840200,
        shift: 6,
        offset: 576,
    },
    Magic {
        magic: 0x1008000800,
        mask: 0x140200100040200,
        shift: 5,
        offset: 640,
    },
    Magic {
        magic: 0x40008080800,
        mask: 0x1C2000803C0000,
        shift: 6,
        offset: 704,
    },
    Magic {
        magic: 0x1100202000000,
        mask: 0x1C38003C3C0000,
        shift: 6,
        offset: 768,
    },
    Magic {
        magic: 0x4001102000800,
        mask: 0x1C20003C3C0000,
        shift: 6,
        offset: 832,
    },
    Magic {
        magic: 0x100200040800,
        mask: 0x140200100040200,
        shift: 5,
        offset: 896,
    },
    Magic {
        magic: 0x1000104002000,
        mask: 0x142002010840200,
        shift: 6,
        offset: 960,
    },
    Magic {
        magic: 0x40000808000,
        mask: 0x40201008040200,
        shift: 6,
        offset: 1024,
    },
    Magic {
        magic: 0x20008081000,
        mask: 0x40201008040200,
        shift: 6,
        offset: 1088,
    },
    Magic {
        magic: 0x20011040000,
        mask: 0x140200100040200,
        shift: 5,
        offset: 1152,
    },
    Magic {
        magic: 0x208088000800,
        mask: 0x1C2000803C0000,
        shift: 6,
        offset: 1216,
    },
    Magic {
        magic: 0x804000810000,
        mask: 0x1C38003C3C0000,
        shift: 6,
        offset: 1280,
    },
    Magic {
        magic: 0x200204000,
        mask: 0x1C20003C3C0000,
        shift: 6,
        offset: 1344,
    },
    Magic {
        magic: 0x40020081000,
        mask: 0x140200100040200,
        shift: 5,
        offset: 1408,
    },
    Magic {
        magic: 0x4020080800,
        mask: 0x142002010840200,
        shift: 6,
        offset: 1472,
    },
    Magic {
        magic: 0x1000802008000,
        mask: 0x40201008040200,
        shift: 6,
        offset: 1536,
    },
    Magic {
        magic: 0x1040004004000,
        mask: 0x40201008040200,
        shift: 6,
        offset: 1600,
    },
    Magic {
        magic: 0x1000800800,
        mask: 0x140200100040200,
        shift: 5,
        offset: 1664,
    },
    Magic {
        magic: 0x1002000800,
        mask: 0x1C2000803C0000,
        shift: 6,
        offset: 1728,
    },
    Magic {
        magic: 0x202000800800,
        mask: 0x1C38003C3C0000,
        shift: 6,
        offset: 1792,
    },
    Magic {
        magic: 0x400802001000,
        mask: 0x1C20003C3C0000,
        shift: 6,
        offset: 1856,
    },
    Magic {
        magic: 0x20020001000,
        mask: 0x140200100040200,
        shift: 5,
        offset: 1920,
    },
    Magic {
        magic: 0x4000802008000,
        mask: 0x142002010840200,
        shift: 6,
        offset: 1984,
    },
    Magic {
        magic: 0x8102000810000,
        mask: 0x40201008040200,
        shift: 6,
        offset: 2048,
    },
    Magic {
        magic: 0x400810008000,
        mask: 0x40201008040200,
        shift: 6,
        offset: 2112,
    },
    Magic {
        magic: 0x10020004000,
        mask: 0x140200100040200,
        shift: 5,
        offset: 2176,
    },
    Magic {
        magic: 0x822000800200,
        mask: 0x1C2000803C0000,
        shift: 6,
        offset: 2240,
    },
    Magic {
        magic: 0x20000080800,
        mask: 0x1C38003C3C0000,
        shift: 6,
        offset: 2304,
    },
    Magic {
        magic: 0x8002000800,
        mask: 0x1C20003C3C0000,
        shift: 6,
        offset: 2368,
    },
    Magic {
        magic: 0x2004000800,
        mask: 0x140200100040200,
        shift: 5,
        offset: 2432,
    },
    Magic {
        magic: 0x80020008,
        mask: 0x142002010840200,
        shift: 6,
        offset: 2496,
    },
    Magic {
        magic: 0x200802008000,
        mask: 0x40201008040200,
        shift: 6,
        offset: 2560,
    },
    Magic {
        magic: 0x200808008000,
        mask: 0x40201008040200,
        shift: 6,
        offset: 2624,
    },
    Magic {
        magic: 0x20020004000,
        mask: 0x140200100040200,
        shift: 5,
        offset: 2688,
    },
    Magic {
        magic: 0x400810808000,
        mask: 0x1C2000803C0000,
        shift: 6,
        offset: 2752,
    },
    Magic {
        magic: 0x100020200000,
        mask: 0x1C38003C3C0000,
        shift: 6,
        offset: 2816,
    },
    Magic {
        magic: 0x400020200000,
        mask: 0x1C20003C3C0000,
        shift: 6,
        offset: 2880,
    },
    Magic {
        magic: 0x208000800,
        mask: 0x140200100040200,
        shift: 5,
        offset: 2944,
    },
    Magic {
        magic: 0x2000400800,
        mask: 0x142002010840200,
        shift: 6,
        offset: 3008,
    },
    Magic {
        magic: 0x1000080800,
        mask: 0x40201008040200,
        shift: 6,
        offset: 3072,
    },
    Magic {
        magic: 0x40008080800,
        mask: 0x40201008040200,
        shift: 6,
        offset: 3136,
    },
    Magic {
        magic: 0x100010800,
        mask: 0x140200100040200,
        shift: 5,
        offset: 3200,
    },
    Magic {
        magic: 0x200110404000,
        mask: 0x1C2000803C0000,
        shift: 6,
        offset: 3264,
    },
    Magic {
        magic: 0x82200080800,
        mask: 0x1C38003C3C0000,
        shift: 6,
        offset: 3328,
    },
    Magic {
        magic: 0x2000802000,
        mask: 0x1C20003C3C0000,
        shift: 6,
        offset: 3392,
    },
    Magic {
        magic: 0x800200400,
        mask: 0x140200100040200,
        shift: 5,
        offset: 3456,
    },
    Magic {
        magic: 0x20004021000,
        mask: 0x142002010840200,
        shift: 6,
        offset: 3520,
    },
    Magic {
        magic: 0x40000080800,
        mask: 0x40201008040200,
        shift: 6,
        offset: 3584,
    },
    Magic {
        magic: 0x40000808000,
        mask: 0x40201008040200,
        shift: 6,
        offset: 3648,
    },
    Magic {
        magic: 0x200100200,
        mask: 0x140200100040200,
        shift: 5,
        offset: 3712,
    },
    Magic {
        magic: 0x40011008000,
        mask: 0x1C2000803C0000,
        shift: 6,
        offset: 3776,
    },
    Magic {
        magic: 0x208002000800,
        mask: 0x1C38003C3C0000,
        shift: 6,
        offset: 3840,
    },
    Magic {
        magic: 0x40000200800,
        mask: 0x1C20003C3C0000,
        shift: 6,
        offset: 3904,
    },
    Magic {
        magic: 0x80001000,
        mask: 0x140200100040200,
        shift: 5,
        offset: 3968,
    },
    Magic {
        magic: 0x20021001000,
        mask: 0x142002010840200,
        shift: 6,
        offset: 4032,
    },
];

/// Rook magic numbers (from chessprogramming.org)
const ROOK_MAGICS: [Magic; 64] = [
    Magic {
        magic: 0x8002020202000,
        mask: 0x1010101010170,
        shift: 12,
        offset: 0,
    },
    Magic {
        magic: 0x4040200020080,
        mask: 0x1010101010170,
        shift: 12,
        offset: 4096,
    },
    Magic {
        magic: 0x4008081000,
        mask: 0x1010101010100,
        shift: 11,
        offset: 8192,
    },
    Magic {
        magic: 0x40100020080,
        mask: 0x1010101010100,
        shift: 11,
        offset: 12288,
    },
    Magic {
        magic: 0x80008008000,
        mask: 0x1010101010100,
        shift: 11,
        offset: 16384,
    },
    Magic {
        magic: 0x1004008000,
        mask: 0x1010101010100,
        shift: 11,
        offset: 20480,
    },
    Magic {
        magic: 0x8020104000,
        mask: 0x1010101010100,
        shift: 11,
        offset: 24576,
    },
    Magic {
        magic: 0x80000100400,
        mask: 0x1010101010170,
        shift: 12,
        offset: 28672,
    },
    Magic {
        magic: 0x400220200,
        mask: 0x1010101010170,
        shift: 12,
        offset: 32768,
    },
    Magic {
        magic: 0x800801000,
        mask: 0x10101A1010170,
        shift: 12,
        offset: 36864,
    },
    Magic {
        magic: 0x1001000,
        mask: 0x1010101010100,
        shift: 11,
        offset: 40960,
    },
    Magic {
        magic: 0x2004000200,
        mask: 0x1010101010100,
        shift: 11,
        offset: 45056,
    },
    Magic {
        magic: 0x804001000,
        mask: 0x1010101010100,
        shift: 11,
        offset: 49152,
    },
    Magic {
        magic: 0x400100200,
        mask: 0x1010101010100,
        shift: 11,
        offset: 53248,
    },
    Magic {
        magic: 0x8001000,
        mask: 0x1010101010100,
        shift: 11,
        offset: 57344,
    },
    Magic {
        magic: 0x1002080200,
        mask: 0x1010101010170,
        shift: 12,
        offset: 61440,
    },
    Magic {
        magic: 0x220080200,
        mask: 0x1010101010170,
        shift: 12,
        offset: 65536,
    },
    Magic {
        magic: 0x802001000,
        mask: 0x10101A1010170,
        shift: 12,
        offset: 69632,
    },
    Magic {
        magic: 0x400080200,
        mask: 0x1010101010100,
        shift: 11,
        offset: 73728,
    },
    Magic {
        magic: 0x2001000,
        mask: 0x1010101010100,
        shift: 11,
        offset: 77824,
    },
    Magic {
        magic: 0x802000100,
        mask: 0x1010101010100,
        shift: 11,
        offset: 81920,
    },
    Magic {
        magic: 0x100800200,
        mask: 0x1010101010100,
        shift: 11,
        offset: 86016,
    },
    Magic {
        magic: 0x400200200,
        mask: 0x1010101010100,
        shift: 11,
        offset: 90112,
    },
    Magic {
        magic: 0x2008000800,
        mask: 0x10101A1010170,
        shift: 12,
        offset: 94208,
    },
    Magic {
        magic: 0x40001001000,
        mask: 0x1010101010170,
        shift: 12,
        offset: 98304,
    },
    Magic {
        magic: 0x800040100,
        mask: 0x1010101010170,
        shift: 12,
        offset: 102400,
    },
    Magic {
        magic: 0x400100400,
        mask: 0x1010101010100,
        shift: 11,
        offset: 106496,
    },
    Magic {
        magic: 0x801001000,
        mask: 0x1010101010100,
        shift: 11,
        offset: 110592,
    },
    Magic {
        magic: 0x100801000,
        mask: 0x1010101010100,
        shift: 11,
        offset: 114688,
    },
    Magic {
        magic: 0x200200200,
        mask: 0x1010101010100,
        shift: 11,
        offset: 118784,
    },
    Magic {
        magic: 0x1000200100,
        mask: 0x10101A1010170,
        shift: 12,
        offset: 122880,
    },
    Magic {
        magic: 0x4000808000,
        mask: 0x1010101010170,
        shift: 12,
        offset: 126976,
    },
    Magic {
        magic: 0x8020802000,
        mask: 0x1010101010170,
        shift: 12,
        offset: 131072,
    },
    Magic {
        magic: 0x801000200,
        mask: 0x1010101010100,
        shift: 11,
        offset: 135168,
    },
    Magic {
        magic: 0x2001000,
        mask: 0x1010101010100,
        shift: 11,
        offset: 139264,
    },
    Magic {
        magic: 0x2000800,
        mask: 0x1010101010100,
        shift: 11,
        offset: 143360,
    },
    Magic {
        magic: 0x200100400,
        mask: 0x1010101010100,
        shift: 11,
        offset: 147456,
    },
    Magic {
        magic: 0x4004000800,
        mask: 0x10101A1010170,
        shift: 12,
        offset: 151552,
    },
    Magic {
        magic: 0x20001010000,
        mask: 0x1010101010170,
        shift: 12,
        offset: 155648,
    },
    Magic {
        magic: 0x408002002000,
        mask: 0x1010101010170,
        shift: 12,
        offset: 159744,
    },
    Magic {
        magic: 0x808002000,
        mask: 0x1010101010100,
        shift: 11,
        offset: 163840,
    },
    Magic {
        magic: 0x2001000,
        mask: 0x1010101010100,
        shift: 11,
        offset: 167936,
    },
    Magic {
        magic: 0x802000200,
        mask: 0x1010101010100,
        shift: 11,
        offset: 172032,
    },
    Magic {
        magic: 0x200080200,
        mask: 0x1010101010100,
        shift: 11,
        offset: 176128,
    },
    Magic {
        magic: 0x800801000,
        mask: 0x10101A1010170,
        shift: 12,
        offset: 180224,
    },
    Magic {
        magic: 0x2001000100,
        mask: 0x1010101010170,
        shift: 12,
        offset: 184320,
    },
    Magic {
        magic: 0x8040008000,
        mask: 0x1010101010170,
        shift: 12,
        offset: 188416,
    },
    Magic {
        magic: 0x4000200,
        mask: 0x1010101010100,
        shift: 11,
        offset: 192512,
    },
    Magic {
        magic: 0x1002000,
        mask: 0x1010101010100,
        shift: 11,
        offset: 196608,
    },
    Magic {
        magic: 0x802002000,
        mask: 0x1010101010100,
        shift: 11,
        offset: 200704,
    },
    Magic {
        magic: 0x10000800,
        mask: 0x1010101010100,
        shift: 11,
        offset: 204800,
    },
    Magic {
        magic: 0x4000002010,
        mask: 0x10101A1010170,
        shift: 12,
        offset: 208896,
    },
    Magic {
        magic: 0x8000802000,
        mask: 0x1010101010170,
        shift: 12,
        offset: 212992,
    },
    Magic {
        magic: 0x100808000,
        mask: 0x1010101010170,
        shift: 12,
        offset: 217088,
    },
    Magic {
        magic: 0x4000800,
        mask: 0x1010101010100,
        shift: 11,
        offset: 221184,
    },
    Magic {
        magic: 0x2004000,
        mask: 0x1010101010100,
        shift: 11,
        offset: 225280,
    },
    Magic {
        magic: 0x1002000,
        mask: 0x1010101010100,
        shift: 11,
        offset: 229376,
    },
    Magic {
        magic: 0x4000800,
        mask: 0x1010101010100,
        shift: 11,
        offset: 233472,
    },
    Magic {
        magic: 0x20800800,
        mask: 0x10101A1010170,
        shift: 12,
        offset: 237568,
    },
    Magic {
        magic: 0x2000800800,
        mask: 0x1010101010170,
        shift: 12,
        offset: 241664,
    },
    Magic {
        magic: 0x800200400,
        mask: 0x1010101010170,
        shift: 12,
        offset: 245760,
    },
    Magic {
        magic: 0x4008081000,
        mask: 0x1010101010170,
        shift: 12,
        offset: 249856,
    },
    Magic {
        magic: 0x8020008000,
        mask: 0x1010101010170,
        shift: 12,
        offset: 253952,
    },
    Magic {
        magic: 0x880801000,
        mask: 0x1010101010170,
        shift: 12,
        offset: 258048,
    },
];

/// Attack table for bishops
const BISHOP_TABLE_SIZE: usize = 5328; // Sum of 2^bits for all squares

/// Attack table for rooks
const ROOK_TABLE_SIZE: usize = 49248; // Sum of 2^bits for all squares

/// Combined attack table size
const ATTACK_TABLE_SIZE: usize = BISHOP_TABLE_SIZE + ROOK_TABLE_SIZE;

/// Combined attack table for both bishops and rooks
static mut ATTACK_TABLE: [u64; ATTACK_TABLE_SIZE] = [0u64; ATTACK_TABLE_SIZE];

/// Get bishop attacks for a given square and occupancy
#[inline]
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let magic = &BISHOP_MAGICS[sq as usize];
    let index = magic.index(occupied);
    unsafe { Bitboard::new(ATTACK_TABLE[magic.offset + index]) }
}

/// Get rook attacks for a given square and occupancy
#[inline]
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let magic = &ROOK_MAGICS[sq as usize];
    let index = magic.index(occupied);
    unsafe { Bitboard::new(ATTACK_TABLE[magic.offset + index]) }
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
        let occupied_masked = occupied.as_u64() & self.mask;
        let magic_mult = (occupied_masked as u128 * self.magic as u128) >> (64 - self.shift);
        magic_mult as usize
    }
}

/// Initialize the attack table
pub fn init_attack_table() {
    unsafe {
        // For each bishop square
        for sq in 0..64 {
            let magic = &BISHOP_MAGICS[sq];
            let mask = Bitboard::new(magic.mask);

            // Generate all possible occupancies
            let mut occ = Bitboard::EMPTY;
            loop {
                let index = magic.index(occ);
                ATTACK_TABLE[magic.offset + index] =
                    generate_bishop_attacks(sq as u8, occ).as_u64();

                if occ.as_u64() == mask.as_u64() {
                    break;
                }
                occ =
                    Bitboard::new(((occ.as_u64() - mask.as_u64()) & mask.as_u64()) + mask.as_u64());
            }
        }

        // For each rook square
        for sq in 0..64 {
            let magic = &ROOK_MAGICS[sq];
            let mask = Bitboard::new(magic.mask);

            // Generate all possible occupancies
            let mut occ = Bitboard::EMPTY;
            loop {
                let index = magic.index(occ);
                ATTACK_TABLE[magic.offset + index] = generate_rook_attacks(sq as u8, occ).as_u64();

                if occ.as_u64() == mask.as_u64() {
                    break;
                }
                occ =
                    Bitboard::new(((occ.as_u64() - mask.as_u64()) & mask.as_u64()) + mask.as_u64());
            }
        }
    }
}

/// Generate bishop attacks for a square (slow but correct, used for initialization)
fn generate_bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let mut attacks = Bitboard::EMPTY;
    let rank = sq / 8;
    let file = sq % 8;

    // Northeast
    for i in 1..8 {
        let r = rank + i;
        let f = file + i;
        if r >= 8 || f >= 8 {
            break;
        }
        let target = r * 8 + f;
        attacks.set(target);
        if occupied.get(target) {
            break;
        }
    }

    // Northwest
    for i in 1..8 {
        let r = rank + i;
        let f = file as i32 - i as i32;
        if r >= 8 || f < 0 {
            break;
        }
        let target = r * 8 + f as u8;
        attacks.set(target);
        if occupied.get(target) {
            break;
        }
    }

    // Southeast
    for i in 1..8 {
        let r = rank as i32 - i as i32;
        let f = file + i;
        if r < 0 || f >= 8 {
            break;
        }
        let target = r as u8 * 8 + f;
        attacks.set(target);
        if occupied.get(target) {
            break;
        }
    }

    // Southwest
    for i in 1..8 {
        let r = rank as i32 - i as i32;
        let f = file as i32 - i as i32;
        if r < 0 || f < 0 {
            break;
        }
        let target = r as u8 * 8 + f as u8;
        attacks.set(target);
        if occupied.get(target) {
            break;
        }
    }

    attacks
}

/// Generate rook attacks for a square (slow but correct, used for initialization)
fn generate_rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let mut attacks = Bitboard::EMPTY;
    let rank = sq / 8;
    let file = sq % 8;

    // North
    for i in (rank + 1)..8 {
        let target = i * 8 + file;
        attacks.set(target);
        if occupied.get(target) {
            break;
        }
    }

    // South
    for i in (0..rank).rev() {
        let target = i * 8 + file;
        attacks.set(target);
        if occupied.get(target) {
            break;
        }
    }

    // East
    for i in (file + 1)..8 {
        let target = rank * 8 + i;
        attacks.set(target);
        if occupied.get(target) {
            break;
        }
    }

    // West
    for i in (0..file).rev() {
        let target = rank * 8 + i;
        attacks.set(target);
        if occupied.get(target) {
            break;
        }
    }

    attacks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bishop_attacks_center() {
        init_attack_table();

        let occupied = Bitboard::EMPTY;
        let attacks = bishop_attacks(28, occupied); // e4

        // e4 should have 13 squares on diagonals (from center)
        assert_eq!(attacks.count(), 13);
    }

    #[test]
    fn test_bishop_attacks_blocked() {
        init_attack_table();

        // Place a piece on f5 (blocks one diagonal)
        let occupied = Bitboard::from_square(37);
        let attacks = bishop_attacks(28, occupied); // e4

        // Should have fewer squares due to blocking
        assert!(attacks.count() < 13);
    }

    #[test]
    fn test_rook_attacks_center() {
        init_attack_table();

        let occupied = Bitboard::EMPTY;
        let attacks = rook_attacks(28, occupied); // e4

        // e4 should have 14 squares on ranks and files (from center)
        assert_eq!(attacks.count(), 14);
    }

    #[test]
    fn test_queen_attacks() {
        init_attack_table();

        let occupied = Bitboard::EMPTY;
        let attacks = queen_attacks(28, occupied); // e4

        // Queen attacks = bishop + rook
        assert_eq!(attacks.count(), 27); // 13 + 14
    }
}
