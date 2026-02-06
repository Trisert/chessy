use crate::piece::Piece;
use crate::utils::Square;
use std::sync::LazyLock;

/// Zobrist hashing tables for incremental position hashing
///
/// # Design
/// This struct uses fixed-size arrays for optimal performance and cache locality.
/// All fields are private - access is provided through associated functions on the
/// global ZOBRIST instance (e.g., `Zobrist::piece(...)` rather than `ZOBRIST.piece(...)`).
/// This design ensures the global tables remain immutable and accessible from anywhere.
pub struct Zobrist {
    /// Piece squares: [piece_type][color][square]
    pieces: [[[u64; 64]; 2]; 6],
    /// Castling rights: [4] (KQkq)
    castling: [u64; 4],
    /// En passant square: [64]
    en_passant: [u64; 64],
    /// Side to move
    side_to_move: u64,
}

/// Global Zobrist tables (initialized once at first access)
///
/// # Usage
/// Access the hash values through the associated functions:
/// - `Zobrist::piece(piece, square)` - hash for a piece on a square
/// - `Zobrist::castling(rights)` - hash for castling rights
/// - `Zobrist::en_passant(square)` - hash for en passant square
/// - `Zobrist::side()` - hash for side to move
///
/// These functions access the global ZOBRIST instance internally.
pub static ZOBRIST: LazyLock<Zobrist> = LazyLock::new(Zobrist::new);

impl Zobrist {
    /// Create new Zobrist tables with random 64-bit numbers
    fn new() -> Self {
        // Fixed seed for reproducibility (non-zero to avoid degeneracy)
        const FIXED_ZOBRIST_SEED: u64 = 0x9E3779B97F4A7C15;

        let mut rng = FIXED_ZOBRIST_SEED;

        // Initialize fixed-size arrays
        let mut pieces = [[[0u64; 64]; 2]; 6];
        for pt in 0..6 {
            for color in 0..2 {
                for sq in 0..64 {
                    pieces[pt][color][sq] = random_u64(&mut rng);
                }
            }
        }

        let mut castling = [0u64; 4];
        for i in 0..4 {
            castling[i] = random_u64(&mut rng);
        }

        let mut en_passant = [0u64; 64];
        for sq in 0..64 {
            en_passant[sq] = random_u64(&mut rng);
        }

        let side_to_move = random_u64(&mut rng);

        Zobrist {
            pieces,
            castling,
            en_passant,
            side_to_move,
        }
    }

    /// Get hash for a piece on a square
    #[inline]
    pub fn piece(piece: Piece, square: Square) -> u64 {
        let pt = piece.piece_type as usize;
        let c = piece.color as usize;
        ZOBRIST.pieces[pt][c][square as usize]
    }

    /// Get hash for castling rights
    #[inline]
    pub fn castling(rights: u8) -> u64 {
        let mut hash = 0;
        if rights & 0x01 != 0 {
            hash ^= ZOBRIST.castling[0]; // K
        }
        if rights & 0x02 != 0 {
            hash ^= ZOBRIST.castling[1]; // Q
        }
        if rights & 0x04 != 0 {
            hash ^= ZOBRIST.castling[2]; // k
        }
        if rights & 0x08 != 0 {
            hash ^= ZOBRIST.castling[3]; // q
        }
        hash
    }

    /// Get hash for en passant square
    #[inline]
    pub fn en_passant(square: Square) -> u64 {
        ZOBRIST.en_passant[square as usize]
    }

    /// Get hash for side to move
    #[inline]
    pub fn side() -> u64 {
        ZOBRIST.side_to_move
    }
}

/// Simple random number generator (xorshift64* variant)
///
/// This is a variant of xorshift64* where the multiply is applied to the state
/// itself rather than just being an output scrambler. This preserves the
/// non-zero invariant (unlike canonical xorshift64* which returns state * mult
/// without modifying state).
fn random_u64(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    *state = state.wrapping_mul(2685821657736338717);
    *state
}
