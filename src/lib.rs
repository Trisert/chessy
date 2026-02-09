//! Chessy - A UCI-compliant chess engine in Rust
//!
//! This library provides a complete chess engine implementation including:
//! - Bitboard-based board representation
//! - Fast move generation with magic bitboards
//! - Alpha-beta search with various pruning techniques
//! - Static evaluation function
//! - Transposition table
//! - UCI protocol support

pub mod bitboard;
pub mod board;
pub mod castling;
pub mod evaluation;
pub mod magic;
pub mod r#match;
pub mod movelist;
pub mod moveorder;
pub mod position;
pub mod r#move;
pub mod movegen;
pub mod piece;
pub mod search;
pub mod transposition;
pub mod utils;
pub mod zobrist;

use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize magic bitboard attack tables. Safe to call multiple times.
pub fn init() {
    INIT.call_once(|| {
        magic::init_attack_table();
    });
}

// Re-export commonly used types
pub use bitboard::Bitboard;
pub use board::Board;
pub use evaluation::Evaluation;
pub use movelist::{MoveList, ScoredMoveList};
pub use position::Position;
pub use r#move::{Move, PromotionType};
pub use piece::{Color, Piece, PieceType};
pub use search::Search;
pub use utils::{square_from_string, square_to_string, Square};
