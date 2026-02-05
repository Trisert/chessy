use crate::board::Board;
use crate::piece::Color;
use crate::r#move::Move;
use crate::utils::Square;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Position state for undo/redo functionality
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionState {
    /// Side to move
    pub side_to_move: Color,
    /// Castling rights (KQkq)
    pub castling_rights: u8,
    /// En passant square (if any)
    pub ep_square: Option<Square>,
    /// Halfmove clock (for 50-move rule)
    pub halfmove_clock: u32,
    /// Fullmove number
    pub fullmove_number: u32,
    /// Zobrist hash of the position
    pub hash: u64,
}

impl PositionState {
    /// Create a new position state
    pub fn new() -> Self {
        PositionState {
            side_to_move: Color::White,
            castling_rights: 0x0F, // All castling rights available
            ep_square: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            hash: 0,
        }
    }
}

impl Default for PositionState {
    fn default() -> Self {
        Self::new()
    }
}

/// Position with board and state
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Position {
    /// Board representation
    pub board: Board,
    /// Current state
    pub state: PositionState,
    /// Position history for repetition detection
    pub history: Vec<PositionState>,
    /// Hash of history positions
    pub history_hashes: Vec<u64>,
}

impl Position {
    /// Create a new position from the starting position
    pub fn from_start() -> Self {
        let mut position = Position {
            board: Board::from_start(),
            state: PositionState::new(),
            history: Vec::new(),
            history_hashes: Vec::new(),
        };
        position.state.hash = position.compute_hash();
        position
    }

    /// Create a position from a FEN string
    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 6 {
            return Err("Invalid FEN: must have at least 6 fields".to_string());
        }

        // Parse board
        let board = Board::from_fen(parts[0])?;

        // Parse side to move
        let side_to_move = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err("Invalid side to move".to_string()),
        };

        // Parse castling rights
        let castling_rights = if parts[2] == "-" {
            0u8
        } else {
            let mut rights = 0u8;
            for c in parts[2].chars() {
                match c {
                    'K' => rights |= 0x01,
                    'Q' => rights |= 0x02,
                    'k' => rights |= 0x04,
                    'q' => rights |= 0x08,
                    _ => {}
                }
            }
            rights
        };

        // Parse en passant square
        let ep_square = if parts[3] == "-" {
            None
        } else {
            Some(crate::utils::square_from_string(parts[3])
                .ok_or_else(|| format!("Invalid en passant square: {}", parts[3]))?)
        };

        // Parse halfmove clock
        let halfmove_clock = parts[4].parse()
            .map_err(|_| format!("Invalid halfmove clock: {}", parts[4]))?;

        // Parse fullmove number
        let fullmove_number = parts[5].parse()
            .map_err(|_| format!("Invalid fullmove number: {}", parts[5]))?;

        let mut position = Position {
            board,
            state: PositionState {
                side_to_move,
                castling_rights,
                ep_square,
                halfmove_clock,
                fullmove_number,
                hash: 0,
            },
            history: Vec::new(),
            history_hashes: Vec::new(),
        };

        position.state.hash = position.compute_hash();
        Ok(position)
    }

    /// Create a position from a board with custom state (for move validation)
    pub fn from_board_with_state(
        board: Board,
        side_to_move: Color,
        castling_rights: u8,
        ep_square: Option<Square>,
    ) -> Self {
        let mut position = Position {
            board,
            state: PositionState {
                side_to_move,
                castling_rights,
                ep_square,
                halfmove_clock: 0,
                fullmove_number: 1,
                hash: 0,
            },
            history: Vec::new(),
            history_hashes: Vec::new(),
        };
        position.state.hash = position.compute_hash();
        position
    }

    /// Make a move on the position
    pub fn make_move(&mut self, mv: Move) {
        // Save current state to history
        self.history.push(self.state.clone());
        self.history_hashes.push(self.state.hash);

        let from = mv.from();
        let to = mv.to();
        let color = self.state.side_to_move;

        // Debug: Check if move is legal before making it
        #[cfg(debug_assertions)]
        {
            use crate::movegen::MoveGen;
            let legal_moves = MoveGen::generate_legal_moves_ep(
                &self.board,
                color,
                self.state.ep_square,
                self.state.castling_rights,
            );
            let mut found = false;
            for i in 0..legal_moves.len() {
                if legal_moves.get(i) == mv {
                    found = true;
                    break;
                }
            }
            if !found && !mv.is_null() {
                eprintln!("WARNING: make_move called with illegal move: {}", mv);
                eprintln!("  Side to move: {:?}", color);
                eprintln!("  Position:\n{}", self.board.to_string());
            }
        }

        // Update halfmove clock
        let is_capture = self.board.get_piece(to).is_some();
        let is_pawn_move = self.board.get_piece(from).map_or(false, |p| p.piece_type == crate::piece::PieceType::Pawn);

        if is_capture || is_pawn_move {
            self.state.halfmove_clock = 0;
        } else {
            self.state.halfmove_clock += 1;
        }

        // Update fullmove number
        if color == Color::Black {
            self.state.fullmove_number += 1;
        }

        // Debug: Check specific moves BEFORE making them
        if mv.to() == 3 || mv.from() == 10 || mv.from() == 3 || mv.from() == 6 || mv.to() == 6 {
            eprintln!("=== DEBUG: BEFORE Move involving d1/c2/g1 ===");
            eprintln!("Move: {}", mv);
            eprintln!("From: {} ({})", mv.from(), crate::utils::square_to_string(mv.from()));
            eprintln!("To: {} ({})", mv.to(), crate::utils::square_to_string(mv.to()));
            eprintln!("Is promotion: {}", mv.is_promotion());

            // Check what pieces are on the FROM square before move
            let pieces_on_from = self.board.debug_get_pieces_on(mv.from());
            eprintln!("Pieces on from square: {:?}", pieces_on_from);

            // Check king squares before
            let white_king = self.board.king_square(crate::piece::Color::White);
            eprintln!("White king square BEFORE: {:?}", white_king);

            if let Err(err) = self.board.validate() {
                eprintln!("VALIDATION ERROR BEFORE MOVE:\n{}", err);
                self.board.debug_print();
            }
            eprintln!("====================================");
        }

        // Make the move on the board
        self.make_move_on_board(mv, color);

        // Debug: Check same moves AFTER making them
        if mv.to() == 3 || mv.from() == 10 || mv.from() == 3 || mv.from() == 6 || mv.to() == 6 {
            eprintln!("=== DEBUG: AFTER Move involving d1/c2/g1 ===");

            // Check what pieces are on the TO square after move
            let pieces_on_to = self.board.debug_get_pieces_on(mv.to());
            eprintln!("Pieces on to square: {:?}", pieces_on_to);

            // Check king squares after
            let white_king = self.board.king_square(crate::piece::Color::White);
            eprintln!("White king square AFTER: {:?}", white_king);

            if let Err(err) = self.board.validate() {
                eprintln!("VALIDATION ERROR AFTER MOVE:\n{}", err);
                self.board.debug_print();
            }
            eprintln!("====================================");
        }

        // Validate board state after move
        if let Err(err) = self.board.validate() {
            eprintln!("=== BOARD STATE CORRUPTION DETECTED ===");
            eprintln!("After move: {}", mv);
            eprintln!("Side to move: {:?}", color);
            eprintln!("ERROR:\n{}", err);
            eprintln!("\nCurrent board:");
            self.board.debug_print();
            eprintln!("====================================");
        }

        // Update side to move
        self.state.side_to_move = color.flip();

        // Update castling rights
        self.update_castling_rights(from, to);

        // Update en passant square
        self.update_ep_square(mv, color);

        // Recompute hash
        self.state.hash = self.compute_hash();
    }

    /// Undo a move
    pub fn undo_move(&mut self) {
        if let Some(prev_state) = self.history.pop() {
            self.state = prev_state;
            self.history_hashes.pop();

            // Restore board from previous state
            // Note: This is a simplified version - a full implementation would
            // need to track the board state changes more carefully
            // For now, we'll rebuild the position from the current state
        }
    }

    /// Check for threefold repetition
    pub fn is_threefold_repetition(&self) -> bool {
        let current_hash = self.state.hash;
        let mut count = 0;

        for &hash in &self.history_hashes {
            if hash == current_hash {
                count += 1;
                if count >= 2 {
                    return true; // Current position + 2 previous = 3fold
                }
            }
        }

        false
    }

    /// Check for fifty-move rule draw
    pub fn is_fifty_move_rule(&self) -> bool {
        self.state.halfmove_clock >= 100
    }

    /// Check if current position is checkmate
    pub fn is_checkmate(&self) -> bool {
        use crate::movegen::MoveGen;

        let color = self.state.side_to_move;

        // Check if king is in check
        let king_sq = match self.board.king_square(color) {
            Some(sq) => sq,
            None => return false,
        };

        if !MoveGen::is_square_attacked(&self.board, king_sq, color.flip()) {
            return false;
        }

        // Check if there are no legal moves
        let moves = MoveGen::generate_legal_moves(&self.board, color);
        moves.is_empty()
    }

    /// Check if current position is stalemate
    pub fn is_stalemate(&self) -> bool {
        use crate::movegen::MoveGen;

        let color = self.state.side_to_move;

        // Check if king is NOT in check
        let king_sq = match self.board.king_square(color) {
            Some(sq) => sq,
            None => return false,
        };

        if MoveGen::is_square_attacked(&self.board, king_sq, color.flip()) {
            return false;
        }

        // Check if there are no legal moves
        let moves = MoveGen::generate_legal_moves(&self.board, color);
        moves.is_empty()
    }

    /// Check if the game is over
    pub fn is_game_over(&self) -> bool {
        self.is_checkmate() || self.is_stalemate() || self.is_threefold_repetition() || self.is_fifty_move_rule()
    }

    /// Get game result string
    pub fn game_result(&self) -> Option<&'static str> {
        if self.is_checkmate() {
            Some(if self.state.side_to_move == Color::White {
                "0-1 (Black wins)"
            } else {
                "1-0 (White wins)"
            })
        } else if self.is_stalemate() {
            Some("1/2-1/2 (Stalemate)")
        } else if self.is_threefold_repetition() {
            Some("1/2-1/2 (Threefold repetition)")
        } else if self.is_fifty_move_rule() {
            Some("1/2-1/2 (Fifty-move rule)")
        } else {
            None
        }
    }

    /// Make a move on the board (internal)
    fn make_move_on_board(&mut self, mv: Move, color: Color) {
        use crate::piece::{Piece, PieceType};
        use crate::r#move::PromotionType;

        let from = mv.from();
        let to = mv.to();

        // Handle castling
        if mv.is_castle() {
            // Move the king
            if let Some(king) = self.board.get_piece(from) {
                self.board.remove_piece(king, from);
                self.board.set_piece(king, to);
            }

            // Move the rook
            let rook_from = if color == Color::White {
                if to == 6 { 7 } else { 0 }  // Kingside: h1(7), Queenside: a1(0)
            } else {
                if to == 62 { 63 } else { 56 }  // Kingside: h8(63), Queenside: a8(56)
            };

            let rook_to = if color == Color::White {
                if to == 6 { 5 } else { 3 }  // Kingside: f1(5), Queenside: d1(3)
            } else {
                if to == 62 { 61 } else { 59 }  // Kingside: f8(61), Queenside: d8(59)
            };

            if let Some(rook) = self.board.get_piece(rook_from) {
                self.board.remove_piece(rook, rook_from);
                self.board.set_piece(rook, rook_to);
            }

            return;
        }

        // Handle en passant
        if mv.is_en_passant() {
            if let Some(piece) = self.board.get_piece(from) {
                self.board.remove_piece(piece, from);
                self.board.set_piece(piece, to);

                // Remove the captured pawn
                let captured_pawn_sq = if color == Color::White {
                    to - 8  // White captures "behind" the EP square
                } else {
                    to + 8  // Black captures "behind" the EP square
                };

                if let Some(captured) = self.board.get_piece(captured_pawn_sq) {
                    self.board.remove_piece(captured, captured_pawn_sq);
                }
            }
            return;
        }

        // Normal move
        if let Some(piece) = self.board.get_piece(from) {
            // Debug: check for d1b3 specifically
            if from == 3 && to == 17 {
                eprintln!("=== DEBUG: Making move from d1 to b3 ===");
                eprintln!("Piece from d1: {:?}", piece);
                eprintln!("Piece type: {:?}, Color: {:?}", piece.piece_type, piece.color);

                // Check what's actually on d1 in the bitboards
                let pieces_on_d1 = self.board.debug_get_pieces_on(3);
                eprintln!("Pieces on d1: {:?}", pieces_on_d1);
                for (c, pt) in pieces_on_d1 {
                    eprintln!("  {:?} {:?}", c, pt);
                }
                self.board.debug_print();
            }

            self.board.remove_piece(piece, from);

            // Handle captures
            if let Some(captured) = self.board.get_piece(to) {
                self.board.remove_piece(captured, to);
            }

            // Handle promotions
            if mv.is_promotion() {
                if let Some(promo_type) = mv.promotion_type() {
                    let promo_piece = Piece::new(
                        color,
                        match promo_type {
                            PromotionType::Knight => PieceType::Knight,
                            PromotionType::Bishop => PieceType::Bishop,
                            PromotionType::Rook => PieceType::Rook,
                            PromotionType::Queen => PieceType::Queen,
                        },
                    );
                    self.board.set_piece(promo_piece, to);
                }
            } else {
                self.board.set_piece(piece, to);
            }
        } else {
            // CRITICAL BUG: No piece on the 'from' square!
            // Skip the ENTIRE move processing - don't update board OR state
            // This prevents position state desynchronization
            eprintln!("=== WARNING: get_piece({}) returned None ===", from);
            eprintln!("  Square: {} ({})", from, crate::utils::square_to_string(from));
            eprintln!("  Move: {} (from {} to {})", mv, from, to);
            eprintln!("  This means the square is EMPTY or piece tracking is corrupted");
            eprintln!("  Board state at this point:");
            self.board.debug_print();
            eprintln!("=== Skipping this move, continuing with rest of sequence ===");
            return;
        }

        // Update side to move
        self.state.side_to_move = color.flip();

        // Update castling rights
        self.update_castling_rights(from, to);

        // Update en passant square
        self.update_ep_square(mv, color);

        // Recompute hash
        self.state.hash = self.compute_hash();
    }

    /// Update castling rights after a move
    fn update_castling_rights(&mut self, from: Square, _to: Square) {
        const WHITE_KINGSIDE: u8 = 0x01;
        const WHITE_QUEENSIDE: u8 = 0x02;
        const BLACK_KINGSIDE: u8 = 0x04;
        const BLACK_QUEENSIDE: u8 = 0x08;

        // White king moves
        if from == 4 && self.state.castling_rights & 0x03 != 0 {
            self.state.castling_rights &= !(WHITE_KINGSIDE | WHITE_QUEENSIDE);
        }

        // Black king moves
        if from == 60 && self.state.castling_rights & 0x0C != 0 {
            self.state.castling_rights &= !(BLACK_KINGSIDE | BLACK_QUEENSIDE);
        }

        // Rook moves
        if from == 0 {
            self.state.castling_rights &= !WHITE_QUEENSIDE;
        }
        if from == 7 {
            self.state.castling_rights &= !WHITE_KINGSIDE;
        }
        if from == 56 {
            self.state.castling_rights &= !BLACK_QUEENSIDE;
        }
        if from == 63 {
            self.state.castling_rights &= !BLACK_KINGSIDE;
        }
    }

    /// Update en passant square
    fn update_ep_square(&mut self, mv: Move, color: Color) {
        use crate::piece::PieceType;

        let from = mv.from();
        let to = mv.to();

        // Check for double pawn push
        if let Some(piece) = self.board.get_piece(to) {
            if piece.piece_type == PieceType::Pawn {
                let rank_diff = (from as i8 - to as i8).abs();

                if rank_diff == 16 {
                    // Double push - set EP square
                    self.state.ep_square = Some(if color == Color::White {
                        to - 8
                    } else {
                        to + 8
                    });
                    return;
                }
            }
        }

        // No EP square
        self.state.ep_square = None;
    }

    /// Compute Zobrist hash (simplified version)
    fn compute_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Hash board state
        for sq in 0..64 {
            if let Some(piece) = self.board.get_piece(sq) {
                piece.hash_to(sq, &mut hasher);
            }
        }

        // Hash side to move
        self.state.side_to_move.hash(&mut hasher);

        // Hash castling rights
        self.state.castling_rights.hash(&mut hasher);

        // Hash EP square
        self.state.ep_square.hash(&mut hasher);

        hasher.finish()
    }
}

/// Extension trait for piece hashing
trait Hashable {
    fn hash_to(&self, sq: Square, hasher: &mut DefaultHasher);
}

impl Hashable for crate::piece::Piece {
    fn hash_to(&self, sq: Square, hasher: &mut DefaultHasher) {
        (self.color, self.piece_type, sq).hash(hasher);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_from_start() {
        let pos = Position::from_start();

        assert_eq!(pos.state.side_to_move, Color::White);
        assert_eq!(pos.state.castling_rights, 0x0F);
        assert_eq!(pos.state.halfmove_clock, 0);
        assert_eq!(pos.state.fullmove_number, 1);
        assert!(!pos.is_checkmate());
        assert!(!pos.is_stalemate());
        assert!(!pos.is_threefold_repetition());
        assert!(!pos.is_fifty_move_rule());
    }
}
