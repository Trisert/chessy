use crate::board::Board;
use crate::piece::Color;
use crate::r#move::Move;
use crate::utils::*;

/// Castling utilities
pub struct Castling;

impl Castling {
    /// Generate all castling moves given castling rights
    pub fn generate_castling(
        board: &Board,
        color: Color,
        castling_rights: u8,
    ) -> [Option<Move>; 2] {
        let mut moves = [None, None];

        const WHITE_KINGSIDE: u8 = 0x01;
        const WHITE_QUEENSIDE: u8 = 0x02;
        const BLACK_KINGSIDE: u8 = 0x04;
        const BLACK_QUEENSIDE: u8 = 0x08;

        // Check if king is in check
        let king_sq = match board.king_square(color) {
            Some(sq) => sq,
            None => return moves,
        };

        // King must be on starting square
        let king_start = if color == Color::White { 4 } else { 60 };
        if king_sq != king_start {
            return moves;
        }

        // Kingside castle
        if color == Color::White {
            if castling_rights & WHITE_KINGSIDE != 0 {
                if Self::can_kingside_castle(board, color) {
                    moves[0] = Some(Move::castle(king_sq, 6));
                }
            }
        } else {
            if castling_rights & BLACK_KINGSIDE != 0 {
                if Self::can_kingside_castle(board, color) {
                    moves[0] = Some(Move::castle(king_sq, 62));
                }
            }
        }

        // Queenside castle
        if color == Color::White {
            if castling_rights & WHITE_QUEENSIDE != 0 {
                if Self::can_queenside_castle(board, color) {
                    moves[1] = Some(Move::castle(king_sq, 2));
                }
            }
        } else {
            if castling_rights & BLACK_QUEENSIDE != 0 {
                if Self::can_queenside_castle(board, color) {
                    moves[1] = Some(Move::castle(king_sq, 58));
                }
            }
        }

        moves
    }

    /// Check if kingside castling is legal
    fn can_kingside_castle(board: &Board, color: Color) -> bool {
        let (king_start, king_end) = if color == Color::White {
            (4, 6) // e1 to g1
        } else {
            (60, 62) // e8 to g8
        };

        // Check if king is in check
        if Self::is_square_attacked(board, king_start, color.flip()) {
            return false;
        }

        // Check if squares king passes through are attacked
        let mid_sq = (king_start + king_end) / 2;
        if Self::is_square_attacked(board, mid_sq, color.flip()) {
            return false;
        }

        // Check if destination is attacked
        if Self::is_square_attacked(board, king_end, color.flip()) {
            return false;
        }

        // Check if squares between king and rook are empty
        for sq in (king_start + 1)..=king_end {
            if board.is_occupied(sq as Square) {
                return false;
            }
        }

        // Rook must be on corner square
        let rook_sq = if color == Color::White { 7 } else { 63 };
        if let Some(piece) = board.get_piece(rook_sq) {
            if piece.piece_type != crate::piece::PieceType::Rook {
                return false;
            }
        } else {
            return false;
        }

        true
    }

    /// Check if queenside castling is legal
    fn can_queenside_castle(board: &Board, color: Color) -> bool {
        let (king_start, king_end) = if color == Color::White {
            (4, 2) // e1 to c1
        } else {
            (60, 58) // e8 to c8
        };

        // Check if king is in check
        if Self::is_square_attacked(board, king_start, color.flip()) {
            return false;
        }

        // Check if squares king passes through are attacked
        let mid_sq = (king_start + king_end) / 2;
        if Self::is_square_attacked(board, mid_sq, color.flip()) {
            return false;
        }

        // Check if destination is attacked
        if Self::is_square_attacked(board, king_end, color.flip()) {
            return false;
        }

        // Check if squares between king and rook are empty
        let rook_sq = if color == Color::White { 0 } else { 56 };
        for sq in (rook_sq + 1)..king_start {
            if board.is_occupied(sq as Square) {
                return false;
            }
        }

        // Rook must be on corner square
        if let Some(piece) = board.get_piece(rook_sq) {
            if piece.piece_type != crate::piece::PieceType::Rook {
                return false;
            }
        } else {
            return false;
        }

        // Also need to check if the b-square is empty (for queenside)
        let b_sq = if color == Color::White { 1 } else { 57 };
        if board.is_occupied(b_sq as Square) {
            return false;
        }

        true
    }

    /// Check if a square is attacked by the opponent
    fn is_square_attacked(board: &Board, sq: Square, by_color: Color) -> bool {
        use crate::movegen::MoveGen;
        MoveGen::is_square_attacked(board, sq, by_color)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::{Piece, PieceType};

    #[test]
    fn test_kingside_castle_available() {
        // Create a board with pieces cleared for kingside castling
        let mut board = Board::new();
        board.set_piece(Piece::new(Color::White, PieceType::King), 4); // e1
        board.set_piece(Piece::new(Color::White, PieceType::Rook), 7); // h1

        let moves = Castling::generate_castling(&board, Color::White, 0x0F);

        // Should have kingside castling available
        assert!(moves[0].is_some());
    }

    #[test]
    fn test_queenside_castle_available() {
        // Create a board with pieces cleared for queenside castling
        let mut board = Board::new();
        board.set_piece(Piece::new(Color::White, PieceType::King), 4); // e1
        board.set_piece(Piece::new(Color::White, PieceType::Rook), 0); // a1

        let moves = Castling::generate_castling(&board, Color::White, 0x0F);

        // Should have queenside castling available
        assert!(moves[1].is_some());
    }

    #[test]
    fn test_kingside_castle_blocked() {
        // Create a board with bishop blocking kingside castling
        let mut board = Board::new();
        board.set_piece(Piece::new(Color::White, PieceType::King), 4); // e1
        board.set_piece(Piece::new(Color::White, PieceType::Rook), 7); // h1
        board.set_piece(Piece::new(Color::White, PieceType::Bishop), 5); // f1

        let moves = Castling::generate_castling(&board, Color::White, 0x0F);

        // Should NOT have kingside castling (blocked by bishop)
        assert!(moves[0].is_none());
    }
}
