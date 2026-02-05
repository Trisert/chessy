use crate::bitboard::Bitboard;
use crate::piece::{Color, Piece, PieceType};
use crate::utils::{square_of, Square};

/// Board representation using bitboards
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    /// Piece bitboards [piece_type][color]
    piece_bb: [[Bitboard; 2]; 6],
    /// Color bitboards [color]
    color_bb: [Bitboard; 2],
    /// Combined occupancy
    occupied: Bitboard,
}

impl Board {
    /// Create a new empty board
    pub fn new() -> Self {
        Board {
            piece_bb: [[Bitboard::EMPTY; 2]; 6],
            color_bb: [Bitboard::EMPTY; 2],
            occupied: Bitboard::EMPTY,
        }
    }

    /// Create a board from the starting position
    pub fn from_start() -> Self {
        let mut board = Board::new();

        // Set up pawns
        for file in 0..8 {
            board.set_piece(Piece::new(Color::White, PieceType::Pawn), square_of(1, file));
            board.set_piece(Piece::new(Color::Black, PieceType::Pawn), square_of(6, file));
        }

        // Set up pieces
        // Rooks
        board.set_piece(Piece::new(Color::White, PieceType::Rook), square_of(0, 0));
        board.set_piece(Piece::new(Color::White, PieceType::Rook), square_of(0, 7));
        board.set_piece(Piece::new(Color::Black, PieceType::Rook), square_of(7, 0));
        board.set_piece(Piece::new(Color::Black, PieceType::Rook), square_of(7, 7));

        // Knights
        board.set_piece(Piece::new(Color::White, PieceType::Knight), square_of(0, 1));
        board.set_piece(Piece::new(Color::White, PieceType::Knight), square_of(0, 6));
        board.set_piece(Piece::new(Color::Black, PieceType::Knight), square_of(7, 1));
        board.set_piece(Piece::new(Color::Black, PieceType::Knight), square_of(7, 6));

        // Bishops
        board.set_piece(Piece::new(Color::White, PieceType::Bishop), square_of(0, 2));
        board.set_piece(Piece::new(Color::White, PieceType::Bishop), square_of(0, 5));
        board.set_piece(Piece::new(Color::Black, PieceType::Bishop), square_of(7, 2));
        board.set_piece(Piece::new(Color::Black, PieceType::Bishop), square_of(7, 5));

        // Queens
        board.set_piece(Piece::new(Color::White, PieceType::Queen), square_of(0, 3));
        board.set_piece(Piece::new(Color::Black, PieceType::Queen), square_of(7, 3));

        // Kings
        board.set_piece(Piece::new(Color::White, PieceType::King), square_of(0, 4));
        board.set_piece(Piece::new(Color::Black, PieceType::King), square_of(7, 4));

        board
    }

    /// Create a board from a FEN board string (first field of FEN)
    pub fn from_fen(fen_board: &str) -> Result<Self, String> {
        let mut board = Board::new();

        let ranks: Vec<&str> = fen_board.split('/').collect();
        if ranks.len() != 8 {
            return Err("Invalid FEN: must have 8 ranks".to_string());
        }

        for (rank_idx, rank_str) in ranks.iter().enumerate() {
            let rank = (7 - rank_idx) as u8; // FEN starts from rank 8
            let mut file = 0u8;

            for c in rank_str.chars() {
                if c.is_ascii_digit() {
                    // Empty squares
                    let empty_count = c.to_digit(10).unwrap() as u8;
                    file += empty_count;
                } else {
                    // Piece
                    let square = square_of(rank, file);

                    let (piece_type, color) = match c {
                        'P' => (PieceType::Pawn, Color::White),
                        'N' => (PieceType::Knight, Color::White),
                        'B' => (PieceType::Bishop, Color::White),
                        'R' => (PieceType::Rook, Color::White),
                        'Q' => (PieceType::Queen, Color::White),
                        'K' => (PieceType::King, Color::White),
                        'p' => (PieceType::Pawn, Color::Black),
                        'n' => (PieceType::Knight, Color::Black),
                        'b' => (PieceType::Bishop, Color::Black),
                        'r' => (PieceType::Rook, Color::Black),
                        'q' => (PieceType::Queen, Color::Black),
                        'k' => (PieceType::King, Color::Black),
                        _ => return Err(format!("Invalid piece character: {}", c)),
                    };

                    board.set_piece(Piece::new(color, piece_type), square);
                    file += 1;
                }

                if file > 8 {
                    return Err("Too many squares on a rank".to_string());
                }
            }

            if file != 8 {
                return Err("Not enough squares on a rank".to_string());
            }
        }

        Ok(board)
    }

    /// Set a piece on a square
    #[inline]
    pub fn set_piece(&mut self, piece: Piece, sq: Square) {
        let piece_idx = piece.piece_type.index();
        let color_idx = piece.color.index();

        // Clear any existing piece on this square
        self.clear_square(sq);

        // Set the piece
        self.piece_bb[piece_idx][color_idx].set(sq);
        self.color_bb[color_idx].set(sq);
        self.occupied.set(sq);
    }

    /// Remove a piece from a square
    #[inline]
    pub fn remove_piece(&mut self, piece: Piece, sq: Square) {
        let piece_idx = piece.piece_type.index();
        let color_idx = piece.color.index();

        self.piece_bb[piece_idx][color_idx].clear(sq);
        self.color_bb[color_idx].clear(sq);
        self.occupied.clear(sq);
    }

    /// Clear a square (remove any piece)
    #[inline]
    fn clear_square(&mut self, sq: Square) {
        if self.occupied.get(sq) {
            // Find which color and piece
            for color_idx in 0..2 {
                if self.color_bb[color_idx].get(sq) {
                    for piece_idx in 0..6 {
                        if self.piece_bb[piece_idx][color_idx].get(sq) {
                            self.piece_bb[piece_idx][color_idx].clear(sq);
                            break;
                        }
                    }
                    self.color_bb[color_idx].clear(sq);
                    break;
                }
            }
            self.occupied.clear(sq);
        }
    }

    /// Get the piece on a square (if any)
    pub fn get_piece(&self, sq: Square) -> Option<Piece> {
        if !self.occupied.get(sq) {
            return None;
        }

        for color_idx in 0..2 {
            if self.color_bb[color_idx].get(sq) {
                for piece_idx in 0..6 {
                    if self.piece_bb[piece_idx][color_idx].get(sq) {
                        return Some(Piece::new(
                            if color_idx == 0 { Color::White } else { Color::Black },
                            match piece_idx {
                                0 => PieceType::Pawn,
                                1 => PieceType::Knight,
                                2 => PieceType::Bishop,
                                3 => PieceType::Rook,
                                4 => PieceType::Queen,
                                5 => PieceType::King,
                                _ => unreachable!(),
                            },
                        ));
                    }
                }
            }
        }

        None
    }

    /// Get bitboard for a specific piece type and color
    #[inline]
    pub fn piece_bb(&self, piece_type: PieceType, color: Color) -> Bitboard {
        self.piece_bb[piece_type.index()][color.index()]
    }

    /// Get bitboard for a color (all pieces of that color)
    #[inline]
    pub fn color_bb(&self, color: Color) -> Bitboard {
        self.color_bb[color.index()]
    }

    /// Get occupied squares bitboard
    #[inline]
    pub fn occupied(&self) -> Bitboard {
        self.occupied
    }

    /// Get empty squares bitboard
    #[inline]
    pub fn empty(&self) -> Bitboard {
        self.occupied.not() & Bitboard::FULL
    }

    /// Get all pieces bitboard
    #[inline]
    pub fn all_pieces(&self) -> Bitboard {
        self.occupied
    }

    /// Get king square for a color
    pub fn king_square(&self, color: Color) -> Option<Square> {
        let king_bb = self.piece_bb(PieceType::King, color);
        king_bb.lsb()
    }

    /// Check if a square is occupied
    #[inline]
    pub fn is_occupied(&self, sq: Square) -> bool {
        self.occupied.get(sq)
    }

    /// Check if a square is occupied by a specific color
    #[inline]
    pub fn is_occupied_by(&self, sq: Square, color: Color) -> bool {
        self.color_bb[color.index()].get(sq)
    }

    /// Move a piece from one square to another
    pub fn move_piece(&mut self, from: Square, to: Square) {
        if let Some(piece) = self.get_piece(from) {
            self.remove_piece(piece, from);
            self.set_piece(piece, to);
        }
    }

    /// Convert board to a string representation
    pub fn to_string(&self) -> String {
        let mut result = String::new();
        result.push_str("  a b c d e f g h\n");
        for rank in (0..8).rev() {
            result.push_str(&(rank + 1).to_string());
            result.push(' ');
            for file in 0..8 {
                let sq = square_of(rank, file);
                let ch = match self.get_piece(sq) {
                    None => '.',
                    Some(piece) => match (piece.color, piece.piece_type) {
                        (Color::White, PieceType::Pawn) => 'P',
                        (Color::White, PieceType::Knight) => 'N',
                        (Color::White, PieceType::Bishop) => 'B',
                        (Color::White, PieceType::Rook) => 'R',
                        (Color::White, PieceType::Queen) => 'Q',
                        (Color::White, PieceType::King) => 'K',
                        (Color::Black, PieceType::Pawn) => 'p',
                        (Color::Black, PieceType::Knight) => 'n',
                        (Color::Black, PieceType::Bishop) => 'b',
                        (Color::Black, PieceType::Rook) => 'r',
                        (Color::Black, PieceType::Queen) => 'q',
                        (Color::Black, PieceType::King) => 'k',
                    },
                };
                result.push(ch);
                result.push(' ');
            }
            result.push('\n');
        }
        result
    }

    /// Count material for a color
    pub fn count_material(&self, color: Color) -> i32 {
        let pawn_value = 100;
        let knight_value = 320;
        let bishop_value = 330;
        let rook_value = 500;
        let queen_value = 900;
        let king_value = 0; // King has no material value

        self.piece_bb(PieceType::Pawn, color).count() as i32 * pawn_value
            + self.piece_bb(PieceType::Knight, color).count() as i32 * knight_value
            + self.piece_bb(PieceType::Bishop, color).count() as i32 * bishop_value
            + self.piece_bb(PieceType::Rook, color).count() as i32 * rook_value
            + self.piece_bb(PieceType::Queen, color).count() as i32 * queen_value
            + self.piece_bb(PieceType::King, color).count() as i32 * king_value
    }

    /// Validate board state consistency (for debugging)
    pub fn validate(&self) -> Result<(), String> {
        let mut errors = Vec::new();

        // Check that occupied squares match piece placements
        let mut occupied_from_pieces = Bitboard::EMPTY;
        for piece_idx in 0..6 {
            for color_idx in 0..2 {
                occupied_from_pieces = occupied_from_pieces | self.piece_bb[piece_idx][color_idx];
            }
        }

        if occupied_from_pieces.as_u64() != self.occupied.as_u64() {
            errors.push(format!("Occupied bitboard mismatch!\n  From pieces: {:064b}\n  Occupied BB:  {:064b}",
                occupied_from_pieces.as_u64(), self.occupied.as_u64()));
        }

        // Check that each square has at most one piece
        for sq in 0..64 {
            let mut piece_count = 0;
            let mut found_pieces = Vec::new();
            for piece_idx in 0..6 {
                for color_idx in 0..2 {
                    if self.piece_bb[piece_idx][color_idx].get(sq) {
                        piece_count += 1;
                        let color_name = if color_idx == 0 { "White" } else { "Black" };
                        let piece_name = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"][piece_idx];
                        found_pieces.push(format!("{} {}", color_name, piece_name));
                    }
                }
            }
            if self.occupied.get(sq) && piece_count != 1 {
                errors.push(format!("Square {} ({}) has {} pieces: {}",
                    sq, crate::utils::square_to_string(sq), piece_count, found_pieces.join(", ")));
            } else if !self.occupied.get(sq) && piece_count > 0 {
                errors.push(format!("Square {} ({}) is marked empty but has pieces: {}",
                    sq, crate::utils::square_to_string(sq), found_pieces.join(", ")));
            } else if self.occupied.get(sq) && piece_count == 0 {
                errors.push(format!("Square {} ({}) is marked occupied but has no pieces",
                    sq, crate::utils::square_to_string(sq)));
            }
        }

        // Check that each color has exactly one king
        for color_idx in 0..2 {
            let king_count = self.piece_bb[5][color_idx].count();
            let color_name = if color_idx == 0 { "White" } else { "Black" };
            if king_count != 1 {
                errors.push(format!("{} has {} kings (should have 1)", color_name, king_count));
                // List all king squares
                for sq in self.piece_bb[5][color_idx].squares() {
                    errors.push(format!("  {} king on square {}", color_name, crate::utils::square_to_string(sq)));
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("\n"))
        }
    }

    /// Print board state with all pieces (for debugging)
    pub fn debug_print(&self) {
        eprintln!("=== Board Debug State ===");
        eprintln!("Occupied: {:064b}", self.occupied.as_u64());
        for piece_idx in 0..6 {
            let piece_name = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"][piece_idx];
            for color_idx in 0..2 {
                let color_name = if color_idx == 0 { "White" } else { "Black" };
                let bb = self.piece_bb[piece_idx][color_idx];
                if !bb.is_empty() {
                    eprintln!("{} {}: {:064b}", color_name, piece_name, bb.as_u64());
                    for sq in bb.squares() {
                        eprintln!("  {} on {}", piece_name, crate::utils::square_to_string(sq));
                    }
                }
            }
        }
        eprintln!("========================");
    }

    /// Get all pieces on a specific square (for debugging)
    pub fn debug_get_pieces_on(&self, sq: Square) -> Vec<(Color, PieceType)> {
        let mut pieces = Vec::new();
        for piece_idx in 0..6 {
            for color_idx in 0..2 {
                if self.piece_bb[piece_idx][color_idx].get(sq) {
                    pieces.push((
                        if color_idx == 0 { Color::White } else { Color::Black },
                        unsafe { std::mem::transmute(piece_idx as u8) }
                    ));
                }
            }
        }
        pieces
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_board() {
        let board = Board::new();
        assert!(board.occupied().is_empty());
    }

    #[test]
    fn test_start_position() {
        let board = Board::from_start();

        // Should have 32 pieces total
        assert_eq!(board.occupied().count(), 32);

        // Each color should have 16 pieces
        assert_eq!(board.color_bb(Color::White).count(), 16);
        assert_eq!(board.color_bb(Color::Black).count(), 16);

        // Should have specific pieces
        assert_eq!(board.piece_bb(PieceType::Pawn, Color::White).count(), 8);
        assert_eq!(board.piece_bb(PieceType::Knight, Color::White).count(), 2);
        assert_eq!(board.piece_bb(PieceType::Bishop, Color::White).count(), 2);
        assert_eq!(board.piece_bb(PieceType::Rook, Color::White).count(), 2);
        assert_eq!(board.piece_bb(PieceType::Queen, Color::White).count(), 1);
        assert_eq!(board.piece_bb(PieceType::King, Color::White).count(), 1);

        // Kings should be in correct positions
        assert_eq!(board.king_square(Color::White), Some(4)); // e1
        assert_eq!(board.king_square(Color::Black), Some(60)); // e8
    }

    #[test]
    fn test_set_remove_piece() {
        let mut board = Board::new();

        board.set_piece(Piece::new(Color::White, PieceType::Knight), 28); // e4
        assert!(board.is_occupied(28));
        assert!(board.is_occupied_by(28, Color::White));
        assert_eq!(board.get_piece(28), Some(Piece::new(Color::White, PieceType::Knight)));

        board.remove_piece(Piece::new(Color::White, PieceType::Knight), 28);
        assert!(!board.is_occupied(28));
        assert_eq!(board.get_piece(28), None);
    }

    #[test]
    fn test_move_piece() {
        let mut board = Board::new();

        board.set_piece(Piece::new(Color::White, PieceType::Knight), 0); // a1
        board.move_piece(0, 28); // a1 to e4

        assert!(!board.is_occupied(0));
        assert!(board.is_occupied(28));
    }

    #[test]
    fn test_material_count() {
        let board = Board::from_start();

        let white_material = board.count_material(Color::White);
        let black_material = board.count_material(Color::Black);

        // Starting position: 8*100 + 2*320 + 2*330 + 2*500 + 1*900 = 3900
        assert_eq!(white_material, 3900);
        assert_eq!(black_material, 3900);
    }

    #[test]
    fn test_board_to_string() {
        let board = Board::from_start();
        let s = board.to_string();

        assert!(s.contains("r n b q k b n r"));
        assert!(s.contains("p p p p p p p p"));
        assert!(s.contains("P P P P P P P P"));
        assert!(s.contains("R N B Q K B N R"));
    }
}
