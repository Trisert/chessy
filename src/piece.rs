/// Chess colors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    /// Flip the color (White -> Black, Black -> White)
    #[inline]
    pub fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    /// Get color index (0 for White, 1 for Black)
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Chess piece types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl PieceType {
    /// Get piece type index
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Check if piece is a knight, bishop, rook, or queen (sliding or knight)
    #[inline]
    pub fn is_sliding_or_knight(self) -> bool {
        matches!(self, PieceType::Knight | PieceType::Bishop | PieceType::Rook | PieceType::Queen)
    }

    /// Check if piece is a sliding piece (bishop, rook, queen)
    #[inline]
    pub fn is_sliding(self) -> bool {
        matches!(self, PieceType::Bishop | PieceType::Rook | PieceType::Queen)
    }
}

/// A piece with both color and type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    pub color: Color,
    pub piece_type: PieceType,
}

impl Piece {
    /// Create a new piece
    #[inline]
    pub fn new(color: Color, piece_type: PieceType) -> Self {
        Piece { color, piece_type }
    }

    /// Get piece index for array access (0-11)
    /// White: 0-5, Black: 6-11
    #[inline]
    pub fn index(self) -> usize {
        self.color.index() * 6 + self.piece_type.index()
    }

    /// Get piece from index (0-11)
    #[inline]
    pub fn from_index(idx: usize) -> Self {
        let color = if idx < 6 { Color::White } else { Color::Black };
        let piece_type = match idx % 6 {
            0 => PieceType::Pawn,
            1 => PieceType::Knight,
            2 => PieceType::Bishop,
            3 => PieceType::Rook,
            4 => PieceType::Queen,
            5 => PieceType::King,
            _ => unreachable!(),
        };
        Piece { color, piece_type }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_flip() {
        assert_eq!(Color::White.flip(), Color::Black);
        assert_eq!(Color::Black.flip(), Color::White);
    }

    #[test]
    fn test_piece_index() {
        let white_pawn = Piece::new(Color::White, PieceType::Pawn);
        assert_eq!(white_pawn.index(), 0);

        let black_king = Piece::new(Color::Black, PieceType::King);
        assert_eq!(black_king.index(), 11);

        let white_queen = Piece::new(Color::White, PieceType::Queen);
        assert_eq!(white_queen.index(), 4);
    }

    #[test]
    fn test_piece_from_index() {
        let piece = Piece::from_index(0);
        assert_eq!(piece.color, Color::White);
        assert_eq!(piece.piece_type, PieceType::Pawn);

        let piece = Piece::from_index(7);
        assert_eq!(piece.color, Color::Black);
        assert_eq!(piece.piece_type, PieceType::Bishop);
    }
}
