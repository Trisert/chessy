use crate::utils::Square;

/// Move representation - 16-bit encoding
///
/// Bits 0-5:   Source square (0-63)
/// Bits 6-11:  Destination square (0-63)
/// Bits 12-14: Promotion piece (0-3) or special flag
/// Bit 15:     Special move flag (castle, en passant, promotion)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move(pub u16);

impl Move {
    /// Create a new normal move
    #[inline]
    pub fn new(from: Square, to: Square) -> Self {
        Move((from as u16) | ((to as u16) << 6))
    }

    /// Create a promotion move
    #[inline]
    pub fn promotion(from: Square, to: Square, promotion: PromotionType) -> Self {
        Move((from as u16) | ((to as u16) << 6) | ((promotion as u16) << 12) | (1 << 15))
    }

    /// Create an en passant move
    #[inline]
    pub fn en_passant(from: Square, to: Square) -> Self {
        Move((from as u16) | ((to as u16) << 6) | (1 << 12) | (1 << 15))
    }

    /// Create a castling move
    #[inline]
    pub fn castle(from: Square, to: Square) -> Self {
        Move((from as u16) | ((to as u16) << 6) | (2 << 12) | (1 << 15))
    }

    /// Get the source square
    #[inline]
    pub const fn from(self) -> Square {
        (self.0 & 0x3F) as Square
    }

    /// Get the destination square
    #[inline]
    pub const fn to(self) -> Square {
        ((self.0 >> 6) & 0x3F) as Square
    }

    /// Check if this is a promotion move
    #[inline]
    pub const fn is_promotion(self) -> bool {
        ((self.0 >> 15) & 1) == 1 && ((self.0 >> 12) & 7) >= 4 && ((self.0 >> 12) & 7) <= 7
    }

    /// Check if this is an en passant move
    #[inline]
    pub const fn is_en_passant(self) -> bool {
        (self.0 & 0xF000) == 0x9000
    }

    /// Check if this is a castling move
    #[inline]
    pub const fn is_castle(self) -> bool {
        (self.0 & 0xF000) == 0xA000
    }

    /// Check if this is a capture move
    /// Note: This needs board state to be accurate for en passant
    #[inline]
    pub const fn is_capture(self) -> bool {
        // This is a simplified check - actual implementation needs board context
        (self.0 & 0x1000) != 0
    }

    /// Get the promotion piece type (if any)
    #[inline]
    pub const fn promotion_type(self) -> Option<PromotionType> {
        if self.is_promotion() {
            Some(match (self.0 >> 12) & 0x3 {
                0 => PromotionType::Knight,
                1 => PromotionType::Bishop,
                2 => PromotionType::Rook,
                3 => PromotionType::Queen,
                _ => unreachable!(),
            })
        } else {
            None
        }
    }

    /// Check if move is null (no move)
    #[inline]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    /// Create a null move
    #[inline]
    pub const fn null() -> Self {
        Move(0)
    }
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_null() {
            write!(f, "0000")
        } else {
            let from = crate::utils::square_to_string(self.from());
            let to = crate::utils::square_to_string(self.to());

            if let Some(promotion) = self.promotion_type() {
                let prom_char = match promotion {
                    PromotionType::Knight => 'n',
                    PromotionType::Bishop => 'b',
                    PromotionType::Rook => 'r',
                    PromotionType::Queen => 'q',
                };
                write!(f, "{}{}{}", from, to, prom_char)
            } else {
                write!(f, "{}{}", from, to)
            }
        }
    }
}

/// Promotion piece types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PromotionType {
    Knight = 0,
    Bishop = 1,
    Rook = 2,
    Queen = 3,
}

impl PromotionType {
    /// Get promotion type from index
    #[inline]
    pub fn from_index(idx: u8) -> Self {
        match idx {
            0 => PromotionType::Knight,
            1 => PromotionType::Bishop,
            2 => PromotionType::Rook,
            3 => PromotionType::Queen,
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_creation() {
        let mv = Move::new(0, 1);
        assert_eq!(mv.from(), 0);
        assert_eq!(mv.to(), 1);
        assert!(!mv.is_promotion());
        assert!(!mv.is_castle());
        assert!(!mv.is_en_passant());
    }

    #[test]
    fn test_promotion_move() {
        let mv = Move::promotion(12, 20, PromotionType::Queen);
        assert_eq!(mv.from(), 12);
        assert_eq!(mv.to(), 20);
        assert!(mv.is_promotion());
        assert_eq!(mv.promotion_type(), Some(PromotionType::Queen));
    }

    #[test]
    fn test_castle_move() {
        let mv = Move::castle(4, 6); // e1g1 (kingside castle)
        assert_eq!(mv.from(), 4);
        assert_eq!(mv.to(), 6);
        assert!(mv.is_castle());
    }

    #[test]
    fn test_en_passant_move() {
        let mv = Move::en_passant(12, 21); // exd5 e.p.
        assert_eq!(mv.from(), 12);
        assert_eq!(mv.to(), 21);
        assert!(mv.is_en_passant());
    }

    #[test]
    fn test_null_move() {
        let mv = Move::null();
        assert!(mv.is_null());
    }

    #[test]
    fn test_move_display() {
        let mv = Move::new(12, 28); // e2e4
        assert_eq!(mv.to_string(), "e2e4");

        let mv = Move::promotion(12, 20, PromotionType::Queen); // e7e8q
        assert_eq!(mv.to_string(), "e7e8q");

        let mv = Move::null();
        assert_eq!(mv.to_string(), "0000");
    }
}
