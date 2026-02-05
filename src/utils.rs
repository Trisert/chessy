/// Square representation (0-63)
///
/// Mapping:
/// A1=0, B1=1, C1=2, D1=3, E1=4, F1=5, G1=6, H1=7
/// A2=8, B2=9, ...                              H2=15
/// ...
/// A8=56, ...                                    H8=63
pub type Square = u8;

/// Rank representation (0-7)
pub type Rank = u8;

/// File representation (0-7)
pub type File = u8;

/// Rank constants
pub const RANK_1: Rank = 0;
pub const RANK_2: Rank = 1;
pub const RANK_3: Rank = 2;
pub const RANK_4: Rank = 3;
pub const RANK_5: Rank = 4;
pub const RANK_6: Rank = 5;
pub const RANK_7: Rank = 6;
pub const RANK_8: Rank = 7;

/// File constants
pub const FILE_A: File = 0;
pub const FILE_B: File = 1;
pub const FILE_C: File = 2;
pub const FILE_D: File = 3;
pub const FILE_E: File = 4;
pub const FILE_F: File = 5;
pub const FILE_G: File = 6;
pub const FILE_H: File = 7;

/// Get rank from square (0-7)
#[inline]
pub const fn rank_of(sq: Square) -> Rank {
    sq / 8
}

/// Get file from square (0-7)
#[inline]
pub const fn file_of(sq: Square) -> File {
    sq % 8
}

/// Get square from rank and file
#[inline]
pub const fn square_of(rank: Rank, file: File) -> Square {
    rank * 8 + file
}

/// Convert square to algebraic notation (e.g., "e4")
#[inline]
pub fn square_to_string(sq: Square) -> String {
    let file = file_of(sq);
    let rank = rank_of(sq);
    let file_char = (b'a' + file) as char;
    let rank_char = (b'1' + rank) as char;
    format!("{}{}", file_char, rank_char)
}

/// Convert algebraic notation to square (e.g., "e4" -> 28)
#[inline]
pub fn square_from_string(s: &str) -> Option<Square> {
    let bytes = s.as_bytes();
    if bytes.len() != 2 {
        return None;
    }

    let file_char = bytes[0];
    let rank_char = bytes[1];

    if file_char < b'a' || file_char > b'h' || rank_char < b'1' || rank_char > b'8' {
        return None;
    }

    Some(square_of(rank_char - b'1', file_char - b'a'))
}

/// Square distance (taxicab distance)
#[inline]
pub const fn square_distance(sq1: Square, sq2: Square) -> u8 {
    let rank_diff = (rank_of(sq1) as i8 - rank_of(sq2) as i8).abs();
    let file_diff = (file_of(sq1) as i8 - file_of(sq2) as i8).abs();
    (rank_diff + file_diff) as u8
}

/// Check if square is on a given rank
#[inline]
pub const fn rank_match(sq: Square, rank: Rank) -> bool {
    rank_of(sq) == rank
}

/// Check if square is on a given file
#[inline]
pub const fn file_match(sq: Square, file: File) -> bool {
    file_of(sq) == file
}

/// Check if square is on same rank as another square
#[inline]
pub const fn same_rank(sq1: Square, sq2: Square) -> bool {
    rank_of(sq1) == rank_of(sq2)
}

/// Check if square is on same file as another square
#[inline]
pub const fn same_file(sq1: Square, sq2: Square) -> bool {
    file_of(sq1) == file_of(sq2)
}

/// Check if square is on same diagonal as another square
#[inline]
pub fn same_diagonal(sq1: Square, sq2: Square) -> bool {
    let rank_diff = (rank_of(sq1) as i8 - rank_of(sq2) as i8).abs();
    let file_diff = (file_of(sq1) as i8 - file_of(sq2) as i8).abs();
    rank_diff == file_diff
}

/// Check if square is light-colored
#[inline]
pub const fn is_light_square(sq: Square) -> bool {
    ((rank_of(sq) + file_of(sq)) % 2) == 1
}

/// Check if square is dark-colored
#[inline]
pub const fn is_dark_square(sq: Square) -> bool {
    ((rank_of(sq) + file_of(sq)) % 2) == 0
}

/// Get the square in front of the given square for a color
#[inline]
pub fn square_in_front(sq: Square, color: crate::piece::Color) -> Option<Square> {
    let rank = rank_of(sq) as i8;
    let new_rank = match color {
        crate::piece::Color::White => rank + 1,
        crate::piece::Color::Black => rank - 1,
    };

    if new_rank < 0 || new_rank > 7 {
        return None;
    }

    Some(square_of(new_rank as u8, file_of(sq)))
}

/// Flipped square (for mirroring positions)
#[inline]
pub const fn flip_square(sq: Square) -> Square {
    sq ^ 56
}

/// Rank masks for each rank
pub const RANK_BB: [u64; 8] = [
    0x00000000000000FF, // Rank 1
    0x000000000000FF00, // Rank 2
    0x0000000000FF0000, // Rank 3
    0x00000000FF000000, // Rank 4
    0x000000FF00000000, // Rank 5
    0x0000FF0000000000, // Rank 6
    0x00FF000000000000, // Rank 7
    0xFF00000000000000, // Rank 8
];

/// File masks for each file
pub const FILE_BB: [u64; 8] = [
    0x0101010101010101, // File A
    0x0202020202020202, // File B
    0x0404040404040404, // File C
    0x0808080808080808, // File D
    0x1010101010101010, // File E
    0x2020202020202020, // File F
    0x4040404040404040, // File G
    0x8080808080808080, // File H
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::Color;

    #[test]
    fn test_square_conversion() {
        // A1 = 0
        assert_eq!(square_of(0, 0), 0);
        assert_eq!(rank_of(0), 0);
        assert_eq!(file_of(0), 0);

        // H1 = 7
        assert_eq!(square_of(0, 7), 7);
        assert_eq!(rank_of(7), 0);
        assert_eq!(file_of(7), 7);

        // A8 = 56
        assert_eq!(square_of(7, 0), 56);
        assert_eq!(rank_of(56), 7);
        assert_eq!(file_of(56), 0);

        // H8 = 63
        assert_eq!(square_of(7, 7), 63);
        assert_eq!(rank_of(63), 7);
        assert_eq!(file_of(63), 7);

        // e4 = 28
        assert_eq!(square_of(3, 4), 28);
        assert_eq!(square_to_string(28), "e4");
        assert_eq!(square_from_string("e4"), Some(28));
    }

    #[test]
    fn test_square_distance() {
        // Same square
        assert_eq!(square_distance(0, 0), 0);

        // Adjacent squares
        assert_eq!(square_distance(0, 1), 1); // A1 to B1
        assert_eq!(square_distance(0, 8), 1); // A1 to A2

        // Knight's move
        assert_eq!(square_distance(0, 10), 3); // A1 to C2

        // Longer distance
        assert_eq!(square_distance(0, 63), 14); // A1 to H8
    }

    #[test]
    fn test_same_diagonal() {
        // Main diagonal
        assert!(same_diagonal(0, 9));   // A1 to B2
        assert!(same_diagonal(0, 63));  // A1 to H8
        assert!(same_diagonal(7, 56));  // H1 to A8

        // Anti-diagonal
        assert!(same_diagonal(7, 14));  // H1 to B2
        assert!(same_diagonal(56, 49)); // A8 to H1

        // Not on same diagonal
        assert!(!same_diagonal(0, 1));  // A1 to B1
    }

    #[test]
    fn test_square_colors() {
        // A1 is dark (rank 0 + file 0 = 0, even = dark)
        assert!(is_dark_square(0));
        assert!(!is_light_square(0));

        // B1 is light (rank 0 + file 1 = 1, odd = light)
        assert!(is_light_square(1));
        assert!(!is_dark_square(1));

        // e4 (square 28) is light
        assert!(is_light_square(28));
    }

    #[test]
    fn test_square_in_front() {
        // White pawn on e2 (square 12) should have e3 in front
        assert_eq!(square_in_front(12, Color::White), Some(20));

        // Black pawn on e7 (square 52) should have e6 in front
        assert_eq!(square_in_front(52, Color::Black), Some(44));

        // White pawn on rank 8 has no square in front
        assert_eq!(square_in_front(63, Color::White), None);

        // Black pawn on rank 1 has no square in front
        assert_eq!(square_in_front(0, Color::Black), None);
    }

    #[test]
    fn test_flip_square() {
        // Flip A1 (0) -> A8 (56)
        assert_eq!(flip_square(0), 56);

        // Flip H1 (7) -> H8 (63)
        assert_eq!(flip_square(7), 63);

        // Flip e4 (28) -> e5 (35)
        assert_eq!(flip_square(28), 35);
    }
}
