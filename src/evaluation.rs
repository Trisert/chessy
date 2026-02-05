use crate::board::Board;
use crate::bitboard::Bitboard;
use crate::piece::{Color, PieceType};
use crate::position::Position;
use crate::utils::*;

/// Evaluation function
pub struct Evaluation;

impl Evaluation {
    /// Evaluate a position from the perspective of the side to move
    /// Positive score means advantage for white, negative for black
    pub fn evaluate(board: &Board) -> i32 {
        let mut score = 0;

        // Material evaluation
        let white_material = Self::material(board, Color::White);
        let black_material = Self::material(board, Color::Black);
        score += white_material - black_material;

        // Piece-square evaluation
        let white_pst = Self::piece_square_tables(board, Color::White);
        let black_pst = Self::piece_square_tables(board, Color::Black);
        score += white_pst - black_pst;

        // Pawn structure evaluation
        let white_pawns = Self::pawn_structure(board, Color::White);
        let black_pawns = Self::pawn_structure(board, Color::Black);
        score += white_pawns - black_pawns;

        // Piece activity
        let white_activity = Self::piece_activity(board, Color::White);
        let black_activity = Self::piece_activity(board, Color::Black);
        score += white_activity - black_activity;

        // King safety
        let white_king = Self::king_safety(board, Color::White);
        let black_king = Self::king_safety(board, Color::Black);
        score += white_king - black_king;

        score
    }

    /// Evaluate a position considering repetition history
    pub fn evaluate_with_history(position: &Position) -> i32 {
        let mut score = Self::evaluate(&position.board);

        // Penalize positions that have occurred before
        let current_hash = position.state.hash;
        let mut repetition_count = 0;

        for &hash in &position.history_hashes {
            if hash == current_hash {
                repetition_count += 1;
            }
        }

        // Heavy penalty for repetitions to avoid draws
        if repetition_count >= 2 {
            score = 0; // This is a threefold repetition - draw
        } else if repetition_count == 1 {
            // Penalize second occurrence to discourage further repetitions
            score /= 4;
        }

        score
    }

    /// Calculate material score for a color
    fn material(board: &Board, color: Color) -> i32 {
        const PAWN_VALUE: i32 = 100;
        const KNIGHT_VALUE: i32 = 320;
        const BISHOP_VALUE: i32 = 330;
        const ROOK_VALUE: i32 = 500;
        const QUEEN_VALUE: i32 = 900;
        const KING_VALUE: i32 = 0;

        let pawns = board.piece_bb(PieceType::Pawn, color).count() as i32 * PAWN_VALUE;
        let knights = board.piece_bb(PieceType::Knight, color).count() as i32 * KNIGHT_VALUE;
        let bishops = board.piece_bb(PieceType::Bishop, color).count() as i32 * BISHOP_VALUE;
        let rooks = board.piece_bb(PieceType::Rook, color).count() as i32 * ROOK_VALUE;
        let queens = board.piece_bb(PieceType::Queen, color).count() as i32 * QUEEN_VALUE;
        let kings = board.piece_bb(PieceType::King, color).count() as i32 * KING_VALUE;

        pawns + knights + bishops + rooks + queens + kings
    }

    /// Calculate piece-square table score for a color
    fn piece_square_tables(board: &Board, color: Color) -> i32 {
        let mut score = 0;

        // Pawn PST
        for sq in board.piece_bb(PieceType::Pawn, color).squares() {
            score += Self::pawn_pst(sq, color);
        }

        // Knight PST
        for sq in board.piece_bb(PieceType::Knight, color).squares() {
            score += Self::knight_pst(sq, color);
        }

        // Bishop PST
        for sq in board.piece_bb(PieceType::Bishop, color).squares() {
            score += Self::bishop_pst(sq, color);
        }

        // Rook PST
        for sq in board.piece_bb(PieceType::Rook, color).squares() {
            score += Self::rook_pst(sq, color);
        }

        // Queen PST
        for sq in board.piece_bb(PieceType::Queen, color).squares() {
            score += Self::queen_pst(sq, color);
        }

        // King PST (midgame)
        for sq in board.piece_bb(PieceType::King, color).squares() {
            score += Self::king_pst(sq, color);
        }

        score
    }

    /// Pawn piece-square table
    fn pawn_pst(sq: Square, color: Color) -> i32 {
        let sq = if color == Color::White { sq } else { flip_square(sq) };
        let rank = rank_of(sq);
        let file = file_of(sq);

        // Encourage central pawns and pawn advancement
        let base = [0, 5, 10, 20, 30, 40, 50, 0][rank as usize];

        // Center bonus
        let center_bonus = match file {
            3 | 4 => 5, // d and e files
            2 | 5 => 2, // c and f files
            _ => 0,
        };

        base + center_bonus
    }

    /// Knight piece-square table
    fn knight_pst(sq: Square, color: Color) -> i32 {
        let sq = if color == Color::White { sq } else { flip_square(sq) };
        const PST: [i32; 64] = [
            -50, -40, -30, -30, -30, -30, -40, -50, // Rank 8
            -40, -20, 0, 0, 0, 0, -20, -40,         // Rank 7
            -30, 0, 10, 15, 15, 10, 0, -30,        // Rank 6
            -30, 5, 15, 20, 20, 15, 5, -30,        // Rank 5
            -30, 0, 15, 20, 20, 15, 0, -30,        // Rank 4
            -30, 5, 10, 15, 15, 10, 5, -30,        // Rank 3
            -40, -20, 0, 5, 5, 0, -20, -40,        // Rank 2
            -50, -40, -30, -30, -30, -30, -40, -50, // Rank 1
        ];
        PST[sq as usize]
    }

    /// Bishop piece-square table
    fn bishop_pst(sq: Square, color: Color) -> i32 {
        let sq = if color == Color::White { sq } else { flip_square(sq) };
        const PST: [i32; 64] = [
            -20, -10, -10, -10, -10, -10, -10, -20, // Rank 8
            -10, 0, 0, 0, 0, 0, 0, -10,             // Rank 7
            -10, 0, 5, 10, 10, 5, 0, -10,           // Rank 6
            -10, 5, 5, 10, 10, 5, 5, -10,           // Rank 5
            -10, 0, 10, 10, 10, 10, 0, -10,         // Rank 4
            -10, 10, 10, 10, 10, 10, 10, -10,       // Rank 3
            -10, 5, 0, 0, 0, 0, 5, -10,             // Rank 2
            -20, -10, -10, -10, -10, -10, -10, -20, // Rank 1
        ];
        PST[sq as usize]
    }

    /// Rook piece-square table
    fn rook_pst(sq: Square, color: Color) -> i32 {
        let sq = if color == Color::White { sq } else { flip_square(sq) };
        const PST: [i32; 64] = [
            0, 0, 0, 0, 0, 0, 0, 0,   // Rank 8
            5, 10, 10, 10, 10, 10, 10, 5, // Rank 7
            -5, 0, 0, 0, 0, 0, 0, -5, // Rank 6
            -5, 0, 0, 0, 0, 0, 0, -5, // Rank 5
            -5, 0, 0, 0, 0, 0, 0, -5, // Rank 4
            -5, 0, 0, 0, 0, 0, 0, -5, // Rank 3
            -5, 0, 0, 0, 0, 0, 0, -5, // Rank 2
            0, 0, 0, 5, 5, 0, 0, 0,   // Rank 1
        ];
        PST[sq as usize]
    }

    /// Queen piece-square table
    fn queen_pst(sq: Square, color: Color) -> i32 {
        let sq = if color == Color::White { sq } else { flip_square(sq) };
        const PST: [i32; 64] = [
            -20, -10, -10, -5, -5, -10, -10, -20, // Rank 8
            -10, 0, 0, 0, 0, 0, 0, -10,           // Rank 7
            -10, 0, 5, 5, 5, 5, 0, -10,           // Rank 6
            -5, 0, 5, 5, 5, 5, 0, -5,             // Rank 5
            0, 0, 5, 5, 5, 5, 0, -5,              // Rank 4
            -10, 5, 5, 5, 5, 5, 0, -10,           // Rank 3
            -10, 0, 5, 0, 0, 0, 0, -10,           // Rank 2
            -20, -10, -10, -5, -5, -10, -10, -20, // Rank 1
        ];
        PST[sq as usize]
    }

    /// King piece-square table (midgame)
    fn king_pst(sq: Square, color: Color) -> i32 {
        let sq = if color == Color::White { sq } else { flip_square(sq) };
        const PST: [i32; 64] = [
            -30, -40, -40, -50, -50, -40, -40, -30, // Rank 8
            -30, -40, -40, -50, -50, -40, -40, -30, // Rank 7
            -30, -40, -40, -50, -50, -40, -40, -30, // Rank 6
            -30, -40, -40, -50, -50, -40, -40, -30, // Rank 5
            -20, -30, -30, -40, -40, -30, -30, -20, // Rank 4
            -10, -20, -20, -20, -20, -20, -20, -10, // Rank 3
            20, 20, 0, 0, 0, 0, 20, 20,             // Rank 2
            20, 30, 10, 0, 0, 10, 30, 20,           // Rank 1
        ];
        PST[sq as usize]
    }

    /// Evaluate pawn structure
    fn pawn_structure(board: &Board, color: Color) -> i32 {
        let mut score = 0;
        let pawns = board.piece_bb(PieceType::Pawn, color);

        // Count doubled pawns
        for file in 0..8 {
            let pawns_on_file = (pawns & Bitboard::file_mask(file)).count();
            if pawns_on_file > 1 {
                score -= (pawns_on_file as i32 - 1) * 10;
            }
        }

        // Count isolated pawns
        for sq in pawns.squares() {
            let file = file_of(sq);
            let has_left_support = file > 0
                && (pawns & Bitboard::file_mask(file - 1)).as_u64() != 0;
            let has_right_support = file < 7
                && (pawns & Bitboard::file_mask(file + 1)).as_u64() != 0;

            if !has_left_support && !has_right_support {
                score -= 20; // Isolated pawn penalty
            }
        }

        // Count passed pawns (simplified)
        for sq in pawns.squares() {
            if Self::is_passed_pawn(board, sq, color) {
                let rank = rank_of(sq);
                let bonus = match color {
                    Color::White => [0, 10, 20, 30, 50, 80, 120, 0][rank as usize],
                    Color::Black => [0, 120, 80, 50, 30, 20, 10, 0][rank as usize],
                };
                score += bonus;
            }
        }

        score
    }

    /// Check if a pawn is passed
    fn is_passed_pawn(board: &Board, sq: Square, color: Color) -> bool {
        let file = file_of(sq);
        let rank = rank_of(sq);

        let opponent = color.flip();
        let opponent_pawns = board.piece_bb(PieceType::Pawn, opponent);

        // Check files ahead
        for check_file in if file > 0 { file - 1 } else { file }..=if file < 7 {
            file + 1
        } else {
            file
        } {
            let file_mask = Bitboard::file_mask(check_file);

            let mask = match color {
                Color::White => {
                    // White pawn: check ranks above
                    let mut mask = Bitboard::EMPTY;
                    for r in (rank + 1)..8 {
                        mask = mask | Bitboard::rank_mask(r);
                    }
                    mask
                }
                Color::Black => {
                    // Black pawn: check ranks below
                    let mut mask = Bitboard::EMPTY;
                    for r in 0..rank {
                        mask = mask | Bitboard::rank_mask(r);
                    }
                    mask
                }
            };

            if (opponent_pawns & file_mask & mask).as_u64() != 0 {
                return false;
            }
        }

        true
    }

    /// Evaluate piece activity (mobility)
    fn piece_activity(board: &Board, color: Color) -> i32 {
        let mut score = 0;
        let occupied = board.occupied();

        // Bishop pair bonus
        let bishops = board.piece_bb(PieceType::Bishop, color).count();
        if bishops >= 2 {
            score += 30;
        }

        // Knight mobility
        for from in board.piece_bb(PieceType::Knight, color).squares() {
            let moves = crate::movegen::MoveGen::knight_attacks(from);
            let legal_moves = moves & !board.color_bb(color);
            score += (legal_moves.count() as i32 - 4) * 4; // 4 points per available square
        }

        // Bishop mobility
        for from in board.piece_bb(PieceType::Bishop, color).squares() {
            let moves = crate::movegen::MoveGen::bishop_attacks_on_the_fly(from, occupied);
            let legal_moves = moves & !board.color_bb(color);
            score += (legal_moves.count() as i32 - 7) * 3; // 3 points per available square
        }

        // Rook mobility
        for from in board.piece_bb(PieceType::Rook, color).squares() {
            let moves = crate::movegen::MoveGen::rook_attacks_on_the_fly(from, occupied);
            let legal_moves = moves & !board.color_bb(color);
            score += (legal_moves.count() as i32 - 7) * 2; // 2 points per available square
        }

        // Queen mobility
        for from in board.piece_bb(PieceType::Queen, color).squares() {
            let moves = crate::movegen::MoveGen::queen_attacks_on_the_fly(from, occupied);
            let legal_moves = moves & !board.color_bb(color);
            score += (legal_moves.count() as i32 - 14) * 1; // 1 point per available square
        }

        // Control of center
        let center_squares = Bitboard(0x0000001818000000); // d4, d5, e4, e5
        let center_control = (center_squares & !board.color_bb(color)).count();
        score += center_control as i32 * 10;

        score
    }

    /// Evaluate king safety
    fn king_safety(board: &Board, color: Color) -> i32 {
        let mut score = 0;

        let king_sq = match board.king_square(color) {
            Some(sq) => sq,
            None => return -1000, // King captured - huge penalty
        };

        // Pawn shield
        let king_file = file_of(king_sq);
        let king_rank = rank_of(king_sq);

        let pawns = board.piece_bb(PieceType::Pawn, color);

        // Check pawn shield in front of king
        let shield_rank = if color == Color::White {
            king_rank + 1
        } else {
            if king_rank == 0 {
                return score; // King on back rank, no shield needed
            }
            king_rank - 1
        };

        if shield_rank < 8 {
            // Check pawns on king's file and adjacent files
            for check_file in if king_file > 0 { king_file - 1 } else { king_file }
                ..=if king_file < 7 { king_file + 1 } else { king_file }
            {
                let shield_sq = square_of(shield_rank, check_file);
                if !pawns.get(shield_sq) {
                    score -= 15; // Penalty for missing pawn shield
                }
            }
        }

        // King exposure (penalty for king in center in opening/middlegame)
        let center_files = king_file >= 2 && king_file <= 5;
        let center_ranks = king_rank >= 2 && king_rank <= 5;
        if center_files && center_ranks {
            score -= 20; // Penalty for king in center
        }

        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_start() {
        let board = Board::from_start();
        let score = Evaluation::evaluate(&board);

        // Starting position should be roughly equal
        assert!(score > -100 && score < 100);
    }

    #[test]
    fn test_material_count() {
        let board = Board::from_start();
        let white_mat = Evaluation::material(&board, Color::White);
        let black_mat = Evaluation::material(&board, Color::Black);

        // Starting position material
        assert_eq!(white_mat, 3900);
        assert_eq!(black_mat, 3900);
    }
}
