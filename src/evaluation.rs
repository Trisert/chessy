use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::piece::{Color, PieceType};
use crate::position::Position;
use crate::utils::*;

/// Evaluation function with SIMD optimizations
pub struct Evaluation;

// SIMD support for x86_64
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Material values
const MATERIAL_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 0]; // Pawn, Knight, Bishop, Rook, Queen, King

/// Compute material score using AVX2-optimized parallel operations
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn avx2_material_count(board: &Board, color: Color) -> i32 {
    // Get bitboards for all piece types
    let pieces = [
        board.piece_bb(PieceType::Pawn, color).as_u64(),
        board.piece_bb(PieceType::Knight, color).as_u64(),
        board.piece_bb(PieceType::Bishop, color).as_u64(),
        board.piece_bb(PieceType::Rook, color).as_u64(),
        board.piece_bb(PieceType::Queen, color).as_u64(),
        board.piece_bb(PieceType::King, color).as_u64(),
    ];

    // SAFETY: This function is marked unsafe and requires AVX2 target feature
    // The caller must ensure CPU supports AVX2 (checked via target_feature)
    unsafe {
        // Count bits using hardware popcnt
        let counts = _mm256_set_epi32(
            pieces[5].count_ones() as i32, // King (0)
            pieces[4].count_ones() as i32, // Queen (900)
            pieces[3].count_ones() as i32, // Rook (500)
            pieces[2].count_ones() as i32, // Bishop (330)
            pieces[1].count_ones() as i32, // Knight (320)
            pieces[0].count_ones() as i32, // Pawn (100)
            0,
            0, // Padding
        );

        // Load material values
        let values = _mm256_set_epi32(
            MATERIAL_VALUES[5], // King
            MATERIAL_VALUES[4], // Queen
            MATERIAL_VALUES[3], // Rook
            MATERIAL_VALUES[2], // Bishop
            MATERIAL_VALUES[1], // Knight
            MATERIAL_VALUES[0], // Pawn
            0,
            0, // Padding
        );

        // Parallel multiply all 6 piece types at once
        let products = _mm256_mullo_epi32(counts, values);

        // Horizontal sum using AVX2
        // Extract lower and upper 128-bit halves
        let low = _mm256_castsi256_si128(products);
        let high = _mm256_extracti128_si256(products, 1);

        // Add them together
        let sum128 = _mm_add_epi32(low, high);

        // Horizontal sum within 128-bit
        let sum1 = _mm_hadd_epi32(sum128, sum128);
        let sum2 = _mm_hadd_epi32(sum1, sum1);

        _mm_extract_epi32(sum2, 0)
    }
}

/// SSE4.2 fallback for material counting
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
#[inline]
unsafe fn avx2_material_count(board: &Board, color: Color) -> i32 {
    const PAWN_VALUE: i32 = 100;
    const KNIGHT_VALUE: i32 = 320;
    const BISHOP_VALUE: i32 = 330;
    const ROOK_VALUE: i32 = 500;
    const QUEEN_VALUE: i32 = 900;

    // Get bitboards for all piece types
    let pawns = board.piece_bb(PieceType::Pawn, color).as_u64();
    let knights = board.piece_bb(PieceType::Knight, color).as_u64();
    let bishops = board.piece_bb(PieceType::Bishop, color).as_u64();
    let rooks = board.piece_bb(PieceType::Rook, color).as_u64();
    let queens = board.piece_bb(PieceType::Queen, color).as_u64();

    unsafe {
        // Use hardware popcnt for each - already fast on modern CPUs
        // Pack counts into SIMD registers for parallel multiply
        let counts = _mm_set_epi32(
            queens.count_ones() as i32,
            rooks.count_ones() as i32,
            bishops.count_ones() as i32,
            knights.count_ones() as i32,
        );

        let values = _mm_set_epi32(QUEEN_VALUE, ROOK_VALUE, BISHOP_VALUE, KNIGHT_VALUE);

        // Parallel multiply
        let products = _mm_mullo_epi32(counts, values);

        // Horizontal sum
        let sum1 = _mm_hadd_epi32(products, products);
        let sum2 = _mm_hadd_epi32(sum1, sum1);
        let minor_score = _mm_extract_epi32(sum2, 0);

        // Add pawns separately (can't fit in 128-bit with 5 values easily)
        let pawn_score = (pawns.count_ones() as i32) * PAWN_VALUE;

        minor_score + pawn_score
    }
}

/// Fallback material counting for non-x86_64
#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn avx2_material_count(board: &Board, color: Color) -> i32 {
    const PAWN_VALUE: i32 = 100;
    const KNIGHT_VALUE: i32 = 320;
    const BISHOP_VALUE: i32 = 330;
    const ROOK_VALUE: i32 = 500;
    const QUEEN_VALUE: i32 = 900;

    let pawns = board.piece_bb(PieceType::Pawn, color).count() as i32 * PAWN_VALUE;
    let knights = board.piece_bb(PieceType::Knight, color).count() as i32 * KNIGHT_VALUE;
    let bishops = board.piece_bb(PieceType::Bishop, color).count() as i32 * BISHOP_VALUE;
    let rooks = board.piece_bb(PieceType::Rook, color).count() as i32 * ROOK_VALUE;
    let queens = board.piece_bb(PieceType::Queen, color).count() as i32 * QUEEN_VALUE;

    pawns + knights + bishops + rooks + queens
}

/// Wrapper for material counting that handles unsafe block
#[inline]
fn simd_material_count(board: &Board, color: Color) -> i32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        avx2_material_count(board, color)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        avx2_material_count(board, color)
    }
}

/// Piece-square table evaluation
#[inline]
fn pst_eval(pst: &[i32; 64], bitboard: u64) -> i32 {
    let mut score = 0;
    let mut bb = bitboard;
    while bb != 0 {
        let sq = bb.trailing_zeros() as usize;
        bb &= bb - 1;
        score += pst[sq];
    }
    score
}

impl Evaluation {
    /// Evaluate a position from the perspective of the side to move
    /// Positive score means advantage for white, negative for black
    pub fn evaluate(board: &Board) -> i32 {
        let mut score = 0;

        // Material evaluation (SIMD-optimized)
        let white_material = simd_material_count(board, Color::White);
        let black_material = simd_material_count(board, Color::Black);
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

        // Threat detection - penalty for pieces under attack
        let white_threats = Self::count_threats(board, Color::White);
        let black_threats = Self::count_threats(board, Color::Black);
        score -= white_threats * 20; // Penalty for our threatened pieces
        score += black_threats * 20; // Bonus for threatening opponent

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

    /// Calculate material score for a color (SIMD-optimized)
    #[allow(dead_code)]
    fn material(board: &Board, color: Color) -> i32 {
        simd_material_count(board, color)
    }

    /// Calculate piece-square table score for a color
    fn piece_square_tables(board: &Board, color: Color) -> i32 {
        let mut score = 0;

        // Pawn PST
        let pawn_bb = board.piece_bb(PieceType::Pawn, color).as_u64();
        score += pst_eval(&Self::get_pawn_pst(), pawn_bb);

        // Knight PST
        let knight_bb = board.piece_bb(PieceType::Knight, color).as_u64();
        score += pst_eval(&Self::get_knight_pst(), knight_bb);

        // Bishop PST
        let bishop_bb = board.piece_bb(PieceType::Bishop, color).as_u64();
        score += pst_eval(&Self::get_bishop_pst(), bishop_bb);

        // Rook PST
        let rook_bb = board.piece_bb(PieceType::Rook, color).as_u64();
        score += pst_eval(&Self::get_rook_pst(), rook_bb);

        // Queen PST
        let queen_bb = board.piece_bb(PieceType::Queen, color).as_u64();
        score += pst_eval(&Self::get_queen_pst(), queen_bb);

        // King PST
        let king_bb = board.piece_bb(PieceType::King, color).as_u64();
        score += pst_eval(&Self::get_king_pst(), king_bb);

        score
    }

    /// Get pawn PST (white perspective) - static const array for performance
    #[inline]
    fn get_pawn_pst() -> &'static [i32; 64] {
        const PAWN_PST: [i32; 64] = {
            let mut pst = [0i32; 64];
            let mut sq = 0;
            while sq < 64 {
                let rank = sq / 8;
                let file = sq % 8;
                let base = [0, 5, 10, 20, 30, 40, 50, 0][rank];
                let center_bonus = match file {
                    3 | 4 => 5,
                    2 | 5 => 2,
                    _ => 0,
                };
                pst[sq] = base + center_bonus;
                sq += 1;
            }
            pst
        };
        &PAWN_PST
    }

    /// Get knight PST - static const array for performance
    #[inline]
    fn get_knight_pst() -> &'static [i32; 64] {
        const KNIGHT_PST: [i32; 64] = [
            -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40, -30, 0, 10, 15,
            15, 10, 0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5,
            10, 15, 15, 10, 5, -30, -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30,
            -40, -50,
        ];
        &KNIGHT_PST
    }

    /// Get bishop PST - static const array for performance
    #[inline]
    fn get_bishop_pst() -> &'static [i32; 64] {
        const BISHOP_PST: [i32; 64] = [
            -20, -10, -10, -10, -10, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 10, 10,
            5, 0, -10, -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 10,
            10, 10, 10, 10, -10, -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10,
            -20,
        ];
        &BISHOP_PST
    }

    /// Get rook PST - static const array for performance
    #[inline]
    fn get_rook_pst() -> &'static [i32; 64] {
        const ROOK_PST: [i32; 64] = [
            0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 10, 10, 10, 5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0,
            0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0,
            0, 0, -5, 0, 0, 0, 5, 5, 0, 0, 0,
        ];
        &ROOK_PST
    }

    /// Get queen PST - static const array for performance
    #[inline]
    fn get_queen_pst() -> &'static [i32; 64] {
        const QUEEN_PST: [i32; 64] = [
            -20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5,
            0, -10, -5, 0, 5, 5, 5, 5, 0, -5, 0, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10,
            -10, 0, 5, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20,
        ];
        &QUEEN_PST
    }

    /// Get king PST (midgame) - static const array for performance
    #[inline]
    fn get_king_pst() -> &'static [i32; 64] {
        const KING_PST: [i32; 64] = [
            -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30,
            -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -20, -30,
            -30, -40, -40, -30, -30, -20, -10, -20, -20, -20, -20, -20, -20, -10, 20, 20, 0, 0, 0,
            0, 20, 20, 20, 30, 10, 0, 0, 10, 30, 20,
        ];
        &KING_PST
    }

    /// Pawn piece-square table lookup
    #[allow(dead_code)]
    fn _pawn_pst(sq: Square, color: Color) -> i32 {
        let sq = if color == Color::White {
            sq
        } else {
            flip_square(sq)
        };
        let pst = Self::get_pawn_pst();
        pst[sq as usize]
    }

    /// Knight piece-square table lookup
    #[allow(dead_code)]
    fn _knight_pst(sq: Square, color: Color) -> i32 {
        let sq = if color == Color::White {
            sq
        } else {
            flip_square(sq)
        };
        let pst = Self::get_knight_pst();
        pst[sq as usize]
    }

    /// Bishop piece-square table lookup
    #[allow(dead_code)]
    fn _bishop_pst(sq: Square, color: Color) -> i32 {
        let sq = if color == Color::White {
            sq
        } else {
            flip_square(sq)
        };
        let pst = Self::get_bishop_pst();
        pst[sq as usize]
    }

    /// Rook piece-square table lookup
    #[allow(dead_code)]
    fn _rook_pst(sq: Square, color: Color) -> i32 {
        let sq = if color == Color::White {
            sq
        } else {
            flip_square(sq)
        };
        let pst = Self::get_rook_pst();
        pst[sq as usize]
    }

    /// Queen piece-square table lookup
    #[allow(dead_code)]
    fn _queen_pst(sq: Square, color: Color) -> i32 {
        let sq = if color == Color::White {
            sq
        } else {
            flip_square(sq)
        };
        let pst = Self::get_queen_pst();
        pst[sq as usize]
    }

    /// King piece-square table lookup
    #[allow(dead_code)]
    fn _king_pst(sq: Square, color: Color) -> i32 {
        let sq = if color == Color::White {
            sq
        } else {
            flip_square(sq)
        };
        let pst = Self::get_king_pst();
        pst[sq as usize]
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
            let has_left_support =
                file > 0 && (pawns & Bitboard::file_mask(file - 1)).as_u64() != 0;
            let has_right_support =
                file < 7 && (pawns & Bitboard::file_mask(file + 1)).as_u64() != 0;

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
        for check_file in
            if file > 0 { file - 1 } else { file }..=if file < 7 { file + 1 } else { file }
        {
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

        let king_file = file_of(king_sq);
        let king_rank = rank_of(king_sq);
        let opponent = color.flip();

        let pawns = board.piece_bb(PieceType::Pawn, color);
        let opponent_pawns = board.piece_bb(PieceType::Pawn, opponent);

        // === CHECK DETECTION - CRITICAL ===
        // Huge penalty if king is in check
        if Self::is_square_attacked_by(board, king_sq, opponent) {
            score -= 150; // Massive penalty for being in check
        }

        // Count attackers near king (within 2 squares)
        let king_danger = Self::count_king_attackers(board, king_sq, opponent);
        score -= king_danger * 30; // 30 points per attacker near king

        // Determine game phase based on total material
        let total_material =
            (board.count_material(Color::White) + board.count_material(Color::Black)) as usize;
        let is_middlegame = total_material > 3000; // More than queen + minor piece remaining

        // === Pawn Shield Evaluation ===
        if is_middlegame {
            // Check pawn shield on two ranks in front of king
            let shield_ranks: Vec<u8> = if color == Color::White {
                vec![king_rank + 1, king_rank + 2]
            } else {
                if king_rank >= 2 {
                    vec![king_rank - 1, king_rank - 2]
                } else {
                    vec![king_rank - 1]
                }
            };

            for shield_rank in shield_ranks {
                if shield_rank >= 8 {
                    continue;
                }

                let rank_weight = if color == Color::White {
                    (shield_rank - king_rank) as i32
                } else {
                    (king_rank - shield_rank) as i32
                };

                // Check pawns on king's file and adjacent files
                for check_file in if king_file > 0 {
                    king_file - 1
                } else {
                    king_file
                }..=if king_file < 7 {
                    king_file + 1
                } else {
                    king_file
                } {
                    let shield_sq = square_of(shield_rank, check_file);
                    if pawns.get(shield_sq) {
                        // Bonus for having pawn shield
                        score += 20 / rank_weight.max(1);
                    } else {
                        // Penalty for missing pawn shield (higher penalty for first rank)
                        let penalty = if rank_weight == 1 { 25 } else { 15 };
                        score -= penalty;

                        // Extra penalty if the square is attacked by opponent
                        if board.is_occupied_by(shield_sq, opponent) {
                            score -= 10;
                        }
                    }
                }
            }
        }

        // === King Exposure Penalty (for middlegame) ===
        if is_middlegame {
            let center_files = king_file >= 2 && king_file <= 5;
            let center_ranks = king_rank >= 2 && king_rank <= 5;

            if center_files && center_ranks {
                score -= 30; // Penalty for king in center
            }

            // Penalty for king on open/semi-open files
            let file_mask = Bitboard::file_mask(king_file);
            let pawns_on_file = (pawns & file_mask).count();

            if pawns_on_file == 0 {
                // Open file - no pawns on king's file
                score -= 20;
            } else if pawns_on_file == 1 {
                // Semi-open file - only one pawn
                score -= 10;
            }
        }

        // === Pawn Storm Penalties ===
        // Penalty when opponent pawns are advancing near our king
        if is_middlegame {
            let storm_distance = if color == Color::White {
                // For white king, check black pawns on higher ranks
                (7 - king_rank).min(4)
            } else {
                // For black king, check white pawns on lower ranks
                king_rank.min(4)
            };

            for check_file in if king_file > 1 {
                king_file - 1
            } else {
                king_file
            }..=if king_file < 6 {
                king_file + 1
            } else {
                king_file
            } {
                // Check for opponent pawns in storm zone
                let storm_ranks: Vec<u8> = if color == Color::White {
                    ((king_rank + 1)..=(king_rank + storm_distance))
                        .filter(|&r| r < 8)
                        .collect()
                } else {
                    (if king_rank >= storm_distance {
                        king_rank - storm_distance
                    } else {
                        0
                    }..king_rank)
                        .collect()
                };

                for storm_rank in storm_ranks {
                    let storm_sq = square_of(storm_rank, check_file);
                    if opponent_pawns.get(storm_sq) {
                        // Penalty based on how close the pawn is
                        let distance = if color == Color::White {
                            (storm_rank - king_rank) as i32
                        } else {
                            (king_rank - storm_rank) as i32
                        };
                        score -= 15 / distance.max(1);
                    }
                }
            }
        }

        // Endgame: King should be more centralized
        if !is_middlegame {
            let center_squares = Bitboard(0x0000001818000000); // d4, d5, e4, e5
            let king_center_dist = if center_squares.get(king_sq) {
                0
            } else {
                // Calculate distance to center (simplified)
                let file_dist = if king_file <= 3 {
                    3 - king_file
                } else {
                    king_file - 4
                };
                let rank_dist = if king_rank <= 3 {
                    3 - king_rank
                } else {
                    king_rank - 4
                };
                (file_dist + rank_dist) as i32
            };
            score -= king_center_dist * 5; // Penalty for king away from center in endgame
        }

        score
    }

    /// Count how many opponent pieces can attack squares near the king
    fn count_king_attackers(board: &Board, king_sq: Square, by_color: Color) -> i32 {
        let mut attackers = 0;
        let king_file = file_of(king_sq);
        let king_rank = rank_of(king_sq);

        // Check the 3x3 area around the king (king + adjacent squares)
        for rank_offset in -1i8..=1 {
            for file_offset in -1i8..=1 {
                let new_rank = (king_rank as i8 + rank_offset) as u8;
                let new_file = (king_file as i8 + file_offset) as u8;

                if new_rank >= 8 || new_file >= 8 {
                    continue;
                }

                let target_sq = square_of(new_rank, new_file);

                // Check if opponent has pieces attacking this square
                // We use a simplified check - is there an opponent piece that could move here?
                let occupied = board.occupied();

                // Check knights
                let knight_sources = crate::movegen::MoveGen::knight_attacks(target_sq);
                if (knight_sources & board.piece_bb(PieceType::Knight, by_color)).as_u64() != 0 {
                    attackers += 1;
                }

                // Check bishops/queens (diagonal)
                let bishop_attacks = crate::movegen::MoveGen::bishop_attacks_on_the_fly(target_sq, occupied);
                let bishop_queen = board.piece_bb(PieceType::Bishop, by_color)
                    | board.piece_bb(PieceType::Queen, by_color);
                if (bishop_attacks & bishop_queen).as_u64() != 0 {
                    attackers += 1;
                }

                // Check rooks/queens (straight)
                let rook_attacks = crate::movegen::MoveGen::rook_attacks_on_the_fly(target_sq, occupied);
                let rook_queen =
                    board.piece_bb(PieceType::Rook, by_color) | board.piece_bb(PieceType::Queen, by_color);
                if (rook_attacks & rook_queen).as_u64() != 0 {
                    attackers += 1;
                }
            }
        }

        attackers
    }

    /// Count how many pieces of a color are under attack
    fn count_threats(board: &Board, color: Color) -> i32 {
        let mut threatened = 0;
        let opponent = color.flip();

        // Check each piece type
        for &piece_type in &[
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ] {
            let pieces = board.piece_bb(piece_type, color);
            for sq in pieces.squares() {
                // Check if this square is attacked by opponent
                if Self::is_square_attacked_by(board, sq, opponent) {
                    threatened += 1;
                }
            }
        }

        threatened
    }

    /// Check if a square is attacked by a specific color
    fn is_square_attacked_by(board: &Board, sq: Square, by_color: Color) -> bool {
        // Check pawn attacks
        let pawn_attacks = if by_color == Color::White {
            // White pawns attack from below (rank-1 to sq)
            let rank = rank_of(sq);
            let file = file_of(sq);
            let mut attacks = Bitboard::EMPTY;
            if rank > 0 && file > 0 {
                attacks.set(square_of(rank - 1, file - 1));
            }
            if rank > 0 && file < 7 {
                attacks.set(square_of(rank - 1, file + 1));
            }
            attacks
        } else {
            // Black pawns attack from above (rank+1 to sq)
            let rank = rank_of(sq);
            let file = file_of(sq);
            let mut attacks = Bitboard::EMPTY;
            if rank < 7 && file > 0 {
                attacks.set(square_of(rank + 1, file - 1));
            }
            if rank < 7 && file < 7 {
                attacks.set(square_of(rank + 1, file + 1));
            }
            attacks
        };

        if (pawn_attacks & board.piece_bb(PieceType::Pawn, by_color)).as_u64() != 0 {
            return true;
        }

        // Check knight attacks
        let knight_attacks = crate::movegen::MoveGen::knight_attacks(sq);
        if (knight_attacks & board.piece_bb(PieceType::Knight, by_color)).as_u64() != 0 {
            return true;
        }

        // Check king attacks (for nearby kings)
        let king_attacks = crate::movegen::MoveGen::king_attacks(sq);
        if (king_attacks & board.piece_bb(PieceType::King, by_color)).as_u64() != 0 {
            return true;
        }

        let occupied = board.occupied();

        // Check bishop/queen diagonal attacks
        let diagonal_attacks = crate::movegen::MoveGen::bishop_attacks_on_the_fly(sq, occupied);
        let bishop_queen = board.piece_bb(PieceType::Bishop, by_color)
            | board.piece_bb(PieceType::Queen, by_color);
        if (diagonal_attacks & bishop_queen).as_u64() != 0 {
            return true;
        }

        // Check rook/queen straight attacks
        let straight_attacks = crate::movegen::MoveGen::rook_attacks_on_the_fly(sq, occupied);
        let rook_queen =
            board.piece_bb(PieceType::Rook, by_color) | board.piece_bb(PieceType::Queen, by_color);
        if (straight_attacks & rook_queen).as_u64() != 0 {
            return true;
        }

        false
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
