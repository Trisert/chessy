use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::movelist::MoveList;
use crate::piece::{Color, Piece, PieceType};
use crate::position::Position;
use crate::r#move::Move;
use crate::utils::*;
use crate::PromotionType;

/// Move generator
pub struct MoveGen;

impl MoveGen {
    /// Generate all pseudo-legal moves
    pub fn generate_moves(board: &Board, side_to_move: Color) -> MoveList {
        Self::generate_moves_ep(board, side_to_move, None, 0x0F)
    }

    /// Generate all pseudo-legal moves with en passant
    pub fn generate_moves_ep(
        board: &Board,
        side_to_move: Color,
        ep_square: Option<Square>,
        castling_rights: u8,
    ) -> MoveList {
        let mut moves = MoveList::new();

        // Generate moves for each piece type
        Self::generate_pawn_moves_ep(board, side_to_move, ep_square, &mut moves);
        Self::generate_knight_moves(board, side_to_move, &mut moves);
        Self::generate_bishop_moves(board, side_to_move, &mut moves);
        Self::generate_rook_moves(board, side_to_move, &mut moves);
        Self::generate_queen_moves(board, side_to_move, &mut moves);
        Self::generate_king_moves(board, side_to_move, &mut moves);
        Self::generate_castling_moves(board, side_to_move, castling_rights, &mut moves);

        moves
    }

    /// Generate all legal moves (pseudo-legal with check filter)
    pub fn generate_legal_moves(board: &Board, side_to_move: Color) -> MoveList {
        Self::generate_legal_moves_ep(board, side_to_move, None, 0x0F)
    }

    /// Generate all legal moves with en passant and castling rights
    pub fn generate_legal_moves_ep(
        board: &Board,
        side_to_move: Color,
        ep_square: Option<Square>,
        castling_rights: u8,
    ) -> MoveList {
        // Create a temporary Position for legal move checking
        let temp_position = Position::from_board_with_state(
            board.clone(),
            side_to_move,
            castling_rights,
            ep_square,
        );

        let pseudo_moves = Self::generate_moves_ep(board, side_to_move, ep_square, castling_rights);
        let mut legal_moves = MoveList::new();

        for mv in pseudo_moves.iter() {
            if Self::is_move_legal(&temp_position, mv, side_to_move) {
                legal_moves.push(mv);
            }
        }

        legal_moves
    }

    /// Generate moves from a Position (includes en passant and castling rights)
    pub fn generate_from_position(position: &crate::position::Position) -> MoveList {
        Self::generate_legal_moves_ep(
            &position.board,
            position.state.side_to_move,
            position.state.ep_square,
            position.state.castling_rights,
        )
    }

    /// Check if a move is legal (doesn't leave king in check)
    /// Uses Position for proper state handling (castling, en passant, etc.)
    pub fn is_move_legal(position: &Position, mv: Move, side_to_move: Color) -> bool {
        // Make the move using Position to properly update all state
        let mut test_position = position.clone();
        test_position.make_move(mv);

        // Check if king is in check
        let king_sq = match test_position.board.king_square(side_to_move) {
            Some(sq) => sq,
            None => return false, // King captured (shouldn't happen)
        };

        !Self::is_square_attacked(&test_position.board, king_sq, side_to_move.flip())
    }

    /// Check if a square is attacked by the given color
    pub fn is_square_attacked(board: &Board, sq: Square, by_color: Color) -> bool {
        // Check for pawn attacks
        let pawn_attacks = Self::pawn_attacks_to(sq, by_color.flip());
        if (pawn_attacks & board.piece_bb(PieceType::Pawn, by_color)).as_u64() != 0 {
            return true;
        }

        // Check for knight attacks
        let knight_attacks = Self::knight_attacks(sq);
        if (knight_attacks & board.piece_bb(PieceType::Knight, by_color)).as_u64() != 0 {
            return true;
        }

        // Check for king attacks
        let king_attacks = Self::king_attacks(sq);
        if (king_attacks & board.piece_bb(PieceType::King, by_color)).as_u64() != 0 {
            return true;
        }

        // Check for bishop/queen diagonal attacks
        let diagonal_attacks = Self::bishop_attacks_on_the_fly(sq, board.occupied());
        let bishop_queen = board.piece_bb(PieceType::Bishop, by_color)
            | board.piece_bb(PieceType::Queen, by_color);
        if (diagonal_attacks & bishop_queen).as_u64() != 0 {
            return true;
        }

        // Check for rook/queen straight attacks
        let straight_attacks = Self::rook_attacks_on_the_fly(sq, board.occupied());
        let rook_queen =
            board.piece_bb(PieceType::Rook, by_color) | board.piece_bb(PieceType::Queen, by_color);
        if (straight_attacks & rook_queen).as_u64() != 0 {
            return true;
        }

        false
    }

    /// Generate pawn moves with optional en passant
    fn generate_pawn_moves_ep(
        board: &Board,
        color: Color,
        ep_square: Option<Square>,
        moves: &mut MoveList,
    ) {
        let pawns = board.piece_bb(PieceType::Pawn, color);
        let _occupied = board.occupied();
        let empty = board.empty();
        let opponent = color.flip();

        let (start_rank, promotion_rank, forward): (Rank, Rank, i8) = match color {
            Color::White => (RANK_2, RANK_8, 8),
            Color::Black => (RANK_7, RANK_1, -8i8),
        };

        // Single pawn pushes
        let single_pushes = {
            let pushed = if forward > 0 {
                pawns.as_u64() << forward
            } else {
                pawns.as_u64() >> (-forward)
            };
            Bitboard::new(pushed) & empty
        };

        for to in single_pushes.squares() {
            let from = if forward > 0 {
                to - forward as u8
            } else {
                to + (-forward) as u8
            };
            if rank_of(to) == promotion_rank {
                // Promotions
                moves.push(Move::promotion(from, to, PromotionType::Queen));
                moves.push(Move::promotion(from, to, PromotionType::Rook));
                moves.push(Move::promotion(from, to, PromotionType::Bishop));
                moves.push(Move::promotion(from, to, PromotionType::Knight));
            } else {
                moves.push(Move::new(from, to));
            }
        }

        // Double pawn pushes (from starting rank)
        let double_pushes = {
            // Only pawns on their starting rank can double push
            let pawns_on_start = pawns & Bitboard::rank_mask(start_rank);
            let pushed = if forward > 0 {
                pawns_on_start.as_u64() << forward
            } else {
                pawns_on_start.as_u64() >> (-forward)
            };
            // Check if the square one step ahead is empty
            let one_step_empty = (Bitboard::new(pushed) & empty).as_u64();
            let two_step = if forward > 0 {
                one_step_empty << forward
            } else {
                one_step_empty >> (-forward)
            };
            Bitboard::new(two_step) & empty
        };

        for to in double_pushes.squares() {
            let from = if forward > 0 {
                to - 2 * forward as u8
            } else {
                to + 2 * (-forward) as u8
            };
            moves.push(Move::new(from, to));
        }

        // Pawn captures (including en passant)
        let opponent_pieces = board.color_bb(opponent);
        let left_captures = if color == Color::White {
            (pawns.as_u64() & !FILE_BB[FILE_A as usize]) << 7
        } else {
            (pawns.as_u64() & !FILE_BB[FILE_H as usize]) >> 9
        };
        let left_captures = Bitboard::new(left_captures) & opponent_pieces;

        for to in left_captures.squares() {
            let from = if color == Color::White {
                to - 7
            } else {
                to + 9
            };

            // Validate: ensure from square is on the board and move is forward
            if from >= 64 {
                continue;
            }

            // Validate rank difference - pawns must move forward exactly one rank
            let from_rank = rank_of(from);
            let to_rank = rank_of(to);
            let rank_diff = if color == Color::White {
                to_rank as i8 - from_rank as i8
            } else {
                from_rank as i8 - to_rank as i8
            };

            if rank_diff != 1 {
                continue; // Invalid - not moving forward one rank
            }

            // Validate file difference - pawn captures must be diagonal
            let from_file = file_of(from);
            let to_file = file_of(to);
            if (from_file as i8 - to_file as i8).abs() != 1 {
                continue; // Invalid - not a diagonal capture
            }

            if rank_of(to) == promotion_rank {
                moves.push(Move::promotion(from, to, PromotionType::Queen));
                moves.push(Move::promotion(from, to, PromotionType::Rook));
                moves.push(Move::promotion(from, to, PromotionType::Bishop));
                moves.push(Move::promotion(from, to, PromotionType::Knight));
            } else {
                moves.push(Move::new(from, to));
            }
        }

        let right_captures = if color == Color::White {
            (pawns.as_u64() & !FILE_BB[FILE_H as usize]) << 9
        } else {
            (pawns.as_u64() & !FILE_BB[FILE_A as usize]) >> 7
        };
        let right_captures = Bitboard::new(right_captures) & opponent_pieces;

        for to in right_captures.squares() {
            let from = if color == Color::White {
                to - 9
            } else {
                to + 7
            };

            // Validate: ensure from square is on the board and move is forward
            if from >= 64 {
                continue;
            }

            // Validate rank difference - pawns must move forward exactly one rank
            let from_rank = rank_of(from);
            let to_rank = rank_of(to);
            let rank_diff = if color == Color::White {
                to_rank as i8 - from_rank as i8
            } else {
                from_rank as i8 - to_rank as i8
            };

            if rank_diff != 1 {
                continue; // Invalid - not moving forward one rank
            }

            // Validate file difference - pawn captures must be diagonal
            let from_file = file_of(from);
            let to_file = file_of(to);
            if (from_file as i8 - to_file as i8).abs() != 1 {
                continue; // Invalid - not a diagonal capture
            }

            if rank_of(to) == promotion_rank {
                moves.push(Move::promotion(from, to, PromotionType::Queen));
                moves.push(Move::promotion(from, to, PromotionType::Rook));
                moves.push(Move::promotion(from, to, PromotionType::Bishop));
                moves.push(Move::promotion(from, to, PromotionType::Knight));
            } else {
                moves.push(Move::new(from, to));
            }
        }

        // En passant captures
        if let Some(ep_sq) = ep_square {
            let ep_rank = rank_of(ep_sq);
            let ep_file = file_of(ep_sq);

            // Only generate en passant if the EP square is on the correct rank
            // After Black's double push to rank 5, EP square is on rank 6
            // After White's double push to rank 4, EP square is on rank 3
            let valid_ep_rank = if color == Color::White {
                RANK_6
            } else {
                RANK_3
            };

            if ep_rank == valid_ep_rank {
                // Find pawns that can capture en passant
                for sq in pawns.squares() {
                    let pawn_rank = rank_of(sq);
                    let pawn_file = file_of(sq);

                    // To capture en passant on rank 6 (white), the white pawn must be on rank 5
                    // To capture en passant on rank 3 (black), the black pawn must be on rank 4
                    let correct_rank = if color == Color::White {
                        RANK_5  // White pawns on rank 5 capture on rank 6
                    } else {
                        RANK_4  // Black pawns on rank 4 capture on rank 3
                    };

                    if pawn_rank == correct_rank {
                        // Check if pawn is adjacent to EP file
                        if (pawn_file as i8 - ep_file as i8).abs() == 1 {
                            // Generate en passant capture
                            moves.push(Move::en_passant(sq, ep_sq));
                        }
                    }
                }
            }
        }
    }

    /// Generate knight moves
    fn generate_knight_moves(board: &Board, color: Color, moves: &mut MoveList) {
        let knights = board.piece_bb(PieceType::Knight, color);
        let not_occupied = board.empty();
        let opponent_pieces = board.color_bb(color.flip());

        for from in knights.squares() {
            let attacks = Self::knight_attacks(from);
            let targets = attacks & (not_occupied | opponent_pieces);
            for to in targets.squares() {
                moves.push(Move::new(from, to));
            }
        }
    }

    /// Generate bishop moves
    fn generate_bishop_moves(board: &Board, color: Color, moves: &mut MoveList) {
        let bishops = board.piece_bb(PieceType::Bishop, color);
        let occupied = board.occupied();
        let not_occupied = board.empty();
        let opponent_pieces = board.color_bb(color.flip());

        for from in bishops.squares() {
            let attacks = Self::bishop_attacks_on_the_fly(from, occupied);
            let targets = attacks & (not_occupied | opponent_pieces);
            for to in targets.squares() {
                moves.push(Move::new(from, to));
            }
        }
    }

    /// Generate rook moves
    fn generate_rook_moves(board: &Board, color: Color, moves: &mut MoveList) {
        let rooks = board.piece_bb(PieceType::Rook, color);
        let occupied = board.occupied();
        let not_occupied = board.empty();
        let opponent_pieces = board.color_bb(color.flip());

        for from in rooks.squares() {
            let attacks = Self::rook_attacks_on_the_fly(from, occupied);
            let targets = attacks & (not_occupied | opponent_pieces);
            for to in targets.squares() {
                moves.push(Move::new(from, to));
            }
        }
    }

    /// Generate queen moves
    fn generate_queen_moves(board: &Board, color: Color, moves: &mut MoveList) {
        let queens = board.piece_bb(PieceType::Queen, color);
        let occupied = board.occupied();
        let not_occupied = board.empty();
        let opponent_pieces = board.color_bb(color.flip());

        for from in queens.squares() {
            let attacks = Self::bishop_attacks_on_the_fly(from, occupied)
                | Self::rook_attacks_on_the_fly(from, occupied);
            let targets = attacks & (not_occupied | opponent_pieces);
            for to in targets.squares() {
                moves.push(Move::new(from, to));
            }
        }
    }

    /// Generate king moves
    fn generate_king_moves(board: &Board, color: Color, moves: &mut MoveList) {
        let king = board.piece_bb(PieceType::King, color);
        let not_occupied = board.empty();
        let opponent_pieces = board.color_bb(color.flip());

        for from in king.squares() {
            let attacks = Self::king_attacks(from);
            let targets = attacks & (not_occupied | opponent_pieces);
            for to in targets.squares() {
                // Don't generate moves that would leave the king in check
                // Make a temporary move and check if the king would be in check
                let mut test_board = board.clone();
                if let Some(king_piece) = test_board.get_piece(from) {
                    test_board.remove_piece(king_piece, from);

                    // Handle capture
                    if let Some(captured) = test_board.get_piece(to) {
                        test_board.remove_piece(captured, to);
                    }

                    test_board.set_piece(king_piece, to);

                    // Check if the king is in check after the move
                    if !Self::is_square_attacked(&test_board, to, color.flip()) {
                        moves.push(Move::new(from, to));
                    }
                }
            }
        }
    }

    /// Generate castling moves
    fn generate_castling_moves(
        board: &Board,
        color: Color,
        castling_rights: u8,
        moves: &mut MoveList,
    ) {
        let castling_moves =
            crate::castling::Castling::generate_castling(board, color, castling_rights);

        for mv in castling_moves {
            if let Some(m) = mv {
                moves.push(m);
            }
        }
    }

    /// Knight attacks from a square
    pub fn knight_attacks(sq: Square) -> Bitboard {
        const KNIGHT_DELTAS: [i8; 8] = [-17, -15, -10, -6, 6, 10, 15, 17];
        let mut attacks = Bitboard::EMPTY;

        for &delta in &KNIGHT_DELTAS {
            let target = (sq as i8 + delta) as u8;
            if delta < 0 && target > sq {
                continue; // Underflow
            }
            if delta > 0 && target < sq {
                continue; // Overflow
            }
            if target < 64 {
                if (rank_of(sq).abs_diff(rank_of(target)) <= 2)
                    && (file_of(sq).abs_diff(file_of(target)) <= 2)
                {
                    attacks.set(target);
                }
            }
        }

        attacks
    }

    /// King attacks from a square
    fn king_attacks(sq: Square) -> Bitboard {
        const KING_DELTAS: [i8; 8] = [-9, -8, -7, -1, 1, 7, 8, 9];
        let mut attacks = Bitboard::EMPTY;

        for &delta in &KING_DELTAS {
            let target = (sq as i8 + delta) as u8;
            // Check for underflow/overflow and that the target is on an adjacent square
            if target < 64 {
                let rank_diff = rank_of(sq).abs_diff(rank_of(target));
                let file_diff = file_of(sq).abs_diff(file_of(target));
                if rank_diff <= 1 && file_diff <= 1 {
                    attacks.set(target);
                }
            }
        }

        attacks
    }

    /// Bishop attacks (simple implementation, will use magic bitboards)
    pub fn bishop_attacks_on_the_fly(sq: Square, occupied: Bitboard) -> Bitboard {
        let mut attacks = Bitboard::EMPTY;
        let rank = rank_of(sq);
        let file = file_of(sq);

        // Northeast
        for i in 1..8 {
            let r = rank + i;
            let f = file + i;
            if r >= 8 || f >= 8 {
                break;
            }
            let target = square_of(r, f);
            attacks.set(target);
            if occupied.get(target) {
                break;
            }
        }

        // Northwest
        for i in 1..8 {
            let r = rank + i;
            let f = file as i8 - i as i8;
            if r >= 8 || f < 0 {
                break;
            }
            let target = square_of(r, f as u8);
            attacks.set(target);
            if occupied.get(target) {
                break;
            }
        }

        // Southeast
        for i in 1..8 {
            let r = rank as i8 - i as i8;
            let f = file + i;
            if r < 0 || f >= 8 {
                break;
            }
            let target = square_of(r as u8, f);
            attacks.set(target);
            if occupied.get(target) {
                break;
            }
        }

        // Southwest
        for i in 1..8 {
            let r = rank as i8 - i as i8;
            let f = file as i8 - i as i8;
            if r < 0 || f < 0 {
                break;
            }
            let target = square_of(r as u8, f as u8);
            attacks.set(target);
            if occupied.get(target) {
                break;
            }
        }

        attacks
    }

    /// Rook attacks (simple implementation, will use magic bitboards)
    pub fn rook_attacks_on_the_fly(sq: Square, occupied: Bitboard) -> Bitboard {
        let mut attacks = Bitboard::EMPTY;
        let rank = rank_of(sq);
        let file = file_of(sq);

        // North
        for i in (rank + 1)..8 {
            let target = square_of(i, file);
            attacks.set(target);
            if occupied.get(target) {
                break;
            }
        }

        // South
        for i in (0..rank).rev() {
            let target = square_of(i, file);
            attacks.set(target);
            if occupied.get(target) {
                break;
            }
        }

        // East
        for i in (file + 1)..8 {
            let target = square_of(rank, i);
            attacks.set(target);
            if occupied.get(target) {
                break;
            }
        }

        // West
        for i in (0..file).rev() {
            let target = square_of(rank, i);
            attacks.set(target);
            if occupied.get(target) {
                break;
            }
        }

        attacks
    }

    /// Queen attacks (combines rook and bishop)
    pub fn queen_attacks_on_the_fly(sq: Square, occupied: Bitboard) -> Bitboard {
        Self::rook_attacks_on_the_fly(sq, occupied) | Self::bishop_attacks_on_the_fly(sq, occupied)
    }

    /// Pawn attacks TO a square (for checking if square is attacked by pawns)
    fn pawn_attacks_to(sq: Square, by_color: Color) -> Bitboard {
        if by_color == Color::White {
            // White pawns attack from below
            let mut attacks = Bitboard::EMPTY;
            let rank = rank_of(sq);
            let file = file_of(sq);

            if rank > 0 && file > 0 {
                attacks.set(square_of(rank - 1, file - 1));
            }
            if rank > 0 && file < 7 {
                attacks.set(square_of(rank - 1, file + 1));
            }
            attacks
        } else {
            // Black pawns attack from above
            let mut attacks = Bitboard::EMPTY;
            let rank = rank_of(sq);
            let file = file_of(sq);

            if rank < 7 && file > 0 {
                attacks.set(square_of(rank + 1, file - 1));
            }
            if rank < 7 && file < 7 {
                attacks.set(square_of(rank + 1, file + 1));
            }
            attacks
        }
    }

    /// Make a move on the board (handles all special cases)
    pub fn make_move_raw(board: &mut Board, mv: Move, color: Color) {
        let from = mv.from();
        let to = mv.to();

        // Handle castling first (before removing the piece from 'from')
        if mv.is_castle() {
            // Move the king
            if let Some(king) = board.get_piece(from) {
                board.remove_piece(king, from);
                board.set_piece(king, to);
            }

            // Move the rook
            let (rook_from, rook_to) = if color == Color::White {
                if to == 6 {
                    (7, 5) // Kingside: h1 to f1
                } else {
                    (0, 3) // Queenside: a1 to d1
                }
            } else {
                if to == 62 {
                    (63, 61) // Kingside: h8 to f8
                } else {
                    (56, 59) // Queenside: a8 to d8
                }
            };

            if let Some(rook) = board.get_piece(rook_from) {
                board.remove_piece(rook, rook_from);
                board.set_piece(rook, rook_to);
            }
            return;
        }

        if let Some(piece) = board.get_piece(from) {
            board.remove_piece(piece, from);

            // Handle en passant capture
            if mv.is_en_passant() {
                // Captured pawn is behind the EP square
                let captured_pawn_sq = if color == Color::White {
                    to - 8 // White captures black pawn "below" the EP square
                } else {
                    to + 8 // Black captures white pawn "above" the EP square
                };
                if let Some(captured) = board.get_piece(captured_pawn_sq) {
                    board.remove_piece(captured, captured_pawn_sq);
                }
            } else if let Some(captured) = board.get_piece(to) {
                // Regular capture
                board.remove_piece(captured, to);
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
                    board.set_piece(promo_piece, to);
                }
            } else {
                board.set_piece(piece, to);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knight_attacks() {
        let attacks = MoveGen::knight_attacks(28); // e4

        // Knight on e4 should attack: c5, d6, f6, g5, g3, f2, d2, c3
        assert_eq!(attacks.count(), 8);
    }

    #[test]
    fn test_king_attacks() {
        let attacks = MoveGen::king_attacks(28); // e4

        // King on e4 should attack 8 squares (or fewer on edges)
        assert!(attacks.count() <= 8);
    }

    #[test]
    fn test_generate_moves_start() {
        let board = Board::from_start();
        let moves = MoveGen::generate_moves(&board, Color::White);

        // Should have 20 moves from start position
        assert_eq!(moves.len(), 20);
    }
}
