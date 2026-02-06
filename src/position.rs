use crate::board::Board;
use crate::piece::{Color, Piece};
use crate::r#move::Move;
use crate::utils::Square;

/// Information needed to undo a move
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UndoInfo {
    /// The move that was made
    pub mv: Move,
    /// Captured piece (if any)
    pub captured: Option<Piece>,
    /// Castling rights before the move
    pub castling_rights: u8,
    /// En passant square before the move
    pub ep_square: Option<Square>,
    /// Halfmove clock before the move
    pub halfmove_clock: u32,
    /// For en passant: the captured pawn position
    pub ep_captured_sq: Option<Square>,
    /// For promotions: the promoted piece type
    pub promoted_piece: Option<Piece>,
    /// Fullmove number before the move
    pub fullmove_number: u32,
    /// Hash before the move
    pub hash: u64,
}

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
    /// Undo stack for fast make/undo during search
    pub undo_stack: Vec<UndoInfo>,
}

impl Position {
    /// Create a new position from the starting position
    pub fn from_start() -> Self {
        let mut position = Position {
            board: Board::from_start(),
            state: PositionState::new(),
            history: Vec::new(),
            history_hashes: Vec::new(),
            undo_stack: Vec::new(),
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
            Some(
                crate::utils::square_from_string(parts[3])
                    .ok_or_else(|| format!("Invalid en passant square: {}", parts[3]))?,
            )
        };

        // Parse halfmove clock
        let halfmove_clock = parts[4]
            .parse()
            .map_err(|_| format!("Invalid halfmove clock: {}", parts[4]))?;

        // Parse fullmove number
        let fullmove_number = parts[5]
            .parse()
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
            undo_stack: Vec::new(),
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
            undo_stack: Vec::new(),
        };
        position.state.hash = position.compute_hash();
        position
    }

    /// Make a move on the position
    pub fn make_move(&mut self, mv: Move) {
        let from = mv.from();
        let to = mv.to();
        let color = self.state.side_to_move;

        // CRITICAL: Check if there's actually a piece to move BEFORE doing anything else
        let maybe_piece = self.board.get_piece(from);
        if maybe_piece.is_none() {
            eprintln!(
                "ERROR: make_move called with no piece on from square {} for move {}",
                crate::utils::square_to_string(from),
                mv
            );
            eprintln!("  Side to move: {:?}", color);
            eprintln!("  Board state:\n{}", self.board.to_string());
            // DO NOT update any state - just return
            return;
        }
        let piece = maybe_piece.unwrap();

        // CRITICAL: Verify the piece belongs to the side to move
        if piece.color != color {
            eprintln!(
                "ERROR: make_move called with wrong color piece on from square {} for move {}",
                crate::utils::square_to_string(from),
                mv
            );
            eprintln!("  Expected color: {:?}, Got: {:?}", color, piece.color);
            eprintln!("  Board state:\n{}", self.board.to_string());
            return;
        }

        // Save current state to history
        self.history.push(self.state.clone());
        self.history_hashes.push(self.state.hash);

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
        let is_pawn_move = piece.piece_type == crate::piece::PieceType::Pawn;

        if is_capture || is_pawn_move {
            self.state.halfmove_clock = 0;
        } else {
            self.state.halfmove_clock += 1;
        }

        // Update fullmove number
        if color == Color::Black {
            self.state.fullmove_number += 1;
        }

        // === Incremental hash updates (MUCH faster than recomputing!) ===
        // Save state before modifications
        let old_rights = self.state.castling_rights;
        let old_ep = self.state.ep_square;

        // Remove piece from old square
        self.update_hash_piece(piece, from);

        // Remove captured piece (if any) from destination
        // Need to check for en passant capture (different square)
        let captured = if mv.is_en_passant() {
            // En passant captures the pawn on a different square
            let ep_capture_sq = if color == Color::White {
                to - 8
            } else {
                to + 8
            };
            self.board.get_piece(ep_capture_sq)
        } else {
            // Normal capture
            self.board.get_piece(to)
        };

        if let Some(captured_piece) = captured {
            let capture_sq = if mv.is_en_passant() {
                if color == Color::White {
                    to - 8
                } else {
                    to + 8
                }
            } else {
                to
            };
            self.update_hash_piece(captured_piece, capture_sq);
        }

        // Make the move on the board
        self.make_move_on_board(mv, color, piece);

        // Add piece to new square (or promoted piece)
        if mv.is_promotion() {
            use crate::r#move::PromotionType;
            use crate::piece::PieceType;
            if let Some(promo_type) = mv.promotion_type() {
                let promoted_piece = Piece::new(
                    color,
                    match promo_type {
                        PromotionType::Knight => PieceType::Knight,
                        PromotionType::Bishop => PieceType::Bishop,
                        PromotionType::Rook => PieceType::Rook,
                        PromotionType::Queen => PieceType::Queen,
                    },
                );
                self.update_hash_piece(promoted_piece, to);
            }
        } else {
            // Check if it's castling (encoded as king captures rook)
            if mv.is_castle() {
                // Castling: we need to hash the rook move too
                let rook_from = to;
                let rook_to = mv.castle_rook_destination();
                if let Some(rook) = self.board.get_piece(rook_to) {
                    self.update_hash_piece(rook, rook_from); // Remove rook from old position
                    self.update_hash_piece(rook, rook_to);   // Add rook to new position
                }
                // King's new position
                let king_dest = mv.castle_king_destination();
                self.update_hash_piece(piece, king_dest);
            } else {
                // Normal move - add piece to destination
                self.update_hash_piece(piece, to);
            }
        }

        // Update side to move (flip)
        self.state.side_to_move = color.flip();
        self.update_hash_side_to_move();

        // Update castling rights
        self.update_castling_rights(from, to);
        self.update_hash_castling(old_rights, self.state.castling_rights);

        // Update en passant square
        self.update_ep_square(mv, color);
        self.update_hash_en_passant(old_ep, self.state.ep_square);
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

    /// Make a move on the position (fast version for search)
    /// Returns UndoInfo that can be used to undo the move
    /// This does NOT update the history vectors (for performance during search)
    pub fn make_move_fast(&mut self, mv: Move) {
        use crate::piece::PieceType;
        use crate::r#move::PromotionType;

        let from = mv.from();
        let to = mv.to();
        let color = self.state.side_to_move;

        // Get the piece to move
        let piece = match self.board.get_piece(from) {
            Some(p) => p,
            None => {
                eprintln!("ERROR: make_move_fast called with no piece on from square {}", from);
                return;
            }
        };

        // Verify piece color
        if piece.color != color {
            eprintln!("ERROR: make_move_fast called with wrong color piece");
            return;
        }

        // Save state for incremental hash updates
        let old_rights = self.state.castling_rights;
        let old_ep = self.state.ep_square;

        // Determine captured piece (handle en passant specially)
        let captured = if mv.is_en_passant() {
            let ep_capture_sq = if color == Color::White {
                to - 8
            } else {
                to + 8
            };
            self.board.get_piece(ep_capture_sq)
        } else {
            self.board.get_piece(to)
        };

        // Collect undo info BEFORE making any changes
        let ep_captured_sq = if mv.is_en_passant() {
            let ep_sq = if color == Color::White { to - 8 } else { to + 8 };
            Some(ep_sq)
        } else {
            None
        };

        let undo_info = UndoInfo {
            mv,
            captured,
            castling_rights: self.state.castling_rights,
            ep_square: self.state.ep_square,
            halfmove_clock: self.state.halfmove_clock,
            ep_captured_sq,
            promoted_piece: None,
            fullmove_number: self.state.fullmove_number,
            hash: self.state.hash,
        };

        // Update halfmove clock
        let is_capture = captured.is_some();
        let is_pawn_move = piece.piece_type == PieceType::Pawn;

        if is_capture || is_pawn_move {
            self.state.halfmove_clock = 0;
        } else {
            self.state.halfmove_clock += 1;
        }

        // Update fullmove number
        if color == Color::Black {
            self.state.fullmove_number += 1;
        }

        // === Incremental hash updates ===
        // Remove piece from old square
        self.update_hash_piece(piece, from);

        // Handle castling (king captures rook encoding)
        if mv.is_castle() {
            let king_dest = mv.castle_king_destination();
            let rook_from = to;
            let rook_dest = mv.castle_rook_destination();

            // Move the king
            self.board.remove_piece(piece, from);
            self.board.set_piece(piece, king_dest);

            // Move the rook
            if let Some(rook) = self.board.get_piece(rook_from) {
                self.board.remove_piece(rook, rook_from);
                self.board.set_piece(rook, rook_dest);

                // Hash rook move
                self.update_hash_piece(rook, rook_from);
                self.update_hash_piece(rook, rook_dest);
            }

            // Hash king to new position
            self.update_hash_piece(piece, king_dest);

            // Update side to move, castling, EP
            self.state.side_to_move = color.flip();
            self.update_hash_side_to_move();
            self.update_castling_rights(from, to);
            self.update_hash_castling(old_rights, self.state.castling_rights);
            self.state.ep_square = None;
            self.update_hash_en_passant(old_ep, None);

            self.undo_stack.push(undo_info);
            return;
        }

        // Handle en passant
        if mv.is_en_passant() {
            self.board.remove_piece(piece, from);
            self.board.set_piece(piece, to);

            let captured_pawn_sq = if color == Color::White {
                to - 8
            } else {
                to + 8
            };

            if let Some(captured_pawn) = captured {
                self.board.remove_piece(captured_pawn, captured_pawn_sq);
                // Hash the captured pawn removal
                self.update_hash_piece(captured_pawn, captured_pawn_sq);
            }

            // Hash pawn to new position
            self.update_hash_piece(piece, to);

            // Update side to move, castling, EP
            self.state.side_to_move = color.flip();
            self.update_hash_side_to_move();
            self.update_castling_rights(from, to);
            self.update_hash_castling(old_rights, self.state.castling_rights);
            self.state.ep_square = None;
            self.update_hash_en_passant(old_ep, None);

            self.undo_stack.push(undo_info);
            return;
        }

        // Handle promotions
        let promoted_piece = if mv.is_promotion() {
            if let Some(promo_type) = mv.promotion_type() {
                Some(Piece::new(
                    color,
                    match promo_type {
                        PromotionType::Knight => PieceType::Knight,
                        PromotionType::Bishop => PieceType::Bishop,
                        PromotionType::Rook => PieceType::Rook,
                        PromotionType::Queen => PieceType::Queen,
                    },
                ))
            } else {
                None
            }
        } else {
            None
        };

        // Make the move on the board
        self.board.remove_piece(piece, from);

        // Handle captures (but NOT castling - that's handled above)
        if let Some(captured_piece) = captured {
            self.board.remove_piece(captured_piece, to);
            // Hash captured piece removal
            self.update_hash_piece(captured_piece, to);
        }

        // Place the piece (promoted or normal)
        if let Some(promo) = promoted_piece {
            self.board.set_piece(promo, to);
            // Hash promoted piece
            self.update_hash_piece(promo, to);
        } else {
            self.board.set_piece(piece, to);
            // Hash piece to new position
            self.update_hash_piece(piece, to);
        }

        // Update castling rights
        self.update_castling_rights(from, to);

        // Update en passant square
        self.update_ep_square(mv, color);

        // Update side to move
        self.state.side_to_move = color.flip();
        self.update_hash_side_to_move();

        // Update castling and EP hashes
        self.update_hash_castling(old_rights, self.state.castling_rights);
        self.update_hash_en_passant(old_ep, self.state.ep_square);

        // Store undo info with promoted piece
        let mut undo_final = undo_info;
        undo_final.promoted_piece = promoted_piece;
        self.undo_stack.push(undo_final);
    }

    /// Undo a move (fast version for search)
    /// Uses UndoInfo from make_move_fast to efficiently undo
    pub fn undo_move_fast(&mut self) {
        use crate::piece::PieceType;

        let undo_info = match self.undo_stack.pop() {
            Some(info) => info,
            None => {
                eprintln!("ERROR: undo_move_fast called with empty undo stack");
                return;
            }
        };

        let mv = undo_info.mv;
        let from = mv.from();
        let to = mv.to();

        // Restore side to move
        let color = self.state.side_to_move.flip();

        // Handle castling undo
        if mv.is_castle() {
            let king_dest = mv.castle_king_destination();
            let rook_from = to;
            let rook_dest = mv.castle_rook_destination();

            // Get the king and rook
            let king = match self.board.get_piece(king_dest) {
                Some(k) => k,
                None => {
                    eprintln!("ERROR: undo_move_fast castling but no king on dest");
                    return;
                }
            };

            let rook = match self.board.get_piece(rook_dest) {
                Some(r) => r,
                None => {
                    eprintln!("ERROR: undo_move_fast castling but no rook on dest");
                    return;
                }
            };

            // Move king back
            self.board.remove_piece(king, king_dest);
            self.board.set_piece(king, from);

            // Move rook back
            self.board.remove_piece(rook, rook_dest);
            self.board.set_piece(rook, rook_from);
        } else if mv.is_en_passant() {
            // Undo en passant
            let piece = match self.board.get_piece(to) {
                Some(p) => p,
                None => {
                    eprintln!("ERROR: undo_move_fast en passant but no piece on dest");
                    return;
                }
            };

            // Move pawn back
            self.board.remove_piece(piece, to);
            self.board.set_piece(piece, from);

            // Restore captured pawn
            if let Some(captured_sq) = undo_info.ep_captured_sq {
                if let Some(captured) = undo_info.captured {
                    self.board.set_piece(captured, captured_sq);
                }
            }
        } else if mv.is_promotion() {
            // Undo promotion
            let promoted = match self.board.get_piece(to) {
                Some(p) => p,
                None => {
                    eprintln!("ERROR: undo_move_fast promotion but no piece on dest");
                    return;
                }
            };

            // Remove promoted piece
            self.board.remove_piece(promoted, to);

            // Restore original pawn (with the mover's color, not flipped!)
            let original_piece = Piece::new(color, PieceType::Pawn);
            self.board.set_piece(original_piece, from);

            // Restore captured piece if any
            if let Some(captured) = undo_info.captured {
                self.board.set_piece(captured, to);
            }
        } else {
            // Normal move
            let piece = match self.board.get_piece(to) {
                Some(p) => p,
                None => {
                    eprintln!("ERROR: undo_move_fast normal but no piece on dest");
                    return;
                }
            };

            // Move piece back
            self.board.remove_piece(piece, to);
            self.board.set_piece(piece, from);

            // Restore captured piece if any
            if let Some(captured) = undo_info.captured {
                self.board.set_piece(captured, to);
            }
        }

        // Restore state
        self.state.castling_rights = undo_info.castling_rights;
        self.state.ep_square = undo_info.ep_square;
        self.state.halfmove_clock = undo_info.halfmove_clock;
        self.state.fullmove_number = undo_info.fullmove_number;
        self.state.hash = undo_info.hash;
        self.state.side_to_move = color;
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
        self.is_checkmate()
            || self.is_stalemate()
            || self.is_threefold_repetition()
            || self.is_fifty_move_rule()
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
    fn make_move_on_board(&mut self, mv: Move, color: Color, piece: crate::piece::Piece) {
        use crate::piece::PieceType;
        use crate::r#move::PromotionType;

        let from = mv.from();
        let to = mv.to();

        // Handle castling
        // NOTE: Castling is encoded as "king captures friendly rook"
        // So 'to' is the ROOK'S square, not the king's destination
        if mv.is_castle() {
            let king_dest = mv.castle_king_destination();
            let rook_from = to; // The 'to' square in a castling move is the rook's position
            let rook_dest = mv.castle_rook_destination();

            // Move the king
            self.board.remove_piece(piece, from);
            self.board.set_piece(piece, king_dest);

            // Move the rook (the 'captured' piece should be our own rook)
            if let Some(rook) = self.board.get_piece(rook_from) {
                if rook.piece_type == PieceType::Rook && rook.color == color {
                    self.board.remove_piece(rook, rook_from);
                    self.board.set_piece(rook, rook_dest);
                } else {
                    eprintln!(
                        "ERROR: Castling move {} expected rook on {} but found {:?}",
                        mv, rook_from, rook
                    );
                }
            } else {
                eprintln!(
                    "ERROR: Castling move {} but no rook on {} (king on {})",
                    mv, rook_from, from
                );
            }

            return;
        }

        // Handle en passant
        if mv.is_en_passant() {
            self.board.remove_piece(piece, from);
            self.board.set_piece(piece, to);

            // Remove the captured pawn
            let captured_pawn_sq = if color == Color::White {
                to - 8 // White captures "behind" the EP square
            } else {
                to + 8 // Black captures "behind" the EP square
            };

            if let Some(captured) = self.board.get_piece(captured_pawn_sq) {
                self.board.remove_piece(captured, captured_pawn_sq);
            }
            return;
        }

        // Normal move
        self.board.remove_piece(piece, from);

        // Handle captures
        if let Some(captured) = self.board.get_piece(to) {
            self.board.remove_piece(captured, to);
        }

        // Handle promotions
        if mv.is_promotion() {
            if let Some(promo_type) = mv.promotion_type() {
                let promo_piece = crate::piece::Piece::new(
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
    }

    /// Update castling rights after a move
    fn update_castling_rights(&mut self, from: Square, to: Square) {
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

        // Rook moves from home square
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

        // Rook captures on home square
        if to == 0 {
            self.state.castling_rights &= !WHITE_QUEENSIDE;
        }
        if to == 7 {
            self.state.castling_rights &= !WHITE_KINGSIDE;
        }
        if to == 56 {
            self.state.castling_rights &= !BLACK_QUEENSIDE;
        }
        if to == 63 {
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

    /// Compute Zobrist hash (incremental version - only used for initialization)
    fn compute_hash(&self) -> u64 {
        use crate::zobrist::Zobrist;

        let mut hash = 0u64;

        // Hash pieces
        for sq in 0..64 {
            if let Some(piece) = self.board.get_piece(sq) {
                hash ^= Zobrist::piece(piece, sq);
            }
        }

        // Hash side to move
        if self.state.side_to_move == Color::Black {
            hash ^= Zobrist::side();
        }

        // Hash castling rights
        hash ^= Zobrist::castling(self.state.castling_rights);

        // Hash en passant square
        if let Some(ep) = self.state.ep_square {
            hash ^= Zobrist::en_passant(ep);
        }

        hash
    }

    /// Update hash incrementally when a piece is added/removed/moved
    fn update_hash_piece(&mut self, piece: Piece, square: Square) {
        use crate::zobrist::Zobrist;
        self.state.hash ^= Zobrist::piece(piece, square);
    }

    /// Update hash incrementally for side to move change
    fn update_hash_side_to_move(&mut self) {
        use crate::zobrist::Zobrist;
        self.state.hash ^= Zobrist::side();
    }

    /// Update hash incrementally for castling rights change
    fn update_hash_castling(&mut self, old_rights: u8, new_rights: u8) {
        use crate::zobrist::Zobrist;
        // Remove old castling rights
        self.state.hash ^= Zobrist::castling(old_rights);
        // Add new castling rights
        self.state.hash ^= Zobrist::castling(new_rights);
    }

    /// Update hash incrementally for en passant square change
    fn update_hash_en_passant(&mut self, old_ep: Option<Square>, new_ep: Option<Square>) {
        use crate::zobrist::Zobrist;
        if let Some(old_sq) = old_ep {
            self.state.hash ^= Zobrist::en_passant(old_sq);
        }
        if let Some(new_sq) = new_ep {
            self.state.hash ^= Zobrist::en_passant(new_sq);
        }
    }

    /// Convert position to FEN string
    pub fn to_fen(&self) -> String {
        use crate::piece::{Color, PieceType};
        use crate::utils::{file_of, rank_of};

        let mut fen = String::new();

        // 1. Piece placement
        for rank in (0..8).rev() {
            let mut empty_count = 0;
            for file in 0..8 {
                let sq = rank * 8 + file;
                if let Some(piece) = self.board.get_piece(sq) {
                    if empty_count > 0 {
                        fen.push_str(&empty_count.to_string());
                        empty_count = 0;
                    }
                    let piece_char = match (piece.color, piece.piece_type) {
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
                    };
                    fen.push(piece_char);
                } else {
                    empty_count += 1;
                }
            }
            if empty_count > 0 {
                fen.push_str(&empty_count.to_string());
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        // 2. Side to move
        fen.push(' ');
        fen.push(if self.state.side_to_move == Color::White {
            'w'
        } else {
            'b'
        });

        // 3. Castling rights
        fen.push(' ');
        let mut has_castling = false;
        if self.state.castling_rights & 0x01 != 0 {
            fen.push('K');
            has_castling = true;
        }
        if self.state.castling_rights & 0x02 != 0 {
            fen.push('Q');
            has_castling = true;
        }
        if self.state.castling_rights & 0x04 != 0 {
            fen.push('k');
            has_castling = true;
        }
        if self.state.castling_rights & 0x08 != 0 {
            fen.push('q');
            has_castling = true;
        }
        if !has_castling {
            fen.push('-');
        }

        // 4. En passant square
        fen.push(' ');
        if let Some(ep_sq) = self.state.ep_square {
            let file = (file_of(ep_sq) as u8 + b'a') as char;
            let rank = (rank_of(ep_sq) as u8 + b'1') as char;
            fen.push(file);
            fen.push(rank);
        } else {
            fen.push('-');
        }

        // 5. Halfmove clock
        fen.push(' ');
        fen.push_str(&self.state.halfmove_clock.to_string());

        // 6. Fullmove number
        fen.push(' ');
        fen.push_str(&self.state.fullmove_number.to_string());

        fen
    }

    /// Comprehensive position validation (Stockfish-style pos_is_ok)
    /// Returns Ok(()) if position is valid, Err with description if not
    pub fn validate_position(&self) -> Result<(), String> {
        let mut errors = Vec::new();

        // Check side to move is valid
        if self.state.side_to_move != Color::White && self.state.side_to_move != Color::Black {
            errors.push(format!(
                "Invalid side to move: {:?}",
                self.state.side_to_move
            ));
        }

        // Check both kings exist
        let white_king_sq = self.board.king_square(Color::White);
        let black_king_sq = self.board.king_square(Color::Black);

        if white_king_sq.is_none() {
            errors.push("White king is missing".to_string());
        }
        if black_king_sq.is_none() {
            errors.push("Black king is missing".to_string());
        }

        // Check kings are not in invalid positions
        if let Some(sq) = white_king_sq {
            if let Some(piece) = self.board.get_piece(sq) {
                if piece.piece_type != crate::piece::PieceType::King || piece.color != Color::White
                {
                    errors.push(format!(
                        "White king square {} has wrong piece: {:?}",
                        sq, piece
                    ));
                }
            }
        }
        if let Some(sq) = black_king_sq {
            if let Some(piece) = self.board.get_piece(sq) {
                if piece.piece_type != crate::piece::PieceType::King || piece.color != Color::Black
                {
                    errors.push(format!(
                        "Black king square {} has wrong piece: {:?}",
                        sq, piece
                    ));
                }
            }
        }

        // Check EP square is valid
        if let Some(ep_sq) = self.state.ep_square {
            let ep_rank = crate::utils::rank_of(ep_sq);
            let expected_rank = if self.state.side_to_move == Color::White {
                5
            } else {
                2
            };
            if ep_rank != expected_rank {
                errors.push(format!(
                    "Invalid EP square {} for side {:?}",
                    ep_sq, self.state.side_to_move
                ));
            }
        }

        // Check hash matches board state
        let computed_hash = self.compute_hash();
        if computed_hash != self.state.hash {
            errors.push(format!(
                "Hash mismatch: computed {}, stored {}",
                computed_hash, self.state.hash
            ));
        }

        // Check board consistency
        if let Err(board_err) = self.board.validate() {
            errors.push(format!("Board validation failed: {}", board_err));
        }

        // Check for overlapping pieces between colors
        let white_pieces = self.board.color_bb(Color::White);
        let black_pieces = self.board.color_bb(Color::Black);
        if !(white_pieces & black_pieces).is_empty() {
            errors.push("Overlapping pieces between colors".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
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
