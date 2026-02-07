use crate::evaluation::Evaluation;
use crate::movegen::MoveGen;
use crate::moveorder::{is_capture, HistoryTable, KillerTable, MovePicker};
use crate::piece::Color;
use crate::position::Position;
use crate::r#move::Move;
use crate::transposition::{TTFlag, TranspositionTable};
use rayon::prelude::*;
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Search engine with parallel support
pub struct Search {
    /// Flag to stop the search (internal)
    stop: AtomicBool,
    /// External stop signal (shared with main thread)
    external_stop: Option<Arc<AtomicBool>>,
    /// Nodes searched
    nodes: AtomicU64,
    /// Search start time (interior mutability for single-threaded access)
    start_time: Cell<Instant>,
    /// Time limit (interior mutability for single-threaded access)
    time_limit: Cell<Option<Duration>>,
    /// Transposition table (interior mutability for TT updates)
    tt: RefCell<TranspositionTable>,
    /// History table for quiet move ordering
    history: HistoryTable,
    /// Killer move table
    killers: KillerTable,
    /// Number of threads for parallel search
    threads: usize,
}

impl Search {
    /// Create a new search
    pub fn new() -> Self {
        Search {
            stop: AtomicBool::new(false),
            external_stop: None,
            nodes: AtomicU64::new(0),
            start_time: Cell::new(Instant::now()),
            time_limit: Cell::new(None),
            tt: RefCell::new(TranspositionTable::new(256)), // 256 MB TT
            history: HistoryTable::new(),
            killers: KillerTable::new(),
            threads: rayon::current_num_threads(),
        }
    }

    /// Create a new search with external stop signal
    pub fn with_stop_signal(external_stop: Arc<AtomicBool>) -> Self {
        Search {
            stop: AtomicBool::new(false),
            external_stop: Some(external_stop),
            nodes: AtomicU64::new(0),
            start_time: Cell::new(Instant::now()),
            time_limit: Cell::new(None),
            tt: RefCell::new(TranspositionTable::new(256)),
            history: HistoryTable::new(),
            killers: KillerTable::new(),
            threads: rayon::current_num_threads(),
        }
    }

    /// Create a new search with custom TT size
    pub fn with_tt_size(tt_size_mb: usize) -> Self {
        Search {
            stop: AtomicBool::new(false),
            external_stop: None,
            nodes: AtomicU64::new(0),
            start_time: Cell::new(Instant::now()),
            time_limit: Cell::new(None),
            tt: RefCell::new(TranspositionTable::new(tt_size_mb)),
            history: HistoryTable::new(),
            killers: KillerTable::new(),
            threads: rayon::current_num_threads(),
        }
    }

    /// Create a new search with custom TT size and external stop signal
    pub fn with_config(tt_size_mb: usize, external_stop: Arc<AtomicBool>) -> Self {
        Search {
            stop: AtomicBool::new(false),
            external_stop: Some(external_stop),
            nodes: AtomicU64::new(0),
            start_time: Cell::new(Instant::now()),
            time_limit: Cell::new(None),
            tt: RefCell::new(TranspositionTable::new(tt_size_mb)),
            history: HistoryTable::new(),
            killers: KillerTable::new(),
            threads: rayon::current_num_threads(),
        }
    }

    /// Set the number of threads
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
    }

    /// Get the number of threads
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Get the TT size in MB
    pub fn tt_size_mb(&self) -> usize {
        self.tt.borrow().size_mb()
    }

    /// Set external stop signal (for use in spawned threads)
    pub fn set_stop_signal(&mut self, external_stop: Arc<AtomicBool>) {
        self.external_stop = Some(external_stop);
    }

    /// Calculate dynamic depth adjustment based on position complexity
    /// Returns a depth modifier (negative for simpler positions, positive for complex ones)
    pub fn calculate_depth_modifier(position: &Position) -> i32 {
        

        let mut modifier = 0;

        // Get static evaluation
        let eval = Evaluation::evaluate(&position.board);
        let eval_abs = eval.abs();

        // 1. King safety: extend when king is under attack
        let color = position.state.side_to_move;
        let king_sq = position.board.king_square(color);
        if let Some(sq) = king_sq {
            let king_attacked = MoveGen::is_square_attacked(&position.board, sq, color.flip());
            if king_attacked {
                modifier += 2; // Extend when in check
            }

            // Check for attackers near king (within 3 squares)
            let king_file = (sq % 8) as i32;
            let king_rank = (sq / 8) as i32;

            for from_sq in 0..64 {
                if let Some(piece) = position.board.get_piece(from_sq) {
                    if piece.color == color.flip() {
                        let file = (from_sq % 8) as i32;
                        let rank = (from_sq / 8) as i32;
                        let dist = (king_file - file).abs() + (king_rank - rank).abs();

                        if dist <= 2 {
                            modifier += 1; // Opponent piece near king
                        }
                    }
                }
            }
        }

        // 2. Material imbalance: extend in unbalanced positions
        if eval_abs > 200 {
            modifier += 1; // Significant material advantage/disadvantage
        }
        if eval_abs > 500 {
            modifier += 1; // Large material difference
        }

        // 3. Game phase: deeper search in endgame
        let total_material = Self::count_total_material(position);
        if total_material < 30 {
            // Endgame: pieces < 30 (roughly)
            modifier += 1;
        }
        if total_material < 20 {
            // Late endgame: very tactical
            modifier += 1;
        }

        // 4. Pawn structure: extend with weak pawns/isolated pawns
        let weak_pawns = Self::count_weak_pawns(position);
        modifier += (weak_pawns / 2) as i32; // +1 depth per 2 weak pawns

        // 5. Mobility: extend with many legal moves (complex position)
        let legal_moves = MoveGen::generate_legal_moves_ep(
            &position.board,
            color,
            position.state.ep_square,
            position.state.castling_rights,
        );
        let move_count = legal_moves.len();
        if move_count > 40 {
            modifier += 1; // Many moves available
        }
        if move_count < 15 {
            modifier -= 1; // Few moves (forced line)
        }

        // Clamp the modifier to reasonable bounds
        modifier.max(-2).min(3)
    }

    /// Count total material on the board (simplified)
    fn count_total_material(position: &Position) -> u32 {
        use crate::piece::PieceType;

        let mut count = 0;
        for sq in 0..64 {
            if let Some(piece) = position.board.get_piece(sq) {
                match piece.piece_type {
                    PieceType::Pawn => count += 1,
                    PieceType::Knight => count += 3,
                    PieceType::Bishop => count += 3,
                    PieceType::Rook => count += 5,
                    PieceType::Queen => count += 9,
                    PieceType::King => {}
                }
            }
        }
        count
    }

    /// Count weak pawns (isolated or backward)
    fn count_weak_pawns(position: &Position) -> u32 {
        use crate::piece::PieceType;

        let mut weak_pawns = 0;

        for sq in 0..64 {
            if let Some(piece) = position.board.get_piece(sq) {
                if piece.piece_type == PieceType::Pawn {
                    let file = sq % 8;
                    let rank = sq / 8;

                    // Check for pawn neighbors (same color, adjacent files)
                    let has_support = if file > 0 {
                        Self::has_friendly_pawn(position, sq - 1, piece.color, rank)
                    } else {
                        false
                    } || if file < 7 {
                        Self::has_friendly_pawn(position, sq + 1, piece.color, rank)
                    } else {
                        false
                    };

                    if !has_support {
                        weak_pawns += 1;
                    }
                }
            }
        }

        weak_pawns
    }

    /// Check if there's a friendly pawn on a specific file
    fn has_friendly_pawn(position: &Position, file_sq: u8, color: Color, _rank: u8) -> bool {
        use crate::piece::PieceType;

        for r in 0..8 {
            let sq = file_sq + r * 8;
            if let Some(piece) = position.board.get_piece(sq) {
                if piece.piece_type == PieceType::Pawn && piece.color == color {
                    return true;
                }
            }
        }
        false
    }

    /// Run an iterative deepening search
    pub fn search(&mut self, position: &Position, depth: u32, time_ms: Option<u64>) -> (Move, i32) {
        if self.threads > 1 {
            self.search_parallel(position, depth, time_ms)
        } else {
            self.search_single_thread(position, depth, time_ms)
        }
    }

    /// Single-threaded search
    fn search_single_thread(
        &mut self,
        position: &Position,
        depth: u32,
        time_ms: Option<u64>,
    ) -> (Move, i32) {
        self.search_internal(position, depth, time_ms, false)
    }

    /// Parallel search using rayon
    fn search_parallel(
        &mut self,
        position: &Position,
        depth: u32,
        time_ms: Option<u64>,
    ) -> (Move, i32) {
        // Reset search state
        self.stop.store(false, Ordering::Relaxed);
        self.nodes.store(0, Ordering::Relaxed);
        self.start_time.set(Instant::now());
        self.time_limit.set(time_ms.map(Duration::from_millis));

        // New search generation for TT
        self.tt.borrow_mut().new_generation();

        // Clear history and killers for new search
        self.history.clear();
        self.killers.clear();

        // Get legal moves first
        let legal_moves = MoveGen::generate_legal_moves_ep(
            &position.board,
            position.state.side_to_move,
            position.state.ep_square,
            position.state.castling_rights,
        );

        // No legal moves - checkmate or stalemate
        if legal_moves.is_empty() {
            let king_sq = match position.board.king_square(position.state.side_to_move) {
                Some(sq) => sq,
                None => return (Move::null(), -32000),
            };

            let in_check = MoveGen::is_square_attacked(
                &position.board,
                king_sq,
                position.state.side_to_move.flip(),
            );

            if in_check {
                return (Move::null(), -32000);
            } else {
                return (Move::null(), 0);
            }
        }

        let mut best_move = Move::null();
        let mut best_score = 0;

        // Iterative deepening
        for d in 1..=depth {
            if self.should_stop() {
                break;
            }

            // For parallel search, split root moves among threads
            let depth_best = self.search_root_parallel(position, d, &legal_moves);

            if self.should_stop() {
                break;
            }

            if let Some((mv, score)) = depth_best {
                // Validate the move
                if Self::is_move_in_list(mv, &legal_moves) {
                    best_move = mv;
                    best_score = score;

                    // Print UCI info
                    let elapsed = self.start_time.get().elapsed();
                    let nodes = self.nodes.load(Ordering::Relaxed);
                    let nps = if elapsed.as_secs_f64() > 0.0 {
                        nodes as f64 / elapsed.as_secs_f64()
                    } else {
                        0.0
                    };

                    let stats = self.tt.borrow().stats();
                    println!(
                        "info depth {} score cp {} nodes {} nps {:.0} time {} hashfull {}",
                        d,
                        score,
                        nodes,
                        nps,
                        elapsed.as_millis(),
                        stats.hashfull
                    );
                }
            }
        }

        // Final validation
        if best_move.is_null() {
            best_move = legal_moves.get(0);
        } else if !Self::is_move_in_list(best_move, &legal_moves) {
            best_move = legal_moves.get(0);
        }

        (best_move, best_score)
    }

    /// Search root position in parallel using Lazy SMP
    fn search_root_parallel(
        &self,
        position: &Position,
        depth: u32,
        legal_moves: &crate::movelist::MoveList,
    ) -> Option<(Move, i32)> {
        let num_moves = legal_moves.len();
        if num_moves == 0 {
            return None;
        }

        // Extract data we need before entering parallel section
        let stop_signal = self
            .external_stop
            .clone()
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        let start_time = self.start_time.get();
        let time_limit = self.time_limit.get();
        let _tt_size = self.tt.borrow().size_mb();

        // Convert moves to a vector for parallel processing
        let moves: Vec<Move> = (0..num_moves).map(|i| legal_moves.get(i)).collect();

        // Use thread-local results
        let results: Vec<(Move, i32)> = moves
            .par_iter()
            .filter_map(|&mv| {
                // Check stop signal
                if stop_signal.load(Ordering::Relaxed) {
                    return None;
                }

                // Check time limit before starting work
                if let Some(limit) = time_limit {
                    let effective_limit = limit.mul_f64(0.95);
                    if start_time.elapsed() >= effective_limit {
                        return None;
                    }
                }

                // Clone position for this thread
                let mut thread_pos = position.clone();

                // Make the move
                thread_pos.make_move(mv);

                // Create a thread-local search instance
                let mut thread_search = Search::with_stop_signal(stop_signal.clone());

                // Set time tracking
                thread_search.start_time.set(start_time);
                thread_search.time_limit.set(time_limit);

                // Search this line - alphabeta checks time internally via should_stop()
                let (_, score) =
                    thread_search.alphabeta(&mut thread_pos, depth - 1, -32000, 32000, 1);
                let score = -score; // Negate for the other side's perspective

                Some((mv, score))
            })
            .collect();

        // Update total node count
        // TODO: This is an inaccurate estimate. The proper fix requires each thread-local
        // Search instance to track and return its actual node count. For now, we use a rough
        // estimate based on the number of moves searched.
        let total_nodes: u64 = results.len() as u64 * 1000; // Rough estimate
        self.nodes.fetch_add(total_nodes, Ordering::Relaxed);

        // Find best result
        results.into_iter().max_by_key(|(_, score)| *score)
    }

    /// Internal search implementation
    fn search_internal(
        &mut self,
        position: &Position,
        depth: u32,
        time_ms: Option<u64>,
        _parallel: bool,
    ) -> (Move, i32) {
        // Debug: Validate position before search
        if let Err(err) = position.board.validate() {
            eprintln!("=== INVALID POSITION AT SEARCH START ===");
            eprintln!("ERROR:\n{}", err);
            position.board.debug_print();
            eprintln!("======================================");
        }

        // Reset search state
        self.stop.store(false, Ordering::Relaxed);
        self.nodes.store(0, Ordering::Relaxed);
        self.start_time.set(Instant::now());
        self.time_limit.set(time_ms.map(Duration::from_millis));

        // New search generation for TT
        self.tt.borrow_mut().new_generation();

        // Clear history and killers for new search
        self.history.clear();
        self.killers.clear();

        let mut best_move = Move::null();
        let mut best_score = 0;

        // Get legal moves first
        let legal_moves = MoveGen::generate_legal_moves_ep(
            &position.board,
            position.state.side_to_move,
            position.state.ep_square,
            position.state.castling_rights,
        );

        // No legal moves - checkmate or stalemate
        if legal_moves.is_empty() {
            let king_sq = match position.board.king_square(position.state.side_to_move) {
                Some(sq) => sq,
                None => return (Move::null(), -32000),
            };

            let in_check = MoveGen::is_square_attacked(
                &position.board,
                king_sq,
                position.state.side_to_move.flip(),
            );

            if in_check {
                return (Move::null(), -32000); // Checkmate
            } else {
                return (Move::null(), 0); // Stalemate
            }
        }

        // Iterative deepening
        for d in 1..=depth {
            if self.should_stop() {
                break;
            }

            // Clone position for this depth iteration (alphabeta uses make_move_fast/undo_move_fast)
            let mut search_position = position.clone();
            let (mv, score) = self.alphabeta(&mut search_position, d, -32000, 32000, 0);

            // Check time after each depth iteration for bullet games
            // This is crucial for 1+0 bullet where each move must be very fast
            if self.should_stop() {
                // Don't use this iteration's result if we're out of time
                break;
            }

            // Validate the move from this iteration
            let mv_is_legal = !mv.is_null() && Self::is_move_in_list(mv, &legal_moves);

            // Only update best_move if the returned move is actually legal
            if mv_is_legal {
                best_move = mv;
                best_score = score;

                // Print UCI info
                let elapsed = self.start_time.get().elapsed();
                let nodes = self.nodes.load(Ordering::Relaxed);
                let nps = if elapsed.as_secs_f64() > 0.0 {
                    nodes as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };

                let stats = self.tt.borrow().stats();
                println!(
                    "info depth {} score cp {} nodes {} nps {:.0} time {} hashfull {}",
                    d,
                    score,
                    nodes,
                    nps,
                    elapsed.as_millis(),
                    stats.hashfull
                );
            }
        }

        // Final validation - ensure we always return a legal move
        if best_move.is_null() {
            // If no best move found, return the first legal move
            best_move = legal_moves.get(0);
        } else if !Self::is_move_in_list(best_move, &legal_moves) {
            best_move = legal_moves.get(0);
        }

        (best_move, best_score)
    }

    /// Helper to check if a move is in a move list
    fn is_move_in_list(mv: Move, moves: &crate::movelist::MoveList) -> bool {
        (0..moves.len()).any(|i| moves.get(i) == mv)
    }

    /// Check if a move is legal using fast validation
    /// Returns true if the move is both pseudo-legal and doesn't leave the king in check
    fn is_move_legal_optimized(position: &Position, mv: Move) -> bool {
        // First check if move is pseudo-legal
        if !MoveGen::is_pseudo_legal(
            &position.board,
            mv,
            position.state.side_to_move,
            position.state.ep_square,
            position.state.castling_rights,
        ) {
            return false;
        }
        // Then check if it leaves king in check
        MoveGen::is_move_legal(position, mv, position.state.side_to_move)
    }

    /// Quiescent search - extends search through captures and checks at leaf nodes
    /// This prevents the "horizon effect" where the engine makes bad moves
    /// because it doesn't see tactical sequences just beyond the search depth
    fn quiescent(
        &mut self,
        position: &mut Position,
        mut alpha: i32,
        beta: i32,
        qs_depth: usize,  // Quiescent search depth (starts at 0, not global ply)
    ) -> i32 {
        if self.should_stop() {
            return 0;
        }

        // Limit quiescent search depth to prevent stack overflow
        if qs_depth >= 10 {
            // At max depth, return static evaluation
            let static_eval = Evaluation::evaluate(&position.board);
            return if position.state.side_to_move == Color::White {
                static_eval
            } else {
                -static_eval
            };
        }

        let color = position.state.side_to_move;

        // Stand pat: use static evaluation as a lower bound
        let static_eval = Evaluation::evaluate(&position.board);
        let eval = if color == Color::White {
            static_eval
        } else {
            -static_eval
        };

        if eval >= beta {
            return beta;
        }

        if eval > alpha {
            alpha = eval;
        }

        // Generate captures only for quiescent search
        let mut move_picker = MovePicker::new_quiescent(position);

        while let Some(mv) = move_picker.next_move() {
            // Delta pruning: if capture can't improve alpha enough, skip it
            // (unless it's a promotion which can be worth much more)
            if !mv.is_promotion() {
                let captured_value = match position.board.get_piece(mv.to()) {
                    Some(piece) => match piece.piece_type {
                        crate::piece::PieceType::Pawn => 100,
                        crate::piece::PieceType::Knight => 320,
                        crate::piece::PieceType::Bishop => 330,
                        crate::piece::PieceType::Rook => 500,
                        crate::piece::PieceType::Queen => 900,
                        crate::piece::PieceType::King => 0,
                    },
                    None => 0,
                };

                // If even with the captured piece we can't exceed alpha by a queen, skip
                if eval + captured_value + 900 < alpha {
                    continue;
                }
            }

            position.make_move_fast(mv);
            let score = -self.quiescent(position, -beta, -alpha, qs_depth + 1);
            position.undo_move_fast();

            if score >= beta {
                return beta;
            }

            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }

    /// Alpha-beta search - SIMPLIFIED FOR DEBUGGING
    fn alphabeta(
        &mut self,
        position: &mut Position,
        depth: u32,
        alpha: i32,
        beta: i32,
        ply: usize,
    ) -> (Move, i32) {
        // Use mutable alpha for the search (allows TT to improve alpha)
        let mut alpha = alpha;

        // Prevent stack overflow
        if ply > 100 {
            return (Move::null(), alpha);
        }

        if self.should_stop() {
            return (Move::null(), 0);
        }

        let color = position.state.side_to_move;

        // Check if king is in check for check extension
        let king_sq = match position.board.king_square(color) {
            Some(sq) => sq,
            None => return (Move::null(), -32000),
        };
        let in_check = MoveGen::is_square_attacked(&position.board, king_sq, color.flip());

        // TEMPORARILY DISABLED: Check extension causing performance issues
        // Apply check extension: search deeper when in check
        let search_depth = depth; // if in_check { (depth + 2).min(20) } else { depth };

        // Get TT move for move ordering
        let tt_entry = self.tt.borrow().probe(position.state.hash);
        let tt_move = tt_entry.map(|e| e.best_move());

        // Validate TT move
        let validated_tt_move = tt_move.and_then(|mv| {
            if Self::is_move_legal_optimized(position, mv) {
                Some(mv)
            } else {
                None
            }
        });

        // Create move picker with history and killer tables
        let mut move_picker = MovePicker::new(
            position,
            validated_tt_move,
            ply,
            Some(&self.history),
            Some(&self.killers),
        );

        // No legal moves
        if move_picker.is_empty() {
            if in_check {
                return (Move::null(), -32000 + (search_depth as i32));
            } else {
                return (Move::null(), 0);
            }
        }

        // TT cutoff
        if let Some(entry) = tt_entry {
            if entry.depth() >= search_depth as u8 {
                let tt_mv = entry.best_move();
                if Self::is_move_legal_optimized(position, tt_mv) {
                    match entry.flag() {
                        TTFlag::Exact => return (tt_mv, entry.score()),
                        TTFlag::Lower => {
                            // Lower bound: score >= actual, only cutoff on beta
                            if entry.score() >= beta {
                                return (tt_mv, entry.score());
                            }
                            // Update alpha but continue searching
                            if entry.score() > alpha {
                                alpha = entry.score();
                            }
                        }
                        TTFlag::Upper => {
                            if entry.score() <= alpha {
                                return (tt_mv, entry.score());
                            }
                        }
                    }
                }
            }
        }

        // Leaf node - use quiescent search to extend through captures/checks
        if search_depth == 0 {
            let nodes = self.nodes.fetch_add(1, Ordering::Relaxed);
            if nodes % 1000000 == 0 {
                self.print_info();
            }
            // Use quiescent search to avoid horizon effect (start qs_depth at 0)
            let score = self.quiescent(position, alpha, beta, 0);
            return (Move::null(), score);
        }

        let mut best_move = Move::null();
        let mut best_score = alpha;
        let mut tt_flag = TTFlag::Upper;
        let mut searched_quiets: Vec<Move> = Vec::new();

        while let Some(mv) = move_picker.next_move() {
            // Track quiet moves for history updates
            let is_quiet = !is_capture(&position.board, color, mv)
                && !mv.is_promotion()
                && !mv.is_castle();
            if is_quiet {
                searched_quiets.push(mv);
            }

            if self.should_stop() {
                break;
            }

            // Simple alpha-beta search
            position.make_move_fast(mv);
            let score = -self.alphabeta(position, search_depth - 1, -beta, -alpha, ply + 1).1;
            position.undo_move_fast();

            if score > best_score {
                best_score = score;
                best_move = mv;
                tt_flag = TTFlag::Exact;

                if best_score >= beta {
                    tt_flag = TTFlag::Lower;

                    // Update killer move for quiet moves
                    if is_quiet {
                        self.killers.update(ply, mv);

                        // Update history: success for cutoff move, failure for other searched quiets
                        if let Some(piece) = position.board.get_piece(mv.from()) {
                            self.history.update_success(color, piece.piece_type, mv.to(), search_depth);

                            // Penalize all other quiet moves that were searched before this cutoff
                            for &quiet_mv in &searched_quiets {
                                if quiet_mv != mv {
                                    if let Some(p) = position.board.get_piece(quiet_mv.from()) {
                                        self.history.update_failure(color, p.piece_type, quiet_mv.to(), search_depth);
                                    }
                                }
                            }
                        }
                    }

                    break;
                }
            }
        }

        // Store in TT
        self.tt
            .borrow_mut()
            .store(position.state.hash, best_move, best_score, search_depth as u8, tt_flag);

        (best_move, best_score)
    }

    /// Check if search should stop
    fn should_stop(&self) -> bool {
        // Check internal stop flag
        if self.stop.load(Ordering::Relaxed) {
            return true;
        }

        // Check external stop signal (from main thread)
        // Check immediately without any delay - we want to respond to UCI stop instantly
        if let Some(ref external) = self.external_stop {
            if external.load(Ordering::Relaxed) {
                return true;
            }
        }

        // Check time limit - use 95% of allocated time to leave safety margin
        if let Some(limit) = self.time_limit.get() {
            // Use 95% of time limit to leave margin for move output
            let effective_limit = limit.mul_f64(0.95);
            let elapsed = self.start_time.get().elapsed();
            if elapsed >= effective_limit {
                return true;
            }
        }

        false
    }

    /// Stop the search
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Get the number of nodes searched
    pub fn nodes(&self) -> u64 {
        self.nodes.load(Ordering::Relaxed)
    }

    /// Print search info
    fn print_info(&self) {
        let elapsed = self.start_time.get().elapsed();
        let nodes = self.nodes.load(Ordering::Relaxed);
        let nps = if elapsed.as_secs_f64() > 0.0 {
            nodes as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        println!(
            "info nodes {} nps {:.0} time {}",
            nodes,
            nps,
            elapsed.as_millis()
        );
    }
}

impl Default for Search {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_start() {
        let position = Position::from_start();
        let mut search = Search::new();
        let (best_move, score) = search.search(&position, 3, None);

        assert!(!best_move.is_null());
        println!("Best move: {}, score: {}", best_move, score);
    }
}
