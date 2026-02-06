use crate::evaluation::Evaluation;
use crate::movegen::MoveGen;
use crate::piece::Color;
use crate::position::Position;
use crate::r#move::Move;
use crate::transposition::{TTFlag, TranspositionTable};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Search engine
pub struct Search {
    /// Flag to stop the search (internal)
    stop: AtomicBool,
    /// External stop signal (shared with main thread)
    external_stop: Option<Arc<AtomicBool>>,
    /// Nodes searched
    nodes: AtomicU64,
    /// Search start time (wrapped for interior mutability)
    start_time: UnsafeCell<Instant>,
    /// Time limit (wrapped for interior mutability)
    time_limit: UnsafeCell<Option<Duration>>,
    /// Transposition table
    tt: TranspositionTable,
}

impl Search {
    /// Create a new search
    pub fn new() -> Self {
        Search {
            stop: AtomicBool::new(false),
            external_stop: None,
            nodes: AtomicU64::new(0),
            start_time: UnsafeCell::new(Instant::now()),
            time_limit: UnsafeCell::new(None),
            tt: TranspositionTable::new(256), // 256 MB TT
        }
    }

    /// Create a new search with external stop signal
    pub fn with_stop_signal(external_stop: Arc<AtomicBool>) -> Self {
        Search {
            stop: AtomicBool::new(false),
            external_stop: Some(external_stop),
            nodes: AtomicU64::new(0),
            start_time: UnsafeCell::new(Instant::now()),
            time_limit: UnsafeCell::new(None),
            tt: TranspositionTable::new(256),
        }
    }

    /// Create a new search with custom TT size
    pub fn with_tt_size(tt_size_mb: usize) -> Self {
        Search {
            stop: AtomicBool::new(false),
            external_stop: None,
            nodes: AtomicU64::new(0),
            start_time: UnsafeCell::new(Instant::now()),
            time_limit: UnsafeCell::new(None),
            tt: TranspositionTable::new(tt_size_mb),
        }
    }

    /// Create a new search with custom TT size and external stop signal
    pub fn with_config(tt_size_mb: usize, external_stop: Arc<AtomicBool>) -> Self {
        Search {
            stop: AtomicBool::new(false),
            external_stop: Some(external_stop),
            nodes: AtomicU64::new(0),
            start_time: UnsafeCell::new(Instant::now()),
            time_limit: UnsafeCell::new(None),
            tt: TranspositionTable::new(tt_size_mb),
        }
    }

    /// Run an iterative deepening search
    pub fn search(&mut self, position: &Position, depth: u32, time_ms: Option<u64>) -> (Move, i32) {
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
        unsafe {
            *self.start_time.get() = Instant::now();
            *self.time_limit.get() = time_ms.map(Duration::from_millis);
        }

        // New search generation for TT
        self.tt.new_generation();

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

            let (mv, score) = self.alphabeta(position, d, -32000, 32000);

            // Validate the move from this iteration
            let mut mv_is_legal = false;
            if !mv.is_null() {
                for i in 0..legal_moves.len() {
                    if legal_moves.get(i) == mv {
                        mv_is_legal = true;
                        break;
                    }
                }
            }

            // Only update best_move if the returned move is actually legal
            if mv_is_legal {
                best_move = mv;
                best_score = score;

                // Print UCI info
                let elapsed = unsafe { (*self.start_time.get()).elapsed() };
                let nodes = self.nodes.load(Ordering::Relaxed);
                let nps = if elapsed.as_secs_f64() > 0.0 {
                    nodes as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };

                let stats = self.tt.stats();
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
        } else {
            // Double-check the move is still legal
            let mut move_is_legal = false;
            for i in 0..legal_moves.len() {
                if legal_moves.get(i) == best_move {
                    move_is_legal = true;
                    break;
                }
            }
            if !move_is_legal {
                best_move = legal_moves.get(0);
            }
        }

        (best_move, best_score)
    }

    /// Alpha-beta search
    fn alphabeta(&self, position: &Position, depth: u32, mut alpha: i32, beta: i32) -> (Move, i32) {
        if self.should_stop() {
            return (Move::null(), 0);
        }

        let color = position.state.side_to_move;

        // Generate legal moves first (needed for TT validation)
        let moves = MoveGen::generate_legal_moves_ep(
            &position.board,
            color,
            position.state.ep_square,
            position.state.castling_rights,
        );

        // No legal moves - checkmate or stalemate
        if moves.is_empty() {
            // Check if king is in check
            let king_sq = match position.board.king_square(color) {
                Some(sq) => sq,
                None => return (Move::null(), -32000), // King captured
            };

            let in_check = MoveGen::is_square_attacked(&position.board, king_sq, color.flip());

            if in_check {
                // Checkmate
                return (Move::null(), -32000 + (depth as i32));
            } else {
                // Stalemate
                return (Move::null(), 0);
            }
        }

        // Check transposition table
        if let Some(entry) = self.tt.probe(position.state.hash) {
            if entry.depth >= depth as u8 {
                // Verify TT move is actually legal before returning
                let tt_move = entry.best_move;

                // First check if move is pseudo-legal (catches most illegal moves quickly)
                let tt_move_pseudo_legal = MoveGen::is_pseudo_legal(
                    &position.board,
                    tt_move,
                    color,
                    position.state.ep_square,
                    position.state.castling_rights,
                );

                // Then check if it's in our generated legal moves list
                let mut tt_move_legal = false;
                if tt_move_pseudo_legal {
                    for i in 0..moves.len() {
                        if moves.get(i) == tt_move {
                            tt_move_legal = true;
                            break;
                        }
                    }
                }

                if tt_move_legal {
                    // Be more conservative with TT moves - only use them if they're from
                    // a sufficiently deep search and the score is within bounds
                    if entry.depth >= depth as u8 {
                        match entry.flag {
                            TTFlag::Exact => {
                                // For exact scores, we can trust the TT move more
                                if depth <= 3 || entry.depth >= depth as u8 + 2 {
                                    return (tt_move, entry.score);
                                }
                            }
                            TTFlag::Lower => {
                                if entry.score >= beta {
                                    // For beta cutoffs, be more careful
                                    if depth <= 2 || entry.depth >= depth as u8 + 1 {
                                        return (tt_move, entry.score);
                                    }
                                }
                            }
                            TTFlag::Upper => {
                                if entry.score <= alpha {
                                    // For alpha cutoffs, be more careful
                                    if depth <= 2 || entry.depth >= depth as u8 + 1 {
                                        return (tt_move, entry.score);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Leaf node
        if depth == 0 {
            let nodes = self.nodes.fetch_add(1, Ordering::Relaxed);
            if nodes % 1000000 == 0 {
                self.print_info();
            }
            let score = Evaluation::evaluate(&position.board);
            return (
                Move::null(),
                if color == Color::White { score } else { -score },
            );
        }

        let mut best_move = Move::null();
        let mut tt_flag = TTFlag::Upper;

        for i in 0..moves.len() {
            let mv = moves.get(i);

            // Make move by cloning (safe for recursion)
            let mut new_position = position.clone();
            new_position.make_move(mv);

            // Search with negated score
            let (_, score) = self.alphabeta(&new_position, depth - 1, -beta, -alpha);

            // Negate score for opponent's perspective
            let score = -score;

            if score > alpha {
                alpha = score;
                best_move = mv;
                tt_flag = TTFlag::Exact;
            }

            if alpha >= beta {
                tt_flag = TTFlag::Lower;
                break; // Beta cutoff
            }
        }

        // Store in transposition table
        self.tt
            .store(position.state.hash, best_move, alpha, depth as u8, tt_flag);

        // Final validation: ensure best_move is actually legal
        if !best_move.is_null() {
            let mut move_is_legal = false;
            for i in 0..moves.len() {
                if moves.get(i) == best_move {
                    move_is_legal = true;
                    break;
                }
            }
            if !move_is_legal {
                // Debug: This should never happen if our search is working correctly
                println!(
                    "WARNING: Alpha-beta returned illegal move {} at depth {}",
                    best_move, depth
                );
                println!("Available legal moves: {}", moves.len());

                // If best_move is not legal, return the first legal move instead
                if moves.len() > 0 {
                    best_move = moves.get(0);
                    println!("Fallback: Using first legal move {}", best_move);
                } else {
                    best_move = Move::null();
                    println!("Fallback: No legal moves available");
                }
            }
        }

        (best_move, alpha)
    }

    /// Check if search should stop
    fn should_stop(&self) -> bool {
        // Check internal stop flag
        if self.stop.load(Ordering::Relaxed) {
            return true;
        }

        // Check external stop signal (from main thread)
        // BUT only after minimum search time to avoid race conditions
        let elapsed = unsafe { (*self.start_time.get()).elapsed() };
        if elapsed >= Duration::from_millis(50) {
            if let Some(ref external) = self.external_stop {
                if external.load(Ordering::Relaxed) {
                    return true;
                }
            }
        }

        // Check time limit
        unsafe {
            if let Some(limit) = *self.time_limit.get() {
                if elapsed >= limit {
                    return true;
                }
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
        let (elapsed, nodes) = unsafe {
            (
                (*self.start_time.get()).elapsed(),
                self.nodes.load(Ordering::Relaxed),
            )
        };
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
