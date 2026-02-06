use chessy::movegen::MoveGen;
use chessy::piece::Color;
use chessy::position::Position;
use chessy::r#move::{Move, PromotionType};
use chessy::search::Search;
use chessy::utils::{square_from_string, square_to_string};
use std::io::{self, BufRead};
use std::sync::atomic::AtomicBool;
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "--perft" => {
                let depth = args.get(2).and_then(|d| d.parse::<u32>().ok()).unwrap_or(1);
                run_perft(depth);
            }
            "--test-illegal" => {
                test_illegal_move();
            }
            _ => {
                eprintln!("Unknown argument: {}", args[1]);
                eprintln!("Usage: chessy [--perft depth] [--test-illegal]");
                std::process::exit(1);
            }
        }
    } else {
        run_uci_mode();
    }
}

fn run_uci_mode() {
    let stdin = io::stdin();
    let mut position = Position::from_start();
    let stop_signal = Arc::new(AtomicBool::new(false));
    let mut search = Search::with_stop_signal(stop_signal.clone());

    loop {
        let mut line = String::new();
        if !stdin.lock().read_line(&mut line).is_ok() {
            break;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();

        match parts[0] {
            "uci" => {
                println!("id name Chessy");
                println!("id author Chessy");
                println!("option name Hash type spin default 256 min 1 max 2048");
                println!("uciok");
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                search = Search::with_stop_signal(stop_signal.clone());
                position = Position::from_start();
            }
            "position" => {
                handle_position(&parts, &mut position);
            }
            "go" => {
                let stop_signal_clone = stop_signal.clone();
                handle_go(&parts, &mut position, &mut search, &stop_signal_clone);
            }
            "stop" => {
                stop_signal.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            "quit" => {
                break;
            }
            "setoption" => {
                // Handle setoption if needed
                if parts.len() >= 5 && parts[1] == "name" && parts[3] == "value" {
                    let name = parts[2];
                    let value = parts[4];
                    if name == "Hash" {
                        if let Ok(size_mb) = value.parse::<usize>() {
                            search = Search::with_config(size_mb, stop_signal.clone());
                        }
                    }
                }
            }
            "d" => {
                // Display current position (debug command)
                println!("{}", position.board.to_string());
                println!("Side to move: {:?}", position.state.side_to_move);
                println!("Castling rights: {:04b}", position.state.castling_rights);
                if let Some(ep) = position.state.ep_square {
                    println!("En passant: {}", chessy::utils::square_to_string(ep));
                }
                println!("Halfmove clock: {}", position.state.halfmove_clock);
                println!("Fullmove number: {}", position.state.fullmove_number);
            }
            "perft" => {
                // Run perft test
                if parts.len() > 1 {
                    if let Ok(depth) = parts[1].parse::<u32>() {
                        let nodes = perft(&position, depth);
                        println!("nodes {}", nodes);
                    }
                }
            }
            _ => {
                // Ignore unknown commands
            }
        }
    }
}

fn handle_position(parts: &[&str], position: &mut Position) {
    if parts.len() < 2 {
        return;
    }

    if parts[1] == "startpos" {
        *position = Position::from_start();

        // Parse moves if present
        if parts.len() > 2 && parts[2] == "moves" {
            for i in 3..parts.len() {
                if let Ok(mv) = move_from_string(parts[i]) {
                    let from_sq = mv.from();
                    // Check if there's actually a piece on the from square BEFORE making the move
                    if position.board.get_piece(from_sq).is_none() {
                        eprintln!(
                            "STATE CORRUPTION: No piece on {} for move '{}' (move #{}/{})",
                            square_to_string(from_sq),
                            parts[i],
                            i - 2,
                            parts.len() - 3
                        );
                        eprintln!("  Skipping move and continuing");
                        continue; // Skip this move instead of making it
                    }
                    position.make_move(mv);
                } else {
                    eprintln!("ERROR: Failed to parse move '{}'", parts[i]);
                }
            }
        }
    } else if parts[1] == "fen" {
        // Extract FEN string (parts 2..7)
        let fen: String = parts[2..8.min(parts.len())].join(" ");
        if let Ok(new_position) = Position::from_fen(&fen) {
            *position = new_position;

            // Parse moves if present
            if let Some(moves_idx) = parts.iter().position(|&x| x == "moves") {
                for i in (moves_idx + 1)..parts.len() {
                    if let Ok(mv) = move_from_string(parts[i]) {
                        let from_sq = mv.from();
                        // Check if there's actually a piece on the from square BEFORE making the move
                        if position.board.get_piece(from_sq).is_none() {
                            eprintln!(
                                "STATE CORRUPTION (FEN): No piece on {} for move '{}'",
                                square_to_string(from_sq),
                                parts[i]
                            );
                            eprintln!("  Skipping move and continuing");
                            continue; // Skip this move instead of making it
                        }
                        position.make_move(mv);
                    } else {
                        eprintln!("ERROR: Failed to parse move '{}'", parts[i]);
                    }
                }
            }
        }
    }
}

fn handle_go(
    parts: &[&str],
    position: &mut Position,
    search: &mut Search,
    stop_signal: &Arc<AtomicBool>,
) {
    // IMPORTANT: Reset stop signal before starting a new search
    stop_signal.store(false, std::sync::atomic::Ordering::Relaxed);

    // Parse go command parameters
    let mut wtime: Option<u64> = None;
    let mut btime: Option<u64> = None;
    let mut winc: u64 = 0;
    let mut binc: u64 = 0;
    let mut movestogo: Option<u32> = None;
    let mut depth: u32 = 10;
    let mut nodes: Option<u64> = None;
    let mut mate: Option<u32> = None;
    let mut movetime: Option<u64> = None;

    let mut i = 1;
    while i < parts.len() {
        match parts[i] {
            "wtime" => {
                i += 1;
                wtime = parts.get(i).and_then(|t| t.parse().ok());
            }
            "btime" => {
                i += 1;
                btime = parts.get(i).and_then(|t| t.parse().ok());
            }
            "winc" => {
                i += 1;
                winc = parts.get(i).and_then(|t| t.parse().ok()).unwrap_or(0);
            }
            "binc" => {
                i += 1;
                binc = parts.get(i).and_then(|t| t.parse().ok()).unwrap_or(0);
            }
            "movestogo" => {
                i += 1;
                movestogo = parts.get(i).and_then(|t| t.parse().ok());
            }
            "depth" => {
                i += 1;
                depth = parts.get(i).and_then(|d| d.parse().ok()).unwrap_or(10);
            }
            "nodes" => {
                i += 1;
                nodes = parts.get(i).and_then(|n| n.parse().ok());
            }
            "mate" => {
                i += 1;
                mate = parts.get(i).and_then(|m| m.parse().ok());
            }
            "movetime" => {
                i += 1;
                movetime = parts.get(i).and_then(|t| t.parse().ok());
            }
            _ => {
                i += 1;
            }
        }
    }

    // Calculate time budget
    let time_ms = calculate_time_budget(
        position.state.side_to_move,
        wtime,
        btime,
        winc,
        binc,
        movestogo,
        position.state.fullmove_number,
    );

    // Adaptive depth based on time budget for bullet games
    let time_budget_ms = time_ms.unwrap_or(0);
    if time_budget_ms > 0 && time_budget_ms < 50 {
        // Ultra-fast bullet: extremely shallow search (depth 1 only)
        depth = depth.min(1); // Depth 1 for <50ms budget
    } else if time_budget_ms > 0 && time_budget_ms < 100 {
        // Fast bullet: very shallow search
        depth = depth.min(2); // Cap at depth 2 for <100ms per move
    } else if time_budget_ms > 0 && time_budget_ms < 200 {
        // Bullet game: shallow search
        depth = depth.min(3); // Cap at depth 3 for very fast games
    } else if time_budget_ms > 0 && time_budget_ms < 500 {
        // Fast blitz: moderate depth
        depth = depth.min(5);
    }

    // Validate position before cloning (comprehensive check)
    if let Err(err) = position.validate_position() {
        eprintln!("ERROR: Invalid position state before search!");
        eprintln!("{}", err);
        eprintln!("Position FEN: {}", position.to_fen());
        position.board.debug_print();
        // Position is corrupted - return null move immediately
        println!("bestmove 0000");
        return;
    }

    // Also check board consistency
    if let Err(err) = position.board.validate() {
        eprintln!("ERROR: Invalid board state before search!");
        eprintln!("{}", err);
        position.board.debug_print();
        // Board is corrupted - return null move immediately
        println!("bestmove 0000");
        return;
    }

    // Clone position for thread BEFORE search starts
    // Create a fresh copy to avoid race conditions during search
    let position_clone = position.clone();

    // Debug: Check if the cloned position is identical to the original
    if position_clone.state.hash != position.state.hash {
        eprintln!(
            "WARNING: Position hash mismatch! Clone: {}, Original: {}",
            position_clone.state.hash, position.state.hash
        );
    }

    // Create channel for sending best move back
    let (tx, rx) = mpsc::channel();

    // Clone stop signal for the search thread
    let stop_signal_for_search = stop_signal.clone();

    // Move the configured search into the thread (TT configuration is preserved)
    // Note: After this move, the main thread's search reference is invalidated
    // It will be recreated on the next ucinewgame or setoption command
    let mut search_to_move = std::mem::replace(search, Search::new());

    // Spawn search thread
    thread::spawn(move || {
        // Attach the stop signal to the existing search (preserves TT)
        search_to_move.set_stop_signal(stop_signal_for_search);

        let time_limit = match (movetime, mate, nodes) {
            (Some(mt), _, _) => Some(mt),
            (_, Some(_), _) | (_, _, Some(_)) => None,
            _ => time_ms,
        };

        let search_depth = mate.unwrap_or(depth);

        let (best_move, _score) = search_to_move.search(&position_clone, search_depth, time_limit);

        tx.send((best_move, search_to_move.nodes())).ok();
    });

    // Wait for result or stop signal
    let mut best_move = Move::null();
    let start = std::time::Instant::now();

    if let Some(mt) = movetime {
        // Wait for movetime or thread completion
        let timeout = Duration::from_millis(mt + 100);
        let mut elapsed = Duration::from_secs(0);

        while elapsed < timeout {
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok((mv, _nodes)) => {
                    best_move = mv;
                    break;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    elapsed = start.elapsed();
                    if stop_signal.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    break;
                }
            }
        }

        // Stop the search
        stop_signal.store(true, std::sync::atomic::Ordering::Relaxed);
    } else if nodes.is_some() {
        // Search until node count reached (enforced internally)
        // Note: The search thread sends only one result when complete
        loop {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok((mv, _searched_nodes)) => {
                    // Always accept the result when the search completes
                    // The node limit is enforced inside the search itself
                    best_move = mv;
                    break;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if stop_signal.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    break;
                }
            }
        }

        stop_signal.store(true, std::sync::atomic::Ordering::Relaxed);
    } else {
        // Wait for search completion or timeout
        loop {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok((mv, _nodes)) => {
                    best_move = mv;
                    break;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if stop_signal.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    break;
                }
            }
        }

        stop_signal.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    // Receive final result (in case search completed)
    if let Ok((mv, _nodes)) = rx.try_recv() {
        best_move = mv;
    }

    // Validate the move before outputting
    if !best_move.is_null() {
        let legal_moves = MoveGen::generate_legal_moves_ep(
            &position.board,
            position.state.side_to_move,
            position.state.ep_square,
            position.state.castling_rights,
        );
        let mut move_is_legal = false;
        for i in 0..legal_moves.len() {
            if legal_moves.get(i) == best_move {
                move_is_legal = true;
                break;
            }
        }
        if !move_is_legal {
            // Debug output to stderr for visibility
            eprintln!("");
            eprintln!("=== ILLEGAL MOVE DETECTED ===");
            eprintln!(
                "Move: {} for side {:?}",
                best_move, position.state.side_to_move
            );
            eprintln!("Position state:");
            eprintln!("  Side to move: {:?}", position.state.side_to_move);
            eprintln!("  Fullmove: {}", position.state.fullmove_number);
            eprintln!("  Hash: {}", position.state.hash);
            eprintln!("Board:");
            eprintln!("{}", position.board.to_string());

            // Output FEN for debugging
            eprintln!("Position FEN: {}", position.to_fen());

            // Check what piece (if any) is on the from square
            let from_sq = best_move.from();
            let to_sq = best_move.to();
            if let Some(piece) = position.board.get_piece(from_sq) {
                eprintln!(
                    "Piece on {}: {:?} {:?}",
                    from_sq, piece.color, piece.piece_type
                );
                if piece.color != position.state.side_to_move {
                    eprintln!(
                        "ERROR: Wrong color piece! Expected {:?}, got {:?}",
                        position.state.side_to_move, piece.color
                    );
                }
            } else {
                eprintln!("ERROR: No piece on from square {}", from_sq);
            }

            // Check what's on the to square
            if let Some(piece) = position.board.get_piece(to_sq) {
                eprintln!(
                    "Destination {} has: {:?} {:?}",
                    to_sq, piece.color, piece.piece_type
                );
            } else {
                eprintln!("Destination {} is empty", to_sq);
            }

            // Check king position
            let king_sq = position.board.king_square(position.state.side_to_move);
            if let Some(sq) = king_sq {
                let in_check = MoveGen::is_square_attacked(
                    &position.board,
                    sq,
                    position.state.side_to_move.flip(),
                );
                eprintln!("King at {} in_check={}", sq, in_check);
            } else {
                eprintln!(
                    "ERROR: No king found for side {:?}",
                    position.state.side_to_move
                );
            }
            eprintln!("==============================");

            // Try to find a better fallback move instead of just using the first one
            let mut fallback_move = Move::null();
            if legal_moves.len() > 0 {
                // Try to find a move that's similar to the illegal one
                // (same piece type, similar direction, etc.)
                let illegal_from = best_move.from();
                let illegal_to = best_move.to();

                // First try: same from and to squares (ideal)
                for i in 0..legal_moves.len() {
                    let candidate = legal_moves.get(i);
                    if candidate.from() == illegal_from && candidate.to() == illegal_to {
                        fallback_move = candidate;
                        println!(
                            "info string Found fallback with same from/to: {}",
                            candidate
                        );
                        break;
                    }
                }

                // Second try: same to square (similar destination)
                if fallback_move.is_null() {
                    for i in 0..legal_moves.len() {
                        let candidate = legal_moves.get(i);
                        if candidate.to() == illegal_to {
                            fallback_move = candidate;
                            println!(
                                "info string Found fallback with same to square: {}",
                                candidate
                            );
                            break;
                        }
                    }
                }

                // Third try: same from square (same piece)
                if fallback_move.is_null() {
                    for i in 0..legal_moves.len() {
                        let candidate = legal_moves.get(i);
                        if candidate.from() == illegal_from {
                            fallback_move = candidate;
                            println!(
                                "info string Found fallback with same from square: {}",
                                candidate
                            );
                            break;
                        }
                    }
                }

                // Fallback: first legal move
                if fallback_move.is_null() {
                    fallback_move = legal_moves.get(0);
                    println!(
                        "info string Using first legal move as fallback: {}",
                        fallback_move
                    );
                }
            } else {
                println!("info string No legal moves available!");
            }

            if !fallback_move.is_null() {
                println!(
                    "info string Replacing illegal {} with {}",
                    best_move, fallback_move
                );
                best_move = fallback_move;
            } else {
                best_move = Move::null();
            }
        }
    }

    // Output best move
    println!("bestmove {}", best_move);

    // Reset stop signal
    stop_signal.store(false, std::sync::atomic::Ordering::Relaxed);
}

fn calculate_time_budget(
    side_to_move: Color,
    wtime: Option<u64>,
    btime: Option<u64>,
    winc: u64,
    binc: u64,
    movestogo: Option<u32>,
    fullmove_number: u32,
) -> Option<u64> {
    let (my_time, my_inc) = if side_to_move == Color::White {
        (wtime?, winc)
    } else {
        (btime?, binc)
    };

    let moves_left = movestogo.unwrap_or_else(|| {
        // Estimate remaining moves, with minimum of 5 to avoid overly aggressive allocation
        let estimated_moves: u32 = 40;
        estimated_moves.saturating_sub(fullmove_number).max(5)
    });

    // Very aggressive time allocation for bullet (1+0) games
    // For bullet, use 1/10 to 1/20 of remaining time based on game phase
    // This is extremely aggressive but necessary for 1+0
    let is_bullet = my_time < 2000; // Less than 2 seconds = bullet
    let time_fraction = if is_bullet {
        if moves_left <= 10 {
            20
        } else if moves_left >= 30 {
            30
        } else {
            25
        }
    } else {
        if moves_left <= 10 {
            3
        } else if moves_left >= 30 {
            5
        } else {
            4
        }
    };

    let mut allocated = my_time / time_fraction;
    allocated = allocated.saturating_add(my_inc);

    // For bullet, use extremely small minimum and proportional buffer
    let allocated = if is_bullet {
        // For bullet: minimum 10ms, with proportional buffer (1/4 to 200ms max)
        // Ensure we don't underflow when subtracting buffer
        let buffer = (my_time / 4).min(200);
        let capped = my_time.saturating_sub(buffer).max(10);
        allocated.max(10).min(capped)
    } else {
        allocated.max(100).min(my_time.saturating_sub(50))
    };

    Some(allocated)
}

fn move_from_string(s: &str) -> Result<Move, String> {
    if s.len() < 4 {
        return Err("Move string too short".to_string());
    }

    let chars: Vec<char> = s.chars().collect();

    let from = square_from_string(&s[0..2]).ok_or("Invalid from square")?;
    let to = square_from_string(&s[2..4]).ok_or("Invalid to square")?;

    // Detect castling moves in UCI format
    // Kingside: e1g1 (white), e8g8 (black)
    // Queenside: e1c1 (white), e8c8 (black)
    // Our encoding uses the king's destination directly
    let is_castle = (from == 4 && (to == 6 || to == 2)) ||   // White: e1 to g1 or c1
                    (from == 60 && (to == 62 || to == 58)); // Black: e8 to g8 or c8

    if is_castle {
        return Ok(Move::castle(from, to));
    }

    // Handle promotion
    if s.len() >= 5 {
        let promo_char = chars[4];
        let promo_type = match promo_char {
            'n' | 'N' => PromotionType::Knight,
            'b' | 'B' => PromotionType::Bishop,
            'r' | 'R' => PromotionType::Rook,
            'q' | 'Q' => PromotionType::Queen,
            _ => return Err(format!("Invalid promotion piece: {}", promo_char)),
        };
        Ok(Move::promotion(from, to, promo_type))
    } else {
        Ok(Move::new(from, to))
    }
}

fn run_perft(depth: u32) {
    let position = Position::from_start();
    println!("Running perft to depth {}", depth);

    let start = std::time::Instant::now();
    let nodes = perft(&position, depth);
    let elapsed = start.elapsed();

    println!("\nPerft {} result: {}", depth, nodes);
    println!("Time: {:.2}s", elapsed.as_secs_f64());
    println!("Nodes/second: {:.0}", nodes as f64 / elapsed.as_secs_f64());
}

fn perft(position: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = MoveGen::generate_from_position(position);
    let mut nodes = 0u64;

    for i in 0..moves.len() {
        let mv = moves.get(i);
        let mut new_pos = position.clone();
        new_pos.make_move(mv);
        nodes += perft(&new_pos, depth - 1);
    }

    nodes
}

fn test_illegal_move() {
    use chessy::position::Position;
    use chessy::r#move::Move;

    // Create a test to investigate the illegal move
    let mut pos = Position::from_start();

    // Let's try to reproduce a scenario where c8g4 might be played
    println!("Starting position:");
    println!("{}", pos.board.to_string());
    println!();

    // Make some moves that could lead to c8g4
    let test_moves = [
        (12, 28), // e2e4
        (52, 36), // e7e5
        (6, 21),  // g1f3
        (62, 45), // g8f6
        (5, 12),  // f1e2
        (51, 35), // d7d5
        (3, 19),  // d2d4
    ];

    for (from, to) in test_moves {
        let mv = Move::new(from, to);
        pos.make_move(mv);
        println!("After {}:", mv);
        println!("Side to move: {:?}", pos.state.side_to_move);
        println!(
            "King square: {:?}",
            pos.board.king_square(pos.state.side_to_move)
        );
        println!();
    }

    println!("Current position:");
    println!("{}", pos.board.to_string());
    println!();

    // Check if c8g4 would be legal
    let _c8g4 = Move::new(59, 46); // c8 to g4 (using correct squares)

    println!("Checking if c8g4 is legal...");
    use chessy::movegen::MoveGen;

    let legal_moves = MoveGen::generate_from_position(&pos);
    println!("Number of legal moves: {}", legal_moves.len());

    // Check if c8g4 is in legal moves
    let mut found = false;
    for i in 0..legal_moves.len() {
        let mv = legal_moves.get(i);
        if mv.from() == 59 && mv.to() == 46 {
            println!("Found c8g4 in legal moves!");
            found = true;
            break;
        }
    }

    if !found {
        println!("c8g4 NOT found in legal moves");

        // Check if it's in pseudo-legal moves
        let pseudo_moves = MoveGen::generate_moves_ep(
            &pos.board,
            pos.state.side_to_move,
            pos.state.ep_square,
            pos.state.castling_rights,
        );
        println!("Number of pseudo-legal moves: {}", pseudo_moves.len());

        for i in 0..pseudo_moves.len() {
            let mv = pseudo_moves.get(i);
            if mv.from() == 59 && mv.to() == 46 {
                println!("c8g4 found in pseudo-legal moves!");
                println!("This is the bug - pseudo-legal move passed legal check!");
                break;
            }
        }
    }

    // Check if black king is in check
    let king_sq = pos.board.king_square(chessy::piece::Color::Black);
    println!("\nBlack king square: {:?}", king_sq);

    if let Some(sq) = king_sq {
        let in_check = MoveGen::is_square_attacked(&pos.board, sq, chessy::piece::Color::White);
        println!("Black king in check: {}", in_check);
    }
}
