// Analyze tactical weaknesses in Chessy
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

fn main() {
    // Connect to Stockfish
    let mut stockfish = TcpStream::connect("127.0.0.1:4000").expect("Failed to connect to Stockfish");

    // Read greeting
    let mut reader = BufReader::new(stockfish.try_clone().expect("Failed to clone"));
    let mut line = String::new();
    reader.read_line(&mut line).ok();
    println!("Stockfish: {}", line.trim());

    // Initialize Stockfish
    writeln!(stockfish, "uci").ok();
    let mut response = read_until(&mut reader, "uciok").unwrap();
    println!("UCI response:\n{}", response);

    writeln!(stockfish, "isready").ok();
    reader.read_line(&mut line).ok();

    // Play a game and analyze positions
    writeln!(stockfish, "ucinewgame").ok();
    writeln!(stockfish, "position startpos").ok();

    // Set Stockfish to low skill level (approx 1350 ELO)
    writeln!(stockfish, "setoption name Skill Level value 5").ok();
    writeln!(stockfish, "setoption name Contempt value 100").ok();

    let positions = Arc::new(Mutex::new(Vec::new()));

    // Play several games and collect critical positions
    for game in 0..5 {
        println!("\n=== Game {} ===", game + 1);
        writeln!(stockfish, "ucinewgame").ok();
        writeln!(stockfish, "position startpos").ok();

        let mut movenum = 0;
        let game_over = false;
        let mut fen = String::from("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        while !game_over && movenum < 100 {
            // Get Stockfish's evaluation and best move
            writeln!(stockfish, "go depth 10").ok();

            let stockfish_move = match read_move(&mut reader, 30000) {
                Some(mv) => mv,
                None => break,
            };

            if stockfish_move == "(none)" || stockfish_move.is_empty() {
                println!("Game over - Stockfish has no moves");
                break;
            }

            // Get evaluation before move
            writeln!(stockfish, "eval {}", fen).ok();
            let eval_before = read_eval(&mut reader);

            println!("Move {}: Stockfish plays {} (eval: {})", movenum + 1, stockfish_move, eval_before);

            // Make the move
            fen = apply_move(&fen, &stockfish_move);
            movenum += 1;

            // Check for checkmate
            if is_game_over(&fen) {
                println!("Game over!");
                break;
            }

            // Get position after move
            writeln!(stockfish, "position fen {}", fen).ok();

            // Analyze if there are tactical opportunities
            writeln!(stockfish, "go depth 12 searchmoves q").ok();
            let tactics = read_analysis(&mut reader, 5000);

            if !tactics.is_empty() && eval_before.parse::<i32>().unwrap_or(0) < 200 {
                println!("  TACTICAL OPPORTUNITY: {}", tactics);
                positions.lock().unwrap().push((fen.clone(), eval_before.clone(), tactics));
            }
        }
    }

    // Summary
    println!("\n\n=== TACTICAL WEAKNESS ANALYSIS ===");
    println!("Collected {} critical positions", positions.lock().unwrap().len());
}

fn read_until(reader: &mut BufReader<TcpStream>, until: &str) -> Result<String, String> {
    let mut output = String::new();
    let mut line = String::new();

    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => return Err("EOF".to_string()),
            Ok(_) => {
                output.push_str(&line);
                if line.trim() == until {
                    return Ok(output);
                }
            }
            Err(e) => return Err(e.to_string()),
        }
    }
}

fn read_move(reader: &mut BufReader<TcpStream>, timeout_ms: u64) -> Option<String> {
    let start = std::time::Instant::now();
    let mut line = String::new();
    let mut bestmove = String::new();

    while start.elapsed().as_millis() < timeout_ms {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                if line.trim().starts_with("bestmove") {
                    let parts: Vec<&str> = line.trim().split_whitespace().collect();
                    if parts.len() >= 2 {
                        bestmove = parts[1].to_string();
                    }
                    return Some(bestmove);
                }
            }
            Err(_) => break,
        }
        thread::sleep(Duration::from_millis(10));
    }

    if bestmove.is_empty() { None } else { Some(bestmove) }
}

fn read_eval(reader: &mut BufReader<TcpStream>) -> String {
    let mut line = String::new();
    let mut last_final = String::new();

    for _ in 0..100 {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                if line.trim().starts_with("final") {
                    last_final = line.trim().to_string();
                }
            }
            Err(_) => break,
        }
    }

    if let Some(eval) = last_final.split_whitespace().nth(1) {
        eval.to_string()
    } else {
        "0".to_string()
    }
}

fn read_analysis(reader: &mut BufReader<TcpStream>, timeout_ms: u64) -> String {
    let start = std::time::Instant::now();
    let mut line = String::new();

    while start.elapsed().as_millis() < timeout_ms {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                if line.trim().starts_with("bestmove") && line.contains("q") {
                    let parts: Vec<&str> = line.trim().split_whitespace().collect();
                    if parts.len() >= 2 {
                        return parts[1].to_string();
                    }
                }
                if line.trim() == "readyok" {
                    return String::new();
                }
            }
            Err(_) => break,
        }
    }

    String::new()
}

fn apply_move(fen: &str, mv: &str) -> String {
    // Simple placeholder - in reality would use the engine's position
    // For now, return the same FEN (this is just for illustration)
    fen.to_string()
}

fn is_game_over(fen: &str) -> bool {
    fen.contains(" K ") || fen.contains(" k ")
}
