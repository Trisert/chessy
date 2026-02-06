// Simple benchmark for profiling
use chessy::movegen::MoveGen;
use chessy::piece::Color;
use chessy::position::Position;
use chessy::search::Search;

fn main() {
    // Create starting position
    let mut position = Position::from_start();

    // Make a few moves to get a more complex position
    let moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"];
    for mv_str in moves {
        use chessy::r#move::Move;
        use chessy::utils::square_from_string;

        let from = square_from_string(&mv_str[0..2]).unwrap();
        let to = square_from_string(&mv_str[2..4]).unwrap();
        let mv = Move::new(from, to);
        position.make_move(mv);
    }

    println!("Starting benchmark search...");
    println!("Position FEN: {}", position.to_fen());

    // Run search with profiling
    let mut search = Search::new();
    let depth = 8; // Moderate depth for better profiling

    let start = std::time::Instant::now();
    let (best_move, score) = search.search(&mut position, depth, None);
    let elapsed = start.elapsed();

    println!("Best move: {}, Score: {}", best_move, score);
    println!("Time: {:?}", elapsed);
    println!("Nodes: {}", search.nodes());
}
