use chessy::position::Position;
use chessy::movegen::MoveGen;
use chessy::piece::Color;

fn main() {
    // Test position: white pawn on g7, ready to promote
    let fen = "8/6P1/8/8/8/8/8/4K2k w - - 0 1";
    let pos = Position::from_fen(fen).unwrap();
    
    println!("Position:\n{}", pos.board.to_string());
    println!("Side to move: {:?}", pos.state.side_to_move);
    
    let moves = MoveGen::generate_legal_moves(&pos.board, Color::White);
    println!("\nLegal moves ({}):", moves.len());
    for i in 0..moves.len() {
        let mv = moves.get(i);
        println!("  {} - promotion: {:?}", mv, mv.promotion_type());
    }
}
