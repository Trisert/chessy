use chessy::r#move::{Move, PromotionType};

fn main() {
    // Create a regular move b7b8
    let regular = Move::new(49, 57); // b7 to b8
    
    // Create promotion moves b7b8q, b7b8r, etc.
    let queen_promo = Move::promotion(49, 57, PromotionType::Queen);
    let rook_promo = Move::promotion(49, 57, PromotionType::Rook);
    
    println!("Regular move: {} (raw: {:016b})", regular, regular.0);
    println!("Queen promo:  {} (raw: {:016b})", queen_promo, queen_promo.0);
    println!("Rook promo:   {} (raw: {:016b})", rook_promo, rook_promo.0);
    
    println!("\nRegular == Queen promo: {}", regular == queen_promo);
    println!("Regular.from() == Queen promo.from(): {}", regular.from() == queen_promo.from());
    println!("Regular.to() == Queen promo.to(): {}", regular.to() == queen_promo.to());
    println!("Regular.is_promotion(): {}", regular.is_promotion());
    println!("Queen promo.is_promotion(): {}", queen_promo.is_promotion());
}
