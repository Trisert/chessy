use chessy::r#move::{Move, PromotionType};

fn main() {
    let queen_promo = Move::promotion(49, 57, PromotionType::Queen);
    
    println!("Raw value: {:016b} ({})", queen_promo.0, queen_promo.0);
    println!("Bit 15: {}", (queen_promo.0 >> 15) & 1);
    println!("Bits 12-14: {}", (queen_promo.0 >> 12) & 7);
    println!("is_promotion check: {}", ((queen_promo.0 >> 15) & 1) == 1 && ((queen_promo.0 >> 12) & 7) >= 4 && ((queen_promo.0 >> 12) & 7) <= 7);
    println!("is_promotion(): {}", queen_promo.is_promotion());
    println!("promotion_type(): {:?}", queen_promo.promotion_type());
}
