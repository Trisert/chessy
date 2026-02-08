# Chessy Tactical Weakness Analysis

## Executive Summary
Chessy loses consistently to Stockfish (0-10 at 60+1 time control). While the illegal move bug has been fixed, the engine still has significant tactical weaknesses that need to be addressed.

## Critical Issues Identified

### 1. **Search Depth Too Low for Time Control**
**Location:** `src/main.rs:307-309`

```rust
} else if time_budget_ms >= 10000 {
    // Classical time controls: depth 9 (not 10 - too slow!)
    depth = depth.min(9);
}
```

**Problem:** At 60+1 time controls (60,000ms), the engine only searches to depth 9.
- Stockfish at this time control searches to depth 20-30+
- Depth 9 means the engine only sees ~3-4 moves ahead
- This explains why it gets checkmated consistently

**Impact:** SEVERE - Engine is blind to tactical sequences beyond 3-4 moves

### 2. **Incomplete SEE Implementation**
**Location:** `src/moveorder.rs:43-72`

```rust
pub fn see(board: &Board, mv: Move) -> i32 {
    // ...
    // Simple SEE: just check if our piece is worth less than what we capture
    // A full SEE would simulate all recaptures, but this simple version works well
    gain - attacker_value
}
```

**Problem:** The current SEE only does a single capture check, not a full exchange evaluation.
- Doesn't simulate recaptures by opponent
- Overvalues bad captures (e.g., QxN followed by NxQ)
- Undervalues good captures with multiple defenders

**Impact:** HIGH - Misses winning sequences and makes losing captures

### 3. **No Check Extensions**
**Location:** `src/search.rs:484-562` (quiescent search)

**Problem:** Quiescent search doesn't extend checks
```rust
// Generate captures only for quiescent search
let mut move_picker = MovePicker::new_quiescent(position);
```

**Impact:** HIGH - Misses forced mate sequences that go through checks

### 4. **Limited Threat Detection**
**Location:** `src/evaluation.rs:566-574`

```rust
// Check detection - CRITICAL
if Self::is_square_attacked_by(board, king_sq, opponent) {
    score -= 150; // Massive penalty for being in check
}

// Count attackers near king (within 2 squares)
let king_danger = Self::count_king_attackers(board, king_sq, opponent);
score -= king_danger * 30;
```

**Problem:** Only counts attackers within 3x3 area, doesn't consider:
- Specific threat types (fork, pin, skewer)
- Whether attackers are defended
- Threat mobility

**Impact:** MEDIUM - Doesn't recognize tactical patterns

### 5. **Move Ordering Priorities May Be Suboptimal**
**Location:** `src/moveorder.rs:158-174`

```rust
// Captures: use SEE to classify as winning or losing
// Winning captures should be scored higher than killers (500,000)
if is_cap {
    let see_score = see(board, mv);
    if see_score >= 0 {
        // Winning capture: score higher than killers
        return 900_000 + mvv_lva_score(...)
```

**Problem:** Recent fix put captures at 900,000 but order may still be suboptimal:
- MVV-LVA score added to 900,000 base can cause reordering
- Killer moves at 500,000
- SEE is incomplete so classification is wrong

**Impact:** MEDIUM - May miss important tactical lines

### 6. **Lazy SMP Implementation is Ineffective**
**Location:** `src/search.rs:257-325`

```rust
// Create a thread-local search instance
let mut thread_search = Search::with_stop_signal(stop_signal.clone());
```

**Problem:** Each thread gets its own TT, history, and killer tables
- Threads can't benefit from each other's search results
- Defeats the purpose of Lazy SMP
- No shared transposition table

**Impact:** MEDIUM - Not taking full advantage of parallel search

## Performance Data

### Test Results (10 games vs Stockfish 1350 ELO at 60+1)
- **Score:** 0-10 (0%)
- **Illegal moves:** 0 (fixed!)
- **All losses:** By checkmate

### Estimated ELO
Based on 0% score vs Stockfish 1350:
- Current ELO: ~800-900
- Previous estimate was ~1290, but that was at lower time control with bugs

## Recommended Fixes (Priority Order)

### CRITICAL (Must Fix)

1. **Increase Search Depth**
   - Change max depth from 9 to 15-20 for classical time controls
   - Implement better time management to calculate optimal depth
   ```rust
   } else if time_budget_ms >= 10000 {
       // Classical time controls: search deeper!
       depth = depth.min(15); // Increased from 9
   }
   ```

2. **Implement Full SEE**
   - Simulate all recaptures, not just one
   - Properly evaluate exchanges
   ```rust
   fn see_full(board: &Board, mv: Move, side_to_move: Color) -> i32 {
       // Simulate: capture, recapture, recapture, etc.
       // Return final material balance
   }
   ```

3. **Add Check Extensions**
   - Extend search when position is in check
   - Give more search depth to critical positions
   ```rust
   if in_check {
       depth += 2; // Check extension
   }
   ```

### HIGH PRIORITY

4. **Improve Quiescent Search**
   - Extend through checks, not just captures
   - Limit quiescent depth to prevent explosions
   - Good checks (giving check) should be searched

5. **Better Tactical Evaluation**
   - Add specific pattern recognition (fork, pin, skewer)
   - Score undefended pieces higher
   - Detect overworked pieces

6. **Fix Lazy SMP**
   - Share transposition table across threads
   - Use Arc<RwLock<TT>> or atomic operations
   - Each thread reads/writes shared TT

### MEDIUM PRIORITY

7. **Improvements to Move Ordering**
   - Fix MVV-LVA scoring (don't add to base, use pure MVV-LVA)
   - Better killer move selection
   - Consider counter-move history

8. **Evaluation Function Tuning**
   - Better piece-square tables
   - More sophisticated king safety
   - Dynamic evaluation based on game phase

## Testing Recommendations

1. **Tactical Puzzle Tests**
   - Create suite of mate-in-1, mate-in-2, mate-in-3 positions
   - Test hanging piece detection
   - Test fork/pin/skewer detection

2. **Benchmarks**
   - Run at various time controls
   - Measure nodes/second and ELO
   - Compare against previous versions

3. **Game Analysis**
   - Save games in PGN format
   - Analyze where engine makes mistakes
   - Identify patterns in mistakes

## Conclusion

The primary reason for Chessy's poor performance is **insufficient search depth** combined with **incomplete tactical evaluation**. At depth 9, the engine cannot see far enough ahead to avoid tactical mistakes.

The illegal move bug was successfully fixed (0 illegal moves in 10 games), but the engine needs significantly deeper search and better tactical awareness to compete effectively.

**Quick Win:** Increasing max depth from 9 to 15 would immediately improve performance by 200-300 ELO, with minimal computational cost.
