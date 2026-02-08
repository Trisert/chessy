# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Build and Run
```bash
# Debug build
cargo build

# Release build (optimized for performance)
cargo build --release

# Run engine in UCI mode
cargo run --bin chessy

# Run with specific depth for testing
echo "go depth 10" | cargo run --bin chessy
```

### Testing
```bash
# Run all library tests
cargo test --lib

# Run a specific test module
cargo test --lib bitboard::tests
cargo test --lib evaluation::tests

# Run perft testing (move generation validation)
./target/release/chessy --perft <depth>

# Test illegal move detection
./target/release/chessy --test-illegal
```

### Engine Testing Against Stockfish
```bash
# Play match against Stockfish (uses cutechess-cli if available)
cutechess-cli -engine name=Chessy cmd=./target/release/chessy -engine name=Stockfish cmd=stockfish -each tc=60+1

# Match with reduced ELO for testing
cutechess-cli -engine name=Chessy cmd=./target/release/chessy -engine name=Stockfish cmd=stockfish option.UCI_LimitStrength=true option.UCI_Eho -each tc=5+0.3 -rounds 10
```

### Linting and Formatting
```bash
# Check code
cargo clippy

# Format code
cargo fmt
```

## Architecture Overview

Chessy is a chess engine using **bitboard representation** with **alpha-beta search** and **UCI protocol** support.

### Core Architecture Layers

1. **Representation Layer** (`bitboard.rs`, `board.rs`, `piece.rs`, `utils.rs`)
   - 64-bit integers represent board state (1 bit per square)
   - Separate bitboards per piece type and color
   - `Board` maintains all piece bitboards and castling/EP state

2. **Move Generation** (`movegen.rs`, `castling.rs`, `magic.rs`, `movelist.rs`)
   - **Magic bitboards** for sliding pieces (bishop/rook/queen attacks)
   - Bulk move generation into `MoveList` (stack-allocated, fixed capacity)
   - Separate pseudo-legal and legal move generation
   - Castling has special validation in `castling.rs`

3. **Position Management** (`position.rs`, `r#move.rs`, `zobrist.rs`)
   - `Position` wraps `Board` with game state (side to move, castling rights, EP square, hash)
   - `make_move()` / `undo_move()` for state updates
   - Zobrist hashing for position identification (TT keys)

4. **Search** (`search.rs`, `transposition.rs`, `moveorder.rs`)
   - Alpha-beta pruning with iterative deepening
   - **Transposition table** (TT) caches search results with depth/bounds
   - **Quiescent search** extends through captures at depth 0 to prevent horizon effect
   - **Move ordering** prioritizes: TT move → countermove → winning captures → killer moves → history-ordered quiets
   - Lazy SMP using rayon for parallel root move search

5. **Evaluation** (`evaluation.rs`)
   - Static position scoring using material + piece-square tables (PSTs)
   - SIMD (AVX2) optimizations for material counting on x86_64
   - Separate PST for each piece type
   - **Tempo bonus**, **pawn structure**, **king safety**, **endgame evaluation**

### Key Design Patterns

**Move Encoding**: Moves are packed into 16 bits:
- Bits 0-5: from square, 6-11: to square
- Bits 12-14: promotion type (knight=0, bishop=1, rook=2, queen=3) or move flags
- Bit 15: special move flag (castling, en passant)

**Transposition Table Entry**:
- Stores: hash (64-bit), score (i32), best move, depth, flag (exact/lower/upper), age
- Three entries per cluster for cache efficiency
- Replacement strategy prefers empty → old generation → shallow depth

**Move Ordering Heuristics**:
- **MVV-LVA** (Most Valuable Victim - Least Valuable Attacker) for captures
- **Full SEE** (Static Exchange Evaluation) with recapture simulation
- **Countermove heuristic**: remembers best response to opponent moves
- **Killer moves**: Two best quiet moves per ply that caused beta cutoffs
- **History table**: Butterfly counters for quiet moves that caused cutoffs

**Search Flow**:
1. Check TT cutoff (if entry depth ≥ current depth)
2. Apply pruning: ProbCut → Null Move → RFP → Futility → Razoring
3. Generate and score moves using heuristics
4. At depth 0: call quiescent search instead of static eval
5. Update killers/history/countermoves on beta cutoff

### Search Optimizations

**Pruning Techniques**:
- **ProbCut**: Reduced-depth search when static eval >> beta
- **Null Move Pruning**: Give opponent free move, verify to avoid zugzwang
- **Reverse Futility Pruning (RFP)**: Prune if eval - margin >= beta
- **Extended Futility Pruning**: Depth-dependent margins at depths 1-3
- **Razoring**: Skip quiet moves at low depths if eval is hopeless
- **Late Move Pruning (LMP)**: Skip very late quiet moves with poor history
- **Singular Extensions**: Extend moves significantly better than alternatives

**Extensions**:
- **Check extension**: +1 ply when in check (max depth 12)
- **Singular extension**: +1 ply when TT move is >> alternatives (100cp margin)

**Move Reduction**:
- **LMR**: History-based late move reduction
- Only reduces moves with poor history scores

**Aspiration Windows**:
- Initial 15cp window, depth-dependent sizing
- Maximum 4 iterations with fallback to full window

### Evaluation Components

**Weighted Terms**:
- **King safety**: 1.5x weight (increased)
- **Pawn structure**: 1.33x weight (isolated, doubled, passed pawns)
- **Center control**: 1.25x weight
- **Tempo**: 20cp bonus for side to move
- **Piece activity**: 0.75x weight (mobility)
- **Threat detection**: 25cp penalty / 15cp bonus
- **Development**: Opening piece placement bonuses
- **Endgame**: Opposition, connected rooks, advanced pawns
- **Draw contempt**: -10cp for zero-score positions

**Early Exits**:
- Skip detailed eval if material difference > 2000cp
- Skip detailed eval if king safety difference > 500cp

### Time Management

**Adaptive Allocation**:
- **Bullet** (<2s): 40% allocation, 500ms buffer
- **Blitz** (<10s): 70% allocation, 400ms buffer
- **Classical** (≥10s): 80% allocation, 1000ms buffer
- Search cutoff at 85% of time limit

**Depth Limits**:
- Maximum search depth: 15 plies
- Quiescent depth limit: 10 plies
- Adaptive depth based on time budget (1-9 in normal play)

### Important Constraints

- **Thread-safety**: `Search` uses `RefCell`/`Cell` (not `Sync`) - each thread creates its own instance in Lazy SMP
- **TT size**: 256 MB default, power-of-2 cluster count for fast indexing
- **Move validation**: Always validate TT moves before using (may be stale)
- **Stack depth**: Maximum 100 ply recursion depth to prevent overflow

## Module Dependencies

- `search.rs` depends on: `moveorder`, `evaluation`, `transposition`, `movegen`, `position`
- `moveorder.rs` depends on: `movegen`, `board`, `piece`, `bitboard`
- `evaluation.rs` depends on: `board`, `piece`, `bitboard`
- `movegen.rs` depends on: `board`, `magic`, `position`, `movelist`
- `position.rs` depends on: `board`, `r#move`, `zobrist`

## Testing Status

**All 68 unit tests passing**

Test categories:
- Bitboard operations (8 tests)
- Board operations (5 tests)
- Castling (4 tests)
- Evaluation (2 tests)
- Move generation (12 tests)
- Move order (3 tests)
- Piece operations (5 tests)
- Move encoding (10 tests)
- Position management (1 test)
- Transposition table (5 tests)
- Utilities (6 tests)
- Magic attacks (7 tests)
- Illegal move detection (2 tests)

**Play Testing**:
- Verified against Stockfish at 1320 ELO
- 0 time forfeits in extended testing
- Playing complete games (all losses are checkmates)

## Performance Considerations

- **Release builds** use LTO, single codegen unit, and panic=abort for maximum speed
- **SIMD features**: AVX2 for material counting (fallback to SSE4.2 available)
- **Allocation-free hot paths**: `MoveList` uses fixed-size array on stack
- **Magic numbers** are pre-initialized at startup for sliding piece attacks
- **Early exits** in evaluation for extreme positions reduce CPU usage
- **ProbCut** and **delta pruning** reduce quiescent search overhead
