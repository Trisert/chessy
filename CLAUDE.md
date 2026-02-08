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

# Or use the play_match.sh script
chmod +x play_match.sh
./play_match.sh  # Chessy vs Stockfish with 2s per move
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
   - **Move ordering** prioritizes: TT move → winning captures → killer moves → history-ordered quiets
   - Lazy SMP using rayon for parallel root move search

5. **Evaluation** (`evaluation.rs`)
   - Static position scoring using material + piece-square tables (PSTs)
   - SIMD (AVX2) optimizations for material counting on x86_64
   - Separate PST for each piece type

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
- **SEE** (Static Exchange Evaluation) to classify winning/losing captures
- **Killer moves**: Two best quiet moves per ply that caused beta cutoffs
- **History table**: Butterfly counters for quiet moves that caused cutoffs

**Search Flow**:
1. Check TT cutoff (if entry depth ≥ current depth)
2. Generate and score moves using heuristics
3. At depth 0: call quiescent search instead of static eval
4. Update killers/history on beta cutoff

### Important Constraints

- **Time management**: Uses adaptive depth capping based on time budget (depth 1-9 scaled by time)
- **Thread-safety**: `Search` uses `RefCell`/`Cell` (not `Sync`) - each thread creates its own instance in Lazy SMP
- **TT size**: 256 MB default, power-of-2 cluster count for fast indexing
- **Move validation**: Always validate TT moves before using (may be stale)

## Module Dependencies

- `search.rs` depends on: `moveorder`, `evaluation`, `transposition`, `movegen`, `position`
- `moveorder.rs` depends on: `movegen`, `board`, `piece`
- `evaluation.rs` depends on: `board`, `piece`, `bitboard`
- `movegen.rs` depends on: `board`, `magic`, `position`, `movelist`
- `position.rs` depends on: `board`, `r#move`, `zobrist`

## Testing Notes

**⚠️ Known Test Failures:**
Pre-existing test failures exist in: `bitboard`, `board`, `evaluation`, `magic`, `piece`, `utils` modules.

**Risk Assessment:**
Relying solely on "play testing" without unit test validation can hide regressions and make debugging more difficult. The lack of unit test coverage creates technical debt that increases maintenance costs.

**Remediation Plan:**
1. Run full test suite to capture all failing tests with specific error messages
2. Create tracked GitHub issues for each failing module with:
   - List of failing test names
   - Root cause analysis
   - Assigned owner
   - Target timeline for resolution
3. Priority order: core modules (board, position, movegen) → evaluation → utilities
4. Restore unit-test coverage and enable CI gating to prevent future regressions

**Current Workaround:**
Engine functionality is verified through play testing against Stockfish. While not ideal, this provides real-world validation of chess logic correctness.

## Performance Considerations

- **Release builds** use LTO, single codegen unit, and panic=abort for maximum speed
- **SIMD features**: AVX2 for material counting (fallback to SSE4.2 available)
- **Allocation-free hot paths**: `MoveList` uses fixed-size array on stack
- **Magic numbers** are pre-initialized at startup for sliding piece attacks
