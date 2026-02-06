# Chessy

A performant chess engine written in Rust, implementing bitboard representation with magic move generation, alpha-beta search with transposition table, and full UCI protocol support.

## Features

- **Bitboard Representation**: Efficient 64-bit board representation using magic bitboards for sliding pieces (rooks, bishops, queens)
- **Move Generation**: Fast pseudo-legal and legal move generation with bulk generation
- **Search Algorithm**: Alpha-beta search with:
  - Transposition table for memoization
  - Iterative deepening
  - Adaptive depth based on time controls
  - Move ordering heuristics
- **UCI Protocol**: Full Universal Chess Interface support for integration with chess GUIs
- **Position Handling**: Full FEN parsing and generation, castling rights, en passant squares
- **Perft Testing**: Built-in perft testing for move generation validation

## Building

```bash
cargo build --release
```

The binary will be at `target/release/chessy`.

## Usage

### UCI Mode

Run the engine in UCI mode for use with chess GUIs:

```bash
./target/release/chessy
```

The engine supports the following UCI commands:
- `uci` - Identify the engine
- `isready` - Check if engine is ready
- `position [startpos|fen <fen>] [moves <moves>...]` - Set up position
- `go [depth <n>] [nodes <n>] [movetime <ms>] [wtime <ms>] [btime <ms>] [movestogo <n>]` - Start search
- `stop` - Stop ongoing search
- `quit` - Exit

### Perft Testing

Test move generation performance:

```bash
./target/release/chessy --perft <depth>
```

Default depth is 1 if not specified.

### Illegal Move Testing

Debug illegal move detection:

```bash
./target/release/chessy --test-illegal
```

## Architecture

### Core Modules

| Module | Purpose |
|--------|---------|
| `bitboard.rs` | Bitboard utilities and operations |
| `board.rs` | Board state management |
| `movegen.rs` | Move generation with magic bitboards |
| `move.rs` | Move representation and encoding |
| `position.rs` | Position state with full game state |
| `search.rs` | Alpha-beta search with transposition table |
| `evaluation.rs` | Static position evaluation |
| `zobrist.rs` | Zobrist hashing for position identification |
| `transposition.rs` | Transposition table implementation |
| `castling.rs` | Castling move generation and validation |
| `magic.rs` | Magic number initialization for sliding pieces |
| `utils.rs` | Square conversion utilities |

### Key Data Structures

- **Bitboard**: `u64` representing board occupancy
- **Square**: 0-63 representing board squares
- **Move**: Encoded move with from/to squares and flags
- **Position**: Full game state including board, side to move, castling rights, etc.

## Performance

The engine includes optimizations for:
- Magic bitboard move generation for sliding pieces
- Bulk move generation to reduce allocations
- Transposition table for search memoization
- Adaptive depth for bullet time controls
- Move legality checking before outputting moves

## License

MIT
