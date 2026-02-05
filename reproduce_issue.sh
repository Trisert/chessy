#!/bin/bash
set -euo pipefail

CHESSY_BIN="${CHESSY_BIN:-./target/debug/chessy}"

if [[ ! -x "$CHESSY_BIN" ]]; then
    echo "Error: $CHESSY_BIN not found or not executable. Run 'cargo build' first."
    exit 1
fi

trap 'rm -f test_e1d1.uci' EXIT

cat > test_e1d1.uci << 'UCI'
uci
ucinewgame
position startpos moves b2b4 e7e5 d2d4 e5d4 g2g3 f8b4 c2c3 d4c3 d1b3 b8c6 e2e3 d8e7 g1e2 c6e5 b7b6 b1a3 c2c4
go depth 5
quit
UCI

echo "=== Testing position that leads to e1d1 illegal move ==="
"$CHESSY_BIN" < test_e1d1.uci 2>&1 | grep -E "(bestmove|ERROR|CORRUPTION|illegal|WARNING)" | head -20 || true
