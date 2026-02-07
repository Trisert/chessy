#!/bin/bash

# Allow overriding engine paths via environment variables
ENGINE1="${ENGINE1:-./target/release/chessy}"
ENGINE2="${ENGINE2:-stockfish}"

moves=""
game_over=0
max_moves=50

# Cleanup function for trap
cleanup() {
  if [ -n "$PID1" ]; then
    kill $PID1 2>/dev/null
  fi
  if [ -n "$PID2" ]; then
    kill $PID2 2>/dev/null
  fi
  rm -f /tmp/engine1_in /tmp/engine1_out /tmp/engine2_in /tmp/engine2_out
}

# Set trap for cleanup on exit or interruption
trap cleanup EXIT INT TERM

echo "Chessy (White) vs Stockfish (Black)"
echo "===================================="

# Use temporary log files instead of FIFOs for proper reading
rm -f /tmp/engine1_out.log /tmp/engine2_out.log

$ENGINE1 < /tmp/engine1_in > /tmp/engine1_out.log 2>&1 &
PID1=$!
$ENGINE2 < /tmp/engine2_in > /tmp/engine2_out.log 2>&1 &
PID2=$!

sleep 0.5

echo "uci" > /tmp/engine1_in
echo "uci" > /tmp/engine2_in
sleep 0.5

echo "isready" > /tmp/engine1_in
echo "isready" > /tmp/engine2_in
sleep 0.2

echo "position startpos" > /tmp/engine1_in
echo "position startpos" > /tmp/engine2_in

for i in $(seq 1 $max_moves); do
  if [ $((i % 2)) -eq 1 ]; then
    echo "go movetime 2000" > /tmp/engine1_in
    sleep 2.5
    best=$(grep "bestmove" /tmp/engine1_out.log | tail -1 | awk '{print $2}')
    if [ -z "$best" ]; then
      echo "Chessy failed to move"
      break
    fi
    echo "White (Chessy): $best"
    moves="$moves $best"
    echo "position startpos moves$moves" > /tmp/engine2_in
  else
    echo "go movetime 2000" > /tmp/engine2_in
    sleep 2.5
    best=$(grep "bestmove" /tmp/engine2_out.log | tail -1 | awk '{print $2}')
    if [ -z "$best" ]; then
      echo "Stockfish failed to move"
      break
    fi
    echo "Black (Stockfish): $best"
    moves="$moves $best"
    echo "position startpos moves$moves" > /tmp/engine1_in
  fi

  # Only "(none)" indicates checkmate/stalemate (no legal moves)
  # Promotion moves like "e7e8q" are 5 characters but are valid
  if [ "$best" = "(none)" ]; then
    echo "Game ended"
    break
  fi
done

# Cleanup is handled by trap
echo "Final moves: $moves"
