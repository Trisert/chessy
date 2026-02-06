#!/bin/bash

# Simulate a game and capture stderr to see validation errors
OUTPUT_FILE="game_debug.log"

echo "Starting simulated game..." > "$OUTPUT_FILE"
echo "==============================" >> "$OUTPUT_FILE"

(
  echo "uci"
  sleep 0.1
  echo "isready"
  sleep 0.1
  echo "position startpos"
  echo "go depth 5"
  sleep 3
  echo "position startpos moves e2e4"
  echo "d"
  sleep 0.1
  echo "go depth 5"
  sleep 3
  echo "quit"
) | ./target/release/chessy 2>&1 | tee -a "$OUTPUT_FILE"

echo "Done. Output saved to $OUTPUT_FILE"
cat "$OUTPUT_FILE" | grep -E "(ERROR|ILLEGAL|WARNING|Side to move)" | head -20
