#!/bin/bash
# Simple UCI test to trigger search for profiling

echo "uci"
sleep 0.1
echo "isready"
sleep 0.1
echo "position startpos moves e2e4 e7e5 g1f3 b8c6"
sleep 0.1
echo "go depth 8"
sleep 5
echo "quit"
