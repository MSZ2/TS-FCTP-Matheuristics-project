#!/bin/bash

. ../venv/bin/activate

INSTANCE_DIR="../generated_instances"
SCRIPT="main.py"

MODE="multi"
ITERS=100
RUNS=1

for size in medium large; do
  for instance in "$INSTANCE_DIR"/${size}*; do
    [ -f "$instance" ] || continue

    echo "Running: $instance"

    python "$SCRIPT" "$instance" \
      --mode "$MODE" \
      --iters "$ITERS" \
      --runs "$RUNS"
  done
done
