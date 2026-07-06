#!/bin/bash

. ../venv/bin/activate

INSTANCE_DIR="../generated_instances"
SCRIPT="main.py"

MODE="multi"
RUNS=3

declare -A ITERS_BY_SIZE=( [small]=500 [medium]=200 [large]=100 )

for size in small medium large; do
  ITERS="${ITERS_BY_SIZE[$size]}"

  for instance in "$INSTANCE_DIR"/${size}*; do
    [ -f "$instance" ] || continue

    echo "Running: $instance (iters=$ITERS, runs=$RUNS)"

    python "$SCRIPT" "$instance" \
      --mode "$MODE" \
      --iters "$ITERS" \
      --runs "$RUNS"
  done
done
