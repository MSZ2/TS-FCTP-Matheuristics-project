#!/bin/bash

. ../venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_DIR="$SCRIPT_DIR/../generated_instances"
SCRIPT="$SCRIPT_DIR/main.py"
OUT_DIR="$SCRIPT_DIR/reports_improved_medium"

MODE="multi"
RUNS=10
ITERS=1000

# main.py writes to ./reports (relative to cwd) — run from a dedicated
# output dir so this doesn't overwrite GRASP_part_parallel/reports/.
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

for instance in "$INSTANCE_DIR"/medium_{19,20}.txt; do
  [ -f "$instance" ] || continue

  echo "Running: $instance (iters=$ITERS, runs=$RUNS)"

  python "$SCRIPT" "$instance" \
    --mode "$MODE" \
    --iters "$ITERS" \
    --runs "$RUNS"
done
