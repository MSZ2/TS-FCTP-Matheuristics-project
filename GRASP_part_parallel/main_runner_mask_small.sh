#!/bin/bash

. ../venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_DIR="$SCRIPT_DIR/../generated_instances"
SCRIPT="$SCRIPT_DIR/main.py"
OUT_DIR="$SCRIPT_DIR/reports_mask_small"

MODE="multi"
RUNS=10
ITERS=1000

# small_1..small_10 only: those are the instances CPLEX solved to
# OPTIMAL, so the gap column means what it says, and they are the ones
# every comparison table in the writeup uses.
INSTANCES=$(seq 1 10)

# main.py writes to ./reports (relative to cwd) — run from a dedicated
# output dir so this doesn't overwrite GRASP_part_parallel/reports/.
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

for n in $INSTANCES; do
  instance="$INSTANCE_DIR/small_$n.txt"
  [ -f "$instance" ] || continue

  echo "=== Running: small_$n (iters=$ITERS, runs=$RUNS)"

  python "$SCRIPT" "$instance" \
    --mode "$MODE" \
    --iters "$ITERS" \
    --runs "$RUNS"
done

echo "=== ALL DONE"
