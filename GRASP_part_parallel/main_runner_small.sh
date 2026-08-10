#!/bin/bash
#
# small_1..small_10 at full budget, 10 runs each.
#
# Those ten are the instances CPLEX solved to OPTIMAL, so their gap
# column means what it says; small_11..small_20 have no such reference.
# main_runner.sh covers the whole suite at a lower budget instead.
#
# Writes to ./reports, same as main_runner.sh.

. ../venv/bin/activate

INSTANCE_DIR="../generated_instances"
SCRIPT="main.py"

MODE="multi"
RUNS=10
ITERS=1000

for n in $(seq 1 10); do
  instance="$INSTANCE_DIR/small_$n.txt"
  [ -f "$instance" ] || continue

  echo "=== Running: small_$n (iters=$ITERS, runs=$RUNS)"

  python "$SCRIPT" "$instance" \
    --mode "$MODE" \
    --iters "$ITERS" \
    --runs "$RUNS"
done

echo "=== ALL DONE"
