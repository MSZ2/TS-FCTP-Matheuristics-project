#!/bin/bash
#
# The whole medium suite, 10 runs x 1000 iterations (~5.4 days).
#
# Same recipe as main_runner_small.sh, so the two tables are directly
# comparable and both carry a sigma.
#
# CPLEX left every medium instance at FEASIBLE after its 2 h limit, so
# the reference is its best feasible solution, not a proven optimum.
#
# Writes to ./reports, same as the other runners.

. ../venv/bin/activate

INSTANCE_DIR="../generated_instances"
SCRIPT="main.py"
MODE="multi"
RUNS=10
ITERS=1000

FROM="${1:-1}"
TO="${2:-20}"

for n in $(seq "$FROM" "$TO"); do
  instance="$INSTANCE_DIR/medium_$n.txt"
  [ -f "$instance" ] || continue
  echo "=== Running: medium_$n (iters=$ITERS, runs=$RUNS)"
  python "$SCRIPT" "$instance" --mode "$MODE" --iters "$ITERS" --runs "$RUNS"
done
echo "=== MEDIUM DONE"
