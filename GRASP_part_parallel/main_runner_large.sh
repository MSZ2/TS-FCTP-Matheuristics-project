#!/bin/bash
#
# The whole large suite, 3 runs x 100 iterations (~6 days).
#
# 100 iterations matches the July run this is meant to be compared
# against: a single pass over the 20 large instances took 21 h then and
# the search is ~2.8x more expensive per iteration now. Three runs is
# the most that fits -- it buys a sigma (weak at n=3, but far better
# than a single run); 10 runs x 1000 iterations would be ~6 months.
#
# CPLEX left every large instance at FEASIBLE after its 2 h limit.
#
# Writes to ./reports, same as the other runners.

. ../venv/bin/activate

INSTANCE_DIR="../generated_instances"
SCRIPT="main.py"
MODE="multi"
RUNS=3
ITERS=100

FROM="${1:-1}"
TO="${2:-20}"

for n in $(seq "$FROM" "$TO"); do
  instance="$INSTANCE_DIR/large_$n.txt"
  [ -f "$instance" ] || continue
  echo "=== [$(date '+%F %T')] Running: large_$n (iters=$ITERS, runs=$RUNS)"
  python "$SCRIPT" "$instance" --mode "$MODE" --iters "$ITERS" --runs "$RUNS"
done
echo "=== [$(date '+%F %T')] LARGE DONE"
