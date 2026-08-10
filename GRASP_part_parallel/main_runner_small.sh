#!/bin/bash
#
# The whole small suite at full budget, 10 runs each.
#
# CPLEX solved 14 of the 20 to OPTIMAL (1-11, 13, 14, 15), so for those
# the gap column is a true optimality gap; for 12 and 16-20 it is a gap
# against CPLEX's best feasible solution after its 2 h limit, which is
# still a fair comparison -- just not proof of optimality.
#
# main_runner.sh covers small+medium+large at a much lower budget.
#
# Writes to ./reports, same as main_runner.sh.
#
# Pass a range to do part of the suite, e.g.  ./main_runner_small.sh 11 20

. ../venv/bin/activate

INSTANCE_DIR="../generated_instances"
SCRIPT="main.py"

MODE="multi"
RUNS=10
ITERS=1000

FROM="${1:-1}"
TO="${2:-20}"

for n in $(seq "$FROM" "$TO"); do
  instance="$INSTANCE_DIR/small_$n.txt"
  [ -f "$instance" ] || continue

  echo "=== Running: small_$n (iters=$ITERS, runs=$RUNS)"

  python "$SCRIPT" "$instance" \
    --mode "$MODE" \
    --iters "$ITERS" \
    --runs "$RUNS"
done

echo "=== ALL DONE"
