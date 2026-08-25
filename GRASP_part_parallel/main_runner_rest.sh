#!/bin/bash
#
# The tail of the medium suite (15-20) at 10 runs x 1000 iterations,
# then the whole large suite via main_runner_large.sh (3 runs x 100).
#
# Runtime: medium 15-20 is ~1.6 days, large is ~6 days, so ~8 days
# total. Each instance writes its report as soon as it finishes, so
# stopping partway still leaves usable progress.
#
# CPLEX left every medium and large instance at FEASIBLE after its
# 2 h limit, so the reference is its best feasible solution, not a
# proven optimum.
#
# Writes to ./reports, same as the other runners.

cd "$(dirname "${BASH_SOURCE[0]}")"
. ../venv/bin/activate

INSTANCE_DIR="../generated_instances"
SCRIPT="main.py"
MODE="multi"

echo "=== [$(date '+%F %T')] medium 15-20 (iters=1000, runs=10)"
for n in $(seq 15 20); do
  instance="$INSTANCE_DIR/medium_$n.txt"
  [ -f "$instance" ] || continue
  echo "=== [$(date '+%F %T')] Running: medium_$n (iters=1000, runs=10)"
  python "$SCRIPT" "$instance" --mode "$MODE" --iters 1000 --runs 10
done
echo "=== [$(date '+%F %T')] MEDIUM DONE"

bash main_runner_large.sh
echo "=== [$(date '+%F %T')] ALL DONE"
