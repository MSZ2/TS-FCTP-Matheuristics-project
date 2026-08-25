#!/bin/bash
#
# small (already running) -> medium -> large, unattended.
#
# Waits on the small run's PID rather than on `pgrep -f <script>`: a
# pattern match also matches this script's own command line, so such a
# loop never exits (it cost 40 minutes and one immortal process on
# 2026-08-11).

SMALL_PID="$1"
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -n "$SMALL_PID" ]; then
  echo "[$(date '+%F %T')] waiting for small suite (PID $SMALL_PID)"
  while kill -0 "$SMALL_PID" 2>/dev/null; do sleep 60; done
  echo "[$(date '+%F %T')] small suite finished"
fi

echo "[$(date '+%F %T')] starting medium (10 runs x 1000 iters, ~5.4 days)"
bash main_runner_medium.sh
echo "[$(date '+%F %T')] medium finished"

echo "[$(date '+%F %T')] starting large (1 run x 100 iters, ~2 days)"
bash main_runner_large.sh
echo "[$(date '+%F %T')] large finished"

echo "[$(date '+%F %T')] CHAIN COMPLETE"
