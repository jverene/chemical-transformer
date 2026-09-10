#!/bin/bash
# Local billing watcher v2: stops the instance VIA THE AUTHENTICATED CLI when
# the sentinel file (written by the manifest itself) appears — immune to the
# stale MANIFEST_DONE hazard. The remote curl watchdog is dead: vast requires
# a 2FA session for state changes, so the stop must happen locally.
ID=${1:?instance id}
PORT=${2:?ssh port}
SENTINEL=${3:-P1B_DONE}
HOURS=${4:-8}
DEADLINE=$(($(date +%s) + HOURS*3600))
command -v vastai >/dev/null 2>&1 || PATH="$(pwd)/.venv/bin:$PATH"
SSHO="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20"
echo "$(date) stop-watcher v2 armed for $ID on sentinel=$SENTINEL (${HOURS}h deadline)"
while true; do
  DONE=$(ssh $SSHO -p $PORT root@45.135.56.10 "ls /workspace/$SENTINEL 2>/dev/null" 2>/dev/null)
  if [ -n "$DONE" ]; then
    echo "$(date) sentinel found — final results pull, then stop"
    rsync -az -e "ssh $SSHO -p $PORT" \
      root@45.135.56.10:/workspace/chemical-transformer/results-p1/ ./results-p1/ 2>/dev/null
    sleep 90   # let any other watcher do its final pull too
    vastai stop instance $ID && echo "$(date) INSTANCE STOPPED — billing ended"
    exit 0
  fi
  [ "$(date +%s)" -ge "$DEADLINE" ] && {
    echo "$(date) 8h deadline — stopping unconditionally"
    rsync -az -e "ssh $SSHO -p $PORT" \
      root@45.135.56.10:/workspace/chemical-transformer/results-p1/ ./results-p1/ 2>/dev/null
    vastai stop instance $ID && echo "$(date) INSTANCE STOPPED"; exit 0; }
  sleep 300
done
