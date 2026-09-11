#!/bin/bash
# Auto-pull + auto-push (neuromodulation-style): every 10 min, pull the
# running instance's results, commit and push to GitHub. Tolerates SSH
# flaps (retries), and exits cleanly once the instance is stopped, doing a
# final pull+push first. Git credentials come from the local machine.
ID=${1:?instance id}
DEADLINE=$(($(date +%s) + 20*3600))
command -v vastai >/dev/null 2>&1 || PATH="$(pwd)/.venv/bin:$PATH"
SSHO="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20"
echo "$(date) auto-push watcher armed for $ID"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  NURL=$(vastai ssh-url $ID 2>/dev/null)
  if [ -n "$NURL" ]; then
    NHP=${NURL#ssh://root@}; NH=${NHP%%:*}; NP=${NHP##*:}
    for t in 1 2 3; do
      rsync -az --timeout=60 --partial -e "ssh $SSHO -p $NP" \
        root@$NH:/workspace/chemical-transformer/results-p3/ ./results-p3/ 2>/dev/null && break
      sleep 45
    done
  fi
  git add results-p3 results-p1 2>/dev/null
  if ! git diff --cached --quiet 2>/dev/null; then
    git commit -q -m "data: 150m grid incremental (auto $(date +%m-%d\ %H:%M))" 2>/dev/null
    for t in 1 2 3; do git push -q origin main 2>/dev/null && { echo "$(date +%H:%M) pushed"; break; }; sleep 30; done
  fi
  S=$(vastai --raw show instances 2>/dev/null | python3 -c "
import json,sys
try: print(json.load(sys.stdin)[0]['actual_status'])
except Exception: print('unknown')" 2>/dev/null)
  if [ "$S" != "running" ]; then
    echo "$(date) instance $S — final pull + push, exiting"
    for t in 1 2 3 4 5; do
      rsync -az --timeout=60 --partial -e "ssh $SSHO -p $NP" \
        root@$NH:/workspace/chemical-transformer/results-p3/ ./results-p3/ 2>/dev/null && break
      sleep 60
    done
    git add results-p3 2>/dev/null
    git diff --cached --quiet 2>/dev/null || { git commit -q -m "data: 150m grid final (auto)"; for t in 1 2 3; do git push -q origin main 2>/dev/null && break; sleep 30; done; }
    exit 0
  fi
  sleep 600
done
echo "$(date) deadline reached"
