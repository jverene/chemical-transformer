#!/bin/bash
# Mirror results-nl/ + phase logs from instance 52311159 (seed-2 pair)
# every 15 min, bounded to 14h. Non-fatal per iteration; safe to re-run.
set -u
SSHO="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=25"
END=$(( $(date +%s) + 14*3600 ))
while [ "$(date +%s)" -lt "$END" ]; do
  if rsync -az --timeout=90 -e "ssh $SSHO -p 1766" \
      root@192.234.50.251:/workspace/chemical-transformer/results-nl/ ./results-nl/ 2>/dev/null; then
    echo "$(date '+%m-%d %H:%M') pulled results-nl/"
  fi
  for f in phase_d2_1.log phase_d2_2.log phase_d2_3.log ct.log; do
    scp -O -q $SSHO -P 1766 root@192.234.50.251:/workspace/$f ./mirror_52311159-$f 2>/dev/null
  done
  if ssh $SSHO -p 1766 root@192.234.50.251 \
      "ls /workspace/MANIFEST_DONE" >/dev/null 2>&1; then
    echo "$(date '+%m-%d %H:%M') MANIFEST_DONE found — final mirror in 2 min, exiting"
    sleep 120
    rsync -az --timeout=90 -e "ssh $SSHO -p 1766" \
      root@192.234.50.251:/workspace/chemical-transformer/results-nl/ ./results-nl/ 2>/dev/null
    for f in phase_d2_1.log phase_d2_2.log phase_d2_3.log; do
      scp -O -q $SSHO -P 1766 root@192.234.50.251:/workspace/$f ./mirror_52311159-$f 2>/dev/null
    done
    exit 0
  fi
  sleep 900
done
echo "$(date '+%m-%d %H:%M') 14h bound reached"
