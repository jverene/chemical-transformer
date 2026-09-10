#!/bin/bash
# Remote watchdog: stops the instance via the vastai CLI using the 2FA
# session key staged on the instance (console-API bearer tokens lack
# state-change privileges — vast requires a 2FA session).
# Usage: vast_watchdog.sh <INSTANCE_ID> <KEYFILE> <DEADLINE_EPOCH> <DONEFILE>
ID="$1"; KEYFILE="$2"; DEADLINE="$3"; DONEFILE="$4"
LOG=/workspace/self_stop.log
echo "$(date) watchdog armed: id=$ID deadline=$DEADLINE done=$DONEFILE" >> $LOG
command -v vastai >/dev/null 2>&1 || PATH="/usr/local/bin:$PATH"
while true; do
  if [ -n "$DONEFILE" ] && [ -f "$DONEFILE" ]; then
    echo "$(date) done marker found" >> $LOG; break
  fi
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "$(date) deadline hit" >> $LOG; break; }
  sleep 120
done
sleep 180   # grace for any external final pull
vastai --api-key "$(cat $KEYFILE)" stop instance $ID >> $LOG 2>&1
echo "$(date) stop attempted" >> $LOG
vastai --api-key "$(cat $KEYFILE)" --raw show instances >> $LOG 2>&1
echo "$(date) watchdog exit" >> $LOG
