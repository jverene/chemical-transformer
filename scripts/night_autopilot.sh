#!/bin/bash
# Overnight autopilot (multi-instance): waits for the instance's ssh, stages
# + launches the given manifest, MIRRORS partial results/logs every probe,
# pulls on completion, pushes to git, destroys the instance (verified pull)
# or stops it for salvage. Host unreachable >90 min -> destroy + exit 1.
# Usage: night_autopilot.sh <INSTANCE_ID> [MANIFEST] [LOGFILE] [DONE_MARKER]
logf() { echo "$(date +%H:%M:%S) $*" >> "${LOGFILE:-/tmp/autopilot.log}"; }
SSHO="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
DEADLINE=$(($(date +%s) + 20*3600))
command -v vastai >/dev/null 2>&1 || PATH="$(pwd)/.venv/bin:$PATH"
CURRENT="${1:?usage: night_autopilot.sh INSTANCE_ID [MANIFEST] [LOGFILE] [DONE_MARKER]}"
MANIFEST="${2:-manifests/nl_real.sh}"
LOGFILE="${3:-/tmp/autopilot.log}"
DONE_MARKER="${4:-NL_REAL_DONE}"
EXPECTED="${5:-}"   # globs (relative to results-nl) that must exist to destroy
UNREACH_SINCE=""
STAGED=0
logf "autopilot armed: instance $CURRENT manifest=$MANIFEST marker=$DONE_MARKER"

probe() { ssh $SSHO -o ConnectTimeout=20 -p $2 root@$1 "echo up" 2>/dev/null | grep -q up; }

mirror() {  # partial pull: results + logs (non-fatal; runs every probe)
  rsync -az --timeout=90 -e "ssh $SSHO -p $NP" \
    root@$NH:/workspace/chemical-transformer/results-nl/ ./results-nl/ 2>/dev/null
  for f in nl.log phase1.log phase2.log phase3.log phase_a*.log phase_b*.log \
           smoke.log smoke_data.log self_stop.log; do
    scp -O -q $SSHO -P $NP root@$NH:/workspace/$f ./mirror_$CURRENT-$f 2>/dev/null
  done
}

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  NURL=$(vastai ssh-url $CURRENT 2>/dev/null)
  if [ -z "$NURL" ]; then logf "no ssh-url for $CURRENT"; sleep 300; continue; fi
  NHP=${NURL#ssh://root@}; NH=${NHP%%:*}; NP=${NHP##*:}
  if probe $NH $NP; then
    UNREACH_SINCE=""
    if [ "$STAGED" = "0" ]; then
      logf "reachable — staging"
      rsync -az --timeout=120 -e "ssh $SSHO -p $NP" --exclude .git --exclude .venv \
        --exclude 'results*' --exclude '__pycache__' --exclude '*.pt' --exclude paper \
        ./ root@$NH:/workspace/chemical-transformer/ && STAGE_CODE=0 || STAGE_CODE=1
      for t in 1 2 3; do scp -O -q $SSHO -P $NP $MANIFEST root@$NH:/workspace/manifest.sh && break; sleep 30; done
      for t in 1 2 3; do scp -O -q $SSHO -P $NP ~/.config/vastai/vast_tfa_key root@$NH:/workspace/vast_tfa_key && break; sleep 30; done
      ssh $SSHO -p $NP root@$NH "cd /workspace/chemical-transformer && rm -f /workspace/$DONE_MARKER && /opt/conda/bin/pip install -q --no-input vastai numpy scipy matplotlib 'transformers>=4.40' 'datasets>=2.19' 'accelerate>=0.30' 2>&1 | tail -1; mkdir -p data/nl results-nl results-nl-smoke"
      DEADLINE2=$(($(date +%s) + 16*3600))
      ssh $SSHO -p $NP root@$NH "tmux kill-session -t nl 2>/dev/null; tmux kill-session -t selfstop 2>/dev/null; tmux new-session -d -s nl 'bash /workspace/manifest.sh > /workspace/nl.log 2>&1; touch /workspace/$DONE_MARKER'; tmux new-session -d -s selfstop '/workspace/chemical-transformer/scripts/vast_watchdog.sh $CURRENT /workspace/vast_tfa_key $DEADLINE2 /workspace/$DONE_MARKER'"
      STAGED=1
      logf "launched (internal deadline 16h)"
    fi
    mirror
    DONE=$(ssh $SSHO -o ConnectTimeout=20 -p $NP root@$NH "ls /workspace/$DONE_MARKER 2>/dev/null" 2>/dev/null)
    if [ -n "$DONE" ]; then
      logf "manifest complete — final pull + destroy"
      for t in 1 2 3; do rsync -az --timeout=120 -e "ssh $SSHO -p $NP" root@$NH:/workspace/chemical-transformer/results-nl/ ./results-nl/ 2>/dev/null && break; sleep 30; done
      for t in 1 2 3; do rsync -az --timeout=120 -e "ssh $SSHO -p $NP" root@$NH:/workspace/chemical-transformer/results-p3b/ ./results-p3b/ 2>/dev/null && break; sleep 30; done
      for f in nl.log phase1.log phase2.log phase3.log phase_a*.log phase_b*.log \
               smoke.log smoke_data.log self_stop.log; do
        scp -O -q $SSHO -P $NP root@$NH:/workspace/$f ./$f 2>/dev/null
      done
      git add results-nl results-p3b 2>/dev/null
      git diff --cached --quiet 2>/dev/null || { git commit -q -m "data: NL extras (autopilot)"; git push -q origin main 2>/dev/null; }
      # user directive: destroy (no storage charges) once results are local;
      # stop (keep disk) only if the pull looks empty — manual salvage then.
      GOT=""
      if [ -n "$EXPECTED" ]; then
        for pat in $EXPECTED; do
          F=$(find results-nl -path "results-nl/$pat" 2>/dev/null | head -1)
          [ -n "$F" ] && GOT="$F" && break
        done
      fi
      if [ -n "$GOT" ]; then
        vastai destroy instance $CURRENT -y >/dev/null 2>&1
        logf "COMPLETE — results local ($GOT), instance destroyed"
      else
        vastai stop instance $CURRENT >/dev/null 2>&1
        logf "COMPLETE but expected files missing ($EXPECTED) — instance STOPPED for salvage"
      fi
      exit 0
    fi
  else
    NOW=$(date +%s)
    if [ -z "$UNREACH_SINCE" ]; then UNREACH_SINCE=$NOW; fi
    if [ $((NOW - UNREACH_SINCE)) -gt 5400 ]; then
      logf "unreachable >90min — destroying $CURRENT (no auto re-rent; manual)"
      vastai destroy instance $CURRENT -y >/dev/null 2>&1
      logf "MANUAL INTERVENTION NEEDED — re-rent and relaunch"
      exit 1
    fi
  fi
  sleep 300
done
logf "deadline reached"
