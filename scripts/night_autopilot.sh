#!/bin/bash
# Overnight autopilot: waits for instance ssh, stages + launches the NL
# manifest, monitors completion, pulls results, pushes to git, stops the
# instance. If the host stays unreachable >90 min, destroys it and re-rents
# from the fallback list. All state logged to /tmp/autopilot.log.
log() { echo "$(date +%H:%M:%S) $*" >> /tmp/autopilot.log; }
SSHO="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
DEADLINE=$(($(date +%s) + 20*3600))
command -v vastai >/dev/null 2>&1 || PATH="$(pwd)/.venv/bin:$PATH"
CURRENT=51168200
FALLBACK_IDS="42274230 45716570"
UNREACH_SINCE=""
STAGED=0
log "autopilot armed: instance $CURRENT, fallbacks: $FALLBACK_IDS"

probe() { ssh $SSHO -o ConnectTimeout=20 -p $2 root@$1 "echo up" 2>/dev/null | grep -q up; }

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  NURL=$(vastai ssh-url $CURRENT 2>/dev/null)
  if [ -z "$NURL" ]; then log "no ssh-url for $CURRENT"; sleep 300; continue; fi
  NHP=${NURL#ssh://root@}; NH=${NHP%%:*}; NP=${NHP##*:}
  if probe $NH $NP; then
    UNREACH_SINCE=""
    if [ "$STAGED" = "0" ]; then
      log "reachable — staging"
      rsync -az --timeout=120 -e "ssh $SSHO -p $NP" --exclude .git --exclude .venv \
        --exclude 'results*' --exclude '__pycache__' --exclude '*.pt' --exclude paper \
        ./ root@$NH:/workspace/chemical-transformer/ && STAGE_CODE=0 || STAGE_CODE=1
      for t in 1 2 3; do scp -O -q $SSHO -P $NP manifests/nl_real.sh root@$NH:/workspace/manifest.sh && break; sleep 30; done
      for t in 1 2 3; do scp -O -q $SSHO -P $NP ~/.config/vastai/vast_tfa_key root@$NH:/workspace/vast_tfa_key && break; sleep 30; done
      ssh $SSHO -p $NP root@$NH "cd /workspace/chemical-transformer && /opt/conda/bin/pip install -q --no-input vastai numpy scipy matplotlib 'transformers>=4.40' 'datasets>=2.19' 'accelerate>=0.30' 2>&1 | tail -1; mkdir -p data/nl results-nl results-nl-smoke"
      DEADLINE2=$(($(date +%s) + 8*3600))
      ssh $SSHO -p $NP root@$NH "tmux kill-session -t nl 2>/dev/null; tmux kill-session -t selfstop 2>/dev/null; tmux new-session -d -s nl 'bash /workspace/manifest.sh > /workspace/nl.log 2>&1; touch /workspace/NL_REAL_DONE'; tmux new-session -d -s selfstop '/workspace/chemical-transformer/scripts/vast_watchdog.sh $CURRENT /workspace/vast_tfa_key $DEADLINE2 /workspace/NL_REAL_DONE'"
      STAGED=1
      log "launched (internal deadline 8h)"
    fi
    DONE=$(ssh $SSHO -o ConnectTimeout=20 -p $NP root@$NH "ls /workspace/NL_REAL_DONE 2>/dev/null" 2>/dev/null)
    if [ -n "$DONE" ]; then
      log "manifest complete — final pull + stop"
      for t in 1 2 3; do rsync -az --timeout=120 -e "ssh $SSHO -p $NP" root@$NH:/workspace/chemical-transformer/results-nl/ ./results-nl/ 2>/dev/null && break; sleep 30; done
      for t in 1 2 3; do rsync -az --timeout=120 -e "ssh $SSHO -p $NP" root@$NH:/workspace/chemical-transformer/results-p3b/ ./results-p3b/ 2>/dev/null && break; sleep 30; done
      for f in nl.log phase1.log phase2.log phase3.log smoke.log smoke_data.log self_stop.log; do
        scp -O -q $SSHO -P $NP root@$NH:/workspace/$f ./$f 2>/dev/null
      done
      git add results-nl results-p3b 2>/dev/null
      git diff --cached --quiet 2>/dev/null || { git commit -q -m "data: NL parity pair + grid remainder (autopilot)"; git push -q origin main 2>/dev/null; }
      # user directive: destroy (no storage charges) once results are local;
      # stop (keep disk) only if the pull looks empty — manual salvage then.
      if [ -n "$(ls results-nl/webtext/*.json 2>/dev/null)" ]; then
        vastai destroy instance $CURRENT -y >/dev/null 2>&1
        log "COMPLETE — results local, instance destroyed"
      else
        vastai stop instance $CURRENT >/dev/null 2>&1
        log "COMPLETE but pull looks empty — instance STOPPED for salvage"
      fi
      exit 0
    fi
  else
    NOW=$(date +%s)
    if [ -z "$UNREACH_SINCE" ]; then UNREACH_SINCE=$NOW; fi
    if [ $((NOW - UNREACH_SINCE)) -gt 5400 ]; then
      log "unreachable >90min — destroying $CURRENT and re-renting"
      vastai destroy instance $CURRENT >/dev/null 2>&1
      NEXT=$(echo $FALLBACK_IDS | cut -d' ' -f1)
      if [ -n "$NEXT" ]; then
        vastai create instance $NEXT --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime --disk 60 --ssh --direct --onstart-cmd "nvidia-smi" >/dev/null 2>&1
        log "re-rented $NEXT"
        FALLBACK_IDS=$(echo $FALLBACK_IDS | cut -d' ' -f2-)
        CURRENT=$NEXT
        UNREACH_SINCE=""
        STAGED=0
      else
        log "no fallback offers left — manual intervention needed"
        exit 1
      fi
    fi
  fi
  sleep 300
done
log "deadline reached"
