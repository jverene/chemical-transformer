#!/bin/bash
# Launch a stage of experiments on a rented vast.ai instance and pull results.
#
# Usage:
#   scripts/vast_launch.sh <INSTANCE_ID> <MANIFEST_FILE> [MAX_HOURS] [MAX_DOLLARS]
#
# <MANIFEST_FILE>: local text file, one shell command per line, executed
#                  sequentially inside /workspace/ct on the instance.
# Env: VAST_API_KEY (required, from https://cloud.vast.ai/account/cli/).
#
# Behavior: rsyncs the working tree (no .git/.venv/results*), sets up the env,
# runs the manifest under tmux, arms the self-stop watchdog (hours / dollar
# cap / done-marker, whichever first), then polls, pulling results/ and
# figures/ every 10 min until the manifest completes or the instance stops.
# NEVER destroys the instance — stop only.
set -uo pipefail

ID=${1:?instance id required}
MANIFEST=${2:?manifest file required}
MAX_HOURS=${3:-24}
MAX_DOLLARS=${4:-40}
SSHO="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=25"
KEY=${VAST_API_KEY:?set VAST_API_KEY}
REPO=chemical-transformer
# vastai may live in the project venv
command -v vastai >/dev/null 2>&1 || PATH="$(pwd)/.venv/bin:$PATH"
command -v vastai >/dev/null 2>&1 || { echo "vastai CLI not found"; exit 1; }

[ -f "$MANIFEST" ] || { echo "manifest $MANIFEST missing"; exit 1; }

conn() {
  local U="" i
  for i in $(seq 1 20); do
    NURL=$(vastai ssh-url $ID 2>/dev/null) || { sleep 30; continue; }
    NHP=${NURL#ssh://root@}; NH=${NHP%%:*}; NP=${NHP##*:}
    U=$(ssh $SSHO -p $NP root@$NH "echo ok" 2>/dev/null)
    [ "$U" = ok ] && { echo "$NH $NP"; return 0; }
    sleep 30
  done
  echo "cannot reach instance $ID" >&2; return 1
}
read -r NH NP <<< "$(conn)" || exit 1
echo "[1/5] connected $NH:$NP"

# --- code + manifest up (rsync keeps iterations cheap) ---
# Parent dirs must be included explicitly or the excluded results* tree is
# pruned before the file-level includes are ever reached.
rsync -az -e "ssh $SSHO -p $NP" \
  --include 'results-p0' \
  --include 'results-p0/tagged' \
  --include 'results-p0/tagged/predictor-supervised_stage1_seed*.pt' \
  --include 'results-p0/tagged/predictor-supervised_stage1_seed*.json' \
  --exclude .git --exclude .venv --exclude 'results*' --exclude '__pycache__' \
  --exclude '*.pt' --exclude paper --exclude figures \
  ./ root@$NH:/workspace/$REPO/
scp -q $SSHO -P $NP "$MANIFEST" root@$NH:/workspace/manifest.sh
# 2FA session key: the ONLY credential that can stop the instance (vast
# requires a 2FA session for state changes). Staged so the remote watchdog
# is fully self-sufficient — no local machine in the loop.
scp -q $SSHO -P $NP ~/.config/vastai/vast_tfa_key root@$NH:/workspace/vast_tfa_key
echo "[2/5] code staged"

# --- env setup (idempotent) + start ---
ssh $SSHO -p $NP root@$NH 'bash -s' <<EOF
cd /workspace/$REPO
PY=/opt/conda/bin/python
\$PY -c "import torch, scipy, numpy, matplotlib" 2>/dev/null || \
  /opt/conda/bin/pip install -q --no-input scipy matplotlib >/tmp/pip.log 2>&1
\$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
command -v vastai >/dev/null 2>&1 || /opt/conda/bin/pip install -q --no-input vastai >/tmp/pip2.log 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p results figures
DEADLINE=\$((\$(date +%s) + $MAX_HOURS*3600))
echo "$ID $KEY \$DEADLINE" > /workspace/wdt_args
chmod +x scripts/vast_watchdog.sh
tmux kill-session -t ct 2>/dev/null; tmux kill-session -t selfstop 2>/dev/null
rm -f /workspace/MANIFEST_DONE /workspace/NLX_DONE /workspace/NL_REAL_DONE
tmux new-session -d -s ct 'cd /workspace/$REPO; bash /workspace/manifest.sh 2>&1 | tee /workspace/ct.log; touch /workspace/MANIFEST_DONE'
tmux new-session -d -s selfstop "scripts/vast_watchdog.sh $ID /workspace/vast_tfa_key \$DEADLINE /workspace/MANIFEST_DONE"
echo "watchdog: stops at \$DEADLINE or on MANIFEST_DONE"
EOF
echo "[3/5] launched (max ${MAX_HOURS}h, cap \$$MAX_DOLLARS — check dashboard spend alert)"

# --- local watcher: pull + completion check every 10 min ---
while true; do
  sleep 600
  NURL=$(vastai ssh-url $ID 2>/dev/null) || continue
  NHP=${NURL#ssh://root@}; NH2=${NHP%%:*}; NP2=${NHP##*:}
  rsync -az -e "ssh $SSHO -p $NP2" \
    root@$NH2:/workspace/$REPO/results/ ./results_vast/ 2>/dev/null
  rsync -az -e "ssh $SSHO -p $NP2" \
    root@$NH2:/workspace/$REPO/figures/ ./figures_vast/ 2>/dev/null
  DONE=$(ssh $SSHO -o ConnectTimeout=20 -p $NP2 root@$NH2 \
    "ls /workspace/MANIFEST_DONE 2>/dev/null" 2>/dev/null)
  if [ -n "$DONE" ]; then
    echo "[5/5] MANIFEST complete $(date) — results in ./results_vast/"
    echo "Instance left RUNNING-check: verify pull, then 'vastai stop instance $ID'"
    exit 0
  fi
  ssh $SSHO -o ConnectTimeout=20 -p $NP2 root@$NH2 "echo ok" >/dev/null 2>&1 \
    || { echo "$(date) unreachable (watchdog owns billing)"; exit 0; }
  echo "$(date +%H:%M) pulled results_vast/ (waiting)"
done
