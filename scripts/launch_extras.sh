#!/bin/bash
# One-shot: complete vast.ai 2FA login with the emailed code, rent two GPUs,
# and launch both NL-extras autopilots.
# Usage: bash scripts/launch_extras.sh <6-DIGIT-CODE> [SECRET]
set -eo pipefail
CODE="${1:?usage: launch_extras.sh <6-DIGIT-CODE> [SECRET]}"
SECRET="${2:-82c8916bd51c6c086784ba23eaed6433}"
V=".venv/bin/vastai --api-key $(cat ~/.config/vastai/vast_api_key)"

$V tfa login --method-type email --secret "$SECRET" -c "$CODE" 2>&1 | tail -1
echo "=== 2FA ok — renting ==="

$V create instance 47545530 --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
  --disk 45 --ssh --direct --onstart-cmd "nvidia-smi" 2>&1 | tail -1 | head -c 100; echo " (A: webtext s1 + oracle)"
$V create instance 31632913 --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
  --disk 45 --ssh --direct --onstart-cmd "nvidia-smi" 2>&1 | tail -1 | head -c 100; echo " (B: code+math)"
sleep 5
IDA=$($V show instances --raw 2>/dev/null | .venv/bin/python -c "
import json,sys
rows = json.load(sys.stdin)
rows.sort(key=lambda r: r['id'])
print(' '.join(str(r['id']) for r in rows))")
echo "instances: $IDA"
set -- $IDA
nohup bash scripts/night_autopilot.sh "$1" manifests/nl_extra_a.sh /tmp/autopilot_a.log NLX_DONE > /tmp/ap_a.out 2>&1 &
echo "autopilot A pid $! -> $1"
nohup bash scripts/night_autopilot.sh "$2" manifests/nl_extra_b.sh /tmp/autopilot_b.log NLX_DONE > /tmp/ap_b.out 2>&1 &
echo "autopilot B pid $! -> $2"
