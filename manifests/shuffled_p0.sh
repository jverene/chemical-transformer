# P0 shuffled controls, all 3 seeds (~$1 total; Stage-1 ckpts staged by launcher)
python -m ct.gategrad --seeds 0,1,2 --outdir results-p0 --arms shuffled --device cuda
echo SHUFFLED_DONE
