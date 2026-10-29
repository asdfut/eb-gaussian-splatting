#!/usr/bin/env bash
# 7k-step MCMC stability checks, keeping the 50k LR schedule (stop after the 7k eval).
set -eo pipefail
source "$HOME/miniconda3/bin/activate" gs
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$HOME/gsplat/examples"
run() { name=$1; strat=$2; shift 2; out=$ROOT/results/ablate/$name
  python simple_trainer.py $strat --data-dir "$ROOT/data/joint4_gs" --data-factor 1 \
    --result-dir "$out" --disable-viewer --disable-video \
    --max-steps 50000 --eval-steps 7000 --save-steps 7000 "$@" > "$out.log" 2>&1 &
  pid=$!
  until [ -f "$out/stats/val_step6999.json" ] || ! kill -0 $pid 2>/dev/null; do sleep 5; done
  kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true
  echo "$name $(cat "$out/stats/val_step6999.json")"
  tr '\r' '\n' < "$out.log" | grep -E "/50000 \[" | awk 'NR%600==1' | sed -E 's/\|.*\| ([0-9]+)\/.*/ @\1/' | tr '\n' ' '; echo; }
run M2_mcmc_noise5e4 mcmc --strategy.noise-lr 5e4
run M3_mcmc_opreg1e-3 mcmc --opacity-reg 0.001
