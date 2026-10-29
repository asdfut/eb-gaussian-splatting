#!/usr/bin/env bash
# 7k-step ablations on the 4 fps joint dataset; one changed flag per run.
set -eo pipefail
source "$HOME/miniconda3/bin/activate" gs
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$HOME/gsplat/examples"
run() { name=$1; shift
  python simple_trainer.py default --data-dir "$ROOT/data/joint4_gs" --data-factor 1 \
    --result-dir "$ROOT/results/ablate/$name" --disable-viewer --disable-video \
    --max-steps 7000 --eval-steps 7000 --save-steps 7000 "$@" > "$ROOT/results/ablate/$name.log" 2>&1
  echo "$name $(cat "$ROOT/results/ablate/$name/stats/val_step6999.json")"; }
mkdir -p "$ROOT/results/ablate"
run A_base
run B_pose_opt --pose-opt
run C_app_opt --app-opt
run D_absgrad --strategy.absgrad --strategy.grow-grad2d 6e-4
