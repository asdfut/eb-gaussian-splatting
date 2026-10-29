#!/usr/bin/env bash
# Usage: run_scene.sh <bag.mcap> <scene_name> [fps]
# mcap -> sharp frames -> COLMAP features/matches -> ZED poses -> triangulate -> gsplat
set -eo pipefail
BAG=$(realpath "$1"); NAME=$2; FPS=${3:-2}
ROOT=$(cd "$(dirname "$0")/.." && pwd); S=$ROOT/data/$NAME; G=$ROOT/data/${NAME}_gs
source "$HOME/miniconda3/bin/activate" gs; set -u

python -I "$ROOT/scripts/extract.py" "$BAG" "$S" --fps "$FPS"
read -r W H FX FY CX CY < "$S/intrinsics.txt"
rm -f "$S/database.db"
colmap feature_extractor --database_path "$S/database.db" --image_path "$S/images" \
  --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE \
  --ImageReader.camera_params "$FX,$FY,$CX,$CY" --SiftExtraction.max_num_features 8192
colmap sequential_matcher --database_path "$S/database.db" --SequentialMatching.overlap 15
python -I "$ROOT/scripts/zed_to_colmap.py" "$BAG" "$S"
mkdir -p "$S/zed_tri"
colmap point_triangulator --database_path "$S/database.db" --image_path "$S/images" \
  --input_path "$S/zed_model" --output_path "$S/zed_tri" \
  --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0
colmap model_analyzer --path "$S/zed_tri"

mkdir -p "$G/sparse"; ln -sfn "$S/images" "$G/images"; rm -rf "$G/sparse/0"; cp -r "$S/zed_tri" "$G/sparse/0"
cd "$HOME/gsplat/examples"
python simple_trainer.py default --data-dir "$G" --data-factor 1 --result-dir "$ROOT/results/$NAME" \
  --disable-viewer --save-ply --max-steps 30000
