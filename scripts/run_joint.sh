#!/usr/bin/env bash
# Joint h1+h3 pipeline: extract -> per-bag SIFT -> sequential + spatial cross matching
# -> ZED poses (h3 aligned into h1 with T_h1_h3.npy) -> triangulate/BA x2 -> filter -> gsplat
# Usage: run_joint.sh <fps> <name> [extra simple_trainer args...]
set -eo pipefail
FPS=$1; NAME=$2; shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd); D=$ROOT/data; J=$D/$NAME
EB=$(cd "$ROOT/../eb" && pwd)
declare -A BAG=([h1]=$EB/EB_try_h1_0-002.mcap [h3]=$EB/EB_try-20261029T164219Z-1-001/EB_try/EB_try_h3/EB_try_h3_0.mcap)
source "$HOME/miniconda3/bin/activate" gs; set -u
F="--Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0"
B="--BundleAdjustment.refine_focal_length 0 --BundleAdjustment.refine_principal_point 0 --BundleAdjustment.refine_extra_params 0"

echo "### [1/6] extract frames @ ${FPS} fps"
for b in h1 h3; do python -I "$ROOT/scripts/extract.py" "${BAG[$b]}" "$D/${b}_${NAME}" --fps "$FPS"; done

echo "### [2/6] features + matching"
rm -rf "$J"; mkdir -p "$J/images"; cd "$J"
for b in h1 h3; do
  cp -al "$D/${b}_${NAME}/images" "images/$b"
  ls "images/$b" | sed "s|^|$b/|" > "list_$b.txt"
  read -r W H FX FY CX CY < "$D/${b}_${NAME}/intrinsics.txt"
  colmap feature_extractor --database_path database.db --image_path images --image_list_path "list_$b.txt" \
    --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE --ImageReader.camera_params "$FX,$FY,$CX,$CY" \
    --SiftExtraction.max_num_features 8192 > "fe_$b.log" 2>&1
done
colmap sequential_matcher --database_path database.db --SequentialMatching.overlap 20 > sm.log 2>&1

echo "### [3/6] ZED poses -> COLMAP (h3 aligned into h1 frame)"
python -I "$ROOT/scripts/zed_to_colmap.py" "${BAG[h1]}" "$D/h1_${NAME}" --db database.db --prefix h1/ --out zed_model
python -I "$ROOT/scripts/zed_to_colmap.py" "${BAG[h3]}" "$D/h3_${NAME}" --db database.db --prefix h3/ --out zed_model \
  --align "$D/joint/T_h1_h3.npy" --append

echo "### [4/6] spatial cross-session matching"
python -I "$ROOT/scripts/spatial_pairs.py" zed_model/images.txt pairs_spatial.txt
colmap matches_importer --database_path database.db --match_list_path pairs_spatial.txt --match_type pairs > mi.log 2>&1

echo "### [5/6] triangulate + BA (x2) + filter"
mkdir -p tri1 ba1 tri2 ba2 final
colmap point_triangulator --database_path database.db --image_path images --input_path zed_model --output_path tri1 $F > tri1.log 2>&1
colmap bundle_adjuster --input_path tri1 --output_path ba1 $B > ba1.log 2>&1
colmap point_triangulator --database_path database.db --image_path images --input_path ba1 --output_path tri2 $F > tri2.log 2>&1
colmap bundle_adjuster --input_path tri2 --output_path ba2 $B > ba2.log 2>&1
colmap point_filtering --input_path ba2 --output_path final --max_reproj_error 2 --min_tri_angle 1.5 --min_track_len 2 > pf.log 2>&1
colmap model_analyzer --path final 2>&1 | grep -E "Registered images|Points:|track length|reprojection"

echo "### [6/6] train gsplat"
G=$D/${NAME}_gs; rm -rf "$G"; mkdir -p "$G/sparse"; cp -al images "$G/images"; cp -r final "$G/sparse/0"
cd "$HOME/gsplat/examples"
python simple_trainer.py default --data-dir "$G" --data-factor 1 --result-dir "$ROOT/results/$NAME" \
  --disable-viewer --save-ply "$@"
echo "### done"
