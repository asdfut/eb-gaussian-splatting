**English** | [中文](README.zh-TW.md)

# EB Corridor — Gaussian Splatting from ROS 2 Bags

3D Gaussian Splatting reconstruction of an indoor corridor, trained from two ROS 2 (Jazzy) recordings
(`EB_try_h1`, `EB_try_h3`) captured by a ground robot carrying a ZED stereo camera, a Livox LiDAR and a
RealSense T265. The two recordings are fused into a single scene using the ZED VIO poses, a cross-session
alignment and a joint COLMAP bundle adjustment, then trained with [gsplat](https://github.com/nerfstudio-project/gsplat).

![Input frames](assets/input_frames.jpg)

## Results

All models are evaluated on held-out frames (every 8th image). Trained models (`.ply` / `.pt`) and
fly-through videos are attached to the [Releases](../../releases) page.

| Model | Images | Steps | PSNR ↑ | SSIM ↑ | LPIPS ↓ | #Gaussians |
|---|---|---|---|---|---|---|
| h1 only (baseline) | 512 | 30k | 26.67 | 0.883 | 0.378 | 0.56 M |
| Joint h1+h3, 2 fps | 930 | 30k | 27.38 | 0.905 | 0.324 | 0.62 M |
| **Joint h1+h3, 4 fps** | **1791** | **50k** | **28.34** | **0.909** | **0.313** | **0.81 M** |

> The test sets differ between rows (they are drawn from different frame sets), so the numbers are only
> roughly comparable. The side-by-side comparison below uses near-identical viewpoints (≤ 0.07 s apart).

**Same-view comparison** — left: ground truth · middle: joint 2 fps · right: joint 4 fps

![2 fps vs 4 fps](assets/compare_2fps_vs_4fps.jpg)

The 4 fps model is sharper in most views (fire cabinet, stairs, railings). **Known issue:** a large
floater appears right in front of the camera near the beginning of `h1` (row 1), plus a band artifact on
the floor (row 2). Likely cause: the robot was nearly stationary at the start, so Gaussians close to the
lens are seen by too few views to be corrected. Planned fix: `--near-plane 0.3` and/or dropping the
stationary frames.

### Ablation (4 fps data, 7k steps, one flag changed at a time)

| Run | Flag | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| A | default | **25.23** | **0.885** | **0.398** |
| B | `--pose-opt` | 25.11 | 0.883 | 0.403 |
| C | `--app-opt` | 24.89 | 0.883 | 0.405 |
| D | `--strategy.absgrad --strategy.grow-grad2d 6e-4` | 22.59 | 0.859 | 0.457 |

A first "improved" run that combined `pose_opt + app_opt + absgrad + bilateral grid` collapsed
(PSNR 20.7 at 7k, large floaters — see `assets/failed_absgrad_combo_7k.jpg`). The ablation shows **absgrad
is the culprit** on this low-texture scene; `pose_opt` / `app_opt` bring no gain because the joint BA poses
are already accurate and exposure is consistent across the two sessions. The bilateral grid was dropped for
speed: its TV regulariser runs over every image's grid each step (~190 ms/step with 1791 images).

## Pipeline

```
ROS 2 bag (.mcap)
 ├─ /zed/zed_node/rgb/color/rect/image/compressed  → extract.py: sharpest frame per 1/fps window
 ├─ /zed/zed_node/rgb/color/rect/camera_info       → fixed PINHOLE intrinsics (fx≈528, 1280×720, rectified)
 ├─ /zed/zed_node/pose + /tf_static                → zed_to_colmap.py: camera_link → left optical frame poses
 │
 ├─ COLMAP SIFT + sequential matching (per session)
 ├─ cross-session alignment: register h3 into h1 (image_registrator) → align.py: RANSAC rigid fit
 ├─ spatial_pairs.py: match nearby h1/h3 views using the aligned poses
 ├─ point_triangulator → bundle_adjuster → re-triangulate → BA → point_filtering
 └─ gsplat simple_trainer.py
```

Why ZED poses instead of plain COLMAP SfM: on the corridor (reflective floor, textureless walls) COLMAP's
mapper split `h1` into 4 disconnected models (287/81/104/12 of 512 images). Triangulating with the ZED VIO
poses registered all images with 0.97 px reprojection error.

| Sparse model | Images | Points | Mean track length | Reproj. error |
|---|---|---|---|---|
| h1 only (ZED poses) | 512 | 37.8 k | 6.9 | 0.97 px |
| Joint 2 fps | 930 | 79.3 k | 6.7 | 0.64 px |
| Joint 4 fps | 1791 | 136.1 k | 8.7 | 0.66 px |

Cross-session alignment: 367 / 418 `h3` frames registered into `h1`; RANSAC rigid fit with 250 inliers
(median 18 cm / 2.5° residual due to VIO drift), corrected by the joint BA (median pose change 9 cm).
The transform is in [`results/alignment/T_h1_h3.txt`](results/alignment/T_h1_h3.txt).

## Reproduce

Environment: RTX 5060 Ti 16 GB · PyTorch 2.9.1 + CUDA 12.8 · gsplat 1.5.3 · COLMAP 3.13 (CUDA) ·
`pip install mcap mcap-ros2-support`. Activate the conda env before training (gsplat JIT needs `CUDA_HOME`).

```bash
# Joint 4 fps scene (bag paths are set at the top of the script)
scripts/run_joint.sh 4 joint4 --max-steps 50000 --eval-steps 7000 30000 50000 \
  --save-steps 30000 50000 --ply-steps 30000 50000 --strategy.refine-stop-iter 25000

# Single bag
scripts/run_scene.sh path/to/bag.mcap h1 2
```

`run_joint.sh` reuses the alignment `T_h1_h3.npy` from the 2 fps run (same bags, same map frames).

View a trained model:

```bash
cd gsplat/examples
python simple_viewer.py --ckpt ckpt_49999_rank0.pt --port 8080   # then open http://localhost:8080
```

The `.ply` files use the standard 3DGS format and open in viewers such as SuperSplat.

## Repository layout

```
scripts/            pipeline scripts (extraction, ZED→COLMAP, alignment, pairing, training)
results/
  joint_2fps/       cfg.yml + val/train stats
  joint_4fps/       cfg.yml + val/train stats
  ablation_4fps_7k/ A–D ablation configs and stats
  alignment/        h3→h1 rigid transform
assets/             figures used in this README
```
