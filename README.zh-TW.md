[English](README.md) | **中文**

# EB 走廊 — 由 ROS 2 Bag 訓練高斯潑濺

用兩段 ROS 2（Jazzy）錄影（`EB_try_h1`、`EB_try_h3`）重建室內走廊的 3D Gaussian Splatting 模型。資料由搭載
ZED 雙目相機、Livox LiDAR 與 RealSense T265 的地面機器人錄製。兩段錄影透過 ZED VIO 位姿、跨錄影對齊與
COLMAP 聯合光束法平差（bundle adjustment）融合成同一個場景，再用 [gsplat](https://github.com/nerfstudio-project/gsplat) 訓練。

![輸入影像](assets/input_frames.jpg)

## 結果

所有模型都在保留的測試影像（每 8 張取 1 張）上評估。訓練好的模型（`.ply` / `.pt`）與漫遊影片放在
[Releases](../../releases) 頁面。

| 模型 | 影像數 | 步數 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 高斯數 |
|---|---|---|---|---|---|---|
| 只用 h1（基準） | 512 | 30k | 26.67 | 0.883 | 0.378 | 56 萬 |
| h1+h3 合併，2 fps | 930 | 30k | 27.38 | 0.905 | 0.324 | 62 萬 |
| **h1+h3 合併，4 fps** | **1791** | **50k** | **28.34** | **0.909** | **0.313** | **81 萬** |

> 各列的測試集不同（取自不同的抽幀集合），數字只能大致比較。下面的並排比較使用幾乎相同的視角
> （時間差 ≤ 0.07 秒）。

**同視角比較** — 左：真實影像 · 中：合併 2 fps · 右：合併 4 fps

![2 fps 與 4 fps 比較](assets/compare_2fps_vs_4fps.jpg)

4 fps 模型在大部分視角更清晰（消防箱、樓梯、扶手）。**已知問題：** `h1` 開頭附近的視角，鏡頭正前方
出現一大塊漂浮物（第 1 列），地板也有一條帶狀假影（第 2 列）。可能原因是機器人開頭幾乎靜止，貼近鏡頭
的高斯只被很少的視角看到而無法被修正。預計的修法：`--near-plane 0.3`，或刪除靜止段的影像。

### 消融實驗（4 fps 資料、7k 步、每次只改一個參數）

| 實驗 | 參數 | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| A | 預設 | **25.23** | **0.885** | **0.398** |
| B | `--pose-opt` | 25.11 | 0.883 | 0.403 |
| C | `--app-opt` | 24.89 | 0.883 | 0.405 |
| D | `--strategy.absgrad --strategy.grow-grad2d 6e-4` | 22.59 | 0.859 | 0.457 |

第一次「改進版」同時開了 `pose_opt + app_opt + absgrad + bilateral grid`，結果崩壞（7k 步 PSNR 20.7、
大量漂浮物，見 `assets/failed_absgrad_combo_7k.jpg`）。消融實驗顯示，在這個低紋理場景中 **absgrad 是元兇**；
`pose_opt` / `app_opt` 沒有幫助，因為聯合 BA 後的位姿已經很準，兩段錄影的曝光也一致。bilateral grid 因為
速度被拿掉：它的 TV 正則項每一步都會對全部影像的網格計算（1791 張時約 190 ms/步）。

## 流程

```
ROS 2 bag (.mcap)
 ├─ /zed/zed_node/rgb/color/rect/image/compressed  → extract.py：每 1/fps 秒挑最清晰的一張
 ├─ /zed/zed_node/rgb/color/rect/camera_info       → 固定 PINHOLE 內參（fx≈528，1280×720，已校正）
 ├─ /zed/zed_node/pose + /tf_static                → zed_to_colmap.py：camera_link → 左目光學座標位姿
 │
 ├─ COLMAP SIFT + 序列匹配（各段錄影）
 ├─ 跨錄影對齊：把 h3 註冊進 h1（image_registrator）→ align.py：RANSAC 剛體擬合
 ├─ spatial_pairs.py：用對齊後的位姿匹配空間上相近的 h1/h3 影像
 ├─ point_triangulator → bundle_adjuster → 重新三角化 → BA → point_filtering
 └─ gsplat simple_trainer.py
```

為什麼用 ZED 位姿而不是純 COLMAP SfM：走廊地板反光、牆面無紋理，COLMAP mapper 把 `h1` 切成 4 個互不相連的
模型（512 張中 287/81/104/12 張）。改用 ZED VIO 位姿三角化後，全部影像都註冊成功，重投影誤差 0.97 px。

| 稀疏模型 | 影像數 | 點數 | 平均 track 長度 | 重投影誤差 |
|---|---|---|---|---|
| 只用 h1（ZED 位姿） | 512 | 3.78 萬 | 6.9 | 0.97 px |
| 合併 2 fps | 930 | 7.93 萬 | 6.7 | 0.64 px |
| 合併 4 fps | 1791 | 13.61 萬 | 8.7 | 0.66 px |

跨錄影對齊：418 張 `h3` 中有 367 張註冊進 `h1`；RANSAC 剛體擬合得到 250 個內點（因 VIO 漂移，殘差中位數
18 cm / 2.5°），再由聯合 BA 修正（位姿修正量中位數 9 cm）。轉換矩陣見
[`results/alignment/T_h1_h3.txt`](results/alignment/T_h1_h3.txt)。

## 重現

環境：RTX 5060 Ti 16 GB · PyTorch 2.9.1 + CUDA 12.8 · gsplat 1.5.3 · COLMAP 3.13（CUDA）·
`pip install mcap mcap-ros2-support`。訓練前要先 activate conda 環境（gsplat 的 JIT 編譯需要 `CUDA_HOME`）。

```bash
# 合併 4 fps 場景（bag 路徑見腳本內設定）
scripts/run_joint.sh 4 joint4 --max-steps 50000 --eval-steps 7000 30000 50000 \
  --save-steps 30000 50000 --ply-steps 30000 50000 --strategy.refine-stop-iter 25000

# 單一 bag
scripts/run_scene.sh path/to/bag.mcap h1 2
```

`run_joint.sh` 會沿用 2 fps 那一輪算出的對齊矩陣 `T_h1_h3.npy`（同樣的 bag、同樣的 map 座標系）。

瀏覽訓練好的模型：

```bash
cd gsplat/examples
python simple_viewer.py --ckpt ckpt_49999_rank0.pt --port 8080   # 再用瀏覽器開 http://localhost:8080
```

`.ply` 是標準 3DGS 格式，可以用 SuperSplat 等工具開啟。

## 倉庫結構

```
scripts/            流程腳本（抽幀、ZED→COLMAP、對齊、配對、訓練）
results/
  joint_2fps/       cfg.yml 與驗證/訓練統計
  joint_4fps/       cfg.yml 與驗證/訓練統計
  ablation_4fps_7k/ A–D 消融實驗的設定與統計
  alignment/        h3→h1 剛體轉換
assets/             README 使用的圖片
```
