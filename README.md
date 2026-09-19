# Direct Geometric Supervision for 2D Gaussian Surfels

Official implementation of **"Direct Geometric Supervision: World-Space LiDAR Anchoring for 2D Gaussian Surfels in Indoor Reconstruction"** (SIGGRAPH Asia 2026 Technical Communications).

Built on [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) (Huang et al., SIGGRAPH 2024).

---

## Overview

We introduce **Direct Geometric Supervision (DGS)**, a world-space regularizer that anchors each surfel to a tangent plane fitted from its local LiDAR neighbourhood, and **Geometry-Tethered Refinement (GTR)** for appearance recovery while preserving geometric accuracy.

Training runs in three stages:

| Stage | Iterations | Description |
|-------|-----------|-------------|
| **Densification** | 0–10k | Standard 2DGS + LiDAR depth/normal supervision |
| **DGS** | 10k–20k | Per-surfel KNN plane fit to LiDAR; penalises signed distance and normal misalignment |
| **GTR** | 20k–30k | Position LR drops 160×; DGS acts as geometric tether (λ = 0.05) |

---

## Results

### ScanNet++ (sparse iPhone LiDAR + high-quality RGB)

Average over six evaluation scenes, evaluated against Faro reference:

| Method | F@5 cm ↑ | Precision ↑ | Recall ↑ | Chamfer (cm) ↓ | NC ↑ | PSNR ↑ | SSIM ↑ |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Vanilla 2DGS | 0.399 | 0.431 | 0.381 | 15.92 | 0.797 | **23.88** | 0.897 |
| DN-Splatter | 0.767 | 0.817 | 0.731 | 9.09 | 0.881 | 23.22 | **0.904** |
| **DGS (Ours)** | **0.856** | **0.896** | **0.837** | **4.73** | **0.917** | 23.20 | 0.893 |

### Dense-LiDAR captures (AccP50 and Wall RMS in mm; Chamfer in cm)

| Scene | Method | AccP50 ↓ | Wall RMS ↓ | Chamfer ↓ | NC ↑ |
|-------|--------|:---:|:---:|:---:|:---:|
| ARTLab | Vanilla 2DGS | 73.7 | 9.79 | 33.05 | 0.773 |
| | DN-Splatter | **6.24** | 9.86 | 60.29 | 0.814 |
| | **DGS (Ours)** | 6.25 | **8.29** | **24.74** | **0.830** |
| m1 | Vanilla 2DGS | 34.4 | 13.52 | 7.04 | 0.830 |
| | DN-Splatter | 5.9 | 12.16 | 3.54 | 0.866 |
| | **DGS (Ours)** | **5.4** | **8.31** | **2.60** | **0.872** |
| m2 | Vanilla 2DGS | 62.0 | 12.78 | 11.43 | 0.778 |
| | DN-Splatter | 5.5 | 7.73 | 3.58 | 0.860 |
| | **DGS (Ours)** | **5.0** | **7.33** | **2.82** | **0.873** |
| m3 | Vanilla 2DGS | 69.6 | 13.92 | 12.15 | 0.813 |
| | DN-Splatter | 39.2 | **6.02** | 33.71 | 0.766 |
| | **DGS (Ours)** | **8.4** | 9.18 | **5.45** | **0.883** |
| m4 | Vanilla 2DGS | 125.5 | 14.11 | 27.61 | 0.663 |
| | DN-Splatter | 1214.9† | 4.77† | 99.99† | 0.640 |
| | **DGS (Ours)** | **19.2** | **10.46** | **11.81** | **0.730** |

† DN-Splatter diverged on m4 (degenerate mesh).

---

## Installation

```bash
git clone --recursive https://github.com/<your-org>/DGS-2DGS.git
cd DGS-2DGS

conda create -n dgs python=3.8 && conda activate dgs

# PyTorch — adjust the index URL for your CUDA version
pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu124

# Build CUDA submodules
pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn

# Python dependencies
pip install plyfile tqdm scipy opencv-python open3d trimesh scikit-learn laspy mediapy
```

> **Verified on:** Ubuntu 22.04 / 24.04, CUDA 12.4, Python 3.8, PyTorch 2.4.1, 48 GB VRAM (RTX 6000 Ada).

---

## Quick Start

```bash
# 1. Train
python train.py \
  -s scene/ -m output/my_model \
  --images images_masked --white_background \
  --lambda_depth 0.5 --lambda_lidar_normal 0.5 \
  --lambda_dgs 0.1 --lambda_dgs_normal 0.1 \
  --dgs_start_iter 10000 --dgs_interval 500 --dgs_k 8 --dgs_radius 0.05 \
  --las_path scene/pc_aligned.ply \
  --soft_phase_b --phase_b_start 20000 \
  --phase_b_xyz_lr 1e-6 --phase_b_dgs_lambda 0.05 \
  --densify_until_iter 15000 --iterations 30000

# 2. Extract mesh
python render.py \
  -s scene/ -m output/my_model --iteration 30000 \
  --skip_train --skip_test \
  --mesh_res 512 --depth_trunc 5.0 --num_cluster 50

# 3. Evaluate
python scripts/eval_mesh_scannetpp.py \
  --pred_mesh output/my_model/train/ours_30000/fuse_post.ply \
  --gt_mesh /path/to/gt_mesh_in_training_frame.ply \
  --transforms_json scene/sparse/0/images.txt \
  --camera_params scene/sparse/0/cameras.txt \
  --threshold 0.05 --clip_to_gt_bbox \
  --output output/my_model/eval_metrics.json
```

---

## Data Layout

```
scene/
├── sparse/0/
│   ├── cameras.txt              # COLMAP PINHOLE camera intrinsics
│   ├── images.txt               # COLMAP camera poses (w2c)
│   └── points3D.ply             # Voxel-downsampled LiDAR for Gaussian init
├── images_masked/<subdir>/      # RGB frames (PNG with alpha mask, or plain PNG)
├── depth_maps/<subdir>/         # Per-frame .npy, float32, metres (0 = no data)
├── normal_maps/<subdir>/        # Per-frame .npy, float32, world-space normals
└── pc_aligned.ply               # Full-resolution LiDAR point cloud for DGS
```

**Generating depth and normal maps** from a LiDAR point cloud:

```bash
python scripts/lidar_to_depth_maps.py \
  --las scene/pc_aligned.ply \
  --source scene/ --voxel_size 0.002 --mask_folder images_masked
```

---

## ScanNet++ Data Preparation

ScanNet++ scenes require coordinate-frame alignment between the Faro scanner, the iPhone SLAM poses, and the GT evaluation mesh.

### Step 1: Convert to COLMAP format

```bash
# iPhone captures (stride=5, ~1267 frames) — used for all paper results
python scripts/scannetpp_iphone_to_artlab_format_dense.py \
  --scene_dir /path/to/scannetpp/data/<scene_id> \
  --output_dir /path/to/output/<scene_id>_iphone_dense
```

### Step 2: Coordinate frame alignment

The conversion scripts apply a rotation to bring iPhone poses into a consistent training frame. The **GT mesh for evaluation** must be rotated manually:

```python
import trimesh, numpy as np

mesh = trimesh.load("mesh_aligned_0.05.ply", process=False)
R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
mesh.vertices = (R @ mesh.vertices.T).T
mesh.export("gt_mesh_training_frame.ply")
```

> **If you skip this**, F-score will be exactly 0.0000.

### Step 3: ICP bridge (optional)

Some ScanNet++ scenes have cm-level registration errors between the Faro scanner and iPhone SLAM. The ICP bridge corrects this:

```bash
python scripts/faro_icp_bridge.py \
  --scene_id <scene_id> \
  --scene_dir /path/to/scannetpp/data/<scene_id> \
  --artlab_dense_dir /path/to/existing_scene_data \
  --output_dir /path/to/corrected_output
```

Use this when initial F-scores are lower than expected (0.5–0.7 instead of 0.85+).

---

## Training Details

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train.py \
  -s scene/ -m output/my_model \
  --images images_masked --white_background \
  --lambda_depth 0.5 --lambda_lidar_normal 0.5 \
  --lambda_dgs 0.1 --lambda_dgs_normal 0.1 \
  --dgs_start_iter 10000 --dgs_interval 500 --dgs_k 8 --dgs_radius 0.05 \
  --las_path scene/pc_aligned.ply \
  --soft_phase_b --phase_b_start 20000 \
  --phase_b_xyz_lr 1e-6 --phase_b_dgs_lambda 0.05 \
  --densify_until_iter 15000 --iterations 30000
```

<details>
<summary><b>Flag reference</b></summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--lambda_depth` | 0.0 | L1 depth supervision weight |
| `--lambda_lidar_normal` | 0.0 | Normal supervision weight |
| `--lambda_dgs` | 0.0 | DGS surfel-to-plane distance weight |
| `--lambda_dgs_normal` | 0.5 | DGS normal alignment weight |
| `--las_path` | — | LiDAR point cloud for DGS (.ply / .las) |
| `--dgs_start_iter` | 10000 | Activate DGS after this iteration |
| `--dgs_interval` | 500 | KNN cache refresh interval |
| `--dgs_k` | 8 | Neighbours for plane fitting |
| `--dgs_radius` | 0.05 | Max KNN radius (m) |
| `--soft_phase_b` | False | Enable GTR phase |
| `--phase_b_start` | 20000 | GTR activation iteration |
| `--phase_b_xyz_lr` | 1e-6 | Reduced position LR during GTR |
| `--phase_b_dgs_lambda` | 0.05 | DGS tether weight during GTR |

</details>

### Tuning `--lambda_depth`

| LiDAR source | `lambda_depth` | Reason |
|-------------|:---:|--------|
| Terrestrial scanner (Faro, Xgrids K1) | **3.0** | Sub-mm registration — strong anchoring is safe |
| iPhone LiDAR (with or without ICP) | **0.5** | cm-level residual noise — lighter anchoring |

---

## Mesh Extraction

```bash
python render.py \
  -s scene/ -m output/my_model --iteration 30000 \
  --skip_train --skip_test \
  --mesh_res 512 --depth_trunc <ROOM_DIAMETER> --num_cluster 50
```

Set `--depth_trunc` to roughly the room diameter in metres (e.g. 5.0 for a 3×4 m room).

**OOM with many cameras (>350):** create a sub-sampled source:

```bash
python scripts/build_stub_source_from_cameras_json.py \
  --cameras_json output/my_model/cameras.json \
  --output_dir /tmp/stub_scene \
  --images_folder images_masked --image_ext png

python render.py \
  -s /tmp/stub_scene -m output/my_model --iteration 30000 \
  --skip_train --skip_test \
  --mesh_res 512 --depth_trunc 5.0 --num_cluster 50
```

---

## Evaluation

```bash
python scripts/eval_mesh_scannetpp.py \
  --pred_mesh output/.../fuse_post.ply \
  --gt_mesh /path/to/gt_mesh_training_frame.ply \
  --transforms_json scene/sparse/0/images.txt \
  --camera_params scene/sparse/0/cameras.txt \
  --threshold 0.05 --clip_to_gt_bbox \
  --output output/.../eval_metrics.json
```

> The GT mesh must be in the training coordinate frame. See [Coordinate Frame Alignment](#step-2-coordinate-frame-alignment).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| F-score = 0.000 | GT mesh not rotated to training frame | Apply rotation — see [Step 2](#step-2-coordinate-frame-alignment) |
| F-score 0.3–0.5 (expected 0.85+) | LiDAR–camera registration error | Run [ICP bridge](#step-3-icp-bridge-optional) and retrain |
| OOM during `train.py` | Too many cameras | Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| OOM during `render.py` | System RAM exhausted (>350 cameras) | Use [stub source](#mesh-extraction) |
| Scene explodes | DGS active during densification | Ensure `--dgs_start_iter` ≥ `--densify_until_iter` |
| PSNR drops >1 dB | Hard position freeze | Use `--soft_phase_b` with `--phase_b_xyz_lr 1e-6` |

---

## Project Structure

```
├── train.py                          # Training (DGS + GTR integrated)
├── render.py                         # Rendering + TSDF mesh extraction
├── metrics.py                        # PSNR / SSIM / LPIPS evaluation
├── utils/
│   ├── direct_geometric_supervision.py   # DGS: KNN plane fitting + loss
│   ├── mesh_utils.py                     # TSDF extraction
│   └── loss_utils.py                     # L1, SSIM
├── scripts/
│   ├── lidar_to_depth_maps.py            # LiDAR → per-camera depth/normal maps
│   ├── scannetpp_iphone_to_artlab_format_dense.py  # ScanNet++ → COLMAP
│   ├── faro_icp_bridge.py                # ICP registration correction
│   ├── eval_mesh_scannetpp.py            # F-score / Chamfer evaluation
│   ├── patch_eval.py                     # Spatial patch-based analysis
│   └── wall_roughness_analysis.py        # RANSAC wall-plane roughness
└── submodules/
    ├── diff-surfel-rasterization/        # 2DGS CUDA rasteriser
    └── simple-knn/                       # KNN for densification
```

---


## Acknowledgements

Built on [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) (Huang et al., SIGGRAPH 2024). Evaluated on [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) (Yeshwanth et al., ICCV 2023).

This work was supported by the Ministry of Electronics and Information Technology (MeitY), Government of India; ARTPARK; and Qualcomm.
