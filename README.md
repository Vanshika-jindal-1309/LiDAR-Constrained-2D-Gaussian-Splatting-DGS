# LiDAR-Constrained 2D Gaussian Splatting

Extending [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) with **Direct Geometric Supervision (DGS)** and **Geometry-Tethered Refinement (GTR)** for mm-accurate indoor mesh reconstruction from LiDAR + RGB.

## Results

**ScanNet++ iPhone indoor scenes** (F-score @ 5 cm, visibility-culled):

| Scene | Vanilla 2DGS | Ours | Δ |
|-------|:-----------:|:----:|:-:|
| e8ea9b4da8 | 0.438 | **0.907** | +107 % |
| 56a0ec536c | 0.491 | **0.845** | +72 % |
| be0ed6b33c | 0.413 | **0.880** | +113 % |
| 88cf747085 | 0.511 | **0.893** | +75 % |
| **Average** | 0.463 | **0.881** | **+90 %** |

Published baselines on ScanNet++: DN-Splatter ≈ 0.55–0.65 · PGSR ≈ 0.60–0.70 · 2DGS-Room ≈ 0.65–0.72.

**Xgrids ARTLab indoor scene** (building-scale, 26 M LiDAR points, 1,992 images):

| Metric | Photometric only | Depth supervised | + DGS | + GTR (final) |
|--------|:---:|:---:|:---:|:---:|
| Wall RMS (mm) | 14.46 | 9.79 | 8.70 | **8.29** |
| Chamfer (mm) | 369.8 | 265.6 | 255.5 | **247.4** |
| F@10 mm | 0.052 | 0.204 | 0.205 | **0.208** |
| PSNR (dB) | — | 13.24 | 12.96 | **13.35** |

---

## How It Works

Training runs in three phases on top of vanilla 2DGS:

| Phase | Iterations | What happens |
|-------|-----------|--------------|
| **Densification** | 0 – 15 k | Standard 2DGS + L1 depth supervision from LiDAR depth maps |
| **DGS** | 10 k – 20 k | Per-surfel KNN plane fit to LiDAR; penalises signed distance and normal misalignment — gradient bypasses alpha-compositing dilution |
| **GTR** | 20 k – 30 k | Position LR drops 160×; DGS acts as a spring tether (λ = 0.05); photometric loss provides a counter-spring — surfels settle ~0.5 mm from the LiDAR surface |

---

## Installation

```bash
git clone --recursive https://github.com/<your-org>/LiDAR-Constrained-2DGS.git
cd LiDAR-Constrained-2DGS

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

If you already have a scene directory in the [expected layout](#data-layout), the full pipeline is three commands:

```bash
# 1. Train
python train.py \
  -s scene/ -m output/my_model \
  --images images_masked --white_background \
  --lambda_depth 0.5 --lambda_lidar_normal 0.5 \
  --lambda_dgs 0.1 --lambda_dgs_normal 0.1 \
  --dgs_start_iter 10000 --dgs_interval 500 --dgs_k 8 --dgs_radius 0.05 \
  --las_path scene/pc_aligned_artlab_frame.ply \
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
└── pc_aligned_artlab_frame.ply  # Full-resolution LiDAR point cloud for DGS
```

**Generating depth and normal maps** from a LiDAR point cloud:

```bash
python scripts/lidar_to_depth_maps.py \
  --las scene/pc_aligned_artlab_frame.ply \
  --source scene/ --voxel_size 0.002 --mask_folder images_masked
```

---

## ScanNet++ Data Preparation

ScanNet++ scenes require coordinate-frame alignment between the Faro scanner, the iPhone SLAM poses, and the GT evaluation mesh.

### Step 1: Convert to COLMAP Format

**iPhone captures** (used for all results in this paper):

```bash
# Dense (stride=5, ~1267 frames) — recommended
python scripts/scannetpp_iphone_to_artlab_format_dense.py \
  --scene_dir /path/to/scannetpp/data/<scene_id> \
  --output_dir /path/to/output/<scene_id>_iphone_dense

# Sparse (stride=10, ~324 frames)
python scripts/scannetpp_iphone_to_artlab_format.py \
  --scene_dir /path/to/scannetpp/data/<scene_id> \
  --output_dir /path/to/output/<scene_id>_iphone
```

**DSLR captures** (different input format, uses `transforms_undistorted.json`):

```bash
python scripts/scannetpp_to_artlab_format.py \
  --scene_dir /path/to/scannetpp/data/<scene_id> \
  --output_dir /path/to/output/<scene_id>_dslr
```

### Step 2: Coordinate Frame Alignment

The conversion scripts apply a **swap_XY_negY** rotation to bring iPhone inside-room poses into a consistent training frame:

```python
# x' = y,  y' = -x,  z' = z
R = np.array([[0, 1, 0],
              [-1, 0, 0],
              [0, 0, 1]])
```

This is applied to camera poses and the LiDAR point cloud automatically. However, the **GT mesh for evaluation** must be rotated manually since it lives outside the conversion pipeline:

```python
import trimesh, numpy as np

mesh = trimesh.load("mesh_aligned_0.05.ply", process=False)
R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
mesh.vertices = (R @ mesh.vertices.T).T
mesh.export("gt_mesh_training_frame.ply")
```

> **If you skip this**, the eval script will print `Bboxes overlap: NO` and F-score will be exactly 0.0000.

**Verify alignment:**

```python
import trimesh
pred = trimesh.load("output/.../fuse_post.ply", process=False)
gt = trimesh.load("gt_mesh_training_frame.ply", process=False)
print(f"Pred Y: [{pred.bounds[0,1]:.2f}, {pred.bounds[1,1]:.2f}]")
print(f"GT   Y: [{gt.bounds[0,1]:.2f}, {gt.bounds[1,1]:.2f}]")
# These ranges should overlap substantially
```

### Step 3: ICP Bridge (Optional)

Some ScanNet++ scenes have cm-level registration errors between the Faro scanner and iPhone SLAM. The ICP bridge corrects this using the iPhone's own LiDAR as an intermediary:

```bash
python scripts/faro_icp_bridge.py \
  --scene_id <scene_id> \
  --scene_dir /path/to/scannetpp/data/<scene_id> \
  --artlab_dense_dir /path/to/existing_scene_data \
  --output_dir /path/to/corrected_output
```

See [`docs/icp_bridge.md`](docs/icp_bridge.md) for details. Use this when initial F-scores are lower than expected (0.5–0.7 instead of 0.85+).

---

## Training

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train.py \
  -s scene/ -m output/my_model \
  --images images_masked --white_background \
  --lambda_depth 0.5 --lambda_lidar_normal 0.5 \
  --lambda_dgs 0.1 --lambda_dgs_normal 0.1 \
  --dgs_start_iter 10000 --dgs_interval 500 --dgs_k 8 --dgs_radius 0.05 \
  --las_path scene/pc_aligned_artlab_frame.ply \
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
| `--lambda_concentrate` | 0.0 | Opacity concentration (experimental — keep 0) |
| `--soft_phase_b` | False | Enable GTR phase |
| `--phase_b_start` | 20000 | GTR activation iteration |
| `--phase_b_xyz_lr` | 1e-6 | Reduced position LR during GTR |
| `--phase_b_dgs_lambda` | 0.05 | DGS tether weight during GTR |

</details>

### Tuning `--lambda_depth`

| LiDAR source | `lambda_depth` | Reason |
|-------------|:---:|--------|
| Terrestrial scanner (Faro, Xgrids) | **3.0** | Sub-mm registration — strong anchoring is safe |
| iPhone LiDAR (with or without ICP) | **0.5** | cm-level residual noise — lighter anchoring accommodates errors |

---

## Mesh Extraction

```bash
python render.py \
  -s scene/ -m output/my_model --iteration 30000 \
  --skip_train --skip_test \
  --mesh_res 512 --depth_trunc <ROOM_DIAMETER> --num_cluster 50
```

Set `--depth_trunc` to roughly the room diameter in metres (e.g. 5.0 for a 3 × 4 m room, 10.0 for a large hall). Output: `output/my_model/train/ours_30000/fuse_post.ply`

**OOM with many cameras:** `render.py` loads all cameras into RAM for TSDF fusion. For scenes with > 350 cameras, create a sub-sampled source:

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
# F-score, Chamfer, Normal Consistency (visibility-culled)
python scripts/eval_mesh_scannetpp.py \
  --pred_mesh output/.../fuse_post.ply \
  --gt_mesh /path/to/gt_mesh_training_frame.ply \
  --transforms_json scene/sparse/0/images.txt \
  --camera_params scene/sparse/0/cameras.txt \
  --threshold 0.05 --clip_to_gt_bbox \
  --output output/.../eval_metrics.json

# Patch-based spatial analysis
python scripts/patch_eval.py \
  --gt_mesh /path/to/gt_mesh_training_frame.ply \
  --pred_mesh output/.../fuse_post.ply \
  --output output/.../patch_eval/ \
  --n_patches 100 --samples 200000 --clip_to_gt_bbox
```

> **Important:** the GT mesh must be in the training coordinate frame. See [Coordinate Frame Alignment](#step-2-coordinate-frame-alignment).

---

## Troubleshooting for scannet++ Dataset

| Symptom | Cause | Fix |
|---------|-------|-----|
| F-score = 0.000 | GT mesh not rotated to training frame | Apply `swap_XY_negY` rotation — see [Step 2](#step-2-coordinate-frame-alignment) |
| F-score 0.3–0.5 (expected 0.85+) | LiDAR↔camera registration error | Run [ICP bridge](#step-3-icp-bridge-optional) and retrain |
| OOM during `train.py` | Too many cameras on GPU | Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; reduce `--densify_until_iter` to 10000 |
| Killed during `render.py` | System RAM exhausted (> 350 cameras) | Use [stub source](#oom-with-many-cameras) |
| Scene explodes (surfel→LiDAR > 100 mm) | DGS active during densification | Ensure `--dgs_start_iter` ≥ `--densify_until_iter` |
| PSNR drops > 1 dB vs vanilla | Hard Phase B freeze | Use `--soft_phase_b` with `--phase_b_xyz_lr 1e-6` |

---

## Project Structure

```
├── train.py                          # Training (DGS + GTR integrated)
├── render.py                         # Rendering + TSDF mesh extraction
├── metrics.py                        # PSNR / SSIM / LPIPS evaluation
│
├── arguments/                        # CLI argument definitions
├── gaussian_renderer/                # Differentiable surfel rasteriser interface
├── scene/                            # Scene loading, cameras, Gaussian model
├── lpipsPyTorch/                     # Perceptual loss (LPIPS)
│
├── utils/
│   ├── direct_geometric_supervision.py   # DGS: KNN plane fitting + loss
│   ├── mesh_utils.py                     # TSDF extraction (GaussianExtractor)
│   ├── loss_utils.py                     # L1, SSIM
│   └── ...                               # Camera, graphics, SH utilities
│
├── scripts/
│   ├── lidar_to_depth_maps.py            # LiDAR → per-camera depth/normal maps
│   ├── scannetpp_iphone_to_artlab_format_dense.py  # ScanNet++ iPhone → COLMAP
│   ├── scannetpp_iphone_to_artlab_format.py        # ScanNet++ iPhone (sparse)
│   ├── scannetpp_to_artlab_format.py               # ScanNet++ DSLR → COLMAP
│   ├── faro_icp_bridge.py                # ICP registration correction
│   ├── extract_iphone_depth_maps.py      # iPhone native depth extraction
│   ├── eval_mesh_scannetpp.py            # F-score / Chamfer evaluation
│   ├── patch_eval.py                     # Spatial patch-based analysis
│   ├── wall_roughness_analysis.py        # RANSAC wall-plane roughness
│   ├── build_stub_source_from_cameras_json.py  # Camera subsampling (OOM fix)
│   ├── verify_lidar_camera_alignment.py  # LiDAR-camera overlay diagnostic
│   ├── jpg_to_masked_png.py              # JPG → RGBA PNG (black border masking)
│   └── las_to_ply.py                     # LAS/LAZ → PLY conversion
│
├── submodules/
│   ├── diff-surfel-rasterization/        # 2DGS CUDA rasteriser
│   └── simple-knn/                       # KNN for densification
│
└── docs/
    └── icp_bridge.md                     # ICP bridge documentation
```

---

## Citation

```bibtex
@mastersthesis{jindal2026lidar2dgs,
  title   = {LiDAR-Constrained 2D Gaussian Surfels with Direct Geometric
             Supervision for Indoor Digital Twin Reconstruction},
  author  = {Jindal, Vanshika},
  school  = {Indian Institute of Science, Bengaluru},
  year    = {2026},
  type    = {M.Tech Thesis}
}
```

## Acknowledgements

Built on [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) (Huang et al., SIGGRAPH 2024). Evaluated on [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) (Yeshwanth et al., ICCV 2023).

This work was supported by the Ministry of Electronics and Information Technology (MeitY), Government of India; ARTLabs; and the Autonomous Machines Lab (AML), Indian Institute of Science, Bengaluru.
