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

## How It Works

Training runs in three phases on top of vanilla 2DGS:

| Phase | Iterations | What happens |
|-------|-----------|--------------|
| **Densification** | 0 – 15 k | Standard 2DGS + L1 depth supervision from LiDAR depth maps |
| **DGS** | 10 k – 20 k | Per-surfel KNN plane fit to LiDAR; penalises signed distance and normal misalignment — gradient bypasses alpha-compositing dilution |
| **GTR** | 20 k – 30 k | Position LR drops 160×; DGS acts as a spring tether (λ = 0.05); photometric loss provides a counter-spring — surfels settle ~0.5 mm from the LiDAR surface |

An optional **ICP bridge** step corrects cm-level LiDAR↔camera registration errors using iPhone sensor depth as an intermediary (see `scripts/faro_icp_bridge.py`).

## Installation

```bash
git clone --recursive <this-repo> && cd lidar-2dgs
conda create -n surfel_splatting python=3.8 && conda activate surfel_splatting

# PyTorch — adjust the index URL for your CUDA version
pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu124

pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn
pip install plyfile tqdm scipy opencv-python open3d trimesh scikit-learn
```

## Data Layout

```
scene/
├── sparse/0/
│   ├── cameras.txt            # COLMAP PINHOLE camera
│   ├── images.txt             # COLMAP poses
│   └── points3D.ply           # Voxel-downsampled LiDAR (Gaussian init)
├── images_masked/camera_0/    # RGB frames
├── depth_maps/camera_0/       # Per-frame .npy float32, metres, 0 = no data
├── normal_maps/camera_0/      # Per-frame .npy float32, world-space normals
└── pc_aligned_artlab_frame.ply  # Full-res LiDAR for DGS
```

Generate depth/normal maps from a LiDAR point cloud:

```bash
python scripts/lidar_to_depth_maps.py \
  --las scene/pc_aligned_artlab_frame.ply \
  --source scene/ --voxel_size 0.002 --mask_folder images_masked
```

## Training

```bash
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

**Tuning `--lambda_depth`:** use **0.5** for iPhone-quality depth (ScanNet++); raise to **3.0** for tightly-registered terrestrial scanners (< 1 mm error).

</details>

## Mesh Extraction

```bash
python render.py \
  -s scene/ -m output/my_model --iteration 30000 \
  --skip_train --skip_test \
  --mesh_res 512 --depth_trunc <ROOM_DIAMETER> --num_cluster 50
```

Set `--depth_trunc` to roughly the **room diameter in metres** (e.g. 3.0 for a 3 × 4 m room, 6.0 for a 6 × 8 m room). Too large → TSDF noise; too small → clipped geometry.

Output: `output/my_model/train/ours_30000/fuse_post.ply`

> **OOM tip:** for scenes with > 350 cameras, create a sub-sampled source directory for `render.py` (every 5th frame) using `scripts/build_stub_source_from_cameras_json.py`.

## Evaluation

```bash
# Standard metrics (F-score, Chamfer, Normal Consistency)
python scripts/eval_mesh_scannetpp.py \
  --pred_mesh output/.../fuse_post.ply \
  --gt_mesh /path/to/gt.ply \
  --transforms_json scene/sparse/0/images.txt \
  --camera_params scene/sparse/0/cameras.txt \
  --threshold 0.05 --clip_to_gt_bbox

# Patch-based analysis (100 spatial patches, multi-threshold F-scores, Q-Poor)
python scripts/patch_eval.py \
  --gt_mesh /path/to/gt.ply \
  --pred_mesh output/.../fuse_post.ply \
  --output output/.../patch_eval/ \
  --n_patches 100 --samples 200000 --clip_to_gt_bbox
```

## ICP Bridge (ScanNet++ iPhone ↔ Faro)

Corrects systematic registration errors between a terrestrial LiDAR scanner and iPhone SLAM poses by using the iPhone's own LiDAR as an alignment intermediary:

```bash
python scripts/faro_icp_bridge.py \
  --scene_id <name> \
  --scene_dir /path/to/scannetpp/scene \
  --artlab_dense_dir /path/to/existing_artlab_data \
  --output_dir /path/to/corrected_output
```

See [`docs/icp_bridge.md`](docs/icp_bridge.md) for details.

## Citation

```bibtex
@inproceedings{lidar2dgs2026,
  title     = {LiDAR-Constrained 2D Gaussian Splatting for Indoor Digital Twins},
  author    = {TODO},
  booktitle = {SIGGRAPH Asia Technical Communications},
  year      = {2026}
}
```

## Acknowledgements

Built on [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) (Huang et al., SIGGRAPH 2024). Evaluated on [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) (Yeshwanth et al., ICCV 2023).