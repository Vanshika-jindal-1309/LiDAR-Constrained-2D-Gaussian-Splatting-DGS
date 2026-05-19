# ICP Bridge: LiDAR-Camera Registration Correction

The ICP bridge corrects registration errors between a high-quality LiDAR scanner (e.g., Faro terrestrial scanner) and an iPhone camera rig. It uses iPhone's built-in LiDAR depth sensor as an alignment intermediary — iPhone depth is perfectly aligned to iPhone cameras by construction.

## When to Use

Use the ICP bridge when:
- Your training images come from iPhone (or other consumer camera with built-in LiDAR)
- Your ground-truth LiDAR comes from a separate high-quality scanner (Faro, Leica, Riegl)
- `verify_lidar_camera_alignment.py` shows systematic misalignment > 10–20mm

Symptoms of registration error: rendered depth supervision pulls Gaussians to wrong positions, PSNR is lower than expected, surfel positions don't match visual surfaces.

## How It Works

```
1. Accumulate iPhone LiDAR PC
   Read iPhone depth.bin frames → project to 3D using iPhone intrinsics/poses
   → sparse but well-aligned iPhone point cloud

2. ICP align Faro PC to iPhone PC
   Initial alignment: identity (assumes same coordinate frame after conversion)
   Open3D point-to-plane ICP → correction transform T_correction

3. Apply correction
   Transform full-resolution Faro PC: pts_corrected = T_correction @ pts_faro
   Save corrected PC as pc_aligned_icp.ply

4. Regenerate depth maps
   Project corrected Faro PC through each camera → new depth_maps/ and normal_maps/
   with reduced registration error
```

## Results

On the ScanNet++ inside-room scenes we evaluated:

| Scene | Pre-ICP Registration Error | ICP Correction | F@5cm (ICP-corrected) |
|-------|--------------------------|----------------|----------------------|
| 56a0ec536c | ~9mm | 9.18mm correction | 0.837 |
| e8ea9b4da8 | ~16mm | 15.68mm correction | 0.893 |

## Usage

```bash
python scripts/faro_icp_bridge.py \
  --scene_id <scene_name> \
  --scene_dir /path/to/raw/scannetpp/scene \
  --artlab_dense_dir /path/to/existing/artlab_dense_scene \
  --output_dir /path/to/icp_corrected_output \
  [--depth_stride 10] \
  [--cloud_voxel 0.005] \
  [--icp_voxel 0.02] \
  [--dm_voxel 0.002] \
  [--icp_max_iter 200]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--scene_id` | required | Scene identifier (e.g., `56a0ec536c`) |
| `--scene_dir` | required | Path to raw ScanNet++ scene (contains `iphone/` subdir with `depth.bin`, `poses.txt`, `intrinsics.txt`) |
| `--artlab_dense_dir` | required | Path to existing dense artlab-format directory (contains `pc_aligned_artlab_frame.ply` and trained depth maps) |
| `--output_dir` | required | Output directory for ICP-corrected data |
| `--depth_stride` | 10 | Use every N-th iPhone depth frame for accumulation. Smaller = denser iPhone PC but slower. |
| `--cloud_voxel` | 0.005 | Voxel size (m) for downsampling the accumulated iPhone cloud. 5mm recommended. |
| `--icp_voxel` | 0.02 | Voxel size (m) for ICP registration. 20mm recommended. |
| `--dm_voxel` | 0.002 | Voxel size (m) for regenerating depth maps from corrected Faro PC. 2mm recommended. |
| `--icp_max_iter` | 200 | Maximum ICP iterations. |

### Input Requirements

The `scene_dir` must contain:
```
scene_dir/
└── iphone/
    ├── depth.bin           # Raw iPhone LiDAR (uint16, 256×192 frames, concatenated)
    ├── poses.txt           # Per-frame IMU poses (4×4 matrices)
    └── intrinsics.txt      # iPhone camera intrinsics (4×4 K matrix)
```

The `artlab_dense_dir` must contain:
```
artlab_dense_dir/
├── pc_aligned_artlab_frame.ply   # Faro PC in iPhone training coordinate frame
└── sparse/0/
    ├── cameras.txt
    └── images.txt
```

### Output Structure

```
output_dir/
├── pc_aligned_icp.ply              # ICP-corrected Faro PC
├── icp_transform.json              # 4×4 correction transform + metadata
├── depth_maps/camera_0/*.npy       # Regenerated depth maps from corrected PC
├── normal_maps/camera_0/*.npy      # Regenerated normal maps
└── [other files symlinked from artlab_dense_dir]
```

## After ICP Correction

Train as usual, pointing `--las_path` to the ICP-corrected PC:

```bash
python train.py \
  -s /path/to/output_dir \
  -m output/my_icp_model \
  --images images_masked --white_background \
  --lambda_depth 0.5 --lambda_lidar_normal 0.5 \
  --lambda_dgs 0.1 --lambda_dgs_normal 0.1 \
  --las_path /path/to/output_dir/pc_aligned_icp.ply \
  --dgs_start_iter 10000 \
  --gtr --gtr_start 20000 --gtr_xyz_lr 1e-6 --gtr_dgs_lambda 0.05 \
  --densify_until_iter 15000 --iterations 30000
```

## Coordinate Frame Notes

ScanNet++ iPhone scenes use a specific coordinate transform chain:

```python
T_CAM_FLIP = np.diag([1., -1., -1., 1.])   # OpenGL → COLMAP local axes
R_GLOBAL   = np.diag([1.0, -1.0, -1.0])    # Rx(180°), det=+1 (proper rotation)
T_GLOBAL   = np.eye(4); T_GLOBAL[:3,:3] = R_GLOBAL
T_WORLD    = [[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]]  # IMU → world swap
T_IMU_TO_C2W = T_GLOBAL @ T_WORLD  # R_PC = [[0,1,0],[-1,0,0],[0,0,1]]
```

The same `T_IMU_TO_C2W` transform is applied to the LiDAR PC so it lives in the same frame as the camera poses.

**For GT mesh evaluation**: the GT mesh must be rotated into the iPhone training frame before evaluation:
```python
R_iphone = np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
verts_eval = verts_scan @ R_iphone.T
```

Do NOT use the DSLR rotation (`R_GLOBAL = diag([1,-1,-1])`) for inside-room iPhone scenes — they use different coordinate conventions.
