#!/usr/bin/env python3
"""
Faro-to-iPhone ICP bridge alignment.

Uses iPhone LiDAR depth.bin as an intermediary to ICP-align the Faro
scanner PC to the iPhone's coordinate frame. The Faro PC has cm-level
registration errors; iPhone LiDAR is perfectly aligned to iPhone cameras
by construction.

Pipeline:
  1. Accumulate iPhone LiDAR PC from depth.bin (every --stride-th frame)
  2. ICP-align Faro pc_aligned_artlab_frame.ply to iPhone LiDAR PC
  3. Apply correction transform to full-res Faro PC
  4. Save corrected Faro PC and ICP metadata
  5. Create new artlab-format dir with corrected PC + regenerated depth maps

Usage:
  python scripts/faro_icp_bridge.py \\
    --scene_id <scene_id> \\
    --scene_dir /path/to/scannetpp/data/<scene_id> \\
    --artlab_dense_dir /path/to/artlab_format/<scene_id>_iphone_dense \\
    --output_dir /path/to/artlab_format/<scene_id>_iphone_icp2 \\
    [--depth_stride 10] [--icp_voxel 0.02] [--icp_max_iter 200]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

# ──────────────────────────────────────────────────────────────────────────────
# depth.bin constants
# ──────────────────────────────────────────────────────────────────────────────
DEPTH_W, DEPTH_H = 256, 192
DEPTH_BYTES_PER_FRAME = DEPTH_W * DEPTH_H * 2  # uint16 = 2 bytes


def read_depth_frame(depth_bin_path: str, frame_idx: int) -> np.ndarray:
    with open(depth_bin_path, 'rb') as f:
        f.seek(frame_idx * DEPTH_BYTES_PER_FRAME)
        raw = f.read(DEPTH_BYTES_PER_FRAME)
    if len(raw) < DEPTH_BYTES_PER_FRAME:
        return np.zeros((DEPTH_H, DEPTH_W), dtype=np.uint16)
    return np.frombuffer(raw, dtype=np.uint16).reshape(DEPTH_H, DEPTH_W).copy()


def upsample_nearest(arr_small: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h_s, w_s = arr_small.shape
    rows = np.clip((np.arange(out_h) * h_s / out_h).astype(np.int32), 0, h_s - 1)
    cols = np.clip((np.arange(out_w) * w_s / out_w).astype(np.int32), 0, w_s - 1)
    return arr_small[rows[:, None], cols[None, :]]


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate helpers  (same as scannetpp_iphone_to_artlab_format_dense.py)
# ──────────────────────────────────────────────────────────────────────────────
T_CAM_FLIP   = np.diag([1., -1., -1., 1.])
R_GLOBAL     = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
T_GLOBAL     = np.eye(4, dtype=np.float64)
T_GLOBAL[:3, :3] = R_GLOBAL
T_WORLD      = np.array([[0., 1., 0., 0.],
                          [1., 0., 0., 0.],
                          [0., 0., -1., 0.],
                          [0., 0., 0., 1.]])
T_IMU_TO_C2W = T_GLOBAL @ T_WORLD   # R_PC = [[0,1,0],[-1,0,0],[0,0,1]]


def imu_aligned_pose_to_colmap(T_imu: np.ndarray):
    """aligned_pose (4×4) → COLMAP R_w2c (3×3), t_w2c (3,)."""
    T_c2w = T_IMU_TO_C2W @ T_imu
    T_w2c = np.linalg.inv(T_c2w)
    return T_w2c[:3, :3].astype(np.float32), T_w2c[:3, 3].astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – Accumulate iPhone LiDAR PC
# ──────────────────────────────────────────────────────────────────────────────

def accumulate_iphone_lidar(depth_bin: Path,
                             pose_json: Path,
                             fx: float, fy: float,
                             cx: float, cy: float,
                             W: int, H: int,
                             depth_stride: int = 10,
                             min_depth_m: float = 0.3,
                             max_depth_m: float = 7.0,
                             cloud_voxel: float = 0.005) -> np.ndarray:
    """
    Read every depth_stride-th frame from depth.bin, match to nearest aligned_pose,
    unproject to world space, and return a voxel-downsampled (N,3) float32 array.
    """
    print(f"\n[Step 1] Accumulating iPhone LiDAR PC from depth.bin ...")
    t0 = time.time()

    with open(pose_json) as f:
        imu_data = json.load(f)
    pose_keys = sorted(imu_data.keys(), key=lambda k: int(k.split('_')[-1]))
    timestamps = np.array([imu_data[k]['timestamp'] for k in pose_keys], dtype=np.float64)
    T_start, T_end = timestamps[0], timestamps[-1]

    fsize = os.path.getsize(depth_bin)
    total_depth_frames = fsize // DEPTH_BYTES_PER_FRAME
    depth_ts = np.linspace(T_start, T_end, total_depth_frames)
    print(f"  depth.bin: {total_depth_frames} frames ({total_depth_frames/(T_end-T_start):.1f} fps), "
          f"loading every {depth_stride}th → {total_depth_frames//depth_stride} frames")

    min_mm = min_depth_m * 1000.0
    max_mm = max_depth_m * 1000.0

    all_pts = []
    depth_bin_str = str(depth_bin)

    for i in range(0, total_depth_frames, depth_stride):
        # Find nearest RGB/IMU pose
        ts_i = depth_ts[i]
        nearest_pose_idx = int(np.argmin(np.abs(timestamps - ts_i)))
        key = pose_keys[nearest_pose_idx]
        T_imu = np.array(imu_data[key]['aligned_pose'], dtype=np.float64)
        R_w2c, t_w2c = imu_aligned_pose_to_colmap(T_imu)

        # Read depth frame (native 256×192)
        depth_mm = read_depth_frame(depth_bin_str, i)

        # Convert to metres, filter range
        depth_f = depth_mm.astype(np.float32)
        valid_sm = (depth_mm > 0) & (depth_mm < 65500) & \
                   (depth_f >= min_mm) & (depth_f <= max_mm)
        if valid_sm.sum() < 10:
            continue

        # Upsample to RGB resolution so we can use RGB intrinsics
        depth_small = np.zeros((DEPTH_H, DEPTH_W), dtype=np.float32)
        depth_small[valid_sm] = depth_f[valid_sm] / 1000.0
        depth_full = upsample_nearest(depth_small, H, W)

        # Subsample pixels (every 8th pixel = 1/64 of pixels)
        sub = 8
        us = np.arange(0, W, sub)
        vs = np.arange(0, H, sub)
        ug, vg = np.meshgrid(us, vs)
        d = depth_full[vg, ug]
        valid = d > 0
        if valid.sum() < 5:
            continue

        ug_v = ug[valid].astype(np.float32)
        vg_v = vg[valid].astype(np.float32)
        d_v  = d[valid]

        Xc = (ug_v - cx) / fx * d_v
        Yc = (vg_v - cy) / fy * d_v
        Zc = d_v
        P_cam = np.stack([Xc, Yc, Zc], axis=-1)

        R_c2w = R_w2c.T.astype(np.float64)
        P_world = (R_c2w @ (P_cam.astype(np.float64) - t_w2c.astype(np.float64)).T).T
        all_pts.append(P_world.astype(np.float32))

    if not all_pts:
        raise RuntimeError("No iPhone LiDAR points accumulated — check depth.bin and poses")

    raw_pts = np.concatenate(all_pts, axis=0)
    print(f"  Raw pts: {len(raw_pts):,} (accumulated in {time.time()-t0:.1f}s)")
    print(f"  Raw Z range: [{raw_pts[:,2].min():.2f}, {raw_pts[:,2].max():.2f}]")
    print(f"  Raw X range: [{raw_pts[:,0].min():.2f}, {raw_pts[:,0].max():.2f}]")
    print(f"  Raw Y range: [{raw_pts[:,1].min():.2f}, {raw_pts[:,1].max():.2f}]")

    # Voxel downsample
    pcd_raw = o3d.geometry.PointCloud()
    pcd_raw.points = o3d.utility.Vector3dVector(raw_pts.astype(np.float64))
    pcd_ds  = pcd_raw.voxel_down_sample(cloud_voxel)
    pts_ds  = np.asarray(pcd_ds.points).astype(np.float32)
    print(f"  After {cloud_voxel*1000:.0f}mm voxel: {len(pts_ds):,} pts")
    return pts_ds


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – ICP alignment
# ──────────────────────────────────────────────────────────────────────────────

def icp_align_faro_to_iphone(faro_pts: np.ndarray,
                               iphone_pts: np.ndarray,
                               icp_voxel: float = 0.02,
                               max_iter: int = 200,
                               max_corr_dist: float = 0.05) -> dict:
    """
    Point-to-plane ICP: align Faro PC to iPhone LiDAR PC.
    Returns dict with: transform (4×4), fitness, inlier_rmse, translation_m, rotation_deg.
    """
    print(f"\n[Step 2] ICP alignment (Faro → iPhone LiDAR) ...")
    print(f"  Faro input:   {len(faro_pts):,} pts")
    print(f"  iPhone input: {len(iphone_pts):,} pts")

    def make_pcd(pts):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        return pcd

    # Downsample both to icp_voxel
    faro_pcd  = make_pcd(faro_pts).voxel_down_sample(icp_voxel)
    phone_pcd = make_pcd(iphone_pts).voxel_down_sample(icp_voxel)
    print(f"  After {icp_voxel*100:.0f}cm voxel:  Faro {len(faro_pcd.points):,}  iPhone {len(phone_pcd.points):,}")

    # Estimate normals
    for pcd in [faro_pcd, phone_pcd]:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    # Coarse print — sanity check
    faro_arr  = np.asarray(faro_pcd.points)
    phone_arr = np.asarray(phone_pcd.points)
    print(f"  Faro centroid:  [{faro_arr.mean(0)[0]:.3f}, {faro_arr.mean(0)[1]:.3f}, {faro_arr.mean(0)[2]:.3f}]")
    print(f"  iPhone centroid:[{phone_arr.mean(0)[0]:.3f}, {phone_arr.mean(0)[1]:.3f}, {phone_arr.mean(0)[2]:.3f}]")

    t0 = time.time()
    reg = o3d.pipelines.registration.registration_icp(
        faro_pcd, phone_pcd,
        max_corr_dist,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter,
            relative_fitness=1e-7,
            relative_rmse=1e-7,
        )
    )
    elapsed = time.time() - t0

    T = reg.transformation
    R = T[:3, :3]
    t = T[:3, 3]
    # Rotation angle from trace: cos(θ) = (tr(R)-1)/2
    trace = np.trace(R)
    angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    trans_mm  = np.linalg.norm(t) * 1000.0

    print(f"\n  ICP result ({elapsed:.1f}s):")
    print(f"    fitness:      {reg.fitness:.4f}  (>0.8 is good)")
    print(f"    inlier RMSE:  {reg.inlier_rmse*1000:.2f} mm  (<20mm is good)")
    print(f"    translation:  {trans_mm:.2f} mm  (<50mm expected)")
    print(f"    rotation:     {angle_deg:.3f}°  (<5° expected)")
    print(f"    transform:\n{T}")

    # Hard stops
    if reg.fitness < 0.5:
        raise RuntimeError(
            f"ICP fitness {reg.fitness:.3f} < 0.5 — coordinate frames incompatible. STOP.")
    if angle_deg > 10.0:
        raise RuntimeError(
            f"ICP rotation {angle_deg:.1f}° > 10° — fundamental frame mismatch. STOP.")

    return {
        'transform':    T,
        'fitness':      float(reg.fitness),
        'inlier_rmse_mm': float(reg.inlier_rmse * 1000),
        'translation_mm': float(trans_mm),
        'rotation_deg':   float(angle_deg),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 – Create corrected artlab dir
# ──────────────────────────────────────────────────────────────────────────────

def create_corrected_artlab_dir(base_dir: Path,
                                 output_dir: Path,
                                 faro_pts_corrected: np.ndarray,
                                 icp_meta: dict):
    """
    Copy base artlab structure, replace pc_aligned_artlab_frame.ply +
    sparse/0/points3D.ply with ICP-corrected Faro PC.
    depth_maps/ and normal_maps/ are intentionally NOT copied (will be
    regenerated by lidar_to_depth_maps.py).
    """
    print(f"\n[Step 3] Creating corrected artlab dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    skip = {'depth_maps', 'normal_maps', 'pc_aligned_artlab_frame.ply',
            'pc_aligned_artlab_frame.ply.bak_dslr_rotation'}

    for item in sorted(base_dir.iterdir()):
        if item.name in skip:
            continue
        dst = output_dir / item.name
        if dst.exists():
            print(f"  skip (exists): {item.name}")
            continue
        if item.is_dir():
            shutil.copytree(str(item), str(dst), symlinks=True)
            print(f"  copied dir:  {item.name}/")
        else:
            shutil.copy2(str(item), str(dst))
            print(f"  copied file: {item.name}")

    # Write ICP-corrected full-res Faro PC
    pc_out = output_dir / 'pc_aligned_artlab_frame.ply'
    _write_ply_with_normals(faro_pts_corrected, pc_out)
    print(f"  Wrote corrected Faro PC ({len(faro_pts_corrected):,} pts) → {pc_out.name}")

    # Replace sparse/0/points3D.ply (init PC for Gaussians)
    p3d_out = output_dir / 'sparse' / '0' / 'points3D.ply'
    _write_init_pc_5cm(faro_pts_corrected, p3d_out)
    print(f"  Wrote init PC (5cm voxel) → {p3d_out}")

    # Save ICP metadata
    import json as _json
    meta_out = output_dir / 'icp_alignment_meta.json'
    meta_save = {k: v if not isinstance(v, np.ndarray) else v.tolist()
                 for k, v in icp_meta.items()}
    with open(meta_out, 'w') as f:
        _json.dump(meta_save, f, indent=2)
    print(f"  Saved ICP meta → {meta_out.name}")


def _write_ply_with_normals(pts: np.ndarray, out_path: Path):
    """Write (N,3) float32 points as PLY with zero normals."""
    from plyfile import PlyData, PlyElement
    # Estimate normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.orient_normals_towards_camera_location(pts.mean(axis=0).astype(np.float64))
    nrm = np.asarray(pcd.normals).astype(np.float32)

    vertex = np.zeros(len(pts), dtype=[
        ('x','f4'),('y','f4'),('z','f4'),
        ('nx','f4'),('ny','f4'),('nz','f4'),
    ])
    for i, ax in enumerate('xyz'):
        vertex[ax] = pts[:, i]
    for i, ax in enumerate(['nx','ny','nz']):
        vertex[ax] = nrm[:, i]
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(str(out_path))


def _write_init_pc_5cm(pts: np.ndarray, out_path: Path):
    """Voxel-downsample to 5cm and write PLY with normals (required by fetchPly)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd_ds = pcd.voxel_down_sample(0.05)
    pts_ds = np.asarray(pcd_ds.points).astype(np.float32)

    pcd_ds.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    pcd_ds.orient_normals_towards_camera_location(pts_ds.mean(axis=0).astype(np.float64))
    nrm_ds = np.asarray(pcd_ds.normals).astype(np.float32)

    from plyfile import PlyData, PlyElement
    vertex = np.zeros(len(pts_ds), dtype=[
        ('x','f4'),('y','f4'),('z','f4'),
        ('nx','f4'),('ny','f4'),('nz','f4'),
    ])
    for i, ax in enumerate('xyz'):
        vertex[ax] = pts_ds[:, i]
    for i, ax in enumerate(['nx','ny','nz']):
        vertex[ax] = nrm_ds[:, i]
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(str(out_path))
    print(f"  Init PC: {len(pts_ds):,} pts at 5cm voxel")


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 – Regenerate depth maps
# ──────────────────────────────────────────────────────────────────────────────

def regenerate_depth_maps(output_dir: Path, corrected_pc_path: Path, voxel_size: float = 0.002):
    """Call lidar_to_depth_maps.py with the ICP-corrected Faro PC."""
    print(f"\n[Step 4] Regenerating depth + normal maps ...")
    script = Path(__file__).parent / 'lidar_to_depth_maps.py'
    cmd = [
        sys.executable, str(script),
        '--las', str(corrected_pc_path),
        '--source', str(output_dir),
        '--voxel_size', str(voxel_size),
        '--mask_folder', 'images_masked',
    ]
    print(f"  CMD: {' '.join(cmd)}")
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        print(f"  [WARNING] lidar_to_depth_maps.py exited with code {ret.returncode}")
    return ret.returncode == 0


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 – Coverage / alignment verification
# ──────────────────────────────────────────────────────────────────────────────

def verify_alignment(output_dir: Path, n_sample: int = 20) -> float:
    """Sample depth maps and print mean coverage."""
    import glob, random
    files = glob.glob(str(output_dir / 'depth_maps' / '**' / '*.npy'), recursive=True)
    if not files:
        print("  [WARNING] No depth maps found for verification")
        return 0.0
    sample = random.sample(files, min(n_sample, len(files)))
    covs = [(np.load(f) > 0).mean() * 100 for f in sample]
    mean_cov = float(np.mean(covs))
    print(f"\n[Step 5] Alignment verification:")
    print(f"  Depth map coverage: mean={mean_cov:.1f}%  (sample of {len(sample)} files)")
    return mean_cov


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene_id',         required=True)
    parser.add_argument('--scene_dir',        required=True,
                        help='ScanNet++ raw scene dir (contains iphone/depth.bin)')
    parser.add_argument('--artlab_dense_dir', required=True,
                        help='Existing iphone_dense artlab-format base dir')
    parser.add_argument('--output_dir',       required=True,
                        help='Output dir for ICP-corrected variant')
    parser.add_argument('--depth_stride',     type=int,   default=10,
                        help='Use every Nth depth frame for iPhone LiDAR cloud (default 10)')
    parser.add_argument('--cloud_voxel',      type=float, default=0.005,
                        help='Voxel size for iPhone cloud accumulation (default 5mm)')
    parser.add_argument('--icp_voxel',        type=float, default=0.02,
                        help='Voxel size for ICP (default 2cm)')
    parser.add_argument('--icp_max_iter',     type=int,   default=200)
    parser.add_argument('--icp_max_corr_dist',type=float, default=0.05,
                        help='Max correspondence distance for ICP (default 5cm)')
    parser.add_argument('--dm_voxel',         type=float, default=0.002,
                        help='Voxel size for depth map generation (default 2mm)')
    parser.add_argument('--skip_depth_regen', action='store_true',
                        help='Skip regenerating depth maps (if already done)')
    parser.add_argument('--only_icp',         action='store_true',
                        help='Only run steps 1-2 (ICP) and print result, no dir creation')
    args = parser.parse_args()

    scene_dir   = Path(args.scene_dir).expanduser()
    base_dir    = Path(args.artlab_dense_dir).expanduser()
    output_dir  = Path(args.output_dir).expanduser()

    depth_bin = scene_dir / 'iphone' / 'depth.bin'
    pose_json = scene_dir / 'iphone' / 'pose_intrinsic_imu.json'
    faro_pc   = base_dir / 'pc_aligned_artlab_frame.ply'
    cameras_txt = base_dir / 'sparse' / '0' / 'cameras.txt'

    for p in [depth_bin, pose_json, faro_pc, cameras_txt]:
        assert p.exists(), f"Missing: {p}"

    print(f"[Faro ICP Bridge]")
    print(f"  scene:   {args.scene_id}")
    print(f"  depth_bin: {depth_bin} ({os.path.getsize(depth_bin)//1024//1024} MB)")
    print(f"  faro_pc:   {faro_pc}")
    print(f"  output:    {output_dir}")

    # Read RGB camera intrinsics from cameras.txt
    fx = fy = cx = cy = W = H = None
    with open(cameras_txt) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if parts[1] in ('PINHOLE', 'OPENCV', 'SIMPLE_PINHOLE'):
                W, H = int(parts[2]), int(parts[3])
                if parts[1] == 'SIMPLE_PINHOLE':
                    fx = fy = float(parts[4]); cx = float(parts[5]); cy = float(parts[6])
                else:
                    fx, fy = float(parts[4]), float(parts[5])
                    cx, cy = float(parts[6]), float(parts[7])
                break
    assert fx is not None, "Could not parse cameras.txt"
    print(f"  RGB intrinsics: {W}×{H}  fx={fx:.1f} fy={fy:.1f}")

    # ── Step 1: Accumulate iPhone LiDAR ─────────────────────────────────────
    iphone_pts = accumulate_iphone_lidar(
        depth_bin, pose_json,
        fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H,
        depth_stride=args.depth_stride,
        cloud_voxel=args.cloud_voxel,
    )

    # Save iPhone LiDAR PC for inspection
    iphone_pc_out = output_dir.parent / f'{args.scene_id}_iphone_lidar_cloud.ply'
    if not args.only_icp:
        iphone_pc_out.parent.mkdir(parents=True, exist_ok=True)
        _write_ply_with_normals(iphone_pts, iphone_pc_out)
        print(f"  Saved iPhone LiDAR cloud → {iphone_pc_out}")

    # ── Step 2: Load Faro and ICP ────────────────────────────────────────────
    print(f"\n  Loading Faro PC from {faro_pc} ...")
    pcd_faro = o3d.io.read_point_cloud(str(faro_pc))
    faro_pts = np.asarray(pcd_faro.points).astype(np.float32)
    print(f"  Faro: {len(faro_pts):,} pts  Z=[{faro_pts[:,2].min():.2f},{faro_pts[:,2].max():.2f}]")

    icp_meta = icp_align_faro_to_iphone(
        faro_pts, iphone_pts,
        icp_voxel=args.icp_voxel,
        max_iter=args.icp_max_iter,
        max_corr_dist=args.icp_max_corr_dist,
    )

    if args.only_icp:
        print("\n[--only_icp] Stopping after ICP. Results:")
        print(f"  fitness={icp_meta['fitness']:.4f}  "
              f"RMSE={icp_meta['inlier_rmse_mm']:.2f}mm  "
              f"trans={icp_meta['translation_mm']:.2f}mm  "
              f"rot={icp_meta['rotation_deg']:.3f}°")
        return

    # Apply ICP transform to full-res Faro PC
    T_corr = icp_meta['transform']
    pts_h  = np.concatenate([faro_pts.astype(np.float64),
                              np.ones((len(faro_pts), 1))], axis=1)
    faro_corrected = (T_corr @ pts_h.T).T[:, :3].astype(np.float32)
    print(f"\n  Corrected Faro: {len(faro_corrected):,} pts  "
          f"Z=[{faro_corrected[:,2].min():.2f},{faro_corrected[:,2].max():.2f}]")

    # ── Step 3: Create corrected artlab dir ──────────────────────────────────
    create_corrected_artlab_dir(base_dir, output_dir, faro_corrected, icp_meta)

    # ── Step 4: Regenerate depth maps ────────────────────────────────────────
    corrected_pc = output_dir / 'pc_aligned_artlab_frame.ply'
    if not args.skip_depth_regen:
        regenerate_depth_maps(output_dir, corrected_pc, voxel_size=args.dm_voxel)
    else:
        print("\n[Step 4] Skipping depth map regeneration (--skip_depth_regen)")

    # ── Step 5: Verify coverage ──────────────────────────────────────────────
    cov = verify_alignment(output_dir)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  ICP fitness:     {icp_meta['fitness']:.4f}")
    print(f"  ICP RMSE:        {icp_meta['inlier_rmse_mm']:.2f} mm")
    print(f"  ICP translation: {icp_meta['translation_mm']:.2f} mm")
    print(f"  ICP rotation:    {icp_meta['rotation_deg']:.3f} °")
    print(f"  Depth coverage:  {cov:.1f}%")
    print(f"  Output:          {output_dir}")
    print("[DONE]")


if __name__ == '__main__':
    main()
