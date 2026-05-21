#!/usr/bin/env python3
"""
Convert a ScanNet++ iPhone scene to ARTLab-format using DN-Splatter protocol:
  - Loads every --stride-th frame from pose_intrinsic_imu.json (all 7040 poses)
  - Default stride=5 → ~1408 candidate frames (vs 324 with original stride-10)
  - Holds out every --test_every-th loaded frame for evaluation (DN-Splatter: 10)
  - Uses aligned_pose from pose_intrinsic_imu.json for ALL frames (self-consistent
    coordinate frame, aligned to room-scan LiDAR)

Pose conversion formula (verified against COLMAP-registered transforms.json):
    T_c2w = T_GLOBAL @ T_WORLD @ T_imu_aligned_pose
    where T_GLOBAL = diag([1,-1,-1,1]), T_WORLD = swap-XY, negate-Z

Usage:
  python scripts/scannetpp_iphone_to_artlab_format_dense.py \\
    --scene_dir /path/to/scannetpp/data/<scene_id> \\
    --output_dir /path/to/artlab_format/<scene_id>_iphone_dense
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import time


T_CAM_FLIP = np.diag([1., -1., -1., 1.])
R_GLOBAL   = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
T_GLOBAL   = np.eye(4, dtype=np.float64)
T_GLOBAL[:3, :3] = R_GLOBAL

# World-axis permutation: swaps X↔Y and negates Z (imu aligned_pose → nerfstudio world)
# Derived by comparing aligned_pose vs transform_matrix for stride-10 registered frames.
T_WORLD       = np.array([[0.,1.,0.,0.],[1.,0.,0.,0.],[0.,0.,-1.,0.],[0.,0.,0.,1.]])
T_IMU_TO_C2W  = T_GLOBAL @ T_WORLD   # combined: [0,1,0,0; -1,0,0,0; 0,0,1,0; 0,0,0,1]


def imu_aligned_pose_to_colmap(T_imu):
    """Convert pose_intrinsic_imu.json aligned_pose → COLMAP R_w2c, t_w2c."""
    T_c2w = T_IMU_TO_C2W @ T_imu
    T_w2c = np.linalg.inv(T_c2w)
    return T_w2c[:3, :3].astype(np.float32), T_w2c[:3, 3].astype(np.float32)


def rotmat_to_colmap_quat(R_w2c):
    q_xyzw = Rotation.from_matrix(R_w2c).as_quat()
    return q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]   # qw, qx, qy, qz


def render_depth_normal_map(pts, normals, R_w2c, t_w2c, fx, fy, cx, cy, W, H):
    Xc = (R_w2c @ pts.T).T + t_w2c
    in_front = Xc[:, 2] > 0.01
    if in_front.sum() == 0:
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W, 3), dtype=np.float32)

    Xc_f = Xc[in_front]
    nrm_f = normals[in_front] if normals is not None else None

    u = (fx * Xc_f[:, 0] / Xc_f[:, 2] + cx).astype(np.float32)
    v = (fy * Xc_f[:, 1] / Xc_f[:, 2] + cy).astype(np.float32)
    d = Xc_f[:, 2].astype(np.float32)

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    in_img = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

    depth_map  = np.zeros((H, W), dtype=np.float32)
    normal_map = np.zeros((H, W, 3), dtype=np.float32)

    ui_v = ui[in_img]; vi_v = vi[in_img]; d_v = d[in_img]
    order = np.argsort(d_v)
    ui_v = ui_v[order]; vi_v = vi_v[order]; d_v = d_v[order]

    depth_map[vi_v[::-1], ui_v[::-1]] = d_v[::-1]
    if nrm_f is not None:
        nrm_v = nrm_f[in_img][order][::-1]
        normal_map[vi_v[::-1], ui_v[::-1]] = nrm_v.astype(np.float32)

    return depth_map, normal_map


def extract_frames_from_mkv(mkv_path, frame_indices, output_dir, quality=95):
    import cv2
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    needed = set(frame_indices)
    result_paths = {}

    cap = cv2.VideoCapture(str(mkv_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mkv_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {total_frames} frames @ {fps:.1f} fps, extracting {len(needed)} frames...")

    idx = 0
    extracted = 0
    t0 = time.time()

    while idx <= max(frame_indices):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in needed:
            out_path = output_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            result_paths[idx] = out_path
            extracted += 1
            if extracted % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed / extracted * (len(needed) - extracted)
                print(f"    Extracted {extracted}/{len(needed)} (frame {idx}) ETA={eta:.0f}s")
        idx += 1

    cap.release()
    print(f"  Extracted {extracted}/{len(needed)} frames in {time.time()-t0:.1f}s")
    if extracted < len(needed):
        missing = needed - set(result_paths.keys())
        print(f"  [WARNING] Missing frames: {sorted(missing)[:10]}")
    return result_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir',  required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--stride', type=int, default=5,
                        help='Sample every Nth frame from full 7040-frame sequence (default 5)')
    parser.add_argument('--test_every', type=int, default=10,
                        help='Hold out every Nth loaded frame as test (DN-Splatter default: 10)')
    parser.add_argument('--voxel_size', type=float, default=0.005,
                        help='Voxel size for depth map generation (default 5mm)')
    parser.add_argument('--verify_frames', type=int, default=5)
    parser.add_argument('--min_coverage_pct', type=float, default=5.0)
    args = parser.parse_args()

    scene_dir  = Path(args.scene_dir)
    output_dir = Path(args.output_dir)

    print(f"[ScanNet++ iPhone Dense → ARTLab]")
    print(f"  scene:      {scene_dir}")
    print(f"  output:     {output_dir}")
    print(f"  stride:     {args.stride}  (every {args.stride}th frame from 7040)")
    print(f"  test_every: {args.test_every}")

    imu_json   = scene_dir / 'iphone' / 'pose_intrinsic_imu.json'
    ns_json    = scene_dir / 'iphone' / 'nerfstudio' / 'transforms.json'
    rgb_mkv    = scene_dir / 'iphone' / 'rgb.mkv'
    pc_ply     = scene_dir / 'scans'  / 'pc_aligned.ply'

    for p in [imu_json, ns_json, rgb_mkv, pc_ply]:
        if not p.exists():
            print(f"[ERROR] Missing: {p}")
            sys.exit(1)

    # ---- Camera intrinsics from nerfstudio transforms.json ----
    with open(ns_json) as f:
        ns_data = json.load(f)

    fl_x = float(ns_data['fl_x'])
    fl_y = float(ns_data['fl_y'])
    cx   = float(ns_data['cx'])
    cy   = float(ns_data['cy'])
    W    = int(ns_data['w'])
    H    = int(ns_data['h'])
    print(f"\n[Camera] PINHOLE {W}×{H}  fl_x={fl_x:.2f} fl_y={fl_y:.2f} cx={cx:.2f} cy={cy:.2f}")
    if 'k1' in ns_data:
        print(f"  Distortion (ignored, PINHOLE): k1={ns_data['k1']:.4f}")

    # ---- Load ALL IMU poses ----
    print(f"\n[Step A] Loading IMU poses from {imu_json.name} ...")
    t0 = time.time()
    with open(imu_json) as f:
        imu_data = json.load(f)
    print(f"  Loaded {len(imu_data)} frame poses in {time.time()-t0:.1f}s")

    # ---- Select stride-N frames ----
    total_video_frames = max(
        int(k.replace('frame_','')) for k in imu_data.keys()
    ) + 1
    print(f"  Total video frames: {total_video_frames}")

    candidate_indices = list(range(0, total_video_frames, args.stride))
    # Filter to frames that actually exist in imu_data
    all_frame_indices = [i for i in candidate_indices
                         if f'frame_{i:06d}' in imu_data]
    print(f"  Stride-{args.stride} candidates: {len(candidate_indices)}, "
          f"present in IMU data: {len(all_frame_indices)}")

    # ---- Train/test split ----
    test_set  = set()
    train_set = []
    for i, idx in enumerate(all_frame_indices):
        if i % args.test_every == 0:
            test_set.add(idx)
        else:
            train_set.append(idx)

    n_total = len(all_frame_indices)
    n_train = len(train_set)
    n_test  = len(test_set)
    print(f"\n[Step B] Train/test split (test_every={args.test_every}):")
    print(f"  Total frames: {n_total}  Train: {n_train}  Test: {n_test}")

    # ---- Load LiDAR PC ----
    print(f"\n[Step C] Loading PC from {pc_ply} ...")
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(pc_ply))
    pts_all_orig = np.asarray(pcd.points).astype(np.float32)
    has_norms    = pcd.has_normals()
    nrm_all_orig = np.asarray(pcd.normals).astype(np.float32) if has_norms else None
    print(f"  Loaded PC: {len(pts_all_orig):,} pts")

    # PC is in the same scan/COLMAP world frame as pc_aligned.ply.
    # Camera poses use T_IMU_TO_C2W = T_GLOBAL @ T_WORLD = swap_XY_negY.
    # Apply the same rotation to the PC so both share the artlab world frame.
    # BUG FIX (2026-05-15): was incorrectly using R_GLOBAL=diag([1,-1,-1]) which
    # negates Z, placing PC below cameras (outside-room geometry). Correct rotation
    # is R_IMU_TO_C2W[:3,:3] = [[0,1,0],[-1,0,0],[0,0,1]] (swap_XY_negY, preserves Z).
    R_PC = T_IMU_TO_C2W[:3, :3].astype(np.float64)  # swap_XY_negY
    pts_all = (pts_all_orig @ R_PC.T).astype(np.float32)
    nrm_all = (nrm_all_orig @ R_PC.T).astype(np.float32) if nrm_all_orig is not None else None

    pcd_rotated = o3d.geometry.PointCloud()
    pcd_rotated.points = o3d.utility.Vector3dVector(pts_all.astype(np.float64))
    if nrm_all is not None:
        pcd_rotated.normals = o3d.utility.Vector3dVector(nrm_all.astype(np.float64))
    if pcd.has_colors():
        pcd_rotated.colors = pcd.colors
    print(f"  PC z range: [{pts_all[:,2].min():.2f}, {pts_all[:,2].max():.2f}]")

    # ---- Coverage check on a few train frames ----
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(len(pts_all), min(200_000, len(pts_all)), replace=False)
    pts_sub = pts_all[sub_idx].astype(np.float64)

    print(f"\n[Step C] Verifying coverage on first {args.verify_frames} train frames...")
    coverages_verify = []
    for frame_idx in train_set[:args.verify_frames]:
        key  = f'frame_{frame_idx:06d}'
        T_imu = np.array(imu_data[key]['aligned_pose'])
        R_w2c, t_w2c = imu_aligned_pose_to_colmap(T_imu)

        Xc = (R_w2c.astype(np.float64) @ pts_sub.T).T + t_w2c.astype(np.float64)
        in_front = Xc[:, 2] > 0.01
        if in_front.sum() < 100:
            coverages_verify.append(0.0)
            continue
        Xc_f = Xc[in_front]
        u = fl_x * Xc_f[:, 0] / Xc_f[:, 2] + cx
        v = fl_y * Xc_f[:, 1] / Xc_f[:, 2] + cy
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        cov = in_img.sum() / len(pts_sub) * 100.0
        coverages_verify.append(cov)

    mean_cov = float(np.mean(coverages_verify)) if coverages_verify else 0.0
    print(f"  Mean coverage on {args.verify_frames} frames: {mean_cov:.1f}%")
    if mean_cov < args.min_coverage_pct:
        print(f"[ERROR] Coverage {mean_cov:.1f}% too low — coordinate transform may be wrong!")
        sys.exit(1)

    # Verify camera positions
    cam_z = []
    for frame_idx in train_set[:10]:
        key   = f'frame_{frame_idx:06d}'
        T_imu = np.array(imu_data[key]['aligned_pose'])
        T_c2w = T_IMU_TO_C2W @ T_imu
        cam_z.append(float(T_c2w[2, 3]))
    print(f"  Camera Z (first 10 train): {[f'{z:.2f}' for z in cam_z]}")
    print("[Step C] Transform OK\n")

    # ---- Create output directories ----
    img_out_dir  = output_dir / 'images'        / 'camera_0'
    img_msk_dir  = output_dir / 'images_masked' / 'camera_0'
    depth_dir    = output_dir / 'depth_maps'    / 'camera_0'
    normal_dir   = output_dir / 'normal_maps'   / 'camera_0'
    sparse_dir   = output_dir / 'sparse' / '0'
    tmp_frames   = output_dir / '_tmp_frames'

    for d in [img_out_dir, img_msk_dir, depth_dir, normal_dir, sparse_dir, tmp_frames]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Save rotated PC for DGS ----
    print("[Step A2] Saving rotated PC for DGS...")
    flipped_ply_path = output_dir / 'pc_aligned_artlab_frame.ply'
    o3d.io.write_point_cloud(str(flipped_ply_path), pcd_rotated, write_ascii=False)
    print(f"  Saved {len(pts_all):,} pts → {flipped_ply_path}")

    # ---- cameras.txt ----
    cameras_txt = sparse_dir / 'cameras.txt'
    with open(cameras_txt, 'w') as f:
        f.write("# Camera list\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {W} {H} {fl_x} {fl_y} {cx} {cy}\n")
    print(f"[Step D] cameras.txt: PINHOLE {W}×{H}")

    # ---- Extract frames from MKV ----
    print(f"\n[Step F] Extracting {n_train} train frames from {rgb_mkv.name} ...")
    extracted = extract_frames_from_mkv(str(rgb_mkv), train_set, str(tmp_frames))

    # ---- Copy images + build name mapping ----
    name_mapping = {}
    n_missing = 0
    train_list_sorted = sorted(train_set)
    for img_id, frame_idx in enumerate(train_list_sorted):
        new_name = f"{img_id:05d}.jpg"
        orig_name = f"frame_{frame_idx:06d}.jpg"
        name_mapping[new_name] = orig_name
        src = extracted.get(frame_idx)
        if src and Path(src).exists():
            dst_img = img_out_dir / new_name
            dst_msk = img_msk_dir / new_name
            shutil.copy2(str(src), str(dst_img))
            shutil.copy2(str(src), str(dst_msk))
        else:
            print(f"  [WARNING] Missing frame: {orig_name}")
            n_missing += 1

    import json as _json
    with open(output_dir / 'image_name_mapping.json', 'w') as f:
        _json.dump(name_mapping, f, indent=2)

    # Save frame metadata
    meta = {
        'stride': args.stride,
        'test_every': args.test_every,
        'n_total': n_total,
        'n_train': n_train,
        'n_test': n_test,
        'train_frame_indices': train_list_sorted,
        'test_frame_indices': sorted(test_set),
    }
    with open(output_dir / 'frame_metadata.json', 'w') as f:
        _json.dump(meta, f, indent=2)

    shutil.rmtree(str(tmp_frames), ignore_errors=True)
    print(f"[Step F] Copied {n_train - n_missing}/{n_train} images ({n_missing} missing)")

    # ---- images.txt ----
    images_txt = sparse_dir / 'images.txt'
    with open(images_txt, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img_id, frame_idx in enumerate(train_list_sorted, start=1):
            key   = f'frame_{frame_idx:06d}'
            T_imu = np.array(imu_data[key]['aligned_pose'])
            R_w2c, t_w2c = imu_aligned_pose_to_colmap(T_imu)
            qw, qx, qy, qz = rotmat_to_colmap_quat(R_w2c)
            tx, ty, tz = t_w2c
            new_name = f"{img_id - 1:05d}.jpg"
            f.write(f"{img_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{tx:.9f} {ty:.9f} {tz:.9f} 1 camera_0/{new_name}\n")
            f.write("\n")
    print(f"[Step E] images.txt: {n_train} entries")

    # ---- points3D.ply (dense init at 5cm voxel) ----
    print(f"\n[Step H] Dense init PC (5cm voxel)...")
    pcd_init = pcd_rotated.voxel_down_sample(0.05)
    pts_init = np.asarray(pcd_init.points).astype(np.float32)

    if pcd_init.has_normals():
        norms_init = np.asarray(pcd_init.normals).astype(np.float32)
    else:
        centroid = pts_init.mean(axis=0).astype(np.float64)
        pcd_init.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd_init.orient_normals_towards_camera_location(centroid)
        norms_init = np.asarray(pcd_init.normals).astype(np.float32)

    if pcd_init.has_colors():
        colors_init_u8 = (np.asarray(pcd_init.colors) * 255).clip(0, 255).astype(np.uint8)
    else:
        colors_init_u8 = np.full((len(pts_init), 3), 128, dtype=np.uint8)

    # Z distribution for diagnostics
    z = pts_init[:, 2]
    z_min, z_max = z.min(), z.max()
    floor_frac = (z < z_min + 0.3).mean() * 100
    print(f"  Dense init: {len(pts_init):,} pts  z=[{z_min:.2f},{z_max:.2f}]  "
          f"floor_frac(z<{z_min+0.3:.2f})={floor_frac:.1f}%")

    points3d_txt = sparse_dir / 'points3D.txt'
    with open(points3d_txt, 'w') as f:
        f.write("# 3D point list\n")
        for i, (pt, c) in enumerate(zip(pts_init, colors_init_u8)):
            f.write(f"{i+1} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])} 0.0\n")

    from plyfile import PlyData, PlyElement
    vertex_data = np.zeros(len(pts_init), dtype=[
        ('x','f4'),('y','f4'),('z','f4'),
        ('nx','f4'),('ny','f4'),('nz','f4'),
        ('red','u1'),('green','u1'),('blue','u1'),
    ])
    for attr, col in [('x',0),('y',1),('z',2)]:
        vertex_data[attr] = pts_init[:,col]
    for attr, col in [('nx',0),('ny',1),('nz',2)]:
        vertex_data[attr] = norms_init[:,col]
    vertex_data['red']   = colors_init_u8[:,0]
    vertex_data['green'] = colors_init_u8[:,1]
    vertex_data['blue']  = colors_init_u8[:,2]
    PlyData([PlyElement.describe(vertex_data, 'vertex')]).write(
        str(sparse_dir / 'points3D.ply'))
    print(f"[Step H] Written points3D.ply: {len(pts_init):,} pts")

    # ---- Depth + normal maps ----
    print(f"\n[Step G] Generating depth+normal maps for {n_train} train frames ...")
    print(f"  Downsampling PC (voxel={args.voxel_size}m) ...")
    t_ds = time.time()
    pcd_ds = pcd_rotated.voxel_down_sample(args.voxel_size)
    pts_ds = np.asarray(pcd_ds.points).astype(np.float32)
    if pcd_ds.has_normals():
        nrm_ds = np.asarray(pcd_ds.normals).astype(np.float32)
    else:
        centroid = pts_ds.mean(axis=0).astype(np.float64)
        pcd_ds.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd_ds.orient_normals_towards_camera_location(centroid)
        nrm_ds = np.asarray(pcd_ds.normals).astype(np.float32)
    print(f"  Downsampled to {len(pts_ds):,} pts in {time.time()-t_ds:.1f}s")

    t_total = time.time()
    coverages_all = []
    skipped = 0
    for i, frame_idx in enumerate(train_list_sorted):
        new_name    = f"{i:05d}"
        depth_path  = depth_dir  / f"{new_name}.npy"
        normal_path = normal_dir / f"{new_name}.npy"

        if depth_path.exists() and normal_path.exists():
            skipped += 1
            cov = (np.load(str(depth_path)) > 0).mean() * 100.0
            coverages_all.append(cov)
            continue

        key   = f'frame_{frame_idx:06d}'
        T_imu = np.array(imu_data[key]['aligned_pose'])
        R_w2c, t_w2c = imu_aligned_pose_to_colmap(T_imu)

        depth_map, normal_map = render_depth_normal_map(
            pts_ds.astype(np.float64), nrm_ds,
            R_w2c.astype(np.float64), t_w2c.astype(np.float64),
            fl_x, fl_y, cx, cy, W, H)
        np.save(str(depth_path),  depth_map)
        np.save(str(normal_path), normal_map)

        cov = (depth_map > 0).mean() * 100.0
        coverages_all.append(cov)
        if i % 50 == 0 or i == n_train - 1:
            elapsed = time.time() - t_total
            eta     = elapsed / (i + 1 - skipped) * (n_train - i - 1) if (i + 1 - skipped) > 0 else 0
            print(f"  [{i+1}/{n_train}] frame {frame_idx} cov={cov:.1f}%  "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s (skipped {skipped})")
    if skipped > 0:
        print(f"  Skipped {skipped} already-existing depth+normal maps.")

    print(f"\n[Step G] Done. Coverage: mean={np.mean(coverages_all):.1f}% "
          f"min={min(coverages_all):.1f}% max={max(coverages_all):.1f}%")

    # ---- Summary ----
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"  Stride:              every {args.stride}th frame")
    print(f"  Total candidates:    {n_total}")
    print(f"  Train frames:        {n_train}")
    print(f"  Test frames (held):  {n_test}")
    print(f"  Images missing:      {n_missing}")
    print(f"  Mean depth coverage: {np.mean(coverages_all):.1f}%")
    print(f"  PC (DGS):            {flipped_ply_path}")
    print(f"  Output:              {output_dir}")
    print("[DONE]")


if __name__ == '__main__':
    main()
