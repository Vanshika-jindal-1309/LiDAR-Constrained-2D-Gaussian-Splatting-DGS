#!/usr/bin/env python3
"""
Convert a ScanNet++ scene into the ARTLab-shaped folder layout so the
existing depth-supervised 2DGS training pipeline works unmodified.

Input:  ScanNet++ scene directory (e.g. .../data/8b5caf3398)
Output: ARTLab-format directory (e.g. .../data_artlab_format/8b5caf3398_perspective)

Output layout:
  <output_dir>/
    images_masked/
      camera_0/
        00000.jpg, 00001.jpg, ...        (renamed from DSC0xxxx.JPG)
    sparse/
      0/
        cameras.txt    (COLMAP PINHOLE)
        images.txt     (COLMAP quaternion + translation)
        points3D.txt   (stub or sparse sample from PC)
    depth_maps/
      camera_0/
        00000.npy, ...  (float32 H×W, 0 where no LiDAR coverage)
    normal_maps/
      camera_0/
        00000.npy, ...  (float32 H×W×3, world-space normals)
    pc_aligned_artlab_frame.ply  (copy of original PC for DGS --las_path)
    image_name_mapping.json   (new_name → original_name, for eval traceability)

Coordinate transform (nerfstudio OpenGL → COLMAP camera convention):
  T_cam_flip  = diag([1, -1, -1, 1])  # flip camera local y,z: OpenGL → COLMAP
  R_GLOBAL    = diag([1, -1, -1])     # Rx(180°), det=+1 — world gravity-up fix
  T_c2w       = R_GLOBAL @ T_nerfstudio @ T_cam_flip  (c2w, global-rotated world)
  T_w2c       = inv(T_c2w)                             (what COLMAP stores)

NOTE: R_GLOBAL = diag([1,-1,-1]) is a PROPER rotation (det = +1, no reflection)
equivalent to a 180° rotation around the x-axis.  It maps camera positions from
z ≈ −1.45 m (nerfstudio frame, below the room at z=[0,3]) to z ≈ +1.45 m (above
the flipped room at z=[0,−3]).  This gives the optimizer a geometrically sane
starting configuration: cameras above the scene looking downward.  The SAME
rotation is applied to the LiDAR PC so cameras and PC remain in the same
(rotated) world frame.  All depth/normal maps are generated in this rotated frame.
Scipy Rotation.from_matrix() sees det=+1 throughout — no corruption.

Usage:
  python scripts/scannetpp_to_artlab_format.py \\
    --scene_dir /path/to/scannetpp/data/8b5caf3398 \\
    --output_dir /path/to/artlab_format/8b5caf3398_perspective
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T_CAM_FLIP = np.diag([1., -1., -1., 1.])  # OpenGL camera local axes → COLMAP (det = +1)

# Global world rotation: Rx(180°) = diag([1,-1,-1]), det = +1 (proper rotation).
# Maps camera positions from z<0 (nerfstudio) to z>0, giving sane optimizer init.
R_GLOBAL = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
T_GLOBAL = np.eye(4, dtype=np.float64)
T_GLOBAL[:3, :3] = R_GLOBAL


def nerfstudio_to_colmap(T_ns):
    """
    Convert a nerfstudio c2w (4×4) matrix to COLMAP world-to-camera (R, t).
    Returns R_w2c (3×3), t_w2c (3,) in float32.

    Applies T_GLOBAL (Rx 180°) to the world frame so cameras sit at z>0 above
    the scene instead of z<0 below it.  det(R_w2c) = +1 throughout.
    """
    T_c2w = T_GLOBAL @ T_ns @ T_CAM_FLIP   # c2w in rotated world frame
    T_w2c = np.linalg.inv(T_c2w)
    return T_w2c[:3, :3].astype(np.float32), T_w2c[:3, 3].astype(np.float32)


def rotmat_to_colmap_quat(R_w2c):
    """
    Convert 3×3 rotation matrix to COLMAP quaternion (qw, qx, qy, qz).
    scipy returns (x, y, z, w) order → reorder to (w, x, y, z).
    """
    q_xyzw = Rotation.from_matrix(R_w2c).as_quat()   # (x, y, z, w)
    qw, qx, qy, qz = q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]
    return qw, qx, qy, qz


def render_depth_normal_map(pts, normals, R_w2c, t_w2c, fx, fy, cx, cy, W, H):
    """
    Z-buffer project PC points into a pinhole camera.
    Returns:
        depth_map  (H, W) float32  — camera-space Z, 0 = no data
        normal_map (H, W, 3) float32 — world-space normal, zeros = no data
    """
    Xc = (R_w2c @ pts.T).T + t_w2c       # (N, 3) camera-space coords
    in_front = Xc[:, 2] > 0.01
    Xc_f = Xc[in_front]
    normals_f = normals[in_front] if normals is not None else None

    if len(Xc_f) == 0:
        return np.zeros((H, W), np.float32), np.zeros((H, W, 3), np.float32)

    u = fx * Xc_f[:, 0] / Xc_f[:, 2] + cx
    v = fy * Xc_f[:, 1] / Xc_f[:, 2] + cy
    ui = np.floor(u).astype(np.int32)
    vi = np.floor(v).astype(np.int32)
    in_frame = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

    ui_v = ui[in_frame];  vi_v = vi[in_frame]
    z_v  = Xc_f[in_frame, 2].astype(np.float32)
    n_v  = normals_f[in_frame].astype(np.float32) if normals_f is not None else None

    # Z-buffer: sort descending (far → near) so closer overwrites farther
    order = np.argsort(z_v)[::-1]
    ui_v = ui_v[order];  vi_v = vi_v[order];  z_v = z_v[order]
    if n_v is not None:
        n_v = n_v[order]

    depth_map  = np.zeros((H, W), np.float32)
    normal_map = np.zeros((H, W, 3), np.float32)
    depth_map[vi_v, ui_v] = z_v
    if n_v is not None:
        normal_map[vi_v, ui_v, :] = n_v

    return depth_map, normal_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir',  required=True,
                        help='ScanNet++ scene root (e.g. .../data/8b5caf3398)')
    parser.add_argument('--output_dir', required=True,
                        help='ARTLab-format output directory')
    parser.add_argument('--voxel_size', type=float, default=0.005,
                        help='Voxel downsample size for depth/normal generation (m). '
                             'Default 5mm ≈ ScanNet++ PC density.')
    parser.add_argument('--verify_frames', type=int, default=5,
                        help='Number of frames to verify camera coverage before full run')
    parser.add_argument('--min_coverage_pct', type=float, default=10.0,
                        help='Abort if mean coverage across verify_frames < this percent')
    args = parser.parse_args()

    scene_dir  = Path(args.scene_dir)
    output_dir = Path(args.output_dir)

    print(f"[ScanNet++ → ARTLab] scene:  {scene_dir}")
    print(f"[ScanNet++ → ARTLab] output: {output_dir}")

    # ---- Input paths ----
    transforms_json = scene_dir / 'dslr' / 'nerfstudio' / 'transforms_undistorted.json'
    train_test_json = scene_dir / 'dslr' / 'train_test_lists.json'
    img_src_dir     = scene_dir / 'dslr' / 'resized_undistorted_images'
    pc_ply          = scene_dir / 'scans' / 'pc_aligned.ply'

    for p in [transforms_json, train_test_json, img_src_dir, pc_ply]:
        if not p.exists():
            print(f"[ERROR] Missing: {p}")
            sys.exit(1)

    # ---- Load JSON ----
    with open(transforms_json) as f:
        tf_data = json.load(f)
    with open(train_test_json) as f:
        split_data = json.load(f)

    train_set = set(split_data['train'])
    test_set  = set(split_data.get('test', []))

    fl_x = float(tf_data['fl_x'])
    fl_y = float(tf_data['fl_y'])
    cx   = float(tf_data['cx'])
    cy   = float(tf_data['cy'])
    W    = int(tf_data['w'])
    H    = int(tf_data['h'])

    # ---- Step B: Filter frames (train, not bad) ----
    all_frames = tf_data['frames']
    n_input  = len(all_frames)
    n_bad    = sum(1 for f in all_frames if f.get('is_bad', False))
    n_test   = sum(1 for f in all_frames if f['file_path'] in test_set)

    kept_frames = [f for f in all_frames
                   if not f.get('is_bad', False) and f['file_path'] not in test_set]
    n_kept = len(kept_frames)

    print(f"\n[Step B] Frame filtering:")
    print(f"  Input frames: {n_input}")
    print(f"  Bad frames:   {n_bad}")
    print(f"  Test frames in frames[]: {n_test}")
    print(f"  Kept: {n_kept}")

    if n_kept == 0:
        print("[ERROR] No frames left after filtering!")
        sys.exit(1)

    # Sort by filename for deterministic ordering
    kept_frames.sort(key=lambda f: f['file_path'])

    # ---- Step C: Verify coordinate transform on first N frames ----
    print(f"\n[Step C] Verifying coordinate transform on first {args.verify_frames} frames...")

    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(pc_ply))
    pts_all_orig = np.asarray(pcd.points).astype(np.float32)
    has_norms = pcd.has_normals()
    nrm_all_orig = np.asarray(pcd.normals).astype(np.float32) if has_norms else None
    print(f"  Loaded PC: {len(pts_all_orig):,} pts, normals={'yes' if has_norms else 'no'}")

    # Apply global world rotation (Rx 180°) to both points and normals.
    # pts_all and nrm_all are now in the rotated frame that matches camera poses.
    pts_all = (pts_all_orig @ R_GLOBAL.T).astype(np.float32)
    nrm_all = (nrm_all_orig @ R_GLOBAL.T).astype(np.float32) if nrm_all_orig is not None else None

    # Build a rotated Open3D PC for voxel downsampling in Step G and saving in Step A2
    pcd_rotated = o3d.geometry.PointCloud()
    pcd_rotated.points = o3d.utility.Vector3dVector(pts_all.astype(np.float64))
    if nrm_all is not None:
        pcd_rotated.normals = o3d.utility.Vector3dVector(nrm_all.astype(np.float64))
    if pcd.has_colors():
        pcd_rotated.colors = pcd.colors
    print(f"  Applied R_GLOBAL (Rx 180°): PC z range now [{pts_all[:,2].min():.2f}, {pts_all[:,2].max():.2f}] m")

    # Quick sub-sample for coverage check (200k pts, fast) — uses ROTATED pts
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(len(pts_all), min(200_000, len(pts_all)), replace=False)
    pts_sub = pts_all[sub_idx].astype(np.float64)

    coverages_verify = []
    verify_count = min(args.verify_frames, n_kept)
    for i, fr in enumerate(kept_frames[:verify_count]):
        T_ns = np.array(fr['transform_matrix'])
        R_w2c, t_w2c = nerfstudio_to_colmap(T_ns)
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
        print(f"  Frame {i} ({fr['file_path']}): coverage={cov:.1f}%")

    mean_cov = float(np.mean(coverages_verify))
    print(f"  Mean coverage: {mean_cov:.1f}% (abort threshold: {args.min_coverage_pct:.1f}%)")
    if mean_cov < args.min_coverage_pct:
        print(f"\n[ERROR] Coverage {mean_cov:.1f}% < {args.min_coverage_pct:.1f}% threshold.")
        print("  The coordinate transform is almost certainly wrong.")
        print("  STOPPING — fix the transform math before proceeding.")
        sys.exit(1)

    # Camera positions in rotated world frame (sanity: z is positive after R_GLOBAL)
    cam_pos = []
    for fr in kept_frames[:verify_count]:
        T_ns = np.array(fr['transform_matrix'])
        T_c2w = T_GLOBAL @ T_ns @ T_CAM_FLIP   # rotated world frame
        cam_pos.append(T_c2w[:3, 3])
    cam_pos = np.array(cam_pos)
    pc_min = pts_all.min(axis=0);  pc_max = pts_all.max(axis=0)
    print(f"  PC bbox (rotated): {pc_min.round(2)} → {pc_max.round(2)}")
    print(f"  Camera positions range (rotated world frame, expect z>0 above room):")
    print(f"    X: {cam_pos[:,0].min():.2f} – {cam_pos[:,0].max():.2f}")
    print(f"    Y: {cam_pos[:,1].min():.2f} – {cam_pos[:,1].max():.2f}")
    print(f"    Z: {cam_pos[:,2].min():.2f} – {cam_pos[:,2].max():.2f}  (expect positive; room at z<0)")
    first_c2w = T_GLOBAL @ np.array(kept_frames[0]['transform_matrix']) @ T_CAM_FLIP
    print("  Camera +Z (COLMAP looking direction) of first frame:", first_c2w[:3, 2].round(3))
    print("[Step C] Transform verified. Proceeding.\n")

    # ---- Create output directories ----
    img_out_dir  = output_dir / 'images_masked' / 'camera_0'
    depth_dir    = output_dir / 'depth_maps'    / 'camera_0'
    normal_dir   = output_dir / 'normal_maps'   / 'camera_0'
    sparse_dir   = output_dir / 'sparse' / '0'

    for d in [img_out_dir, depth_dir, normal_dir, sparse_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Save rotated PC for DGS --las_path ----
    # Camera poses in images.txt are in the R_GLOBAL-rotated world frame.
    # The LiDAR PC must be in the SAME rotated frame so DGS spatial matching works.
    print(f"\n[Step A2] Saving R_GLOBAL-rotated PC → pc_aligned_artlab_frame.ply ...")
    flipped_ply_path = output_dir / 'pc_aligned_artlab_frame.ply'
    o3d.io.write_point_cloud(str(flipped_ply_path), pcd_rotated, write_ascii=False)
    print(f"  Saved {len(pts_all):,} pts (rotated) → {flipped_ply_path}")

    # ---- Step D: cameras.txt ----
    cameras_txt = sparse_dir / 'cameras.txt'
    with open(cameras_txt, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {W} {H} {fl_x} {fl_y} {cx} {cy}\n")
    print(f"[Step D] Written cameras.txt: PINHOLE {W}×{H} fx={fl_x:.2f} fy={fl_y:.2f} cx={cx} cy={cy}")

    # ---- Step F: Image renaming + mapping ----
    name_mapping = {}   # "00000.jpg" → "DSC02515.JPG"
    img_index = 0
    for fr in kept_frames:
        orig_name = fr['file_path']            # e.g. "DSC02515.JPG"
        new_name  = f"{img_index:05d}.jpg"
        name_mapping[new_name] = orig_name
        img_index += 1

    mapping_path = output_dir / 'image_name_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(name_mapping, f, indent=2)
    print(f"[Step F] Image mapping saved ({len(name_mapping)} entries) → {mapping_path}")

    # ---- Copy images ----
    print(f"[Step F] Copying {len(kept_frames)} images ...")
    n_missing = 0
    for new_name, orig_name in name_mapping.items():
        src = img_src_dir / orig_name
        dst = img_out_dir / new_name
        if src.exists():
            shutil.copy2(str(src), str(dst))
        else:
            print(f"  [WARNING] Image not found: {src}")
            n_missing += 1
    print(f"  Copied {len(name_mapping) - n_missing}/{len(name_mapping)} images"
          f"{f' ({n_missing} missing)' if n_missing else ''}.")

    # ---- Step E: images.txt ----
    images_txt = sparse_dir / 'images.txt'
    with open(images_txt, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img_id, fr in enumerate(kept_frames, start=1):
            T_ns = np.array(fr['transform_matrix'])
            R_w2c, t_w2c = nerfstudio_to_colmap(T_ns)
            qw, qx, qy, qz = rotmat_to_colmap_quat(R_w2c)
            tx, ty, tz = t_w2c
            new_name = f"{img_id - 1:05d}.jpg"
            f.write(f"{img_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{tx:.9f} {ty:.9f} {tz:.9f} 1 camera_0/{new_name}\n")
            f.write("\n")  # empty POINTS2D line
    print(f"[Step E] Written images.txt: {len(kept_frames)} image entries")

    # ---- Step H: points3D.{txt,ply} — dense init from voxel-downsampled LiDAR ----
    # 5cm voxel matches ARTLab init density. Without this, init was 10k random
    # samples → Gaussians had ~12cm scale → photometric densification couldn't
    # cover the floor (ceiling-biased gradients) → recall collapsed.
    init_voxel_size = 0.05  # 5cm
    pcd_init = pcd_rotated.voxel_down_sample(init_voxel_size)
    pts_init = np.asarray(pcd_init.points).astype(np.float32)
    if pcd_init.has_normals():
        norms_init = np.asarray(pcd_init.normals).astype(np.float32)
    else:
        centroid = pts_init.mean(axis=0).astype(np.float64)
        pcd_init.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd_init.orient_normals_towards_camera_location(centroid)
        norms_init = np.asarray(pcd_init.normals).astype(np.float32)
    if pcd_init.has_colors():
        colors_init_u8 = (np.asarray(pcd_init.colors) * 255).clip(0, 255).astype(np.uint8)
    else:
        colors_init_u8 = np.full((len(pts_init), 3), 128, dtype=np.uint8)
    print(f"[Step H] Voxel-downsampled init PC: {len(pts_init):,} points "
          f"(voxel={init_voxel_size}m, was {len(pts_all):,})")

    # Write points3D.txt (compatibility — read only when .ply missing)
    points3d_txt = sparse_dir / 'points3D.txt'
    with open(points3d_txt, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i, (pt, c) in enumerate(zip(pts_init, colors_init_u8)):
            f.write(f"{i+1} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])} 0.0\n")
    print(f"[Step H] Written points3D.txt: {len(pts_init):,} points")

    # Write points3D.ply directly with normals (dataset_readers.fetchPly reads .ply first)
    from plyfile import PlyData, PlyElement
    vertex_data = np.zeros(len(pts_init), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vertex_data['x'] = pts_init[:, 0]
    vertex_data['y'] = pts_init[:, 1]
    vertex_data['z'] = pts_init[:, 2]
    vertex_data['nx'] = norms_init[:, 0]
    vertex_data['ny'] = norms_init[:, 1]
    vertex_data['nz'] = norms_init[:, 2]
    vertex_data['red'] = colors_init_u8[:, 0]
    vertex_data['green'] = colors_init_u8[:, 1]
    vertex_data['blue'] = colors_init_u8[:, 2]
    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el], text=False).write(str(sparse_dir / 'points3D.ply'))
    print(f"[Step H] Written points3D.ply: {len(pts_init):,} points with normals")

    # ---- Step G: Depth & normal maps ----
    print(f"\n[Step G] Generating depth + normal maps for {n_kept} frames ...")

    # Voxel downsample ROTATED PC for depth projection
    print(f"  Voxel downsampling rotated PC (voxel={args.voxel_size}m) ...")
    t_ds = time.time()
    if has_norms:
        pcd_ds = pcd_rotated.voxel_down_sample(args.voxel_size)
        pts_ds  = np.asarray(pcd_ds.points).astype(np.float32)
        nrm_ds  = np.asarray(pcd_ds.normals).astype(np.float32)
    else:
        pcd_ds  = pcd_rotated.voxel_down_sample(args.voxel_size)
        pts_ds  = np.asarray(pcd_ds.points).astype(np.float32)
        # Estimate normals (orient toward scene centroid = rough camera viewpoint)
        centroid = pts_ds.mean(axis=0).astype(np.float64)
        pcd_ds.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd_ds.orient_normals_towards_camera_location(centroid)
        nrm_ds = np.asarray(pcd_ds.normals).astype(np.float32)
    print(f"  Downsampled to {len(pts_ds):,} pts in {time.time()-t_ds:.1f}s")

    t_total = time.time()
    coverages_all = []
    for i, fr in enumerate(kept_frames):
        new_name  = f"{i:05d}"
        depth_path  = depth_dir  / f"{new_name}.npy"
        normal_path = normal_dir / f"{new_name}.npy"

        T_ns = np.array(fr['transform_matrix'])
        R_w2c, t_w2c = nerfstudio_to_colmap(T_ns)

        depth_map, normal_map = render_depth_normal_map(
            pts_ds.astype(np.float64), nrm_ds,
            R_w2c.astype(np.float64), t_w2c.astype(np.float64),
            fl_x, fl_y, cx, cy, W, H
        )

        np.save(str(depth_path),  depth_map)
        np.save(str(normal_path), normal_map)

        cov = (depth_map > 0).mean() * 100.0
        coverages_all.append(cov)

        if i % 20 == 0 or i == n_kept - 1:
            elapsed = time.time() - t_total
            eta = elapsed / (i + 1) * (n_kept - i - 1)
            print(f"  [{i+1}/{n_kept}] coverage={cov:.1f}%  "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")

    print(f"\n[Step G] Depth/normal maps done. "
          f"Coverage: mean={np.mean(coverages_all):.1f}% "
          f"min={min(coverages_all):.1f}% max={max(coverages_all):.1f}%")

    # ---- Final summary ----
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"  Input frames:           {n_input}")
    print(f"  Bad frames skipped:     {n_bad}")
    print(f"  Test frames (in set):   {n_test}")
    print(f"  Output frames:          {n_kept}")
    print(f"  Images copied:          {len(name_mapping) - n_missing}")
    print(f"  Depth maps written:     {n_kept}")
    print(f"  Normal maps written:    {n_kept}")
    print(f"  Mean depth coverage:    {np.mean(coverages_all):.1f}%")
    print(f"  PC copy for DGS:        {flipped_ply_path}  ({len(pts_all):,} pts)")
    print(f"  Output directory:       {output_dir}")

    import subprocess
    try:
        du = subprocess.check_output(['du', '-sh', str(output_dir)],
                                     stderr=subprocess.DEVNULL).decode().split()[0]
        print(f"  Disk usage:             {du}")
    except Exception:
        pass

    print("\ncameras.txt content:")
    with open(cameras_txt) as f:
        print("  " + f.read().strip().replace('\n', '\n  '))

    print("\nFirst 3 lines of images.txt:")
    with open(images_txt) as f:
        lines = f.readlines()
    for ln in lines[:5]:
        if ln.strip():
            print("  " + ln.rstrip())

    print("\n[DONE] Conversion complete.")


if __name__ == '__main__':
    main()
