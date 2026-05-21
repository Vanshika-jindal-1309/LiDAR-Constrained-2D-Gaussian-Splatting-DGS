#!/usr/bin/env python3
"""
Convert a ScanNet++ scene (iPhone capture) into ARTLab-format for 2DGS training.

Adapts scannetpp_to_artlab_format.py for iPhone data, which differs from DSLR:
  - Images stored in rgb.mkv video (not pre-extracted files)
  - nerfstudio/transforms.json (distorted OPENCV, not transforms_undistorted.json)
  - No train_test_lists.json — uses every-8th-frame test split
  - Standard OPENCV lens (mild distortion k1≈0.07) vs DSLR fisheye

Usage:
  python scripts/scannetpp_iphone_to_artlab_format.py \\
    --scene_dir /path/to/scannetpp/data/<scene_id> \\
    --output_dir /path/to/artlab_format/<scene_id>_iphone
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


def nerfstudio_to_colmap(T_ns):
    T_c2w = T_GLOBAL @ T_ns @ T_CAM_FLIP
    T_w2c = np.linalg.inv(T_c2w)
    return T_w2c[:3, :3].astype(np.float32), T_w2c[:3, 3].astype(np.float32)


def rotmat_to_colmap_quat(R_w2c):
    q_xyzw = Rotation.from_matrix(R_w2c).as_quat()
    qw, qx, qy, qz = q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]
    return qw, qx, qy, qz


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
    # Z-buffer: sort near-to-far so near points overwrite far ones
    order = np.argsort(d_v)
    ui_v = ui_v[order]; vi_v = vi_v[order]; d_v = d_v[order]

    # Fill depth (far first, then overwrite with near — reversed = near last)
    depth_map[vi_v[::-1], ui_v[::-1]] = d_v[::-1]

    if nrm_f is not None:
        nrm_v = nrm_f[in_img][order][::-1]
        normal_map[vi_v[::-1], ui_v[::-1]] = nrm_v.astype(np.float32)

    return depth_map, normal_map


def extract_frames_from_mkv(mkv_path, frame_indices, output_dir, quality=95):
    """
    Extract specific frames from an MKV video using OpenCV.
    frame_indices: list of 0-based frame indices to extract
    Returns dict: {frame_idx: output_path}
    """
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
            if extracted % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / extracted * (len(needed) - extracted)
                print(f"    Extracted {extracted}/{len(needed)} frames "
                      f"(video frame {idx}) ETA={eta:.0f}s")
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
    parser.add_argument('--voxel_size', type=float, default=0.005)
    parser.add_argument('--test_every', type=int, default=8,
                        help='Hold out every Nth frame as test (default 8)')
    parser.add_argument('--verify_frames', type=int, default=5)
    parser.add_argument('--min_coverage_pct', type=float, default=10.0)
    args = parser.parse_args()

    scene_dir  = Path(args.scene_dir)
    output_dir = Path(args.output_dir)

    print(f"[ScanNet++ iPhone → ARTLab] scene:  {scene_dir}")
    print(f"[ScanNet++ iPhone → ARTLab] output: {output_dir}")

    transforms_json = scene_dir / 'iphone' / 'nerfstudio' / 'transforms.json'
    rgb_mkv         = scene_dir / 'iphone' / 'rgb.mkv'
    pc_ply          = scene_dir / 'scans'  / 'pc_aligned.ply'

    for p in [transforms_json, rgb_mkv, pc_ply]:
        if not p.exists():
            print(f"[ERROR] Missing: {p}")
            sys.exit(1)

    with open(transforms_json) as f:
        tf_data = json.load(f)

    fl_x = float(tf_data['fl_x'])
    fl_y = float(tf_data['fl_y'])
    cx   = float(tf_data['cx'])
    cy   = float(tf_data['cy'])
    W    = int(tf_data['w'])
    H    = int(tf_data['h'])
    print(f"\n[Camera] OPENCV {W}×{H}, fl_x={fl_x:.2f} fl_y={fl_y:.2f} "
          f"cx={cx:.2f} cy={cy:.2f}")
    if 'k1' in tf_data:
        print(f"  Distortion (ignored, writing PINHOLE): "
              f"k1={tf_data['k1']:.4f} k2={tf_data.get('k2',0):.4f} "
              f"p1={tf_data.get('p1',0):.6f} p2={tf_data.get('p2',0):.6f}")

    # ---- Step B: Frame filtering (train/test split) ----
    all_frames = tf_data['frames']
    n_input = len(all_frames)
    all_frames.sort(key=lambda f: f['file_path'])  # sort by frame name

    # Hold out every test_every-th frame
    test_set = set()
    for i, fr in enumerate(all_frames):
        if i % args.test_every == 0:
            test_set.add(fr['file_path'])

    bad_set = set(f['file_path'] for f in all_frames if f.get('is_bad', False))
    kept_frames = [f for f in all_frames
                   if f['file_path'] not in test_set and f['file_path'] not in bad_set]
    n_kept = len(kept_frames)
    n_test = len(test_set)
    n_bad  = len(bad_set)

    print(f"\n[Step B] Frame filtering:")
    print(f"  Input frames: {n_input}")
    print(f"  Bad frames:   {n_bad}")
    print(f"  Test frames (every {args.test_every}th): {n_test}")
    print(f"  Train frames: {n_kept}")

    if n_kept == 0:
        print("[ERROR] No frames left after filtering!")
        sys.exit(1)

    # ---- Load LiDAR PC ----
    print(f"\n[Step C] Loading PC from {pc_ply} ...")
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(pc_ply))
    pts_all_orig = np.asarray(pcd.points).astype(np.float32)
    has_norms = pcd.has_normals()
    nrm_all_orig = np.asarray(pcd.normals).astype(np.float32) if has_norms else None
    print(f"  Loaded PC: {len(pts_all_orig):,} pts, normals={'yes' if has_norms else 'no'}")

    pts_all = (pts_all_orig @ R_GLOBAL.T).astype(np.float32)
    nrm_all = (nrm_all_orig @ R_GLOBAL.T).astype(np.float32) if nrm_all_orig is not None else None

    pcd_rotated = o3d.geometry.PointCloud()
    pcd_rotated.points = o3d.utility.Vector3dVector(pts_all.astype(np.float64))
    if nrm_all is not None:
        pcd_rotated.normals = o3d.utility.Vector3dVector(nrm_all.astype(np.float64))
    if pcd.has_colors():
        pcd_rotated.colors = pcd.colors
    print(f"  R_GLOBAL applied: PC z range [{pts_all[:,2].min():.2f}, {pts_all[:,2].max():.2f}]")

    # Coverage check
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(len(pts_all), min(200_000, len(pts_all)), replace=False)
    pts_sub = pts_all[sub_idx].astype(np.float64)

    print(f"\n[Step C] Verifying coverage on {args.verify_frames} frames...")
    coverages_verify = []
    for i, fr in enumerate(kept_frames[:args.verify_frames]):
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

    mean_cov = float(np.mean(coverages_verify)) if coverages_verify else 0.0
    print(f"  Mean coverage: {mean_cov:.1f}%")
    if mean_cov < args.min_coverage_pct:
        print(f"[ERROR] Coverage {mean_cov:.1f}% too low — coordinate transform may be wrong!")
        sys.exit(1)

    # Check camera positions in rotated frame
    cam_z = []
    for fr in kept_frames[:10]:
        T_ns = np.array(fr['transform_matrix'])
        T_c2w = T_GLOBAL @ T_ns @ T_CAM_FLIP
        cam_z.append(T_c2w[2, 3])
    print(f"  Camera Z (rotated, first 10): {[f'{z:.2f}' for z in cam_z]}")
    print("[Step C] Transform OK\n")

    # ---- Create output directories ----
    img_out_dir  = output_dir / 'images'         / 'camera_0'
    img_msk_dir  = output_dir / 'images_masked'  / 'camera_0'
    depth_dir    = output_dir / 'depth_maps'     / 'camera_0'
    normal_dir   = output_dir / 'normal_maps'    / 'camera_0'
    sparse_dir   = output_dir / 'sparse' / '0'
    tmp_frames   = output_dir / '_tmp_frames'

    for d in [img_out_dir, img_msk_dir, depth_dir, normal_dir, sparse_dir, tmp_frames]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Save rotated PC ----
    print("[Step A2] Saving rotated PC for DGS...")
    flipped_ply_path = output_dir / 'pc_aligned_artlab_frame.ply'
    o3d.io.write_point_cloud(str(flipped_ply_path), pcd_rotated, write_ascii=False)
    print(f"  Saved {len(pts_all):,} pts → {flipped_ply_path}")

    # ---- cameras.txt (PINHOLE, ignoring distortion) ----
    cameras_txt = sparse_dir / 'cameras.txt'
    with open(cameras_txt, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {W} {H} {fl_x} {fl_y} {cx} {cy}\n")
    print(f"[Step D] cameras.txt: PINHOLE {W}×{H} fx={fl_x:.2f} fy={fl_y:.2f}")

    # ---- Extract frames from MKV ----
    print(f"\n[Step F] Extracting {n_kept} train frames from {rgb_mkv.name} ...")
    frame_indices_needed = []
    for fr in kept_frames:
        idx = int(fr['file_path'].split('_')[1].split('.')[0])
        frame_indices_needed.append(idx)

    extracted = extract_frames_from_mkv(str(rgb_mkv), frame_indices_needed, str(tmp_frames))

    # ---- Build name mapping and copy images ----
    name_mapping = {}
    n_missing = 0
    for img_id, fr in enumerate(kept_frames):
        new_name = f"{img_id:05d}.jpg"
        orig_name = fr['file_path']
        name_mapping[new_name] = orig_name
        frame_idx = int(orig_name.split('_')[1].split('.')[0])
        src = extracted.get(frame_idx)
        if src and Path(src).exists():
            dst_img = img_out_dir / new_name
            dst_msk = img_msk_dir / new_name
            shutil.copy2(str(src), str(dst_img))
            shutil.copy2(str(src), str(dst_msk))
        else:
            print(f"  [WARNING] Missing extracted frame: {orig_name}")
            n_missing += 1

    import json as _json
    mapping_path = output_dir / 'image_name_mapping.json'
    with open(mapping_path, 'w') as f:
        _json.dump(name_mapping, f, indent=2)
    print(f"[Step F] Copied {n_kept - n_missing}/{n_kept} images "
          f"({n_missing} missing)")

    # Clean up temp frames
    shutil.rmtree(str(tmp_frames), ignore_errors=True)

    # ---- images.txt ----
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
            f.write("\n")
    print(f"[Step E] images.txt: {n_kept} entries")

    # ---- points3D.ply (dense init, 5cm voxel) ----
    print(f"\n[Step H] Computing dense init PC (5cm voxel) ...")
    pcd_init = pcd_rotated.voxel_down_sample(0.05)
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
        colors_init_u8 = (np.asarray(pcd_init.colors) * 255).clip(0,255).astype(np.uint8)
    else:
        colors_init_u8 = np.full((len(pts_init), 3), 128, dtype=np.uint8)
    print(f"  Dense init: {len(pts_init):,} points")

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
    vertex_data['x'] = pts_init[:,0]; vertex_data['y'] = pts_init[:,1]; vertex_data['z'] = pts_init[:,2]
    vertex_data['nx'] = norms_init[:,0]; vertex_data['ny'] = norms_init[:,1]; vertex_data['nz'] = norms_init[:,2]
    vertex_data['red'] = colors_init_u8[:,0]; vertex_data['green'] = colors_init_u8[:,1]; vertex_data['blue'] = colors_init_u8[:,2]
    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el], text=False).write(str(sparse_dir / 'points3D.ply'))
    print(f"[Step H] Written points3D.ply: {len(pts_init):,} points")

    # ---- Depth & normal maps ----
    print(f"\n[Step G] Generating depth+normal maps for {n_kept} frames ...")
    print(f"  Voxel downsampling PC (voxel={args.voxel_size}m) ...")
    t_ds = time.time()
    pcd_ds = pcd_rotated.voxel_down_sample(args.voxel_size)
    pts_ds  = np.asarray(pcd_ds.points).astype(np.float32)
    if pcd_ds.has_normals():
        nrm_ds = np.asarray(pcd_ds.normals).astype(np.float32)
    else:
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
            print(f"  [{i+1}/{n_kept}] cov={cov:.1f}%  elapsed={elapsed:.0f}s ETA={eta:.0f}s")

    print(f"\n[Step G] Done. Coverage: mean={np.mean(coverages_all):.1f}% "
          f"min={min(coverages_all):.1f}% max={max(coverages_all):.1f}%")

    # ---- Summary ----
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"  Train frames:        {n_kept}")
    print(f"  Test frames (held):  {n_test}")
    print(f"  Images missing:      {n_missing}")
    print(f"  Mean depth coverage: {np.mean(coverages_all):.1f}%")
    print(f"  PC (DGS):            {flipped_ply_path}")
    print(f"  Output:              {output_dir}")
    print("[DONE]")


if __name__ == '__main__':
    main()
