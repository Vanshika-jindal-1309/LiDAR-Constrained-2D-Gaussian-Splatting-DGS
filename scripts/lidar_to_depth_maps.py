#!/usr/bin/env python3
"""
Generate per-camera LiDAR depth maps and surface normal maps for depth-supervised 2DGS.

For each camera in images.txt:
  - Projects all LiDAR points through the pinhole camera model
  - Z-buffers to keep the closest point per pixel (camera-space Z coordinate)
  - Saves a (H, W) float32 depth map as .npy  (0 = no data)
  - Saves a (H, W, 3) float32 normal map as .npy in world space (zeros = no data)
  - Applies the alpha mask from images_masked/ (depth=0 where alpha=0)

Output structure:
    {source}/depth_maps/camera_0/img_name.npy
    {source}/depth_maps/camera_1/img_name.npy
    {source}/normal_maps/camera_0/img_name.npy
    {source}/normal_maps/camera_1/img_name.npy

Usage:
    python scripts/lidar_to_depth_maps.py \
        --las /path/to/scan.las \
        --source /path/to/perspective \
        [--voxel_size 0.02] \
        [--normal_radius 0.15] \
        [--normal_max_nn 30] \
        [--mask_folder images_masked] \
        [--workers 1]
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import time

# ---------------------------------------------------------------------------
# COLMAP parsers (copied from verify_lidar_camera_alignment.py)
# ---------------------------------------------------------------------------

def parse_cameras_txt(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = [float(x) for x in parts[4:]]
            if model == 'PINHOLE':
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model == 'SIMPLE_PINHOLE':
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                fx = fy = params[0]
                cx, cy = width / 2.0, height / 2.0
            cameras[cam_id] = dict(width=width, height=height,
                                   fx=fx, fy=fy, cx=cx, cy=cy)
    return cameras


def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,  2*qx*qy - 2*qw*qz,  2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz,  1 - 2*qx**2 - 2*qz**2,  2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy,  2*qy*qz + 2*qw*qx,  1 - 2*qx**2 - 2*qy**2],
    ])


def parse_images_txt(path):
    """Parse images.txt handling Windows CRLF and both empty and non-empty POINTS2D lines."""
    images = []
    with open(path, 'rb') as f:
        raw = f.read().decode('utf-8', errors='replace')
    all_lines = raw.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    data_lines = [l for l in all_lines if not l.startswith('#')]

    i = 0
    while i < len(data_lines):
        line = data_lines[i].strip()
        if not line:
            i += 1
            continue
        parts = line.split()
        try:
            img_id = int(parts[0])
        except (ValueError, IndexError):
            i += 1
            continue
        qvec = np.array([float(x) for x in parts[1:5]])
        tvec = np.array([float(x) for x in parts[5:8]])
        camera_id = int(parts[8])
        name = parts[9]
        images.append(dict(id=img_id, qvec=qvec, tvec=tvec,
                           camera_id=camera_id, name=name))
        i += 2
    return images


def parse_nerf_transforms(transforms_path):
    """Parse NeRF/Blender transforms JSON → list of camera dicts with R_w2c, t_w2c."""
    import json, math
    with open(transforms_path) as f:
        data = json.load(f)

    w, h = int(data['w']), int(data['h'])

    # Compute pixel focal lengths from camera_angle (fl_x in these JSONs can be in mm, not px)
    if 'camera_angle_x' in data:
        fx = w / (2.0 * math.tan(data['camera_angle_x'] / 2.0))
    elif 'fl_x' in data:
        fx = float(data['fl_x'])
    else:
        raise ValueError("No camera_angle_x or fl_x in transforms JSON")

    if 'camera_angle_y' in data:
        fy = h / (2.0 * math.tan(data['camera_angle_y'] / 2.0))
    elif 'fl_y' in data:
        fy = float(data['fl_y'])
    else:
        fy = fx

    cx, cy = w / 2.0, h / 2.0

    cameras = []
    for frame in data['frames']:
        c2w = np.array(frame['transform_matrix'], dtype=np.float64)  # (4,4)
        cam_world_pos = c2w[:3, 3].copy()  # actual camera position in world space (col 3, unaffected by flip)
        # Convert from OpenGL/Blender (Y up, Z back) to OpenCV (Y down, Z forward)
        # This matches the convention used in dataset_readers.py line ~209
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R_w2c = w2c[:3, :3].astype(np.float32)
        t_w2c = w2c[:3, 3].astype(np.float32)

        file_path = frame['file_path']
        # Ensure extension present
        if not file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path += '.jpg'

        cameras.append(dict(
            file_path=file_path,   # e.g. './images/000132.jpg'
            R_w2c=R_w2c, t_w2c=t_w2c,
            cam_world_pos=cam_world_pos,
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=w, height=h,
        ))
    return cameras


# ---------------------------------------------------------------------------
# LiDAR loading and normal estimation
# ---------------------------------------------------------------------------

def load_lidar(las_path, voxel_size=None):
    """Load LAS/PLY and optionally voxel-downsample. Returns (N,3) float64 xyz."""
    if las_path.lower().endswith('.ply'):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(las_path)
        xyz = np.asarray(pcd.points).astype(np.float64)
        print(f"Loaded {len(xyz):,} LiDAR points from PLY")
    else:
        import laspy
        las = laspy.read(las_path)
        xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
        print(f"Loaded {len(xyz):,} LiDAR points")

    if voxel_size is not None:
        xyz = voxel_downsample(xyz, voxel_size)

    return xyz


def voxel_downsample(xyz, voxel_size):
    """Keep one point per voxel (fast hash-based)."""
    mins = xyz.min(axis=0)
    keys = ((xyz - mins) / voxel_size).astype(np.int64)
    _, idx = np.unique(
        keys[:, 0] * 1_000_000_007 + keys[:, 1] * 1_000_003 + keys[:, 2],
        return_index=True)
    xyz_ds = xyz[idx]
    print(f"Voxel downsample (size={voxel_size}m): {len(xyz):,} → {len(xyz_ds):,} points")
    return xyz_ds


def estimate_normals_open3d(xyz, radius=0.15, max_nn=30, viewpoint=None):
    """
    Estimate surface normals using Open3D KNN search.

    Args:
        xyz: (N,3) float64 point positions
        radius: KNN search radius in metres
        max_nn: max neighbours for normal fitting
        viewpoint: (3,) world-space point to orient normals TOWARD (e.g. mean camera position).
                   If None, uses consistent tangent plane orientation (global consistency,
                   may have sign ambiguity at boundaries).

    Returns:
        normals: (N,3) float32 world-space surface normals, pointing toward `viewpoint`
    """
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    if viewpoint is not None:
        # Orient all normals to point toward the camera centroid (room interior).
        # This ensures normals point "inward" (toward cameras) consistently.
        vp = np.array(viewpoint, dtype=np.float64)
        pcd.orient_normals_towards_camera_location(vp)
        print(f"Oriented normals toward viewpoint centroid {vp.round(2)}")
    else:
        # Fallback: global minimum spanning tree consistency (sign may still be ambiguous at edges)
        pcd.orient_normals_consistent_tangent_plane(k=15)
    normals = np.asarray(pcd.normals).astype(np.float32)
    print(f"Estimated {len(normals):,} surface normals (radius={radius}m, max_nn={max_nn})")
    return normals


# ---------------------------------------------------------------------------
# Per-camera depth/normal map generation
# ---------------------------------------------------------------------------

def render_depth_normal_map(xyz, normals, R_w2c, t_w2c, fx, fy, cx, cy, width, height):
    """
    Z-buffer project LiDAR points into a pinhole camera.

    Returns:
        depth_map  (H, W) float32  -- camera-space Z, 0 where no LiDAR data
        normal_map (H, W, 3) float32 -- world-space normal, zeros where no data
    """
    # Camera-space coordinates
    Xc = (R_w2c @ xyz.T).T + t_w2c   # (N, 3)

    # Keep only points in front of camera
    in_front = Xc[:, 2] > 0.01
    Xc_f = Xc[in_front]
    if normals is not None:
        normals_f = normals[in_front]
    else:
        normals_f = None

    if len(Xc_f) == 0:
        return np.zeros((height, width), np.float32), np.zeros((height, width, 3), np.float32)

    u = fx * Xc_f[:, 0] / Xc_f[:, 2] + cx
    v = fy * Xc_f[:, 1] / Xc_f[:, 2] + cy

    ui = np.floor(u).astype(np.int32)
    vi = np.floor(v).astype(np.int32)
    in_frame = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)

    ui_v = ui[in_frame]
    vi_v = vi[in_frame]
    z_v = Xc_f[in_frame, 2].astype(np.float32)
    if normals_f is not None:
        n_v = normals_f[in_frame].astype(np.float32)  # world-space normals

    # Z-buffer: sort by depth descending so that closer (smaller z) overwrites farther
    sort_idx = np.argsort(z_v)[::-1]
    ui_v = ui_v[sort_idx]
    vi_v = vi_v[sort_idx]
    z_v = z_v[sort_idx]
    if normals_f is not None:
        n_v = n_v[sort_idx]

    depth_map = np.zeros((height, width), np.float32)
    normal_map = np.zeros((height, width, 3), np.float32)

    depth_map[vi_v, ui_v] = z_v
    if normals_f is not None:
        normal_map[vi_v, ui_v, :] = n_v

    return depth_map, normal_map


def apply_alpha_mask(depth_map, normal_map, mask_path):
    """Zero out depth/normal where the alpha mask is 0 (black border)."""
    if mask_path is None or not os.path.exists(mask_path):
        return depth_map, normal_map
    try:
        from PIL import Image
        img = Image.open(mask_path)
        if img.mode == 'RGBA':
            alpha = np.array(img.split()[3])  # (H, W) uint8
            valid = alpha > 10
            h, w = depth_map.shape
            if alpha.shape == (h, w):
                depth_map = depth_map * valid
                normal_map = normal_map * valid[:, :, None]
    except Exception as e:
        print(f"  [WARNING] Could not load mask from {mask_path}: {e}")
    return depth_map, normal_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate per-camera LiDAR depth+normal maps')
    parser.add_argument('--las', required=True, help='Path to .las / .laz file')
    parser.add_argument('--source', required=True,
                        help='COLMAP perspective/ directory (cameras.txt, images.txt, images/)')
    parser.add_argument('--voxel_size', type=float, default=0.02,
                        help='Voxel downsampling (m). 0.02=2cm recommended. Use 0 to skip. (default 0.02)')
    parser.add_argument('--normal_radius', type=float, default=0.15,
                        help='Radius for normal estimation in Open3D (m, default 0.15)')
    parser.add_argument('--normal_max_nn', type=int, default=30,
                        help='Max nearest neighbours for normal estimation (default 30)')
    parser.add_argument('--mask_folder', type=str, default='images_masked',
                        help='Subfolder containing alpha-masked PNGs (default: images_masked)')
    parser.add_argument('--skip_normals', action='store_true',
                        help='Skip normal map generation (faster, depth only)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip cameras whose depth map already exists')
    args = parser.parse_args()

    source = args.source

    # ---- Auto-detect dataset format ----
    transforms_train = os.path.join(source, 'transforms_train.json')
    use_nerf_format = os.path.exists(transforms_train)

    cameras_txt = os.path.join(source, 'cameras.txt')
    images_txt = os.path.join(source, 'images.txt')
    if not use_nerf_format and not os.path.exists(cameras_txt):
        cameras_txt = os.path.join(source, 'sparse', '0', 'cameras.txt')
        images_txt = os.path.join(source, 'sparse', '0', 'images.txt')

    # ---- load LiDAR ----
    voxel = args.voxel_size if args.voxel_size > 0 else None
    xyz = load_lidar(args.las, voxel_size=voxel)
    xyz_f32 = xyz.astype(np.float32)

    # ---- compute camera centroid for normal orientation ----
    if use_nerf_format:
        print(f"Detected NeRF/Blender format: {transforms_train}")
        nerf_cams = parse_nerf_transforms(transforms_train)
        print(f"Processing {len(nerf_cams)} camera views (NeRF format)...")
        cam_positions = []
        for nc in nerf_cams:
            cam_positions.append(nc['cam_world_pos'])  # from c2w[:3,3] before flip
    else:
        cameras = parse_cameras_txt(cameras_txt)
        images = parse_images_txt(images_txt)
        print(f"Processing {len(images)} camera views (COLMAP format)...")
        cam_positions = []
        for img_info in images:
            R_w2c = qvec2rotmat(img_info['qvec'])
            t_w2c = img_info['tvec']
            cam_pos = -R_w2c.T @ t_w2c
            cam_positions.append(cam_pos)

    cam_centroid = np.mean(cam_positions, axis=0)
    print(f"Camera centroid (room interior): {cam_centroid.round(3)}")

    # ---- estimate normals ----
    normals = None
    if not args.skip_normals:
        print("Estimating surface normals (this may take a minute)...")
        t0 = time.time()
        normals = estimate_normals_open3d(xyz, radius=args.normal_radius,
                                          max_nn=args.normal_max_nn,
                                          viewpoint=cam_centroid)
        print(f"Normal estimation took {time.time()-t0:.1f}s")

    # ---- output dirs ----
    depth_root = os.path.join(source, 'depth_maps')
    normal_root = os.path.join(source, 'normal_maps')

    # ---- build unified iteration list ----
    # Each entry: dict with R_w2c, t_w2c, fx, fy, cx, cy, width, height,
    #             depth_path, normal_path, mask_path
    render_jobs = []
    if use_nerf_format:
        for nc in nerf_cams:
            file_path = nc['file_path']
            if file_path.startswith('./'):
                file_path = file_path[2:]          # 'images/000132.jpg'
            name_noext = os.path.splitext(file_path)[0]   # 'images/000132'
            stem = os.path.basename(name_noext)            # '000132'
            # camera_utils._load_depth_normal uses parts[1:] → subpath = stem (no subdir)
            depth_path  = os.path.join(depth_root, stem + '.npy')
            normal_path = os.path.join(normal_root, stem + '.npy')
            mask_path = None  # NeRF TnT images have no alpha masks
            render_jobs.append(dict(
                R_w2c=nc['R_w2c'], t_w2c=nc['t_w2c'],
                fx=nc['fx'], fy=nc['fy'], cx=nc['cx'], cy=nc['cy'],
                width=nc['width'], height=nc['height'],
                depth_path=depth_path, normal_path=normal_path,
                mask_path=mask_path, label=file_path,
            ))
    else:
        for img_info in images:
            cam = cameras[img_info['camera_id']]
            R_w2c = qvec2rotmat(img_info['qvec']).astype(np.float32)
            t_w2c = img_info['tvec'].astype(np.float32)
            img_name = img_info['name']
            name_noext = os.path.splitext(img_name)[0]
            subdir = os.path.dirname(name_noext)
            stem = os.path.basename(name_noext)
            depth_dir  = os.path.join(depth_root, subdir) if subdir else depth_root
            normal_dir = os.path.join(normal_root, subdir) if subdir else normal_root
            os.makedirs(depth_dir, exist_ok=True)
            if not args.skip_normals:
                os.makedirs(normal_dir, exist_ok=True)
            depth_path  = os.path.join(depth_dir, stem + '.npy')
            normal_path = os.path.join(normal_dir, stem + '.npy')
            mask_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(source, args.mask_folder, name_noext + ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            render_jobs.append(dict(
                R_w2c=R_w2c, t_w2c=t_w2c,
                fx=cam['fx'], fy=cam['fy'], cx=cam['cx'], cy=cam['cy'],
                width=cam['width'], height=cam['height'],
                depth_path=depth_path, normal_path=normal_path,
                mask_path=mask_path, label=img_name,
            ))

    os.makedirs(depth_root, exist_ok=True)
    if not args.skip_normals:
        os.makedirs(normal_root, exist_ok=True)

    t_start = time.time()
    for idx, job in enumerate(render_jobs):
        # Resume logic
        depth_done = os.path.exists(job['depth_path'])
        normal_done = args.skip_normals or os.path.exists(job['normal_path'])
        if args.resume and depth_done and normal_done:
            continue

        depth_map, normal_map = render_depth_normal_map(
            xyz_f32, normals,
            job['R_w2c'], job['t_w2c'],
            job['fx'], job['fy'], job['cx'], job['cy'],
            job['width'], job['height'],
        )

        if job['mask_path'] is not None:
            depth_map, normal_map = apply_alpha_mask(depth_map, normal_map, job['mask_path'])

        np.save(job['depth_path'], depth_map)
        if not args.skip_normals:
            np.save(job['normal_path'], normal_map.astype(np.float32))

        # Progress
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (idx + 1) * (len(render_jobs) - idx - 1)
            n_valid = (depth_map > 0).sum()
            pct = 100.0 * n_valid / (job['width'] * job['height'])
            depth_vals = depth_map[depth_map > 0]
            drange = f"{depth_vals.min():.2f}–{depth_vals.max():.2f}m" if len(depth_vals) > 0 else "no data"
            print(f"  [{idx+1:4d}/{len(render_jobs)}] {job['label']} | "
                  f"coverage {pct:.1f}% | depth {drange} | ETA {eta/60:.1f}min")

    total = time.time() - t_start
    print(f"\nDone! Generated depth maps in {total/60:.1f} minutes.")
    print(f"Depth maps → {depth_root}")
    if not args.skip_normals:
        print(f"Normal maps → {normal_root}")

    # Quick stats on one of the depth maps
    import glob
    sample_files = glob.glob(os.path.join(depth_root, '**', '*.npy'), recursive=True)
    if sample_files:
        depth_map = np.load(sample_files[len(sample_files) // 2])
        if (depth_map > 0).sum() > 0:
            valid_depths = depth_map[depth_map > 0]
            print(f"\nSample frame stats: {(depth_map>0).mean()*100:.1f}% coverage, "
                  f"depth {valid_depths.min():.3f}–{valid_depths.max():.3f}m, "
                  f"mean {valid_depths.mean():.3f}m")


if __name__ == '__main__':
    main()
