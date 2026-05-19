#!/usr/bin/env python3
"""
Extract iPhone-native LiDAR depth from depth.bin and create per-frame depth maps
aligned to the RGB camera resolution for 2DGS depth supervision.

The iPhone LiDAR captures depth at 256x192 at ~40fps. This script:
1. Maps each training frame to the nearest depth frame by timestamp
2. Reads 256x192 uint16 depth (mm) from depth.bin
3. Cleans invalid values (0 = no data, >=65500 = out-of-range sentinel)
4. Optionally filters to a valid depth range (default 0.3-7m for indoor)
5. Upsamples to RGB resolution using nearest-neighbor (preserves boundaries)
6. Saves as float32 .npy files matching the existing depth_maps/ format

Usage:
  python scripts/extract_iphone_depth_maps.py \
    --scene_dir /path/to/scannetpp/data/<scene_id> \
    --artlab_dir /path/to/artlab_format/<scene_id>_iphone_dense \
    --output_dir /path/to/artlab_format/<scene_id>_iphone_phonedepth

The output directory is a copy of artlab_dir with depth_maps/ replaced by
phone-LiDAR-derived depth maps (and normal_maps/ regenerated from phone depth gradients
or left empty if --skip_normals is set).
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Depth.bin reader
# ---------------------------------------------------------------------------

DEPTH_W, DEPTH_H = 256, 192
DEPTH_BYTES_PER_FRAME = DEPTH_W * DEPTH_H * 2  # uint16


def read_depth_frame(depth_path: str, frame_idx: int) -> np.ndarray:
    """Read one 256x192 uint16 depth frame (in mm) from depth.bin."""
    with open(depth_path, 'rb') as f:
        f.seek(frame_idx * DEPTH_BYTES_PER_FRAME)
        raw = f.read(DEPTH_BYTES_PER_FRAME)
    if len(raw) < DEPTH_BYTES_PER_FRAME:
        return np.zeros((DEPTH_H, DEPTH_W), dtype=np.uint16)
    return np.frombuffer(raw, dtype=np.uint16).reshape(DEPTH_H, DEPTH_W).copy()


def depth_mm_to_meters(depth_mm: np.ndarray,
                        min_mm: float = 300.0,
                        max_mm: float = 7000.0) -> np.ndarray:
    """Convert uint16 mm depth to float32 meters, masking invalid pixels."""
    depth_f = depth_mm.astype(np.float32)
    valid = (depth_mm > 0) & (depth_mm < 65500) & \
            (depth_f >= min_mm) & (depth_f <= max_mm)
    out = np.zeros((DEPTH_H, DEPTH_W), dtype=np.float32)
    out[valid] = depth_f[valid] / 1000.0
    return out


# ---------------------------------------------------------------------------
# Upsampling (nearest-neighbour — preserves depth discontinuities)
# ---------------------------------------------------------------------------

def upsample_depth_nearest(depth_small: np.ndarray,
                            out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbour resize from (H_s, W_s) to (out_h, out_w)."""
    h_s, w_s = depth_small.shape
    # Map output pixel → input pixel
    rows = (np.arange(out_h) * h_s / out_h).astype(np.int32)
    cols = (np.arange(out_w) * w_s / out_w).astype(np.int32)
    rows = np.clip(rows, 0, h_s - 1)
    cols = np.clip(cols, 0, w_s - 1)
    return depth_small[rows[:, None], cols[None, :]]


def upsample_depth_bilinear_valid(depth_small: np.ndarray,
                                   out_h: int, out_w: int) -> np.ndarray:
    """
    Bilinear upsample, but only for valid-to-valid transitions.
    Pixels where any bilinear neighbour is invalid (=0) use nearest-neighbour
    fallback to avoid bleeding across depth discontinuities.
    """
    from scipy.ndimage import zoom as sp_zoom
    h_s, w_s = depth_small.shape
    scale_h = out_h / h_s
    scale_w = out_w / w_s

    # Bilinear upscale of the depth values (including zeros)
    bilinear = sp_zoom(depth_small, (scale_h, scale_w), order=1, mode='nearest')

    # Nearest-neighbour upscale of a validity mask
    mask_small = (depth_small > 0).astype(np.float32)
    mask_bilinear = sp_zoom(mask_small, (scale_h, scale_w), order=1, mode='nearest')

    # Use bilinear only where all neighbours were valid (mask_bilinear == 1.0)
    near = upsample_depth_nearest(depth_small, out_h, out_w)
    result = np.where(mask_bilinear > 0.99, bilinear, near)
    # Mask out pixels where even the nearest neighbour was 0
    result[near == 0] = 0
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Normal map from depth (finite differences)
# ---------------------------------------------------------------------------

def normals_from_depth(depth_m: np.ndarray, fx: float, fy: float,
                        cx: float, cy: float) -> np.ndarray:
    """
    Compute surface normals from a depth map using finite differences.
    Returns (H, W, 3) float32, length-1 normals; zero vector at invalid pixels.
    """
    H, W = depth_m.shape
    valid = depth_m > 0

    # Back-project to 3D
    uu = (np.arange(W)[None, :] - cx) / fx
    vv = (np.arange(H)[:, None] - cy) / fy

    X = uu * depth_m
    Y = vv * depth_m
    Z = depth_m

    # Finite difference gradients (central diff where possible)
    dXdx = np.gradient(X, axis=1)
    dXdy = np.gradient(X, axis=0)
    dYdx = np.gradient(Y, axis=1)
    dYdy = np.gradient(Y, axis=0)
    dZdx = np.gradient(Z, axis=1)
    dZdy = np.gradient(Z, axis=0)

    # Tangents
    tx = np.stack([dXdx, dYdx, dZdx], axis=-1)  # (H, W, 3)
    ty = np.stack([dXdy, dYdy, dZdy], axis=-1)

    # Normal = tx × ty
    nx = tx[..., 1] * ty[..., 2] - tx[..., 2] * ty[..., 1]
    ny = tx[..., 2] * ty[..., 0] - tx[..., 0] * ty[..., 2]
    nz = tx[..., 0] * ty[..., 1] - tx[..., 1] * ty[..., 0]

    norm = np.stack([nx, ny, nz], axis=-1)
    length = np.linalg.norm(norm, axis=-1, keepdims=True)
    length = np.maximum(length, 1e-8)
    norm = norm / length

    # Flip to point toward camera (z should be negative in camera space → nz < 0)
    flip = (norm[..., 2] > 0).astype(np.float32)[..., None]
    norm = norm * (1 - 2 * flip)

    # Zero out invalid pixels
    out = norm.astype(np.float32)
    out[~valid] = 0
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene_dir', required=True,
                        help='ScanNet++ raw scene dir (contains iphone/depth.bin etc.)')
    parser.add_argument('--artlab_dir', required=True,
                        help='Existing artlab_format dir for this scene')
    parser.add_argument('--output_dir', required=True,
                        help='New artlab_format dir with phone-depth maps')
    parser.add_argument('--min_depth_m', type=float, default=0.3,
                        help='Min valid depth in metres (default 0.3)')
    parser.add_argument('--max_depth_m', type=float, default=7.0,
                        help='Max valid depth in metres (default 7.0)')
    parser.add_argument('--upsample', choices=['nearest', 'bilinear'],
                        default='nearest',
                        help='Upsampling method: nearest (default) or bilinear')
    parser.add_argument('--skip_normals', action='store_true',
                        help='Skip normal map generation (saves time)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite output_dir if it exists')
    args = parser.parse_args()

    scene_dir  = Path(args.scene_dir).expanduser()
    artlab_dir = Path(args.artlab_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    depth_bin  = scene_dir / 'iphone' / 'depth.bin'
    pose_json  = scene_dir / 'iphone' / 'pose_intrinsic_imu.json'
    mapping_json = artlab_dir / 'image_name_mapping.json'
    cameras_colmap = artlab_dir / 'sparse' / '0' / 'cameras.txt'

    assert depth_bin.exists(), f"Missing: {depth_bin}"
    assert pose_json.exists(), f"Missing: {pose_json}"
    assert mapping_json.exists(), f"Missing: {mapping_json}"

    # ---- Read camera intrinsics from COLMAP cameras.txt ----
    fx, fy, cx, cy = None, None, None, None
    with open(cameras_colmap) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            # PINHOLE: cam_id MODEL W H fx fy cx cy
            # OPENCV:  cam_id MODEL W H fx fy cx cy k1 k2 p1 p2
            if parts[1] in ('PINHOLE', 'OPENCV', 'SIMPLE_PINHOLE'):
                out_w, out_h = int(parts[2]), int(parts[3])
                fx, fy = float(parts[4]), float(parts[5])
                cx, cy = float(parts[6]), float(parts[7])
                break
    assert fx is not None, "Could not read camera intrinsics"
    print(f"RGB intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}, W={out_w}, H={out_h}")

    # ---- Load timestamps ----
    print("Loading pose timestamps ...")
    with open(pose_json) as f:
        poses = json.load(f)
    pose_keys = sorted(poses.keys(), key=lambda x: int(x.split('_')[-1]))
    rgb_timestamps = [poses[k]['timestamp'] for k in pose_keys]  # 7040 entries at 60fps
    T_start = rgb_timestamps[0]
    T_end   = rgb_timestamps[-1]
    total_depth_frames = os.path.getsize(depth_bin) // DEPTH_BYTES_PER_FRAME
    depth_timestamps = np.linspace(T_start, T_end, total_depth_frames)
    print(f"RGB frames: {len(rgb_timestamps)}, depth frames: {total_depth_frames}")
    print(f"Time range: {T_start:.3f} - {T_end:.3f} s, depth fps ≈ {total_depth_frames/(T_end-T_start):.1f}")

    # ---- Load training frame → RGB frame ID mapping ----
    with open(mapping_json) as f:
        name_map = json.load(f)  # {train_name: original_frame_name}
    # Map training frame index (00000, 00001, ...) → RGB frame integer ID
    def get_rgb_id(orig_name: str) -> int:
        # orig_name like "frame_000010.jpg"
        return int(orig_name.replace('frame_', '').replace('.jpg', '').replace('.png', ''))

    sorted_train = sorted(name_map.items())  # [(00000.jpg, frame_000010.jpg), ...]
    print(f"Training frames: {len(sorted_train)}")

    # ---- Set up output directory ----
    if output_dir.exists():
        if not args.overwrite:
            print(f"Output dir exists: {output_dir}. Use --overwrite to replace.")
            # Still update depth_maps only
        else:
            shutil.rmtree(output_dir)

    # Symlink or copy everything except depth_maps and normal_maps
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in artlab_dir.iterdir():
        if item.name in ('depth_maps', 'normal_maps'):
            continue
        dst = output_dir / item.name
        if not dst.exists():
            if item.is_dir():
                shutil.copytree(str(item), str(dst), symlinks=True)
            else:
                shutil.copy2(str(item), str(dst))

    depth_out_dir  = output_dir / 'depth_maps'  / 'camera_0'
    normal_out_dir = output_dir / 'normal_maps' / 'camera_0'
    depth_out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_normals:
        normal_out_dir.mkdir(parents=True, exist_ok=True)

    # ---- LiDAR-scale intrinsics for normal computation ----
    # We compute normals at RGB scale using RGB intrinsics (after upsampling)
    fx_rgb, fy_rgb, cx_rgb, cy_rgb = fx, fy, cx, cy

    # ---- Process each training frame ----
    min_mm = args.min_depth_m * 1000
    max_mm = args.max_depth_m * 1000
    coverages = []

    t0 = time.time()
    for i, (train_name, orig_name) in enumerate(sorted_train):
        rgb_id = get_rgb_id(orig_name)

        # Find nearest depth frame by timestamp
        if rgb_id < len(rgb_timestamps):
            rgb_ts = rgb_timestamps[rgb_id]
        else:
            rgb_ts = T_start + rgb_id / 60.0  # fallback: assume 60fps

        depth_idx = int(np.argmin(np.abs(depth_timestamps - rgb_ts)))
        depth_idx = min(depth_idx, total_depth_frames - 1)

        # Read depth frame
        depth_mm = read_depth_frame(str(depth_bin), depth_idx)
        depth_m  = depth_mm_to_meters(depth_mm, min_mm=min_mm, max_mm=max_mm)

        # Upsample to RGB resolution
        if args.upsample == 'nearest':
            depth_full = upsample_depth_nearest(depth_m, out_h, out_w)
        else:
            depth_full = upsample_depth_bilinear_valid(depth_m, out_h, out_w)

        # Save depth map
        stem = os.path.splitext(train_name)[0]
        np.save(str(depth_out_dir / f'{stem}.npy'), depth_full)

        # Coverage stats
        cov = (depth_full > 0).mean() * 100
        coverages.append(cov)

        # Normal map (computed at full resolution)
        if not args.skip_normals:
            nrm = normals_from_depth(depth_full, fx_rgb, fy_rgb, cx_rgb, cy_rgb)
            np.save(str(normal_out_dir / f'{stem}.npy'), nrm)

        if i % 20 == 0 or i == len(sorted_train) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(sorted_train) - i - 1) if i > 0 else 0
            print(f"  [{i+1}/{len(sorted_train)}] depth_idx={depth_idx} "
                  f"cov={cov:.1f}% elapsed={elapsed:.0f}s ETA={eta:.0f}s")

    print(f"\nDone. Coverage: mean={np.mean(coverages):.1f}% "
          f"min={min(coverages):.1f}% max={max(coverages):.1f}%")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
