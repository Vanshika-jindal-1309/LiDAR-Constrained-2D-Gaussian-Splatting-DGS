#!/usr/bin/env python3
"""
Verify LiDAR-camera alignment by projecting LiDAR points onto camera images.

Loads the raw LAS file (or a voxel-downsampled subset), reads COLMAP cameras.txt
and images.txt, then projects LiDAR points into 5 sample camera views and saves
color-coded overlay images.

Usage:
    python scripts/verify_lidar_camera_alignment.py \
        --las /path/to/scan.las \
        --source /path/to/perspective \
        --output /path/to/output_dir \
        [--n_samples 5] [--max_pts 500000]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


# ---------------------------------------------------------------------------
# COLMAP file parsers (minimal, text-only)
# ---------------------------------------------------------------------------

def parse_cameras_txt(path):
    """Return dict: camera_id -> dict(model, width, height, fx, fy, cx, cy)."""
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            if model == 'PINHOLE':
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model == 'SIMPLE_PINHOLE':
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                # fallback: assume fx=fy=params[0], cx=width/2, cy=height/2
                fx = fy = params[0]
                cx, cy = width / 2.0, height / 2.0
            cameras[cam_id] = dict(model=model, width=width, height=height,
                                   fx=fx, fy=fy, cx=cx, cy=cy)
    return cameras


def qvec2rotmat(qvec):
    """COLMAP quaternion (qw, qx, qy, qz) → 3×3 rotation matrix (world-to-cam)."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,  2*qx*qy - 2*qw*qz,  2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz,  1 - 2*qx**2 - 2*qz**2,  2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy,  2*qy*qz + 2*qw*qx,  1 - 2*qx**2 - 2*qy**2],
    ])


def parse_images_txt(path):
    """
    Return list of dicts with keys: id, qvec, tvec, camera_id, name.
    Handles Windows CRLF and both empty and non-empty POINTS2D lines.
    images.txt format: pose line, then POINTS2D line (possibly empty), alternating.
    """
    images = []
    with open(path, 'rb') as f:
        raw = f.read().decode('utf-8', errors='replace')
    # Normalise line endings
    all_lines = raw.replace('\r\n', '\n').replace('\r', '\n').split('\n')

    # Collect non-comment lines while preserving blank lines (they are POINTS2D placeholders)
    data_lines = [l for l in all_lines if not l.startswith('#')]

    # Process in pairs: even index = pose, odd index = POINTS2D (may be empty)
    i = 0
    while i < len(data_lines):
        line = data_lines[i].strip()
        if not line:
            i += 1
            continue
        parts = line.split()
        # A pose line starts with an integer IMAGE_ID
        try:
            img_id = int(parts[0])
        except (ValueError, IndexError):
            i += 1
            continue
        qvec = np.array([float(x) for x in parts[1:5]])  # qw qx qy qz
        tvec = np.array([float(x) for x in parts[5:8]])
        camera_id = int(parts[8])
        name = parts[9]
        images.append(dict(id=img_id, qvec=qvec, tvec=tvec,
                           camera_id=camera_id, name=name))
        i += 2  # skip the POINTS2D line (may be empty)
    return images


# ---------------------------------------------------------------------------
# LiDAR loading
# ---------------------------------------------------------------------------

def load_lidar_points(las_path, max_pts=500_000):
    """Load LAS/LAZ and return (N,3) float32 xyz array, optionally downsampled."""
    import laspy
    las = laspy.read(las_path)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    print(f"Loaded {len(xyz):,} LiDAR points from {las_path}")
    if len(xyz) > max_pts:
        idx = np.random.default_rng(42).choice(len(xyz), max_pts, replace=False)
        xyz = xyz[idx]
        print(f"  Randomly downsampled to {len(xyz):,} points for alignment check")
    return xyz


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_points(xyz, R_w2c, t_w2c, fx, fy, cx, cy, width, height):
    """
    Project world-space points through a pinhole camera.

    Returns:
        uv      (M, 2) pixel coordinates of visible points
        z_vals  (M,)   camera-space Z of those points
        mask    (N,)   bool mask of which input points are visible
    """
    # Transform to camera space: X_cam = R @ X_world + t
    X_cam = (R_w2c @ xyz.T).T + t_w2c   # (N, 3)

    in_front = X_cam[:, 2] > 0.01
    Xc = X_cam[in_front]

    u = fx * Xc[:, 0] / Xc[:, 2] + cx
    v = fy * Xc[:, 1] / Xc[:, 2] + cy

    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    uv = np.stack([u[valid], v[valid]], axis=1)
    z_vals = Xc[valid, 2]

    full_mask = np.zeros(len(xyz), dtype=bool)
    idx_front = np.where(in_front)[0]
    idx_valid = idx_front[valid]
    full_mask[idx_valid] = True

    return uv, z_vals, full_mask


# ---------------------------------------------------------------------------
# Overlay image generation
# ---------------------------------------------------------------------------

def make_overlay(img_pil, uv, z_vals, title=""):
    """Return matplotlib figure with LiDAR points overlaid on the camera image."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_pil)

    if len(uv) > 0:
        # Color by depth using viridis; clip extreme values
        z_clipped = np.clip(z_vals, np.percentile(z_vals, 2), np.percentile(z_vals, 98))
        z_norm = (z_clipped - z_clipped.min()) / (z_clipped.ptp() + 1e-8)
        colors = cm.viridis(z_norm)

        # Thin scatter for visibility
        step = max(1, len(uv) // 8000)
        ax.scatter(uv[::step, 0], uv[::step, 1],
                   c=colors[::step], s=1.5, linewidths=0, alpha=0.8)

    ax.set_title(title, fontsize=9)
    ax.axis('off')
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify LiDAR–camera alignment")
    parser.add_argument('--las', required=True, help='Path to .las / .laz file')
    parser.add_argument('--source', required=True,
                        help='Path to COLMAP perspective/ directory (contains cameras.txt, images.txt, images/)')
    parser.add_argument('--output', required=True, help='Directory to save overlay images')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of sample cameras to visualise (default 5)')
    parser.add_argument('--max_pts', type=int, default=500_000,
                        help='Max LiDAR points to load (default 500k for speed)')
    parser.add_argument('--images_folder', type=str, default='images',
                        help='Subfolder under source containing original images (default: images)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ---- load data ----
    xyz = load_lidar_points(args.las, args.max_pts)

    cameras_txt = os.path.join(args.source, 'cameras.txt')
    images_txt = os.path.join(args.source, 'images.txt')
    # Try sparse/0/ fallback
    if not os.path.exists(cameras_txt):
        cameras_txt = os.path.join(args.source, 'sparse', '0', 'cameras.txt')
        images_txt = os.path.join(args.source, 'sparse', '0', 'images.txt')

    cameras = parse_cameras_txt(cameras_txt)
    images = parse_images_txt(images_txt)

    print(f"Loaded {len(cameras)} camera intrinsics, {len(images)} image poses")

    # ---- pick sample images evenly spaced ----
    step = max(1, len(images) // args.n_samples)
    samples = images[::step][:args.n_samples]

    print(f"\nProjecting {len(xyz):,} points onto {len(samples)} sample cameras...\n")

    results = []
    for img_info in samples:
        cam = cameras[img_info['camera_id']]
        R_w2c = qvec2rotmat(img_info['qvec'])
        t_w2c = img_info['tvec']

        uv, z_vals, mask = project_points(
            xyz, R_w2c, t_w2c,
            cam['fx'], cam['fy'], cam['cx'], cam['cy'],
            cam['width'], cam['height']
        )

        pct_visible = 100.0 * mask.sum() / len(xyz)
        print(f"  {img_info['name']}: {mask.sum():,} / {len(xyz):,} pts visible "
              f"({pct_visible:.1f}%), depth range [{z_vals.min():.2f}, {z_vals.max():.2f}] m")

        results.append(dict(img_info=img_info, cam=cam, uv=uv, z_vals=z_vals,
                            pct_visible=pct_visible))

        # Load the actual camera image for overlay
        img_path = None
        for subfolder in [args.images_folder, 'images_masked', 'images']:
            candidate = os.path.join(args.source, subfolder, img_info['name'])
            if os.path.exists(candidate):
                img_path = candidate
                break
            # try with .png extension
            base = os.path.splitext(candidate)[0]
            for ext in ['.png', '.jpg', '.jpeg']:
                if os.path.exists(base + ext):
                    img_path = base + ext
                    break
            if img_path:
                break

        if img_path is None:
            print(f"  [WARNING] Could not find image file for {img_info['name']}, "
                  f"saving point-only overlay")
            img_pil = Image.fromarray(np.zeros((cam['height'], cam['width'], 3), np.uint8))
        else:
            img_pil = Image.open(img_path).convert('RGB')

        title = (f"{img_info['name']} | {mask.sum():,} pts ({pct_visible:.1f}%) | "
                 f"Z: {z_vals.min():.2f}–{z_vals.max():.2f} m")
        fig = make_overlay(img_pil, uv, z_vals, title=title)

        safe_name = img_info['name'].replace('/', '_').replace('.', '_')
        out_path = os.path.join(args.output, f"alignment_{safe_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  → Saved overlay to {out_path}")

    # ---- summary ----
    print("\n=== Alignment Summary ===")
    pcts = [r['pct_visible'] for r in results]
    depths_min = [r['z_vals'].min() for r in results if len(r['z_vals']) > 0]
    depths_max = [r['z_vals'].max() for r in results if len(r['z_vals']) > 0]
    print(f"Visibility: {np.mean(pcts):.1f}% avg ({np.min(pcts):.1f}%–{np.max(pcts):.1f}%)")
    if depths_min:
        print(f"Depth range across samples: {np.min(depths_min):.3f} – {np.max(depths_max):.3f} m")

    # Sanity check: if less than 1% of points land on any image, flag it
    if np.max(pcts) < 1.0:
        print("\n[ERROR] Very few LiDAR points project onto images. "
              "Possible coordinate system mismatch — DO NOT proceed to depth supervision.")
        sys.exit(1)
    elif np.mean(pcts) < 5.0:
        print("\n[WARNING] Low projection coverage. Check that LiDAR and cameras share "
              "the same coordinate frame and scale.")
    else:
        print("\n[OK] LiDAR-camera alignment looks reasonable. "
              "Visually inspect the saved overlays to confirm points land on correct surfaces.")

    print(f"\nOverlay images saved to: {args.output}")


if __name__ == '__main__':
    main()
