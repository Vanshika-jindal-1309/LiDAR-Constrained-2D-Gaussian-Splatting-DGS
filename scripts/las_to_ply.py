#!/usr/bin/env python3
"""
Convert a LAS/LAZ LiDAR point cloud to the PLY format expected by 2D Gaussian Splatting.

The output PLY has per-vertex properties: x, y, z, nx, ny, nz, red, green, blue
matching the format used by storePly/fetchPly in scene/dataset_readers.py.

Usage:
    python las_to_ply.py <input.las> <output.ply> [--voxel_size 0.05] [--max_points 500000]

Examples:
    # Voxel downsample to ~0.05m spacing (recommended for room-scale scenes)
    python las_to_ply.py scan.las points3D.ply --voxel_size 0.05

    # Random downsample to at most 500k points
    python las_to_ply.py scan.las points3D.ply --max_points 500000

    # No downsampling (use all points — may be very slow to train)
    python las_to_ply.py scan.las points3D.ply
"""

import argparse
import numpy as np

def read_las(path):
    """Read a LAS/LAZ file and return xyz (Nx3) and rgb (Nx3, uint8)."""
    import laspy
    las = laspy.read(path)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

    # LAS RGB can be 8-bit or 16-bit; normalise to 0-255
    if hasattr(las, 'red'):
        r, g, b = las.red, las.green, las.blue
        mx = max(r.max(), g.max(), b.max(), 1)
        if mx > 255:
            r = (r / 65535.0 * 255).astype(np.uint8)
            g = (g / 65535.0 * 255).astype(np.uint8)
            b = (b / 65535.0 * 255).astype(np.uint8)
        rgb = np.vstack([r, g, b]).T.astype(np.uint8)
    else:
        # No colour — default to mid-grey
        rgb = np.full((xyz.shape[0], 3), 128, dtype=np.uint8)

    print(f"Loaded {xyz.shape[0]:,} points from {path}")
    return xyz, rgb


def voxel_downsample(xyz, rgb, voxel_size):
    """Voxel-grid downsample: keep one random point per voxel."""
    mins = xyz.min(axis=0)
    keys = ((xyz - mins) / voxel_size).astype(np.int64)
    # Cantor-style hash per voxel
    _, idx = np.unique(
        keys[:, 0] * 1000000007 + keys[:, 1] * 1000003 + keys[:, 2],
        return_index=True,
    )
    print(f"Voxel downsample (size={voxel_size}): {xyz.shape[0]:,} → {len(idx):,} points")
    return xyz[idx], rgb[idx]


def random_downsample(xyz, rgb, max_points):
    """Randomly keep at most max_points."""
    if xyz.shape[0] <= max_points:
        return xyz, rgb
    idx = np.random.default_rng(42).choice(xyz.shape[0], max_points, replace=False)
    idx.sort()
    print(f"Random downsample: {xyz.shape[0]:,} → {max_points:,} points")
    return xyz[idx], rgb[idx]


def write_ply(path, xyz, rgb):
    """Write PLY in the exact format expected by 2DGS fetchPly()."""
    from plyfile import PlyData, PlyElement

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb.astype(np.float32)), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(path)
    print(f"Wrote {xyz.shape[0]:,} points to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LAS LiDAR point cloud to 2DGS-compatible PLY"
    )
    parser.add_argument("input", help="Path to input .las or .laz file")
    parser.add_argument("output", help="Path to output .ply file")
    parser.add_argument("--voxel_size", type=float, default=None,
                        help="Voxel grid spacing in metres for downsampling (e.g. 0.05)")
    parser.add_argument("--max_points", type=int, default=None,
                        help="Maximum number of points to keep (random sampling)")
    args = parser.parse_args()

    xyz, rgb = read_las(args.input)

    if args.voxel_size is not None:
        xyz, rgb = voxel_downsample(xyz, rgb, args.voxel_size)

    if args.max_points is not None:
        xyz, rgb = random_downsample(xyz, rgb, args.max_points)

    write_ply(args.output, xyz, rgb)


if __name__ == "__main__":
    main()
