"""
Wall roughness analysis for 2DGS reconstructed mesh.

Loads fuse_post.ply, detects major planar surfaces via RANSAC,
calibrates scale using expected room height (~3m), then reports
RMS roughness and peak-to-valley deviation per plane.

Usage:
    python scripts/wall_roughness_analysis.py \
        --mesh output/ARTLab_1901_masked/train/ours_30000/fuse_post.ply \
        --output output/ARTLab_1901_masked/roughness_analysis
"""

import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def classify_plane_normal(normal):
    """Return 'floor', 'ceiling', or 'wall' based on plane normal direction."""
    n = np.asarray(normal)
    n = n / (np.linalg.norm(n) + 1e-9)
    # vertical planes have normals mostly in XY; horizontal planes mostly in Z
    abs_z = abs(n[2])
    if abs_z > 0.85:
        return "floor_or_ceiling"
    else:
        return "wall"


def _ransac_loop(pcd, n_planes, dist_threshold, num_iterations, min_inlier_fraction):
    """Core RANSAC loop on a given pcd; returns list of plane dicts."""
    pts = np.asarray(pcd.points)
    remaining_mask = np.ones(len(pts), dtype=bool)
    planes = []

    for _ in range(n_planes):
        remaining_idx = np.where(remaining_mask)[0]
        if len(remaining_idx) < 500:
            break
        sub = pcd.select_by_index(remaining_idx.tolist())
        try:
            plane_model, inliers = sub.segment_plane(
                distance_threshold=dist_threshold,
                ransac_n=3,
                num_iterations=num_iterations,
            )
        except Exception:
            break
        if len(inliers) < min_inlier_fraction * len(remaining_idx):
            break
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        global_inlier_idx = remaining_idx[inliers]
        inlier_pts = pts[global_inlier_idx]
        planes.append({
            "normal": normal, "d": d,
            "inlier_pts": inlier_pts, "n_inliers": len(inlier_pts),
            "kind": classify_plane_normal(normal),
        })
        remaining_mask[global_inlier_idx] = False

    return planes


def fit_planes_iteratively(pcd, n_planes=8, dist_threshold=0.02,
                            ransac_n=3, num_iterations=1000,
                            min_inlier_fraction=0.005):
    """
    Two-pass RANSAC:
    - Pass 1: restrict to mid-height Z band to prioritise wall detection.
    - Pass 2: full point cloud for floor / ceiling.

    Returns combined list of plane dicts sorted by n_inliers descending.
    """
    pts_all = np.asarray(pcd.points)
    z_min, z_max = pts_all[:, 2].min(), pts_all[:, 2].max()
    z_range = z_max - z_min

    # --- Pass 1: wall-height band (middle 70 % of Z) ---
    wall_mask = (
        (pts_all[:, 2] > z_min + 0.15 * z_range)
        & (pts_all[:, 2] < z_max - 0.15 * z_range)
    )
    wall_idx = np.where(wall_mask)[0]
    wall_sub = pcd.select_by_index(wall_idx.tolist())

    wall_planes = _ransac_loop(
        wall_sub, n_planes=max(4, n_planes // 2),
        dist_threshold=dist_threshold,
        num_iterations=num_iterations,
        min_inlier_fraction=min_inlier_fraction,
    )
    # Remap inlier_pts back to world coords (already in world, just stored correctly)

    # --- Pass 2: full pcd for floor / ceiling ---
    floor_ceil_planes = _ransac_loop(
        pcd, n_planes=4,
        dist_threshold=dist_threshold,
        num_iterations=num_iterations,
        min_inlier_fraction=min_inlier_fraction,
    )
    # Keep only horizontal ones from pass 2
    floor_ceil_planes = [p for p in floor_ceil_planes if p["kind"] == "floor_or_ceiling"]

    all_planes = wall_planes + floor_ceil_planes
    all_planes.sort(key=lambda p: p["n_inliers"], reverse=True)
    return all_planes


def point_to_plane_distances(pts, normal, d):
    """Signed distances from pts to plane defined by normal·x + d = 0."""
    n = normal / (np.linalg.norm(normal) + 1e-9)
    return pts @ n + d


def roughness_stats(distances):
    """Return dict with RMS, peak-to-valley, p95, mean_abs."""
    d = np.asarray(distances)
    rms = float(np.sqrt(np.mean(d ** 2)))
    ptv = float(np.max(d) - np.min(d))
    p5, p95 = float(np.percentile(d, 5)), float(np.percentile(d, 95))
    mean_abs = float(np.mean(np.abs(d)))
    return {"rms": rms, "peak_to_valley": ptv, "p5_p95_range": p95 - p5,
            "mean_abs": mean_abs, "n_points": len(d)}


def scale_from_room_height(planes, expected_floor_ceil_gap_m=3.0):
    """
    Estimate meters_per_unit by finding floor and ceiling planes and
    computing their separation vs expected room height.

    Falls back to 1.0 if < 2 horizontal planes found.
    """
    horiz = [p for p in planes if p["kind"] == "floor_or_ceiling"]
    if len(horiz) < 2:
        print("[scale] Could not find 2 horizontal planes — assuming 1 unit = 1 m")
        return 1.0

    # Sort by mean Z of inliers
    horiz.sort(key=lambda p: float(np.mean(p["inlier_pts"][:, 2])))
    z_floor = float(np.mean(horiz[0]["inlier_pts"][:, 2]))
    z_ceil  = float(np.mean(horiz[-1]["inlier_pts"][:, 2]))
    gap = abs(z_ceil - z_floor)
    if gap < 1e-3:
        return 1.0
    scale = expected_floor_ceil_gap_m / gap
    print(f"[scale] Floor Z={z_floor:.3f}, Ceiling Z={z_ceil:.3f}, "
          f"gap={gap:.3f} units → scale={scale:.4f} m/unit")
    return scale


# ---------------------------------------------------------------------------
# heatmap
# ---------------------------------------------------------------------------

def project_to_plane_2d(pts, normal):
    """
    Project 3D points onto the plane and return 2D coordinates using
    two orthogonal basis vectors tangent to the plane.
    """
    n = normal / (np.linalg.norm(normal) + 1e-9)
    # Pick a reference vector not parallel to n
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return pts @ u, pts @ v


def save_heatmap(pts, distances, normal, label, out_path, scale_m_per_unit,
                 vmax_mm=20.0):
    u_coords, v_coords = project_to_plane_2d(pts, normal)
    dist_mm = np.abs(distances) * scale_m_per_unit * 1000.0

    # Build a grid
    u_min, u_max = u_coords.min(), u_coords.max()
    v_min, v_max = v_coords.min(), v_coords.max()
    grid_res = 400
    ug = np.linspace(u_min, u_max, grid_res)
    vg = np.linspace(v_min, v_max, grid_res)
    uu, vv = np.meshgrid(ug, vg)

    # KD-tree interpolation: mean distance of nearest neighbours
    tree = cKDTree(np.column_stack([u_coords, v_coords]))
    flat = np.column_stack([uu.ravel(), vv.ravel()])
    _, idx = tree.query(flat, k=min(8, len(u_coords)))
    if idx.ndim == 1:
        idx = idx[:, None]
    grid_dist = dist_mm[idx].mean(axis=1).reshape(grid_res, grid_res)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        grid_dist,
        extent=[u_min, u_max, v_min, v_max],
        origin="lower",
        cmap="hot_r",
        vmin=0,
        vmax=vmax_mm,
        aspect="auto",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Roughness (mm)", fontsize=11)
    ax.set_title(f"{label}\nRMS roughness shown as heatmap (capped at {vmax_mm:.0f} mm)",
                 fontsize=10)
    ax.set_xlabel("u (m)")
    ax.set_ylabel("v (m)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Path to fuse_post.ply")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n_planes", type=int, default=8,
                        help="Max number of planes to detect")
    parser.add_argument("--dist_threshold", type=float, default=0.02,
                        help="RANSAC inlier distance threshold (mesh units)")
    parser.add_argument("--room_height_m", type=float, default=3.0,
                        help="Expected room height for scale calibration (m)")
    parser.add_argument("--force_scale", type=float, default=None,
                        help="Force scale (m/unit) instead of calibrating from floor/ceiling. "
                             "Use 1.0 for scenes already in metres (e.g. Replica).")
    parser.add_argument("--sample_pts", type=int, default=500000,
                        help="Number of points to sample from mesh surface")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load mesh → point cloud
    # ------------------------------------------------------------------
    print(f"Loading mesh: {args.mesh}")
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh.compute_vertex_normals()
    print(f"  Vertices: {len(mesh.vertices):,}  Triangles: {len(mesh.triangles):,}")

    print(f"Sampling {args.sample_pts:,} points from surface …")
    pcd = mesh.sample_points_uniformly(args.sample_pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    pts_all = np.asarray(pcd.points)
    bb = pcd.get_axis_aligned_bounding_box()
    extent = bb.get_extent()
    print(f"  Bounding box extent: {extent[0]:.2f} × {extent[1]:.2f} × {extent[2]:.2f} units")

    # ------------------------------------------------------------------
    # 2. RANSAC plane detection
    # ------------------------------------------------------------------
    print(f"\nFitting up to {args.n_planes} planes with RANSAC "
          f"(dist_threshold={args.dist_threshold}) …")
    planes = fit_planes_iteratively(
        pcd,
        n_planes=args.n_planes,
        dist_threshold=args.dist_threshold,
        num_iterations=2000,
        min_inlier_fraction=0.005,
    )
    print(f"  Found {len(planes)} planes")
    for i, p in enumerate(planes):
        print(f"  Plane {i}: {p['kind']:20s}  n={p['n_inliers']:>8,}  "
              f"normal=({p['normal'][0]:+.3f}, {p['normal'][1]:+.3f}, {p['normal'][2]:+.3f})")

    # ------------------------------------------------------------------
    # 3. Scale calibration
    # ------------------------------------------------------------------
    if args.force_scale is not None:
        scale = args.force_scale
        print(f"\n[scale] Forced scale: 1 unit = {scale:.4f} m (--force_scale)")
    else:
        scale = scale_from_room_height(planes, expected_floor_ceil_gap_m=args.room_height_m)
    print(f"\nScale: 1 unit = {scale:.4f} m  →  1 m = {1/scale:.4f} units")

    # ------------------------------------------------------------------
    # 4. Per-plane roughness analysis
    # ------------------------------------------------------------------
    results = {
        "mesh": args.mesh,
        "scale_m_per_unit": scale,
        "room_height_m_assumed": args.room_height_m,
        "n_planes_detected": len(planes),
        "planes": [],
    }

    wall_count = 0
    horiz_count = 0

    for i, plane in enumerate(planes):
        distances = point_to_plane_distances(plane["inlier_pts"], plane["normal"], plane["d"])
        stats = roughness_stats(distances)

        kind = plane["kind"]
        if kind == "floor_or_ceiling":
            label = f"plane_{i}_horiz_{horiz_count}"
            horiz_count += 1
        else:
            label = f"plane_{i}_wall_{wall_count}"
            wall_count += 1

        rms_mm = stats["rms"] * scale * 1000.0
        ptv_mm = stats["peak_to_valley"] * scale * 1000.0
        p5p95_mm = stats["p5_p95_range"] * scale * 1000.0

        print(f"\n  [{label}]")
        print(f"    RMS roughness       : {rms_mm:.2f} mm")
        print(f"    Peak-to-valley      : {ptv_mm:.2f} mm")
        print(f"    P5–P95 range        : {p5p95_mm:.2f} mm")
        print(f"    Points on plane     : {stats['n_points']:,}")

        plane_result = {
            "label": label,
            "kind": kind,
            "normal": plane["normal"].tolist(),
            "d": float(plane["d"]),
            "n_points": stats["n_points"],
            "rms_mm": rms_mm,
            "peak_to_valley_mm": ptv_mm,
            "p5_p95_range_mm": p5p95_mm,
            "mean_abs_mm": stats["mean_abs"] * scale * 1000.0,
        }
        results["planes"].append(plane_result)

        # Heatmap (skip if too few points or heatmap would be trivial)
        if stats["n_points"] >= 500:
            heatmap_path = os.path.join(args.output, f"{label}_heatmap.png")
            print(f"    Saving heatmap → {heatmap_path}")
            try:
                save_heatmap(
                    plane["inlier_pts"], distances, plane["normal"],
                    label, heatmap_path, scale,
                    vmax_mm=max(10.0, rms_mm * 5),
                )
            except Exception as e:
                print(f"    [warn] Heatmap failed: {e}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    walls_only = [p for p in results["planes"] if p["kind"] == "wall"]
    if walls_only:
        mean_wall_rms = np.mean([p["rms_mm"] for p in walls_only])
        max_wall_ptv  = np.max([p["peak_to_valley_mm"] for p in walls_only])
        results["summary"] = {
            "n_walls": len(walls_only),
            "mean_wall_rms_mm": float(mean_wall_rms),
            "max_wall_ptv_mm": float(max_wall_ptv),
        }
        print(f"\n{'='*60}")
        print(f"SUMMARY — {len(walls_only)} wall plane(s)")
        print(f"  Mean RMS roughness : {mean_wall_rms:.2f} mm")
        print(f"  Max peak-to-valley : {max_wall_ptv:.2f} mm")
        print(f"  (Typical drywall: 1–3 mm RMS, painted concrete 3–8 mm)")
        print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 6. Save JSON
    # ------------------------------------------------------------------
    json_path = os.path.join(args.output, "roughness_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # ------------------------------------------------------------------
    # 7. Overview bar chart
    # ------------------------------------------------------------------
    if results["planes"]:
        labels = [p["label"] for p in results["planes"]]
        rms_vals = [p["rms_mm"] for p in results["planes"]]
        colors = ["#2196F3" if p["kind"] == "wall" else "#FF9800"
                  for p in results["planes"]]

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(range(len(labels)), rms_vals, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("RMS Roughness (mm)")
        ax.set_title("Plane roughness — blue=wall, orange=floor/ceiling")
        ax.axhline(y=3.0, color="red", linestyle="--", label="3 mm ref")
        ax.legend()
        plt.tight_layout()
        chart_path = os.path.join(args.output, "roughness_summary.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Summary chart saved to: {chart_path}")


if __name__ == "__main__":
    main()
