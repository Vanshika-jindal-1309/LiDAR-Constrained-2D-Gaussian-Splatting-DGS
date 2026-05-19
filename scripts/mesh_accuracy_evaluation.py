"""
Mesh accuracy evaluation against LiDAR ground truth.

Metrics:
  1. Accuracy (mesh→LiDAR):    mean/median/RMS distance from sampled mesh points to nearest LiDAR point
  2. Completeness (LiDAR→mesh): mean/median/RMS distance from LiDAR points to nearest mesh point
  3. Chamfer Distance:         (accuracy + completeness) / 2
  4. F-score at various thresholds (5mm, 10mm, 20mm, 50mm): harmonic mean of precision and recall
  5. Normal consistency:       mean |dot| of mesh normals vs nearest LiDAR normals

Usage:
  python scripts/mesh_accuracy_evaluation.py \
    --mesh output/dgs_v8b/train/ours_30000/fuse_post.ply \
    --lidar /path/to/scan.las \
    --output output/dgs_v8b/mesh_accuracy \
    --mesh_samples 500000 \
    --lidar_voxel 0.005 \
    --thresholds 0.005 0.01 0.02 0.05
"""

import argparse
import json
import os
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_lidar(path, voxel_size, gt_mesh_samples=1000000):
    """Load LAS, PLY point cloud, or PLY mesh; voxel-downsample, estimate normals if missing.

    If the PLY file is a triangle mesh (has triangles), sample gt_mesh_samples points
    uniformly from its surface and treat as GT reference.
    """
    if path.lower().endswith(".ply"):
        # Try reading as mesh first to detect GT mesh vs point cloud
        mesh_test = o3d.io.read_triangle_mesh(path)
        if len(mesh_test.triangles) > 100:
            print(f"  GT is a mesh ({len(mesh_test.triangles):,} O3D triangles) — "
                  f"sampling {gt_mesh_samples:,} surface points")
            # Use trimesh for robust polygon-mesh support (Replica meshes have polygon faces)
            try:
                import trimesh as _trimesh
                tm = _trimesh.load(path, process=False)
                _n_tm_faces = len(tm.faces)
                print(f"  (trimesh: {len(tm.vertices):,} verts, {_n_tm_faces:,} triangulated faces)")
                _pts_tm, _ = _trimesh.sample.sample_surface(tm, gt_mesh_samples)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(_pts_tm.astype(np.float64))
            except Exception as _e:
                print(f"  trimesh failed ({_e}), using Open3D")
                mesh_test.compute_vertex_normals()
                pcd = mesh_test.sample_points_uniformly(number_of_points=gt_mesh_samples)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
            )
            n_raw = len(pcd.points)
            # For mesh GT, voxel downsample to avoid extremely dense reference
            pcd_ds = pcd.voxel_down_sample(voxel_size)
            n_ds = len(pcd_ds.points)
            print(f"  GT mesh pts: {n_raw:,} → {n_ds:,} at {voxel_size}m voxel")
            return pcd_ds
        else:
            pcd = o3d.io.read_point_cloud(path)
    else:
        import laspy
        las = laspy.read(path)
        pts = np.stack([las.x, las.y, las.z], axis=1).astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

    n_raw = len(pcd.points)
    pcd_ds = pcd.voxel_down_sample(voxel_size)
    n_ds = len(pcd_ds.points)
    print(f"  LiDAR: {n_raw:,} raw → {n_ds:,} at {voxel_size}m voxel")

    if not pcd_ds.has_normals():
        print("  Estimating LiDAR normals...")
        t0 = time.time()
        pcd_ds.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        centroid = np.asarray(pcd_ds.points).mean(axis=0)
        pcd_ds.orient_normals_towards_camera_location(centroid)
        print(f"  Normals estimated in {time.time()-t0:.1f}s")

    return pcd_ds


def sample_mesh(mesh, n_points):
    """Sample n_points uniformly from mesh surface; estimate normals."""
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    return pcd


def crop_to_lidar_bounds(pts, lidar_pts, margin=0.5):
    """Return boolean mask of pts inside LiDAR bounding box + margin."""
    lo = lidar_pts.min(axis=0) - margin
    hi = lidar_pts.max(axis=0) + margin
    return np.all((pts >= lo) & (pts <= hi), axis=1)


def compute_distances(pts_a, pts_b):
    """Return distances from each point in A to its nearest neighbour in B."""
    tree = cKDTree(pts_b)
    dists, _ = tree.query(pts_a, workers=-1)
    return dists


def chamfer_and_fscore(mesh_pts, lidar_pts, thresholds):
    """
    Compute accuracy (A→B), completeness (B→A), Chamfer, F-scores.
    All distances in metres; output in mm.
    """
    print("  Computing mesh→LiDAR distances...")
    t0 = time.time()
    d_a2b = compute_distances(mesh_pts, lidar_pts)
    print(f"    done in {time.time()-t0:.1f}s")

    print("  Computing LiDAR→mesh distances...")
    t0 = time.time()
    d_b2a = compute_distances(lidar_pts, mesh_pts)
    print(f"    done in {time.time()-t0:.1f}s")

    acc_mean   = float(d_a2b.mean())
    acc_median = float(np.median(d_a2b))
    acc_rms    = float(np.sqrt(np.mean(d_a2b ** 2)))

    comp_mean   = float(d_b2a.mean())
    comp_median = float(np.median(d_b2a))
    comp_rms    = float(np.sqrt(np.mean(d_b2a ** 2)))

    chamfer = (acc_mean + comp_mean) / 2.0

    fscores = {}
    for t in thresholds:
        precision = float((d_a2b < t).mean())   # % mesh pts within t of LiDAR
        recall    = float((d_b2a < t).mean())   # % LiDAR pts within t of mesh
        if precision + recall > 0:
            fs = 2.0 * precision * recall / (precision + recall)
        else:
            fs = 0.0
        key = f"F{int(round(t * 1000))}mm"
        fscores[key] = {
            "threshold_m": t,
            "precision": precision,
            "recall": recall,
            "fscore": fs,
        }

    pcts = {
        "p50":  float(np.percentile(d_a2b, 50)  * 1000),
        "p90":  float(np.percentile(d_a2b, 90)  * 1000),
        "p95":  float(np.percentile(d_a2b, 95)  * 1000),
        "p99":  float(np.percentile(d_a2b, 99)  * 1000),
    }

    return {
        "accuracy_mean_mm":       acc_mean   * 1000,
        "accuracy_median_mm":     acc_median * 1000,
        "accuracy_rms_mm":        acc_rms    * 1000,
        "completeness_mean_mm":   comp_mean  * 1000,
        "completeness_median_mm": comp_median* 1000,
        "completeness_rms_mm":    comp_rms   * 1000,
        "chamfer_mm":             chamfer    * 1000,
        "fscores":                fscores,
        "accuracy_percentiles_mm": pcts,
        "n_mesh_pts":             int(len(mesh_pts)),
        "n_lidar_pts":            int(len(lidar_pts)),
    }


def normal_consistency(mesh_pcd, lidar_pcd):
    """Mean |dot| of mesh normals vs nearest LiDAR normals."""
    if not mesh_pcd.has_normals() or not lidar_pcd.has_normals():
        return {"normal_consistency_mean": None, "normal_consistency_median": None}

    m_pts = np.asarray(mesh_pcd.points)
    m_nrm = np.asarray(mesh_pcd.normals)
    l_pts = np.asarray(lidar_pcd.points)
    l_nrm = np.asarray(lidar_pcd.normals)

    tree = cKDTree(l_pts)
    _, idx = tree.query(m_pts, workers=-1)
    nn_nrm = l_nrm[idx]

    dots = np.abs(np.einsum("ij,ij->i", m_nrm, nn_nrm))
    dots = np.clip(dots, 0, 1)

    return {
        "normal_consistency_mean":   float(np.mean(dots)),
        "normal_consistency_median": float(np.median(dots)),
        "normal_consistency_std":    float(np.std(dots)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mesh accuracy vs LiDAR ground truth")
    parser.add_argument("--mesh",         required=True,  help="fuse_post.ply path")
    parser.add_argument("--lidar",        required=True,  help="LiDAR .las or .ply path")
    parser.add_argument("--output",       required=True,  help="Output directory")
    parser.add_argument("--mesh_samples", type=int,   default=500000,
                        help="Points sampled from mesh (default 500k)")
    parser.add_argument("--lidar_voxel",  type=float, default=0.005,
                        help="LiDAR voxel downsampling in metres (default 5mm)")
    parser.add_argument("--thresholds",   nargs="+",  type=float,
                        default=[0.005, 0.01, 0.02, 0.05],
                        help="F-score thresholds in metres (default 5 10 20 50 mm)")
    parser.add_argument("--margin",       type=float, default=0.5,
                        help="Margin (m) to crop mesh points to LiDAR bbox (default 0.5m)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ---- Load mesh ----
    print(f"\nLoading mesh: {args.mesh}")
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh.compute_vertex_normals()
    print(f"  Vertices: {len(mesh.vertices):,}  Triangles: {len(mesh.triangles):,}")

    print(f"Sampling {args.mesh_samples:,} points from mesh surface...")
    mesh_pcd = sample_mesh(mesh, args.mesh_samples)
    mesh_pts = np.asarray(mesh_pcd.points)
    print(f"  Sampled {len(mesh_pts):,} points")

    # ---- Load LiDAR ----
    print(f"\nLoading LiDAR: {args.lidar}")
    lidar_pcd = load_lidar(args.lidar, args.lidar_voxel)
    lidar_pts = np.asarray(lidar_pcd.points)

    # ---- Crop mesh to LiDAR extent ----
    in_bounds = crop_to_lidar_bounds(mesh_pts, lidar_pts, margin=args.margin)
    frac = in_bounds.mean()
    print(f"\nMesh pts in LiDAR bbox±{args.margin}m: "
          f"{in_bounds.sum():,}/{len(mesh_pts):,} ({100*frac:.1f}%)")

    mesh_pts_crop = mesh_pts[in_bounds]
    mesh_nrm_crop = np.asarray(mesh_pcd.normals)[in_bounds] \
        if mesh_pcd.has_normals() else None

    # Build cropped mesh pcd for normal consistency
    mesh_pcd_crop = o3d.geometry.PointCloud()
    mesh_pcd_crop.points = o3d.utility.Vector3dVector(mesh_pts_crop)
    if mesh_nrm_crop is not None:
        mesh_pcd_crop.normals = o3d.utility.Vector3dVector(mesh_nrm_crop)

    # ---- Compute metrics ----
    print("\nComputing Chamfer distance + F-scores...")
    metrics = chamfer_and_fscore(mesh_pts_crop, lidar_pts, args.thresholds)

    print("\nComputing normal consistency...")
    nc = normal_consistency(mesh_pcd_crop, lidar_pcd)
    metrics.update(nc)

    metrics["mesh_path"]   = args.mesh
    metrics["lidar_path"]  = args.lidar
    metrics["lidar_voxel"] = args.lidar_voxel
    metrics["mesh_in_lidar_bbox_fraction"] = float(frac)

    # ---- Print summary ----
    print("\n" + "=" * 65)
    print("MESH ACCURACY RESULTS")
    print("=" * 65)
    print(f"  Accuracy   (mesh→LiDAR): "
          f"mean={metrics['accuracy_mean_mm']:.2f}mm  "
          f"median={metrics['accuracy_median_mm']:.2f}mm  "
          f"RMS={metrics['accuracy_rms_mm']:.2f}mm")
    print(f"  Completeness (LiDAR→mesh): "
          f"mean={metrics['completeness_mean_mm']:.2f}mm  "
          f"median={metrics['completeness_median_mm']:.2f}mm  "
          f"RMS={metrics['completeness_rms_mm']:.2f}mm")
    print(f"  Chamfer distance: {metrics['chamfer_mm']:.2f}mm")
    if metrics.get("normal_consistency_mean") is not None:
        print(f"  Normal consistency: "
              f"mean={metrics['normal_consistency_mean']:.4f}  "
              f"median={metrics['normal_consistency_median']:.4f}")
    p = metrics["accuracy_percentiles_mm"]
    print(f"  Accuracy percentiles (mesh→LiDAR): "
          f"P50={p['p50']:.2f}mm  P90={p['p90']:.2f}mm  "
          f"P95={p['p95']:.2f}mm  P99={p['p99']:.2f}mm")
    for k, v in metrics["fscores"].items():
        print(f"  {k}: precision={v['precision']:.4f}  "
              f"recall={v['recall']:.4f}  F={v['fscore']:.4f}")
    print("=" * 65)

    # ---- Save ----
    out_path = os.path.join(args.output, "mesh_accuracy.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
