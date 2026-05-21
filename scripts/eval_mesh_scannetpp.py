#!/usr/bin/env python3
"""
ScanNet++ mesh evaluation with DN-Splatter visibility-culled protocol.

Usage:
    python scripts/eval_mesh_scannetpp.py \
        --pred_mesh <path> \
        --gt_mesh   <path> \
        --transforms_json <images.txt> \
        --camera_params   <cameras.txt> \
        --threshold 0.05
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Camera file parsers
# ---------------------------------------------------------------------------

def parse_cameras_txt(path):
    """Return dict: camera_id -> (W, H, fx, fy, cx, cy)."""
    cams = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            cid = int(parts[0])
            model = parts[1]
            W, H = int(parts[2]), int(parts[3])
            if model == 'PINHOLE':
                fx, fy, cx, cy = (float(parts[4]), float(parts[5]),
                                   float(parts[6]), float(parts[7]))
            elif model in ('SIMPLE_RADIAL', 'RADIAL', 'OPENCV'):
                # Use first two focal lengths / principal point
                fx = fy = float(parts[4])
                cx, cy = float(parts[5]), float(parts[6])
            else:
                raise ValueError(f"Unsupported camera model: {model}")
            cams[cid] = (W, H, fx, fy, cx, cy)
    return cams


def parse_images_txt(path):
    """Return list of dicts with w2c pose + camera_id."""
    with open(path, 'rb') as f:
        content = f.read().decode('utf-8', errors='replace')
    lines = content.replace('\r\n', '\n').replace('\r', '\n').split('\n')

    images = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        parts = line.split()
        if len(parts) < 9:
            i += 1
            continue
        try:
            int(parts[0])  # IMAGE_ID guard
        except ValueError:
            i += 1
            continue
        qw, qx, qy, qz = (float(parts[1]), float(parts[2]),
                           float(parts[3]), float(parts[4]))
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        images.append(dict(qw=qw, qx=qx, qy=qy, qz=qz,
                           tx=tx, ty=ty, tz=tz, cam_id=cam_id))
        i += 2  # skip POINTS2D line
    return images


def build_camera_list(images, cameras):
    """Return list of (cam_pos_world, R_w2c, W, H, fx, fy, cx, cy)."""
    result = []
    for img in images:
        # scipy uses (x, y, z, w) ordering
        R_w2c = Rotation.from_quat(
            [img['qx'], img['qy'], img['qz'], img['qw']]
        ).as_matrix()
        t_w2c = np.array([img['tx'], img['ty'], img['tz']])
        # Camera centre in world: c = -R^T @ t
        cam_pos = -R_w2c.T @ t_w2c
        W, H, fx, fy, cx, cy = cameras[img['cam_id']]
        result.append((cam_pos, R_w2c, t_w2c, W, H, fx, fy, cx, cy))
    return result


# ---------------------------------------------------------------------------
# Mesh sampling
# ---------------------------------------------------------------------------

def sample_mesh(mesh_tri, n_samples):
    """Return (pts float32 (N,3), normals float32 (N,3)) from surface."""
    pts, face_idx = trimesh.sample.sample_surface(mesh_tri, n_samples)
    normals = mesh_tri.face_normals[face_idx]
    return pts.astype(np.float32), normals.astype(np.float32)


# ---------------------------------------------------------------------------
# Visibility culling
# ---------------------------------------------------------------------------

def build_raycasting_scene(mesh_tri):
    """Build Open3D tensor RaycastingScene from a trimesh object."""
    verts = np.asarray(mesh_tri.vertices, dtype=np.float32)
    faces = np.asarray(mesh_tri.faces, dtype=np.uint32)
    mesh_t = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(verts),
        o3d.core.Tensor(faces)
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    return scene


def visibility_cull(points, cam_list, scene, epsilon=0.05, batch_size=100_000):
    """
    Return bool mask (N,) — True if visible from ≥1 camera.

    A point is visible from camera C if:
      1. It projects inside the camera image (frustum check).
      2. The first ray-mesh intersection from C towards the point
         occurs within `epsilon` metres of the point's actual distance.
    """
    N = len(points)
    visible = np.zeros(N, dtype=bool)
    t0 = time.time()

    for ci, (cam_pos, R_w2c, t_w2c, W, H, fx, fy, cx, cy) in enumerate(cam_list):
        if (ci % 30 == 0) or ci == len(cam_list) - 1:
            elapsed = time.time() - t0
            print(f"    cam {ci+1:3d}/{len(cam_list)}  "
                  f"visible={visible.sum():6d}/{N}  "
                  f"elapsed={elapsed:.0f}s", flush=True)

        # --- frustum filter ---
        # Transform points to camera space: p_c = R_w2c @ (p - cam_pos)
        # equivalently: p_c = R_w2c @ p + t_w2c
        pts_cam = (R_w2c @ points.T).T + t_w2c  # (N, 3)
        in_front = pts_cam[:, 2] > 0.05          # depth > 5 cm

        u = fx * pts_cam[:, 0] / (pts_cam[:, 2] + 1e-9) + cx
        v = fy * pts_cam[:, 1] / (pts_cam[:, 2] + 1e-9) + cy
        in_frustum = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        idx = np.where(in_frustum)[0]
        if idx.size == 0:
            continue

        pts_f = points[idx]
        dirs = pts_f - cam_pos[None, :]
        dists = np.linalg.norm(dirs, axis=1)

        valid = dists > 0.01
        idx = idx[valid]
        pts_f = pts_f[valid]
        dirs = dirs[valid]
        dists = dists[valid]
        dirs_n = dirs / dists[:, None]

        origins = np.tile(cam_pos.astype(np.float32), (len(idx), 1))
        rays_np = np.hstack([origins, dirs_n.astype(np.float32)])

        # batch raycasting
        hit_good = np.zeros(len(idx), dtype=bool)
        for s in range(0, len(idx), batch_size):
            e = min(s + batch_size, len(idx))
            rays_t = o3d.core.Tensor(rays_np[s:e], dtype=o3d.core.Dtype.Float32)
            ans = scene.cast_rays(rays_t)
            hit_d = ans['t_hit'].numpy()
            hit_good[s:e] = np.abs(hit_d - dists[s:e]) < epsilon

        visible[idx[hit_good]] = True

    return visible


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(pred_pts, pred_nrm, gt_pts, gt_nrm, threshold):
    """Compute accuracy, completeness, Chamfer, F-score, normal consistency."""
    print("  Building KD-trees ...", flush=True)
    tree_gt = cKDTree(gt_pts)
    tree_pr = cKDTree(pred_pts)

    print("  pred → GT distances ...", flush=True)
    d_p2g, idx_p2g = tree_gt.query(pred_pts, workers=-1)

    print("  GT → pred distances ...", flush=True)
    d_g2p, idx_g2p = tree_pr.query(gt_pts, workers=-1)

    accuracy       = float(np.mean(d_p2g))
    completeness   = float(np.mean(d_g2p))
    chamfer        = (accuracy + completeness) / 2.0
    precision      = float(np.mean(d_p2g < threshold))
    recall         = float(np.mean(d_g2p < threshold))
    fscore = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

    # normal consistency: |n_pred · n_gt_nearest|
    nrm_gt_nn = gt_nrm[idx_p2g]
    dot = np.abs(np.sum(pred_nrm * nrm_gt_nn, axis=1))
    normal_cons = float(np.mean(dot))

    return dict(
        accuracy=accuracy,
        completeness=completeness,
        chamfer=chamfer,
        precision=precision,
        recall=recall,
        fscore=fscore,
        normal_consistency=normal_cons,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_mesh',       required=True)
    ap.add_argument('--gt_mesh',         required=True)
    ap.add_argument('--transforms_json', required=True,
                    help='Path to COLMAP images.txt')
    ap.add_argument('--camera_params',   required=True,
                    help='Path to COLMAP cameras.txt')
    ap.add_argument('--threshold', type=float, default=0.05,
                    help='Precision/recall threshold in metres')
    ap.add_argument('--n_samples', type=int, default=200_000,
                    help='Surface sample count per mesh')
    ap.add_argument('--epsilon', type=float, default=0.05,
                    help='Visibility ray-hit tolerance in metres')
    ap.add_argument('--output', default=None,
                    help='Optional path to save JSON results')
    ap.add_argument('--clip_to_gt_bbox', action='store_true',
                    help='Clip pred mesh to GT bbox + margin before evaluation')
    ap.add_argument('--bbox_margin', type=float, default=0.1,
                    help='Padding added to GT bbox for clipping (metres)')
    args = ap.parse_args()

    THR = args.threshold
    sep = '=' * 60

    print(sep)
    print(f"EVALUATION: ScanNet++ visibility-culled protocol")
    print(sep)
    print(f"Predicted mesh: {args.pred_mesh}")
    print(f"GT mesh:        {args.gt_mesh}")
    print(f"Threshold:      {THR*100:.1f} cm")
    print()

    # ------------------------------------------------------------------
    # 1. Load meshes
    # ------------------------------------------------------------------
    print("Loading meshes ...")
    t0 = time.time()
    pred_tri = trimesh.load(os.path.expanduser(args.pred_mesh), process=False)
    gt_tri   = trimesh.load(os.path.expanduser(args.gt_mesh),   process=False)
    print(f"  Pred: {len(pred_tri.vertices):,} verts  {len(pred_tri.faces):,} faces  [{time.time()-t0:.1f}s]")
    print(f"  GT:   {len(gt_tri.vertices):,} verts  {len(gt_tri.faces):,} faces")

    pb = pred_tri.bounds
    gb = gt_tri.bounds
    print()
    print("Mesh sanity:")
    print(f"  Pred bbox:  X=[{pb[0,0]:.2f},{pb[1,0]:.2f}]  "
          f"Y=[{pb[0,1]:.2f},{pb[1,1]:.2f}]  Z=[{pb[0,2]:.2f},{pb[1,2]:.2f}]")
    print(f"  GT bbox:    X=[{gb[0,0]:.2f},{gb[1,0]:.2f}]  "
          f"Y=[{gb[0,1]:.2f},{gb[1,1]:.2f}]  Z=[{gb[0,2]:.2f},{gb[1,2]:.2f}]")

    pred_ext = pb[1] - pb[0]
    gt_ext   = gb[1] - gb[0]
    overlap_lo = np.maximum(pb[0], gb[0])
    overlap_hi = np.minimum(pb[1], gb[1])
    overlap = np.all(overlap_hi > overlap_lo)
    print(f"  Bboxes overlap: {'YES' if overlap else 'NO — possible frame mismatch!'}")

    # Warn if extents are wildly different (>2× in any axis)
    ratio = pred_ext / (gt_ext + 1e-6)
    if np.any(ratio > 2.5) or np.any(ratio < 0.4):
        print(f"  WARNING: extent ratios pred/gt = {ratio.round(2)}  "
              f"— possible coordinate-frame mismatch")

    # ------------------------------------------------------------------
    # 1b. Optional: clip pred mesh to GT bbox + margin
    # ------------------------------------------------------------------
    if args.clip_to_gt_bbox:
        margin = args.bbox_margin
        clip_min = gb[0] - margin
        clip_max = gb[1] + margin
        print()
        print(f"Clipping pred mesh to GT bbox + {margin}m margin ...")
        print(f"  Clip box: X=[{clip_min[0]:.2f},{clip_max[0]:.2f}]  "
              f"Y=[{clip_min[1]:.2f},{clip_max[1]:.2f}]  "
              f"Z=[{clip_min[2]:.2f},{clip_max[2]:.2f}]")

        verts = np.asarray(pred_tri.vertices)
        vert_in = np.all((verts >= clip_min) & (verts <= clip_max), axis=1)

        faces = np.asarray(pred_tri.faces)
        face_in = vert_in[faces].all(axis=1)

        n_verts_before = len(verts)
        n_faces_before = len(faces)

        kept_faces = faces[face_in]
        used_verts = np.unique(kept_faces)
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[used_verts] = np.arange(len(used_verts))
        new_verts = verts[used_verts]
        new_faces = remap[kept_faces]

        # preserve vertex colors if present
        vc = None
        if hasattr(pred_tri.visual, 'vertex_colors') and pred_tri.visual.vertex_colors is not None:
            try:
                vc = np.asarray(pred_tri.visual.vertex_colors)[used_verts]
            except Exception:
                pass

        pred_tri = trimesh.Trimesh(vertices=new_verts, faces=new_faces,
                                   vertex_colors=vc, process=False)

        print(f"  Verts: {n_verts_before:,} → {len(new_verts):,}  "
              f"({100*len(new_verts)/n_verts_before:.1f}% retained)")
        print(f"  Faces: {n_faces_before:,} → {len(new_faces):,}  "
              f"({100*len(new_faces)/n_faces_before:.1f}% retained)")

        pb = pred_tri.bounds
        print(f"  Clipped pred bbox:  "
              f"X=[{pb[0,0]:.2f},{pb[1,0]:.2f}]  "
              f"Y=[{pb[0,1]:.2f},{pb[1,1]:.2f}]  "
              f"Z=[{pb[0,2]:.2f},{pb[1,2]:.2f}]")

    # ------------------------------------------------------------------
    # 2. Load cameras
    # ------------------------------------------------------------------
    print()
    print("Loading cameras ...")
    cameras = parse_cameras_txt(os.path.expanduser(args.camera_params))
    images  = parse_images_txt(os.path.expanduser(args.transforms_json))
    cam_list = build_camera_list(images, cameras)
    print(f"  {len(cam_list)} cameras loaded")
    cam_positions = np.array([c[0] for c in cam_list])
    print(f"  Camera positions bbox:  "
          f"X=[{cam_positions[:,0].min():.2f},{cam_positions[:,0].max():.2f}]  "
          f"Y=[{cam_positions[:,1].min():.2f},{cam_positions[:,1].max():.2f}]  "
          f"Z=[{cam_positions[:,2].min():.2f},{cam_positions[:,2].max():.2f}]")

    # ------------------------------------------------------------------
    # 3. Sample points + normals
    # ------------------------------------------------------------------
    print()
    print(f"Sampling {args.n_samples:,} points per mesh ...")
    pred_pts, pred_nrm = sample_mesh(pred_tri, args.n_samples)
    gt_pts,   gt_nrm   = sample_mesh(gt_tri,   args.n_samples)
    print(f"  Done. pred_pts={pred_pts.shape}  gt_pts={gt_pts.shape}")

    # ------------------------------------------------------------------
    # 4. Visibility culling
    # ------------------------------------------------------------------
    print()
    print("Building raycasting scenes ...")
    scene_pred = build_raycasting_scene(pred_tri)
    scene_gt   = build_raycasting_scene(gt_tri)
    print("  Done.")

    print()
    print(f"Visibility culling PRED points (epsilon={args.epsilon}m) ...")
    t0 = time.time()
    pred_vis = visibility_cull(pred_pts, cam_list, scene_pred,
                                epsilon=args.epsilon)
    print(f"  PRED visible: {pred_vis.sum():,} / {len(pred_pts):,} "
          f"({100*pred_vis.mean():.1f}%)  [{time.time()-t0:.0f}s]")

    print()
    print(f"Visibility culling GT points (epsilon={args.epsilon}m) ...")
    t0 = time.time()
    gt_vis = visibility_cull(gt_pts, cam_list, scene_gt,
                              epsilon=args.epsilon)
    print(f"  GT   visible: {gt_vis.sum():,} / {len(gt_pts):,} "
          f"({100*gt_vis.mean():.1f}%)  [{time.time()-t0:.0f}s]")

    pred_pts_v = pred_pts[pred_vis];  pred_nrm_v = pred_nrm[pred_vis]
    gt_pts_v   = gt_pts[gt_vis];      gt_nrm_v   = gt_nrm[gt_vis]

    if len(pred_pts_v) == 0 or len(gt_pts_v) == 0:
        print()
        print("ERROR: No visible points after culling — "
              "check coordinate frame alignment.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------------
    print()
    print("Computing metrics ...")
    metrics = compute_metrics(pred_pts_v, pred_nrm_v,
                               gt_pts_v,   gt_nrm_v,   THR)

    # ------------------------------------------------------------------
    # 6. Print results
    # ------------------------------------------------------------------
    print()
    print(sep)
    print(f"  Sampled points (visibility-culled):")
    print(f"    Pred points visible: {pred_vis.sum():,} / {len(pred_pts):,}  "
          f"({100*pred_vis.mean():.1f}%)")
    print(f"    GT   points visible: {gt_vis.sum():,} / {len(gt_pts):,}  "
          f"({100*gt_vis.mean():.1f}%)")
    print()
    print(f"  Metrics (threshold = {THR*100:.0f} cm):")
    print(f"    Accuracy (mean):       {metrics['accuracy']*100:.4f} cm")
    print(f"    Completeness (mean):   {metrics['completeness']*100:.4f} cm")
    print(f"    Chamfer:               {metrics['chamfer']*100:.4f} cm")
    print(f"    Precision@{THR*100:.0f}cm:       {metrics['precision']:.4f}")
    print(f"    Recall@{THR*100:.0f}cm:          {metrics['recall']:.4f}")
    print(f"    F-score@{THR*100:.0f}cm:         {metrics['fscore']:.4f}")
    print(f"    Normal Consistency:    {metrics['normal_consistency']:.4f}")
    print(sep)
    print()
    print("  CONTEXT (literature F-score@5cm for ScanNet++ indoor scenes):")
    print("    Vanilla 2DGS:  ~0.40–0.50")
    print("    DN-Splatter:   ~0.55–0.65")
    print("    PGSR:          ~0.60–0.70")
    print("    2DGS-Room:     ~0.65–0.72")
    print(sep)

    # ------------------------------------------------------------------
    # 7. Save JSON
    # ------------------------------------------------------------------
    out_path = args.output
    if out_path is None:
        pred_dir = os.path.dirname(os.path.abspath(
            os.path.expanduser(args.pred_mesh)))
        # go up two levels: .../train/ours_30000/ -> .../  (model root)
        out_dir = os.path.dirname(os.path.dirname(pred_dir))
        out_path = os.path.join(out_dir, 'eval_metrics.json')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result = dict(
        pred_mesh=args.pred_mesh,
        gt_mesh=args.gt_mesh,
        threshold=THR,
        n_samples=args.n_samples,
        epsilon=args.epsilon,
        pred_visible=int(pred_vis.sum()),
        gt_visible=int(gt_vis.sum()),
        **metrics,
    )
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == '__main__':
    main()
