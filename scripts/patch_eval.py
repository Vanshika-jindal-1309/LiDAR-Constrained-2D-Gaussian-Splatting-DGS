#!/usr/bin/env python3
"""
Patch-based mesh accuracy evaluation.

Divides the GT mesh surface into K spatial patches (k-means on GT points),
then for each patch reports F@1cm/2cm/5cm/10cm, per-patch roughness (RMS
residual from a best-fit plane), and computes Q-Poor (fraction of patches
with F@5cm < 0.5).

Usage:
  conda run -n surfel_splatting python scripts/patch_eval.py \\
    --gt_mesh  scans/mesh_aligned_0.05.ply \\
    --pred_mesh output/.../fuse_post.ply \\
    --output    output/.../patch_eval/ \\
    --n_patches 100 \\
    --samples 500000 \\
    --clip_to_gt_bbox
"""

import argparse
import json
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_mesh(mesh_path, n_samples=500000):
    """Sample points from mesh surface. Returns (N,3) float32."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no triangles: {mesh_path}")
    pcd = mesh.sample_points_uniformly(n_samples)
    return np.asarray(pcd.points).astype(np.float32)


def compute_f_score(pred_pts, gt_pts, thresh):
    """F-score, precision, recall at given threshold (metres)."""
    tree_gt   = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)
    d_pred, _ = tree_gt.query(pred_pts, workers=-1)
    d_gt,   _ = tree_pred.query(gt_pts, workers=-1)
    prec   = (d_pred < thresh).mean()
    recall = (d_gt   < thresh).mean()
    denom  = prec + recall
    f      = 2 * prec * recall / denom if denom > 0 else 0.0
    return float(f), float(prec), float(recall)


def plane_roughness_rms(pts):
    """RMS residual of pts from their best-fit plane (mm output)."""
    if len(pts) < 4:
        return float('nan')
    c = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    normal = Vt[-1]               # smallest singular vector = plane normal
    residuals = (pts - c) @ normal
    return float(np.sqrt((residuals**2).mean()) * 1000)   # → mm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt_mesh',    required=True)
    ap.add_argument('--pred_mesh',  required=True)
    ap.add_argument('--output',     required=True)
    ap.add_argument('--n_patches',  type=int,   default=100,
                    help='Number of k-means patches on GT surface')
    ap.add_argument('--samples',    type=int,   default=500_000,
                    help='Surface samples from each mesh')
    ap.add_argument('--clip_to_gt_bbox', action='store_true',
                    help='Clip pred_mesh samples to GT bounding box + margin')
    ap.add_argument('--bbox_margin', type=float, default=0.3,
                    help='Margin around GT bbox for clipping (m)')
    ap.add_argument('--q_poor_thresh', type=float, default=0.5,
                    help='F@5cm threshold below which a patch is "poor"')
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PatchEval] GT:   {args.gt_mesh}")
    print(f"[PatchEval] Pred: {args.pred_mesh}")

    # ---- Sample both meshes ----
    print(f"  Sampling {args.samples:,} pts from GT mesh ...")
    gt_pts = sample_mesh(args.gt_mesh, args.samples)
    print(f"  Sampling {args.samples:,} pts from pred mesh ...")
    pred_pts = sample_mesh(args.pred_mesh, args.samples)

    # ---- Clip pred to GT bbox ----
    if args.clip_to_gt_bbox:
        lo = gt_pts.min(axis=0) - args.bbox_margin
        hi = gt_pts.max(axis=0) + args.bbox_margin
        mask = ((pred_pts >= lo) & (pred_pts <= hi)).all(axis=1)
        pred_pts = pred_pts[mask]
        print(f"  After bbox clip: {len(pred_pts):,} pred pts "
              f"(GT bbox {lo.round(2)} → {hi.round(2)})")

    if len(pred_pts) < 100:
        print("[ERROR] Too few pred points after clipping!")
        return

    # ---- Global metrics (for reference) ----
    print("\n[Global metrics]")
    thresholds = [0.01, 0.02, 0.05, 0.10]
    global_metrics = {}
    for t in thresholds:
        f, p, r = compute_f_score(pred_pts, gt_pts, t)
        global_metrics[f'F@{int(t*100)}cm'] = f
        global_metrics[f'P@{int(t*100)}cm'] = p
        global_metrics[f'R@{int(t*100)}cm'] = r
        print(f"  F@{int(t*100)}cm = {f:.4f}  prec={p:.4f}  recall={r:.4f}")

    # ---- K-means patch assignment on GT ----
    K = args.n_patches
    print(f"\n[Patches] K-means K={K} on GT surface ...")
    km = MiniBatchKMeans(n_clusters=K, random_state=42, n_init=3, max_iter=100)
    gt_labels = km.fit_predict(gt_pts)
    centroids = km.cluster_centers_

    # Assign pred pts to nearest centroid
    tree_centroids = cKDTree(centroids)
    _, pred_labels = tree_centroids.query(pred_pts, workers=-1)

    # ---- Per-patch metrics ----
    print(f"  Computing per-patch metrics ...")
    patch_results = []
    for k in range(K):
        gt_k   = gt_pts[gt_labels == k]
        pred_k = pred_pts[pred_labels == k]
        n_gt   = len(gt_k)
        n_pred = len(pred_k)

        if n_gt < 10 or n_pred < 10:
            patch_results.append({
                'patch_id': k,
                'centroid': centroids[k].tolist(),
                'n_gt': int(n_gt),
                'n_pred': int(n_pred),
                'F@1cm': None, 'F@2cm': None, 'F@5cm': None, 'F@10cm': None,
                'roughness_mm': None,
                'skipped': True,
            })
            continue

        f1,  _, _ = compute_f_score(pred_k, gt_k, 0.01)
        f2,  _, _ = compute_f_score(pred_k, gt_k, 0.02)
        f5,  _, _ = compute_f_score(pred_k, gt_k, 0.05)
        f10, _, _ = compute_f_score(pred_k, gt_k, 0.10)
        rms = plane_roughness_rms(gt_k)   # GT roughness (flat → 0mm)

        patch_results.append({
            'patch_id': k,
            'centroid': centroids[k].tolist(),
            'n_gt': int(n_gt),
            'n_pred': int(n_pred),
            'F@1cm':  float(f1),
            'F@2cm':  float(f2),
            'F@5cm':  float(f5),
            'F@10cm': float(f10),
            'roughness_mm': float(rms),
            'skipped': False,
        })

    # ---- Aggregate ----
    valid = [p for p in patch_results if not p['skipped']]
    n_valid = len(valid)
    print(f"  Valid patches: {n_valid}/{K}")

    def mean_metric(key):
        vals = [p[key] for p in valid if p[key] is not None]
        return float(np.mean(vals)) if vals else float('nan')

    q_poor_thresh = args.q_poor_thresh
    q_poor = sum(1 for p in valid if p['F@5cm'] is not None and p['F@5cm'] < q_poor_thresh) / max(1, n_valid)

    summary = {
        'global_metrics': global_metrics,
        'patch_summary': {
            'K': K,
            'n_valid': n_valid,
            'mean_F@1cm':  mean_metric('F@1cm'),
            'mean_F@2cm':  mean_metric('F@2cm'),
            'mean_F@5cm':  mean_metric('F@5cm'),
            'mean_F@10cm': mean_metric('F@10cm'),
            'mean_roughness_mm': mean_metric('roughness_mm'),
            'Q_Poor': float(q_poor),
            'Q_Poor_threshold': q_poor_thresh,
        },
        'per_patch': patch_results,
    }

    # ---- Save results ----
    json_path = out_dir / 'patch_eval.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Results saved] {json_path}")

    # ---- Console summary ----
    ps = summary['patch_summary']
    print("\n" + "="*55)
    print("PATCH EVALUATION SUMMARY")
    print("="*55)
    print(f"  Patches: {K}  (valid: {n_valid})")
    print(f"  Global  F@5cm:  {global_metrics['F@5cm']:.4f}")
    print(f"  Mean    F@1cm:  {ps['mean_F@1cm']:.4f}")
    print(f"  Mean    F@2cm:  {ps['mean_F@2cm']:.4f}")
    print(f"  Mean    F@5cm:  {ps['mean_F@5cm']:.4f}")
    print(f"  Mean    F@10cm: {ps['mean_F@10cm']:.4f}")
    print(f"  Mean roughness: {ps['mean_roughness_mm']:.2f} mm")
    print(f"  Q-Poor (F@5cm < {q_poor_thresh:.2f}): {ps['Q_Poor']:.3f} ({q_poor*n_valid:.0f}/{n_valid} patches)")

    # ---- Patch heatmap ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        f5_vals = [p['F@5cm'] if p['F@5cm'] is not None else 0 for p in patch_results]
        rms_vals = [p['roughness_mm'] if p['roughness_mm'] is not None else 0 for p in patch_results]
        centers  = np.array([p['centroid'] for p in patch_results])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, vals, title, cmap, vmin, vmax in [
            (axes[0], f5_vals,  'F@5cm per patch',        'RdYlGn', 0, 1),
            (axes[1], [p['F@1cm'] or 0 for p in patch_results], 'F@1cm per patch', 'RdYlGn', 0, 1),
            (axes[2], rms_vals,  'GT roughness (mm)',      'viridis', 0, None),
        ]:
            if vmax is None:
                vmax = max(vals) if vals else 1
            sc = ax.scatter(centers[:,0], centers[:,1], c=vals,
                            cmap=cmap, vmin=vmin, vmax=vmax, s=80, edgecolors='k', linewidths=0.3)
            plt.colorbar(sc, ax=ax)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_aspect('equal')

        plt.suptitle(f'Patch Eval: Q-Poor={q_poor:.3f}  F@5cm={global_metrics["F@5cm"]:.4f}', fontsize=13)
        plt.tight_layout()
        fig_path = out_dir / 'patch_heatmap.png'
        plt.savefig(str(fig_path), dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Heatmap saved: {fig_path}")
    except Exception as e:
        print(f"  [WARN] heatmap failed: {e}")

    # ---- Per-patch F-score distribution ----
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        f5_valid = [p['F@5cm'] for p in valid]
        axes[0].hist(f5_valid, bins=20, color='steelblue', edgecolor='k')
        axes[0].axvline(q_poor_thresh, color='r', linestyle='--', label=f'Q-Poor threshold={q_poor_thresh}')
        axes[0].set_xlabel('F@5cm'); axes[0].set_ylabel('# patches')
        axes[0].set_title(f'F@5cm distribution  Q-Poor={q_poor:.3f}')
        axes[0].legend()

        thresholds_plot = [0.01, 0.02, 0.05, 0.10]
        means_plot = [ps[f'mean_F@{int(t*100)}cm'] for t in thresholds_plot]
        axes[1].bar([f'F@{int(t*100)}cm' for t in thresholds_plot], means_plot,
                    color=['#d73027','#fc8d59','#fee090','#91cf60'])
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel('Mean F-score (patches)')
        axes[1].set_title('Per-patch mean F-scores')
        for i, (t, v) in enumerate(zip([f'F@{int(t*100)}cm' for t in thresholds_plot], means_plot)):
            axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

        plt.tight_layout()
        dist_path = out_dir / 'patch_distribution.png'
        plt.savefig(str(dist_path), dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Distribution saved: {dist_path}")
    except Exception as e:
        print(f"  [WARN] distribution plot failed: {e}")


if __name__ == '__main__':
    main()
