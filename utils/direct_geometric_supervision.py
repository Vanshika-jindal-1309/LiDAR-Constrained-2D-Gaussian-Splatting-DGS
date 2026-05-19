"""
Direct Geometric Supervision (DGS) for 2D Gaussian Surfels.

Mathematical insight: standard depth supervision goes through alpha-compositing
  d(p) = Σ αᵢTᵢdᵢ  →  gradient to each surfel diluted by rendering weight (~1/N)

DGS bypasses the rendering equation entirely. For each surfel center xᵢ, find
nearby LiDAR points, fit a local plane, and penalize the signed distance from
that plane. Gradient flows directly to surfel xyz — no compositing dilution.

Loss:
  L_DGS = mean_i(|signed_dist(xᵢ, planeᵢ)|)          ← position
         + λ * mean_i(1 - |n_surfelᵢ · n_planeᵢ|)    ← normal alignment

Differentiable parts:
  - signed_dist = (surfel_xyz · plane_n) - plane_offset  →  grad flows to surfel_xyz
  - cos_sim = surfel_normals · plane_n                    →  grad flows to rotation quaternions

Non-differentiable (fixed LiDAR geometry):
  - KNN query, local plane fitting (PCA) — cached every dgs_interval iterations
"""

import numpy as np
import torch
import time

import laspy
import open3d as o3d
from scipy.spatial import cKDTree


class LiDARSurfaceField:
    """
    Precomputed LiDAR surface field for Direct Geometric Supervision.

    Usage:
        field = LiDARSurfaceField(las_path)          # once, before training loop
        field.update_cache(surfel_xyz_np, k=8)       # every dgs_interval iters
        L_pos, L_norm, n = field.compute_loss(xyz_t, normals_t)   # differentiable
    """

    def __init__(self, las_path, voxel_size=0.005, normal_radius=0.05, normal_max_nn=30):
        """
        Load LiDAR, voxel-downsample, estimate normals, build KD-tree.

        Args:
            las_path:       Path to .las LiDAR file.
            voxel_size:     Downsampling voxel (default 5mm → ~10-15M pts for 26M input).
            normal_radius:  Radius for Open3D normal estimation (default 50mm).
            normal_max_nn:  Max neighbors for normal estimation.
        """
        print(f"[DGS] Initialising LiDARSurfaceField from {las_path}")
        t0 = time.time()

        # ---- Load LAS or PLY ----
        ply_has_normals = False
        if str(las_path).lower().endswith('.ply'):
            pcd_raw = o3d.io.read_point_cloud(str(las_path))
            pts_np = np.asarray(pcd_raw.points).astype(np.float64)
            ply_has_normals = pcd_raw.has_normals()
            print(f"[DGS]   Loaded {len(pts_np):,} raw LiDAR points from PLY "
                  f"(normals={'present' if ply_has_normals else 'absent'})")
        else:
            las = laspy.read(las_path)
            pts_np = np.stack([las.x, las.y, las.z], axis=1).astype(np.float64)
            pcd_raw = None
            print(f"[DGS]   Loaded {len(pts_np):,} raw LiDAR points")

        # ---- Voxel downsample ----
        # For PLY with normals: downsample while keeping normals so we can skip estimation.
        if ply_has_normals:
            pcd = pcd_raw.voxel_down_sample(voxel_size)
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_np)
            pcd = pcd.voxel_down_sample(voxel_size)
        pts_down = np.asarray(pcd.points).astype(np.float32)
        print(f"[DGS]   Downsampled to {len(pts_down):,} pts (voxel={voxel_size}m), "
              f"t={time.time()-t0:.1f}s")

        # ---- Normals: use PLY normals if present, else estimate ----
        t1 = time.time()
        if ply_has_normals and pcd.has_normals():
            normals_down = np.asarray(pcd.normals).astype(np.float32)
            print(f"[DGS]   Using pre-computed normals from PLY ({len(normals_down):,} pts), "
                  f"t={time.time()-t1:.1f}s")
        else:
            pcd_for_normals = o3d.geometry.PointCloud()
            pcd_for_normals.points = o3d.utility.Vector3dVector(pts_down.astype(np.float64))
            pcd_for_normals.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_radius, max_nn=normal_max_nn
                )
            )
            centroid = pts_down.mean(axis=0).astype(np.float64)
            pcd_for_normals.orient_normals_towards_camera_location(centroid)
            normals_down = np.asarray(pcd_for_normals.normals).astype(np.float32)
            print(f"[DGS]   Normals estimated, t={time.time()-t1:.1f}s")

        # ---- Store on CPU ----
        self.lidar_pts = pts_down        # (N_lidar, 3) float32
        self.lidar_normals = normals_down  # (N_lidar, 3) float32

        # ---- Build KD-tree (stays on CPU; queries via scipy) ----
        t2 = time.time()
        self.kdtree = cKDTree(pts_down)
        print(f"[DGS]   KD-tree built ({len(pts_down):,} pts), t={time.time()-t2:.1f}s")

        # ---- Cache state ----
        self._cache_valid = False
        self._plane_normals = None   # (M, 3) float32 numpy
        self._plane_offsets = None   # (M,)   float32 numpy
        self._valid_indices = None   # (M,)   int64   numpy  (indices into surfel array)
        self._planarity = None       # (M,)   float32 numpy  λ₁/λ₂ ratio; low = good plane
        self._n_surfels_at_cache = 0

        # MLS quadric cache (v8q)
        self._mls_valid_indices = None   # (M_q,) int64  numpy
        self._mls_R   = None             # (M_q, 3, 3) float32 numpy — local frame [x_hat|y_hat|n_hat]
        self._mls_x_bar = None           # (M_q, 3)    float32 numpy — weighted centroid
        self._mls_theta = None           # (M_q, 6)    float32 numpy — [a,b,c,d,e,g]
        self._quadric_n_skip = 0         # diagnostic: total skipped surfels

        total = time.time() - t0
        print(f"[DGS] Initialisation done in {total:.1f}s")

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    @property
    def cache_invalid(self):
        return not self._cache_valid

    def invalidate_cache(self):
        """Must be called after any densify/prune step."""
        self._cache_valid = False

    def get_pruning_mask(self, xyz_np, max_distance=0.3):
        """Return boolean mask: True = KEEP (close to LiDAR), False = PRUNE (far from LiDAR).

        Uses the same KD-tree as DGS — no extra data structure needed.

        Args:
            xyz_np:       (N, 3) numpy float32/64 — detached surfel positions.
            max_distance: Surfels with nearest-LiDAR distance >= this are pruned. Default 0.3m.

        Returns:
            keep_mask: (N,) numpy bool — True = keep, False = prune.
        """
        distances, _ = self.kdtree.query(xyz_np.astype(np.float32), k=1, workers=-1)
        return distances < max_distance

    def update_cache(self, surfel_xyz_np, k=8, max_radius=0.05, min_neighbors=4,
                     surfel_normals_np=None, normal_cos_threshold=0.7):
        """
        For each surfel, find K nearest LiDAR points and fit a local plane via PCA.
        Results cached as (plane_normals, plane_offsets, valid_indices).

        Args:
            surfel_xyz_np:       (N, 3) numpy float32/64 — detached surfel positions.
            k:                   Number of nearest LiDAR neighbours.
            max_radius:          Only surfels with >= min_neighbors within this radius are valid.
            min_neighbors:       Minimum neighbours required for plane fitting.
            surfel_normals_np:   (N, 3) numpy float32 — detached surfel normals (optional).
                                 When provided, LiDAR neighbours whose normal differs from the
                                 surfel normal by more than ~arccos(normal_cos_threshold) are
                                 replaced by the surfel's own position before PCA fitting.
                                 This prevents cross-surface plane fits at geometry edges.
            normal_cos_threshold: Cosine similarity threshold for normal filtering.
                                 0.7 ≈ 45°. Higher = stricter. Default: 0.7.
        """
        t0 = time.time()
        N = len(surfel_xyz_np)

        # KNN query (all CPU cores)
        dists, idxs = self.kdtree.query(surfel_xyz_np.astype(np.float32), k=k, workers=-1)
        # dists, idxs: (N, k) — always k valid indices (no distance_upper_bound)

        # Filter: surfels with enough neighbours within max_radius
        valid_nn = dists < max_radius             # (N, k) bool
        valid_counts = valid_nn.sum(axis=1)       # (N,)
        valid_surf_mask = valid_counts >= min_neighbors
        valid_idx = np.where(valid_surf_mask)[0]  # (M,)
        M = len(valid_idx)

        if M == 0:
            print(f"[DGS] WARNING: 0 valid surfels within {max_radius}m. "
                  f"Check LiDAR-scene registration.")
            self._plane_normals = np.zeros((0, 3), dtype=np.float32)
            self._plane_offsets = np.zeros(0, dtype=np.float32)
            self._valid_indices = valid_idx
            self._n_surfels_at_cache = N
            self._cache_valid = True
            return

        # Gather k nearest LiDAR points for valid surfels
        nn_idx = idxs[valid_idx]                  # (M, k) int
        nn_pts = self.lidar_pts[nn_idx]           # (M, k, 3) float32

        # ---- Normal-consistent filtering (optional) ----
        # Replace LiDAR neighbours whose normal disagrees with the surfel normal with
        # the surfel's own position (Approach A). This neutralises cross-surface KNN
        # neighbours at wall ledges and object boundaries before PCA plane fitting.
        n_normal_filtered_total = 0
        n_too_few_consistent = 0
        if surfel_normals_np is not None:
            surf_n = surfel_normals_np[valid_idx].astype(np.float32)   # (M, 3)
            lidar_n_k = self.lidar_normals[nn_idx]                      # (M, k, 3)

            # |cos_sim| handles orientation ambiguity (normals may point inward or outward)
            cos_sim = np.abs((surf_n[:, None, :] * lidar_n_k).sum(axis=-1))  # (M, k)
            consistent = cos_sim >= normal_cos_threshold  # (M, k) bool

            n_consistent = consistent.sum(axis=1)  # (M,)
            n_normal_filtered_total = int((k - n_consistent).sum())
            n_too_few_consistent = int((n_consistent < min_neighbors).sum())

            # Replace inconsistent neighbours with the surfel's own position.
            # These points all map to the same location → their centred displacement
            # is equal → they don't introduce spurious PCA directions.
            surfel_pos = surfel_xyz_np[valid_idx].astype(np.float32)   # (M, 3)
            surfel_expanded = np.broadcast_to(
                surfel_pos[:, None, :], (M, k, 3)
            ).copy()                                                     # (M, k, 3) writable
            inconsistent_3d = ~consistent[:, :, None]                   # (M, k, 1)
            nn_pts = np.where(inconsistent_3d, surfel_expanded, nn_pts)  # (M, k, 3)

        # ---- Vectorised PCA plane fitting on CPU via torch ----
        # Faster than numpy SVD because torch uses batched LAPACK DSYEVD on (M,3,3).
        pts_t = torch.from_numpy(nn_pts)          # (M, k, 3) CPU tensor
        centroids = pts_t.mean(dim=1)             # (M, 3)
        centered = pts_t - centroids.unsqueeze(1)  # (M, k, 3)

        # Covariance C = centered^T @ centered  →  (M, 3, 3)
        cov = torch.bmm(centered.transpose(1, 2), centered)  # (M, 3, 3)

        # eigh returns eigenvalues in ascending order; eigenvectors are columns.
        # Smallest eigenvalue direction = plane normal.
        # Planarity: λ₁/λ₂ — near 0 = good plane; near 1 = edge/corner.
        try:
            eigenvalues, eigvecs = torch.linalg.eigh(cov)  # eigenvalues: (M,3) ascending
            plane_normals = eigvecs[:, :, 0].numpy()        # (M, 3) — min eigenvector
            # λ₁/(λ₂+ε): low → flat plane, high → edge/corner (two dominant directions)
            planarity_ratio = (eigenvalues[:, 0] / (eigenvalues[:, 1] + 1e-8)).cpu().numpy()
            self._planarity = planarity_ratio.astype(np.float32)
        except Exception as exc:
            # Fallback: average nearby LiDAR normals; mark all as non-planar (conservative)
            print(f"[DGS] eigh failed ({exc}), falling back to normal averaging.")
            nn_norms = self.lidar_normals[nn_idx]  # (M, k, 3)
            plane_normals = nn_norms.mean(axis=1)   # (M, 3)
            self._planarity = np.ones(M, dtype=np.float32)  # conservative: treat as non-planar

        # Ensure unit length
        norms = np.linalg.norm(plane_normals, axis=1, keepdims=True)
        plane_normals = plane_normals / np.maximum(norms, 1e-8)

        # Plane offsets:  d = n · centroid  (plane eq: n·x = d)
        centroids_np = centroids.numpy()          # (M, 3)
        plane_offsets = (plane_normals * centroids_np).sum(axis=1)  # (M,)

        self._plane_normals = plane_normals.astype(np.float32)
        self._plane_offsets = plane_offsets.astype(np.float32)
        self._valid_indices = valid_idx
        self._n_surfels_at_cache = N
        self._cache_valid = True

        elapsed = time.time() - t0
        coverage = 100.0 * M / N
        nf_msg = ""
        if surfel_normals_np is not None:
            nf_pct = 100.0 * n_normal_filtered_total / max(M * k, 1)
            nf_msg = (f" | normal_filter: {n_normal_filtered_total:,} neighbours replaced "
                      f"({nf_pct:.1f}% of M×k), {n_too_few_consistent:,} surfels <{min_neighbors} consistent")
        print(f"[DGS] Cache updated: {M:,}/{N:,} surfels valid ({coverage:.1f}%), "
              f"time: {elapsed:.1f}s{nf_msg}")

    # ------------------------------------------------------------------
    # Loss computation (differentiable)
    # ------------------------------------------------------------------

    def compute_loss(self, surfel_xyz, surfel_normals,
                     planarity_threshold=0.3, distance_gate_sigma=0.02, iteration=0):
        """
        Compute DGS position + normal alignment losses with planarity filter + distance gate.

        signed_dist(xᵢ) = (xᵢ · n_plane_i) - d_plane_i     ← differentiable w.r.t. xᵢ
        cos_sim(nᵢ)     = nᵢ · n_plane_i                    ← differentiable w.r.t. nᵢ

        Sign of normal does not matter for position gradient (|·| is used) and
        abs() handles normal orientation ambiguity in the alignment loss.

        Planarity filter: skip surfels whose KNN neighborhood spans two surfaces
          (λ₁/λ₂ ≥ planarity_threshold). Typical values: 0.3 = moderate, 0.15 = strict.

        Distance gate: soft-weight by exp(-|dist|/σ). Surfels already far from any
          LiDAR plane (>2σ) are weighted down; avoids overcorrection where LiDAR coverage
          is sparse and the wrong plane may have been fitted.

        Args:
            surfel_xyz:            (N, 3) tensor, requires_grad=True (gaussians.get_xyz)
            surfel_normals:        (N, 3) tensor (from Gaussian rotation matrices)
            planarity_threshold:   λ₁/λ₂ cutoff; surfels above are filtered out. Default 0.3.
            distance_gate_sigma:   Soft-gate length scale (m). Default 0.02 (20mm).
            iteration:             Current training iteration (for diagnostic logging).

        Returns:
            L_pos:    scalar tensor — mean gated absolute signed distance to fitted planes
            L_norm:   scalar tensor — mean 1 - |cosine similarity| with plane normals
            n_planar: int — number of planar surfels used after filtering
        """
        device = surfel_xyz.device
        zero = torch.tensor(0.0, device=device)

        if not self._cache_valid or self._valid_indices is None or len(self._valid_indices) == 0:
            return zero, zero, 0

        N = surfel_xyz.shape[0]
        valid_idx = self._valid_indices
        M = len(valid_idx)

        # Safety check: surfel count changed since last cache update
        if valid_idx.max() >= N:
            return zero, zero, 0

        # Move cached plane params to GPU (cheap — already float32)
        plane_n = torch.tensor(self._plane_normals, device=device, dtype=torch.float32)  # (M,3)
        plane_d = torch.tensor(self._plane_offsets, device=device, dtype=torch.float32)  # (M,)
        idx_t   = torch.tensor(valid_idx,           device=device, dtype=torch.long)     # (M,)

        # ---- Position (signed distance) — needed for gate even before masking ----
        valid_xyz   = surfel_xyz[idx_t]                             # (M, 3)
        signed_dist = (valid_xyz * plane_n).sum(dim=1) - plane_d   # (M,)

        # ---- Planarity filter (CPU numpy → GPU bool mask) ----
        if self._planarity is not None and planarity_threshold > 0:
            planarity_t  = torch.tensor(self._planarity, device=device, dtype=torch.float32)
            planar_mask  = planarity_t < planarity_threshold        # True = flat surface
        else:
            planar_mask  = torch.ones(M, device=device, dtype=torch.bool)

        # ---- Distance gate (soft weight, no masking) ----
        # exp(-|d|/σ): weight ≈ 1 when surfel is near the plane, decays for outliers
        gate = torch.exp(-signed_dist.detach().abs() / (distance_gate_sigma + 1e-8))  # (M,)

        # ---- Apply planarity mask ----
        n_planar = planar_mask.sum().item()
        if n_planar == 0:
            if iteration % 1000 == 0:
                mean_plan = float(self._planarity.mean()) if self._planarity is not None else -1
                print(f"[DGS] iter={iteration} WARNING: 0 planar surfels after filtering! "
                      f"mean_planarity={mean_plan:.4f}. "
                      f"Try increasing --dgs_planarity_threshold (current={planarity_threshold})")
            return zero, zero, 0

        plane_n_f   = plane_n[planar_mask]                          # (P, 3)
        plane_d_f   = plane_d[planar_mask]                          # (P,)
        idx_f       = idx_t[planar_mask]                            # (P,)
        gate_f      = gate[planar_mask]                             # (P,)

        # ---- Position loss with distance gate (gradient → surfel_xyz) ----
        valid_xyz_f   = surfel_xyz[idx_f]                           # (P, 3)
        signed_dist_f = (valid_xyz_f * plane_n_f).sum(dim=1) - plane_d_f  # (P,)
        L_pos = (signed_dist_f.abs() * gate_f).mean()

        # ---- Normal alignment loss (gradient → surfel rotation) ----
        valid_n_f = surfel_normals[idx_f]                           # (P, 3)
        cos_sim_f = (valid_n_f * plane_n_f).sum(dim=1)             # (P,)
        L_norm    = (1.0 - cos_sim_f.abs()).mean()

        # ---- Diagnostic logging every 1000 iters ----
        if iteration % 1000 == 0:
            n_filtered   = M - n_planar
            mean_plan    = float(self._planarity.mean()) if self._planarity is not None else -1
            mean_gate    = gate_f.mean().item()
            print(f"[DGS] iter={iteration} total_valid={M} planar={n_planar} "
                  f"filtered={n_filtered} ({100*n_filtered/max(M,1):.1f}%) "
                  f"mean_planarity={mean_plan:.4f} mean_gate={mean_gate:.3f} "
                  f"L_pos={L_pos.item():.4f} L_norm={L_norm.item():.4f}")

        return L_pos, L_norm, n_planar

    # ------------------------------------------------------------------
    # Adaptive DGS — per-surfel geometry classification + weighted loss
    # ------------------------------------------------------------------

    def update_cache_adaptive(self, surfel_xyz_np, planarity_k=64,
                              k_flat=8, k_ambig=16, max_radius=0.05,
                              flat_threshold=0.85, skip_threshold=0.5,
                              min_neighbors=4):
        """
        Two-stage adaptive DGS cache update.

        Stage 1 (classification, k=planarity_k):
          For every surfel find planarity_k LiDAR neighbours, fit PCA, compute:
            planarity = 1 - λ_min / (λ_max + ε)   (high = flat surface)
          Classify into: flat (plan>flat_thr) / ambiguous / skip (plan≤skip_thr).

        Stage 2 (plane fitting):
          - Flat group:      k=k_flat  KNN + PCA  (tight fit on clear planar patches)
          - Ambiguous group: k=k_ambig KNN + PCA  (wider neighbourhood for stability)
          - Skip group:      no plane fit

        Cached state (replaces normal cache):
          _adap_flat_normals   (F,3)  _adap_flat_offsets   (F,)
          _adap_flat_idx       (F,)   _adap_flat_plan       (F,)
          _adap_ambig_normals  (A,3)  _adap_ambig_offsets  (A,)
          _adap_ambig_idx      (A,)   _adap_ambig_plan      (A,)
          _adap_stats          dict   {n_flat, n_ambig, n_skip, n_total_valid}
        """
        t0 = time.time()
        N = len(surfel_xyz_np)
        xyz_f32 = surfel_xyz_np.astype(np.float32)

        # ----------------------------------------------------------------
        # Stage 1: k=planarity_k KNN for geometry classification
        # ----------------------------------------------------------------
        dists_cls, idxs_cls = self.kdtree.query(xyz_f32, k=planarity_k, workers=-1)
        valid_nn_cls = dists_cls < max_radius           # (N, k)
        valid_counts_cls = valid_nn_cls.sum(axis=1)    # (N,)
        has_enough = valid_counts_cls >= min_neighbors  # (N,) bool

        # Compute planarity for surfels with enough neighbours
        valid_all_idx = np.where(has_enough)[0]        # (V,)
        V = len(valid_all_idx)
        if V == 0:
            print(f"[DGS-adaptive] WARNING: 0 surfels with ≥{min_neighbors} neighbours in {max_radius}m")
            self._init_empty_adaptive_cache(N)
            self._cache_valid = True
            return

        nn_pts_cls = self.lidar_pts[idxs_cls[valid_all_idx]]  # (V, k, 3)
        pts_t = torch.from_numpy(nn_pts_cls)
        centroids_t = pts_t.mean(dim=1)                         # (V, 3)
        centered = pts_t - centroids_t.unsqueeze(1)             # (V, k, 3)
        cov = torch.bmm(centered.transpose(1, 2), centered)     # (V, 3, 3)
        try:
            evals, _ = torch.linalg.eigh(cov)                   # (V, 3) ascending
            evals_np = evals.numpy().astype(np.float32)
            planarity_all = 1.0 - evals_np[:, 0] / (evals_np[:, 2] + 1e-8)  # (V,)
        except Exception as exc:
            print(f"[DGS-adaptive] eigh failed ({exc}), skipping adaptive update.")
            self._init_empty_adaptive_cache(N)
            self._cache_valid = True
            return

        # Classify
        flat_local   = planarity_all > flat_threshold             # (V,) bool
        skip_local   = planarity_all <= skip_threshold
        ambig_local  = (~flat_local) & (~skip_local)

        flat_global  = valid_all_idx[flat_local]
        ambig_global = valid_all_idx[ambig_local]
        flat_plan    = planarity_all[flat_local]
        ambig_plan   = planarity_all[ambig_local]

        n_flat  = len(flat_global)
        n_ambig = len(ambig_global)
        n_skip  = (skip_local).sum()

        # ----------------------------------------------------------------
        # Stage 2a: Plane fitting for flat group (k=k_flat)
        # ----------------------------------------------------------------
        flat_n, flat_d = self._fit_planes(xyz_f32, flat_global, k=k_flat,
                                          max_radius=max_radius, min_neighbors=min_neighbors)

        # ----------------------------------------------------------------
        # Stage 2b: Plane fitting for ambiguous group (k=k_ambig)
        # ----------------------------------------------------------------
        ambig_n, ambig_d = self._fit_planes(xyz_f32, ambig_global, k=k_ambig,
                                            max_radius=max_radius, min_neighbors=min_neighbors)

        # Cache
        self._adap_flat_normals  = flat_n
        self._adap_flat_offsets  = flat_d
        self._adap_flat_idx      = flat_global
        self._adap_flat_plan     = flat_plan.astype(np.float32)
        self._adap_ambig_normals = ambig_n
        self._adap_ambig_offsets = ambig_d
        self._adap_ambig_idx     = ambig_global
        self._adap_ambig_plan    = ambig_plan.astype(np.float32)
        self._adap_stats = dict(n_flat=n_flat, n_ambig=n_ambig, n_skip=int(n_skip),
                                n_total_valid=V, N=N)
        self._n_surfels_at_cache = N
        self._cache_valid = True

        elapsed = time.time() - t0
        print(f"[DGS-adaptive] Cache updated in {elapsed:.1f}s — "
              f"flat={n_flat:,} ({100*n_flat/V:.0f}%), "
              f"ambig={n_ambig:,} ({100*n_ambig/V:.0f}%), "
              f"skip={n_skip:,} ({100*n_skip/V:.0f}%) "
              f"of {V:,} valid/{N:,} total surfels")

    def _fit_planes(self, xyz_f32, global_idx, k, max_radius, min_neighbors):
        """
        KNN + PCA plane fit for a subset of surfels.

        Args:
            xyz_f32   (N, 3) full surfel array
            global_idx (M,) indices into xyz_f32 to process
        Returns:
            plane_normals (M, 3) float32
            plane_offsets (M,)   float32
        """
        M = len(global_idx)
        if M == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

        batch = xyz_f32[global_idx]                     # (M, 3)
        dists, idxs = self.kdtree.query(batch, k=k, workers=-1)  # (M, k)

        valid_nn = dists < max_radius
        valid_counts = valid_nn.sum(axis=1)
        valid_m = valid_counts >= min_neighbors          # (M,) — may drop some

        # For surfels without enough tight neighbours, expand to whatever is available
        # (we still need a plane; fall back to averaging LiDAR normals for those)
        nn_pts = self.lidar_pts[idxs]                   # (M, k, 3)
        pts_t  = torch.from_numpy(nn_pts)
        centroids_t = pts_t.mean(dim=1)                 # (M, 3)
        centered = pts_t - centroids_t.unsqueeze(1)
        cov = torch.bmm(centered.transpose(1, 2), centered)

        try:
            _, eigvecs = torch.linalg.eigh(cov)
            plane_normals = eigvecs[:, :, 0].numpy()    # (M, 3) — min-eigenvector = normal
        except Exception:
            nn_norms = self.lidar_normals[idxs]         # (M, k, 3)
            plane_normals = nn_norms.mean(axis=1)

        norms = np.linalg.norm(plane_normals, axis=1, keepdims=True)
        plane_normals = plane_normals / np.maximum(norms, 1e-8)

        centroids_np = centroids_t.numpy()
        plane_offsets = (plane_normals * centroids_np).sum(axis=1)

        return plane_normals.astype(np.float32), plane_offsets.astype(np.float32)

    # ------------------------------------------------------------------
    # Bilateral-weighted PCA plane fitting (v8o)
    # ------------------------------------------------------------------

    def update_cache_bilateral(self, surfel_xyz_np, surfel_normals_np,
                               k=8, max_radius=0.05, min_neighbors=4,
                               sigma_c=0.5, sigma_x=0.03):
        """
        Bilateral-weighted PCA plane fitting for each surfel.

        For each surfel i and its k nearest LiDAR neighbors x_j, n_j:
          w_j = exp(-(1 - |n_surfel_i · n_j|) / sigma_c^2)   ← normal alignment kernel
               * exp(-||x_j - mu_i||^2 / sigma_x^2)           ← spatial proximity kernel
          where mu_i is the surfel centre position.

        Weighted centroid: x_bar = Σ w_j x_j / Σ w_j
        Weighted covariance: C = Σ w_j (x_j - x_bar)(x_j - x_bar)^T / Σ w_j
        Plane normal = min eigenvector of C.

        A surfel is skipped if Σ w_j < 0.5 * k (insufficient effective neighbors).
        Results are stored in the same cache fields as update_cache(), so
        compute_loss() works unchanged.

        Args:
            surfel_xyz_np:      (N, 3) numpy float32 — detached surfel positions.
            surfel_normals_np:  (N, 3) numpy float32 — detached surfel normals.
            k:                  Number of nearest LiDAR neighbours.
            max_radius:         Maximum distance for a neighbour to count.
            min_neighbors:      Minimum neighbours within max_radius before attempting fit.
            sigma_c:            Normal-kernel bandwidth. Default 0.5 (~60° half-angle).
            sigma_x:            Spatial-kernel bandwidth (m). Default 0.03 (30mm).
        """
        t0 = time.time()
        N = len(surfel_xyz_np)
        xyz_f32 = surfel_xyz_np.astype(np.float32)

        # KNN query (all CPU cores)
        dists, idxs = self.kdtree.query(xyz_f32, k=k, workers=-1)  # (N, k)

        # Filter: surfels with enough neighbours within max_radius
        valid_nn = dists < max_radius              # (N, k)
        valid_counts = valid_nn.sum(axis=1)        # (N,)
        valid_surf_mask = valid_counts >= min_neighbors
        valid_idx = np.where(valid_surf_mask)[0]   # (M,)
        M = len(valid_idx)

        if M == 0:
            print(f"[DGS-bilateral] WARNING: 0 valid surfels within {max_radius}m. "
                  f"Check LiDAR-scene registration.")
            self._plane_normals = np.zeros((0, 3), dtype=np.float32)
            self._plane_offsets = np.zeros(0, dtype=np.float32)
            self._valid_indices = valid_idx
            self._planarity = np.zeros(0, dtype=np.float32)
            self._n_surfels_at_cache = N
            self._cache_valid = True
            return

        # Gather k nearest LiDAR points and normals for valid surfels
        nn_idx   = idxs[valid_idx]                         # (M, k)
        nn_pts   = self.lidar_pts[nn_idx]                  # (M, k, 3)
        nn_norms = self.lidar_normals[nn_idx]              # (M, k, 3)
        surf_xyz = xyz_f32[valid_idx]                      # (M, 3) — surfel centres (mu_i)
        surf_n   = surfel_normals_np[valid_idx].astype(np.float32)  # (M, 3)

        # ---- Compute bilateral weights ----
        # Normal alignment kernel: exp(-(1 - |n_i · n_j|) / sigma_c^2)
        cos_sim  = np.abs((surf_n[:, None, :] * nn_norms).sum(axis=-1))  # (M, k)
        w_normal = np.exp(-(1.0 - cos_sim) / (sigma_c ** 2 + 1e-12))     # (M, k)

        # Spatial proximity kernel: exp(-||x_j - mu_i||^2 / sigma_x^2)
        diff      = nn_pts - surf_xyz[:, None, :]                          # (M, k, 3)
        sq_dist   = (diff ** 2).sum(axis=-1)                               # (M, k)
        w_spatial = np.exp(-sq_dist / (sigma_x ** 2 + 1e-12))             # (M, k)

        weights = (w_normal * w_spatial).astype(np.float32)  # (M, k)

        # ---- Skip surfels where sum(w_j) < 0.5 * k ----
        w_sum = weights.sum(axis=1)            # (M,)
        weight_threshold = 0.5 * k
        pass_mask = w_sum >= weight_threshold  # (M,) bool
        n_below   = int((~pass_mask).sum())

        valid_idx_final = valid_idx[pass_mask]   # (M_pass,)
        M_pass = len(valid_idx_final)

        if M_pass == 0:
            print(f"[DGS-bilateral] WARNING: 0 surfels pass weight threshold "
                  f"(sum_w >= {weight_threshold:.1f}). "
                  f"sigma_c={sigma_c} may be too tight.")
            self._plane_normals = np.zeros((0, 3), dtype=np.float32)
            self._plane_offsets = np.zeros(0, dtype=np.float32)
            self._valid_indices = valid_idx_final
            self._planarity = np.zeros(0, dtype=np.float32)
            self._n_surfels_at_cache = N
            self._cache_valid = True
            return

        # ---- Weighted PCA via torch (batched LAPACK) ----
        pts_t   = torch.from_numpy(nn_pts[pass_mask])          # (M_pass, k, 3)
        w_t     = torch.from_numpy(weights[pass_mask]).unsqueeze(-1)  # (M_pass, k, 1)
        w_sum_t = w_t.sum(dim=1)                               # (M_pass, 1)

        # Weighted centroid: x_bar = Σ w_j x_j / Σ w_j
        centroids = (pts_t * w_t).sum(dim=1) / w_sum_t         # (M_pass, 3)

        # Weighted covariance: C = Σ w_j (x_j - x_bar)^T (x_j - x_bar) / Σ w_j
        centered          = pts_t - centroids.unsqueeze(1)     # (M_pass, k, 3)
        weighted_centered = centered * w_t                     # (M_pass, k, 3)
        cov = torch.bmm(centered.transpose(1, 2),
                        weighted_centered) / w_sum_t.unsqueeze(-1)  # (M_pass, 3, 3)

        try:
            eigenvalues, eigvecs = torch.linalg.eigh(cov)  # eigenvalues (M_pass, 3) ascending
            plane_normals   = eigvecs[:, :, 0].numpy()     # (M_pass, 3) — min eigenvector
            planarity_ratio = (eigenvalues[:, 0] /
                               (eigenvalues[:, 1] + 1e-8)).cpu().numpy()  # (M_pass,)
        except Exception as exc:
            print(f"[DGS-bilateral] eigh failed ({exc}), falling back to normal averaging.")
            nn_norms_pass   = self.lidar_normals[nn_idx[pass_mask]]
            plane_normals   = nn_norms_pass.mean(axis=1)
            planarity_ratio = np.ones(M_pass, dtype=np.float32)

        # Ensure unit length
        norms_len     = np.linalg.norm(plane_normals, axis=1, keepdims=True)
        plane_normals = plane_normals / np.maximum(norms_len, 1e-8)

        # Plane offsets: d = n · x_bar
        centroids_np  = centroids.numpy()
        plane_offsets = (plane_normals * centroids_np).sum(axis=1)

        # ---- Store cache (same fields as update_cache) ----
        self._plane_normals        = plane_normals.astype(np.float32)
        self._plane_offsets        = plane_offsets.astype(np.float32)
        self._valid_indices        = valid_idx_final
        self._planarity            = planarity_ratio.astype(np.float32)
        self._bilateral_n_skip     = n_below   # diagnostic: surfels below weight threshold
        self._n_surfels_at_cache   = N
        self._cache_valid          = True

        elapsed    = time.time() - t0
        coverage   = 100.0 * M_pass / N
        skip_pct   = 100.0 * n_below / max(M, 1)
        print(f"[DGS-bilateral] Cache updated: {M_pass:,}/{N:,} surfels valid ({coverage:.1f}%), "
              f"skipped {n_below:,}/{M:,} ({skip_pct:.1f}%) below weight threshold "
              f"(sum_w<{weight_threshold:.1f}), time: {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # MLS Quadric DGS (v8q) — height-field quadric fit in local PCA frame
    # ------------------------------------------------------------------

    def update_cache_mls_quadric(self, surfel_xyz_np, surfel_normals_np,
                                  k=8, max_radius=0.05, min_neighbors=4,
                                  sigma_c=0.5, sigma_x=0.03,
                                  uniform_weights=False, min_weight_frac=0.5):
        """
        MLS quadric DGS cache update.

        For each surfel:
          1. KNN + bilateral weights (same as v8o)
          2. Weighted PCA → local frame R = [x_hat | y_hat | n_hat]
          3. Transform k neighbors into local (u,v,z) coordinates
          4. Fit height-field quadric z = a*u^2 + b*u*v + c*v^2 + d*u + e*v + g
             via weighted least squares on the 6×6 system A^T W A θ = A^T W z
          5. Skip if: sum(w) < 0.5*k, cond(A^T W A) > 1e6, |a|+|c| > 500, or NaN

        Cache: _mls_valid_indices, _mls_R, _mls_x_bar, _mls_theta, _quadric_n_skip
        """
        t0 = time.time()
        N = len(surfel_xyz_np)
        xyz_f32 = surfel_xyz_np.astype(np.float32)

        # KNN query
        dists, idxs = self.kdtree.query(xyz_f32, k=k, workers=-1)

        valid_nn = dists < max_radius
        valid_counts = valid_nn.sum(axis=1)
        valid_surf_mask = valid_counts >= min_neighbors
        valid_idx = np.where(valid_surf_mask)[0]   # (M,)
        M = len(valid_idx)

        def _empty_mls_cache(n_skip=0):
            self._mls_valid_indices = np.zeros(0, dtype=np.int64)
            self._mls_R             = np.zeros((0, 3, 3), dtype=np.float32)
            self._mls_x_bar         = np.zeros((0, 3),    dtype=np.float32)
            self._mls_theta         = np.zeros((0, 6),    dtype=np.float32)
            self._quadric_n_skip    = n_skip
            self._n_surfels_at_cache = N
            self._cache_valid = True

        if M == 0:
            print(f"[DGS-MLS] WARNING: 0 valid surfels within {max_radius}m.")
            _empty_mls_cache()
            return

        # Gather neighbor data
        nn_idx   = idxs[valid_idx]                                       # (M, k)
        nn_pts   = self.lidar_pts[nn_idx]                                # (M, k, 3)
        surf_xyz = xyz_f32[valid_idx]                                    # (M, 3)

        # ---- Weights: uniform (v8r) or bilateral (v8q/v8s) ----
        if uniform_weights:
            # All neighbors contribute equally — isolates the quadric contribution.
            # sum_w = k always, so the low-weight skip is never triggered.
            weights     = np.ones((M, k), dtype=np.float32)
            pass_mask   = np.ones(M, dtype=bool)
            n_low_w     = 0
            valid_idx_w = valid_idx
            M_pass      = M
        else:
            nn_norms = self.lidar_normals[nn_idx]                             # (M, k, 3)
            surf_n   = surfel_normals_np[valid_idx].astype(np.float32)       # (M, 3)
            # Normal alignment kernel: exp(-(1 - |n_i · n_j|) / sigma_c^2)
            cos_sim   = np.abs((surf_n[:, None, :] * nn_norms).sum(axis=-1)) # (M, k)
            w_normal  = np.exp(-(1.0 - cos_sim) / (sigma_c ** 2 + 1e-12))
            # Spatial proximity kernel: exp(-||x_j - mu_i||^2 / sigma_x^2)
            diff      = nn_pts - surf_xyz[:, None, :]
            sq_dist   = (diff ** 2).sum(axis=-1)
            w_spatial = np.exp(-sq_dist / (sigma_x ** 2 + 1e-12))
            weights   = (w_normal * w_spatial).astype(np.float32)             # (M, k)
            # Skip surfels where effective weight sum < min_weight_frac * k
            w_sum       = weights.sum(axis=1)                                  # (M,)
            pass_mask   = w_sum >= min_weight_frac * k
            n_low_w     = int((~pass_mask).sum())
            valid_idx_w = valid_idx[pass_mask]
            M_pass      = len(valid_idx_w)

        if M_pass == 0:
            threshold_str = "k (uniform)" if uniform_weights else f"{min_weight_frac:.2f}*k={min_weight_frac*k:.1f}"
            print(f"[DGS-MLS] WARNING: 0 surfels pass weight threshold (need sum_w >= {threshold_str}).")
            _empty_mls_cache(n_skip=n_low_w)
            return

        # ---- Weighted PCA on CPU (torch, same as bilateral) ----
        pts_t    = torch.from_numpy(nn_pts[pass_mask])        # (M_pass, k, 3)
        w_t      = torch.from_numpy(weights[pass_mask])       # (M_pass, k)
        w_t3     = w_t.unsqueeze(-1)                          # (M_pass, k, 1)
        w_sum_t  = w_t.sum(dim=1, keepdim=True)              # (M_pass, 1)

        centroids = (pts_t * w_t3).sum(dim=1) / w_sum_t      # (M_pass, 3)
        centered  = pts_t - centroids.unsqueeze(1)            # (M_pass, k, 3)
        wc        = centered * w_t3
        cov       = torch.bmm(centered.transpose(1, 2), wc) / w_sum_t.unsqueeze(-1)  # (M_pass, 3, 3)

        try:
            eigenvalues, eigvecs = torch.linalg.eigh(cov)    # ascending eigenvalues
            n_hat = eigvecs[:, :, 0]                          # (M_pass, 3) smallest eigenvec
        except Exception as exc:
            print(f"[DGS-MLS] eigh failed ({exc}), falling back to LiDAR normal averaging.")
            n_hat = torch.from_numpy(self.lidar_normals[nn_idx[pass_mask]].mean(axis=1))

        n_hat = n_hat / n_hat.norm(dim=1, keepdim=True).clamp(min=1e-8)  # (M_pass, 3)

        # ---- Gram-Schmidt tangent basis ----
        world_x = torch.tensor([[1., 0., 0.]], dtype=torch.float32).expand(M_pass, -1)
        world_y = torch.tensor([[0., 1., 0.]], dtype=torch.float32).expand(M_pass, -1)
        dot_x   = (n_hat * world_x).sum(dim=1, keepdim=True).abs()       # (M_pass, 1)
        seed    = torch.where(dot_x > 0.9, world_y, world_x)             # (M_pass, 3)
        proj    = (seed * n_hat).sum(dim=1, keepdim=True)
        x_hat   = seed - proj * n_hat
        x_hat   = x_hat / x_hat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        y_hat   = torch.linalg.cross(n_hat, x_hat)
        y_hat   = y_hat / y_hat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        # R columns: x_hat, y_hat, n_hat  →  (M_pass, 3, 3)
        R_t     = torch.stack([x_hat, y_hat, n_hat], dim=2)

        # ---- Transform neighbors into local frame ----
        # local[m,j,:] = (x_j - x_bar) @ R[m]  →  (u, v, z_local)
        local_coords = torch.bmm(centered, R_t)                          # (M_pass, k, 3)
        u = local_coords[:, :, 0]                                        # (M_pass, k)
        v = local_coords[:, :, 1]
        z = local_coords[:, :, 2]

        # ---- Coordinate normalisation (per-surfel) ----
        # Scale u, v by their per-neighbourhood max so that all design-matrix
        # columns are O(1). Without this, when the radius is ~50mm the u²
        # columns are ~1e-3 while the constant column is 1 → cond(ATA) ≈ 1e7,
        # causing the condition-number filter to skip almost every surfel.
        u_scale = u.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)     # (M_pass, 1)
        v_scale = v.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)     # (M_pass, 1)
        u_n = u / u_scale                                                # (M_pass, k)
        v_n = v / v_scale

        # ---- Weighted least-squares quadric fit (in normalised û, v̂ coords) ----
        ones   = torch.ones_like(u_n)
        A      = torch.stack([u_n**2, u_n*v_n, v_n**2, u_n, v_n, ones], dim=2)  # (M_pass, k, 6)
        sqrt_w = w_t.sqrt().unsqueeze(2)                                 # (M_pass, k, 1)
        A_w    = A * sqrt_w                                              # (M_pass, k, 6)  √w·A
        z_w    = z * w_t.sqrt()                                          # (M_pass, k)     √w·z

        ATA    = torch.bmm(A_w.transpose(1, 2), A_w)                    # (M_pass, 6, 6)
        ATz    = torch.bmm(A_w.transpose(1, 2), z_w.unsqueeze(2)).squeeze(2)  # (M_pass, 6)

        reg    = 1e-6 * torch.eye(6, dtype=torch.float32).unsqueeze(0).expand(M_pass, -1, -1)
        ATA_r  = ATA + reg

        # Condition number via eigenvalues of symmetric ATA (faster than SVD)
        try:
            evals_ata = torch.linalg.eigvalsh(ATA_r)                    # (M_pass, 6) ascending
            cond_est  = evals_ata[:, -1] / (evals_ata[:, 0].abs() + 1e-12)
            good_cond = cond_est < 1e6
        except Exception:
            good_cond = torch.ones(M_pass, dtype=torch.bool)

        # Solve M_pass × (6×6) systems (solution is θ in normalised coords)
        try:
            theta_n = torch.linalg.solve(ATA_r, ATz)                    # (M_pass, 6)
        except Exception as exc:
            print(f"[DGS-MLS] linalg.solve failed ({exc}), using zeros.")
            theta_n = torch.zeros(M_pass, 6)

        # Un-normalise θ: û=u/us, v̂=v/vs  →  a=a'/us², b=b'/(us·vs), c=c'/vs², d=d'/us, e=e'/vs, g=g
        us = u_scale.squeeze(1)   # (M_pass,)
        vs = v_scale.squeeze(1)
        col_scales = torch.stack([us**2, us*vs, vs**2, us, vs,
                                   torch.ones(M_pass, dtype=torch.float32)], dim=1)  # (M_pass, 6)
        theta_all  = theta_n / col_scales                                # (M_pass, 6) in original coords

        # Curvature and NaN filters (check on original-space curvatures)
        a_coef   = theta_all[:, 0]
        c_coef   = theta_all[:, 2]
        good_curv = (a_coef.abs() + c_coef.abs()) <= 500.0
        good_nan  = ~torch.isnan(theta_all).any(dim=1)
        valid_q   = good_cond & good_curv & good_nan                    # (M_pass,)
        n_bad_q   = int((~valid_q).sum())
        n_total_skip = n_low_w + n_bad_q

        valid_q_np      = valid_q.numpy()
        valid_idx_final = valid_idx_w[valid_q_np]
        M_final         = len(valid_idx_final)

        self._mls_valid_indices = valid_idx_final.astype(np.int64)
        self._mls_R             = R_t[valid_q].numpy().astype(np.float32)
        self._mls_x_bar         = centroids[valid_q].numpy().astype(np.float32)
        self._mls_theta         = theta_all[valid_q].numpy().astype(np.float32)
        self._quadric_n_skip    = n_total_skip
        self._n_surfels_at_cache = N
        self._cache_valid = True

        elapsed   = time.time() - t0
        coverage  = 100.0 * M_final / N
        skip_pct  = 100.0 * n_total_skip / max(M, 1)
        print(f"[DGS-MLS] Cache updated: {M_final:,}/{N:,} surfels valid ({coverage:.1f}%), "
              f"skipped {n_total_skip:,}/{M:,} ({skip_pct:.1f}%) "
              f"(low_weight={n_low_w}, ill-cond/curv/nan={n_bad_q}), time: {elapsed:.1f}s")

    def compute_mls_loss(self, xyz, normals, lambda_dgs_eff, lambda_dgs_normal,
                         distance_gate_sigma=0.02, iteration=0):
        """
        Differentiable MLS quadric DGS loss.

        For each valid surfel i:
          (u, v, z) = R_i^T (mu_i - x_bar_i)         [R_i, x_bar_i: constants from cache]
          z_target  = a*u^2 + b*u*v + c*v^2 + d*u + e*v + g   [theta: constant]
          delta     = z - z_target
          L_pos    += |delta| * exp(-|delta|/sigma)

        Gradient (autograd, theta/R/x_bar all detached):
          d(delta)/d(mu_i) = n_hat - (2a*u + b*v + d)*x_hat - (2c*v + b*u + e)*y_hat
        On flat regions (a=b=c=0, d=e=0) this reduces to n_hat, matching DGS plane behaviour.

        Returns:
            dgs_pos_loss:  scalar tensor, already multiplied by lambda_dgs_eff
            dgs_norm_loss: scalar tensor, already multiplied by lambda_dgs_normal
            n_valid:       int — number of surfel contributions used
        """
        device = xyz.device
        zero   = torch.tensor(0.0, device=device)

        if (not self._cache_valid or
                self._mls_valid_indices is None or
                len(self._mls_valid_indices) == 0):
            return zero, zero, 0

        N = xyz.shape[0]
        valid_idx = self._mls_valid_indices
        M = len(valid_idx)

        if valid_idx.max() >= N:
            return zero, zero, 0

        # Move cached arrays to GPU as detached constants
        idx_t   = torch.tensor(valid_idx,       device=device, dtype=torch.long)
        R_t     = torch.tensor(self._mls_R,     device=device, dtype=torch.float32)  # (M,3,3)
        xbar_t  = torch.tensor(self._mls_x_bar, device=device, dtype=torch.float32)  # (M,3)
        theta_t = torch.tensor(self._mls_theta, device=device, dtype=torch.float32)  # (M,6)

        a = theta_t[:, 0]; b = theta_t[:, 1]; c = theta_t[:, 2]
        d = theta_t[:, 3]; e = theta_t[:, 4]; g = theta_t[:, 5]

        # Transform surfel positions into per-surfel local frames (gradients flow through xyz)
        mu          = xyz[idx_t]                                          # (M, 3)  grad OK
        delta_mu    = mu - xbar_t                                         # (M, 3)
        # (delta_mu @ R_t)[m] = R_t[m]^T delta_mu[m] (since R columns are basis vectors)
        local       = torch.bmm(delta_mu.unsqueeze(1), R_t).squeeze(1)   # (M, 3)
        u_mu = local[:, 0]; v_mu = local[:, 1]; z_mu = local[:, 2]

        # Quadric height at (u_mu, v_mu) — theta is constant (detached)
        z_target    = a*u_mu**2 + b*u_mu*v_mu + c*v_mu**2 + d*u_mu + e*v_mu + g
        delta       = z_mu - z_target                                     # (M,)

        # Distance gate
        gate        = torch.exp(-delta.detach().abs() / (distance_gate_sigma + 1e-8))

        # Position loss (with gate, scaled by lambda)
        dgs_pos_loss  = lambda_dgs_eff * (delta.abs() * gate).mean()

        # Normal alignment loss — use n_hat (3rd column of R)
        n_hat_t       = R_t[:, :, 2]                                      # (M, 3)
        valid_n       = normals[idx_t]                                    # (M, 3)
        cos_sim       = (valid_n * n_hat_t).sum(dim=1)                   # (M,)
        dgs_norm_loss = lambda_dgs_normal * (1.0 - cos_sim.abs()).mean()

        if iteration % 2000 == 0:
            N_surf   = self._n_surfels_at_cache
            sfrac    = self._quadric_n_skip / max(N_surf, 1) * 100
            raw_pos  = (delta.abs() * gate).mean().item()
            raw_norm = (1.0 - cos_sim.abs()).mean().item()
            print(f"[DGS-MLS] iter={iteration} valid={M:,} "
                  f"skip_frac={sfrac:.1f}% "
                  f"L_pos={raw_pos:.5f} L_norm={raw_norm:.5f}")

        return dgs_pos_loss, dgs_norm_loss, M

    # ------------------------------------------------------------------
    # Tangent-plane reparameterization support
    # ------------------------------------------------------------------

    def compute_tangent_frames(self, surfel_xyz_np, k=8, max_radius=0.05, min_neighbors=4):
        """
        For each surfel compute a local tangent-plane frame from nearby LiDAR points via PCA.

        Returns a dict:
            anchor_points : (N, 3) float32 — nearest LiDAR point for each surfel
            tangent_u     : (N, 3) float32 — largest-eigenvalue PCA direction (in-plane)
            tangent_v     : (N, 3) float32 — middle-eigenvalue PCA direction (in-plane)
            normals       : (N, 3) float32 — smallest-eigenvalue direction (surface normal)
            planarity     : (N,)   float32 — λ_min/(λ_mid+ε): near 0 = flat plane, near 1 = edge
            valid_mask    : (N,)   bool    — True if surfel has ≥ min_neighbors within max_radius
        """
        t0 = time.time()
        N = len(surfel_xyz_np)
        xyz_f32 = surfel_xyz_np.astype(np.float32)

        # 1-NN for anchor points (nearest LiDAR point to each surfel)
        _, nearest_idx = self.kdtree.query(xyz_f32, k=1, workers=-1)
        anchor_points = self.lidar_pts[nearest_idx.flatten()]  # (N, 3)

        # k-NN for plane fitting
        dists, idxs = self.kdtree.query(xyz_f32, k=k, workers=-1)  # (N, k)

        # Valid surfels: enough neighbours within max_radius
        valid_nn = dists < max_radius
        valid_counts = valid_nn.sum(axis=1)
        valid_mask = valid_counts >= min_neighbors
        valid_idx = np.where(valid_mask)[0]   # (M,)
        M = len(valid_idx)

        # Default frames: world-axis aligned (used for invalid surfels)
        tangent_u = np.tile(np.array([1., 0., 0.], dtype=np.float32), (N, 1))
        tangent_v = np.tile(np.array([0., 1., 0.], dtype=np.float32), (N, 1))
        normals   = np.tile(np.array([0., 0., 1.], dtype=np.float32), (N, 1))
        planarity = np.ones(N, dtype=np.float32)  # default: treat as non-planar

        if M > 0:
            nn_idx = idxs[valid_idx]           # (M, k)
            nn_pts = self.lidar_pts[nn_idx]    # (M, k, 3)

            pts_t    = torch.from_numpy(nn_pts)           # (M, k, 3)
            centroids = pts_t.mean(dim=1)                 # (M, 3)
            centered  = pts_t - centroids.unsqueeze(1)    # (M, k, 3)
            cov       = torch.bmm(centered.transpose(1, 2), centered)  # (M, 3, 3)

            try:
                eigenvalues, eigvecs = torch.linalg.eigh(cov)  # ascending eigenvalues
                # Columns of eigvecs: eigvecs[:,i] = eigenvector for eigenvalues[:,i]
                # [0] = smallest (normal), [1] = middle (tangent_v), [2] = largest (tangent_u)
                normals_m   = eigvecs[:, :, 0].numpy().astype(np.float32)   # (M, 3) surface normal
                tangent_v_m = eigvecs[:, :, 1].numpy().astype(np.float32)   # (M, 3) second tangent
                tangent_u_m = eigvecs[:, :, 2].numpy().astype(np.float32)   # (M, 3) first tangent
                planarity_m = (eigenvalues[:, 0] /
                               (eigenvalues[:, 1] + 1e-8)).cpu().numpy().astype(np.float32)
            except Exception as exc:
                print(f"[Tangent] eigh failed ({exc}), falling back to LiDAR normal averaging.")
                nn_norms  = self.lidar_normals[nn_idx].mean(axis=1)   # (M, 3)
                normals_m = nn_norms / np.maximum(
                    np.linalg.norm(nn_norms, axis=1, keepdims=True), 1e-8)

                # Gram-Schmidt tangent basis from normal
                world_x   = np.tile([1., 0., 0.], (M, 1)).astype(np.float32)
                world_y   = np.tile([0., 1., 0.], (M, 1)).astype(np.float32)
                dot_x     = np.abs((normals_m * world_x).sum(axis=1, keepdims=True))
                seed      = np.where(dot_x > 0.9, world_y, world_x)
                tangent_u_m = seed - (seed * normals_m).sum(axis=1, keepdims=True) * normals_m
                tangent_u_m /= np.maximum(np.linalg.norm(tangent_u_m, axis=1, keepdims=True), 1e-8)
                tangent_v_m  = np.cross(normals_m, tangent_u_m)
                tangent_v_m /= np.maximum(np.linalg.norm(tangent_v_m, axis=1, keepdims=True), 1e-8)
                planarity_m  = np.ones(M, dtype=np.float32)

            # Normalise (eigh returns orthonormal columns but numerical noise accumulates)
            for arr in (normals_m, tangent_u_m, tangent_v_m):
                nrm = np.linalg.norm(arr, axis=1, keepdims=True)
                arr /= np.maximum(nrm, 1e-8)

            normals[valid_idx]   = normals_m
            tangent_u[valid_idx] = tangent_u_m
            tangent_v[valid_idx] = tangent_v_m
            planarity[valid_idx] = planarity_m

        coverage = 100.0 * M / N
        print(f"[Tangent] compute_tangent_frames: {M:,}/{N:,} valid ({coverage:.1f}%), "
              f"t={time.time()-t0:.1f}s")

        return dict(
            anchor_points=anchor_points,
            tangent_u=tangent_u,
            tangent_v=tangent_v,
            normals=normals,
            planarity=planarity,
            valid_mask=valid_mask,
        )

    def compute_quadric_coefficients(self, surfel_xyz_np, k=8, max_radius=0.05,
                                       min_neighbors=4, epsilon_base=0.002,
                                       ridge_lambda=0.01, cond_threshold=1000.0):
        """
        For each surfel, compute a local PCA frame + second-order quadric fit.

        The quadric is: h = a*u² + b*u*v + c*v²  (centred at nearest LiDAR anchor point)
        where (u, v, h) are local coordinates in the PCA frame.

        Planarity-based epsilon: ε_i = epsilon_base × clamp(ρ_i, 0.01, 1.0)
        where ρ = λ_smallest / λ_middle (small ρ = flat surface = tight constraint).

        Ridge regularization: (ATA + ridge_lambda * I) θ = ATh
        Biases a,b,c toward zero on flat surfaces where LiDAR noise dominates.

        Returns dict with keys:
            anchor_points : (N, 3) float32 — nearest LiDAR point per surfel
            tangent_u     : (N, 3) float32 — PCA largest-eigenvalue direction
            tangent_v     : (N, 3) float32 — PCA middle-eigenvalue direction
            normals       : (N, 3) float32 — PCA smallest-eigenvalue (surface normal)
            quad_abc      : (N, 3) float32 — quadric coefficients [a, b, c]
            epsilon       : (N,)   float32 — adaptive per-surfel epsilon
            valid_mask    : (N,)   bool    — True if >= min_neighbors within max_radius
            dist          : (N,)   float32 — distance to nearest LiDAR point
        """
        t0 = time.time()
        N = len(surfel_xyz_np)
        xyz_f32 = surfel_xyz_np.astype(np.float32)

        # 1-NN for anchor points and distance
        dists_1, nearest_idx = self.kdtree.query(xyz_f32, k=1, workers=-1)
        nearest_idx = nearest_idx.flatten()
        anchor_points = self.lidar_pts[nearest_idx]          # (N, 3)
        dist_to_anchor = dists_1.flatten().astype(np.float32)  # (N,)

        # k-NN for PCA + quadric
        dists, idxs = self.kdtree.query(xyz_f32, k=k, workers=-1)  # (N, k)

        # Valid surfels: enough neighbours within max_radius
        valid_nn = dists < max_radius
        valid_counts = valid_nn.sum(axis=1)
        valid_mask = valid_counts >= min_neighbors
        valid_idx = np.where(valid_mask)[0]
        M = len(valid_idx)

        # Default frames: world-axis aligned
        tangent_u = np.tile([1., 0., 0.], (N, 1)).astype(np.float32)
        tangent_v = np.tile([0., 1., 0.], (N, 1)).astype(np.float32)
        normals   = np.tile([0., 0., 1.], (N, 1)).astype(np.float32)
        quad_abc  = np.zeros((N, 3), dtype=np.float32)        # default: flat (a=b=c=0)
        epsilon   = np.full(N, epsilon_base, dtype=np.float32)

        if M > 0:
            nn_idx = idxs[valid_idx]           # (M, k)
            nn_pts = self.lidar_pts[nn_idx]    # (M, k, 3)

            # PCA plane fitting via torch (batched LAPACK)
            pts_t     = torch.from_numpy(nn_pts)
            centroids = pts_t.mean(dim=1)         # (M, 3)
            centered  = pts_t - centroids.unsqueeze(1)
            cov       = torch.bmm(centered.transpose(1, 2), centered)  # (M, 3, 3)

            try:
                eigenvalues, eigvecs = torch.linalg.eigh(cov)  # ascending
                # [0]=smallest→normal, [1]=middle→tv, [2]=largest→tu
                normals_m   = eigvecs[:, :, 0].numpy().astype(np.float32)
                tangent_v_m = eigvecs[:, :, 1].numpy().astype(np.float32)
                tangent_u_m = eigvecs[:, :, 2].numpy().astype(np.float32)
            except Exception as exc:
                print(f"[QuadricReparam] eigh failed ({exc}), using LiDAR normal averaging.")
                # Fallback eigenvalues: treat all as flat (planarity=1 → mid epsilon)
                eigenvalues = torch.ones(M, 3, dtype=torch.float32)
                nn_norms  = self.lidar_normals[nn_idx].mean(axis=1)
                normals_m = nn_norms / np.maximum(np.linalg.norm(nn_norms, axis=1, keepdims=True), 1e-8)
                world_x   = np.tile([1., 0., 0.], (M, 1)).astype(np.float32)
                world_y   = np.tile([0., 1., 0.], (M, 1)).astype(np.float32)
                dot_x     = np.abs((normals_m * world_x).sum(axis=1, keepdims=True))
                seed      = np.where(dot_x > 0.9, world_y, world_x)
                tangent_u_m = seed - (seed * normals_m).sum(axis=1, keepdims=True) * normals_m
                tangent_u_m /= np.maximum(np.linalg.norm(tangent_u_m, axis=1, keepdims=True), 1e-8)
                tangent_v_m  = np.cross(normals_m, tangent_u_m)
                tangent_v_m /= np.maximum(np.linalg.norm(tangent_v_m, axis=1, keepdims=True), 1e-8)

            # Normalise eigenvectors
            for arr in (normals_m, tangent_u_m, tangent_v_m):
                nrm = np.linalg.norm(arr, axis=1, keepdims=True)
                arr /= np.maximum(nrm, 1e-8)

            # Quadric fit centred at anchor (nearest LiDAR point)
            anchor_valid = anchor_points[valid_idx]   # (M, 3)
            nn_offset    = nn_pts - anchor_valid[:, np.newaxis, :]  # (M, k, 3)

            # Project neighbors into local PCA frame relative to anchor
            u_local = (nn_offset * tangent_u_m[:, np.newaxis, :]).sum(axis=-1)  # (M, k)
            v_local = (nn_offset * tangent_v_m[:, np.newaxis, :]).sum(axis=-1)
            h_local = (nn_offset * normals_m[:, np.newaxis, :]).sum(axis=-1)

            # Design matrix: [u^2, u*v, v^2]  → (M, k, 3)
            A_mat = np.stack([u_local**2, u_local * v_local, v_local**2], axis=-1)

            # Normal equations: ATA θ = ATh
            ATA = np.einsum('mkd,mke->mde', A_mat, A_mat)    # (M, 3, 3)
            ATh = np.einsum('mkd,mk->md',  A_mat, h_local)   # (M, 3)

            # Ridge regularization: biases a,b,c toward zero.
            # Suppresses noise-driven curvature on flat surfaces while
            # preserving genuine curvature where the data signal is strong.
            reg   = np.eye(3, dtype=np.float32)[np.newaxis] * ridge_lambda
            ATA_r = ATA + reg

            try:
                evals_ata = np.linalg.eigvalsh(ATA_r)        # (M, 3) ascending
                cond_est  = evals_ata[:, -1] / (np.abs(evals_ata[:, 0]) + 1e-12)
                good_cond = cond_est < cond_threshold
            except Exception:
                good_cond = np.ones(M, dtype=bool)

            try:
                abc = np.linalg.solve(ATA_r, ATh)            # (M, 3)
            except Exception:
                abc = np.zeros((M, 3), dtype=np.float32)

            # Clamp ill-conditioned fits to zero (plain tangent plane fallback)
            abc[~good_cond] = 0.0
            # Clamp extreme curvature values and remove NaN/Inf
            curvature_max = 500.0
            abc = np.clip(abc, -curvature_max, curvature_max)
            bad_abc = ~np.isfinite(abc).all(axis=1)
            abc[bad_abc] = 0.0

            # Planarity-based epsilon (same as pure tangent reparam)
            # ρ = λ_smallest / λ_middle — small ρ = flat surface = tight constraint
            eigenvalues_np = eigenvalues.numpy()  # (M, 3) ascending: [smallest, middle, largest]
            lambda_3 = eigenvalues_np[:, 0]  # smallest eigenvalue (normal direction)
            lambda_2 = eigenvalues_np[:, 1]  # middle eigenvalue
            planarity = lambda_3 / np.maximum(np.abs(lambda_2), 1e-12)  # ρ ∈ [0, ∞)
            planarity = np.clip(planarity, 0.01, 1.0).astype(np.float32)
            eps_m = (epsilon_base * planarity).astype(np.float32)

            # Store valid results
            normals[valid_idx]   = normals_m
            tangent_u[valid_idx] = tangent_u_m
            tangent_v[valid_idx] = tangent_v_m
            quad_abc[valid_idx]  = abc.astype(np.float32)
            epsilon[valid_idx]   = eps_m

            abc_v = abc
            n_flat     = int((np.abs(abc_v).max(axis=1) < 1e-4).sum())
            coverage   = 100.0 * M / N
            print(f"[QuadricReparam] compute_quadric_coefficients: {M:,}/{N:,} valid ({coverage:.1f}%), "
                  f"n_flat={n_flat:,} (|abc|<1e-4), t={time.time()-t0:.1f}s")
            print(f"[QuadricReparam] Quadric |abc|: a={np.abs(abc_v[:,0]).mean():.5f}, "
                  f"b={np.abs(abc_v[:,1]).mean():.5f}, c={np.abs(abc_v[:,2]).mean():.5f}")
            print(f"[QuadricReparam] Planarity ρ: mean={planarity.mean():.4f}, "
                  f"min={planarity.min():.4f}, max={planarity.max():.4f}")
            print(f"[QuadricReparam] Epsilon: mean={eps_m.mean()*1000:.2f}mm, "
                  f"max={eps_m.max()*1000:.2f}mm (ε_base={epsilon_base*1000:.1f}mm, "
                  f"ridge_λ={ridge_lambda})")
            n_regularized = int((np.abs(abc_v).max(axis=1) < 0.1).sum())
            print(f"[QuadricReparam] Ridge λ={ridge_lambda}: {n_regularized:,}/{M:,} surfels "
                  f"have |abc|<0.1 (effectively flat after regularization)")

        return dict(
            anchor_points=anchor_points,
            tangent_u=tangent_u,
            tangent_v=tangent_v,
            normals=normals,
            quad_abc=quad_abc,
            epsilon=epsilon,
            valid_mask=valid_mask,
            dist=dist_to_anchor,
        )

    def _init_empty_adaptive_cache(self, N):
        self._adap_flat_normals  = np.zeros((0, 3), dtype=np.float32)
        self._adap_flat_offsets  = np.zeros(0, dtype=np.float32)
        self._adap_flat_idx      = np.zeros(0, dtype=np.int64)
        self._adap_flat_plan     = np.zeros(0, dtype=np.float32)
        self._adap_ambig_normals = np.zeros((0, 3), dtype=np.float32)
        self._adap_ambig_offsets = np.zeros(0, dtype=np.float32)
        self._adap_ambig_idx     = np.zeros(0, dtype=np.int64)
        self._adap_ambig_plan    = np.zeros(0, dtype=np.float32)
        self._adap_stats = dict(n_flat=0, n_ambig=0, n_skip=0, n_total_valid=0, N=N)
        self._n_surfels_at_cache = N

    def compute_adaptive_loss(self, surfel_xyz, surfel_normals,
                              lambda_dgs_eff, lambda_dgs_normal,
                              distance_gate_sigma=0.02, iteration=0):
        """
        Differentiable adaptive DGS loss using per-group cached planes.

        Per-surfel effective weight = planarity × exp(-|dist|/σ) so:
          - Clearly flat surfels (plan≈1.0) near the plane get full weight
          - Ambiguous surfels (plan≈0.65) get ~65% base weight
          - Surfels far from any plane get down-weighted by the gate

        Loss decomposed into flat + ambiguous contributions and combined as a
        count-weighted mean so group sizes don't bias the gradient magnitude.

        Returns:
            dgs_pos_loss  scalar tensor — already multiplied by lambda_dgs_eff
            dgs_norm_loss scalar tensor — already multiplied by lambda_dgs_normal
            stats         dict {n_flat, n_ambig, n_skip, mean_planarity, mean_gate}
        """
        device = surfel_xyz.device
        zero   = torch.tensor(0.0, device=device)

        if not self._cache_valid:
            return zero, zero, {}

        N = surfel_xyz.shape[0]

        # Safety: stale cache
        if self._n_surfels_at_cache != N:
            return zero, zero, {}

        stats = dict(self._adap_stats)
        n_flat  = len(self._adap_flat_idx)
        n_ambig = len(self._adap_ambig_idx)
        n_total = n_flat + n_ambig

        if n_total == 0:
            if iteration % 1000 == 0:
                print(f"[DGS-adaptive] iter={iteration} WARNING: 0 planar/ambiguous surfels. "
                      f"Check --dgs_flat_threshold / --dgs_skip_threshold.")
            return zero, zero, stats

        # Helper: compute gated + planarity-weighted position + normal losses for one group
        def _group_loss(idx_np, normals_np, offsets_np, plan_np):
            if len(idx_np) == 0:
                return None, None, None, None

            idx_t   = torch.tensor(idx_np,  device=device, dtype=torch.long)
            plane_n = torch.tensor(normals_np, device=device, dtype=torch.float32)
            plane_d = torch.tensor(offsets_np, device=device, dtype=torch.float32)
            plan_t  = torch.tensor(plan_np,  device=device, dtype=torch.float32)

            # Safety
            if idx_t.max() >= N:
                return None, None, None, None

            xyz_g   = surfel_xyz[idx_t]                            # (G, 3)
            norm_g  = surfel_normals[idx_t]                        # (G, 3)
            sdist   = (xyz_g * plane_n).sum(dim=1) - plane_d      # (G,)

            gate    = torch.exp(-sdist.detach().abs() /
                                (distance_gate_sigma + 1e-8))      # (G,)
            weight  = plan_t * gate                                 # (G,)  planarity × gate

            pos_loss  = (sdist.abs()              * weight).mean()
            norm_loss = ((1.0 - (norm_g * plane_n).sum(dim=1).abs()) * plan_t).mean()

            return pos_loss, norm_loss, weight.mean().item(), plan_t.mean().item()

        fp, fn, fg, fplan = _group_loss(self._adap_flat_idx,
                                        self._adap_flat_normals,
                                        self._adap_flat_offsets,
                                        self._adap_flat_plan)
        ap, an, ag, aplan = _group_loss(self._adap_ambig_idx,
                                        self._adap_ambig_normals,
                                        self._adap_ambig_offsets,
                                        self._adap_ambig_plan)

        # Count-weighted combination
        if fp is not None and ap is not None:
            wf, wa = n_flat / n_total, n_ambig / n_total
            pos_combined  = wf * fp + wa * ap
            norm_combined = wf * fn + wa * an
            mean_gate = wf * fg + wa * ag
            mean_plan = wf * fplan + wa * aplan
        elif fp is not None:
            pos_combined, norm_combined = fp, fn
            mean_gate, mean_plan = fg, fplan
        else:
            pos_combined, norm_combined = ap, an
            mean_gate, mean_plan = ag, aplan

        dgs_pos_loss  = lambda_dgs_eff    * pos_combined
        dgs_norm_loss = lambda_dgs_normal * norm_combined

        if iteration % 500 == 0:
            print(f"[DGS-adaptive] iter={iteration} "
                  f"flat={n_flat:,} ambig={n_ambig:,} skip={stats.get('n_skip',0):,} | "
                  f"mean_plan={mean_plan:.3f} mean_gate={mean_gate:.3f} | "
                  f"λ_eff={lambda_dgs_eff:.4f} "
                  f"L_pos={pos_combined.item():.5f} L_norm={norm_combined.item():.5f}")

        stats.update(mean_planarity=mean_plan, mean_gate=mean_gate)
        return dgs_pos_loss, dgs_norm_loss, stats
