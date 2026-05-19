#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
# gaussian_model.py

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation


def normals_to_quaternions(normals_np):
    """Convert (N,3) unit normals to (N,4) quaternions in wxyz format.

    Computes the shortest rotation from (0,0,1) to each normal, so that
    the surfel z-axis (R[:,2]) aligns with the surface normal.

    Args:
        normals_np: (N, 3) float32 array of surface normals (need not be unit).

    Returns:
        quats: (N, 4) float32 array of quaternions in wxyz order.
    """
    normals_np = normals_np.astype(np.float32)
    norms = np.linalg.norm(normals_np, axis=1, keepdims=True)
    normals_np = normals_np / np.maximum(norms, 1e-8)

    # Rotation from (0,0,1) to n:
    #   v = (0,0,1) × n = (-n_y, n_x, 0)
    #   w = 1 + n_z      (= 1 + dot product)
    # quaternion (w, v_x, v_y, v_z) = (1+n_z, -n_y, n_x, 0)
    N = normals_np.shape[0]
    w = 1.0 + normals_np[:, 2]     # 1 + n_z
    qx = -normals_np[:, 1]          # -n_y
    qy = normals_np[:, 0]           # n_x
    qz = np.zeros(N, dtype=np.float32)

    quats = np.stack([w, qx, qy, qz], axis=1)

    # Edge case: n ≈ (0,0,-1) → w ≈ 0 → degenerate. Use 180° around x-axis.
    anti_parallel = normals_np[:, 2] < -0.9999
    quats[anti_parallel] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    # Normalize
    q_norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.maximum(q_norms, 1e-8)
    return quats


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # Per-surfel LiDAR anchor weights [0,1] — NOT optimizable, propagated on densify/prune
        self._anchor_weight = torch.empty(0)
        # Phase C intrinsic decomposition: per-surfel albedo SH (separate from _features which bake shadows)
        # Initialized empty; populated via init_albedo_from_features() at phase_c_start.
        self._albedo_dc = torch.empty(0)    # (N, 1, 3) — DC albedo term
        self._albedo_rest = torch.empty(0)  # (N, (max_sh+1)^2-1, 3) — higher-order albedo SH
        # v8j dual positions: rendering offset, inactive (zeros) until GTR
        self._xyz_offset = torch.empty(0)   # (N, 3) — added to _xyz for rendering
        self._offset_active = False          # True after GTR activation
        self._offset_optimizer = None        # Created at GTR start
        # Tangent-plane reparameterization: constrains surface Gaussians to LiDAR geometry
        # Activated via enable_tangent_reparam() at tangent_reparam_iter (after densification).
        self._use_tangent_reparam = False
        self._anchor_points = None    # (N_surf, 3) fixed LiDAR anchor positions, no grad
        self._anchor_tu = None        # (N_surf, 3) tangent u vectors, no grad
        self._anchor_tv = None        # (N_surf, 3) tangent v vectors, no grad
        self._anchor_normal = None    # (N_surf, 3) surface normals, no grad
        self._tangent_uv = None       # (N_surf, 2) nn.Parameter — in-plane offset
        self._normal_delta = None     # (N_surf, 1) nn.Parameter — bounded normal offset
        self._surface_mask = None     # (N,) bool — True for surface-constrained Gaussians
        self._planarity_t = None      # (N_surf,) float32, no grad — for adaptive epsilon
        self._epsilon_base = 0.002    # 2mm default normal displacement bound
        # Quadric tangent-plane reparameterization (v10): extends tangent reparam with
        # second-order height-field Q(u,v) = a*u^2 + b*u*v + c*v^2 to handle curved surfaces.
        # Mutually exclusive with _use_tangent_reparam.
        self._quadric_active = False
        self._quad_abc = None            # (N_surf, 3) fixed, no grad: [a, b, c]
        self._quadric_epsilon = None     # (N_surf,)   fixed, no grad: per-surfel epsilon
        # v13: opacity temperature annealing — sharpens sigmoid toward binary during Phase A→B
        self.opacity_temperature = 1.0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        """Rendering position.

        Standard mode: _xyz (+ optional _xyz_offset in dual_position GTR mode).
        Quadric-reparam mode: surface Gaussians use quadric-constrained positions.
        Tangent-reparam mode: surface Gaussians use first-order tangent-constrained positions.
        Gradients flow through the reparameterization to _tangent_uv and _normal_delta.
        """
        if self._quadric_active and self._anchor_points is not None:
            return self._get_xyz_quadric()
        if self._use_tangent_reparam and self._anchor_points is not None:
            return self._get_xyz_tangent()
        if self._offset_active and self._xyz_offset.shape[0] == self._xyz.shape[0]:
            # Detach _xyz so gradients flow ONLY to _xyz_offset (not to frozen geo position)
            return self._xyz.detach() + self._xyz_offset
        return self._xyz

    def _get_xyz_tangent(self):
        """Compute reparameterized positions for tangent-plane mode.

        μ_i = anchor_i + u_i * tu_i + v_i * tv_i + clamp(δ_i, −ε_i, +ε_i) * n_i

        For surface Gaussians (surface_mask=True):  uses _tangent_uv and _normal_delta
        For free Gaussians   (surface_mask=False): uses _xyz directly
        Gradients to _tangent_uv/_normal_delta and _xyz are preserved.
        """
        # Adaptive epsilon: tight on flat surfaces, looser on edges
        eps = self._epsilon_base * torch.clamp(self._planarity_t, min=0.01, max=1.0)  # (N_surf,)
        delta_clamped = torch.clamp(
            self._normal_delta.squeeze(1), -eps, eps
        ).unsqueeze(1)  # (N_surf, 1)

        surface_xyz = (self._anchor_points
                       + self._tangent_uv[:, 0:1] * self._anchor_tu
                       + self._tangent_uv[:, 1:2] * self._anchor_tv
                       + delta_clamped * self._anchor_normal)  # (N_surf, 3)

        if self._surface_mask.all():
            return surface_xyz

        # Mix: surface positions from reparam, free positions from _xyz.
        # Uses masked linear combination to keep autograd graph intact for both paths.
        surf_idx = torch.where(self._surface_mask)[0]  # (N_surf,)
        free_mask_f = (~self._surface_mask).float().unsqueeze(1)  # (N, 1) 0 for surface
        free_contrib = self._xyz * free_mask_f                     # (N, 3) only free positions
        surf_contrib = torch.zeros_like(self._xyz).index_put(      # (N, 3) only surface positions
            (surf_idx,), surface_xyz)
        return free_contrib + surf_contrib

    def _get_xyz_quadric(self):
        """Compute reparameterized positions with second-order quadric correction.

        μ_i = anchor_i + u_i*tu_i + v_i*tv_i + [Q_i(u_i,v_i) + clamp(δ_i, −ε_i, +ε_i)]*n_i

        Q_i(u,v) = a_i*u² + b_i*u*v + c_i*v²   (fixed from LiDAR fit, no grad)
        Gradients flow through u, v (→ _tangent_uv) and δ (→ _normal_delta when unclamped).
        """
        u = self._tangent_uv[:, 0]    # (N_surf,) with grad
        v = self._tangent_uv[:, 1]    # (N_surf,) with grad

        # Quadric height — fixed coefficients, gradient flows through u, v
        a = self._quad_abc[:, 0]      # (N_surf,) no grad
        b = self._quad_abc[:, 1]
        c = self._quad_abc[:, 2]
        Q = a * u**2 + b * u * v + c * v**2  # (N_surf,)

        # Adaptive clamped normal offset — per-surfel epsilon from fit residual
        delta_clamped = torch.clamp(
            self._normal_delta.squeeze(1),    # (N_surf,) with grad
            -self._quadric_epsilon,            # (N_surf,) no grad
            self._quadric_epsilon              # (N_surf,) no grad
        )  # (N_surf,)

        # Surface surfel positions: (N_surf, 3)
        surface_xyz = (self._anchor_points                        # (N_surf, 3) no grad
                       + u.unsqueeze(1) * self._anchor_tu         # tangential
                       + v.unsqueeze(1) * self._anchor_tv
                       + (Q + delta_clamped).unsqueeze(1) * self._anchor_normal)  # normal

        if self._surface_mask.all():
            return surface_xyz

        # Mix: surface surfels via quadric reparam, free surfels via raw _xyz.
        surf_idx = torch.where(self._surface_mask)[0]        # (N_surf,)
        free_mask_f = (~self._surface_mask).float().unsqueeze(1)  # (N, 1)
        free_contrib = self._xyz * free_mask_f
        surf_contrib = torch.zeros_like(self._xyz).index_put(
            (surf_idx,), surface_xyz)
        return free_contrib + surf_contrib

    @property
    def get_xyz_geo(self):
        """Geometric position: pure _xyz without offset. Used for DGS and mesh extraction."""
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity / self.opacity_temperature)

    # ---- Phase C albedo helpers ----
    @property
    def has_albedo(self):
        """True after init_albedo_from_features() has been called."""
        return self._albedo_dc.numel() > 0

    @property
    def get_albedo_dc(self):
        return self._albedo_dc

    @property
    def get_albedo_rest(self):
        return self._albedo_rest

    @property
    def get_albedo(self):
        """Full albedo SH tensor (N, (max_sh+1)^2, 3) — same layout as get_features."""
        return torch.cat((self._albedo_dc, self._albedo_rest), dim=1)

    def init_albedo_from_features(self, feature_lr):
        """Copy _features_dc → _albedo_dc and zeros → _albedo_rest, register in optimizer.

        Call once at phase_c_start.  Returns the phase-C optimizer that covers
        _albedo_dc, _albedo_rest, and _features_rest (residual).
        _features_dc is frozen by the caller after this call.
        """
        n = self._xyz.shape[0]
        n_rest = self._features_rest.shape[1]

        # Initialise albedo from current (GTR-refined) features
        albedo_dc_data = self._features_dc.detach().clone()   # (N, 1, 3)
        albedo_rest_data = torch.zeros(n, n_rest, 3, dtype=torch.float32, device="cuda")

        self._albedo_dc = nn.Parameter(albedo_dc_data.requires_grad_(True))
        self._albedo_rest = nn.Parameter(albedo_rest_data.requires_grad_(True))

        # Freeze features_dc so it cannot absorb the albedo signal
        self._features_dc.requires_grad_(False)

        # Only albedo params here — _features_rest stays in the main optimizer (no double-stepping).
        # env_map is added to this optimizer by the caller (train.py).
        phase_c_params = [
            {'params': [self._albedo_dc],   'lr': feature_lr,        'name': 'alb_dc'},
            {'params': [self._albedo_rest], 'lr': feature_lr / 20.0, 'name': 'alb_rest'},
        ]
        phase_c_optimizer = torch.optim.Adam(phase_c_params, lr=0.0, eps=1e-15)
        n_params = sum(p.numel() for g in phase_c_params for p in g['params'])
        print(f"\n[Phase C] Albedo initialised from features_dc ({n:,} surfels). "
              f"Phase-C params: {n_params:,}. features_dc frozen. "
              f"features_rest continues in main optimizer.")
        return phase_c_optimizer

    def get_albedo_for_eval(self):
        """Return albedo DC as RGB (N,3) in [0,1], for visual inspection."""
        from utils.sh_utils import SH2RGB
        return SH2RGB(self._albedo_dc.squeeze(1)).clamp(0.0, 1.0)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)

        # Normal-aligned initialization: if the point cloud has surface normals,
        # orient each surfel so its z-axis (local normal) matches the LiDAR surface normal.
        normals_np = np.asarray(pcd.normals).astype(np.float32) if pcd.normals is not None else None
        if (normals_np is not None and
                normals_np.shape[0] == fused_point_cloud.shape[0] and
                np.any(normals_np != 0)):
            rots = torch.from_numpy(normals_to_quaternions(normals_np)).cuda()
            print(f"[Init] Normal-aligned rotation initialised from point cloud normals "
                  f"({fused_point_cloud.shape[0]:,} surfels).")
        else:
            rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._anchor_weight = torch.zeros(fused_point_cloud.shape[0], dtype=torch.float32, device="cuda")
        # v8j: xyz offset starts as plain zero tensor (not Parameter; no grad in Phase A)
        self._xyz_offset = torch.zeros_like(fused_point_cloud, device="cuda")
        # Albedo empty until Phase C
        self._albedo_dc = torch.empty(0)
        self._albedo_rest = torch.empty(0)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # Phase C albedo (only when initialised)
        if self.has_albedo:
            for i in range(self._albedo_dc.shape[1]*self._albedo_dc.shape[2]):
                l.append('alb_dc_{}'.format(i))
            for i in range(self._albedo_rest.shape[1]*self._albedo_rest.shape[2]):
                l.append('alb_rest_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        # When quadric reparam is active, save the constrained positions (what the rasterizer uses)
        # so that the PLY faithfully represents the reconstructed surface for mesh extraction.
        if self._quadric_active and self._anchor_points is not None:
            xyz = self.get_xyz.detach().cpu().numpy()
        else:
            xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.has_albedo:
            alb_dc = self._albedo_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            alb_rest = self._albedo_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation,
                                         alb_dc, alb_rest), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_anchor_weights(self, path):
        """Load precomputed LiDAR anchor weights from a .npy file.

        Weights must have been computed for the same initial point cloud
        (sparse/0/points3D.ply) before densification. Call this right after
        Scene initialisation, before the training loop.
        """
        if not os.path.exists(path):
            print(f"[AnchorWeights] File not found: {path}. Using zero weights.")
            return
        weights = np.load(path)
        n_surf = self._xyz.shape[0]
        if weights.shape[0] != n_surf:
            print(f"[AnchorWeights] Warning: {weights.shape[0]} weights ≠ {n_surf} surfels. "
                  f"Weights NOT loaded (did you resume from a checkpoint?)")
            return
        self._anchor_weight = torch.tensor(weights, dtype=torch.float32, device="cuda")
        pct_high = (self._anchor_weight > 0.8).float().mean().item() * 100
        print(f"[AnchorWeights] Loaded {n_surf:,} weights. "
              f"High (>0.8): {pct_high:.1f}%  mean: {self._anchor_weight.mean():.3f}")

    @property
    def get_anchor_weight(self):
        return self._anchor_weight

    def activate_xyz_offset(self, lr=5e-5):
        """GTR (dual_position): promote _xyz_offset to nn.Parameter and create its optimizer."""
        n = self._xyz.shape[0]
        # Ensure offset matches current surfel count (may have changed during densification)
        if self._xyz_offset.shape[0] != n:
            self._xyz_offset = torch.zeros(n, 3, dtype=torch.float32, device="cuda")
        self._xyz_offset = nn.Parameter(self._xyz_offset.detach().clone().requires_grad_(True))
        self._offset_optimizer = torch.optim.Adam(
            [{'params': [self._xyz_offset], 'lr': lr, 'name': 'xyz_offset'}],
            lr=lr, eps=1e-15
        )
        self._offset_active = True
        print(f"[DualPos] _xyz_offset activated: {n:,} surfels, lr={lr:.2e}")

    def enable_tangent_reparam(self, lidar_field, gap_threshold=0.10, epsilon_base=0.002,
                               k=8, max_radius=0.05, min_neighbors=4,
                               planarity_threshold=0.10):
        """Switch eligible surface Gaussians to tangent-plane reparameterization.

        Call AFTER densification ends (tangent_reparam_iter).  Must NOT be called during
        densification — the densification code does not update the tangent params.

        For each Gaussian within gap_threshold of its nearest LiDAR point AND with enough
        LiDAR neighbours for a reliable plane fit AND planarity < planarity_threshold:
          _xyz reparameterized as:  anchor + u*tu + v*tv + clamp(δ, -ε, +ε)*n

        Gaussians near LiDAR but with high planarity (curved geometry) remain as free _xyz
        and continue to receive soft DGS.

        Returns the number of surface-constrained Gaussians.
        """
        self._epsilon_base = epsilon_base
        xyz_np = self._xyz.detach().cpu().numpy().astype(np.float32)
        N = xyz_np.shape[0]

        # Compute tangent frames from LiDAR
        frames = lidar_field.compute_tangent_frames(
            xyz_np, k=k, max_radius=max_radius, min_neighbors=min_neighbors)

        # Surface mask: valid LiDAR neighbours AND close to anchor AND locally flat
        distances = np.linalg.norm(xyz_np - frames['anchor_points'], axis=1)
        near_mask = frames['valid_mask'] & (distances < gap_threshold)
        surface_mask_np = near_mask & (frames['planarity'] < planarity_threshold)
        n_surface = int(surface_mask_np.sum())

        n_near = int(near_mask.sum())
        n_curved = n_near - n_surface
        print(f"[Tangent reparam] Classification: {n_surface:,} planar (hard constraint, "
              f"planarity<{planarity_threshold}), {n_curved:,} curved (soft DGS), "
              f"{N - n_near:,} free (unconstrained)")

        print(f"[Tangent reparam] {n_surface:,}/{N:,} surfels tangent-constrained "
              f"({100.0*n_surface/max(N,1):.1f}%), gap_threshold={gap_threshold}m, "
              f"planarity_threshold={planarity_threshold}")
        print(f"[Tangent reparam] epsilon_base={epsilon_base*1000:.1f}mm, "
              f"adaptive: planarity*epsilon for each surfel")

        device = self._xyz.device

        # Store fixed LiDAR frames (no gradient required)
        self._anchor_points = torch.tensor(
            frames['anchor_points'][surface_mask_np], dtype=torch.float32, device=device)
        self._anchor_tu = torch.tensor(
            frames['tangent_u'][surface_mask_np], dtype=torch.float32, device=device)
        self._anchor_tv = torch.tensor(
            frames['tangent_v'][surface_mask_np], dtype=torch.float32, device=device)
        self._anchor_normal = torch.tensor(
            frames['normals'][surface_mask_np], dtype=torch.float32, device=device)
        self._planarity_t = torch.tensor(
            frames['planarity'][surface_mask_np], dtype=torch.float32, device=device)

        # Compute initial (u, v, δ) from current surfel positions relative to anchors
        current_xyz = self._xyz.detach()[surface_mask_np]  # (N_surf, 3)
        offset = current_xyz - self._anchor_points         # (N_surf, 3)

        u_init = (offset * self._anchor_tu).sum(dim=1, keepdim=True)      # (N_surf, 1)
        v_init = (offset * self._anchor_tv).sum(dim=1, keepdim=True)      # (N_surf, 1)
        delta_init = (offset * self._anchor_normal).sum(dim=1, keepdim=True)  # (N_surf, 1)

        # Clamp initial delta to epsilon bounds (avoid starting outside the constraint)
        eps0 = self._epsilon_base * torch.clamp(self._planarity_t, min=0.01, max=1.0)
        delta_init = torch.clamp(delta_init.squeeze(1), -eps0, eps0).unsqueeze(1)

        self._tangent_uv = nn.Parameter(
            torch.cat([u_init, v_init], dim=1))        # (N_surf, 2)
        self._normal_delta = nn.Parameter(delta_init)  # (N_surf, 1)

        self._surface_mask = torch.tensor(
            surface_mask_np, dtype=torch.bool, device=device)

        # Stats
        mean_plan = float(self._planarity_t.mean().item())
        mean_eps  = float((self._epsilon_base * self._planarity_t.clamp(0.01, 1.0)).mean().item())
        mean_u = float(u_init.abs().mean().item())
        mean_v = float(v_init.abs().mean().item())
        mean_d = float(delta_init.abs().mean().item())
        print(f"[Tangent reparam] Init stats: mean_planarity={mean_plan:.4f}, "
              f"mean_epsilon={mean_eps*1000:.2f}mm")
        print(f"[Tangent reparam] Init offsets: |u|={mean_u*1000:.2f}mm, "
              f"|v|={mean_v*1000:.2f}mm, |δ|={mean_d*1000:.2f}mm")

        self._use_tangent_reparam = True
        return n_surface

    def enable_quadric_reparam(self, lidar_field, gap_threshold=0.10,
                               epsilon_base=0.002, ridge_lambda=0.01,
                               k=8, max_radius=0.05, min_neighbors=4):
        """Activate quadric tangent-plane reparameterization (v10).

        For surfels within gap_threshold of nearest LiDAR point AND with valid PCA frame:
          μ_i = anchor_i + u_i*tu_i + v_i*tv_i + [Q_i(u_i,v_i) + clamp(δ_i,-ε_i,+ε_i)]*n_i

        On flat surfaces: a≈b≈c≈0, reduces to standard tangent plane.
        On curved surfaces: quadric follows local curvature; surfel stays on manifold.

        Call AFTER densification ends (densify_until_iter). Must NOT be called during
        densification — densification code does not update quadric params.

        Returns the number of surface-constrained Gaussians.
        """
        xyz_np = self._xyz.detach().cpu().numpy().astype(np.float32)
        N = xyz_np.shape[0]

        # Get PCA frames + quadric coefficients from LiDAR field
        result = lidar_field.compute_quadric_coefficients(
            xyz_np, k=k, max_radius=max_radius, min_neighbors=min_neighbors,
            epsilon_base=epsilon_base, ridge_lambda=ridge_lambda
        )

        anchor_pts  = result['anchor_points']   # (N, 3)
        tangent_u   = result['tangent_u']        # (N, 3)
        tangent_v   = result['tangent_v']        # (N, 3)
        normal      = result['normals']          # (N, 3)
        quad_abc    = result['quad_abc']         # (N, 3)
        epsilon     = result['epsilon']          # (N,)
        valid_mask  = result['valid_mask']       # (N,) bool
        dist        = result['dist']             # (N,) float32

        # Surface mask: valid PCA neighbours + close to nearest LiDAR point
        surface_mask_np = valid_mask & (dist < gap_threshold)
        n_surface = int(surface_mask_np.sum())
        n_free    = int((~surface_mask_np).sum())
        n_near    = int(valid_mask.sum())

        print(f"[QuadricReparam] Classification: {n_surface:,} surface (quadric constraint), "
              f"{n_free:,} free — {N - n_near:,} no valid LiDAR neighbors, "
              f"{n_near - n_surface:,} valid but dist >= {gap_threshold}m")
        print(f"[QuadricReparam] {n_surface:,}/{N:,} surfels constrained "
              f"({100.0*n_surface/max(N,1):.1f}%), gap_threshold={gap_threshold}m")

        device = self._xyz.device

        # Store surface mask
        self._surface_mask = torch.tensor(surface_mask_np, dtype=torch.bool, device=device)

        # Fixed geometry tensors (no grad)
        self._anchor_points = torch.tensor(
            anchor_pts[surface_mask_np], dtype=torch.float32, device=device)
        self._anchor_tu = torch.tensor(
            tangent_u[surface_mask_np], dtype=torch.float32, device=device)
        self._anchor_tv = torch.tensor(
            tangent_v[surface_mask_np], dtype=torch.float32, device=device)
        self._anchor_normal = torch.tensor(
            normal[surface_mask_np], dtype=torch.float32, device=device)
        self._quad_abc = torch.tensor(
            quad_abc[surface_mask_np], dtype=torch.float32, device=device)
        self._quadric_epsilon = torch.tensor(
            epsilon[surface_mask_np], dtype=torch.float32, device=device)

        # Initialise (u, v, δ) from current surfel positions — ensures continuity (no position jump)
        current_xyz = self._xyz.detach()[surface_mask_np]   # (N_surf, 3)
        offset      = current_xyz - self._anchor_points      # (N_surf, 3)

        u_init = (offset * self._anchor_tu).sum(dim=1)       # (N_surf,)
        v_init = (offset * self._anchor_tv).sum(dim=1)
        h_init = (offset * self._anchor_normal).sum(dim=1)   # normal component of offset

        # δ_init = h_init - Q(u_init, v_init) so that anchor + u*tu + v*tv + (Q+δ)*n == current_xyz
        a = self._quad_abc[:, 0]
        b = self._quad_abc[:, 1]
        c = self._quad_abc[:, 2]
        Q_init = a * u_init**2 + b * u_init * v_init + c * v_init**2
        delta_init = h_init - Q_init

        # Clamp initial delta to epsilon bounds (start within constraint)
        delta_init = torch.clamp(delta_init, -self._quadric_epsilon, self._quadric_epsilon)

        # Learnable parameters
        self._tangent_uv   = nn.Parameter(torch.stack([u_init, v_init], dim=1))   # (N_surf, 2)
        self._normal_delta = nn.Parameter(delta_init.unsqueeze(1))                 # (N_surf, 1)

        # Stats
        eps_mean  = float(self._quadric_epsilon.mean()) * 1000
        eps_max   = float(self._quadric_epsilon.max())  * 1000
        u_mean    = float(u_init.abs().mean()) * 1000
        v_mean    = float(v_init.abs().mean()) * 1000
        d_mean    = float(delta_init.abs().mean()) * 1000
        print(f"[QuadricReparam] Epsilon: mean={eps_mean:.2f}mm, max={eps_max:.2f}mm")
        print(f"[QuadricReparam] Init offsets: |u|={u_mean:.2f}mm, "
              f"|v|={v_mean:.2f}mm, |δ|={d_mean:.2f}mm")

        self._quadric_active     = True
        self._use_tangent_reparam = False   # mutually exclusive
        return n_surface

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        # Anchor weights are not stored in PLY — initialise to zeros and let caller load via load_anchor_weights()
        self._anchor_weight = torch.zeros(xyz.shape[0], dtype=torch.float32, device="cuda")
        # v8j: xyz offset — start as zeros (not active until GTR)
        self._xyz_offset = torch.zeros(xyz.shape[0], 3, dtype=torch.float32, device="cuda")
        self._offset_active = False
        self._offset_optimizer = None

        # Phase C albedo — load if stored in this PLY, otherwise leave empty
        alb_dc_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("alb_dc_")]
        if alb_dc_names:
            alb_dc_names = sorted(alb_dc_names, key=lambda x: int(x.split('_')[-1]))
            alb_dc_data = np.zeros((xyz.shape[0], 3, 1))
            alb_dc_data[:, 0, 0] = np.asarray(plydata.elements[0]["alb_dc_0"])
            alb_dc_data[:, 1, 0] = np.asarray(plydata.elements[0]["alb_dc_1"])
            alb_dc_data[:, 2, 0] = np.asarray(plydata.elements[0]["alb_dc_2"])

            alb_rest_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("alb_rest_")]
            alb_rest_names = sorted(alb_rest_names, key=lambda x: int(x.split('_')[-1]))
            alb_rest_data = np.zeros((xyz.shape[0], len(alb_rest_names)))
            for idx, attr_name in enumerate(alb_rest_names):
                alb_rest_data[:, idx] = np.asarray(plydata.elements[0][attr_name])
            n_rest = (self.max_sh_degree + 1) ** 2 - 1
            alb_rest_data = alb_rest_data.reshape((xyz.shape[0], 3, n_rest))

            self._albedo_dc = nn.Parameter(
                torch.tensor(alb_dc_data, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self._albedo_rest = nn.Parameter(
                torch.tensor(alb_rest_data, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            print(f"[LoadPLY] Loaded albedo params ({xyz.shape[0]:,} surfels)")
        else:
            self._albedo_dc = torch.empty(0)
            self._albedo_rest = torch.empty(0)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self._anchor_weight.shape[0] > 0:
            self._anchor_weight = self._anchor_weight[valid_points_mask]
        if self.has_albedo:
            self._albedo_dc = nn.Parameter(self._albedo_dc[valid_points_mask].requires_grad_(True))
            self._albedo_rest = nn.Parameter(self._albedo_rest[valid_points_mask].requires_grad_(True))
        # v8j: prune offset tensor (plain tensor in Phase A, nn.Parameter in GTR)
        if self._xyz_offset.shape[0] > 0:
            self._xyz_offset = self._xyz_offset[valid_points_mask].detach()
            if self._offset_active:
                self._xyz_offset = nn.Parameter(self._xyz_offset.requires_grad_(True))

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_anchor_weights=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Propagate anchor weights: new surfels inherit parent weight (or zero if none)
        if self._anchor_weight.shape[0] > 0:
            if new_anchor_weights is not None:
                self._anchor_weight = torch.cat([self._anchor_weight, new_anchor_weights], dim=0)
            else:
                self._anchor_weight = torch.cat([
                    self._anchor_weight,
                    torch.zeros(new_xyz.shape[0], dtype=torch.float32, device="cuda")
                ], dim=0)

        # Albedo params: densification happens before Phase C so has_albedo is always False here.
        # Guard included for correctness if ever called after Phase C start.
        if self.has_albedo:
            n_new = new_xyz.shape[0]
            n_rest = self._albedo_rest.shape[1]
            new_alb_dc = torch.zeros(n_new, 1, 3, dtype=torch.float32, device="cuda")
            new_alb_rest = torch.zeros(n_new, n_rest, 3, dtype=torch.float32, device="cuda")
            self._albedo_dc = nn.Parameter(torch.cat([self._albedo_dc.detach(), new_alb_dc], dim=0).requires_grad_(True))
            self._albedo_rest = nn.Parameter(torch.cat([self._albedo_rest.detach(), new_alb_rest], dim=0).requires_grad_(True))

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # v8j: extend _xyz_offset with zeros for new surfels (densification happens in Phase A only)
        if self._xyz_offset.shape[0] > 0:
            n_new = new_xyz.shape[0]
            new_offset = torch.zeros(n_new, 3, dtype=torch.float32, device="cuda")
            self._xyz_offset = torch.cat([self._xyz_offset.detach(), new_offset], dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2,
                          surface_mask=None, lidar_field=None):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # LiDAR-gated densification: only split Gaussians near LiDAR surfaces.
        # surface_mask was computed pre-clone; new clones appended after original N → pad with False.
        if surface_mask is not None:
            n_cur = selected_pts_mask.shape[0]
            if surface_mask.shape[0] < n_cur:
                pad = torch.zeros(n_cur - surface_mask.shape[0], dtype=torch.bool,
                                  device=surface_mask.device)
                surface_mask = torch.cat([surface_mask, pad])
            selected_pts_mask = selected_pts_mask & surface_mask[:n_cur]

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_anchor_weights = (self._anchor_weight[selected_pts_mask].repeat(N)
                              if self._anchor_weight.shape[0] > 0 else None)

        # Surface-directed split positions: snap child Gaussians to nearest LiDAR surface,
        # then reset rotation and scale from local LiDAR geometry.
        if lidar_field is not None and new_xyz.shape[0] > 0:
            new_xyz_np = new_xyz.detach().cpu().numpy().astype(np.float32)
            dists, idx = lidar_field.kdtree.query(new_xyz_np, k=1, workers=-1)
            nearest_pts = lidar_field.lidar_pts[idx]          # (M, 3)
            new_xyz = torch.from_numpy(nearest_pts).float().cuda()
            # Align rotation with LiDAR surface normal
            new_quats = normals_to_quaternions(lidar_field.lidar_normals[idx])
            new_rotation = torch.from_numpy(new_quats).float().cuda()
            # Compute scale from local LiDAR point density
            dists_k, _ = lidar_field.kdtree.query(nearest_pts, k=5, workers=-1)
            local_spacing = np.clip(dists_k.mean(axis=1).astype(np.float32), 1e-6, None)
            new_scale_np = np.stack([local_spacing, local_spacing], axis=1)  # (M, 2)
            new_scaling = torch.from_numpy(np.log(new_scale_np)).float().cuda()

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_anchor_weights)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent,
                          surface_mask=None, lidar_field=None):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        # LiDAR-gated densification: only clone Gaussians near LiDAR surfaces
        if surface_mask is not None:
            selected_pts_mask = selected_pts_mask & surface_mask

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_anchor_weights = (self._anchor_weight[selected_pts_mask]
                              if self._anchor_weight.shape[0] > 0 else None)

        # Surface-directed clone positions: snap to nearest LiDAR surface point,
        # then reset rotation and scale from local LiDAR geometry.
        if lidar_field is not None and new_xyz.shape[0] > 0:
            new_xyz_np = new_xyz.detach().cpu().numpy().astype(np.float32)
            dists, idx = lidar_field.kdtree.query(new_xyz_np, k=1, workers=-1)
            nearest_pts = lidar_field.lidar_pts[idx]          # (M, 3)
            new_xyz = torch.from_numpy(nearest_pts).float().cuda()
            # Align rotation with LiDAR surface normal
            new_quats = normals_to_quaternions(lidar_field.lidar_normals[idx])
            new_rotation = torch.from_numpy(new_quats).float().cuda()
            # Compute scale from local LiDAR point density (k=5 nearest to snapped pos)
            dists_k, _ = lidar_field.kdtree.query(nearest_pts, k=5, workers=-1)
            local_spacing = np.clip(dists_k.mean(axis=1).astype(np.float32), 1e-6, None)
            new_scale_np = np.stack([local_spacing, local_spacing], axis=1)  # (M, 2)
            new_scaling = torch.from_numpy(np.log(new_scale_np)).float().cuda()

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_anchor_weights)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size,
                          lidar_field=None, surface_mask=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent,
                               surface_mask=surface_mask, lidar_field=lidar_field)
        self.densify_and_split(grads, max_grad, extent,
                               surface_mask=surface_mask, lidar_field=lidar_field)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1