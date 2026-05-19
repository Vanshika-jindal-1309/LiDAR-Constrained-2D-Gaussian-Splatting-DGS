#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
# train.py

import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, build_rotation
from utils.sh_utils import eval_sh
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


def get_surfel_normals(gaussians):
    """Extract world-space normals from Gaussian rotation quaternions.

    Each 2D Gaussian has a rotation quaternion → 3×3 matrix R.
    The surfel normal is the third column R[:, :, 2] (z-axis of the local frame).
    Gradients flow back through get_rotation → _rotation for normal alignment loss.
    """
    R = build_rotation(gaussians.get_rotation)       # (N, 3, 3)
    normals = R[:, :, 2]                              # (N, 3)
    norms = normals.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return normals / norms                            # (N, 3) unit normals


def _make_gaussian_kernel_2d(kernel_size, sigma, device):
    """Create a (1,1,k,k) Gaussian blur kernel for image-space anchor weight smoothing."""
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    return kernel_2d.view(1, 1, kernel_size, kernel_size)


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             lambda_depth=0.0, lambda_lidar_normal=0.0, phase_b_start=20000, use_anchor_weights=False,
             lambda_dgs=0.0, lambda_dgs_normal=0.5, lambda_concentrate=0.0,
             dgs_interval=500, dgs_k=8, dgs_radius=0.05, dgs_start_iter=10000, las_path=None,
             dgs_planarity_threshold=0.3, dgs_distance_gate_sigma=0.02,
             dgs_warmup_start=-1, dgs_warmup_lambda=0.02,
             # Geometry-Tethered Refinement (GTR) — small xyz lr + DGS tether instead of hard freeze
             soft_phase_b=False, phase_b_xyz_lr=1e-6, phase_b_dgs_lambda=0.05,
             # Dual positions — _xyz_offset unlocks in GTR phase for photometric tuning
             dual_position=False, lambda_offset=100.0, offset_lr=5e-5,
             # v8l: Opacity entropy — push opacities toward 0 or 1 (binary)
             lambda_opacity_entropy=0.0,
             # v8m: Adaptive DGS — per-surfel planarity classification
             adaptive_dgs=False, dgs_planarity_k=64,
             dgs_flat_threshold=0.85, dgs_skip_threshold=0.5,
             # v8n: Normal-consistent KNN filtering — fix cross-surface plane fits at edges
             dgs_normal_filter=False, dgs_normal_cos_threshold=0.7,
             # v8o: Bilateral-weighted PCA plane fitting
             dgs_bilateral=False, dgs_bilateral_sigma_c=0.5, dgs_bilateral_sigma_x=0.03,
             # v8q: MLS quadric DGS
             dgs_mls_quadric=False,
             # v8r: MLS quadric without bilateral (uniform weights)
             # v8s: MLS quadric with softened bilateral (sigma_x=60mm, min_weight_frac=0.25)
             dgs_mls_uniform_weights=False, dgs_mls_min_weight_frac=0.5,
             # LiDAR-proximity pruning: remove floater Gaussians far from any LiDAR surface
             lidar_prune_distance=0.0,
             # Z-axis pruning: remove Gaussians above a Z threshold (for outside-room cameras)
             prune_z_max=None,
             # GTR: optionally keep scaling/opacity trainable so dragged-in Gaussians can re-optimize
             phase_b_unfreeze_scale_opacity=False,
             # LiDAR-gated densification: only densify near LiDAR surfaces
             lidar_gated_densify=False, densify_surface_radius=0.3,
             # Geometric pruning: remove floaters far from LiDAR AND low opacity AND small scale
             geo_prune_distance=0.5,
             # Tangent-plane reparameterization: constrain surfels to LiDAR manifold
             use_tangent_reparam=False, tangent_reparam_iter=10000,
             epsilon_base=0.002, tangent_gap_threshold=0.10,
             tangent_planarity_threshold=0.10,
             # v10: Quadric tangent-plane reparameterization
             use_quadric_reparam=False, quadric_epsilon_base=0.002,
             quadric_ridge_lambda=0.01, quadric_gap_threshold=0.10,
             # v12: Alternating DGS — even iters use depth+normal, odd iters use DGS (no simultaneous interference)
             alternate_dgs=False):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # Adaptive anchor weights: precomputed per-surfel LiDAR density weights
    anchor_kernel = None
    if use_anchor_weights:
        anchor_path = os.path.join(dataset.model_path, "anchor_weights.npy")
        gaussians.load_anchor_weights(anchor_path)
        # 11×11 Gaussian blur kernel to create smooth falloff at LiDAR coverage boundaries
        anchor_kernel = _make_gaussian_kernel_2d(kernel_size=11, sigma=3.0, device="cuda")
        print(f"[AnchorWeights] Blur kernel ready (11×11, σ=3.0). "
              f"Boundary pixels down-weighted to reduce edge bias.")

    # ---- Direct Geometric Supervision (DGS) / Tangent-Plane Reparameterization ----
    lidar_field = None
    if (lambda_dgs > 0.0 or use_tangent_reparam or use_quadric_reparam) and las_path:
        from utils.direct_geometric_supervision import LiDARSurfaceField
        lidar_field = LiDARSurfaceField(las_path, voxel_size=0.005)
        warmup_desc = (f", warmup λ={dgs_warmup_lambda} from iter {dgs_warmup_start}"
                       if dgs_warmup_start >= 0 else "")
        if adaptive_dgs:
            print(f"[DGS] Enabled (ADAPTIVE): lambda_dgs={lambda_dgs}, lambda_dgs_normal={lambda_dgs_normal}, "
                  f"interval={dgs_interval}, planarity_k={dgs_planarity_k}, "
                  f"k_flat={dgs_k}, k_ambig={dgs_k*2}, radius={dgs_radius}m, "
                  f"start_iter={dgs_start_iter}{warmup_desc}, "
                  f"flat_thr={dgs_flat_threshold}, skip_thr={dgs_skip_threshold}")
        else:
            print(f"[DGS] Enabled: lambda_dgs={lambda_dgs}, lambda_dgs_normal={lambda_dgs_normal}, "
                  f"lambda_concentrate={lambda_concentrate}, interval={dgs_interval}, "
                  f"k={dgs_k}, radius={dgs_radius}m, start_iter={dgs_start_iter}{warmup_desc}, "
                  f"planarity_threshold={dgs_planarity_threshold}, "
                  f"distance_gate_sigma={dgs_distance_gate_sigma}m")
    elif lambda_concentrate > 0.0:
        print(f"[DGS] Depth concentration loss enabled: lambda_concentrate={lambda_concentrate}")

    # Tangent-plane reparameterization optimizer (created at tangent_reparam_iter)
    tangent_reparam_optimizer = None
    if use_tangent_reparam:
        print(f"[TangentReparam] Will activate at iteration {tangent_reparam_iter}. "
              f"epsilon_base={epsilon_base}m, gap_threshold={tangent_gap_threshold}m, "
              f"planarity_threshold={tangent_planarity_threshold}. "
              f"Flat surfels (planarity<threshold) get hard constraint; "
              f"curved surfels continue with soft DGS.")

    quadric_reparam_optimizer = None
    if use_quadric_reparam:
        print(f"[QuadricReparam v10b] Will activate at iteration {opt.densify_until_iter}. "
              f"epsilon_base={quadric_epsilon_base*1000:.1f}mm (planarity-scaled), "
              f"ridge_lambda={quadric_ridge_lambda}, "
              f"gap_threshold={quadric_gap_threshold}m. "
              f"Ridge-regularized quadric + planarity epsilon (fixes v10 noise issue).")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_depth_for_log = 0.0
    ema_dgs_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # ---- GTR: override position LR after scheduler to maintain tether strength ----
        if soft_phase_b and phase_b_start > 0 and iteration > phase_b_start:
            for pg in gaussians.optimizer.param_groups:
                if pg["name"] == "xyz":
                    pg["lr"] = phase_b_xyz_lr
                    break

        # Quadric reparam: keep quadric_uv LR consistent with position LR during GTR
        if (soft_phase_b and phase_b_start > 0 and iteration > phase_b_start
                and quadric_reparam_optimizer is not None):
            for pg in quadric_reparam_optimizer.param_groups:
                if pg['name'] == 'quadric_uv':
                    pg['lr'] = phase_b_xyz_lr

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # ---- Tangent-plane reparameterization: activate BEFORE render so the first
        #      forward pass already uses the reparameterized positions. ----
        if (use_tangent_reparam and lidar_field is not None
                and iteration == tangent_reparam_iter
                and not gaussians._use_tangent_reparam):
            n_surf = gaussians.enable_tangent_reparam(
                lidar_field,
                gap_threshold=tangent_gap_threshold,
                epsilon_base=epsilon_base,
                k=dgs_k,
                max_radius=dgs_radius,
                planarity_threshold=tangent_planarity_threshold,
            )
            n_total = gaussians.get_xyz.shape[0]
            n_free = n_total - n_surf
            tangent_reparam_optimizer = torch.optim.Adam([
                {'params': [gaussians._tangent_uv],   'lr': 1e-4, 'name': 'tangent_uv'},
                {'params': [gaussians._normal_delta], 'lr': 1e-5, 'name': 'normal_delta'},
            ], eps=1e-15)
            print(f"\n[TangentReparam] Activated at iter {iteration}: "
                  f"{n_surf}/{n_total} surface Gaussians ({100*n_surf/max(n_total,1):.1f}%), "
                  f"{n_free} free. epsilon_base={epsilon_base}m")

        # ---- Quadric tangent-plane reparameterization: activate at densify_until_iter ----
        if (use_quadric_reparam and lidar_field is not None
                and iteration == opt.densify_until_iter
                and not gaussians._quadric_active):
            n_surf = gaussians.enable_quadric_reparam(
                lidar_field,
                gap_threshold=quadric_gap_threshold,
                epsilon_base=quadric_epsilon_base,
                ridge_lambda=quadric_ridge_lambda,
                k=dgs_k,
                max_radius=dgs_radius,
            )
            n_total = gaussians.get_xyz.shape[0]
            quadric_reparam_optimizer = torch.optim.Adam([
                {'params': [gaussians._tangent_uv],   'lr': 1e-4, 'name': 'quadric_uv'},
                {'params': [gaussians._normal_delta], 'lr': 1e-5, 'name': 'quadric_delta'},
            ], eps=1e-15)
            print(f"\n[QuadricReparam] Activated at iter {iteration}: "
                  f"{n_surf:,}/{n_total:,} surface Gaussians "
                  f"({100*n_surf/max(n_total,1):.1f}%), "
                  f"{n_total-n_surf:,} free. "
                  f"quadric_uv lr=1e-4, quadric_delta lr=1e-5")

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        if viewpoint_cam.gt_alpha_mask is not None:
            mask = viewpoint_cam.gt_alpha_mask.cuda()  # 1=valid, 0=masked
            Ll1 = l1_loss(image * mask, gt_image * mask)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image * mask, gt_image * mask))
        else:
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        if viewpoint_cam.gt_alpha_mask is not None:
            mask = viewpoint_cam.gt_alpha_mask.cuda()
            normal_loss = lambda_normal * (normal_error * mask).mean()
            dist_loss = lambda_dist * (rend_dist * mask).mean()
        else:
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()

        # ---- Geometry-Tethered Refinement (GTR): tether/freeze geometry for appearance refinement ----
        if iteration == phase_b_start and phase_b_start > 0:
            if soft_phase_b:
                # GTR mode: rotation always frozen; scaling/opacity frozen unless
                # --phase_b_unfreeze_scale_opacity is set (lets dragged-in Gaussians re-optimize)
                gaussians._rotation.requires_grad_(False)
                if not phase_b_unfreeze_scale_opacity:
                    gaussians._scaling.requires_grad_(False)
                    gaussians._opacity.requires_grad_(False)
                frozen_str = "rotation frozen" if phase_b_unfreeze_scale_opacity else "rotation/scaling/opacity frozen"
                print(f"\n[GTR] Geometry-Tethered Refinement activated at iteration {iteration}. "
                      f"Position LR → {phase_b_xyz_lr}, DGS tether λ={phase_b_dgs_lambda}. "
                      f"{frozen_str}, position still active.")
            elif dual_position:
                # Dual position: freeze geo xyz, unlock _xyz_offset for photometric tuning
                gaussians._xyz.requires_grad_(False)
                gaussians._rotation.requires_grad_(False)
                gaussians._scaling.requires_grad_(False)
                gaussians._opacity.requires_grad_(False)
                gaussians.activate_xyz_offset(offset_lr)
                print(f"\n[GTR - DUAL POS] Geo xyz frozen. _xyz_offset activated "
                      f"(lr={offset_lr}) at iteration {iteration}.")
            else:
                gaussians._xyz.requires_grad_(False)
                gaussians._rotation.requires_grad_(False)
                gaussians._scaling.requires_grad_(False)
                gaussians._opacity.requires_grad_(False)
                print(f"\n[GTR - Hard Freeze] Froze geometry (xyz/rotation/scaling/opacity) at iteration {iteration}. "
                      f"Training color (SH features) only.")

            # Tangent reparam: reduce in-plane lr, freeze normal displacement
            if gaussians._use_tangent_reparam and tangent_reparam_optimizer is not None:
                for pg in tangent_reparam_optimizer.param_groups:
                    if pg['name'] == 'tangent_uv':
                        pg['lr'] = 1e-6
                    elif pg['name'] == 'normal_delta':
                        pg['lr'] = 0.0
                print(f"[TangentReparam GTR] tangent_uv lr → 1e-6, normal_delta lr → 0.0")

            # Quadric reparam GTR: reduce in-plane lr, freeze normal displacement
            if gaussians._quadric_active and quadric_reparam_optimizer is not None:
                for pg in quadric_reparam_optimizer.param_groups:
                    if pg['name'] == 'quadric_uv':
                        pg['lr'] = phase_b_xyz_lr
                    elif pg['name'] == 'quadric_delta':
                        pg['lr'] = 0.0
                print(f"[QuadricReparam GTR] quadric_uv lr → {phase_b_xyz_lr:.2e}, "
                      f"quadric_delta lr → 0.0 (frozen)")

        # ---- LiDAR depth supervision (Phase A only) ----
        lidar_depth_loss = torch.tensor(0.0, device="cuda")
        lidar_normal_loss = torch.tensor(0.0, device="cuda")

        in_phase_a = (phase_b_start <= 0) or (iteration < phase_b_start)
        if in_phase_a and lambda_depth > 0.0 and viewpoint_cam.gt_depth is not None:
            gt_depth = viewpoint_cam.gt_depth  # (H, W) already on data_device

            # Use median depth for LiDAR comparison — sharper, no blending noise.
            # render_depth_median is the depth of the dominant Gaussian per ray,
            # which matches what LiDAR measures (the actual surface depth).
            render_median = render_pkg['render_depth_median']  # (1, H, W)

            # Valid mask: LiDAR has data AND rendered median depth is non-zero AND pixel not in black border
            valid_depth = (gt_depth > 0) & (render_median[0] > 0)
            if viewpoint_cam.gt_alpha_mask is not None:
                valid_depth = valid_depth & (viewpoint_cam.gt_alpha_mask.squeeze(0) > 0)

            n_valid = valid_depth.sum()
            if n_valid > 100:
                if use_anchor_weights and anchor_kernel is not None:
                    # Per-pixel weight: 1.0 deep inside LiDAR coverage, tapering to ~0 at edges.
                    # This down-weights boundary pixels that may be near depth discontinuities
                    # (occlusion edges, furniture boundaries) where interpolation was unreliable.
                    with torch.no_grad():
                        depth_binary = (gt_depth > 0).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                        anchor_map = F.conv2d(depth_binary, anchor_kernel, padding=5).squeeze()  # (H,W)
                    lidar_depth_loss = lambda_depth * (
                        anchor_map[valid_depth] *
                        torch.abs(render_median[0][valid_depth] - gt_depth[valid_depth])
                    ).mean()
                else:
                    lidar_depth_loss = lambda_depth * l1_loss(
                        render_median[0][valid_depth], gt_depth[valid_depth]
                    )

            # ---- LiDAR normal supervision ----
            if (lambda_lidar_normal > 0.0 and
                    viewpoint_cam.gt_normal is not None and n_valid > 100):
                gt_normal_cam = viewpoint_cam.gt_normal  # (3, H, W) world-space
                rend_normal = render_pkg['rend_normal']  # (3, H, W) world-space

                # Valid where LiDAR normal is non-zero and depth is valid
                normal_magnitude = gt_normal_cam.abs().sum(dim=0)  # (H, W)
                valid_normal = valid_depth & (normal_magnitude > 0.1)

                if valid_normal.sum() > 100:
                    # Use absolute cosine to be robust to normal sign ambiguity
                    cos_sim = (rend_normal * gt_normal_cam).sum(dim=0)  # (H, W)
                    lidar_normal_loss = lambda_lidar_normal * (
                        1.0 - cos_sim[valid_normal].abs()
                    ).mean()

        # ---- Direct Geometric Supervision (Phase A only) ----
        dgs_pos_loss = torch.tensor(0.0, device="cuda")
        dgs_norm_loss = torch.tensor(0.0, device="cuda")
        conc_loss = torch.tensor(0.0, device="cuda")

        # Compute effective DGS lambda based on warmup schedule:
        #   iter < effective_warmup_start:               lambda_eff = 0
        #   effective_warmup_start <= iter < dgs_start_iter: lambda_eff = dgs_warmup_lambda
        #   iter >= dgs_start_iter:                      lambda_eff = lambda_dgs
        #   iter >= phase_b_start (soft_phase_b only):   lambda_eff = phase_b_dgs_lambda (tether)
        effective_warmup_start = dgs_warmup_start if dgs_warmup_start >= 0 else dgs_start_iter
        if soft_phase_b and phase_b_start > 0 and iteration >= phase_b_start:
            # In GTR: DGS continues as a spring tether at reduced lambda
            lambda_dgs_eff = phase_b_dgs_lambda if lambda_dgs > 0.0 else 0.0
        elif iteration >= dgs_start_iter:
            lambda_dgs_eff = lambda_dgs
        elif dgs_warmup_start >= 0 and iteration >= dgs_warmup_start:
            lambda_dgs_eff = dgs_warmup_lambda
        else:
            lambda_dgs_eff = 0.0

        # DGS is active in Phase A always; also in GTR as a spring tether.
        # When tangent reparam is active: DGS still applies to curved (non-surface) Gaussians.
        # _surface_mask.all() == True means every Gaussian is tangent-constrained → no DGS needed.
        in_soft_phase_b = soft_phase_b and phase_b_start > 0 and iteration >= phase_b_start
        _reparam_active = gaussians._use_tangent_reparam or gaussians._quadric_active
        _has_free_surfels = (
            _reparam_active and
            gaussians._surface_mask is not None and
            not gaussians._surface_mask.all()
        )
        dgs_active = (
            (in_phase_a or in_soft_phase_b) and
            (lambda_dgs_eff > 0.0) and
            (lidar_field is not None) and
            (not _reparam_active or _has_free_surfels)
        )

        if in_phase_a or in_soft_phase_b:
            if dgs_active:
                # When tangent reparam is active, restrict DGS to curved (non-surface) Gaussians.
                # Tangent-constrained flat surfels already have hard geometric constraint —
                # applying soft DGS on top would be redundant and could fight the constraint.
                if _has_free_surfels:
                    free_mask_t = ~gaussians._surface_mask  # (N,) bool tensor
                    free_mask_np = free_mask_t.cpu().numpy()
                    dgs_xyz_geo = gaussians.get_xyz_geo[free_mask_t]  # (N_free, 3) with grad
                else:
                    free_mask_t = None
                    free_mask_np = None
                    dgs_xyz_geo = gaussians.get_xyz_geo  # all Gaussians (standard path)

                # Update KNN cache every dgs_interval iters or after densification
                # Cache is built for the DGS subset only (free surfels when hybrid is active)
                if (iteration % dgs_interval == 0 or lidar_field.cache_invalid):
                    if free_mask_np is not None:
                        xyz_np = dgs_xyz_geo.detach().cpu().numpy()
                    else:
                        xyz_np = gaussians.get_xyz_geo.detach().cpu().numpy()
                    if adaptive_dgs:
                        lidar_field.update_cache_adaptive(
                            xyz_np,
                            planarity_k=dgs_planarity_k,
                            k_flat=dgs_k,
                            k_ambig=dgs_k * 2,
                            max_radius=dgs_radius,
                            flat_threshold=dgs_flat_threshold,
                            skip_threshold=dgs_skip_threshold,
                        )
                    elif dgs_bilateral:
                        # v8o: bilateral-weighted PCA — needs surfel normals for normal kernel
                        surfel_normals_np_all = get_surfel_normals(gaussians).detach().cpu().numpy()
                        surfel_normals_np = surfel_normals_np_all[free_mask_np] if free_mask_np is not None else surfel_normals_np_all
                        lidar_field.update_cache_bilateral(
                            xyz_np, surfel_normals_np,
                            k=dgs_k, max_radius=dgs_radius, min_neighbors=4,
                            sigma_c=dgs_bilateral_sigma_c, sigma_x=dgs_bilateral_sigma_x,
                        )
                    elif dgs_mls_quadric:
                        # v8q/v8r/v8s: MLS height-field quadric fit
                        if iteration == dgs_start_iter:
                            if dgs_mls_uniform_weights:
                                mode_str = "uniform"
                            else:
                                mode_str = (f"bilateral (sigma_x={dgs_bilateral_sigma_x}, "
                                            f"sigma_c={dgs_bilateral_sigma_c}, "
                                            f"min_w_frac={dgs_mls_min_weight_frac})")
                            print(f"[DGS-MLS] Mode: {mode_str}")
                        surfel_normals_np_all = get_surfel_normals(gaussians).detach().cpu().numpy()
                        surfel_normals_np = surfel_normals_np_all[free_mask_np] if free_mask_np is not None else surfel_normals_np_all
                        lidar_field.update_cache_mls_quadric(
                            xyz_np, surfel_normals_np,
                            k=dgs_k, max_radius=dgs_radius, min_neighbors=4,
                            sigma_c=dgs_bilateral_sigma_c, sigma_x=dgs_bilateral_sigma_x,
                            uniform_weights=dgs_mls_uniform_weights,
                            min_weight_frac=dgs_mls_min_weight_frac,
                        )
                    else:
                        # Standard DGS — surfel normals optionally used for cross-surface filtering
                        surfel_normals_for_filter = None
                        if dgs_normal_filter:
                            normals_all_np = get_surfel_normals(gaussians).detach().cpu().numpy()
                            surfel_normals_for_filter = normals_all_np[free_mask_np] if free_mask_np is not None else normals_all_np
                        lidar_field.update_cache(xyz_np, k=dgs_k, max_radius=dgs_radius,
                                                 surfel_normals_np=surfel_normals_for_filter,
                                                 normal_cos_threshold=dgs_normal_cos_threshold)

                # Differentiable DGS loss — gradients flow directly to surfel_xyz / rotation
                surfel_normals_all = get_surfel_normals(gaussians)
                surfel_normals = surfel_normals_all[free_mask_t] if free_mask_t is not None else surfel_normals_all
                if adaptive_dgs:
                    dgs_pos_loss, dgs_norm_loss, _adap_stats = lidar_field.compute_adaptive_loss(
                        dgs_xyz_geo, surfel_normals,
                        lambda_dgs_eff=lambda_dgs_eff,
                        lambda_dgs_normal=lambda_dgs_normal,
                        distance_gate_sigma=dgs_distance_gate_sigma,
                        iteration=iteration,
                    )
                    if iteration % 1000 == 0:
                        print(f"[DGS-adaptive] iter={iteration} lambda_eff={lambda_dgs_eff:.4f} "
                              f"flat={_adap_stats.get('n_flat',0):,} "
                              f"ambig={_adap_stats.get('n_ambig',0):,} "
                              f"skip={_adap_stats.get('n_skip',0):,} "
                              f"mean_plan={_adap_stats.get('mean_planarity',0):.3f} "
                              f"dgs_pos={dgs_pos_loss.item():.5f} dgs_norm={dgs_norm_loss.item():.5f}")
                elif dgs_mls_quadric:
                    # v8q: MLS quadric loss — returns already-lambda-weighted losses
                    dgs_pos_loss, dgs_norm_loss, n_valid = lidar_field.compute_mls_loss(
                        dgs_xyz_geo, surfel_normals,
                        lambda_dgs_eff=lambda_dgs_eff,
                        lambda_dgs_normal=lambda_dgs_normal,
                        distance_gate_sigma=dgs_distance_gate_sigma,
                        iteration=iteration,
                    )
                    if iteration % 2000 == 0:
                        N_surf    = dgs_xyz_geo.shape[0]
                        skip_frac = getattr(lidar_field, '_quadric_n_skip', 0) / max(N_surf, 1) * 100
                        print(f"[DGS-MLS] iter={iteration} skip_frac={skip_frac:.1f}% "
                              f"L_pos={dgs_pos_loss.item():.5f}")
                else:
                    L_pos, L_norm, n_valid = lidar_field.compute_loss(
                        dgs_xyz_geo, surfel_normals,
                        planarity_threshold=dgs_planarity_threshold,
                        distance_gate_sigma=dgs_distance_gate_sigma,
                        iteration=iteration,
                    )
                    dgs_pos_loss = lambda_dgs_eff * L_pos
                    dgs_norm_loss = lambda_dgs_normal * L_norm
                    if iteration % 1000 == 0:
                        free_tag = f" [free={dgs_xyz_geo.shape[0]:,}]" if free_mask_t is not None else ""
                        print(f"[DGS{free_tag}] iter={iteration} lambda_eff={lambda_dgs_eff:.4f} "
                              f"n_valid={n_valid} L_pos={L_pos.item():.5f} L_norm={L_norm.item():.5f}")
                    # v8o bilateral sanity check every 2000 iters
                    if dgs_bilateral and iteration % 2000 == 0:
                        n_skip_bi = getattr(lidar_field, '_bilateral_n_skip', 0)
                        n_total_bi = max(getattr(lidar_field, '_n_surfels_at_cache', 1), 1)
                        skip_pct_bi = 100.0 * n_skip_bi / n_total_bi
                        print(f"[DGS-bilateral] iter={iteration} skip_fraction={skip_pct_bi:.2f}% "
                              f"({n_skip_bi:,}/{n_total_bi:,}) below sum_w threshold "
                              f"(sigma_c={dgs_bilateral_sigma_c}, sigma_x={dgs_bilateral_sigma_x})")

            # ---- Depth concentration loss ----
            # Penalises spread of surfels along each ray:
            # when concentrated at one depth, median ≈ expected (surf_depth).
            if lambda_concentrate > 0.0:
                surf_d   = render_pkg['surf_depth']           # (1, H, W)
                median_d = render_pkg['render_depth_median']  # (1, H, W)
                conc_valid = (surf_d[0] > 0) & (median_d[0] > 0)
                if viewpoint_cam.gt_alpha_mask is not None:
                    conc_valid = conc_valid & (viewpoint_cam.gt_alpha_mask.cuda().squeeze(0) > 0)
                if conc_valid.sum() > 100:
                    conc_loss = lambda_concentrate * l1_loss(
                        median_d[0][conc_valid], surf_d[0][conc_valid]
                    )

        # ---- v8l: Opacity entropy loss — push opacities toward 0 or 1 ----
        opacity_entropy_loss = torch.tensor(0.0, device="cuda")
        if lambda_opacity_entropy > 0.0:
            opacity = gaussians.get_opacity  # (N, 1), sigmoid-activated, in [0, 1]
            eps = 1e-6
            entropy = -(opacity * torch.log(opacity + eps) +
                        (1 - opacity) * torch.log(1 - opacity + eps))
            opacity_entropy_loss = lambda_opacity_entropy * entropy.mean()

        # ---- v8j: Offset regularization — keep _xyz_offset close to zero ----
        offset_reg_loss = torch.tensor(0.0, device="cuda")
        if dual_position and lambda_offset > 0.0 and gaussians._offset_active:
            offset_reg_loss = lambda_offset * (gaussians._xyz_offset ** 2).mean()
            if iteration % 1000 == 0:
                off_mag = gaussians._xyz_offset.detach().norm(dim=1)
                print(f"[Offset] iter={iteration} mean={off_mag.mean().item()*1000:.3f}mm "
                      f"max={off_mag.max().item()*1000:.3f}mm "
                      f"std={off_mag.std().item()*1000:.3f}mm")

        # loss — v12: when alternate_dgs active, interleave depth and DGS across iterations
        # to prevent per-iteration gradient interference between the two geometric forces.
        _dgs_active_this_iter = (dgs_pos_loss.item() != 0.0 or dgs_norm_loss.item() != 0.0)
        if alternate_dgs and _dgs_active_this_iter:
            if iteration % 2 == 0:
                # Depth iteration: depth + normal supervision; skip DGS
                _eff_dgs_pos = torch.tensor(0.0, device="cuda")
                _eff_dgs_norm = torch.tensor(0.0, device="cuda")
                _eff_depth = lidar_depth_loss
                _eff_normal = lidar_normal_loss
            else:
                # DGS iteration: DGS losses only; skip depth + normal
                _eff_dgs_pos = dgs_pos_loss
                _eff_dgs_norm = dgs_norm_loss
                _eff_depth = torch.tensor(0.0, device="cuda")
                _eff_normal = torch.tensor(0.0, device="cuda")
            geo_loss = (dist_loss + normal_loss + _eff_depth + _eff_normal
                        + _eff_dgs_pos + _eff_dgs_norm + conc_loss
                        + opacity_entropy_loss + offset_reg_loss)
        else:
            geo_loss = (dist_loss + normal_loss + lidar_depth_loss + lidar_normal_loss
                        + dgs_pos_loss + dgs_norm_loss + conc_loss
                        + opacity_entropy_loss + offset_reg_loss)

        _backward_ok = True
        total_loss = loss + geo_loss
        try:
            total_loss.backward()
        except RuntimeError as _bwd_err:
            if "invalid gradient" in str(_bwd_err) or "_RasterizeGaussians" in str(_bwd_err):
                _backward_ok = False
            else:
                raise

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_depth_for_log = 0.4 * lidar_depth_loss.item() + 0.6 * ema_depth_for_log
            ema_dgs_for_log = 0.4 * (dgs_pos_loss.item() + dgs_norm_loss.item()) + 0.6 * ema_dgs_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "depth": f"{ema_depth_for_log:.{5}f}",
                    "dgs": f"{ema_dgs_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Quadric reparam diagnostics every 1000 iters
            if use_quadric_reparam and gaussians._quadric_active and iteration % 1000 == 0:
                with torch.no_grad():
                    surf_xyz = gaussians.get_xyz[gaussians._surface_mask]
                    dist_to_anc = (surf_xyz - gaussians._anchor_points).norm(dim=-1)
                    delta_abs = gaussians._normal_delta.abs().squeeze(1)
                    uv_norms  = gaussians._tangent_uv.norm(dim=-1)
                    at_clamp  = int((delta_abs >= gaussians._quadric_epsilon).sum().item())
                    eps_median = float(gaussians._quadric_epsilon.median()) * 1000
                    abc_max = float(gaussians._quad_abc.abs().max())
                    print(f"[Quadric v10b iter {iteration}] "
                          f"anchor_dist: {dist_to_anc.median()*1000:.2f}mm, "
                          f"|δ|: {delta_abs.median()*1000:.2f}mm (ε_med={eps_median:.2f}mm), "
                          f"|uv|: {uv_norms.median()*1000:.2f}mm, "
                          f"at_clamp: {at_clamp}/{len(delta_abs)}, "
                          f"|abc|_max: {abc_max:.4f}")

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/lidar_depth_loss', ema_depth_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                if _backward_ok:  # grad is valid only when backward succeeded
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                _did_densify = False
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    # Fix 2a: LiDAR-gated densification — compute surface proximity mask
                    _surface_mask = None
                    _lidar_field_for_densify = None
                    if lidar_gated_densify and lidar_field is not None:
                        import numpy as _np
                        _xyz_np = gaussians.get_xyz_geo.detach().cpu().numpy()
                        _near_lidar = lidar_field.get_pruning_mask(_xyz_np, max_distance=densify_surface_radius)
                        _surface_mask = torch.from_numpy(_near_lidar).cuda()
                        _lidar_field_for_densify = lidar_field
                        n_near = int(_near_lidar.sum())
                        if iteration % 1000 == 0:
                            print(f"[LiDAR-gate iter {iteration}] {n_near}/{len(_xyz_np)} Gaussians "
                                  f"within {densify_surface_radius}m of LiDAR surface "
                                  f"({100*n_near/max(len(_xyz_np),1):.1f}% eligible for densification)",
                                  flush=True)

                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull,
                                                scene.cameras_extent, size_threshold,
                                                lidar_field=_lidar_field_for_densify,
                                                surface_mask=_surface_mask)
                    _did_densify = True

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

                    # Fix 5: Surface-biased opacity reset — boost Gaussians near LiDAR surfaces
                    # so they survive the next few iters while floaters stay faint.
                    if lidar_gated_densify and lidar_field is not None:
                        import numpy as _np2
                        _xyz_np2 = gaussians.get_xyz_geo.detach().cpu().numpy().astype(_np2.float32)
                        _dists2, _ = lidar_field.kdtree.query(_xyz_np2, k=1, workers=-1)
                        # Gaussian proximity kernel: σ=50mm → weight≈1 within 5cm of LiDAR
                        _prox = _np2.exp(-_dists2 ** 2 / (2 * 0.05 ** 2)).astype(_np2.float32)
                        _boost = torch.from_numpy(0.04 * _prox).float().cuda()
                        _cur_op = torch.sigmoid(gaussians._opacity.squeeze())
                        _boosted = (_cur_op + _boost).clamp(0.001, 0.999)
                        gaussians._opacity.data = torch.logit(_boosted).unsqueeze(-1)
                        n_boosted = int((_prox > 0.5).sum())
                        print(f"[OpacityBoost iter {iteration}] Boosted {n_boosted} surfels "
                              f"near LiDAR (prox>0.5)", flush=True)

                    _did_densify = True

                # Invalidate DGS plane cache: surfel count changed after split/prune
                if _did_densify and lidar_field is not None:
                    lidar_field.invalidate_cache()

            # Fix 4: Geometric pruning — remove floaters far from LiDAR AND low opacity AND small scale.
            # The AND logic preserves high-opacity ray-interior Gaussians (needed for rendering/TSDF)
            # while culling small faint floaters that are purely photometric noise.
            if lidar_gated_densify and lidar_field is not None and iteration % 1000 == 0:
                import numpy as _np3
                _xyz_np3 = gaussians.get_xyz_geo.detach().cpu().numpy().astype(_np3.float32)
                _dists3, _ = lidar_field.kdtree.query(_xyz_np3, k=1, workers=-1)
                _far = torch.from_numpy(_dists3 > geo_prune_distance).cuda()
                _low_op = (gaussians.get_opacity.squeeze() < 0.3)
                _small_sc = (gaussians.get_scaling.max(dim=1).values < 0.01)
                _geo_prune = _far & _low_op & _small_sc
                n_geo_prune = int(_geo_prune.sum().item())
                if n_geo_prune > 0:
                    gaussians.prune_points(_geo_prune)
                    if lidar_field is not None:
                        lidar_field.invalidate_cache()
                print(f"[Geo-prune iter {iteration}] Pruned {n_geo_prune} floaters "
                      f"(dist>{geo_prune_distance}m & opacity<0.3 & scale<1cm)", flush=True)

            # ---- Z-axis pruning: remove Gaussians above z_max threshold every 1000 iters ----
            # Use for scenes where cameras are outside/above the room (e.g. ScanNet++).
            # Unlike LiDAR-proximity pruning, Z-prune keeps ray-interior Gaussians that lie
            # between camera and room surfaces but are far from any LiDAR surface point.
            if prune_z_max is not None and iteration % 1000 == 0:
                xyz = gaussians.get_xyz_geo.detach()
                prune_mask = xyz[:, 2] > prune_z_max
                n_prune = int(prune_mask.sum().item())
                n_keep = int((~prune_mask).sum().item())
                if n_prune > 0:
                    gaussians.prune_points(prune_mask)
                    if lidar_field is not None:
                        lidar_field.invalidate_cache()
                print(f"[Z-prune iter {iteration}] Pruned {n_prune} (z>{prune_z_max}), kept {n_keep}", flush=True)

            # ---- LiDAR-proximity pruning: remove floater Gaussians every 1000 iters ----
            # Runs both during and after densification — floaters get created by densification
            # and must be removed promptly to prevent them from compounding.
            if lidar_prune_distance > 0.0 and lidar_field is not None and iteration % 1000 == 0:
                xyz_np = gaussians.get_xyz_geo.detach().cpu().numpy()
                keep_mask = lidar_field.get_pruning_mask(xyz_np, lidar_prune_distance)
                prune_mask = torch.from_numpy(~keep_mask).cuda()
                n_prune = int(prune_mask.sum().item())
                n_keep = int(keep_mask.sum())
                if n_prune > 0:
                    gaussians.prune_points(prune_mask)
                    lidar_field.invalidate_cache()
                print(f"[LiDAR prune iter {iteration}] Pruned {n_prune}, kept {n_keep} "
                      f"(dist<{lidar_prune_distance}m)", flush=True)

            # Optimizer step
            if iteration < opt.iterations:
                if _backward_ok:
                    gaussians.optimizer.step()
                    if gaussians._offset_optimizer is not None:
                        gaussians._offset_optimizer.step()
                    if tangent_reparam_optimizer is not None:
                        tangent_reparam_optimizer.step()
                    if quadric_reparam_optimizer is not None:
                        quadric_reparam_optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if gaussians._offset_optimizer is not None:
                    gaussians._offset_optimizer.zero_grad(set_to_none=True)
                if tangent_reparam_optimizer is not None:
                    tangent_reparam_optimizer.zero_grad(set_to_none=True)
                if quadric_reparam_optimizer is not None:
                    quadric_reparam_optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Phase 3: LiDAR depth supervision
    parser.add_argument("--lambda_depth", type=float, default=0.0,
                        help="Weight for LiDAR depth supervision loss (0=disabled). Recommended: 1.0")
    parser.add_argument("--lambda_lidar_normal", type=float, default=0.0,
                        help="Weight for LiDAR normal supervision loss (0=disabled). Recommended: 0.1")
    parser.add_argument("--phase_b_start", type=int, default=20000,
                        help="Iteration to begin Geometry-Tethered Refinement (GTR): freeze/tether geometry "
                             "and refine appearance. 0=disable. Default: 20000. Use --gtr_start as alias.")
    parser.add_argument("--use_anchor_weights", action="store_true", default=False,
                        help="Use adaptive per-pixel anchor weights for depth loss. "
                             "Requires anchor_weights.npy in model_path (run compute_anchor_weights.py first).")
    # Direct Geometric Supervision (DGS)
    parser.add_argument("--lambda_dgs", type=float, default=0.0,
                        help="Weight for DGS position loss (signed distance surfel→LiDAR plane). "
                             "Recommended: 1.0. 0=disabled.")
    parser.add_argument("--lambda_dgs_normal", type=float, default=0.5,
                        help="Weight for DGS normal alignment loss. Recommended: 0.5.")
    parser.add_argument("--lambda_concentrate", type=float, default=0.0,
                        help="Weight for depth concentration loss |median_depth - surf_depth|. "
                             "Recommended: 0.5. 0=disabled.")
    parser.add_argument("--dgs_interval", type=int, default=500,
                        help="Iterations between DGS KNN cache updates. Default: 500.")
    parser.add_argument("--dgs_start_iter", type=int, default=10000,
                        help="Iteration to start DGS within Phase A. Default: 10000 (matches densify_until_iter).")
    parser.add_argument("--dgs_k", type=int, default=8,
                        help="Number of nearest LiDAR neighbours for DGS plane fitting. Default: 8.")
    parser.add_argument("--dgs_radius", type=float, default=0.05,
                        help="Max radius (m) for LiDAR neighbour search in DGS. Default: 0.05 (50mm).")
    parser.add_argument("--las_path", type=str, default=None,
                        help="Path to .las LiDAR file (required when --lambda_dgs > 0).")
    parser.add_argument("--dgs_warmup_start", type=int, default=-1,
                        help="Iteration to start DGS warmup phase (-1 = disabled). "
                             "Between dgs_warmup_start and dgs_start_iter, uses dgs_warmup_lambda. "
                             "Example: --dgs_warmup_start 5000 --dgs_warmup_lambda 0.02 "
                             "--dgs_start_iter 10000 starts warmup at 5k, full DGS at 10k.")
    parser.add_argument("--dgs_warmup_lambda", type=float, default=0.02,
                        help="DGS lambda during warmup phase (dgs_warmup_start to dgs_start_iter). "
                             "Default: 0.02. Only used when dgs_warmup_start >= 0.")
    parser.add_argument("--dgs_planarity_threshold", type=float, default=0.3,
                        help="λ₁/λ₂ planarity ratio cutoff for DGS. Surfels at edges/corners "
                             "(ratio ≥ threshold) are excluded from DGS. Default: 0.3. "
                             "Tighter=0.15, looser=0.5.")
    parser.add_argument("--dgs_distance_gate_sigma", type=float, default=0.02,
                        help="Soft-gate length scale (m) for DGS: weight=exp(-|dist|/σ). "
                             "Surfels >2σ from their fitted plane are down-weighted. "
                             "Default: 0.02 (20mm).")
    # Geometry-Tethered Refinement (GTR)
    parser.add_argument("--soft_phase_b", action="store_true", default=False,
                        help="Enable Geometry-Tethered Refinement (GTR): reduce position LR to "
                             "phase_b_xyz_lr instead of hard freeze. rotation/scaling/opacity still frozen. "
                             "DGS continues as a spring tether at phase_b_dgs_lambda. Use --gtr as alias.")
    parser.add_argument("--phase_b_xyz_lr", type=float, default=1e-6,
                        help="Position learning rate during Geometry-Tethered Refinement. Default: 1e-6. "
                             "Use --gtr_xyz_lr as alias.")
    parser.add_argument("--phase_b_dgs_lambda", type=float, default=0.05,
                        help="DGS tether strength during Geometry-Tethered Refinement. Default: 0.05. "
                             "Use --gtr_dgs_lambda as alias.")
    # GTR aliases (backward-compatible shorthand)
    parser.add_argument("--gtr", dest="soft_phase_b", action="store_true",
                        help="Alias for --soft_phase_b. Enable Geometry-Tethered Refinement (GTR).")
    parser.add_argument("--gtr_start", dest="phase_b_start", type=int, default=20000,
                        help="Alias for --phase_b_start. GTR start iteration. Default: 20000.")
    parser.add_argument("--gtr_xyz_lr", dest="phase_b_xyz_lr", type=float, default=1e-6,
                        help="Alias for --phase_b_xyz_lr. Position LR during GTR. Default: 1e-6.")
    parser.add_argument("--gtr_dgs_lambda", dest="phase_b_dgs_lambda", type=float, default=0.05,
                        help="Alias for --phase_b_dgs_lambda. DGS tether strength during GTR. Default: 0.05.")
    # v8j: Dual positions
    parser.add_argument("--dual_position", action="store_true", default=False,
                        help="Dual positions: geo _xyz for DGS/mesh, geo+_xyz_offset for rendering. "
                             "_xyz_offset activates during GTR.")
    parser.add_argument("--lambda_offset", type=float, default=100.0,
                        help="L2 regularization weight for _xyz_offset. Higher = smaller offsets. Default: 100.")
    parser.add_argument("--offset_lr", type=float, default=5e-5,
                        help="Learning rate for _xyz_offset during GTR. Default: 5e-5.")
    # v8l: Opacity entropy
    parser.add_argument("--lambda_opacity_entropy", type=float, default=0.0,
                        help="Weight for opacity entropy loss (pushes opacities toward 0 or 1). "
                             "Recommended: 0.01. 0=disabled.")
    # v8m: Adaptive DGS
    parser.add_argument("--adaptive_dgs", action="store_true", default=False,
                        help="Adaptive DGS: classify surfels by local geometry planarity, then apply "
                             "different k and per-surfel planarity weights instead of uniform λ_dgs. "
                             "Flat surfels (plan>flat_thr) use k=dgs_k; ambiguous use k=dgs_k*2; "
                             "volumetric (plan≤skip_thr) skip DGS entirely.")
    parser.add_argument("--dgs_planarity_k", type=int, default=64,
                        help="Neighbours for planarity classification in adaptive DGS. Default: 64.")
    parser.add_argument("--dgs_flat_threshold", type=float, default=0.85,
                        help="Planarity threshold above which surfel is treated as flat (full DGS). "
                             "planarity=1-λ_min/λ_max. Default: 0.85 (~58%% of surfels in ARTLab).")
    parser.add_argument("--dgs_skip_threshold", type=float, default=0.5,
                        help="Planarity threshold below which surfel skips DGS (volumetric/isotropic). "
                             "Default: 0.5 (~9%% of surfels in ARTLab).")
    # v8n: Normal-consistent KNN filtering
    parser.add_argument("--dgs_normal_filter", action="store_true", default=False,
                        help="Filter DGS KNN neighbours by surfel-LiDAR normal consistency before PCA. "
                             "Rejects neighbours whose normal disagrees with the surfel normal by "
                             ">arccos(dgs_normal_cos_threshold). Fixes cross-surface plane fits at "
                             "wall ledges and object boundaries. Zero extra compute (dot product only).")
    parser.add_argument("--dgs_normal_cos_threshold", type=float, default=0.7,
                        help="Cosine similarity threshold for DGS normal filtering. "
                             "0.7 ≈ 45°, 0.866 ≈ 30°, 0.5 ≈ 60°. Lower = more permissive. Default: 0.7")
    # v8o: Bilateral-weighted PCA plane fitting
    parser.add_argument("--dgs_bilateral", action="store_true", default=False,
                        help="Use bilateral-weighted PCA for DGS plane fitting. "
                             "Each LiDAR neighbour is weighted by a normal-alignment kernel "
                             "exp(-(1-|n_surfel·n_j|)/sigma_c^2) times a spatial kernel "
                             "exp(-||x_j-mu_i||^2/sigma_x^2). Surfels where sum(w_j)<0.5*k are skipped. "
                             "Incompatible with --adaptive_dgs and --dgs_normal_filter.")
    parser.add_argument("--dgs_bilateral_sigma_c", type=float, default=0.5,
                        help="Normal-kernel bandwidth for bilateral DGS. sigma_c=0.5 → "
                             "exp(-2)≈0.135 at 60° misalignment, near-zero at 90°. Default: 0.5.")
    parser.add_argument("--dgs_bilateral_sigma_x", type=float, default=0.03,
                        help="Spatial-kernel bandwidth (m) for bilateral DGS. "
                             "sigma_x=0.03 → half-weight at ~30mm from surfel centre. Default: 0.03.")
    # v8q: MLS quadric DGS
    parser.add_argument("--dgs_mls_quadric", action="store_true", default=False,
                        help="Use MLS height-field quadric fit for DGS. Fits z=a*u^2+b*u*v+c*v^2+d*u+e*v+g "
                             "in the bilateral-weighted local PCA frame, recovering local surface curvature. "
                             "Uses same sigma_c/sigma_x as --dgs_bilateral. "
                             "Incompatible with --adaptive_dgs, --dgs_bilateral, --dgs_normal_filter.")
    # v8r: MLS quadric without bilateral weighting
    parser.add_argument("--dgs_mls_uniform_weights", action="store_true", default=False,
                        help="v8r mode: skip bilateral weighting, use uniform weights w_j=1.0 for "
                             "MLS quadric fit. Isolates the quadric contribution from bilateral "
                             "weighting confound. Only active when --dgs_mls_quadric is set.")
    # v8s: MLS quadric with softened bilateral threshold
    parser.add_argument("--dgs_mls_min_weight_frac", type=float, default=0.5,
                        help="Minimum sum_w / k threshold to keep surfel valid in MLS quadric. "
                             "Default 0.5 (v8q). Lower (e.g. 0.25) allows surfels with softer "
                             "bilateral weights to contribute (v8s). "
                             "Ignored if --dgs_mls_uniform_weights is set.")
    # LiDAR-proximity pruning
    parser.add_argument("--lidar_prune_distance", type=float, default=0.0,
                        help="Max distance (m) from any LiDAR point for a Gaussian to survive pruning. "
                             "Gaussians farther than this are removed every 1000 iters. "
                             "0=disabled (default). WARNING: do NOT use on ScanNet++ or other "
                             "outside-room camera configs — kills ray-interior Gaussians needed for TSDF.")
    # Z-axis pruning for outside-room cameras
    parser.add_argument("--prune_z_max", type=float, default=None,
                        help="Prune Gaussians with z > this threshold every 1000 iters. "
                             "Intended for scenes where cameras are above/outside the room "
                             "(e.g. ScanNet++ cameras at z~+1.5, room at z=[-4,0]). "
                             "Set to slightly above max camera z (e.g. 2.0). "
                             "None=disabled (default). Safe for ray-interior Gaussians unlike "
                             "--lidar_prune_distance.")
    parser.add_argument("--phase_b_unfreeze_scale_opacity", action="store_true", default=False,
                        help="When set with --soft_phase_b/--gtr, keep _scaling and _opacity trainable in "
                             "GTR. Allows Gaussians dragged to room surfaces by depth+DGS to "
                             "re-optimize their scale and opacity for photometric quality. "
                             "rotation still frozen. Default: False (standard behavior).")
    # LiDAR-gated densification
    parser.add_argument("--lidar_gated_densify", action="store_true", default=False,
                        help="Gate densification by LiDAR surface proximity: only clone/split "
                             "Gaussians within --densify_surface_radius of a LiDAR surface point. "
                             "Also enables geometric pruning and surface-biased opacity reset. "
                             "Requires --las_path. Default: False.")
    parser.add_argument("--densify_surface_radius", type=float, default=0.3,
                        help="Max distance (m) from LiDAR for a Gaussian to be eligible for "
                             "densification when --lidar_gated_densify is set. Default: 0.3m.")
    parser.add_argument("--geo_prune_distance", type=float, default=0.5,
                        help="Distance threshold (m) for geometric pruning. Gaussians BEYOND this "
                             "distance AND with opacity<0.3 AND scale<1cm are pruned every 1000 iters. "
                             "Only active when --lidar_gated_densify is set. Default: 0.5m.")
    # Tangent-plane reparameterization
    parser.add_argument("--use_tangent_reparam", action="store_true", default=False,
                        help="Enable tangent-plane reparameterization: μ_i = p_i + u*t_u + v*t_v + "
                             "clamp(δ, -ε, +ε)*n. Constrains surfels to LiDAR surface manifold "
                             "with bounded normal displacement (ε=epsilon_base). Replaces DGS "
                             "soft penalty with a hard geometric constraint. Requires --las_path.")
    parser.add_argument("--tangent_reparam_iter", type=int, default=10000,
                        help="Iteration to enable tangent-plane reparameterization. "
                             "Must be >= densify_until_iter (default 10000).")
    parser.add_argument("--epsilon_base", type=float, default=0.002,
                        help="Base normal displacement bound (m). Actual ε_i = "
                             "epsilon_base * clamp(planarity_i, 0.01, 1.0). Default: 0.002 (2mm).")
    parser.add_argument("--tangent_gap_threshold", type=float, default=0.10,
                        help="Max distance (m) from LiDAR anchor for a surfel to be constrained. "
                             "Surfels beyond this remain as unconstrained free Gaussians. "
                             "Default: 0.10 (10cm).")
    parser.add_argument("--tangent_planarity_threshold", type=float, default=0.10,
                        help="PCA planarity ratio (lambda3/lambda2) threshold for tangent reparam. "
                             "Surfels with planarity < threshold get hard tangent constraint (flat surfaces). "
                             "Surfels with planarity >= threshold stay as free _xyz with soft DGS (curved geometry). "
                             "Default: 0.10.")
    # v10: Quadric tangent-plane reparameterization
    parser.add_argument("--quadric_reparam", action="store_true", default=False,
                        help="Enable quadric tangent-plane reparameterization (v10b) at densify_until_iter: "
                             "μ_i = anchor_i + u*tu + v*tv + [Q(u,v) + clamp(δ,-ε,+ε)]*n. "
                             "Q(u,v) = a*u² + b*u*v + c*v² captures local surface curvature from LiDAR. "
                             "Epsilon uses planarity ratio (same as tangent reparam). "
                             "Quadric fit is ridge-regularized toward zero to suppress noise. "
                             "Requires --las_path.")
    parser.add_argument("--quadric_epsilon_base", type=float, default=0.002,
                        help="Base epsilon for quadric normal clamp (metres). "
                             "ε_i = ε_base × clamp(ρ_i, 0.01, 1.0) where ρ=planarity ratio. "
                             "Flat walls: ε≈0.02mm, edges: ε≈0.6mm, curved: ε≈2mm. Default: 0.002.")
    parser.add_argument("--quadric_ridge_lambda", type=float, default=0.01,
                        help="Ridge regularization for quadric fit: (ATA + λI)θ = ATh. "
                             "Suppresses noise-driven curvature on flat surfaces. "
                             "Higher = stronger bias toward a=b=c=0 (tangent plane). Default: 0.01.")
    parser.add_argument("--quadric_gap_threshold", type=float, default=0.10,
                        help="Max distance (m) from nearest LiDAR point for quadric constraint. "
                             "Surfels beyond this remain as free _xyz. Default: 0.10 (10cm).")
    # v12: Alternating DGS
    parser.add_argument("--alternate_dgs", action="store_true", default=False,
                        help="Alternate depth/normal supervision and DGS supervision across iterations "
                             "(even iters: depth+normal only; odd iters: DGS only). "
                             "Tests whether per-iteration gradient interference between depth and DGS "
                             "causes the non-monotonic lambda sweep behavior. "
                             "Photometric + 2DGS regularization losses are included on ALL iterations.")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint,
             lambda_depth=args.lambda_depth, lambda_lidar_normal=args.lambda_lidar_normal, phase_b_start=args.phase_b_start,
             use_anchor_weights=args.use_anchor_weights,
             lambda_dgs=args.lambda_dgs, lambda_dgs_normal=args.lambda_dgs_normal,
             lambda_concentrate=args.lambda_concentrate, dgs_interval=args.dgs_interval,
             dgs_k=args.dgs_k, dgs_radius=args.dgs_radius, dgs_start_iter=args.dgs_start_iter,
             las_path=args.las_path,
             dgs_planarity_threshold=args.dgs_planarity_threshold,
             dgs_distance_gate_sigma=args.dgs_distance_gate_sigma,
             dgs_warmup_start=args.dgs_warmup_start,
             dgs_warmup_lambda=args.dgs_warmup_lambda,
             soft_phase_b=args.soft_phase_b,
             phase_b_xyz_lr=args.phase_b_xyz_lr,
             phase_b_dgs_lambda=args.phase_b_dgs_lambda,
             dual_position=args.dual_position,
             lambda_offset=args.lambda_offset,
             offset_lr=args.offset_lr,
             lambda_opacity_entropy=args.lambda_opacity_entropy,
             adaptive_dgs=args.adaptive_dgs,
             dgs_planarity_k=args.dgs_planarity_k,
             dgs_flat_threshold=args.dgs_flat_threshold,
             dgs_skip_threshold=args.dgs_skip_threshold,
             dgs_normal_filter=args.dgs_normal_filter,
             dgs_normal_cos_threshold=args.dgs_normal_cos_threshold,
             dgs_bilateral=args.dgs_bilateral,
             dgs_bilateral_sigma_c=args.dgs_bilateral_sigma_c,
             dgs_bilateral_sigma_x=args.dgs_bilateral_sigma_x,
             dgs_mls_quadric=args.dgs_mls_quadric,
             dgs_mls_uniform_weights=args.dgs_mls_uniform_weights,
             dgs_mls_min_weight_frac=args.dgs_mls_min_weight_frac,
             lidar_prune_distance=args.lidar_prune_distance,
             prune_z_max=args.prune_z_max,
             phase_b_unfreeze_scale_opacity=args.phase_b_unfreeze_scale_opacity,
             lidar_gated_densify=args.lidar_gated_densify,
             densify_surface_radius=args.densify_surface_radius,
             geo_prune_distance=args.geo_prune_distance,
             use_tangent_reparam=args.use_tangent_reparam,
             tangent_reparam_iter=args.tangent_reparam_iter,
             epsilon_base=args.epsilon_base,
             tangent_gap_threshold=args.tangent_gap_threshold,
             tangent_planarity_threshold=args.tangent_planarity_threshold,
             use_quadric_reparam=args.quadric_reparam,
             quadric_epsilon_base=args.quadric_epsilon_base,
             quadric_ridge_lambda=args.quadric_ridge_lambda,
             quadric_gap_threshold=args.quadric_gap_threshold,
             alternate_dgs=args.alternate_dgs)

    # All done
    print("\nTraining complete.")
