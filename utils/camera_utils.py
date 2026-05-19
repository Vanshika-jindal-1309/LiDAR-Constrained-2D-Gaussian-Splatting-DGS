#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
import os
import torch
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False


def _load_depth_normal(source_path, image_path, target_hw):
    """
    Derive and load depth/normal .npy maps for a given image_path.

    Depth map: {source_path}/depth_maps/{camera_subdir}/{stem}.npy  → (H,W) float32
    Normal map: {source_path}/normal_maps/{camera_subdir}/{stem}.npy → (H,W,3) float32

    Returns:
        gt_depth   torch.Tensor (H, W)  or None
        gt_normal  torch.Tensor (3, H, W) or None
    """
    try:
        # image_path: .../source_path/images_masked/camera_0/img.png
        rel = os.path.relpath(image_path, source_path)   # images_masked/camera_0/img.png
        parts = rel.replace('\\', '/').split('/')
        # parts[0] = images folder (e.g. 'images_masked'), parts[1:] = subdir + filename
        if len(parts) < 2:
            return None, None
        rest = parts[1:]   # ['camera_0', 'img.png']
        stem = os.path.splitext(rest[-1])[0]   # 'img'
        subpath = os.path.join(*rest[:-1], stem) if len(rest) > 1 else stem

        depth_path  = os.path.join(source_path, 'depth_maps',  subpath + '.npy')
        normal_path = os.path.join(source_path, 'normal_maps', subpath + '.npy')
    except Exception:
        return None, None

    gt_depth = gt_normal = None

    if os.path.exists(depth_path):
        d = np.load(depth_path)   # (H, W) float32
        if d.shape != target_hw:
            # Bilinear resize depth
            import torch.nn.functional as F
            d_t = torch.from_numpy(d).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            d_t = F.interpolate(d_t, size=target_hw, mode='nearest')
            d = d_t.squeeze(0).squeeze(0).numpy()
        gt_depth = torch.from_numpy(d.astype(np.float32))   # (H, W)

    if os.path.exists(normal_path):
        n = np.load(normal_path)   # (H, W, 3) float32
        if n.shape[:2] != target_hw:
            import torch.nn.functional as F
            n_t = torch.from_numpy(n).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
            n_t = F.interpolate(n_t, size=target_hw, mode='nearest')
            n = n_t.squeeze(0).permute(1, 2, 0).numpy()
        gt_normal = torch.from_numpy(n.astype(np.float32)).permute(2, 0, 1)  # (3,H,W)

    return gt_depth, gt_normal


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if len(cam_info.image.split()) > 3:
        import torch
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    # Load LiDAR depth / normal maps if available
    target_hw = (gt_image.shape[1], gt_image.shape[2])
    gt_depth, gt_normal = _load_depth_normal(args.source_path, cam_info.image_path, target_hw)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  gt_depth=gt_depth, gt_normal=gt_normal)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry