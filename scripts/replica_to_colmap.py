"""
Convert Replica NICE-SLAM format to COLMAP format for 2DGS training.

Input (per scene):
  results/frame000000.jpg, frame000001.jpg, ...   (RGB images)
  results/frame000000.depth.png, ...               (16-bit depth)
  traj.txt                                         (N lines, each: 4x4 pose matrix flattened)
  mesh.ply                                         (GT mesh)

Output:
  images/                (symlinks to RGB frames)
  sparse/0/cameras.txt   (single PINHOLE camera)
  sparse/0/images.txt    (camera poses in COLMAP format)
  sparse/0/points3D.txt  (sampled from GT mesh)
  sparse/0/points3D.ply  (PLY version for 2DGS init)
  gt_mesh.ply            (symlink to mesh.ply)
  depth_maps/            (converted from 16-bit PNG to .npy float32 in metres)
  normal_maps/           (estimated from depth maps)
  lidar_pointcloud.ply   (dense point cloud sampled from GT mesh, for DGS)
"""
import argparse
import numpy as np
import os
import glob
from PIL import Image
import shutil


def read_traj(traj_path):
    """Read Replica trajectory file. Each line is a flattened 4x4 camera-to-world matrix."""
    poses = []
    with open(traj_path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                pose = np.array(vals).reshape(4, 4)
                poses.append(pose)
    return poses


def pose_to_colmap(c2w):
    """Convert camera-to-world (OpenGL/Replica convention) to COLMAP world-to-camera.
    Replica uses OpenGL: Y up, Z back. COLMAP uses: Y down, Z forward.
    """
    # Flip Y and Z axes to go from OpenGL to OpenCV/COLMAP convention
    flip = np.diag([1, -1, -1, 1]).astype(np.float64)
    c2w_colmap = c2w @ flip

    # COLMAP stores world-to-camera
    w2c = np.linalg.inv(c2w_colmap)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    return R, t


def rotation_to_quaternion(R):
    """Convert 3x3 rotation matrix to COLMAP quaternion (w, x, y, z)."""
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(R)
    q = r.as_quat()  # scipy returns [x, y, z, w]
    return [q[3], q[0], q[1], q[2]]  # COLMAP wants [w, x, y, z]


def depth_png_to_metres(depth_path, depth_scale):
    """Convert Replica 16-bit depth PNG to float32 metres.
    depth_scale: divide raw pixel value by this to get metres.
    Common values:
      6553.5  = NICE-SLAM Replica (uint16 max maps to ~10m)
      1000.0  = depth in millimetres stored as uint16
      5000.0  = some MonoSDF versions
    """
    d = np.array(Image.open(depth_path)).astype(np.float32)
    d_m = d / depth_scale
    d_m[d == 0] = 0  # Invalid depth stays 0
    return d_m


def estimate_normals_from_depth(depth, fx, fy, cx, cy):
    """Estimate per-pixel normals from depth map using central differences."""
    H, W = depth.shape
    normals = np.zeros((H, W, 3), dtype=np.float32)

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    X = (uu - cx) * depth / fx
    Y = (vv - cy) * depth / fy
    Z = depth

    dXdu = np.gradient(X, axis=1)
    dXdv = np.gradient(X, axis=0)
    dYdu = np.gradient(Y, axis=1)
    dYdv = np.gradient(Y, axis=0)
    dZdu = np.gradient(Z, axis=1)
    dZdv = np.gradient(Z, axis=0)

    normals[:, :, 0] = dYdu * dZdv - dZdu * dYdv
    normals[:, :, 1] = dZdu * dXdv - dXdu * dZdv
    normals[:, :, 2] = dXdu * dYdv - dYdu * dXdv

    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    norms[norms < 1e-8] = 1
    normals = normals / norms
    normals[depth == 0] = 0

    return normals


def main():
    parser = argparse.ArgumentParser(description="Convert Replica NICE-SLAM scene to COLMAP format")
    parser.add_argument("--scene_dir", required=True, help="Path to Replica scene (e.g. replica_raw/office0)")
    parser.add_argument("--output_dir", required=True, help="Output directory in COLMAP format")
    parser.add_argument("--stride", type=int, default=10,
                        help="Frame stride (use every Nth frame). Default: 10 → ~200 frames")
    parser.add_argument("--width", type=int, default=1200, help="Image width (default: 1200)")
    parser.add_argument("--height", type=int, default=680, help="Image height (default: 680)")
    parser.add_argument("--fx", type=float, default=600.0, help="Focal length x (default: 600)")
    parser.add_argument("--fy", type=float, default=600.0, help="Focal length y (default: 600)")
    parser.add_argument("--depth_scale", type=float, default=6553.5,
                        help="Depth PNG value / depth_scale = metres. NICE-SLAM Replica: 6553.5")
    parser.add_argument("--n_init_points", type=int, default=100000,
                        help="Points sampled from GT mesh for COLMAP points3D (default: 100k)")
    parser.add_argument("--n_lidar_points", type=int, default=1000000,
                        help="Points sampled from GT mesh for DGS lidar_pointcloud.ply (default: 1M)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read trajectory
    traj_path = os.path.join(args.scene_dir, "traj.txt")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"traj.txt not found at {traj_path}")
    poses = read_traj(traj_path)
    print(f"Read {len(poses)} poses from {traj_path}")

    # Find RGB frames
    rgb_frames = sorted(glob.glob(os.path.join(args.scene_dir, "results", "frame*.jpg")))
    if not rgb_frames:
        rgb_frames = sorted(glob.glob(os.path.join(args.scene_dir, "results", "frame*.png")))
    print(f"Found {len(rgb_frames)} RGB frames")

    if not rgb_frames:
        raise FileNotFoundError(f"No RGB frames found in {args.scene_dir}/results/")

    # Subsample frames
    frame_indices = list(range(0, min(len(poses), len(rgb_frames)), args.stride))
    print(f"Using {len(frame_indices)} frames (stride={args.stride})")

    # Create output directories
    img_dir = os.path.join(args.output_dir, "images")
    sparse_dir = os.path.join(args.output_dir, "sparse", "0")
    depth_dir = os.path.join(args.output_dir, "depth_maps")
    normal_dir = os.path.join(args.output_dir, "normal_maps")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    # Camera intrinsics — Replica standard
    cx = args.width / 2.0 - 0.5   # 599.5 for 1200
    cy = args.height / 2.0 - 0.5  # 339.5 for 680

    # Write cameras.txt (single PINHOLE camera)
    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {args.width} {args.height} {args.fx} {args.fy} {cx} {cy}\n")
    print(f"Camera: PINHOLE {args.width}×{args.height} f=({args.fx},{args.fy}) c=({cx},{cy})")

    # Write images.txt and process frames
    depth_coverages = []
    with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for img_id, frame_idx in enumerate(frame_indices, start=1):
            # Symlink RGB frame
            src_rgb = rgb_frames[frame_idx]
            ext = os.path.splitext(src_rgb)[1]
            frame_name = f"frame{frame_idx:06d}{ext}"
            dst_rgb = os.path.join(img_dir, frame_name)
            if not os.path.exists(dst_rgb):
                os.symlink(os.path.abspath(src_rgb), dst_rgb)

            # Convert pose to COLMAP format
            c2w = poses[frame_idx]
            R, t = pose_to_colmap(c2w)
            q = rotation_to_quaternion(R)

            f.write(f"{img_id} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                    f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} 1 {frame_name}\n")
            f.write("\n")  # Empty POINTS2D line (required by COLMAP format)

            # Convert depth map
            stem = f"frame{frame_idx:06d}"
            depth_dst = os.path.join(depth_dir, f"{stem}.npy")
            normal_dst = os.path.join(normal_dir, f"{stem}.npy")

            # Try multiple depth file name patterns
            src_base = os.path.splitext(src_rgb)[0]
            depth_candidates = [
                src_base + ".depth.png",
                src_base.replace("frame", "depth") + ".png",
                os.path.join(os.path.dirname(src_rgb), f"depth{frame_idx:06d}.png"),
            ]
            depth_src = None
            for cand in depth_candidates:
                if os.path.exists(cand):
                    depth_src = cand
                    break

            if depth_src and not os.path.exists(depth_dst):
                d_m = depth_png_to_metres(depth_src, args.depth_scale)
                np.save(depth_dst, d_m)
                cov = (d_m > 0).mean()
                depth_coverages.append(cov)

                if not os.path.exists(normal_dst):
                    normals = estimate_normals_from_depth(d_m, args.fx, args.fy, cx, cy)
                    np.save(normal_dst, normals)

            if img_id % 50 == 0:
                print(f"  Processed {img_id}/{len(frame_indices)} frames...")

    print(f"Wrote {len(frame_indices)} poses to images.txt")

    # Depth coverage report
    if depth_coverages:
        print(f"Depth coverage: {np.mean(depth_coverages)*100:.1f}% mean "
              f"({np.min(depth_coverages)*100:.1f}%-{np.max(depth_coverages)*100:.1f}%)")

    # GT mesh processing
    # NICE-SLAM Replica: mesh is at parent_dir/<scene_name>_mesh.ply (NOT inside scene dir!)
    scene_name = os.path.basename(os.path.normpath(args.scene_dir))
    parent_dir = os.path.dirname(os.path.normpath(args.scene_dir))
    gt_mesh_path = os.path.join(args.scene_dir, "mesh.ply")
    if not os.path.exists(gt_mesh_path):
        for cand in [
            os.path.join(parent_dir, f"{scene_name}_mesh.ply"),  # NICE-SLAM format
            os.path.join(args.scene_dir, "gt_mesh.ply"),
            os.path.join(args.scene_dir, "habitat", "mesh_semantic.ply"),
        ]:
            if os.path.exists(cand):
                gt_mesh_path = cand
                break

    if os.path.exists(gt_mesh_path):
        import open3d as o3d

        # Symlink GT mesh for evaluation
        dst_gt = os.path.join(args.output_dir, "gt_mesh.ply")
        if not os.path.exists(dst_gt):
            os.symlink(os.path.abspath(gt_mesh_path), dst_gt)
        print(f"GT mesh: {gt_mesh_path}")

        # Load mesh — use trimesh first (handles polygon meshes in Replica),
        # fall back to Open3D if trimesh unavailable
        print(f"Sampling {args.n_init_points:,} init points from GT mesh...")
        try:
            import trimesh
            tm = trimesh.load(gt_mesh_path, process=False)
            pts_init, face_idx = trimesh.sample.sample_surface(tm, args.n_init_points)
            # Get colors from vertex attributes if available
            if hasattr(tm.visual, 'vertex_colors') and tm.visual.vertex_colors is not None:
                vc = np.asarray(tm.visual.vertex_colors)[:, :3] / 255.0  # (N,3) float
                # Interpolate vertex colors to sample points via nearest vertex
                from scipy.spatial import cKDTree
                tree = cKDTree(np.asarray(tm.vertices))
                _, vid = tree.query(pts_init, workers=-1)
                colors = vc[vid]
            else:
                colors = np.ones((len(pts_init), 3)) * 0.5
            pts = pts_init
            print(f"  (trimesh: {len(tm.vertices):,} verts, {len(tm.faces):,} triangulated faces)")
        except Exception as e:
            print(f"  trimesh failed ({e}), falling back to Open3D")
            mesh_o3d = o3d.io.read_triangle_mesh(gt_mesh_path)
            mesh_o3d.compute_vertex_normals()
            pcd_init_o3d = mesh_o3d.sample_points_uniformly(number_of_points=args.n_init_points)
            pts = np.asarray(pcd_init_o3d.points)
            colors = np.asarray(pcd_init_o3d.colors) if pcd_init_o3d.has_colors() \
                else np.ones_like(pts) * 0.5

        # Write points3D.txt
        with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
            f.write("# 3D point list with one line per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
            for i in range(len(pts)):
                r, g, b = (np.clip(colors[i], 0, 1) * 255).astype(int)
                f.write(f"{i+1} {pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f} "
                        f"{r} {g} {b} 0.0\n")
        print(f"Wrote {len(pts):,} init points to points3D.txt")

        # Save as PLY for 2DGS initialization (must include normals for vanilla 2DGS fetchPly)
        pts_ply_path = os.path.join(sparse_dir, "points3D.ply")
        pcd_init_save = o3d.geometry.PointCloud()
        pcd_init_save.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd_init_save.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
        pcd_init_save.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        # Write in storePly format (x,y,z,nx,ny,nz,red,green,blue) expected by fetchPly
        from plyfile import PlyData, PlyElement
        _pts = np.asarray(pcd_init_save.points, dtype=np.float32)
        _nrm = np.asarray(pcd_init_save.normals, dtype=np.float32)
        _col = (np.clip(colors[:len(_pts)], 0, 1) * 255).astype(np.uint8)
        _dtype = [('x','f4'),('y','f4'),('z','f4'),
                  ('nx','f4'),('ny','f4'),('nz','f4'),
                  ('red','u1'),('green','u1'),('blue','u1')]
        _data = np.empty(len(_pts), dtype=_dtype)
        _data['x'] = _pts[:,0]; _data['y'] = _pts[:,1]; _data['z'] = _pts[:,2]
        _data['nx'] = _nrm[:,0]; _data['ny'] = _nrm[:,1]; _data['nz'] = _nrm[:,2]
        _data['red'] = _col[:,0]; _data['green'] = _col[:,1]; _data['blue'] = _col[:,2]
        PlyData([PlyElement.describe(_data, 'vertex')]).write(pts_ply_path)
        print(f"Saved {len(_pts):,} init points to points3D.ply (with normals)")

        # Sample denser point cloud for DGS (LiDAR substitute)
        lidar_ply = os.path.join(args.output_dir, "lidar_pointcloud.ply")
        if not os.path.exists(lidar_ply):
            print(f"Sampling {args.n_lidar_points:,} DGS points from GT mesh...")
            try:
                import trimesh
                tm = trimesh.load(gt_mesh_path, process=False)
                pts_dense, _ = trimesh.sample.sample_surface(tm, args.n_lidar_points)
            except Exception:
                mesh_o3d = o3d.io.read_triangle_mesh(gt_mesh_path)
                pcd_dense_o3d = mesh_o3d.sample_points_uniformly(args.n_lidar_points)
                pts_dense = np.asarray(pcd_dense_o3d.points)
            pcd_dense = o3d.geometry.PointCloud()
            pcd_dense.points = o3d.utility.Vector3dVector(pts_dense.astype(np.float64))
            pcd_dense.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            o3d.io.write_point_cloud(lidar_ply, pcd_dense)
            print(f"Saved {len(pcd_dense.points):,} DGS points to lidar_pointcloud.ply")
    else:
        print("WARNING: No GT mesh found — writing empty points3D.txt")
        with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
            f.write("# 3D point list (empty — no GT mesh found)\n")

    # Summary and depth verification
    n_images = len(glob.glob(os.path.join(img_dir, "*")))
    n_depth = len(glob.glob(os.path.join(depth_dir, "*.npy")))
    n_normal = len(glob.glob(os.path.join(normal_dir, "*.npy")))

    print(f"\n=== Summary: {args.output_dir} ===")
    print(f"  Images:      {n_images}")
    print(f"  Depth maps:  {n_depth}")
    print(f"  Normal maps: {n_normal}")
    print(f"  GT mesh:     {'yes' if os.path.exists(os.path.join(args.output_dir, 'gt_mesh.ply')) else 'no'}")
    print(f"  LiDAR PLY:   {'yes' if os.path.exists(os.path.join(args.output_dir, 'lidar_pointcloud.ply')) else 'no'}")

    # Depth sanity check
    depth_files = glob.glob(os.path.join(depth_dir, "*.npy"))
    if depth_files:
        d_sample = np.load(depth_files[len(depth_files)//2])
        valid = d_sample[d_sample > 0]
        if len(valid) > 0:
            print(f"\n  Depth sanity (middle frame):")
            print(f"    Coverage: {(d_sample>0).mean()*100:.1f}%")
            print(f"    Range:    {valid.min():.3f} - {valid.max():.3f} m")
            if valid.max() > 50 or valid.min() < 0.05:
                print(f"  WARNING: Depth range looks wrong for indoor scenes (expected 0.1-10m)!")
                print(f"           Try --depth_scale {args.depth_scale * valid.max() / 10:.1f}")


if __name__ == "__main__":
    main()
