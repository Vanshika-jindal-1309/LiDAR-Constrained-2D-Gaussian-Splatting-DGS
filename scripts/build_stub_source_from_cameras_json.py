"""
Build a minimal COLMAP-format source directory from cameras.json (saved in output dir).
This allows render.py mesh extraction to run without the original SSD source data.

Usage:
    python scripts/build_stub_source_from_cameras_json.py \
        --cameras_json output/<model_dir>/cameras.json \
        --output_dir /tmp/stub_<scene_id> \
        [--images_folder images_masked] [--image_ext png]
"""
import argparse
import json
import os
import struct
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to COLMAP quaternion (qw, qx, qy, qz)."""
    r = Rotation.from_matrix(R)
    q = r.as_quat()  # [qx, qy, qz, qw] (scipy convention)
    return q[3], q[0], q[1], q[2]  # COLMAP: [qw, qx, qy, qz]


def build_stub(cameras_json_path, output_dir, images_folder, image_ext, stub_image_size):
    with open(cameras_json_path) as f:
        cams = json.load(f)

    print(f"Loaded {len(cams)} cameras from {cameras_json_path}")

    sparse_dir = os.path.join(output_dir, "sparse", "0")
    images_dir = os.path.join(output_dir, images_folder)
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Determine camera intrinsics — use mode (most cameras share same params)
    widths = [c["width"] for c in cams]
    heights = [c["height"] for c in cams]
    fxs = [c["fx"] for c in cams]
    fys = [c["fy"] for c in cams]

    # Single camera model (PINHOLE) for all cameras
    W = widths[0]
    H = heights[0]
    fx = np.median(fxs)
    fy = np.median(fys)
    cx = W / 2.0
    cy = H / 2.0

    print(f"Intrinsics: W={W}, H={H}, fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")

    # Write cameras.txt
    cameras_txt = os.path.join(sparse_dir, "cameras.txt")
    with open(cameras_txt, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {W} {H} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
    print(f"Wrote cameras.txt ({cameras_txt})")

    # Write images.txt
    # cameras.json convention:
    #   rotation = W2C[:3,:3] (world-to-camera rotation matrix R_cw)
    #   position = camera center in world (C)
    # COLMAP images.txt: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    #   Q = quaternion of R_cw
    #   T = -R_cw @ C  (translation so P_cam = R_cw @ P_world + T)
    images_txt = os.path.join(sparse_dir, "images.txt")
    with open(images_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(cams)}\n")
        for i, cam in enumerate(cams):
            R_cw = np.array(cam["rotation"])  # 3x3 world-to-cam rotation
            C = np.array(cam["position"])       # camera center in world
            T = -R_cw @ C                        # world-to-cam translation
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R_cw)
            img_name = cam["img_name"]
            img_file = img_name + "." + image_ext
            f.write(f"{i+1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{T[0]:.9f} {T[1]:.9f} {T[2]:.9f} 1 {img_file}\n")
            f.write("\n")  # empty POINTS2D line
    print(f"Wrote images.txt with {len(cams)} cameras ({images_txt})")

    # Write empty points3D.txt
    points_txt = os.path.join(sparse_dir, "points3D.txt")
    with open(points_txt, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0\n")
    print(f"Wrote points3D.txt (empty)")

    # Write minimal points3D.ply (required by scene loader's fetchPly)
    points_ply = os.path.join(sparse_dir, "points3D.ply")
    _write_minimal_ply(points_ply)
    print(f"Wrote points3D.ply (1 dummy point)")

    # Create stub images (very small placeholder PNGs to pass Image.open)
    sw, sh = stub_image_size
    stub_img = Image.fromarray(np.full((sh, sw, 4), 255, dtype=np.uint8), mode="RGBA")
    n_created = 0
    for cam in cams:
        img_name = cam["img_name"] + "." + image_ext
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            stub_img.save(img_path)
            n_created += 1
    print(f"Created {n_created} stub images ({sw}x{sh} RGBA) in {images_dir}/")

    print(f"\nStub source directory ready: {output_dir}")
    print(f"Use this as -s argument in render.py:")
    print(f"  python render.py -s {output_dir} -m <model_path> --iteration 30000 \\")
    print(f"    --skip_train --skip_test --mesh_res 1024 --depth_trunc 10.0 --num_cluster 50")


def _write_minimal_ply(path):
    """Write a PLY file with 1 dummy point (x,y,z,nx,ny,nz,r,g,b)."""
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(struct.pack("<ffffffBBB", 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 128, 128, 128))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build stub COLMAP source dir from cameras.json")
    parser.add_argument("--cameras_json", required=True,
                        help="Path to cameras.json saved by 2DGS training")
    parser.add_argument("--output_dir", required=True,
                        help="Output stub source directory path")
    parser.add_argument("--images_folder", default="images_masked",
                        help="Images subfolder name (default: images_masked)")
    parser.add_argument("--image_ext", default="png",
                        help="Image file extension (default: png)")
    parser.add_argument("--stub_size", type=int, nargs=2, default=[4, 4],
                        metavar=("W", "H"),
                        help="Stub image dimensions (default: 4 4). "
                             "Small enough to be fast, still valid for PIL.open().")
    args = parser.parse_args()
    build_stub(args.cameras_json, args.output_dir, args.images_folder,
               args.image_ext, tuple(args.stub_size))
