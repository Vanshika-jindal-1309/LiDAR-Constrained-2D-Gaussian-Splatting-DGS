#!/usr/bin/env python3
"""
Convert masked JPEG images to RGBA PNG where black pixels become transparent (alpha=0).
The alpha channel is then used by 2DGS to exclude masked pixels from the training loss.

Usage:
    python scripts/jpg_to_masked_png.py <images_dir> [--threshold 10] [--suffix _masked]

    <images_dir>  Root folder containing camera_N sub-folders of JPEGs.
                  Converted PNGs are written to <images_dir>_masked/ (or --out_dir).

Example:
    python scripts/jpg_to_masked_png.py \
        perspective/images \
        --out_dir perspective/images_masked
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def convert(src_path: Path, dst_path: Path, threshold: int):
    img = Image.open(src_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    # Mask = pixels where all channels are <= threshold (solid black)
    mask = (arr.max(axis=2) > threshold).astype(np.uint8) * 255  # 255=valid, 0=masked

    rgba = np.dstack([arr, mask])
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(dst_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert masked JPEGs to RGBA PNGs for 2DGS masked training"
    )
    parser.add_argument("images_dir", help="Directory containing camera_N/*.jpg")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: <images_dir>_masked)")
    parser.add_argument("--threshold", type=int, default=10,
                        help="Pixels with max(R,G,B) <= threshold are treated as masked (default: 10)")
    args = parser.parse_args()

    src_root = Path(args.images_dir)
    dst_root = Path(args.out_dir) if args.out_dir else src_root.parent / (src_root.name + "_masked")

    jpgs = sorted(src_root.rglob("*.jpg")) + sorted(src_root.rglob("*.JPG"))
    if not jpgs:
        print(f"No JPEGs found under {src_root}")
        return

    print(f"Converting {len(jpgs)} images: {src_root} → {dst_root}")
    for src in tqdm(jpgs):
        rel = src.relative_to(src_root)
        dst = (dst_root / rel).with_suffix(".png")
        convert(src, dst, args.threshold)

    print(f"\nDone. Use --images images_masked in your train.py command.")


if __name__ == "__main__":
    main()
