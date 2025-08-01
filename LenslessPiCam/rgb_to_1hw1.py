#!/usr/bin/env python3
"""
rgb_to_1hw1.py
--------------
Convert (H,W,3) RGB NumPy arrays to (1,H,W,1) single-channel stacks
suitable for Lensless-PiCam ADMM.

Usage
-----
    # overwrite in place
    python rgb_to_1hw1.py psf_rgb.npy data_rgb.npy

    # keep originals, create *_1hw1.npy
    python rgb_to_1hw1.py psf_rgb.npy data_rgb.npy --suffix _1hw1
"""

import argparse
from pathlib import Path
import numpy as np
import sys


def convert(path: Path, suffix: str = "") -> None:
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        print(f"[skip] {path.name}: expected (H,W,3), got {arr.shape}", file=sys.stderr)
        return

    # --- choose single channel -------------------------------------------------
    # keep green plane; change to arr.mean(-1, keepdims=True) for luminance
    single = arr[..., 1:2]                    # shape (H,W,1)

    # --- add depth axis, convert to uint16 -------------------------------------
    arr4 = single[np.newaxis, ...].astype(np.uint16)   # (1,H,W,1)

    out_path = path if suffix == "" else path.with_stem(path.stem + suffix)
    np.save(out_path, arr4)
    print(f"[ok] {out_path.name}: shape {arr4.shape}, dtype uint16")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert (H,W,3) .npy to (1,H,W,1).")
    parser.add_argument("files", nargs="+", help="input .npy files")
    parser.add_argument("--suffix", default="", help="append suffix instead of overwriting")
    args = parser.parse_args()

    for fname in args.files:
        p = Path(fname)
        if not p.is_file():
            print(f"[err] {p} not found", file=sys.stderr)
            continue
        convert(p, suffix=args.suffix)


if __name__ == "__main__":
    main()
