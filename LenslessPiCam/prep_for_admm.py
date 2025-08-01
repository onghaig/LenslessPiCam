#!/usr/bin/env python3
"""
prep_for_admm.py
----------------
Convert capture .npy files for Lensless ADMM.

 • PSF  (H,W,3) → (1,H,W,1) uint16
 • Data (H,W,3) → (H,W,1)  uint16  (no leading depth axis)

Usage
-----
    python prep_for_admm.py psf.npy data.npy
    python prep_for_admm.py psf.npy data.npy --suffix _g
"""

import argparse, sys
from pathlib import Path
import numpy as np


def to_uint16(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint16:
        return arr
    arr = arr.astype(np.float32)
    maxv = arr.max() or 1.0
    return np.clip(arr / maxv * 65535, 0, 65535).astype(np.uint16)


def convert_psf(path: Path, suffix: str):
    arr = np.load(path)                   # (H,W,3) or already 4-D
    if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[-1] == 1:
        print(f"[skip] {path.name}: already (1,H,W,1)")
        return
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[..., 1:2]               # green → (H,W,1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        pass
    else:
        sys.exit(f"[err] {path.name}: unexpected shape {arr.shape}")
    arr = arr[np.newaxis, ...]            # (1,H,W,1)
    arr = to_uint16(arr)

    out = path if suffix == "" else path.with_stem(path.stem + suffix)
    np.save(out, arr)
    print(f"[psf ] {out.name}: {arr.shape}, {arr.dtype}")


def convert_data(path: Path, suffix: str):
    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        print(f"[skip] {path.name}: already (H,W,1)")
        return
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr.squeeze(0)              # drop depth
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[..., 1:2]               # green
    elif arr.ndim != 3 or arr.shape[-1] != 1:
        sys.exit(f"[err] {path.name}: unexpected shape {arr.shape}")
    arr = to_uint16(arr)

    out = path if suffix == "" else path.with_stem(path.stem + suffix)
    np.save(out, arr)
    print(f"[data] {out.name}: {arr.shape}, {arr.dtype}")


def main():
    p = argparse.ArgumentParser(description="Prep PSF & data for ADMM.")
    p.add_argument("psf")
    p.add_argument("data")
    p.add_argument("--suffix", default="", help="append suffix instead of overwrite")
    a = p.parse_args()

    convert_psf(Path(a.psf), a.suffix)
    convert_data(Path(a.data), a.suffix)


if __name__ == "__main__":
    main()
