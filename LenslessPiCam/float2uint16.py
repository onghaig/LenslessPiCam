#!/usr/bin/env python3
"""
float2uint16.py
---------------
Convert .npy arrays from float (or uint8) to uint16.

Usage
-----
    python float2uint16.py data.npy psf.npy
    # keep originals, write *_u16.npy
    python float2uint16.py data.npy psf.npy --suffix _u16
"""

import argparse
from pathlib import Path
import numpy as np
import sys


def convert(path: Path, suffix: str = "") -> None:
    arr = np.load(path)
    out_path = path if suffix == "" else path.with_stem(path.stem + suffix)

    # skip if already uint16
    if arr.dtype == np.uint16:
        print(f"[skip] {path.name}: already uint16")
        return

    # promote integers (e.g. uint8) to float for scaling
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32)

    if not np.issubdtype(arr.dtype, np.floating):
        print(f"[err ] {path.name}: unsupported dtype {arr.dtype}", file=sys.stderr)
        return

    max_val = arr.max()
    if max_val == 0:
        print(f"[warn] {path.name}: max value is zero; leaving array zeros")
        arr_u16 = np.zeros_like(arr, dtype=np.uint16)
    else:
        arr_u16 = np.clip(arr / max_val * 65535.0, 0, 65535).astype(np.uint16)

    np.save(out_path, arr_u16)
    print(f"[ok  ] {out_path.name}: shape {arr_u16.shape}, dtype uint16")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .npy float arrays to uint16.")
    parser.add_argument("files", nargs="+", help="input .npy files")
    parser.add_argument("--suffix", default="", help="append suffix before .npy instead of overwriting")
    args = parser.parse_args()

    for f in args.files:
        p = Path(f)
        if not p.is_file():
            print(f"[err ] {p} not found", file=sys.stderr)
            continue
        convert(p, suffix=args.suffix)


if __name__ == "__main__":
    main()
