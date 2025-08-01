#!/usr/bin/env python3
"""
squeeze_depth.py
----------------
Drop the leading depth dimension (axis 0) when it has length 1.

Usage
-----
    python squeeze_depth.py file1.npy file2.npy ...

After running, each file that was (1, H, W, C) or (1, H, W) becomes
(H, W, C) or (H, W), stored as float32.
"""

import argparse
import numpy as np
import sys
from pathlib import Path


def squeeze_file(path: Path) -> None:
    arr = np.load(path, mmap_mode=None)          # load into RAM
    if arr.shape[0] != 1:
        print(f"[skip] {path}: depth axis length is {arr.shape[0]} (expected 1)")
        return

    squeezed = arr.squeeze(0).astype(np.uint16)  # drop axis, cast
    np.save(path, squeezed)
    print(f"[ok]   {path}: shape {squeezed.shape}, dtype {squeezed.dtype}")


def main():
    parser = argparse.ArgumentParser(description="Remove singleton depth axis from .npy files.")
    parser.add_argument("files", nargs="+", help="one or more .npy files")
    args = parser.parse_args()

    for file in args.files:
        p = Path(file)
        if not p.is_file():
            print(f"[err]  {p} does not exist", file=sys.stderr)
            continue
        squeeze_file(p)


if __name__ == "__main__":
    main()
