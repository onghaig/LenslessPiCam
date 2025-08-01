#!/usr/bin/env python3
"""
analyze_pixel_vs_angle.py
=========================

Extract the value of a single pixel from every captured frame in a
wave-plate sweep and inspect how it changes with angle.

Usage examples
--------------
# 1) Quick look with default centre pixel, show plot
python analyze_pixel_vs_angle.py /path/to/Thorlabs/7-21-2025 --show

# 2) Specify pixel (x=150, y=200) and export CSV only
python analyze_pixel_vs_angle.py /path/to/Thorlabs/7-21-2025 \
        --xy 150 200 --outfile pixel150_200.csv
"""
import argparse, csv, re, sys  
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2           # for PNG/JPG
try:
    import rawpy     # for DNG
except ImportError:
    rawpy = None


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
ANGLE_RE = re.compile(r"angle_([+-]?\d+)", re.I)   # filename → ±45, +05, etc.


def extract_angle(path: Path) -> float | None:
    """Return angle in degrees parsed from file stem or None."""
    m = ANGLE_RE.search(path.stem)
    return float(m.group(1)) if m else None


def read_image(path: Path) -> np.ndarray:
    """Load the image/array and return H×W×C ndarray with dtype float32 [0-1]."""
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path).squeeze()
        if arr.ndim == 2:                       # H×W
            arr = arr[..., None]
        return arr.astype(np.float32)
    elif ext in {".png", ".jpg", ".jpeg"}:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"cv2 failed to open {path}")
        if img.ndim == 2:                       # grayscale → H×W×1
            img = img[..., None]
        img = img.astype(np.float32) / img.max()
        return img
    elif ext == ".dng":
        if rawpy is None:
            sys.exit("[Error] pip install rawpy to read DNG/Bayer files.")
        with rawpy.imread(str(path)) as raw:
            # get visible RGB for quick analysis – you can change
            img = raw.postprocess(gamma=(1, 1), no_auto_bright=True)
            return img.astype(np.float32) / 65535.0
    else:
        raise ValueError(f"Unsupported file type: {path.name}")


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("folder", type=Path,
                   help="Directory containing capture files")
    p.add_argument("--xy", type=int, nargs=2, metavar=("X", "Y"),
                   help="Pixel coordinate (origin top-left). Default = centre")
    p.add_argument("--outfile", default="pixel_vs_angle.csv",
                   help="CSV filename to write results")
    p.add_argument("--show", action="store_true",
                   help="Display plot")
    args = p.parse_args()

    # Collect and sort files
    all_files = [f for f in args.folder.iterdir() if f.is_file() and extract_angle(f) is not None]
    npy_files = [f for f in all_files if f.suffix.lower() == ".npy"]
    dng_files = [f for f in all_files if f.suffix.lower() == ".dng"]

    if npy_files:
        files = sorted(npy_files)
        print(f"[Info] Found {len(files)} .npy files, using only those.")
    elif dng_files:
        files = sorted(dng_files)
        print(f"[Info] Found {len(files)} .dng files, using only those.")
    else:
        files = sorted(all_files)
        print(f"[Info] Using all supported files.")

    if not files:
        sys.exit(f"No angle_* files found in {args.folder}")

    data = []   # (angle, ch0, ch1, ch2) or (angle, gray)

    # First pass to find image size and default centre pixel
    sample_img = read_image(files[0])
    h, w = sample_img.shape[:2]
    x, y = args.xy if args.xy else (w//2, h//2)
    if not (0 <= x < w and 0 <= y < h):
        sys.exit(f"Pixel ({x},{y}) outside image bounds ({w}×{h})")

    print(f"[Info] Using pixel ({x}, {y}) on {w}×{h} frames")

    # Loop through captures
    for fp in files:
        ang = extract_angle(fp)
        img = read_image(fp)
        pix = img[y, x]              # shape (C,)  or scalar if 1-chan
        pix = np.atleast_1d(pix)     # force length ≥1
        row = [ang] + pix.tolist()
        data.append(row)

    # Write CSV
    header = ["angle_deg"] + [f"ch{i}" for i in range(len(data[0])-1)]
    with open(args.outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print(f"[Saved] {args.outfile}")

    # Plot
    if args.show:
        data = np.array(data)
        angles = data[:, 0]
        for i in range(data.shape[1]-1):
            plt.plot(angles, data[:, i+1], label=f"ch{i}")
        plt.xlabel("Wave-plate angle (deg)")
        plt.ylabel("Pixel value (normalised)")
        plt.title(f"Pixel ({x},{y}) response vs angle")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
