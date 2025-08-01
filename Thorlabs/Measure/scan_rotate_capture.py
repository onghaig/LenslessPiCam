#!/usr/bin/env python3
"""
scan_rotate_capture.py  (v0.2)
"""

from locale import currency
import argparse, os, subprocess
from datetime import datetime
from pathlib import Path
from pylablib.devices import Thorlabs


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def find_repo_root() -> Path:
    """Two levels up from this file -> <repo_root>/Thorlabs/Setup/.. -> <repo_root>"""
    return Path(__file__).resolve().parents[2]


def run_capture(stem: Path, capture_script: Path,
                sensor: str, exp: float, bayer: bool, extra: list[str]):
    cmd = [
        "python", str(capture_script),
        f"fn={stem}",
        f"sensor={sensor}",
        "legacy=False",
        f"exp={exp}",
        f"bayer={str(bayer).lower()}"
    ] + extra
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(" └─ capture complete\n")


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    root_default = find_repo_root()
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--output-dir", default="./captures")
    p.add_argument("--sensor", default="rpi_gs")
    p.add_argument("--exp", type=float, default=0.02)
    p.add_argument("--bayer", action="store_true")
    p.add_argument("--step", type=float, default=5.0)
    p.add_argument("--start", type=float, default=-45.0)
    p.add_argument("--stop", type=float, default=45.0)
    p.add_argument("--capture-script", type=Path,
                   default=root_default / "/home/user/OriginalLenslessPiCam/LenslessPiCam/scripts/measure/on_device_capture.py",
                   help="Full path to on_device_capture.py")
    p.add_argument("--extra", nargs=argparse.REMAINDER,
                   help="Additional Hydra overrides")
    args = p.parse_args()

    outdir = Path(args.output_dir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    stage = Thorlabs.KinesisMotor(args.port, scale="stage")
    print("[Stage] Device:", stage.get_device_info())

    curr = args.start
    try:
        stage.home(); stage.wait_move()
        stage.move_to(curr); stage.wait_move()

        while curr <= args.stop:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = outdir / f"angle_{curr:+03.0f}_{timestamp}" #negated due to curr angle being flipped.
            run_capture(
                stem, args.capture_script,
                sensor=args.sensor,
                exp=args.exp,
                bayer=args.bayer,
                extra=(args.extra or [])
            )
            if curr + args.step <= args.stop:
                curr += args.step
                stage.move_to(curr); stage.wait_move()
            else:
                break

        stage.move_to(0); stage.wait_move()
    finally:
        stage.close()
        print("[Stage] Connection closed.")


if __name__ == "__main__":
    main()
