"""
Capture raw Bayer data or post-processed RGB data.

```
python scripts/measure/on_device_capture.py legacy=True \
exp=0.02 bayer=True
```

With the Global Shutter sensor, legacy RPi software is not supported.
```
python scripts/measure/on_device_capture.py sensor=rpi_gs \
legacy=False exp=0.02 bayer=True
```

To capture PNG data (bayer=False) and downsample (by factor 2):
```
python scripts/measure/on_device_capture.py sensor=rpi_gs \
legacy=False exp=0.02 bayer=False down=2
```

See these code snippets for setting camera settings and post-processing
- https://github.com/scivision/pibayer/blob/1bb258c0f3f8571d6ded5491c0635686b59a5c4f/pibayer/base.py#L56
- https://picamera.readthedocs.io/en/release-1.13/recipes1.html#capturing-consistent-images
- https://www.strollswithmydog.com/open-raspberry-pi-high-quality-camera-raw

"""

import hydra
import os
import cv2
import numpy as np
from time import sleep
from PIL import Image
from lensless.hardware.utils import get_distro
from lensless.utils.image import bayer2rgb_cc, rgb2gray, resize
from lensless.hardware.constants import RPI_HQ_CAMERA_CCM_MATRIX, RPI_HQ_CAMERA_BLACK_LEVEL
from lensless.hardware.sensor import SensorOptions, sensor_dict, SensorParam
from fractions import Fraction
import time


SENSOR_MODES = [
    "off",
    "auto",
    "sunlight",
    "cloudy",
    "shade",
    "tungsten",
    "fluorescent",
    "incandescent",
    "flash",
    "horizon",
]


@hydra.main(version_base=None, config_path="../../configs", config_name="capture")
def capture(config):

    sensor = config.sensor
    assert sensor in SensorOptions.values(), f"Sensor must be one of {SensorOptions.values()}"

    bayer = config.bayer
    fn = config.fn
    exp = config.exp
    config_pause = config.config_pause
    sensor_mode = config.sensor_mode
    rgb = config.rgb
    gray = config.gray
    greenscale = config.greenscale
    iso = config.iso
    sixteen = config.sixteen
    legacy = config.legacy
    down = config.down
    res = config.res
    nbits_out = config.nbits_out
    rgain = config.awb_gains[0]
    bgain = config.awb_gains[1]

    assert (
        nbits_out in sensor_dict[sensor][SensorParam.BIT_DEPTH]
    ), f"nbits_out must be one of {sensor_dict[sensor][SensorParam.BIT_DEPTH]} for sensor {sensor}"

    # https://www.raspberrypi.com/documentation/accessories/camera.html#hardware-specification
    sensor_param = sensor_dict[sensor]
    assert exp <= sensor_param[SensorParam.MAX_EXPOSURE]
    assert exp >= sensor_param[SensorParam.MIN_EXPOSURE]
    sensor_mode = int(sensor_mode)

    distro = get_distro()
    print("RPi distribution : {}".format(distro))

    if sensor == SensorOptions.ARDU_708.value:
        assert not legacy

    if "bookworm" in distro and not legacy:
        # TODO : grayscale and downsample

        import subprocess

        if bayer:
            assert not rgb
            assert not gray
            assert not greenscale
            assert down is None

            # https://www.raspberrypi.com/documentation/computers/camera_software.html#raw-image-capture
            jpg_fn = fn + ".jpg"
            fn += ".dng"
            pic_command = [
                "libcamera-still",
                "-r",
                "--gain",
                f"{iso / 100}",
                "--shutter",
                f"{int(exp * 1e6)}",
                "-o",
                f"{jpg_fn}",
                # long exposure: https://www.raspberrypi.com/documentation/computers/camera_software.html#very-long-exposures
                # -- setting awbgains caused issues
                "--awbgains",
                f"{rgain},{bgain}"
                # "--immediate"
            ]

            cmd = subprocess.Popen(
                pic_command,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            cmd.stdout.readlines()
            cmd.stderr.readlines()
            # os.remove(jpg_fn)
            os.system(f"exiftool {fn}")
            print("\nJPG saved to : {}".format(jpg_fn))
            # print("\nDNG saved to : {}".format(fn))

        else:  # --- non-Bayer branch -------------------------------------------------
            #
            # Capture a frame with Picamera2 and optionally store:
            #   • grayscale   → .npy + preview .jpg   (gray=True)
            #   • greenscale  → .npy + preview .jpg   (greenscale=True)
            #   • colour      → .png                  (default)
            #
            from picamera2 import Picamera2
            import cv2, numpy as np, time

            picam2 = Picamera2()

            # -----------------------------------------------------------------------
            #  Filenames
            # -----------------------------------------------------------------------                     # stem
            png_fn  = fn + ".png"
            jpg_fn  = fn + ".jpg"
            npy_fn  = fn + ".npy"

            # If both flags set, prefer grayscale
            if gray and greenscale:
                print("[Warn] Both gray and greenscale requested – defaulting to gray.")
                greenscale = False

            # -----------------------------------------------------------------------
            #  Resolution
            # -----------------------------------------------------------------------
            sensor_max = picam2.camera_properties["PixelArraySize"]   # e.g. (4608, 2592)
            res = tuple(res) if res else tuple((np.array(sensor_max) // (down or 1)).astype(int))
            print("Capturing at resolution:", res)

            # -----------------------------------------------------------------------
            #  Build configurations
            # -----------------------------------------------------------------------
            manual_ctrls = {
                "AeEnable": 0,
                "ExposureTime": int(exp * 1e6),            # μs
                "AnalogueGain": iso / 100,
            }
            if config.awb_gains:
                manual_ctrls.update({
                    "AwbEnable": 0,
                    "ColourGains": tuple(config.awb_gains),
                })

            preview_cfg = picam2.create_preview_configuration(
                main={"size": res, "format": "RGB888"},
                controls=manual_ctrls,
            )
            still_cfg = picam2.create_still_configuration(
                main={"size": res, "format": "RGB888"},
                buffer_count=2,
                controls=manual_ctrls,
            )

            picam2.configure(preview_cfg)
            picam2.start(show_preview=False)
            time.sleep(config.config_pause)                # let sensor settle

            # -----------------------------------------------------------------------
            #  Capture & save
            # -----------------------------------------------------------------------
            # helper for directory safety
            def ensure_dir(path):
                d = os.path.dirname(path)
                if d and not os.path.isdir(d):
                    os.makedirs(d, exist_ok=True)

            # -----------------------------------------------------------------
            # Capture & save
            # -----------------------------------------------------------------
            if gray or greenscale:
                frame = picam2.switch_mode_and_capture_array(
                            still_cfg, name="main", delay=3)      # H×W×3

                if gray:
                    green2d = frame[..., 1]                       # (H,W)

                    # ---- make 4-D tensor: (1,H,W,1) ---------------------------------
                    green4d = green2d[np.newaxis, ..., np.newaxis]

                    ensure_dir(npy_fn);  np.save(npy_fn, green4d)
                    print(f"[Saved]   {npy_fn}  (shape {green4d.shape})")

                    ensure_dir(jpg_fn);   cv2.imwrite(jpg_fn, green2d)
                    print(f"[Preview] {jpg_fn}")

                else:  # greenscale → RGB 0,G,0
                    out_arr = np.zeros_like(frame)                # H×W×3
                    out_arr[..., 1] = frame[..., 1]

                    # ---- add ‘depth’ axis → (1,H,W,3) ----------
                    out4d = out_arr[np.newaxis, ...]

                    ensure_dir(npy_fn);  np.save(npy_fn, out4d)
                    print(f"[Saved]   {npy_fn}  (shape {out4d.shape})")

                    ensure_dir(jpg_fn);  cv2.imwrite(jpg_fn, out_arr)
                    print(f"[Preview] {jpg_fn}")

            else:
                picam2.switch_mode_and_capture_file(
                    still_cfg,
                    png_fn,
                    name="main",
                    delay=3,
                )
                print(f"[Saved]   {png_fn}")

            # -----------------------------------------------------------------------
            #  Clean up
            # -----------------------------------------------------------------------
            picam2.stop()
            picam2.close()


    print("Image saved to : {}".format(fn))


if __name__ == "__main__":
    capture()
