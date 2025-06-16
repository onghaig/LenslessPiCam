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
    iso = config.iso
    sixteen = config.sixteen
    legacy = config.legacy
    down = config.down
    res = config.res
    nbits_out = config.nbits_out

    # Create output directory if specified
    if hasattr(config, 'output_dir'):
        os.makedirs(config.output_dir, exist_ok=True)
        fn = os.path.join(config.output_dir, fn)

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

    # Use libcamera for Raspberry Pi 5 and IMX708
    if sensor == SensorOptions.ARDUCAM_708.value or "bookworm" in distro:
        assert not legacy
        import subprocess

        if bayer:
            # Use libcamera-still for raw capture
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
            ]

            # Add sensor-specific settings for IMX708
            if sensor == SensorOptions.ARDUCAM_708.value:
                pic_command.extend([
                    "--sensor-mode", "0",  # Full resolution mode
                    "--awb-mode", "off",
                ])

            cmd = subprocess.Popen(
                pic_command,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            cmd.stdout.readlines()
            cmd.stderr.readlines()
            
            # Check if DNG file was created
            if not os.path.exists(fn):
                print(f"Warning: DNG file {fn} was not created. Check if libcamera-still is working correctly.")
            else:
                os.system(f"exiftool {fn}")
            
            print("\nJPG saved to : {}".format(jpg_fn))
            print("\nDNG saved to : {}".format(fn))
        else:
            # Use libcamera-still for processed capture
            fn += ".jpg"
            pic_command = [
                "libcamera-still",
                "--gain",
                f"{iso / 100}",
                "--shutter",
                f"{int(exp * 1e6)}",
                "-o",
                f"{fn}",
            ]

            # Add sensor-specific settings for IMX708
            if sensor == SensorOptions.ARDUCAM_708.value:
                pic_command.extend([
                    "--sensor-mode", "0",  # Full resolution mode
                    "--awb-mode", "off",
                ])

            if down:
                pic_command.extend(["--width", str(res[0]), "--height", str(res[1])])

            cmd = subprocess.Popen(
                pic_command,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            cmd.stdout.readlines()
            cmd.stderr.readlines()
            print("\nImage saved to : {}".format(fn))

    # legacy camera software for older sensors
    elif sensor == SensorOptions.RPI_GS.value and not "bookworm" in distro:
        assert not legacy
        # ... rest of the legacy code ...
    else:
        raise ValueError(f"Unsupported sensor {sensor} with legacy={legacy} on {distro}")

    print("Image saved to : {}".format(fn))


if __name__ == "__main__":
    capture()
