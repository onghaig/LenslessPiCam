import click
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from lensless.utils.image import rgb2gray, gamma_correction, resize
from lensless.utils.plot import plot_image, pixel_histogram, plot_cross_section, plot_autocorr2d
from lensless.utils.io import load_image, load_psf, save_image


import numpy as np
import cv2

import numpy as np
import cv2

def compute_awb_gains(rgb,
                      mask_pct=(5, 99),        # keep more pixels
                      clip=(0.5, 4.0),         # allow stronger red gain
                      pctl=80,                 # use 80-th percentile
                      target_ratio=0.9):       # R should land ~90 % of G
    """
    AWB tuned for lensless PSF frames with weak red channel.
    """
    lum = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2GRAY)
    lo, hi = np.percentile(lum, mask_pct)
    mask = (lum > lo) & (lum < hi)

    # robust statistic: percentile (defaults to median for 50 %)
    def p(img, q): return np.percentile(img[mask], q)

    r_stat = p(rgb[..., 0], pctl)
    g_stat = p(rgb[..., 1], pctl)
    b_stat = p(rgb[..., 2], pctl)

    eps = 1e-6
    rg = np.clip(target_ratio * g_stat / (r_stat + eps), *clip)
    bg = np.clip(            g_stat / (b_stat + eps), *clip)

    return float(rg), float(bg)




@click.command()
@click.option("--fp", type=str, help="File path for measurement.")
@click.option("--gamma", default=2.2, type=float, help="Gamma factor for plotting.")
@click.option("--width", default=3, type=float, help="dB drop for estimating width")
@click.option("--bayer", is_flag=True, help="Whether image is raw bayer data.")
@click.option("--lens", is_flag=True, help="Whether measurement is PSF of lens.")
@click.option("--lensless", is_flag=True, help="Whether measurement is PSF of a lensless camera.")
@click.option("--bg", type=float, help="Blue gain.")
@click.option("--rg", type=float, help="Red gain.")
@click.option("--plot_width", type=int, help="Width for cross-section.")
@click.option("--save", type=str, help="File name to save color correct bayer as RGB.")
@click.option("--save_auto", is_flag=True, help="Save autocorrelation instead of pop-up window.")
@click.option("--nbits", default=None, type=int, help="Number of bits for output. Only used for Bayer data")
@click.option("--down", default=1, type=int, help="Factor by which to downsample.")
@click.option("--back", type=str, help="File path for background image.")
@click.option("--auto_gain", is_flag=True, help="Automatically compute red and blue gains from image content.")
def analyze_image(
    fp, gamma, width, bayer, lens, lensless, bg, rg, plot_width,
    save, save_auto, nbits, down, back, auto_gain
):
    assert fp is not None, "Must pass file path."

    # Set default gains if not provided
    rg = rg if rg is not None else 1.0
    bg = bg if bg is not None else 1.0

    # Auto-gain step: first load image with neutral gains and analyze
    if auto_gain:
        print("[Auto Gain] Loading image for AWB analysis...")
        if lensless:
            img_temp = load_psf(
                fp,
                verbose=False,
                bayer=bayer,
                blue_gain=1.0,
                red_gain=1.0,
                nbits_out=nbits,
                return_float=False,
                downsample=down,
            )[0]
        else:
            img_temp = load_image(
                fp,
                verbose=False,
                bayer=bayer,
                blue_gain=1.0,
                red_gain=1.0,
                nbits_out=nbits,
                back=back,
                downsample=down,
            )
        rg, bg = compute_awb_gains(img_temp)
        print(f"[Auto Gain] Reloading image with gains: rg={rg:.3f}, bg={bg:.3f}")

    # Final image load with proper gains
    if lensless:
        img = load_psf(
            fp,
            verbose=True,
            bayer=bayer,
            blue_gain=bg,
            red_gain=rg,
            nbits_out=nbits,
            return_float=False,
            downsample=down,
        )[0]
    else:
        img = load_image(
            fp,
            verbose=True,
            bayer=bayer,
            blue_gain=bg,
            red_gain=rg,
            nbits_out=nbits,
            back=back,
            downsample=down,
        )

    # Auto-infer nbits if not provided
    if nbits is None:
        nbits = int(np.ceil(np.log2(img.max())))

    # Initialize plots
    fig_rgb, ax_rgb = plt.subplots(ncols=2, nrows=1, num="RGB", figsize=(15, 5))
    if lens:
        fig_gray, ax_gray = plt.subplots(ncols=3, nrows=1, num="Grayscale", figsize=(15, 5))
    else:
        fig_gray, ax_gray = plt.subplots(ncols=2, nrows=1, num="Grayscale", figsize=(15, 5))

    # Plot RGB
    ax = plot_image(img, gamma=gamma, normalize=True, ax=ax_rgb[0])
    ax.set_title("RGB")
    ax = pixel_histogram(img, ax=ax_rgb[1], nbits=nbits)
    ax.set_title("Histogram")
    fig_rgb.savefig(os.path.join(os.path.dirname(fp), "rgb_analysis.png"))

    # Grayscale image
    img_grey = rgb2gray(img[None, ...])
    ax = plot_image(img_grey, gamma=gamma, normalize=True, ax=ax_gray[0])
    ax.set_title("Grayscale")
    ax = pixel_histogram(img_grey, ax=ax_gray[1], nbits=nbits)
    ax.set_title("Histogram")
    fig_gray.savefig(os.path.join(os.path.dirname(fp), "grey_analysis.png"))

    img_grey = img_grey.squeeze()
    img = img.squeeze()

    # Width and autocorrelations
    if lens:
        plot_cross_section(img_grey, color="gray", plot_db_drop=width, ax=ax_gray[2], plot_width=plot_width)
        fig_auto, ax_cross = plt.subplots(ncols=3, nrows=1, num="RGB widths", figsize=(15, 5))
        for i, c in enumerate(["r", "g", "b"]):
            print(f"-- {c} channel")
            ax, _ = plot_cross_section(
                img[:, :, i],
                color=c,
                ax=ax_cross[i],
                plot_db_drop=width,
                max_val=2**nbits - 1,
                plot_width=plot_width,
            )
            if i > 0:
                ax.set_ylabel("")
    elif lensless:
        fig_auto, ax_auto = plt.subplots(ncols=4, nrows=2, num="Autocorrelations", figsize=(15, 5))
        _, autocorr_grey = plot_autocorr2d(img_grey, ax=ax_auto[0][0])
        print("-- grayscale")
        plot_cross_section(
            autocorr_grey, color="gray", plot_db_drop=width, ax=ax_auto[1][0], plot_width=plot_width
        )
        for i, c in enumerate(["r", "g", "b"]):
            _, autocorr_c = plot_autocorr2d(img[:, :, i], ax=ax_auto[0][i + 1])
            print(f"-- {c} channel")
            ax, _ = plot_cross_section(
                autocorr_c,
                color=c,
                ax=ax_auto[1][i + 1],
                plot_db_drop=width,
                plot_width=plot_width,
            )
            ax.set_ylabel("")

    # Save corrected color image
    if bayer and save is not None:
        cv2.imwrite(save, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"\nColor-corrected RGB image saved to: {save}")
        if gamma is not None:
            img = img / img.max()
            img = gamma_correction(img, gamma=gamma)
        save_8bit = save.replace(".png", "_8bit.png")
        save_image(img, save_8bit, normalize=True)
        print(f"\n8bit version saved to: {save_8bit}")

    # Save or show autocorrelation
    if save_auto:
        auto_fp = os.path.join(os.path.dirname(fp), "autocorrelation.png")
        fig_auto.savefig(auto_fp)
        print(f"\nAutocorrelation saved to: {auto_fp}")
    else:
        plt.show()


if __name__ == "__main__":
    analyze_image()
