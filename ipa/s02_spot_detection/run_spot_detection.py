# ruff: noqa: E402
"""
Run_spot_detection.py
=========================

This script expects spot_detection_config.yaml to be present in the
current working directory.
"""
import sys
from os.path import basename, join, exists, dirname

sys.path.insert(1, join(dirname(__file__), "..", ".."))

from ipa.s01_segmentation.run_segmentation import list_files
from tifffile import imread
from skimage.morphology import h_maxima, local_maxima
from skimage.measure import regionprops
from scipy.ndimage import gaussian_laplace
import logging
from datetime import datetime
import yaml
import pandas as pd
import numpy as np

from skimage.util import img_as_float32
from tqdm import tqdm


def _create_logger(name: str) -> logging.Logger:
    """
    Create logger which logs to <timestamp>-<name>.log inside the current
    working directory.

    Parameters
    ----------
    name
        Name of the logger instance.
    """
    logger = logging.Logger(name.capitalize())
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    handler = logging.FileHandler(f"{now}-{name}.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_yx_sigma(wavelength: float, NA: float, yx_spacing: float):
    """
    Sigma which produces a Gaussian with the same full width
    half maximum as the theoretical Rayleigh Criterion.

    R = 0.61 * lambda / NA

    Parameters
    ----------
    wavelength :
        Emission wavelength
    NA :
        Numerical Aperture
    z_spacing :
        Spacing of z planes

    Returns
    -------
    theoretical sigma
    """
    return 0.61 * wavelength / NA / (2 * np.sqrt(2 * np.log(2))) / yx_spacing


def get_z_sigma(wavelength: float, NA: float, z_spacing: float):
    """
    Sigma which produces a Gaussian with the same full width
    half maximum as the theoretical resolution limit in Z described by E. Abbe.

    d = 2 * lambda / (NA^2)

    Parameters
    ----------
    wavelength :
        Emission wavelength
    NA :
        Numerical Aperture
    z_spacing :
        Spacing of z planes

    Returns
    -------
    theoretical sigma
    """
    return 2 * wavelength / (NA**2) / (2 * np.sqrt(2 * np.log(2))) / z_spacing


def _bbox_to_slices(bbox, padding, shape):
    z_slice = slice(max(0, bbox[0] - padding[0]), min(shape[0], bbox[3] + padding[0]))
    y_slice = slice(max(0, bbox[1] - padding[1]), min(shape[1], bbox[4] + padding[1]))
    x_slice = slice(max(0, bbox[2] - padding[2]), min(shape[2], bbox[5] + padding[2]))
    return z_slice, y_slice, x_slice


def detect_spots(
    img_path: str,
    cell_seg_path: str,
    wavelength: int,
    NA: float,
    spacing: tuple[float, float, float],
    intensity_threshold: int,
) -> pd.DataFrame:
    """
    Detect bright, diffraction limited spots.

    The Laplacian of Gaussian is used to detect diffraction limited spots,
    where the spot size is computed from the emission `wavelength`, `NA` and
    pixel `spacing`. This results in an overdetection of spots and only the
    ones with an intensity larger than `intensity_threshold` relative to
    their immediate neighborhood are kept.

    Parameters
    ----------
    img_path :
        Path to the raw image data.
    cell_seg_path :
        Path to the ROI segmentation image.
    wavelength :
        Emission wavelength of the spot signal.
    NA :
        Numerical aperture.
    spacing :
        Z, Y, X spacing of the data.
    intensity_threshold :
        Minimum spot intensity relative to the immediate background.

    Returns
    -------
    pd.DataFrame with detected spots
    """
    image = imread(img_path)
    segmentation = imread(cell_seg_path).astype(np.uint16)
    yx_sigma = get_yx_sigma(wavelength, NA, spacing[1])
    z_sigma = get_z_sigma(wavelength, NA, spacing[0])
    spots = []
    for roi in tqdm(regionprops(segmentation)):
        roi_img, x_slice, y_slice, z_slice = _crop_padded_roi_img(
            image=image,
            roi=roi,
            segmentation=segmentation,
            yx_sigma=yx_sigma,
            z_sigma=z_sigma,
        )

        log_detections = get_log_detections(roi_img, yx_sigma, z_sigma)
        h_detections = h_maxima(
            roi_img, h=intensity_threshold, footprint=np.ones((1, 1, 1))
        )

        detections = np.array(np.where(np.logical_and(log_detections, h_detections))).T
        detections += np.array([z_slice.start, y_slice.start, x_slice.start])

        spots.extend(detections.tolist())

    return pd.DataFrame(spots, columns=["axis-0", "axis-1", "axis-2"])


def get_log_detections(roi_img, yx_sigma, z_sigma):
    log_img = -gaussian_laplace(
        input=img_as_float32(roi_img), sigma=(z_sigma, yx_sigma, yx_sigma)
    )
    log_detections = local_maxima(log_img, connectivity=0)
    return log_detections


def _crop_padded_roi_img(image, roi, segmentation, yx_sigma, z_sigma):
    z_slice, y_slice, x_slice = _bbox_to_slices(
        roi.bbox,
        padding=(
            min(1, int(3 * z_sigma)),
            min(1, int(3 * yx_sigma)),
            min(1, int(3 * yx_sigma)),
        ),
        shape=segmentation.shape,
    )
    roi_img = image[z_slice, y_slice, x_slice]
    return roi_img, x_slice, y_slice, z_slice


def list_cell_seg_files(input_dir: str, raw_files: list[str]) -> list[str]:
    """
    Load segmentation files, which match the raw files.

    Parameters
    ----------
    input_dir :
        Direcotry containing the segmentation files.
    raw_files :
        List of raw files

    Returns
    -------
    segmentation files
    """
    cell_seg_files = []
    for file in raw_files:
        name = basename(file)
        cell_seg_file = join(input_dir, name.replace("_w1Conf640.stk", "-CELL_SEG.tif"))
        assert exists(cell_seg_file), (
            f"Cell segmentation file is missing " f"for {name}."
        )
        cell_seg_files.append(cell_seg_file)

    return cell_seg_files


def main(
    input_dir: str,
    segmentation_dir: str,
    output_dir: str,
    wavelength_1: int,
    wavelength_2: int,
    NA: float,
    spacing: tuple[float, float, float],
    h_1: int,
    h_2: int,
) -> None:
    """
    Apply feature_extraction step to data in input_dir.

    Parameters
    ----------
    input_dir
        Directory containing raw input data.
    segmentation_dir
        Directory containing segmentation masks.
    output_dir
        Directory where the preprocessed data is saved.
    wavelength_1
        Emission wavelength of channel 1 (Ch1).
    wavelength_2
        Emission wavelength of channel 2 (Ch2).
    NA
        Numerical aperture
    spacing
        Z, Y, X spacing of the data.
    h_1
        Minimum spot intensity relative to the immediate background in Ch1.
    h_2
        Minimum spot intensity relative to the immediate background in Ch2.
    """
    logger = _create_logger(name="spot_detection")

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Segmentation directory: {segmentation_dir}")
    logger.info(f"Output directory: {output_dir}")

    w1_files, w2_files = list_files(input_dir=input_dir)
    cell_seg_files = list_cell_seg_files(input_dir=segmentation_dir, raw_files=w1_files)

    for w1_file, w2_file, cell_seg_file in zip(w1_files, w2_files, cell_seg_files):
        logger.info(f"Processing {basename(w1_file)}.")
        w1_spots = detect_spots(
            img_path=w1_file,
            cell_seg_path=cell_seg_file,
            wavelength=wavelength_1,
            NA=NA,
            spacing=spacing,
            intensity_threshold=h_1,
        )
        (
            w1_spots.to_csv(
                join(output_dir, basename(w1_file).replace(".stk", ".csv")),
                index_label=True,
            )
        )
        logger.info(f"Found {len(w1_spots)} in Channel 1.")

        logger.info(f"Processing {basename(w1_file)}.")
        w2_spots = detect_spots(
            img_path=w2_file,
            cell_seg_path=cell_seg_file,
            wavelength=wavelength_2,
            NA=NA,
            spacing=spacing,
            intensity_threshold=h_2,
        )
        w2_spots.to_csv(
            join(output_dir, basename(w2_file).replace(".stk", ".csv")),
            index_label=True,
        )
        logger.info(f"Found {len(w2_spots)} in Channel 2.")

    logger.info("Done!")


if __name__ == "__main__":
    from build_spot_detection_config import CONFIG_NAME

    with open(CONFIG_NAME, "r") as f:
        config = yaml.safe_load(f)

    main(**config)
