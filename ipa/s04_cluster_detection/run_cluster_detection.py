# ruff: noqa: E402
"""
Run_cluster_detection.py
=========================

This script expects cluster_detection_config.yaml to be present in the
current working directory.
"""
import logging
import sys
from datetime import datetime
from glob import glob
from os.path import basename, join, exists, dirname

import bigfish.detection as detection
import numpy as np
import pandas as pd
import yaml
from tifffile import imread
from tqdm import tqdm

sys.path.insert(1, join(dirname(__file__), "..", ".."))


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


def list_files(input_dir: str) -> tuple[list[str], list[str], list[str]]:
    w1_files = glob(join(input_dir, "*_w1Conf640.stk"))
    w2_files = [w1.replace("w1Conf640", "w2Conf561") for w1 in w1_files]
    w3_files = [w1.replace("w1Conf640", "w3Conf405") for w1 in w1_files]
    return w1_files, w2_files, w3_files


def list_cell_seg_files(input_dir: str, raw_files: list[str]) -> list[str]:
    """
    Load segmentation files, which match the raw files.

    Parameters
    ----------
    input_dir :
        Directory containing the segmentation files.
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


def list_spot_files(input_dir: str, raw_files: list[str]) -> list[str]:
    """
    Load spot files, which match the raw files.

    Parameters
    ----------
    input_dir :
        Directory containing the spot files.

    raw_files :
        List of raw files

    Returns
    -------
    spot files
    """
    spot_files = []
    for file in raw_files:
        name = basename(file)
        spot_file = join(input_dir, name.replace(".stk", "_postprocessed.csv"))
        assert exists(spot_file), (
            f"Spot file is missing " f"for {name}."
        )
        spot_files.append(spot_file)

    return spot_files


def abbe_limit_xy(wavelength: int, NA: float):
    """
    Calculates the Abbe limit in xy

    Parameters
    ----------
    wavelength :
        Wavelength in nm
    NA :
        Numerical aperture

    """
    return wavelength / (2 * NA)


def abbe_limit_z(wavelength: int, NA: float):
    """
    Calculates the Abbe limit in z

    Parameters
    ----------
    wavelength :
        Wavelength in nm
    NA :
        Numerical aperture

    """
    return 2 * wavelength / (NA) ** 2


def cluster_detection(image_path: str, segmentation_path: str, spot_coord: pd.DataFrame, wavelength: int, NA: float,
                      spacing: tuple[float, float, float]) -> pd.DataFrame:
    """
    Using the BigFish package, detect cluster on a single image and assigns them to cells from a mask

    Parameters
    ----------
    image_path :
        Path to raw image
    segmentation_path :
        Path to segmentation mask
    spot_coord :
        Pandas dataframe containing spot coordinates
    wavelength :
        Wavelength used for cluster detection
    NA :
        Numerical aperture
    spacing :
        Spacing in nm [Z, Y, X]

    Returns
    -------
    df_cluster :
        Pandas dataframe containing cluster information
    """

    # read-in image and spot coordinates
    image = imread(image_path)
    segmentation = imread(segmentation_path)
    spots_coord = spot_coord[['z', 'y', 'x']].dropna().to_numpy()

    # define psf/spot size based on Abbe limit
    sample_xy_nm = abbe_limit_xy(wavelength, NA)
    sample_z_nm = abbe_limit_z(wavelength, NA)
    spot_radius_nm = (0.5 * sample_z_nm, 0.5 * sample_xy_nm, 0.5 * sample_xy_nm)

    # run spot decomposition, the function simulates as many median intensity spots as necessary to re-create the
    # original region intensity
    spots_post_decomposition, dense_regions, _ = detection.decompose_dense(
        image=image,
        spots=spots_coord,
        voxel_size=spacing,
        spot_radius=spot_radius_nm,  # radius in nm comes from reighleight criterion
        alpha=0.3,  # relates to the intensity of the average spot. 0.5 means median intensity
    )

    # run cluster detection
    spots_post_clustering, clusters = detection.detect_clusters(
        spots=spots_post_decomposition,
        voxel_size=spacing,
        radius=int(spacing[0]),
        nb_min_spots=5)

    # create dataframe with cluster information
    df_cluster = pd.DataFrame(clusters, columns=['z', 'y', 'x', 'nb_spots', 'index'])
    df_cluster.drop(columns=['index'], inplace=True)

    # assign cluster to labels
    df_cluster['label'] = segmentation[
        df_cluster['z'].astype(np.int64),
        df_cluster['y'].astype(np.int64),
        df_cluster['x'].astype(np.int64)
    ]

    # remove cluster that are not assigned to a label
    df_cluster = df_cluster[df_cluster['label'] != 0].reset_index(drop=True)

    return df_cluster


def main(
        input_dir: str,
        segmentation_dir: str,
        processedspot_dir: str,
        spacing: tuple[float, float, float],
        NA: float,
        wavelength_cluster: int,
        output_dir: str,
) -> None:
    """
    Apply postprocessing step to data in input_dir.

    Parameters
    ----------
    input_dir
    Directory containing raw images.
    segmentation_dir
        Directory containing segmentation masks.
    processedspot_dir
        Directory containing spot information
    spacing
        Spacing in nm [Z, Y, X]
    NA
        Numerical aperture
    wavelength_cluster
        Wavelength used for cluster detection
    output_dir
        Directory where the post-processed data is saved.
    """
    logger = _create_logger(name="clusterdetection")

    logger.info(f"Raw images directory: {input_dir}")
    logger.info(f"Segmentation directory: {segmentation_dir}")
    logger.info(f"Processed Spot directory: {processedspot_dir}")
    logger.info(f"Output directory: {output_dir}")

    w1_files, w2_files, w3_files = list_files(input_dir=input_dir)
    cell_seg_files = list_cell_seg_files(input_dir=segmentation_dir, raw_files=w1_files)
    w1_spot_files = list_spot_files(input_dir=processedspot_dir, raw_files=w1_files)

    for w1_file, cell_seg_file, w1_spot_file in tqdm(zip(w1_files, cell_seg_files, w1_spot_files)):
        logger.info(f"Processing {basename(w1_file)}.")

        # load spot coordinates to I can check if there are enough spots for cluster detection (cutoff is 30 as described in the paper
        df_spots_w1 = pd.read_csv(w1_spot_file)

        # perform cluster detection
        if df_spots_w1.dropna().shape[0] > 30:
            # run cluster detection
            df_cluster = cluster_detection(
                image_path=w1_file,
                segmentation_path=cell_seg_file,
                spot_coord=df_spots_w1,
                wavelength=wavelength_cluster,
                NA=NA,
                spacing=spacing,
            )

            # save cluster information
            df_cluster.to_csv(
                join(output_dir, basename(w1_spot_file).replace("_w1Conf640.csv", "_clusters.csv")),
                index_label=True,
            )
        else:
            logger.info(
                f"Number of spots in channels 1 is too small for automated cluster detection in {basename(w1_file)}.")

    logger.info("Done!")


if __name__ == "__main__":
    from build_cluster_detection_config import CONFIG_NAME

    with open(CONFIG_NAME, "r") as f:
        config = yaml.safe_load(f)

    main(**config)
