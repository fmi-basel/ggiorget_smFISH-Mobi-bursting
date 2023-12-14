# ruff: noqa: E402
"""
Run_postprocessing.py
=========================

This script expects postprocessing_config.yaml to be present in the
current working directory.
"""
import logging
import sys
from datetime import datetime
from glob import glob
from os.path import basename, join, exists, dirname

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage import measure
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
        spot_file = join(input_dir, name.replace(".stk", ".csv"))
        assert exists(spot_file), (
            f"Spot file is missing " f"for {name}."
        )
        spot_files.append(spot_file)

    return spot_files


def readout_dapi(img_path: str, cell_seg_path: str) -> pd.DataFrame:
    """
    Read-out DAPI intensity

    Parameters
    ----------
    img_path :
        Directory containing the DAPI files

    cell_seg_path :
        Directory containing the corresponding segmentation files

    Returns
    -------
    dataframe with DAPI readout
    """

    w3 = imread(img_path)
    segmentation = imread(cell_seg_path)

    def sd(regionmask, intensity):
        return np.std(intensity[regionmask])

    def sumup(regionmask, intensity):
        return np.sum(intensity[regionmask])

    df = pd.DataFrame(measure.regionprops_table(segmentation, w3, properties=('label', 'area', 'intensity_mean',),
                                                extra_properties=(sd, sumup,)))
    df.rename(columns={'area': 'cellarea', 'intensity_mean': 'dapi_mean', 'sd': 'dapi_sd',
                       'sumup': 'dapi_integratedintensity'},
              inplace=True)

    return df


def readout_background_intensity(img_path: str, cell_seg_path: str) -> pd.DataFrame:
    """
    Read-out cell background intensity

    Parameters
    ----------
    img_path :
        Directory containing the intensity images

    cell_seg_path :
        Directory containing the corresponding segmentation files

    Returns
    -------
    dataframe with intensity readout
    """

    image = imread(img_path)
    segmentation = imread(cell_seg_path)

    def median(regionmask, intensity):
        return np.median(intensity[regionmask])

    df = pd.DataFrame(measure.regionprops_table(segmentation, image, properties=('label',),
                                                extra_properties=(median,)))
    df.rename(columns={'median': 'background_median'}, inplace=True)

    return df


def assign_spots(spot_coord_path: str, cell_seg_path: str) -> pd.DataFrame:
    """
    Load spot file files with matching cell segmentation and assign spots to cell labels.

    Parameters
    ----------
    spot_coord_path :
        Directory containing the spot files

    cell_seg_path :
        Directory containing the corresponding segmentation files

    Returns
    -------
    dataframe with spot coordinates and corresponding cell labels
    """
    spot_coord = pd.read_csv(spot_coord_path)
    segmentation = imread(cell_seg_path)

    # assign spots to labels
    spot_coord['label'] = segmentation[
        spot_coord['axis-0'].astype(np.int64),
        spot_coord['axis-1'].astype(np.int64),
        spot_coord['axis-2'].astype(np.int64)
    ]

    # remove spots that are not assigned to a label
    spot_coord = spot_coord[spot_coord['label'] != 0].reset_index(drop=True)

    spot_coord.rename(columns={'axis-0': 'z', 'axis-1': 'y', 'axis-2': 'x'}, inplace=True)

    return spot_coord


def create_3d_ellipsoid_mask(radius_xy: int, radius_z: int) -> np.array:
    """
    Create a 3d ellipsoid mask with given radi (no empty slices)

    Parameters
    ----------
    radius_xy :
        radius in xy plane
    radius_z :
        radius in z plane

    Returns
    -------
    3d ellipsoid mask
    """
    from skimage.draw import ellipsoid

    # create ellipsoid
    ellipse = ellipsoid(radius_z, radius_xy, radius_xy)
    # remove empty slices
    mask_z = np.any(ellipse, axis=(1, 2))
    mask_y = np.any(ellipse, axis=(0, 2))
    mask_x = np.any(ellipse, axis=(0, 1))
    ellipse = ellipse[mask_z, :, :]
    ellipse = ellipse[:, mask_y, :]
    ellipse = ellipse[:, :, mask_x]

    return ellipse


def intensity_readout_spot(image_path: str, spot_coord: pd.DataFrame, yx_radius: int, z_radius: int) -> pd.DataFrame:
    """
    Readout intensity of spots in a given image

    Parameters
    ----------
    image_path :
        Path to the raw image from which the intensity is read out
    spot_coord :
        Dataframe with spot coordinates
    yx_radius :
        radius in xy plane for disk shaped mask (spot radius)
    z_radius :
        radius in z plane for disk shaped mask (spot radius)

    Returns
    -------
    Dataframe containing spot coordinates and spot intensity
    """
    # read in image and pad to avoid boundary effects
    image = imread(image_path)
    image = np.pad(image, ((z_radius, z_radius), (yx_radius, yx_radius), (yx_radius, yx_radius)), 'edge')
    # create spot mask
    ellipse_mask = create_3d_ellipsoid_mask(yx_radius, z_radius)

    def sd(regionmask, intensity):
        return np.std(intensity[regionmask])

    def sumup(regionmask, intensity):
        return np.sum(intensity[regionmask])

    def median(regionmask, intensity):
        return np.median(intensity[regionmask])

    # readout intensity spot by spot using mask
    int_spots = []
    y_coords = spot_coord['y'].values + yx_radius
    x_coords = spot_coord['x'].values + yx_radius
    z_coords = spot_coord['z'].values + z_radius

    for z, y, x in zip(z_coords, y_coords, x_coords):
        image_cut = image[z - z_radius:z + z_radius + 1, y - yx_radius:y + yx_radius + 1,
                    x - yx_radius:x + yx_radius + 1]
        df = pd.DataFrame(measure.regionprops_table(ellipse_mask.astype(np.uint8), image_cut,
                                                    properties=('intensity_mean', 'intensity_max',),
                                                    extra_properties=(sd, sumup, median,)))
        int_spots.append(df)

    int_spots = pd.concat(int_spots).reset_index(drop=True)
    int_spots.rename(
        columns={'intensity_mean': 'spot_mean', 'intensity_max': 'spot_max', 'sd': 'spot_sd',
                 'sumup': 'spot_integratedintensity', 'median': 'spot_median'},
        inplace=True)
    spot_coord = pd.concat([spot_coord, int_spots], axis=1)

    return spot_coord


def linear_sum_colocalisation(spot_coord_1: pd.DataFrame, spot_coord_2: pd.DataFrame, cutoff_dist: int) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare two dataframes with spot coordinates and indicate if spot is present in both dataframes with a certain
    cutoff distance.

    Parameters
    ----------
    spot_coord_1 :
        Dataframe with first set of spot coordinates

    spot_coord_2 :
        Dataframe with second set of spot coordinates

    cutoff_dist :
        cutoff distance in pixels (sum over all axis)

    Returns
    -------
    input dataframes with additional column indicating co-localisation and one additional dataframe with matched spots
    """
    spot_coord_1_values = spot_coord_1[['z', 'y', 'x']].dropna().values
    spot_coord_2_values = spot_coord_2[['z', 'y', 'x']].dropna().values

    # calculate the distance matrix and find the optimal assignment
    global_distances = cdist(spot_coord_1_values, spot_coord_2_values, 'euclidean')
    spot_coord_1_ind, spot_coord_2_ind = linear_sum_assignment(global_distances)

    # use threshold on distance
    assigned_distance = spot_coord_1_values[spot_coord_1_ind] - spot_coord_2_values[spot_coord_2_ind]
    distance_id = np.sum(np.abs(assigned_distance), axis=1)

    spot_coord_1_ind = spot_coord_1_ind[distance_id < cutoff_dist]
    spot_coord_2_ind = spot_coord_2_ind[distance_id < cutoff_dist]

    # create a dataframe only containing the matches spots, where one row corresponds to a match
    match_index_df = pd.DataFrame({'index_w1': spot_coord_1_ind, 'index_w2': spot_coord_2_ind})
    match_spot_coord_1 = spot_coord_1.add_suffix('_w1')
    match_spot_coord_2 = spot_coord_2.add_suffix('_w2')

    match_df = pd.merge(match_index_df, match_spot_coord_1, left_on='index_w1', right_index=True, how='left')
    match_df = pd.merge(match_df, match_spot_coord_2, left_on='index_w2', right_index=True, how='left')
    match_df.drop(columns=['index_w1', 'index_w2', 'label_w2'], inplace=True)
    match_df.rename(columns={'label_w1': 'label'}, inplace=True)

    # add column to indicate co-localisation
    spot_coord_1['coloc'] = spot_coord_1.index.isin(spot_coord_1_ind)
    spot_coord_2['coloc'] = spot_coord_2.index.isin(spot_coord_2_ind)

    return spot_coord_1, spot_coord_2, match_df


def main(
        input_dir: str,
        segmentation_dir: str,
        spot_dir: str,
        cutoff_dist: int,
        spot_radius: tuple[int, int],
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
    spot_dir
        Directory containing spot information
    cutoff_dist
        cutoff distance for co-localisation
    spot_radius
        radius of spot mask [Z, YX]
    output_dir
        Directory where the post-processed data is saved.
    """
    logger = _create_logger(name="postprocessing")

    logger.info(f"Raw images directory: {input_dir}")
    logger.info(f"Segmentation directory: {segmentation_dir}")
    logger.info(f"Spot directory: {spot_dir}")
    logger.info(f"Output directory: {output_dir}")

    w1_files, w2_files, w3_files = list_files(input_dir=input_dir)
    cell_seg_files = list_cell_seg_files(input_dir=segmentation_dir, raw_files=w1_files)
    w1_spot_files = list_spot_files(input_dir=spot_dir, raw_files=w1_files)
    w2_spot_files = list_spot_files(input_dir=spot_dir, raw_files=w2_files)

    for w1_file, w2_file, w3_file, cell_seg_file, w1_spot_file, w2_spot_file in tqdm(zip(w1_files, w2_files, w3_files,
                                                                                         cell_seg_files, w1_spot_files,
                                                                                         w2_spot_files)):
        logger.info(f"Processing {basename(w3_file)}.")

        # assign spots to cells, remove spots that are not assigned
        df_spots_w1 = assign_spots(
            spot_coord_path=w1_spot_file,
            cell_seg_path=cell_seg_file,
        )

        df_spots_w2 = assign_spots(
            spot_coord_path=w2_spot_file,
            cell_seg_path=cell_seg_file,
        )

        # readout spot intensity
        df_spots_w1 = intensity_readout_spot(
            image_path=w1_file,
            spot_coord=df_spots_w1,
            yx_radius=spot_radius[1],
            z_radius=spot_radius[0]
        )

        df_spots_w2 = intensity_readout_spot(
            image_path=w2_file,
            spot_coord=df_spots_w2,
            yx_radius=spot_radius[1],
            z_radius=spot_radius[0]
        )

        # read-out background intensity
        df_background_w1 = readout_background_intensity(
            img_path=w1_file,
            cell_seg_path=cell_seg_file,
        )

        df_background_w2 = readout_background_intensity(
            img_path=w2_file,
            cell_seg_path=cell_seg_file,
        )

        df_spots_w1 = pd.merge(df_spots_w1, df_background_w1, on='label', how='outer')
        df_spots_w2 = pd.merge(df_spots_w2, df_background_w2, on='label', how='outer')

        # check for presence of spot in both channels
        df_spots_w1, df_spots_w2, df_matched_spots = linear_sum_colocalisation(
            spot_coord_1=df_spots_w1,
            spot_coord_2=df_spots_w2,
            cutoff_dist=cutoff_dist,
        )

        # readout DAPI intensity
        df_dapi = readout_dapi(
            img_path=w3_file,
            cell_seg_path=cell_seg_file,
        )

        df_spots_w1 = pd.merge(df_spots_w1, df_dapi, on='label', how='outer')
        df_spots_w2 = pd.merge(df_spots_w2, df_dapi, on='label', how='outer')
        df_matched_spots = pd.merge(df_matched_spots, df_dapi, on='label', how='outer')

        # save spot information
        df_spots_w1.to_csv(
            join(output_dir, basename(w1_spot_file).replace(".csv", "_postprocessed.csv")),
            index_label=True,
        )
        df_spots_w2.to_csv(
            join(output_dir, basename(w2_spot_file).replace(".csv", "_postprocessed.csv")),
            index_label=True,
        )
        df_matched_spots.to_csv(
            join(output_dir, basename(w1_spot_file).replace("_w1Conf640.csv", "_coloc-spots.csv")),
            index_label=True,
        )

    logger.info("Done!")


if __name__ == "__main__":
    from build_postprocessing_config import CONFIG_NAME

    with open(CONFIG_NAME, "r") as f:
        config = yaml.safe_load(f)

    main(**config)
