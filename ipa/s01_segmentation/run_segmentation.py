"""
run_segmentation.py
====================

This script expects segmentation_config.yaml to be present in the current
working directory.
"""
import logging
import os
from datetime import datetime
from glob import glob
from os.path import join, basename
from pathlib import Path

import numpy as np
import yaml
from csbdeep.utils import normalize
# Run on CPU
from numpy._typing import ArrayLike
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
from tifffile import imread, imwrite
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_patched_fun(cache_dir: str):
    from csbdeep.models.pretrained import get_model_details, get_file

    def get_model_folder(cls, key_or_alias):
        key, alias, m = get_model_details(cls, key_or_alias)
        target = str(Path("models") / cls.__name__ / key)
        path = Path(
            get_file(
                fname=key + ".zip",
                origin=m["url"],
                file_hash=m["hash"],
                cache_subdir=target,
                extract=True,
                cache_dir=cache_dir,
            )
        )
        assert path.exists() and path.parent.exists()
        return path.parent

    return get_model_folder


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


def list_files(input_dir: str) -> tuple[list[str], list[str]]:
    w1_files = glob(join(input_dir, "*_w1Conf640.stk"))
    w2_files = [w1.replace("w1Conf640", "w2Conf561") for w1 in w1_files]
    return w1_files, w2_files


def _normalize(img: ArrayLike) -> ArrayLike:
    mi, ma = np.quantile(img[np.argmax(np.mean(img, axis=(1, 2)))], [0.5, 1])
    return np.clip((img - mi) / (ma - mi), 0, 10)


def _merge_2d_to_3d(
        raw_data: ArrayLike, plane_segmentations: list[ArrayLike]
) -> ArrayLike:
    final_labeling = np.zeros_like(raw_data)
    final_labeling[0] = plane_segmentations[0]
    for i, current_plane in tqdm(
            enumerate(plane_segmentations[1:]), total=len(plane_segmentations) - 1
    ):
        previous_plane = final_labeling[i]
        for label_id in filter(None, np.unique(current_plane)):
            current_max_label = np.max(final_labeling)
            roi_in_current_plane = current_plane == label_id
            candidate_labels, intersection_counts = _get_candidate_labels(
                previous_plane, roi_in_current_plane
            )
            new_label_id = _select_label_id(
                candidate_labels=candidate_labels,
                counts=intersection_counts,
                previous_plane=previous_plane,
                roi_in_current_plane=roi_in_current_plane,
                current_max_label=current_max_label,
            )

            final_labeling[i + 1, current_plane == label_id] = new_label_id

    return final_labeling.astype(np.uint16)


def _select_label_id(
        candidate_labels, counts, previous_plane, roi_in_current_plane, current_max_label
):
    if len(candidate_labels) == 0:
        # Nothing in the previous plane
        return current_max_label + 1
    elif len(candidate_labels) == 1:
        # Only a single label in the previous plane which overlaps with this
        # ROI
        candidate_label = candidate_labels[0]
        roi_in_previous_plane = previous_plane == candidate_label
        iou = _iou(
            intersection=counts[0],
            roi_in_previous_plane=roi_in_previous_plane,
            roi_in_current_plane=roi_in_current_plane,
        )
        if iou > 0.5:
            return candidate_label
        else:
            return 0
    else:
        # Multiple labels in the previous plane which overlap with this ROI
        ious = []
        for count, candidate_label in zip(counts, candidate_labels):
            roi_in_previous_plane = previous_plane == candidate_label
            ious.append(
                _iou(
                    intersection=count,
                    roi_in_previous_plane=roi_in_previous_plane,
                    roi_in_current_plane=roi_in_current_plane,
                )
            )

        max = np.argmax(ious)
        return candidate_labels[max]


def _iou(intersection, roi_in_previous_plane, roi_in_current_plane):
    union = np.logical_or(roi_in_previous_plane, roi_in_current_plane)
    return intersection / np.sum(union)


def _get_candidate_labels(previous_plane, roi_in_current_plane):
    candidate_labels, counts = np.unique(
        previous_plane[roi_in_current_plane], return_counts=True
    )
    fg_labels = candidate_labels > 0
    candidate_labels = candidate_labels[fg_labels]
    counts = counts[fg_labels]
    return candidate_labels, counts


def remove_yxborder_cells(labeling: ArrayLike) -> ArrayLike:
    """
    Remove cells which touch the border in the xy plane.

    Parameters
    ----------
    labeling
        label image of shape (z, y, x)

    Returns
    -------
    cleared_labeling
        label image of shape (z, y, x)
    """
    mask = np.zeros_like(labeling)
    mask[:, 1:-1, 1:-1] = 1
    cleared_labeling = clear_border(labeling, mask=mask.astype(bool))

    return cleared_labeling


def segment_cells(
        w1_file: str, w2_file: str, model_name: str, output_dir: str, logger: logging.Logger
):
    logger.info(f"Run segmentation on: " f"{w1_file.replace('_w1Conf640.stk', '')}")

    from stardist.models import StarDist2D

    model = StarDist2D.from_pretrained(model_name)

    w1 = imread(w1_file)
    w2 = imread(w2_file)

    raw_data = normalize(((w1 + w2) // 2), axis=(1, 2))

    plane_segmentations = []
    for plane in raw_data:
        labels, _ = model.predict_instances(plane, scale=0.33, prob_thresh=0.6)
        labels = remove_small_objects(labels, min_size=2000)
        plane_segmentations.append(labels)

    final_labeling = _merge_2d_to_3d(
        raw_data=raw_data, plane_segmentations=plane_segmentations
    )

    # remove border touching cells in xy
    final_labeling = remove_yxborder_cells(final_labeling)

    file_name = join(
        output_dir, basename(w1_file).replace("_w1Conf640.stk", "-CELL_SEG.tif")
    )
    imwrite(file_name, data=final_labeling, compression="zstd")
    logger.info(f"Saved cell segmentation to {file_name}.")


def main(
        input_dir: str, output_dir: str, model_name: str, model_cache_dir: str
) -> None:
    """
    Perform segmentation on all data in the input directory.

    Parameters
    ----------
    input_dir
        Directory containing all input data.
    output_dir
        Directory where the preprocessed data is saved.
    model_name
        Name of the StarDist model.
    model_cache_dir
        Directory where the StarDist models are downloaded to.

    """
    from csbdeep.models import pretrained

    pretrained.get_model_folder = get_patched_fun(cache_dir=model_cache_dir)
    logger = _create_logger(name="segmentation")

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"StarDist model: {model_name}")

    w1_files, w2_files = list_files(input_dir=input_dir)

    for w1_file, w2_file in zip(w1_files, w2_files):
        segment_cells(w1_file, w2_file, model_name, output_dir, logger=logger)

    logger.info("Done!")


if __name__ == "__main__":
    from build_segmentation_config import CONFIG_NAME

    with open(CONFIG_NAME, "r") as f:
        config = yaml.safe_load(f)

    main(**config)
