"""
build_postprocessing_config.py
==============================

Ask for user input via command line interface (CLI) to build the
cluster_detection_config.yaml. The cluster_detection_config.yaml is consumed
by the run_cluster_detection.py script.
"""

import os
from os.path import join, dirname

import questionary
import yaml

CONFIG_NAME = "cluster_detection_config.yaml"


def main() -> None:
    """Create cluster_detection_config.yaml from user inputs."""
    cwd = os.getcwd()

    input_dir = questionary.path("Path to input directory:").ask()
    output_dir = questionary.path("Path to output directory:").ask()
    segmentation_dir = questionary.path(
        "Path to segmentation directory:", default=join(output_dir, "s01_segmentation")
    ).ask()
    processedspot_dir = questionary.path(
        "Path to postprocessed spot directory:", default=join(output_dir, "s03_postprocessing")
    ).ask()
    spacing = tuple(
        int(s)
        for s in questionary.text(
            "Spacing in nm [Z, Y, X]:",
            default="300, 65, 65",
            validate=lambda v: v.replace(",", "")
            .replace(".", "")
            .replace(" ", "")
            .isdigit(),
        )
        .ask()
        .split(",")
    )
    NA = float(
        questionary.text(
            "NA:", default="1.4", validate=lambda v: v.replace(".", "").isdigit()
        ).ask()
    )
    wavelength_cluster = int(
        questionary.text(
            "Wavelength Channel 1 (for clustering):", default="640", validate=lambda v: v.isdigit()
        ).ask()
    )
    output_dir = join(output_dir, "s04_cluster")

    config = {
        "input_dir": os.path.relpath(input_dir, cwd),
        "segmentation_dir": os.path.relpath(segmentation_dir, cwd),
        "processedspot_dir": os.path.relpath(processedspot_dir, cwd),
        "spacing": spacing,
        "NA": NA,
        "wavelength_cluster": wavelength_cluster,
        "output_dir": os.path.relpath(output_dir, cwd),
    }

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(cwd, CONFIG_NAME), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, indent=4)

    _path_to_file = os.path.relpath(dirname(__file__), cwd)
    next_cmd = f"python {_path_to_file}/run_cluster_detection.py"
    print("Run the cluster detection with: ")
    print(next_cmd)


if __name__ == "__main__":
    main()
