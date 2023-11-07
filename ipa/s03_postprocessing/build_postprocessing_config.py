"""
build_postprocessing_config.py
==============================

Ask for user input via command line interface (CLI) to build the
postprocessing_config.yaml. The postprocessing_config.yaml is consumed
by the run_postprocessing.py script.
"""


import os
from os.path import join, dirname

import questionary
import yaml

CONFIG_NAME = "postprocessing_config.yaml"


def main() -> None:
    """Create spot_assignment_config.yaml from user inputs."""
    cwd = os.getcwd()

    input_dir = questionary.path("Path to input directory:").ask()
    output_dir = questionary.path("Path to output directory:").ask()
    segmentation_dir = questionary.path(
        "Path to segmentation directory:", default=join(output_dir, "s01_segmentation")
    ).ask()
    spot_dir = questionary.path(
        "Path to spot directory:", default=join(output_dir, "s02_spot-detection")
    ).ask()
    cutoff_dist = int(
        questionary.text(
            "cutoff distance for coloc.:", default="3", validate=lambda v: v.isdigit()
        ).ask()
    )
    output_dir = join(output_dir, "s03_postprocessing")

    config = {
        "input_dir": os.path.relpath(input_dir, cwd),
        "segmentation_dir": os.path.relpath(segmentation_dir, cwd),
        "spot_dir": os.path.relpath(spot_dir, cwd),
        "cutoff_dist": cutoff_dist,
        "output_dir": os.path.relpath(output_dir, cwd),
    }

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(cwd, CONFIG_NAME), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, indent=4)

    _path_to_file = os.path.relpath(dirname(__file__), cwd)
    next_cmd = f"python {_path_to_file}/run_postprocessing.py"
    print("Run the post-processing with: ")
    print(next_cmd)


if __name__ == "__main__":
    main()
