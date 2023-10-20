"""
build_segmentation_config.py
============================

Ask for user input via command line interface (CLI) to build the
segmentation_config.yaml. The segmentation_config.yaml is consumed
by the run_segmentation.py script.
"""


import os
from os.path import join, dirname

import questionary
import yaml

CONFIG_NAME = "segmentation_config.yaml"


def main() -> None:
    """
    Create segmentation_config.yaml from user inputs.
    """
    cwd = os.getcwd()

    input_dir = questionary.path("Path to input directory:").ask()
    output_dir = questionary.path("Path to output directory:").ask()
    model_name = questionary.text(
        "StarDist model name:", default="2D_versatile_fluo"
    ).ask()
    model_cache_dir = questionary.path(
        "Path to model cache:",
        default="/tungstenfs/scratch/gmicro_prefect/ggiorget/tunnjana/",
    ).ask()

    output_dir = join(output_dir, "s01_segmentation")

    config = {
        "input_dir": os.path.relpath(input_dir, cwd),
        "output_dir": os.path.relpath(output_dir, cwd),
        "model_name": model_name,
        "model_cache_dir": os.path.relpath(model_cache_dir, cwd),
    }

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(cwd, CONFIG_NAME), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, indent=4)

    _path_to_file = os.path.relpath(dirname(__file__), cwd)
    next_cmd = f"python {_path_to_file}/run_segmentation.py"
    print("Run the segmentation with: ")
    print(next_cmd)


if __name__ == "__main__":
    main()
