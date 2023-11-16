# Example
<!-- start processing steps summary -->
The image processing and analysis of this project has following steps:
1. segmentation.
2. spot detection
3. post-processing

The required scripts are organized in the `ipa` directory. Each step follows the pattern of building a config file and then running the respective script.
Each run should have its respective run-directory. Inside the run-directory the config files and log files are stored.
<!-- end processing steps summary -->

Before running the scripts you must [activate your environment](../../infrastructure/apps/README.md).

<!-- start instructions -->
## s01_segmentation
```{note}
The scripts write the config and log files into the current working directory. In this example it is assumed that you are in `runs/example`.
```
1. Build segmentation config:<br>
    `python ../../ipa/s01_segmentation/build_segmentation_config.py`
2. Run segmentation:<br>
    `python ../../ipa/s01_segmentation/run_segmentation.py`


## s02_spot_detection
1. Build spot detection config:<br>
    `python ../../ipa/s02_spot_detection/build_spot_detection_config.py`
2. Run spot detection:<br>
    `python ../../ipa/s02_spot_detection/run_spot_detection.py`
    
## s03_postprocessing
1. Build post-processing config:<br>
    `python ../../ipa/s03_postprocessing/build_postprocessing_config.py`
2. Run spot detection:<br>
    `python ../../ipa/s03_postprocessing/run_postprocessing.py`
<!-- end instructions -->
