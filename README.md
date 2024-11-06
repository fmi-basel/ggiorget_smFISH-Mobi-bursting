<!-- start abstract -->
# smRNA-FISH analysis from 'Enhancer control of transcriptional activity via modulation of burst frequency'

![](/docs/source/resources/FISH.jpg)

This repository contains all code used to analyse smRNA-FISH data from the project 'Enhancer control of transcriptional activity via modulation of burst frequency'. 

Central steps involve:
1. cell segmentation
2. spot detection
3. post-processing: assigning spots to cells, and check co-localisation
<!-- end abstract -->

## Repository Overview
* `docs`: Contains all project documentation.
* `infrastructure`: Contains detailed installation instructions for all required tools.
* `ipa`: Contains all image-processing-and-analysis (ipa) scripts which are used for this project to generate final results.
* `runs`: Contains all config files which were used as inputs to the scripts in `ipa`.
* `scratchpad`: Contains everything that is nice to keep track of, but which is not used for any final results.

## Setup
Detailed install instructions can be found in [infrastructure/README.md](infrastructure/README.md).

## Citation
Do not forget to cite our [publication]() if you use any of our provided materials.
