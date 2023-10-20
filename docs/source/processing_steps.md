# Processing Steps
```{include} ../../runs/example/README.md
:start-after: <!-- start processing steps summary -->
:end-before: <!-- end processing steps summary -->
```

## s01_segmentation
We use [StarDist](https://github.com/stardist/stardist) to segment the cells.

## s02_spot_detection
We use a combination of Laplacian of Gaussian (LoG) and h-maxima for spot detection. The LoG selects all spots which are of the right size and h-maxima filters the regions which have the right intensity. The detected spots must be detected with both of these methods.
