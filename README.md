# Supplementary code for "Identifying regions of concomittant compound precipitation and wind speed extremes over Europe"

This repository contains implementation of clustering methods applied to the ERA5 dataset for compound precipitation and wind speed extremes over Europe.

## File descriptions:

* `analysis_hr.py`, The provided code is a Python script that contains two functions, "secomat" and "chimat", for calculating different types of dissimilarity matrices based on precipitation and wind speed thresholds using ERA5 climate data.
* `clust_hr.py`, This Python code defines functions to perform clustering analysis, and executes the compound clustering analysis algorithm (CAICE) or (HC) on input matrices. 
* `eco_alg.py`: Two functions used to perform (CAICE) clustering algorithm on a 2D array Theta, with the option of specifying the threshold alpha.
* `hc_alg.py`: A function used to perform (HC) clustering algorithm on a 2D array Theta, with the option of specifying the number of clusters.

## Reference
