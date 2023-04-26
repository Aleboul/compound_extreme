""" 
The provided code is a Python script that contains two functions, "secomat" and "chimat", for calculating different types of co-occurrence matrices based on precipitation 
and wind speed thresholds using ERA5 climate data. The script starts by importing necessary Python packages, including NumPy, xarray, pandas, and geopandas. Then, it loads 
two ERA5 datasets for wind and total precipitation from netCDF files and assigns them to "era5_wind" and "era5_tp" variables, respectively. After that, it merges the two 
datasets into a single dataset called "era5" and deletes the two original datasets to free up memory. The first function, "secomat", takes the "era5" dataset and three 
optional parameters: "k", "d1" and "d2". The function calculates the seco matrix based on the given thresholds for precipitation and wind speed. The second function, "chimat"
is similar to "secomat" but calculates the extremal correlation matrix instead. It takes the "era5" dataset and four optional parameters: "k", "d1", "d2", "tp", and "wind".
It calculates the rank threshold, the threshold for either total precipitation or wind speed depending on the input parameters, and the number of co-occurrences. Finally, 
it calculates the extremal correlation matrix and returns it.
"""

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

era5_wind = xr.open_dataset('data/high_res/era5hr_wind.nc')
era5_tp = xr.open_dataset('data/high_res/era5hr_tp.nc')
era5 = era5_wind
era5['tp'] = era5_tp.tp
del era5_wind
del era5_tp


def secomat(era5, k=100, d1=91, d2=116, normalize=True):
    """
    Calculate the seco matrix based on precipitation and wind speed thresholds.

    Parameters:
    era5 (xarray.Dataset): The ERA5 dataset with variables "tp" (total precipitation) and "speed" (wind speed).
    k (int): The threshold value.
    d1 (int): The threshold value for the latitude dimension.
    d2 (int): The threshold value for the longitude dimension.
    normalize (bool): If True, normalize the secondary co-occurrence matrix.

    Returns:
    np.array: The seco matrix.
    """

    # Rank the era5 dataset by date
    erank5 = era5.rank('date')
    # Get the number of dates in the dataset
    n = len(erank5.date)
    # Set the rank threshold to be above n - k + 0.5
    erank_threshold = (erank5 > n - k + 0.5)

    # Get the threshold for total precipitation and wind speed
    indicator_tp = erank_threshold.tp[:, :d1, :d2]
    indicator_speed = erank_threshold.speed[:, :d1, :d2]
    # Combine the two thresholds
    indicator = indicator_tp + indicator_speed
    # Sum the thresholds along the date dimension
    indicator_sum = (indicator + indicator.rename(
        {'latitude': 'latitude1', 'longitude': 'longitude1'})).sum('date')
    # Reshape the sum into a 2D matrix
    value_2 = indicator_sum.values.reshape([d1*d2, d1*d2])
    # Clear the memory for the sum variable
    del indicator_sum
    # Set the number of co-occurence to the diagonal of the value 2 matrix
    ext_coeff = np.diag(value_2)
    # Set the number of co-occurence as the sum of the two indicators
    value_1 = ext_coeff[:, np.newaxis] + ext_coeff[np.newaxis, :]
    # Set the minimum extremal coefficient to be the outer minimum of the external coefficients divided by k
    min_ext_coeff = np.minimum.outer(ext_coeff, ext_coeff) / k
    # Calculate the secondary co-occurrence matrix divided by k
    seco = (value_1 - value_2) / k
    # Normalize the secondary co-occurrence matrix if specified
    if normalize:
        seco = seco / min_ext_coeff

    return seco


def chimat(era5, k=100, d1=91, d2=116, tp=True, wind=False):
    """
    Calculate the seco matrix based on precipitation and wind speed thresholds.

    Parameters:
    era5 (xarray.Dataset): The ERA5 dataset with variables "tp" (total precipitation) and "speed" (wind speed).
    k (int): An integer representing the threshold rank of the data.
    d1 (int): The threshold value for the latitude dimension.
    d2 (int): The threshold value for the longitude dimension.
    tp (bool): A boolean indicating whether to use total precipitation data (default: True).
    wind (bool): A boolean indicating whether to use wind speed data (default: False).

    Returns:
    np.array: The extremal correlation matrix.
    """

    # Rank the era5 dataset by date
    erank5 = era5.rank('date')
    # Get the number of dates in the dataset
    n = len(erank5.date)
    # Set the rank threshold to be above n - k + 0.5
    erank_threshold = (erank5 > n - k + 0.5)

    # Get the threshold for total precipitation and wind speed
    if tp:
        indicator = erank_threshold.tp[:, :d1, :d2]
    if wind:
        indicator = erank_threshold.speed[:, :d1, :d2]
    # Sum the thresholds along the date dimension
    indicator_sum = (indicator + indicator.rename(
        {'latitude': 'latitude1', 'longitude': 'longitude1'})).sum('date')
    # Reshape the sum into a 2D matrix
    value_2 = indicator_sum.values.reshape([d1*d2, d1*d2])
    # Clear the memory for the sum variable
    del indicator_sum
    # Set the number of co-occurence to the diagonal of the value 2 matrix
    ext_coeff = np.diag(value_2)
    # Set the number of co-occurence as the sum of the two indicators
    value_1 = ext_coeff[:, np.newaxis] + ext_coeff[np.newaxis, :]
    # Calculate the secondary co-occurrence matrix divided by k
    chi = (value_1 - value_2) / k
    # Normalize the secondary co-occurrence matrix if specified

    return chi

# test

#print(chimat(era5, d1=10, d2=10, tp=False, wind=True))