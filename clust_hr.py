"""
    This Python code imports necessary libraries, defines functions to perform clustering analysis, and executes the compound clustering analysis algorithm (CAICE) or (HC)
    on input matrices. 
    
    Four functions are defined in the code. The first function, threshold_seco, computes the SECO values for several given thresholds and plots the results if required.
    It takes several parameters including matseco which is a numpy array containing the matrix of SECO values, era5 which is a numpy array containing the data to be analyzed,
    _alpha_ which is a list of values to be used as thresholds in the (CAICE) algorithm, plot which is an optional boolean indicating whether or not to plot the computed SECO
    values against the given thresholds.

    The second function, hr_caice_run, runs clustering using the CAICE algorithm on an input matrix and saves the results to a pickle file if specified. It takes several
    parameters including save which is a boolean flag indicating whether to save the clustering results to a pickle file, tp which is a boolean flag indicating whether
    to use the extremal correlation of daily total precipitation for clustering, and wind which is a boolean flag indicating whether to use the extremal correlation of
    wind speed maxima file for clustering.

    The third function, hr_hc_run, runs clustering using the HC algorithm on an input matrix and saves the results to a pickle file if specified. It takes several parameters
    such as boolean flag.

    The four function, hr_plot, return some figures using as input a dictionary of clusters O_hat and the seco dissimilarity matrix. It randomly shuffles and combines the 
    cluster indices to create a new matrix. Finally, it creates a map of the clustered regions using GeoPandas and matplotlib if specified, sorts the dictionary of clusters
    by the length of the cluster, and creates a plot of the 9 largest clusters using seaborn and GeoPandas if specified.

    The remainder of the code opens two NetCDF files containing high-resolution ERA5 wind and temperature data, stacks the latitude and longitude dimensions of the wind data,
    adds total precipitation data as a new variable, and deletes the original wind and total precipitation data variables.
"""

import eco_alg
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import random
import seaborn as sns
import pickle
import geopandas as gpd
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon
from shapely.ops import unary_union
from itertools import islice
from eco_alg import clust, SECO
from hc_alg import clust_hc
crest = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)


# Define a function to compute SECO values for a given threshold and plot the results if required
def threshold_seco(matseco, era5, _alpha_, plot=False, tp=False, wind=False):
    """
    Function to obtain a data-driven choice of the threshold in the CAICE algorithm.

    Parameters:
    matseco (numpy.ndarray): A numpy array containing the matrix of SECO values.
    era5 (xarray.Dataset): A numpy array containing the data to be analyzed.
    _alpha_ (list): A list of values to be used as thresholds for computing SECO values.
    plot (optional): A boolean indicating whether or not to plot the computed SECO values against the given thresholds. Default value is False.
    tp (optional): A boolean indicating whether or not to use the 'tp' option when computing SECO values. Default value is False.
    wind (optional): A boolean indicating whether or not to use the 'wind' option when computing SECO values. Default value is False.
    """

    # Initialize an empty list to store the computed SECO values
    value_SECO = []

    # Loop through the given list of thresholds and compute SECO values for each threshold
    for alpha in _alpha_:

        # Compute clusters using the given threshold and a pre-defined clustering function
        clusters = clust(matseco+0.00000001, n=1000, alpha=alpha)

        # Compute the SECO value for the given era5 data and clusters, using a pre-defined SECO function
        if tp:
            value = SECO(era5, clusters, k=30, tp=True)
        elif wind:
            value = SECO(era5, clusters, k=30, wind=True)
        else:
            value = SECO(era5, clusters, k=30)

        # Append the computed SECO value to the list of SECO values
        value_SECO.append(value)

    # Convert the list of SECO values to a numpy array
    value_SECO = np.array(value_SECO)

    # Normalize the SECO values to lie in the range (0,inf)
    value_SECO = np.log(1+value_SECO - np.min(value_SECO))

    # If plot is True, plot the computed SECO values against the given thresholds and return the plot object
    if plot:
        fig, ax = plt.subplot()
        ax.plot(_alpha_, value_SECO, marker='o', linestyle='solid',
                markerfacecolor='white', lw=1, color='#6F6F6F')
        ax.set_ylabel(r'$L$')
        ax.set_xlabel(r'Threshold $\tau$')
        return fig, ax

    # If plot is False, return the list of normalized SECO values
    else:
        return (value_SECO)


# Define a function to run compound clustering analysis
# The function opens two NetCDF files containing high-resolution ERA5 wind and temperature data
# Stacks the latitude and longitude dimensions of the wind data and adds temperature data as a new variable
# Deletes the original wind and temperature data variables
# Reads in a CSV file containing a matrix of seco dissimilarities and converts it to a numpy array
# Clusters the matrix using the clust function and saves the resulting dictionary object to a pickle file if specified
# Randomly shuffles and combines the cluster indices to create a new matrix
# Calculates the sizes of each cluster and creates a cumulative sum of cluster sizes
# Creates a map of the clustered regions using GeoPandas and matplotlib if specified
# Sorts the dictionary of clusters by length of cluster and creates a plot of the 9 largest clusters using seaborn and GeoPandas if specified

def hr_caice_run(save=False, tp=False, wind=False):
    """
    This function runs clustering using the CAICE algorithm on an input matrix and saves the results to a pickle file if specified.

    Parameters:
    save (boolean): flag indicating whether to save the clustering results to a pickle file.
    tp (boolean): flag indicating whether to use the eco_tp.csv file for clustering.
    wind (boolean): flag indicating whether to use the eco_speed.csv file for clustering.
    """

    # Read the appropriate matrix based on the input flags
    if tp:
        matseco = np.array(pd.read_csv('data/eco_tp.csv', index_col=0))
    elif wind:
        matseco = np.array(pd.read_csv('data/eco_speed.csv', index_col=0))
    else:
        matseco = np.array(pd.read_csv('data/seco.csv', index_col=0))

    # Cluster the matrix using the CAICE algorithm with appropriate parameters based on the input flags
    if tp:
        O_hat = clust(matseco+0.00000001, n=1000, alpha=0.09)
    elif wind:
        O_hat = clust(matseco+0.00000001, n=1000, alpha=0.07)
    else:
        O_hat = clust(matseco+0.00000001, n=1000, alpha=0.08)

    # Save the results to a pickle file if specified
    if save:
        if tp:
            with open('results/O_hat_tp.pkl', 'wb') as fp:
                pickle.dump(O_hat, fp)
        elif wind:
            with open('results/O_hat_wind.pkl', 'wb') as fp:
                pickle.dump(O_hat, fp)
        else:
            with open('results/O_hat.pkl', 'wb') as fp:
                pickle.dump(O_hat, fp)

    # Return the clustering results and the original matrix
    return O_hat, matseco


def hr_hc_run(save=False, tp=False, wind=False):
    """
    This function runs hierarchical clustering on an input matrix and saves the results to a pickle file if specified.

    Parameters:
    save (boolean): flag indicating whether to save the clustering results to a pickle file.
    tp (boolean): flag indicating whether to use the eco_tp.csv file for clustering.
    wind (boolean): flag indicating whether to use the eco_speed.csv file for clustering.
    """

    # Read the appropriate matrix based on the input flags
    if tp:
        matseco = np.array(pd.read_csv('data/eco_tp.csv', index_col=0))
    elif wind:
        matseco = np.array(pd.read_csv('data/eco_speed.csv', index_col=0))
    else:
        matseco = np.array(pd.read_csv('data/seco.csv', index_col=0))

    # Cluster the matrix using hierarchical clustering
    O_hat = clust_hc(1-matseco, K=25)

    # Save the results to a pickle file if specified
    if save:
        if tp:
            with open('results/O_hat_hc_tp.pkl', 'wb') as fp:
                pickle.dump(O_hat, fp)
        elif wind:
            with open('results/O_hat_hc_wind.pkl', 'wb') as fp:
                pickle.dump(O_hat, fp)
        else:
            with open('results/O_hat_hc.pkl', 'wb') as fp:
                pickle.dump(O_hat, fp)

    # Return the clustering results and the original matrix
    return O_hat, matseco


def hr_plot(O_hat, matseco, save=False, tp=False, wind=False):
    # Randomly shuffle and combine cluster indices to create new matrix
    index = []
    for key, item in O_hat.items():
        shuffled = sorted(item, key=lambda k: random.random())
        index.append(shuffled)
    index = np.hstack(index)
    new_Theta = matseco[index, :][:, index]

    # Calculate cluster sizes and create cumulative sum of cluster sizes
    sizes = np.zeros(len(O_hat)+1)
    for key, item in O_hat.items():
        sizes[key] = len(item)
    cusizes = np.cumsum(sizes)
    if save:
        fig, ax = plt.subplots()
        im = plt.imshow(new_Theta, cmap=crest)
        for i in range(0, len(O_hat)):
            ax.add_patch(Rectangle((cusizes[i], cusizes[i]), sizes[i+1],
                         sizes[i+1], edgecolor='#323232', fill=False, lw=0.1))
        plt.colorbar(im)
        if tp:
            fig.savefig('results/tp/matseco_clust_emphase.pdf')
        elif wind:
            fig.savefig('results/wind/matseco_clust_emphase.pdf')
        else:
            fig.savefig('results/compound/matseco_clust_emphase.pdf')

    # Create map of clustered regions using GeoPandas and matplotlib if specified
    polys1 = gpd.GeoSeries(Polygon([(-16, 29), (43, 29), (43, 76), (-16, 76)]))
    df1 = gpd.GeoDataFrame({'geometry': polys1}).set_crs('epsg:4326')
    world = gpd.read_file("data/world-administrative-boundaries.geojson")
    frankreich = gpd.overlay(world, df1, how='intersection')
    O_hat = dict(sorted(O_hat.items(), key=lambda i: -len(i[1])))
    plt.style.use('seaborn-whitegrid')
    i = 0
    if save:
        fig, ax = plt.subplots()
        qualitative_colors = sns.color_palette('Paired', 12)
        for clst in islice(O_hat, 9):
            coordinate_ = np.array(eco_stack.pos[O_hat[clst]])
            polygon = []
            for coord in coordinate_:
                polygon_geom = Polygon([((coord[1]-0.25), (coord[0]-0.25)), ((coord[1]+0.25), (coord[0]-0.25)),
                                       ((coord[1]+0.25), (coord[0]+0.25)), ((coord[1]-0.25), (coord[0]+0.25))])
                polygon.append(polygon_geom)
                cu = unary_union(polygon)
            if cu.geom_type == 'MultiPolygon':
                for geom in cu.geoms:
                    xs, ys = geom.exterior.xy
                    ax.fill(xs, ys, fc=qualitative_colors[i], ec='none')
                    ax.plot(
                        xs, ys, color=qualitative_colors[i], linewidth=0.75)
            elif cu.geom_type == 'Polygon':
                xs, ys = cu.exterior.coords.xy
                ax.fill(xs, ys, fc=qualitative_colors[i], ec='none')
                ax.plot(xs, ys, color=qualitative_colors[i], linewidth=0.75)
            i += 1

        if tp:
            ax.text(32.5, 73.0, '6', fontsize=12)
            ax.text(11.0, 65.5, '4', fontsize=12)
            ax.text(3.60, 73.0, '2', fontsize=12)
            ax.text(-1.5, 65.5, '8', fontsize=12)
            ax.text(30.0, 65.0, '1', fontsize=12)
            ax.text(34.0, 58.2, '3', fontsize=12)
            ax.text(7.0, 51.5, '10', fontsize=12)
            ax.text(-8.5, 47.0, '18', fontsize=12)
            ax.text(10.5, 32.5, '16', fontsize=12)
        elif wind:
            ax.text(15.5, 70.0, '6', fontsize=12)
            ax.text(-12.0, 65.5, '4', fontsize=12)
            ax.text(-12.0, 57.0, '2', fontsize=12)
            ax.text(17.5, 55.5, '8', fontsize=12)
            ax.text(-8.0, 45.0, '1', fontsize=12)
            ax.text(7.0, 42.2, '3', fontsize=12)
            ax.text(-6.5, 32.2, '10', fontsize=12)
            ax.text(17, 34.0, '18', fontsize=12)
            ax.text(38, 34.0, '16', fontsize=12)
        else:
            ax.text(10.2, 52.5, '4', fontsize=12)
            ax.text(20.4, 71.2, '1', fontsize=12)
            ax.text(34.0, 61.5, '2', fontsize=12)
            ax.text(39.0, 52.6, '5', fontsize=12)
            ax.text(5.10, 34.0, '7', fontsize=12)
            ax.text(31.5, 33.9, '9', fontsize=12)
            ax.text(13.2, 42.8, '14', fontsize=12)
            ax.text(-10.3, 59.1, '6', fontsize=12)
            ax.text(-8.8, 45.0, '10', fontsize=12)

        frankreich.boundary.plot(ax=ax, linewidth=0.5, color='black')
        frankreich.plot(ax=ax, color='white', alpha=0.2)
        ax.set_aspect(1.0)
        if tp:
            fig.savefig('results/tp/clust_1_9_together.pdf')
        elif wind:
            fig.savefig('results/wind/clust_1_9_together.pdf')
        else:
            fig.savefig('results/compound/clust_1_9_together.pdf')
# launch


# Open high-resolution wind and temperature data
era5_wind = xr.open_dataset("data/high_res/era5hr_wind.nc")
era5_tp = xr.open_dataset("data/high_res/era5hr_tp.nc")
# Stack latitude and longitude dimensions of wind data and add temperature data as new variable
era5 = era5_wind
era5['tp'] = era5_tp.tp
# Delete original wind and temperature data variables
del era5_wind
del era5_tp
# Stack the position dimensions of the data and convert seco dissimilarity matrix to numpy array
eco_stack = era5.stack(pos=('latitude', 'longitude'))

O_hat, matseco = hr_hc_run(save=True)
hr_plot(O_hat, matseco, save=True)
