"""
The two functions clust and find_max are related to the problem of clustering, which involves grouping similar objects together based on some similarity measure.

The clust function takes in a 2D array Theta, an integer n, and an optional value alpha. It applies a clustering algorithm to the rows and columns of Theta 
and returns a list of clusters. The algorithm starts with all columns indices iteratively groups these indices until each indices is assigned to a cluster. 
The value of alpha is calculated based on the number of rows and columns in Theta, and a default value is used if alpha is not provided.

The find_max function is a helper function used by clust. It takes in a 2D array M and a 1D array S of indices, and returns the row and column indices of the
maximum element in the submatrix M[S,S]. This function is used in clust to find the two rows or columns with the highest similarity at each iteration of the
clustering algorithm.

Together, these two functions can be used to perform clustering on a 2D array Theta, with the option of specifying the threshold alpha. The resulting clusters
can be used for further analysis or visualization of the data.
"""

import numpy as np
import pandas as pd


def find_max(M, S):
    """
    Find the pair of indices (i,j) that correspond to the maximum element in the submatrix M[S,S],
    where S is a subset of the indices of M.

    Args:
        M (np.ndarray): A 2D array to extract the maximum value from.
        S (np.ndarray): A 1D array of indices to extract the submatrix M[S,S] from.

    Returns:
        i (int): The row index of the maximum element in M[S,S].
        j (int): The column index of the maximum element in M[S,S].
    """
    # Create a boolean mask that selects the submatrix M[S,S].
    mask = np.zeros(M.shape, dtype=bool)
    values = np.ones((len(S), len(S)), dtype=bool)
    mask[np.ix_(S, S)] = values
    np.fill_diagonal(mask, 0)

    # Extract the maximum value from the submatrix M[S,S].
    max_value = M[mask].max()

    # Find the row and column index of the maximum value in the submatrix M[S,S].
    i, j = np.argwhere((M * mask) == max_value)[0]

    return i, j


def clust(Theta, n, alpha=None):
    """
    Cluster the columns of Theta into groups using correlation threshold alpha.
    If alpha is not provided, it is set automatically based on the number of columns and sample size.

    Args:
        Theta (np.ndarray): A 2D array of shape (d, d) containing n samples and d features.
        n (int): The number of samples.
        alpha (float): The correlation threshold to use for clustering the columns of Theta.

    Returns:
        clusters (dict): A list of lists, where each inner list contains the indices of the columns in a cluster.
    """
    d = Theta.shape[1]
    S = np.arange(d)
    l = 0

    # Set the correlation threshold automatically if alpha is not provided.
    if alpha is None:
        alpha = 2 * np.sqrt(np.log(d) / n)

    cluster = {}
    while len(S) > 0:
        l = l + 1
        if len(S) == 1:
            cluster[l] = np.array(S)
        else:
            # Find the pair of columns with the highest correlation.
            a_l, b_l = find_max(Theta, S)
            if Theta[a_l, b_l] < alpha:
                cluster[l] = np.array([a_l])
            else:
                # Find the indices of columns that have high correlation with both a_l and b_l.
                index_a = np.where(Theta[a_l, :] >= alpha)
                index_b = np.where(Theta[b_l, :] >= alpha)

                # Add the intersection of these indices to the current cluster.
                cluster[l] = np.intersect1d(S, index_a, index_b)

        # Remove the columns in the current cluster from S.
        S = np.setdiff1d(S, cluster[l])

    return cluster


def SECO(era5, clst, k, tp=False, wind=False):
    """ evaluation of the criteria

    Input
    -----
        R (np.array(float)) : n x d rank matrix
                  w (float) : element of the simplex
           cols (list(int)) : partition of the columns

    Output
    ------
        Evaluate (theta - theta_\Sigma) 

    """
    eco = era5.rank('date') > len(era5.date) - k + 0.5
    eco = eco.stack(pos=('latitude', 'longitude'))
    d = len(eco.pos)

    # Evaluate the cluster as a whole
    if tp:
        indicator_tp = np.array(eco.tp[:, range(0, d)])
        sum_indicator = np.sum(indicator_tp, axis=1, dtype=bool)
    elif wind:
        indicator_speed = np.array(eco.spped[:, range(0, d)])
        sum_indicator = np.sum(sum_indicator) / k
    else:
        indicator_tp = np.array(eco.tp[:, range(0, d)])
        indicator_speed = np.array(eco.speed[:, range(0, d)])
        sum_indicator = np.sum(
            indicator_tp+indicator_speed, axis=1, dtype=bool)
    value = np.sum(sum_indicator)/k
    _value_ = []
    for c in clst.values():
        indicator_tp = np.array(eco.tp[:, c])
        indicator_speed = np.array(eco.speed[:, c])
        sum_indicator = np.sum(
            indicator_tp+indicator_speed, axis=1, dtype=bool)
        _value_.append(np.sum(sum_indicator)/k)

    value_sum = []
    for i in range(0, len(_value_)):
        value_ = np.sum(np.delete(_value_, i))
        value_sum.append(value_)

    return (np.sum(_value_) - value)


# matseco = np.array(pd.read_csv('data/seco.csv', index_col=0))
# O_hat = clust(matseco - 0.000000001, n=1000, alpha=0.08)
# print(O_hat)
