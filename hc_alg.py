import numpy as np
from sklearn.cluster import AgglomerativeClustering


def clust_hc(Theta, K):
    """
    Performs Hierarchical Agglomerative Clustering on the given matrix Theta with K clusters.

    Parameters:
    Theta (numpy.ndarray): A matrix to be clustered.
    K (int): The number of clusters.

    Returns:
    dict: A dictionary with cluster labels as keys and corresponding indices as values.
    """

    # Initialize AgglomerativeClustering object with given parameters and fit to Theta.
    HC = AgglomerativeClustering(
        n_clusters=K, metric='precomputed', linkage='complete').fit(Theta)

    # Get labels assigned to each data point by the clustering algorithm.
    labels = HC.labels_

    # Get unique labels and create a dictionary to store indices of each cluster.
    label = np.unique(labels)
    cluster = {}

    # Loop through unique labels and store indices of each cluster in the dictionary.
    for lab, l in enumerate(label):
        l += 1
        index = np.where(labels == lab)
        cluster[l] = index[0]

    # Return the dictionary with cluster information.
    return cluster
