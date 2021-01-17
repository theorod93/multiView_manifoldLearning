# -*- coding: utf-8 -*-
"""
@author: Theodoulos Rodosthenous
"""

'''
This script contains the code of the following functions
-> LLE
-> m-LLE
-> multi-LLE
Found in 'Multi-view data visualisation via manifold learning' paper

Outline:
Multi-view ISOMAP:
Input: X = (X_1,..., X_M), where M is the number of views
    X_m \in R^{NxD_m}, for each m = {1,...,M}
Output: Y \in R^{Nxd}, d:usually equals to 2 for good 2D visualisation
Steps: On three separate scenarios:
(A) multi-isomap (graph
    Define graphs G_m~(V, E_m), for each view m = {1,...,M}
    V: Number of n_samples
    E_m: Edges, for which the length is defined by the distance between two vertices
*   Combine all graphs into a single graph \hat{G} by
        (i) SNF
        (ii) Average
    Remaining steps remain the same as standard (single-view) ISOMAP
(B) multi-isomap (path)
    Define graphs G_m~(V, E_m), for each view m = {1,...,M}
    V: Number of n_samples
    E_m: Edges, for which the length is defined by the distance between two vertices
    Compute shortest paths D_m = {d_m(i,j), \forall i,j \in V}, for each view m = {1,...,M}, by taking the distances between
    all pairs of points in G_m
*   Combine all shortest paths into a single shortest path measure by
        (i) Minial between all D_m
            (a) Avoiding zero-valued D_m
            (b) Considering zero-valued D_m
        (ii) Average
    Remaining steps remain the same as standard (single-view) ISOMAP
(C) m-isomap (embeddings)
    Define graphs G_m~(V, E_m), for each view m = {1,...,M}
    V: Number of n_samples
    E_m: Edges, for which the length is defined by the distance between two vertices
    Compute shortest paths D_m = {d_m(i,j), \forall i,j \in V}, for each view m = {1,...,M}, by taking the distances between
    all pairs of points in G_m
    Compute d-dimensional embeddings for each D_m into Y_m, for each m = {1,...,M}
*   Combine the final d-dimensional embedding into a single \hat{Y}  by
        (i) Average

'''

import numpy as np
import pylab
from scipy.linalg import eigh, svd, qr, solve
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import eigsh
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.neighbors import NearestNeighbors
import math
from sklearn.utils.graph import graph_shortest_path

def mds(data, n_components=2):
    """
    Apply multidimensional scaling (aka Principal Coordinates Analysis)
    :param data: nxn square distance matrix
    :param n_components: number of components for projection
    :return: projected output of shape (n_components, n)
    """

    # Center distance matrix
    center(data)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_val_cov, eig_vec_cov = np.linalg.eig(data)
    eig_pairs = [
        (np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))
    ]
    # Select n_components eigenvectors with largest eigenvalues, obtain subspace transform matrix
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    eig_pairs = np.array(eig_pairs)
    matrix_w = np.hstack(
        [eig_pairs[i, 1].reshape(data.shape[1], 1) for i in range(n_components)]
    )

    # Return samples in new subspace
    return matrix_w

def center(K):
    """
    Method to center the distance matrix
    :param K: numpy array of shape mxm
    :return: numpy array of shape mxm
    """
    n_samples = K.shape[0]

    # Mean for each row/column
    meanrows = np.sum(K, axis=0) / n_samples
    meancols = (np.sum(K, axis=1)/n_samples)[:, np.newaxis]

    # Mean across all rows (entire matrix)
    meanall = meanrows.sum() / n_samples

    K -= meanrows
    K -= meancols
    K += meanall
    return K

def distance_mat(X, n_neighbors=6):
    """
    Compute the square distance matrix using Euclidean distance
    :param X: Input data, a numpy array of shape (img_height, img_width)
    :param n_neighbors: Number of nearest neighbors to consider, int
    :return: numpy array of shape (img_height, img_height), numpy array of shape (img_height, n_neighbors)
    """
    def dist(a, b):
        return np.sqrt(sum((a - b)**2))

    # Compute full distance matrix
    distances = np.array([[dist(p1, p2) for p2 in X] for p1 in X])

    # Keep only the 6 nearest neighbors, others set to 0 (= unreachable)
    neighbors = np.zeros_like(distances)
    sort_distances = np.argsort(distances, axis=1)[:, 1:n_neighbors+1]
    for k,i in enumerate(sort_distances):
        neighbors[k,i] = distances[k,i]
    return neighbors, sort_distances

def isomap(data, n_components=2, n_neighbors=6):
    """
    Dimensionality reduction with isomap algorithm
    :param data: input image matrix of shape (n,m) if dist=False, square distance matrix of size (n,n) if dist=True
    :param n_components: number of components for projection
    :param n_neighbors: number of neighbors for distance matrix computation
    :return: Projected output of shape (n_components, n)
    """
    # Compute distance matrix
    data, _ = distance_mat(data, n_neighbors)

    # Compute shortest paths from distance matrix

    graph = graph_shortest_path(data, directed=False)
    graph = -0.5 * (graph ** 2)

    # Return the MDS projection on the shortest paths graph
    return mds(graph, n_components)

def m_isomap(X, n_components=2, n_neighbors=6): 
    # Combines embeddings
    # Follow the (C) scenario for multi-view ISOMAP
    M = X.shape[0]

    embeddings = X
    print("Computing ISOMAP for each view")
    for view in range(M):
        # Compute ISOMAP for each view
        embedding_temp = isomap(X[view], n_components=n_components, n_neighbors=n_neighbors)
        embeddings[view] = embedding_temp

    # Combine d-dimensional embeddings
    print("Combining d-dimensional embeddings")
    embedding_total = embeddings[0]
    for view in range(1,M):
        embedding_total = embedding_total + embeddings[view]
    embedding_final = embedding_total/M

    # Return the MDS projection on the combined shortest paths graph
    return embedding_final

def multi_isomap_path(X, n_components=2, n_neighbors=6, method = 'average', zero_valued = False):
    # Combines paths
    # Follow the (B) scenario for multi-view ISOMAP
    M = X.shape[0]

    all_graphs = X
    print("Computing distance matrix for each view")
    print("Computing the shortest paths for each view")
    for view in range(M):
        # Compute distance matrix for each view
        g_temp,_ = distance_mat(X[view], n_neighbors)
        # Compute shortest paths from distance matrix
        graph_temp = graph_shortest_path(g_temp, directed=False)
        graph_temp = -0.5 * (graph_temp ** 2)
        all_graphs[view] = graph_temp

    # Combine shortest paths
    print("Combining the shortest paths")
    if method == 'average':
        graph_total = all_graphs[0]
        for view in range(1,M):
            graph_total = graph_total + all_graphs[view]
        graphs_final = graph_total/M
    elif method == 'minimal':
        if zero_valued:
            print("TODO")
            # # TODO:
        else:
            print("TODO")
            # # TODO:
        # # TODO:
    else:
        print("Please provide a method between 'average' and 'snf'")


    # Return the MDS projection on the combined shortest paths graph
    return mds(graphs_final, n_components)

def multi_isomap_graph(X, n_components=2, n_neighbors=6, method = 'average'):
    # Combines graphs
    # Follow the (A) scenario for multi-view ISOMAP
    M = X.shape[0]

    # Compute distance matrix for each view
    print("Computing distance matrix for each view")
    G = X
    for view in range(M):
        g_temp,_ = distance_mat(X[view], n_neighbors)
        G[view] = g_temp
    # Combine distance matrix
    print("Combining the distance matrices")
    if method == 'average':
        G_total = G[0]
        for view in range(1,M):
            G_total = G_total + G[view]
        G_final = G_total/M
    elif method == 'snf':
        print("TODO")
        # # TODO:
    else:
        print("Please provide a method between 'average' and 'snf'")

    # Compute shortest paths from distance matrix
    print("Computing the shortest paths")
    graph = graph_shortest_path(G_final, directed=False)
    graph = -0.5 * (graph ** 2)

    # Return the MDS projection on the shortest paths graph
    return mds(graph, n_components)
