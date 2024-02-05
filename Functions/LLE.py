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
Input: X -- Multi-view data
Output: Y -- Single-view data, ideally with n_components = 2
Analysis on two approaches, two functions:
(A)
    1. Get weight matrix for each view (W^v)
    2. Averaged weight matrix, based on a coefficient (\alpha, where \sum_v{\alpha^v} = 1)
    3. Find Y by using the averaged weight matrix
(B)
    1. Get weight matrix for each view (W^v)
    2. Find Y^v based on weight matrix W^v, for each view v
    3. Take the averaged output of all Y^v, denoted Y, based on a coefficient (\beta, where \sum_v{\beta^v} = 1)

NOTE: All functions performed in the following code are part of LocallyLinearEmbedding from sklearn.manifold
        i.e. from sklearn.manifold import LocallyLinearEmbedding

    BUT, to test the performance on the method, we will create a new lle() function to be comparable
        with existing tsne() function. Similarly for multi_lle() against multiSNE()

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
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components, shortest_path



#### Single-view functions ####

def barycenter_weights(X, Z, reg=1e-3):
    """Compute barycenter weights of X from Y along the first axis
    We estimate the weights to assign to each point in Y[i] to recover
    the point X[i]. The barycenter weights sum to 1.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_dim)
    Z : array-like, shape (n_samples, n_neighbors, n_dim)
    reg : float, optional
        amount of regularization to add for the problem to be
        well-posed in the case of n_neighbors > n_dim
    Returns
    -------
    B : array-like, shape (n_samples, n_neighbors)
    Notes
    -----
    See developers note for more information.
    """
    X = check_array(X, dtype=FLOAT_DTYPES)
    Z = check_array(Z, dtype=FLOAT_DTYPES, allow_nd=True)

    n_samples, n_neighbors = X.shape[0], Z.shape[1]
    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)

    # this might raise a LinalgError if G is singular and has trace
    # zero
    for i, A in enumerate(Z.transpose(0, 2, 1)):
        C = A.T - X[i]  # broadcasting
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[::Z.shape[1] + 1] += R
        w = solve(G, v, assume_a='pos')
        B[i, :] = w / np.sum(w)
    return B

def barycenter_kneighbors_graph(X, n_neighbors, reg=1e-3, n_jobs=None):
    """Computes the barycenter weighted graph of k-Neighbors for points in X
    Parameters
    ----------
    X : {array-like, NearestNeighbors}
        Sample data, shape = (n_samples, n_features), in the form of a
        numpy array or a NearestNeighbors object.
    n_neighbors : int
        Number of neighbors for each sample.
    reg : float, optional
        Amount of regularization when solving the least-squares
        problem. Only relevant if mode='barycenter'. If None, use the
        default.
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    Returns
    -------
    A : sparse matrix in CSR format, shape = [n_samples, n_samples]
        A[i, j] is assigned the weight of edge that connects i to j.
    See also
    --------
    sklearn.neighbors.kneighbors_graph
    sklearn.neighbors.radius_neighbors_graph
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs).fit(X)
    X = knn._fit_X
    n_samples = X.shape[0]
    ind = knn.kneighbors(X, return_distance=False)[:, 1:]
    data = barycenter_weights(X, X[ind], reg=reg)
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    return csr_matrix((data.ravel(), ind.ravel(), indptr),
                      shape=(n_samples, n_samples))

def null_space(M, k, k_skip=1, eigen_solver='arpack', tol=1E-6, max_iter=100,
               random_state=None):
    """
    Find the null space of a matrix M.
    Parameters
    ----------
    M : {array, matrix, sparse matrix, LinearOperator}
        Input covariance matrix: should be symmetric positive semi-definite
    k : integer
        Number of eigenvalues/vectors to return
    k_skip : integer, optional
        Number of low eigenvalues to skip.
    eigen_solver : string, {'auto', 'arpack', 'dense'}
        auto : algorithm will attempt to choose the best method for input data
        arpack : use arnoldi iteration in shift-invert mode.
                    For this method, M may be a dense matrix, sparse matrix,
                    or general linear operator.
                    Warning: ARPACK can be unstable for some problems.  It is
                    best to try several random seeds in order to check results.
        dense  : use standard dense matrix operations for the eigenvalue
                    decomposition.  For this method, M must be an array
                    or matrix type.  This method should be avoided for
                    large problems.
    tol : float, optional
        Tolerance for 'arpack' method.
        Not used if eigen_solver=='dense'.
    max_iter : int
        Maximum number of iterations for 'arpack' method.
        Not used if eigen_solver=='dense'
    random_state : int, RandomState instance, default=None
        Determines the random number generator when ``solver`` == 'arpack'.
        Pass an int for reproducible results across multiple function calls.
        See :term: `Glossary <random_state>`.
    """
    if eigen_solver == 'auto':
        if M.shape[0] > 200 and k + k_skip < 10:
            eigen_solver = 'arpack'
        else:
            eigen_solver = 'dense'

    if eigen_solver == 'arpack':
        random_state = check_random_state(random_state)
        # initialize with [-1,1] as in ARPACK
        v0 = random_state.uniform(-1, 1, M.shape[0])
        try:
            eigen_values, eigen_vectors = eigsh(M, k + k_skip, sigma=0.0,
                                                tol=tol, maxiter=max_iter,
                                                v0=v0)
        except RuntimeError as msg:
            raise ValueError("Error in determining null-space with ARPACK. "
                             "Error message: '%s'. "
                             "Note that method='arpack' can fail when the "
                             "weight matrix is singular or otherwise "
                             "ill-behaved.  method='dense' is recommended. "
                             "See online documentation for more information."
                             % msg)

        return eigen_vectors[:, k_skip:], np.sum(eigen_values[k_skip:])
    elif eigen_solver == 'dense':
        if hasattr(M, 'toarray'):
            M = M.toarray()
        eigen_values, eigen_vectors = eigh(
            M, eigvals=(k_skip, k + k_skip - 1), overwrite_a=True)
        index = np.argsort(np.abs(eigen_values))
        return eigen_vectors[:, index], np.sum(eigen_values)
    else:
        raise ValueError("Unrecognized eigen_solver '%s'" % eigen_solver)


def lle(X=np.array([]), n_components=2, n_neighbors = 5, eigen_solver="auto", n_jobs=None, reg = 1e-3):
    '''
        X is the input matrix
        no_dims is the number of components for the embedding matrix Y
        n_neighbors is the number of neighbours to consider in running k-NN
        eigen_solver is the method use to find the lowest eigenvectors -> Y matrix
        n_jobs is the number of parallel jobs to run for neighbours search (optional)
        reg : float :regularization constant, multiplies the trace of the local covariance matrix of the distances.
    '''

    if eigen_solver not in ('auto', 'arpack', 'dense'):
        raise ValueError("unrecognized eigen_solver '%s'" % eigen_solver)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    nbrs.fit(X)
    X = nbrs._fit_X

    N, d_in = X.shape

    if n_components > d_in:
        raise ValueError("output dimension must be less than or equal "
                         "to input dimension")
    if n_neighbors >= N:
        raise ValueError(
            "Expected n_neighbors <= n_samples, "
            " but n_samples = %d, n_neighbors = %d" %
            (N, n_neighbors)
        )

    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")

    M_sparse = (eigen_solver != 'dense')

    W = barycenter_kneighbors_graph(
            X=nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs)

    if M_sparse:
        M = eye(*W.shape, format=W.format) - W
        M = (M.T * M).tocsr()
    else:
        M = (W.T * W - W.T - W).toarray()
        M.flat[::M.shape[0] + 1] += 1  # W = W - I = W - I

    return null_space(M, n_components, k_skip=1, eigen_solver=eigen_solver)

#### Multi-view LLE ####
## (A) -- Averaged Weight Matrix

def multiLLE(X = np.array([[]]), n_components=2, n_neighbors = 5,
                    eigen_solver="auto", n_jobs=None, reg = 1e-3):
        '''
        X is the input matrix
            -- Multi-view data, in the same structure as multi-SNE
        no_dims is the number of components for the embedding matrix Y
        n_neighbors is the number of neighbours to consider in running k-NN
        eigen_solver is the method use to find the lowest eigenvectors -> Y matrix
        n_jobs is the number of parallel jobs to run for neighbours search (optional)
        reg : float :regularization constant, multiplies the trace of the local covariance matrix of the distances.

        In this function, we will follow the (A) solution
        i.e. take the averaged weight matrix out of all views
    '''
    # Now, initialization for each data-set
    dim = X.shape
    V = dim[0] # Number of views
    # Get Weight Matrix for each view
    W = X#.copy(deep=True)
    print("Computing Weight matrix for each view")
    for view in range(V):
        Xtemp = X[view]#.copy(deep=True)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
        nbrs.fit(Xtemp)
        Xtemp = nbrs._fit_X
        N, d_in = Xtemp.shape

        if n_components > d_in:
            raise ValueError("output dimension must be less than or equal "
                             "to input dimension")
        if n_neighbors >= N:
            raise ValueError(
                "Expected n_neighbors <= n_samples, "
                " but n_samples = %d, n_neighbors = %d" %
                (N, n_neighbors)
            )
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")

        Wtemp = barycenter_kneighbors_graph(
            nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs)
        W[view] = Wtemp#.copy(deep=True)

    # Take the averaged Weight Matrix
    print("Combining the weight matrices")
    W_total = W[0]
    for view in range(1,V):
        W_total =  W[view] + W_total
    W_averaged = W_total / V

    M_sparse = (eigen_solver != 'dense')

    print("Computing the d-dimensional embedding")
    if M_sparse:
        M = eye(*W_averaged.shape, format=W_averaged.format) - W_averaged
        M = (M.T * M).tocsr()
    else:
        M = (W_averaged.T * W_averaged - W_averaged.T - W_averaged).toarray()
        M.flat[::M.shape[0] + 1] += 1  #  = W - I = W - I

    return null_space(M, n_components, k_skip=1, eigen_solver=eigen_solver)

## (B) -- Averaged Embeddings

def mLLE(X = np.array([[]]), n_components=2, n_neighbors = 5,
                    eigen_solver="auto", n_jobs=None, reg = 1e-3):
    '''
        X is the input matrix
            -- Multi-view data, in the same structure as multi-SNE
        no_dims is the number of components for the embedding matrix Y
        n_neighbors is the number of neighbours to consider in running k-NN
        eigen_solver is the method use to find the lowest eigenvectors -> Y matrix
        n_jobs is the number of parallel jobs to run for neighbours search (optional)
        reg : float :regularization constant, multiplies the trace of the local covariance matrix of the distances.

        In this function, we will follow the (A) solution
        i.e. take the averaged weight matrix out of all views
    '''
    # Now, initialization for each data-set
    dim = X.shape
    V = dim[0] # Number of views
    # Get Weight Matrix for each view
    W = X#.copy(deep=True)
    Y = X
    print("Computing LLE for each view")
    for view in range(V):
        Xtemp = X[view]#.copy(deep=True)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
        nbrs.fit(Xtemp)
        Xtemp = nbrs._fit_X
        N, d_in = Xtemp.shape

        if n_components > d_in:
            raise ValueError("output dimension must be less than or equal "
                             "to input dimension")
        if n_neighbors >= N:
            raise ValueError(
                "Expected n_neighbors <= n_samples, "
                " but n_samples = %d, n_neighbors = %d" %
                (N, n_neighbors)
            )
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")

        Wtemp = barycenter_kneighbors_graph(
            nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs)

        M_sparse = (eigen_solver != 'dense')

        if M_sparse:
            M = eye(*Wtemp.shape, format=Wtemp.format) - Wtemp
            M = (M.T * M).tocsr()
        else:
            M = (Wtemp.T * Wtemp - Wtemp.T - Wtemp).toarray()
            M.flat[::M.shape[0] + 1] += 1  #  = W - I = W - I

        Y[view] = null_space(M, n_components, k_skip=1, eigen_solver=eigen_solver)

    # Take the averaged Weight Matrix
    print("Combining the d-dimensional embeddings")
    Y_total = Y[0][0]
    for view in range(1,V):
        Y_total =  Y[view][0] + Y_total
    Y_averaged = Y_total / V

    return Y_averaged


