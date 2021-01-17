# -*- coding: utf-8 -*-
"""
@author: Theodoulos Rodosthenous
"""

'''
In this script, we include all solutions in multi-view data visualisation, namely:
    (A) multi-SNE
    (B) multi-LLE
    (C) multi-ISOMAP

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

