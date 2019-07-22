# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:34:39 2019

@author: dreww
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

def norm(x, axis=None):
    return np.linalg.norm(x, axis=axis)

def poincare_dist(u, v, eps=1e-5):
    d = 1 + 2 * norm(u-v)**2 / ((1 - norm(u)**2) * (1 - norm(v)**2) + eps)
    return np.arccosh(d)

class HyperMDS():
    
    def __init__(self, n_components=2, max_iter=100, verbose=0, eps=1e-3, n_jobs=None,
                 random_state=None, dissimilarity="euclidean"):
        self.n_components = n_components
        self.dissimilarity = dissimilarity
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state
    
    def fit(self, X, init=None):
        """
        Uses gradient descent to find the embedding configuration in the Poincaré disk
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.
        init: initial configuration of the embedding coordinates
        """
        self.fit_transform(X, init=init)
        return self
    
    def fit_transform(self, X, init=None):
        """
        Fit the embedding from X, and return the embedding coordinates
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.
        init: initial configuration of the embedding coordinates
        """