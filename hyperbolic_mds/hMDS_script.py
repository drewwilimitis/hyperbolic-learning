# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:34:39 2019

@author: dreww
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.metrics import euclidean_distances

def norm(x, axis=None):
    return np.linalg.norm(x, axis=axis)

def conjugate(z):
    z_bar = np.array([z[0], -z[1]])
    return z_bar

def poincare_dist(u, v, eps=1e-5):
    d = 1 + 2 * norm(u-v)**2 / ((1 - norm(u)**2) * (1 - norm(v)**2) + eps)
    return np.arccosh(d)

def partial_d(theta, x):
    alpha = 1 - norm(theta)**2
    beta = 1 - norm(x)**2
    gamma = 1 + 2/(alpha*beta + eps) * norm(theta-x)**2
    lhs = 4 / (beta*np.sqrt(gamma**2 - 1) + eps)
    rhs = 1/(alpha**2 + eps) * (norm(x)**2 - 2*np.inner(theta,x) + 1) * theta - x/(alpha + eps)
    return lhs*rhs

def step_error(r, Z, g, dissimilarities, n):
        M_r = np.zeros((n, 2))
        for j in range(n):
            M_r[j] = (-r*g[j] + Z[j]) / (-r*g[j] * conjugate(Z[j]) + 1)
            #print(M_r[j])
        return loss_fn(M_r, dissimilarities, n)

def line_search(Z, dissimilarities, g, n, r0, rmax):
    Z_norm = norm(Z, axis=1)**2
    M_prime = g*Z_norm.reshape(-1,1)
    qprime_0 = np.dot(M_prime[:,0].T, g[:,0]) + np.dot(M_prime[:,1].T, g[:,1])
    p = 0.5
    r = r0
    roof_fn = lambda r: step_error(0, Z, g, dissimilarities, n)+p*qprime_0*r
    rmin = 1e-7
    while rmin < r < rmax and step_error(r, Z, g, dissimilarities, n) < roof_fn(r):
        r = 2*r
    while r > rmax or step_error(r, Z, g, dissimilarities, n) > roof_fn(r):
        r = r/2
    return r

class HyperMDS():
    
    def __init__(self, dim=2, max_iter=3, verbose=0, eps=1e-3, alpha=1,
                 random_state=None, dissimilarity="euclidean"):
        self.dim = dim
        self.dissimilarity = dissimilarity
        self.max_iter = max_iter
        self.alpha = alpha
        self.eps = eps
        self.verbose = verbose
        self.random_state = random_state
        
    def init_embed(self, low=-0.1, high=0.1):
        rand_init = np.random.uniform(low, high, size=(self.n, self.dim))
        self.embedding = rand_init
    
    def loss_fn(self):
        loss = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                d_ij = poincare_dist(self.embedding[i], self.embedding[j])
                delta_ij = self.alpha*self.dissimilarity_matrix[i][j]
                loss += (d_ij - delta_ij)**2
        self.loss = loss

    def compute_gradients(self):
        gradients = np.zeros((self.n, 2))
        for i in range(n):
            grad_zi = 0
            for j in range(i+1, n):
                dd_ij = 2*poincare_dist(self.embedding[i], self.embedding[j])
                ddelta_ij = 2*self.alpha*self.dissimilarity_matrix[i][j]
                dd_loss = dd_ij - ddelta_ij
                dd_dist = partial_d(self.embedding[i], self.embedding[j])
                grad_zi += dd_loss * dd_dist
            gradients[i] = grad_zi
        self.gradients = gradients
    
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

    def fit_transform(self, X, init=None, max_epochs = 40):
        """
        Fit the embedding from X, and return the embedding coordinates
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.
        init: initial configuration of the embedding coordinates
        """
        if self.dissimilarity == "precomputed":
            self.dissimilarity_matrix = X
        elif self.dissimilarity == "euclidean":
            self.dissimilarity_matrix = euclidean_distances(X)
        self.n = self.dissimilarity_matrix.shape[0]
        
        self.init_embed()
        
        #error_tol =
        #min_grad = 
        #min_step = 
        smax = 1
        for i in range(max_epochs):
            self.loss_fn()
            self.compute_gradients()
            rmax = 1/(norm(self.gradients, axis=1).max()+eps) * np.tanh(smax/2)
            r = line_search(self.embedding, self.dissimilarity_matrix, self.gradients,
                            self.n, 0.001, rmax)
            for i in range(n):
                zi_num = -r*self.gradients[i] + self.embedding[i]
                zi_denom = -r*self.gradients[i] * conjugate(self.embedding[i]) + 1
                zi_prime = zi_num / zi_denom
                self.embedding[i] = zi_prime
        return self.embedding