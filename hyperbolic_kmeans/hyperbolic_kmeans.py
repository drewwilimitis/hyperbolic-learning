# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 05:46:45 2019

@author: dreww
"""

class PoincareKMeans():
    
    def __init__(self,n_clusters=8,n_init=20,max_iter=300,tol=1e-8,verbose=False):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose =  verbose
        self.labels_ = None
        self.cluster_centers_ = None
        
    def init_assign(self):
        assignments = np.zeros((self.n_samples, self.n_clusters))
        for i in range(self.n_samples):
            # initialize binary assignment vector
            j = np.random.randint(0, self.n_clusters)
            assignments[i][j] = 1
        self.assignments = assignments
        
    def update_centroids(self, X):
        dim = X.shape[1]
        new_centroids = np.empty((self.n_clusters, dim)) 
        for i in range(self.n_clusters):
            # find total observations in cluster
            n_k = np.sum(self.assignments, axis=0)[i]
            if n_k == 0:
                new_centroids[i] = np.zeros((1, dim))
            else:
                new_centroids[i] = np.sum(X[self.assignments[:, i] == 1], axis=0) / n_k
        self.centroids = new_centroids
    
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

    def fit_transform(self, X, max_epochs=40, verbose=False):
        """
        Fit the embedding from X, and return the embedding coordinates
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.
        max_epochs: maximum number of gradient descent iterations
        verbose: optionally print training scores
        """
        
        self.n_samples = X.shape[0]
        
        # initialize random centroid assignments
        self.init_assign()
        
        # loop through the assignment and update steps
        for j in range(max_epochs):
            self.update_centroids(X)
            for i in range(self.n_samples):
                # zero out current cluster assignment
                self.assignments[i, :] = np.zeros((1, self.n_clusters))
                # find closest centroid mean
                centroid_distances = list(np.sqrt(((X[i] - self.centroids)**2).sum(axis=1)))
                cx = centroid_distances.index(np.min(centroid_distances))
                self.assignments[i][cx] = 1
            if verbose:
                print('Epoch ' + str(j) + ' complete')
                print(self.centroids)
        return self.assignments