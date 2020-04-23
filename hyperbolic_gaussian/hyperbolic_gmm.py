# import libraries
import numpy as np
import sys
import os

# import modules within repository
sys.path.append('C:\\Users\\dreww\\Desktop\\hyperbolic-learning\\utils') 
sys.path.append('C:\\Users\\dreww\\Desktop\\hyperbolic-learning\\hyperbolic_gaussian')
from utils import *
from riemann_mean import *
from distributions import *

#----------------------------------------------------------
#----- Gaussian Mixture Model in Hyperboloid Space --------
#----------------------------------------------------------

class HyperbolicGMM():
    """
    Gaussian Mixture Model in hyperbolic space where we use the Wrapped Normal Distribution
    and its p.d.f to determine likelihood of cluster assignments. Applies gradient descent in
    the hyperboloid model to iteratively compute the Riemannian barycenter.
    
    Note: Follows the Expectation-Maximization (EM) approach for Unsupervised Clustering
    """
    
    def __init__(self,n_clusters=3,max_iter=300,tol=1e-5,verbose=False):
        """ Initialize Gaussian Mixture Model and set training parameters """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.labels = None
        self.cluster_weights = np.repeat(1, n_clusters)
        
    def init_gaussians(self, radius=0.3):
        """ Randomly sample starting points (centers) around small uniform ball """
        theta = np.random.uniform(0, 2*np.pi, self.n_clusters)
        u = np.random.uniform(0, radius, self.n_clusters)
        r = np.sqrt(u)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        centers = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
        self.means = centers
        self.variances = np.tile([1, 1], self.n_clusters).reshape((self.n_clusters, 2))
        
    #------------------------------------------------------------------------------
    #------------------------ EXPECTATION STEP ------------------------------------
    #------------------------------------------------------------------------------
        
    def normalization_term(self, xi):
        """ Sum all weighted likelihood assignments for normalization """
        total = 0
        K = len(self.cluster_weights)
        for i in range(K):
            total += self.cluster_weights[i]*log_pdf(z=xi, mu=self.means[i], sigma=self.variances[i])
        return total

    def update_likelihoods(self, X):
        """ Compute likelihoods using log-pdf of Wrapped Normal Distribution """
        N = X.shape[0]
        K = self.n_clusters
        W = np.array((K, N))
        for j in range(K):
            for i in range(N):
                W[j, i] = self.cluster_weights[j] * log_pdf(z=X[i], mu=self.means[j], sigma=self.variances[j])
                W[j, i] /= normalization_term(X[i])
        self.likelihoods = W
                
    #-----------------------------------------------------------------------------
    #------------------------ MAXIMIZATION STEP ----------------------------------
    #-----------------------------------------------------------------------------
    
    def update_cluster_weights(self):
        """ Update new cluster weights based on cluster assignment likelihoods """
        # W: matrix with weights w_1k, ..., w_Nk - shape K x N
        K = self.likelihoods.shape[0]
        N = self.likelihoods.shape[1]
        updated_weights = np.array([np.sum(self.likelihoods[i, :])/N for i in range(K)])
        self.cluster_weights = updated_weights
        
    def update_means(self, X, num_rounds=10, alpha=0.3, tol=1e-4):
        """ Apply weighted barycenter algorithm to update gaussian clusters """
        for i in range(self.n_clusters):
            self.means[i] = weighted_barycenter(self.means[i], X, self.likelihoods[i, :])
            
    def fit(self, X, y=None, max_epochs=40, verbose=False):
        """
        Fit K gaussian distributed clusters to data, return cluster assignments by max likelihood 
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y: optionally train a supervised model with given labels y (in progress)
        max_epochs: maximum number of gradient descent iterations
        verbose: optionally print training scores
        """
        
        # make sure X within poincaré ball
        if (norm(X, axis=1) > 1).any():
            X = X / (np.max(norm(X, axis=1)))
        
        # initialize random gaussian centroids
        self.n_samples = X.shape[0]
        self.init_gaussians()
        
        # compute initial likelihoods for x1, ..., xN
        self.update_likelihoods(X)
        
        # loop through the expectation and maximization steps
        for j in range(max_epochs):
            self.loss = 0
            self.update_cluster_weights()
            self.update_means(X)
            for i in range(self.n_samples):
                # zero out current cluster assignment
                self.assignments[i, :] = np.zeros((1, self.n_clusters))
                # find closest centroid (in Poincare disk)
                centroid_distances = np.array([poincare_dist(X[i], centroid) for centroid in self.centroids])
                cx = np.argmin(centroid_distances)
                self.inertia_ += centroid_distances[cx]**2
                self.assignments[i][cx] = 1
            if verbose:
                print('Epoch ' + str(j) + ' complete')
        self.labels = np.argmax(self.assignments, axis=1)
        self.cluster_var(X)
        return