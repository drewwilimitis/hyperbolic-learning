# import libraries
import numpy as np
import sys
my_path = 'C:\\Users\\dreww\\Desktop\\hyperbolic-learning' # path to utils.py 
sys.path.append(my_path)
from utils import *

################################################
# Frechet Mean Optimization in Hyperboloid Model
################################################

def exp_map(v, theta, eps=1e-6):
    # v: tangent vector in minkowski space
    # theta: parameter vector in hyperboloid with centroid coordinates
    # project vector v from tangent minkowski space -> hyperboloid
    return np.cosh(norm(v))*theta + np.sinh(norm(v)) * v / (norm(v) + eps)

def minkowski_distance_gradient(u, v):
    # u,v in hyperboloid
    # returns gradient with respect to u
    return -1*(hyperboloid_dot(u,v)**2 - 1)**-1/2 * v

def minkowski_loss_gradient(theta, X):
    # X : array with points in hyperboloid cluster
    # theta: parameter vector in hyperboloid with centroid coordinates
    # returns gradient vector
    distances = np.array([-1*hyperboloid_dist(theta, x) for x in X]).reshape(-1,1)
    distance_grads = np.array([minkowski_distance_gradient(theta, x) for x in X])
    grad_loss = 2*np.mean(distances*distance_grads, axis=0)
    if np.isnan(grad_loss).any():
        print('Hyperboloid dist returned nan value')
        return eps
    else:
        return grad_loss

def project_to_tangent(theta, minkowski_grad):
    # grad: gradient vector in ambient Minkowski space
    # theta: parameter vector in hyperboloid with centroid coordinates
    # projects to hyperboloid gradient in tangent space
    return minkowski_grad + hyperboloid_dot(theta, minkowski_grad)*theta

def update_theta(theta, hyperboloid_grad, alpha=0.1):
    # theta: parameter vector in hyperboloid with centroid coordinates
    return exp_map(-1*alpha*hyperboloid_grad, theta)

def frechet_loss(theta, X):
    s = X.shape[0]
    dist_sq = np.array([hyperboloid_dist(theta, x)**2 for x in X])
    return np.sum(dist_sq) / s

def compute_mean(theta, X, num_rounds = 10, alpha=0.3, tol = 1e-4, verbose=False):
    centr_pt = theta.copy()
    centr_pts = []
    losses = []
    for i in range(num_rounds):
        gradient_loss = minkowski_loss_gradient(centr_pt, X)
        tangent_v = project_to_tangent(centr_pt, -gradient_loss)
        centr_pt = update_theta(centr_pt, tangent_v, alpha=alpha)
        centr_pts.append(centr_pt)
        losses.append(frechet_loss(centr_pt, X))
        if verbose:
            print('Epoch ' + str(i+1) + ' complete')
            print('Loss: ', frechet_loss(centr_pt, X))
            print('\n')
    return centr_pt

#####################################
# Hyperbolic K-Means Clustering Model
#####################################

class HyperbolicKMeans():
    """
    Perform K-Means clustering in hyperbolic space. Applies gradient descent in
    the hyperboloid model to iteratively compute Fréchet means, and the Poincaré disk
    model for visualization.
    
    API design is modeled on the standard scikit-learn Classifier API
    """
    
    def __init__(self,n_clusters=6,max_iter=300,tol=1e-8,verbose=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose =  verbose
        self.labels = None
        self.cluster_centers_ = None
        
    def init_centroids(self, radius=0.3):
        # randomly sample starting points on small uniform ball
        theta = np.random.uniform(0, 2*np.pi, self.n_clusters)
        u = np.random.uniform(0, radius, self.n_clusters)
        r = np.sqrt(u)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        centers = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
        self.centroids = centers
        
    def init_assign(self):
        # cluster assignments as indicator matrix
        assignments = np.zeros((self.n_samples, self.n_clusters))
        for i in range(self.n_samples):
            # randomly initialize each binary vector
            j = np.random.randint(0, self.n_clusters)
            assignments[i][j] = 1
        self.assignments = assignments
        
    def update_centroids(self, X):
        """Updates centroids with Fréchet means in Hyperboloid model
        Parameters
        ----------
        X : array of shape (n_samples, dim) with input data.
        First convert X to hyperboloid points
        """
        dim = X.shape[1]
        new_centroids = np.empty((self.n_clusters, dim)) 
        H = poincare_pts_to_hyperboloid(X)
        for i in range(self.n_clusters):
            if np.sum(self.assignments[:, i] ==1) == 0:
                new_centroids[i] = self.centroids[i]
            else:
                # find subset of observations in cluster
                H_k = H[self.assignments[:, i] ==1]
                theta_k = poincare_pt_to_hyperboloid(self.centroids[i])
                # solve for frechet mean
                fmean_k = compute_mean(theta_k, H_k, alpha=0.1)
                # convert back to Poincare disk
                new_centroids[i] = hyperboloid_pt_to_poincare(fmean_k)
        self.centroids = new_centroids
        
    def cluster_var(self, X):
        n = self.centroids.shape[0]
        var_C = []
        for i in range(n):
            var_C.append(np.mean(np.array([poincare_dist(self.centroids[i], x) for x in X])))
        self.variances = np.sort(var_C)[-2]
    
    def fit(self, X):
        """
        Apply K means algorithm in the Poincaré disk for input data X
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        """
        self.fit_predict(X)
        return self

    def fit_predict(self, X, max_epochs=40, verbose=False):
        """
        Fit the K centroids from X, and return the class assignments by nearest centroid
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        max_epochs: maximum number of gradient descent iterations
        verbose: optionally print training scores
        """
        
        # make sure X within poincaré ball
        if (norm(X, axis=1) > 1).any():
            X = X / (np.max(norm(X, axis=1)))
        
        # initialize random centroids and assignments
        self.n_samples = X.shape[0]
        self.init_centroids()
        self.init_assign()
        
        # loop through the assignment and update steps
        for j in range(max_epochs):
            self.inertia_ = 0
            self.update_centroids(X)
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
                print(self.centroids)
        self.labels = np.argmax(self.assignments, axis=1)
        self.cluster_var(X)
        return self.assignments
    
    def predict(self, X):
        """
        Predict class labels for given data points by nearest centroid rule
        Parameters
        ----------
        X : array, shape (n_samples, n_features). Observations to be assigned to the
        class represented by the nearest centroid.
        """
        # zero out current cluster assignment
        n = X.shape[0]
        labels = np.zeros((n, self.n_clusters))
        # find closest centroid (in Poincare disk)
        for i in range(n):
            centroid_distances = np.array([poincare_dist(X[i], centroid) for centroid in self.centroids])
            cx = np.argmin(centroid_distances)
            labels[i][cx] = 1
        return labels