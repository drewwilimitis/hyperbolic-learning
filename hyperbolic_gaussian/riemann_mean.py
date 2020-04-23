# import libraries
import numpy as np
import sys

# import modules within repository
sys.path.append('C:\\Users\\dreww\\Desktop\\hyperbolic-learning\\utils') # path to utils folder
from utils import *

#-------------------------------------------------------------------
#----- Riemannian Barycenter Optimization in Hyperboloid Model -----
#-------------------------------------------------------------------

def exp_map(v, theta_k, eps=1e-6):
    """ Exponential map that projects tangent vector v onto hyperboloid"""
    # v: vector in tangent space at theta_k
    # theta: parameter vector in hyperboloid with centroid coordinates
    # project vector v from tangent minkowski space -> hyperboloid"""
    return np.cosh(norm(v))*theta_k + np.sinh(norm(v)) * v / (norm(v) + eps)

def minkowski_distance_gradient(u, v):
    """ Riemannian gradient of hyperboloid distance w.r.t point u """ 
    # u,v in hyperboloid
    return -1*(hyperboloid_dot(u,v)**2 - 1)**-1/2 * v

def minkowski_loss_gradient(theta_k, X, w):
    """ Riemannian gradient of error function w.r.t theta_k """
    # X : ALL data x1, ..., xN (not just within clusters like K-means) - shape N x 1
    # theta_k: point in hyperboloid at cluster center
    # w: vector with weights w_1k, ..., w_Nk - shape N x 1
    # returns gradient vector
    weighted_distances = w*np.array([-1*hyperboloid_dist(theta_k, x) for x in X]) # scalar
    distance_grads = np.array([minkowski_distance_gradient(theta_k, x) for x in X]) # list of vectors
    grad_loss = 2*np.sum(weighted_distances*distance_grads, axis=0) # summing along list of vectors
    if np.isnan(grad_loss).any():
        #print('Hyperboloid dist returned nan value')
        return eps
    else:
        return grad_loss

def project_to_tangent(theta_k, minkowski_grad):
    """ 
    Projects vector in ambient space to hyperboloid tangent space at theta_k 
    Note: returns our hyperboloid gradient of the error function w.r.t theta_k
    """
    # minkowski_grad: riemannian gradient vector in ambient space
    # theta_k: point in hyperboloid at cluster center
    return minkowski_grad + hyperboloid_dot(theta_k, minkowski_grad)*theta_k

def update_step(theta_k, hyperboloid_grad, alpha=0.1):
    """ 
    Apply exponential map to project the gradient and obtain new cluster center
    Note: returns updated theta_k
    """
    # theta_k: point in hyperboloid at cluster center
    # hyperboloid_grad: hyperboloid gradient in tangent space
    # alpha: learning rate > 0
    new_theta_k = exp_map(-1*alpha*hyperboloid_grad, theta_k)
    return 

def barycenter_loss(theta_k, X, w):
    """ Evaluate barycenter loss for a given gaussian cluster """
    # X : ALL data x1, ..., xN (not just within clusters like K-means) - shape N x 1
    # theta_k: parameter matrix with cluster center points - 1 x n
    # w: weights w_1k, ..., w_Nk - shape N x 1
    distances = np.array([hyperboloid_dist(theta_k, x)**2 for x in X])
    weighted_distances = w * distances
    loss = np.sum(weighted_distances)
    return loss

def overall_loss(theta, X, W):
    """ Evaluate barycenter loss for a given gaussian cluster """
    # X : ALL data x1, ..., xN (not just within clusters like K-means) - shape N x 1
    # theta: parameter matrix with cluster center points - k x n
    # W: matrix with weights w_1k, ..., w_Nk - shape k x N
    loss = 0
    K = W.shape[1]
    for i in range(K):
        distances = np.array([hyperboloid_dist(theta[i], x)**2 for x in X])
        weighted_distances = W[i, :] * distances
        loss += np.sum(weighted_distances)
    return loss

def weighted_barycenter(theta_k, X, w, num_rounds = 10, alpha=0.3, tol = 1e-4, verbose=False):
    """ Estimate weighted barycenter for a gaussian cluster with optimization routine """
    # X : ALL data x1, ..., xN (not just within clusters like K-means) - shape N x 1
    # theta_k: parameter matrix with cluster center points - k x n
    # w: weights w_1k, ..., w_Nk - shape N X 1
    # num_rounds: training iterations
    # alpha: learning rate
    # tol: convergence tolerance, exit if updates smaller than tolerance
    centr_pt = theta_k
    centr_pts = [theta_k]
    losses = []
    for i in range(num_rounds):
        gradient_loss = minkowski_loss_gradient(centr_pt, X, w)
        tangent_grad = project_to_tangent(centr_pt, -gradient_loss)
        centr_pt = update_step(centr_pt, tangent_grad, alpha=alpha)
        centr_pts.append(centr_pt)
        losses.append(barycenter_loss(centr_pt, X, w))
        if verbose:
            print('Epoch ' + str(i+1) + ' complete')
            print('Loss: ', barycenter_loss(centr_pt, X, w))
            print('\n')
        if hyperboloid_dist(centr_pts[i+1], centr_pts[i]) < tol:
            break
    return centr_pt