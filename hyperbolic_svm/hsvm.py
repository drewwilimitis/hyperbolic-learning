# import libraries
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import sys
import logging
logging.basicConfig(level=logging.INFO)

# import modules within repository
my_path = 'C:\\Users\\dreww\\Desktop\\hyperbolic-learning\\utils' # path to utils folder
sys.path.append(my_path)
from utils import *

def loss_fn(w, X, y, C=1.0):
    """
    Compute loss function for HSVM maximum margin formulation
    """
    margin_loss = -1/2 * minkowski_dot(w, w)
    misclass_loss = np.arcsinh(1) - np.arcsinh(y *[minkowski_dot(w, x) for x in X])
    return margin_loss + C*np.sum(np.where(misclass_loss > 0, misclass_loss, 0))

def grad_fn(w, X, y, C=1.0):
    """
    Solve for gradient w.r.t w for loss function L(w, x, y)
    """
    #if len(y.shape) < 2:
    #    y = y.reshape(-1, 1)
    grad_margin = np.hstack((-w[0], w[1:]))
    z = y*np.array([minkowski_dot(w, x) for x in X])
    misclass = (np.arcsinh(1) - np.arcsinh(z)) > 0
    arcsinh_term = -1/np.sqrt(z**2 + 1)
    mink_prod_term = y.reshape(-1,1)*np.hstack((X[:, 0].reshape(-1,1), -1*X[:, 1:]))
    grad_misclass = misclass.reshape(-1,1) * arcsinh_term.reshape(-1,1) * mink_prod_term
    grad_w = grad_margin + C*np.sum(grad_misclass, axis=0)
    return grad_w

def is_feasible(w):
    norm_cond = minkowski_dot(w, w) < 0
    return norm_cond

def proj_boundary(w, alpha, eps=1e-2):
    proj_w = w.copy()
    proj_w[1:] = (1 + alpha)*proj_w[1:]
    proj_w[0] = np.sqrt(np.sum(proj_w[1:]**2) - eps)
    return proj_w

def alpha_search(w):
    res = sp.optimize.minimize_scalar(lambda alpha: np.sum((proj_boundary(w, alpha) - w)**2))
    alpha = res.x
    return alpha