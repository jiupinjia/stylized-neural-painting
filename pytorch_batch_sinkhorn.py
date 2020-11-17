#!/usr/bin/env python
"""
sinkhorn_pointcloud.py

Discrete OT : Sinkhorn algorithm for point cloud marginals.

"""

import torch
from torch.autograd import Variable

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sinkhorn_normalized(x, y, epsilon, niter, mass_x=None, mass_y=None):

    Wxy = sinkhorn_loss(x, y, epsilon, niter, mass_x, mass_y)
    Wxx = sinkhorn_loss(x, x, epsilon, niter, mass_x, mass_x)
    Wyy = sinkhorn_loss(y, y, epsilon, niter, mass_y, mass_y)
    return 2 * Wxy - Wxx - Wyy

def sinkhorn_loss(x, y, epsilon, niter, mass_x=None, mass_y=None):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """

    # The Sinkhorn algorithm takes as input three variables :
    C = cost_matrix(x, y)  # Wasserstein cost function

    nx = x.shape[1]
    ny = y.shape[1]
    batch_size = x.shape[0]

    if mass_x is None:
        # assign marginal to fixed with equal weights
        mu = 1. / nx * torch.ones([batch_size, nx]).to(device)
    else: # normalize
        mass_x.data = torch.clamp(mass_x.data, min=0, max=1e9)
        mass_x = mass_x + 1e-9
        mu = (mass_x / mass_x.sum(dim=-1, keepdim=True)).to(device)

    if mass_y is None:
        # assign marginal to fixed with equal weights
        nu = 1. / ny * torch.ones([batch_size, ny]).to(device)
    else: # normalize
        mass_y.data = torch.clamp(mass_y.data, min=0, max=1e9)
        mass_y = mass_y + 1e-9
        nu = (mass_y / mass_y.sum(dim=-1, keepdim=True)).to(device)

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(2) + v.unsqueeze(1)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(2, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.

    for i in range(niter):
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).transpose(dim0=1, dim1=2)).squeeze()) + v

    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C, dim=[1, 2])  # Sinkhorn cost

    return torch.mean(cost)


def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(2)
    y_lin = y.unsqueeze(1)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
    return c
