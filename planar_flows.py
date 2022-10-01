from distutils.dist import Distribution
from turtle import forward
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import datasets_dict
from torch.distributions import MultivariateNormal, Distribution
import torch.nn.functional as F
        
class PlanarFlowTransform(nn.Module):
    """
    Invertible transformation:
    f(z) = z + u * h(w.T*z + b)
    
    Args: 
    dim: we will you dimension equal to 2 as our datasets are 2-D
    """
    def __init__(self, dim=2):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, dim).normal_(0, 0.1))
        self.b = nn.Parameter(torch.randn(1).normal_(0, 0.1))
        self.w = nn.Parameter(torch.randn(1, dim).normal_(0, 0.1))
        self.h = nn.Tanh() # h(x)=tanh(x)
        
    
    def invertibility(self):
        u = self.u
        w = self.w
        w_dot_u = torch.mm(u, w.t())
        if w_dot_u.item() >= -1.0:
            return

        norm_w = w / torch.norm(w, p=2, dim=1)**2
        bias = -1.0 + F.softplus(w_dot_u)
        u = u + (bias - w_dot_u) * norm_w
        self.u.data = u
        
    
    def forward(self, z):
        self.invertibility()
        f_z = z + self.u * nn.Tanh()(torch.mm(z, self.w.T) + self.b) # h is tanh as mentioned in the appendix of the paper
        psi_z = (1 - self.h(torch.mm(z, self.w.T) + self.b) ** 2) * self.w  # psi_z = h'(w_T z + b)w, where h'(x) = 1 - h(x)^2
        det = (1 + torch.mm(self.u, psi_z.T)).abs()
        logdet = torch.log(det)
        
        return f_z, logdet
    
    '''
    def backward(self, z):
        w_dot_z = torch.mm(z, self.w.t())
        w_dot_u = torch.mm(self.u, self.w.t())

        lo = torch.full_like(w_dot_z, -1.0e3)
        hi = torch.full_like(w_dot_z, 1.0e3)

        for _ in range(100):
            mid = (lo + hi) * 0.5
            val = mid + w_dot_u * torch.tanh(mid + self.b)
            lo = torch.where(val < w_dot_z, mid, lo)
            hi = torch.where(val > w_dot_z, mid, hi)

            if torch.all(torch.abs(hi - lo) < 1.0e-5):
                break

        affine = (lo + hi) * 0.5 + self.b
        z = z - self.u * torch.tanh(affine)
        #det = 1.0 + w_dot_u * deriv_tanh(affine)
        #log_df_dz = log_df_dz - torch.sum(torch.log(torch.abs(det) + 1.0e-5), dim=1)

        return z #, log_df_dz
    '''
    
     
class PlanarFlowModel(nn.Module):
    def __init__(self, q0 , dim = 2, K = 6):
        """Make a planar flow by stacking planar transformations in sequence.
        Args:
            dim: Dimensionality of the distribution to be estimated.
            K: Number of transformations in the flow. 
        """
        super().__init__()
        self.q0 = q0
        self.layers = [PlanarFlowTransform(dim) for _ in range(K)]
        self.model = nn.Sequential(*self.layers)
        
    
    def forward(self, x):
        sum_logdet = 0

        for layer in self.layers:
            z, logdet = layer(x)
            #sum_logdet += layer.log_det_J(x)
            #z, logdet = self.model(x)
            sum_logdet += logdet
            

        return z, sum_logdet # transformed input, Jacobian log-determinants evaluated at z
    '''
    def backward(self, base_sample):
        for layer in reversed(self.layers):
            x = layer.backward(base_sample)
        
        return x
    '''
        