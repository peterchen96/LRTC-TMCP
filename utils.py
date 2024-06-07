import os
import torch
import random
import logging
import datetime

import numpy as np

from scipy import io
from torch.nn import functional as F

def initial_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

'''
# --------------------------------------------
# Tensor Fold and Unfold Operations
# --------------------------------------------
'''
class TrafficTensor:
    def __init__(self, tensor, mode=0):
        self.data = tensor
        self.mode = mode
        self.dimk = self.data.shape[self.mode]
        self.dim_k = [self.data.shape[i] for i in range(len(self.data.shape)) if i != self.mode]

    def ten2mat(self):
        transpose_tensor = torch.moveaxis(self.data, self.mode, 0)
        return transpose_tensor.reshape(self.dimk, -1)

    def mat2ten(self, unfolding_mat):
        # return the unfolding_mat to the corrsponding tensor
        tensor = unfolding_mat.reshape(self.dimk, *self.dim_k)
        return torch.moveaxis(tensor, 0, self.mode)


'''
# --------------------------------------------
# Data Loading
# --------------------------------------------
'''
def load_data(dataset_name, dataroot):
    '''
    Load the dataset.
    Input:
        dataset_name: the name of dataset
        dataroot: the path of dataset
    Output:
        tensor: the original data, (locations, days, time_intervals)
    '''
    dataset_name += "-data-set"
    if dataset_name == "Hangzhou-data-set":
        file = os.path.join(dataroot, dataset_name, "tensor.mat")
        tensor = io.loadmat(file)['tensor']
    
    elif dataset_name == "PeMS-data-set":
        file = os.path.join(dataroot, dataset_name, "pems.npy")
        tensor = np.load(file).reshape(228, -1, 288)
    
    elif dataset_name == "Portland-data-set":
        file = os.path.join(dataroot, dataset_name, "volume.npy") # occupancy, speed and volume
        tensor = np.load(file).reshape(1156, -1, 96)
        
    elif dataset_name == "Seattle-data-set":
        file = os.path.join(dataroot, dataset_name, "tensor.npz")
        tensor = np.load(file)["arr_0"]
        
    try:
        tensor = torch.Tensor(tensor)
    except:
        tensor = torch.Tensor(tensor.astype(np.int16))
        
    return tensor


'''
# --------------------------------------------
# Missing Pattern Generation
# --------------------------------------------
'''
def missing_pattern(dense_tensor, ms, kind="random", block_window=12, seed=1000):
    initial_seed(seed)

    if kind == "random":
        binary_tensor = torch.round(torch.Tensor(np.random.rand(*dense_tensor.shape)) + 0.5 - ms)

    elif kind == "non-random":
        dim1, dim2, _ = dense_tensor.shape
        binary_tensor = torch.round(torch.Tensor(np.random.rand(dim1, dim2)) + 0.5 - ms)[:, :, None]

    elif kind == "blackout":
        dense_mat = dense_tensor.reshape(dense_tensor.shape[0], -1)
        T = dense_mat.shape[1]
        binary_blocks = np.round(np.random.rand(T // block_window) + 0.5 - ms)
        binary_mat = np.array([binary_blocks] * block_window).reshape(T, order="F")[None, :]
        binary_tensor = torch.Tensor(binary_mat.reshape(dense_tensor.shape[1], -1))[None, :, :]

    else:
        raise ValueError("Only 'random', 'non-random', and 'blackout' 3 kinds of missing patterns.")
    
    if kind == "blackout":
        # binary blocks used for showing the missing pattern
        return binary_tensor, binary_blocks
    else:
        return binary_tensor


'''
# --------------------------------------------
# Metrics
# --------------------------------------------
'''
def compute_rmse(var, var_hat):
    return torch.sqrt(torch.sum((var - var_hat) ** 2) / var.shape[0])

def compute_mape(var, var_hat):
    return torch.sum(torch.abs(var - var_hat) / var) / var.shape[0]


'''
# --------------------------------------------
# Faster SVD on CPU
# --------------------------------------------
'''
def svd_(mat):
    # faster SVD
    [m, n] = mat.shape
    try:
        if 2 * m < n:
            u, s, _ = torch.linalg.svd(mat @ mat.T, full_matrices=False)
            s = torch.sqrt(s)
            tol = n * torch.finfo(float).eps
            idx = torch.sum(s > tol)
            return u[:, :idx], s[:idx],  torch.diag(1/s[:idx]) @ u[:, :idx].T @ mat
        elif m > 2 * n:
            v, s, u = svd_(mat.T)
            return u, s, v
    except: 
        pass

    u, s, v = torch.linalg.svd(mat, full_matrices=False)
    return u, s, v
    

'''
# --------------------------------------------
# Shrinkage Operations
# --------------------------------------------
'''
def shrinkage(vec, params, p="soft"):
    if p == "hard":
        return hard_shrinkage(vec, params)
    
    elif p == "soft":
        return soft_shrinkage(vec, params)
    
    elif p == "firm":
        return firm_shrinkage(vec, params[0], params[1])
    
    else:
        try:
            return sp_shrinkage(vec, params, p)
        except:
            raise ValueError("Only 'hard', 'soft', 'firm', 'scad' or 'p < 1' 4 kinds of shrinkage functions.")

def hard_shrinkage(vec, lam):
    ss = F.relu(vec - lam)
    ss[ss > 0] += lam[ss > 0]

    return ss

def soft_shrinkage(vec, lam):
    return F.relu(vec - lam)

def firm_shrinkage(vec, lam, gamma):
    # vec >= 0
    if gamma <= 1:
        return hard_shrinkage(vec, lam)
    
    else:
        ss = gamma / (gamma - 1) * F.relu(vec - lam)
        ss[vec > gamma * lam] = vec[vec > gamma * lam]
    
        return ss
    
def sp_shrinkage(x, w, p, iter=5):
    # generalized soft-thresholding algorithm
    # inner iteration is supposed to be 5
    if torch.sum(w) == 0:
        return x
    else:
        tau = (2 * w * (1 - p)) ** (1 / (2 - p)) + w * p * (2 * w * (1 - p)) ** ((p - 1) / (2 - p))
        ans = F.relu(x - tau)

        ins = torch.where(ans > 0)
        try:
            ans[ins] += tau[ins]
            weight = w[ins]
        except:
            ans[ins] += tau
            weight = w

        x, y = [ans[ins].clone() for _ in range(2)]
        
        for _ in range(iter):
            ans[ins] = y - weight * p * x ** (p - 1)
            x = ans[ins].clone()
        
        return ans

    
'''
# --------------------------------------------
# logger
# --------------------------------------------
'''
def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def logger_info(logger_name, log_path='default_logger.log'):
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)

def logger_close(logger):
    # close the logger
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()

