import torch
from scipy.optimize import nnls
import numpy as np


def FCLSU(Y, A):

    
    Y = Y.numpy()
    A = A.numpy()
    N, T = A.shape[1], Y.shape[1]
    A = np.vstack((A/1000, np.ones(N)))
    Y = np.vstack((Y/1000, np.ones(T)))
    S = np.zeros((N, T))

    for i in range(T):
        S[:, i], _ = nnls(A, Y[:, i])

    return torch.from_numpy(S).float()