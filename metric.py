import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


def reorder_columns_mse(A1, A2):
    # match how first can be derived by second, i.e., A1 = A2[:, order]
    # A1 should be ground truth; A2 should be the estimated matrix to be reordered
    
    # Calculate the MSE between each column in A and A_gt
    cost_matrix = ((A1[:, :, None] - A2[:, None, :])**2).mean(axis=0).detach().cpu().numpy()
    # Use the Hungarian algorithm to find the optimal assignment of columns
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    A_reordered = A2[:, col_ind]
    
    return A_reordered, col_ind



def reorder_columns_angle(A1, A2):
    # match how first can be derived by second, i.e., A1 = A2[:, order]
    # A1 should be ground truth; A2 should be the estimated matrix to be reordered
    
    # Normalize the columns of A1and A2
    A1_norm = A1.float()/ torch.linalg.norm(A1, dim=0)
    A2_norm = A2.float() / torch.linalg.norm(A2, dim=0)
    # Calculate the cosine of the angle between each column in A1 and A2
    cos_angle_matrix = torch.mm(A1_norm.T, A2_norm).detach()
    angle_matrix = torch.acos(torch.clamp(cos_angle_matrix, -1, 1))
    # Use the Hungarian algorithm to find the optimal assignment of columns
    _, col_ind = linear_sum_assignment(angle_matrix.cpu().numpy())
    A_reordered = A2[:, col_ind]

    return A_reordered, col_ind


def mse(A_gt, A_est):
    A_est, _ = reorder_columns_mse(A_gt, A_est)
    mse_A = torch.log10((A_est - A_gt).pow(2).mean()) * 10
    return mse_A

def sam(A_gt, A_est):
    A_est, _ = reorder_columns_angle(A_gt, A_est)
    sam_A = torch.acos((A_est * A_gt).sum(dim=0)/ (A_est.norm(dim=0) * A_gt.norm(dim=0))).mean() / np.pi * 90
    return sam_A

def rmse(S_gt, S_est):
    rmse_S = torch.sqrt( ((S_est - S_gt)**2).mean(dim=0) ).mean()
    return rmse_S