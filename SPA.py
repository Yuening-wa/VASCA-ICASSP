import torch

def pca(Y, N):
    d = torch.mean(Y, dim=1).unsqueeze(1)
    Y_cen = Y - d
    _, C = torch.linalg.eigh(Y_cen @ Y_cen.T)
    ls = list(range(Y.shape[0]))[::-1]
    C = C[:, ls][:, :N-1]
    Y_red = torch.pinverse(C) @ Y_cen    
    return Y_red, C, d

def SVMAX(Y, N):
    # pca
    X, C, d = pca(Y, N)
    # svmax
    X_bar = torch.cat((X, torch.ones((1, X.shape[1]), device=X.device)), dim=0)
    A_est = torch.tensor([], device=X.device)
    idx = []
    P = torch.eye(N, device=X.device)
    for i in range(N):
        _, idx_i = torch.max(torch.sum((P @ X_bar) ** 2, dim=0), 0)
        idx.append(idx_i.item())
        A_est = torch.cat((A_est, X[:, idx_i].unsqueeze(1)), dim=1)
        F = torch.cat((A_est, torch.ones((1, A_est.shape[1]), device=X.device)), dim=0)
        P = torch.eye(N, device=X.device) - F @ torch.pinverse(F)
    A_est = C @ A_est + d
    return A_est


def SPA(Y, N):

    M = Y.shape[0]  # the spectral band
    P = torch.eye(M, device=Y.device)  # initialize the projection with unit matrix
    A_init = torch.zeros(M, N, device=Y.device)  # initialize the endmember matrix
    idx = []
    Y = Y.float()
    
    for i in range(N):
        _, idx_i = torch.max(torch.sum((P @ Y) ** 2, dim=0), 0)
        idx.append(idx_i.item())
        A_init[:, i] = Y[:, idx_i]
        # update the orthogonal projector
        P = P - torch.ger((P @ Y[:, idx_i]), ((P @ Y[:, idx_i]).T @ P)) / torch.norm(P @ Y[:, idx_i], 2) ** 2

    return A_init

