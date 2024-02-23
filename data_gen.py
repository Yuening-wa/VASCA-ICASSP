import torch
from torch.distributions.gamma import Gamma
from scipy.io import loadmat
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from torchvision.transforms.functional import gaussian_blur


class DataGenerator(object):

    def lowpass(self, S, nRow, nCol):
        S_3d = S.view(S.shape[0], nCol, nRow)
        S_lp = gaussian_blur(S_3d, kernel_size=9, sigma=2.0)
        S_lp = S_lp.view(S.shape[0], -1)
        return S_lp
    
    def wo_pure(self, S):
        S[:, torch.any(S>0.8, dim=0)] += 0.2
        S[S>0.8] = 0.8
        S = S / S.sum(dim=0)
        return S

    def gen_S(self, N, T, case='uniform'):
        # uniform Dirichlet
        if case == 'uniform':
            alpha = torch.ones(N,T)
            S = Gamma(alpha, torch.ones(N,T)).sample()
            S = S / S.sum(dim=0)  
        # logistic normal
        elif case == 'LN': 
            mu = torch.zeros(N)
            Cov = 1.65*torch.diag(torch.ones((N,))) + 1.65*torch.ones((N,N))
            S = torch.distributions.MultivariateNormal(mu, Cov).sample((T,))
            S = F.softmax(S, dim=1).T     
        return S
    
    def gen_A(self, A_lib, M, N):
        picked_index = torch.randint(low=0, high=A_lib.shape[1], size=(N,))
        # picked_index = torch.tensor([445, 257, 61])
        interval = A_lib.shape[0] // M
        A = A_lib[0:interval*M:interval, picked_index]
        while torch.linalg.cond(A) > 200.0:
            picked_index = torch.randint(low=0, high=A_lib.shape[1], size=(N,))
            A = A_lib[0:interval*M:interval, picked_index]
        cond_A = torch.linalg.cond(A)
        print("Condition number of A: %.2f" % cond_A)
        return A

    def create(self, A_lib, M=224, N=3, T=1000, case='uniform'):
        S = self.gen_S(N, T, case)
        A = self.gen_A(A_lib, M, N)
        Y = A @ S
        return A, S, Y


    def add_noise(self, Y, SNR=30):
        M, T = Y.shape
        sig = torch.sqrt( Y.pow(2).sum()/T / (torch.tensor(10.0).pow(SNR/10) * M) )
        noise = torch.randn(M, T, dtype=torch.float32, device=Y.device) * sig
        Y = Y + noise
        sig2 = torch.pow(sig, 2)
        
        # sig2 = torch.pow(torch.tensor(0.1194), 2)
        # S  = torch.tensor(loadmat('S.mat')['S']).float()
        # Y = torch.tensor(loadmat('Y_obs.mat')['Y_obs']).float()
        return Y, sig2