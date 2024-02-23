import torch
import numpy as np
import metric
import torch.nn as nn
from torch.optim import lr_scheduler


def S_cal(alpha):
    eta = alpha.sum(dim=0)
    eta_prod = eta * (1+eta)
    alp_eta_prod = alpha / eta_prod
    S = alpha / eta
    SST = torch.diag(alp_eta_prod.sum(dim=-1)) + alpha.mm(alp_eta_prod.T)
    H_s = torch.lgamma(alpha).sum(dim=0) - torch.lgamma(eta) \
         - ((alpha-1) * (torch.special.polygamma(0,alpha) - torch.special.polygamma(0,eta))).sum(dim=0)
    return S, SST, H_s.mean()


def E_step(Y, A, sig2, alpha):
    S, SST, H_s = S_cal(alpha)
    Elog_likelihood = torch.trace(A.T.mm(A).mm(SST)) / alpha.size(1) - 2 * (Y * A.mm(S)).sum(dim=0).mean()
    loss = Elog_likelihood/sig2/2 - H_s
    return loss


def M_step(Y, alpha):
    S, SST, _ = S_cal(alpha)
    A = Y.mm(S.T).mm(torch.inverse(SST))
    return A, S

def train(Y, A, sig2, A_gt, S_gt, num_iter, lr, show_flag):
    M, N = A.shape
    const = (Y**2).sum(dim=0).mean() / sig2 / 2 + M * torch.log(torch.tensor(2*np.pi*sig2)) / 2 - torch.lgamma(torch.tensor(N))
    alpha = nn.Parameter(10*torch.ones(N, Y.shape[1]).to(Y.device))
    optimizer = torch.optim.Adam([alpha], lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    maxLoop = 500
    elbo_record = []
    A_prev = A
    for iter in range(num_iter):
        # E-step
        maxLoop = maxLoop - 5 if maxLoop > 30 else maxLoop
        for i in range(maxLoop):
            loss = E_step(Y, A, sig2, alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            alpha.data = torch.clamp(alpha.data, min=1e-3)
        scheduler.step()

        # M-step
        # S = alpha.detach() / alpha.detach().sum(dim=0)
        A, S = M_step(Y, alpha.detach())
        
        # stop criterion
        A_diff = A - A_prev
        change_A = torch.sqrt(torch.sum(A_diff**2) / torch.sum(A_prev**2))
        if change_A < 1e-4 and iter > 30:
            break
        A_prev = A
        
        # evaluate
        elbo_record.append(-loss.item() - const.item())
        if show_flag:
            _, col_order = metric.reorder_columns_angle(A_gt, A)
            print("Iter[{}/{}], -elbo: {:.2f}, mse_A: {:.2f}, sam_A: {:.2f}, rmse_S: {:.3f}" \
                    .format(iter + 1, num_iter, loss.item() + const, metric.mse(A_gt, A), \
                        metric.sam(A_gt, A), metric.rmse(S_gt, S[col_order, :])))

    return A, S, alpha.detach(), elbo_record[-1]