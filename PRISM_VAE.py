import torch
import numpy as np
import os
import torch.nn.functional as F
import metric
# import visual
# from SPA import SPA
# from model_VAE import VAE
# from matplotlib import pyplot as plt
from torch.optim import lr_scheduler

def calc_posterior_N(mu, log_var, num_samples=100):
    std = torch.exp(log_var / 2)
    samples = mu.unsqueeze(-1) + torch.randn(*std.shape, num_samples, device=mu.device) * std.unsqueeze(-1)
    samples = F.softmax(samples, dim=1)
    mean = torch.mean(samples, dim=-1)
    del samples, std
    return mean.T
    
def calc_loss_N(x_reconst, x, mu, log_var, inv_Cov_0, sigv2):
    # reconst_loss = F.mse_loss(x_reconst, x.expand(x_reconst.size(0),-1,-1), reduction='sum') / x_reconst.size(0) / x_reconst.size(1) / sigv2
    if x_reconst.size() != x.size():
        reconst_loss = (x_reconst - x).pow(2).sum(dim=2).mean(dim=1).mean(dim=0) / sigv2
    else:
        reconst_loss = (x_reconst - x).pow(2).sum(dim=-1).mean(dim=0) / sigv2
    Cov = log_var.exp()
    kl_div =  (torch.diag(inv_Cov_0)*Cov).sum(dim=1).mean(dim=0) \
         + (mu @ inv_Cov_0 * mu).sum(dim=1).mean(dim=0) - log_var.sum(dim=1).mean(dim=0)
    neg_elbo = (reconst_loss + kl_div) / 2
    # neg_elbo = reconst_loss / 2
    return neg_elbo, kl_div/2

def calc_posterior(mu, log_var, num_samples=1000):
    std = torch.exp(log_var / 2)
    eps = torch.randn(num_samples, *std.shape, device=mu.device)
    samples = mu.unsqueeze(0) + eps * std.unsqueeze(0)
    samples = torch.cat((samples, torch.zeros(samples.shape[0], samples.shape[1], 1, device=mu.device)), dim=2)
    samples = F.softmax(samples, dim=-1)
    mean = torch.mean(samples, dim=0)
    del samples, eps
    return mean.T

def calc_loss(x_reconst, x, mu, log_var, sigv2, s_sample):
    # reconst_loss = F.mse_loss(x_reconst, x.expand(x_reconst.size(0),-1,-1), reduction='sum') / x_reconst.size(0) / x_reconst.size(1) / sigv2
    if x_reconst.size() != x.size():
        reconst_loss = (x_reconst - x).pow(2).sum(dim=2).mean(dim=1).mean(dim=0) / sigv2
    else:
        reconst_loss = (x_reconst - x).pow(2).sum(dim=-1).mean(dim=0) / sigv2
    # only consider one sample for s
    s_bar = torch.log(s_sample[:, 0:-1]/s_sample[:, -1].view(-1,1)) - mu
    H_s = log_var.sum(dim=1) + mu.shape[1]*torch.log(torch.tensor(2*np.pi)) \
        + 2*torch.log(s_sample).sum(dim=-1) + (s_bar**2 / log_var.exp()).sum(dim=-1)
    neg_elbo = reconst_loss/2 - H_s.mean()/2    
    return neg_elbo



def train(model, batch_size, lr_decoder, lr_encoder, num_epochs, A_gt, S_gt, Y, sigv2, plot_flag, show_flag):
    num_batches = Y.shape[1] // batch_size
    M, N = A_gt.shape
    # ---- for N dimensional LN
    # Cov_0 = ( 1.65*torch.diag(torch.ones((N,))) + 1.65*torch.ones((N,N)) ).cuda()
    # inv_Cov_0 = torch.inverse(Cov_0)
    # const = ( torch.logdet(Cov_0) - N + M * torch.log(torch.tensor(2*np.pi)*sigv2) ) / 2
    
    # ---- for N-1 dimensional LN
    E_prior = torch.lgamma(torch.tensor(N))
    const = M/2 * torch.log(torch.tensor(2*np.pi)*sigv2) - E_prior

    params = list(model.parameters())
    optimizer = torch.optim.Adam([
        {'params': model.fc_decoder.parameters(), 'lr': lr_decoder},
        {'params': params[0:-2], 'lr': lr_encoder} ])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    # optimizer = torch.optim.SGD([
    #     {'params': model.fc_decoder.parameters(), 'lr': lr_decoder},
    #     {'params': params[0:-2], 'lr': lr_encoder} ], momentum=0.9)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    # if plot_flag:
    #     fig, ax = plt.subplots(1, 2)
    #     for i in range(A_gt.shape[1]):
    #         ax[0].plot(A_gt.detach().cpu().numpy()[:, i], label=str(i+1))        
    elbo_record = []
    A_prev = torch.zeros_like(A_gt)
    for epoch in range(num_epochs):
        S_est, mu_est, log_var_est, elbo = [], [], [], []
        idx = torch.randperm(Y.shape[1])
        Y_shuffle = Y[:, idx]
        for i in range(num_batches):
            if i == num_batches - 1:
                x = Y_shuffle[:, i*batch_size:].T.float()
            else:
                x = Y_shuffle[:, i*batch_size: (i+1)*batch_size].T.float()
            x_reconst, A, mu, log_var, s_sample = model(x)
            loss = calc_loss(x_reconst, x, mu, log_var, sigv2, s_sample)
            # x_reconst, A, mu, log_var = model(x)
            # loss, _ = calc_loss_N(x_reconst, x, mu, log_var, inv_Cov_0, sigv2)
            optimizer.zero_grad()
            loss.backward()
            if epoch < 10:
                list(model.fc_decoder.parameters())[0].grad = torch.zeros_like(A)
            optimizer.step()

            mu_est.append(mu.detach())
            log_var_est.append(log_var.detach())
            elbo.append(loss.item())
            if show_flag:
                S_i = calc_posterior(mu.detach(), log_var.detach())
                S_est.append(S_i)
        
        scheduler.step()
        # show result
        elbo_record.append(-np.mean(elbo)-const.item())
        if show_flag:
            S_est = torch.cat(S_est, dim=1)
            S_est = S_est[:, torch.argsort(idx)]
            _, col_order = metric.reorder_columns_angle(A_gt, A)
            print("Epoch[{}/{}], -elbo: {:.2f}, mse_A: {:.2f}, sam_A: {:.2f}, rmse_S: {:.3f}"
                .format(epoch + 1, num_epochs, np.mean(elbo) + const, metric.mse(A_gt, A), \
                         metric.sam(A_gt, A), metric.rmse(S_gt, S_est[col_order, :])))
    
        A_diff = A - A_prev
        change_A = torch.sqrt(torch.sum(A_diff**2) / torch.sum(A_prev**2))
        # print('change_A: {:.5f}'.format(change_A.item()))
        if change_A < 1e-4 and epoch > 300:
            break
        A_prev = A

    # A, col_order = metric.reorder_columns_angle(A_gt, A)
    # if plot_flag:
    #     for i in range(A.shape[1]):
    #         ax[1].plot(A.detach().cpu().numpy()[:, i], label=str(i+1))        
        # plt.show()
    mu_est = torch.cat(mu_est, dim=0).T
    log_var_est = torch.cat(log_var_est, dim=0).T
    cov_est = log_var_est.exp()
    mu_est = mu_est[:, torch.argsort(idx)]
    cov_est = cov_est[:, torch.argsort(idx)]

    if not show_flag: 
        S_est = calc_posterior(mu_est.T, log_var_est.T)
        # S_est = S_est[:, torch.argsort(idx)]
        # elbo = torch.tensor(elbo).mean()
    # S_est = S_est[col_order, :]
    # save model parameter
    # torch.save(model.state_dict(), 'VAE_HU_state.ckpt')
    return A.detach(), S_est, mu_est, cov_est, elbo_record[-1], model 


def test(model, Y_test):
    Y_test = Y_test.T.float()
    with torch.no_grad():
        _, A, mu, log_var, _ = model(Y_test)
        S_est = calc_posterior(mu.detach(), log_var.detach())
    return A, S_est, mu.T, log_var.exp().T
