import torch
import numpy as np
import os
import torch.nn.functional as F
from torchvision.utils import save_image
import metric
import SPA
from FCLSU import FCLSU
from model_VAE import VAE
from data_gen import DataGenerator
from loadHSI import loadhsi
import visual
from matplotlib import pyplot as plt
import PRISM
import PRISM_VAE
import time
import scipy.io
from scipy.io import loadmat
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def evaluation(A_gt, A_est, S_gt, S_est, plot_flag=False):
    _, col_order = metric.reorder_columns_angle(A_gt, A_est)
    mse_A = metric.mse(A_gt, A_est)
    sam_A = metric.sam(A_gt, A_est)
    rmse_S = metric.rmse(S_gt, S_est[col_order, :])
    print('MSE= %.2f, SAM= %.2f, rmse_S= %.3f' % (mse_A, sam_A, rmse_S))
    if plot_flag:
        S_est = S_est.detach().cpu().numpy()
        S_gt = S_gt.detach().cpu().numpy()
    return mse_A, sam_A, rmse_S


if __name__ == '__main__':
    torch.manual_seed(0)
    case = ['ridge', 'urban'][1]
    _, A_gt, S_gt, nRow, nCol = loadhsi(case)
    generator = DataGenerator()
    # low-pass filter for S to remove pure pixels
    S_gt = generator.wo_pure(S_gt)
    S_gt = generator.lowpass(S_gt, nRow, nCol)
    N, M, T = A_gt.shape[1], A_gt.shape[0], S_gt.shape[1]
    print('N=%d, M=%d, T=%d' % (N, M, T))
    SNR = 25
    
    # genearate semi-real data
    Y, sigv2 = generator.add_noise(A_gt @ S_gt, SNR)

    # initialization
    tic = time.time()
    A_init = SPA.SVMAX(Y, N)
    A_init = np.clip(A_init, 0.001, None)    
    S_init = FCLSU(Y, A_init)
    time_init = time.time()-tic
    # visual.compare_spectra(A_gt, A_init)
    A_init, pos_init = metric.reorder_columns_angle(A_gt, A_init)
    S_init = S_init[pos_init, :]


    Y, A_gt, S_gt, sigv2 = Y.cuda(), A_gt.cuda(), S_gt.cuda(), sigv2.cuda()
    A_init, S_init = A_init.cuda(), S_init.cuda()
    plot_flag = False
    mse_init, sam_init, rmse_init = evaluation(A_gt, A_init, S_gt, S_init, plot_flag=False)

    
    # ---- PRISM
    # start = time.time()
    # A_prism, S_prism, _, elbo_prism = PRISM.train(Y, A_init, sigv2, A_gt, S_gt, \
    #     num_iter=300, lr=5e-2, show_flag=False)
    # time_prism = time.time()-start
    # print('PRISM ELBO=%.2f, (time: %.2f s)' % (elbo_prism, time_prism))
    # mse_prism, sam_prism, rmse_prism = evaluation(A_gt, A_prism, S_gt, S_prism, plot_flag)
    # A_prism, pos_prism = metric.reorder_columns_angle(A_gt, A_prism)
    # S_prism = S_prism[pos_prism, :]

    # # ---- VAE
    reorder = torch.argsort(A_init.mean(dim=0), descending=True)
    start = time.time()
    model = VAE(input_size=M, h1_dim=128, h2_dim=64, h3_dim=32, h4_dim=16, \
                z_dim=N, A_init=A_init, num_samples=1).cuda()  
    num_epochs = 500
    train_num =  int(nRow) * int(nCol)  #250*190  #  307*270
    batch_size = int(train_num // 5)
    lr_encoder, lr_decoder = 5e-3, 1e-2
    A_vae, S_vae,_,_,elbo_vae, model = PRISM_VAE.train(model, batch_size, lr_decoder, lr_encoder,  \
        num_epochs, A_gt, S_gt, Y, sigv2, plot_flag, show_flag=False)
    time_vae = time.time()-start
    print('VAE: ELBO=%.2f, (time: %.2f s)' % (elbo_vae, time_vae))
    A_vae, pos_vae = metric.reorder_columns_angle(A_gt, A_vae)
    S_vae = S_vae[pos_vae, :]
    mse_vae, sam_vae, rmse_vae = evaluation(A_gt, A_vae, S_gt, S_vae)
    # visual.compare_spectra(A_gt.cpu(), A_vae.cpu().detach())
    torch.cuda.empty_cache()
    
    # plot 2 : Abundance maps
    S_gt = S_gt.reshape([N, nCol, nRow]).cpu().numpy()
    S_vae_plot = S_vae.reshape([N, nCol, nRow]).cpu().numpy()
    S_init = S_init.reshape([N, nCol, nRow]).cpu().numpy()
    # S_prism = S_prism.reshape([N, nCol, nRow]).cpu().numpy()
    
    alg = 4
    fig, ax = plt.subplots(alg, N)    
    for i in range(N):
        ax[0][i].imshow(S_gt[i].T, cmap='jet', interpolation='none')
        ax[0][i].axis('off')
        ax[1][i].imshow(S_init[i].T, cmap='jet', interpolation='none')
        ax[1][i].axis('off')
        # ax[2][i].imshow(S_prism[i].T, cmap='jet', interpolation='none')
        # ax[2][i].axis('off')
        ax[3][i].imshow(S_vae_plot[i].T, cmap='jet', interpolation='none')
        ax[3][i].axis('off')
    fig.subplots_adjust(wspace=0, hspace=0.1)
        # aaa.set_clim(vmin=0, vmax=1)
    

fig.savefig('S.png')

# scipy.io.savemat('results_semi-real.mat', {'S_vae': S_vae_plot, 'S_init': S_init, 'S_prism': S_prism, 'S_gt': S_gt, \
    # 'A_vae': A_vae.cpu().numpy(), 'A_init': A_init.cpu().numpy(), 'A_prism': A_prism.cpu().numpy(), 'A_gt': A_gt.cpu().numpy()})