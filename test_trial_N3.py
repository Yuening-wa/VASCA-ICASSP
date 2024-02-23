import torch
import numpy as np
import os
import torch.nn.functional as F
# from torchvision.utils import save_image
import metric
# import visual
import SPA
from FCLSU import FCLSU
from model_VAE import VAE
from data_gen import DataGenerator
# from matplotlib import pyplot as plt
import PRISM
import PRISM_VAE
import time
import scipy.io
from scipy.io import loadmat
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def evaluation(A_gt, A_est, S_gt, S_est, plot_flag=False):
    _, col_order = metric.reorder_columns_angle(A_gt, A_est)
    mse_A = metric.mse(A_gt, A_est)
    sam_A = metric.sam(A_gt, A_est)
    rmse_S = metric.rmse(S_gt, S_est[col_order, :])
    print('MSE= %.2f, SAM= %.2f, rmse_S= %.3f' % (mse_A, sam_A, rmse_S))
    if plot_flag:
        S_est = S_est.detach().cpu().numpy()
        S_gt = S_gt.detach().cpu().numpy()
        # plt.scatter(S_gt[0,:], S_gt[1,:])
        # plt.scatter(S_est[0,:], S_est[1,:])
    return mse_A, sam_A, rmse_S


if __name__ == '__main__':

    print("Loading the USGS.mat file")
    A_lib = torch.tensor(loadmat('USGS.mat')['datalib']).float()
    print("Loaded")

    num_trail, num_case, alg_num = 100, 6, 3
    N, M, T = 3, 50, 10000
    SNR_list = [10, 15, 20, 25, 30, 35]

    generator = DataGenerator()
    sam_record = torch.zeros((num_trail, num_case, alg_num))
    mse_record = torch.zeros((num_trail, num_case, alg_num))
    rmse_record = torch.zeros((num_trail, num_case, alg_num))
    time_record = torch.zeros((num_trail, num_case, alg_num))
    elbo_record = torch.zeros((num_trail, num_case, 2))

    for trial in range(num_trail):
        torch.manual_seed(trial)
        np.random.seed(2)
        print('*'*20, 'trial: %d' % trial, '*'*20)
        A_gt, S_gt, Y_clean = generator.create(A_lib, M, N, 2*T)
        S_gt = S_gt[:, torch.all(S_gt<0.8, dim=0)][:, :T]
        Y_clean = A_gt @ S_gt

        for case, SNR in enumerate(SNR_list): 
            Y, sigv2 = generator.add_noise(Y_clean, SNR)
            print('-'*10, 'SNR= %d' % SNR, '-'*10)

            # initialization
            tic = time.time()
            A_init = SPA.SVMAX(Y, N)
            A_init = np.clip(A_init, 0.01, None)

            # reorder columns of A_init as descending order of mean
            reorder = torch.argsort(A_init.mean(dim=0), descending=True)
            A_init = A_init[:, reorder]
            S_init = FCLSU(Y, A_init)
            time_init = time.time()-tic
            print('init SVMAX: (time: %.2f s)' % (time_init))

            Y, A_gt, S_gt, sigv2 = Y.cuda(), A_gt.cuda(), S_gt.cuda(), sigv2.cuda()
            A_init, S_init = A_init.cuda(), S_init.cuda()
            plot_flag, show_flag = False, False

            mse_init, sam_init, rmse_init = evaluation(A_gt, A_init, S_gt, S_init, plot_flag=False)

            # ---- PRISM
            start = time.time()
            A_prism, S_prism, _, elbo_prism = PRISM.train(Y, A_init, sigv2, A_gt, S_gt, \
                num_iter=300, lr=5e-2, show_flag=show_flag)
            time_prism = time.time()-start
            print('PRISM: ELBO=%.2f, (time: %.2f s)' % (elbo_prism, time_prism))
            mse_prism, sam_prism, rmse_prism = evaluation(A_gt, A_prism, S_gt, S_prism)

            # ---- VAE
            start = time.time()
            model = VAE(input_size=M, h1_dim=32, h2_dim=32, h3_dim=16, h4_dim=8, \
                        z_dim=N, A_init=A_init, num_samples=1).cuda()  
            num_epochs = 500
            batch_size = T // 5
            lr_encoder, lr_decoder = 1e-2, 1e-3
            # train
            A_vae, S_vae,_,_,elbo_vae,_ = PRISM_VAE.train(model, batch_size, lr_decoder, lr_encoder,  \
                num_epochs, A_gt, S_gt, Y, sigv2, plot_flag, show_flag=show_flag)
            time_vae = time.time()-start
            print('VAE: ELBO=%.2f, (time: %.2f s)' % (elbo_vae, time_vae))
            try:
                mse_vae, sam_vae, rmse_vae = evaluation(A_gt, A_vae, S_gt, S_vae)
            except Exception:
                mse_vae, sam_vae, rmse_vae = np.nan, np.nan, np.nan            
            
            # save results
            sam_record[trial, case, :] = torch.tensor([sam_init, sam_prism, sam_vae]).detach().cpu()
            mse_record[trial, case, :] = torch.tensor([mse_init, mse_prism, mse_vae]).detach().cpu()
            rmse_record[trial, case, :] = torch.tensor([rmse_init, rmse_prism, rmse_vae]).detach().cpu()
            time_record[trial, case, :] = torch.tensor([time_init, time_prism, time_vae]).detach().cpu()
            elbo_record[trial, case, :] = torch.tensor([elbo_prism, elbo_vae]).detach().cpu()

            torch.cuda.empty_cache()
        
        # scipy.io.savemat('results_N3_0912_s08_SNR.mat', {'sam_record': sam_record.numpy(), 'mse_record': mse_record.numpy(), \
        #    'rmse_record': rmse_record.numpy(), 'time_record': time_record.numpy(), 'elbo_record': elbo_record.numpy()})
    
    pass
 

 