import numpy as np
import scipy.io as scio
import torch

def loadhsi(case):
    '''
    :input: case: for different datasets,
                 'toy' and 'usgs' are synthetic datasets
    :return: Y : HSI data of size [Bands,N]
             A_ture : Ground Truth of abundance map of size [P,N]
             P : nums of endmembers signature
    '''

    if case == 'ridge':
        file = 'PGMSU/dataset/JasperRidge2_R198.mat'
        data = scio.loadmat(file)
        Y = data['Y']
        nRow, nCol = data['nRow'][0][0], data['nCol'][0][0]
        if np.max(Y) > 1:
            Y = Y / np.max(Y)       
        Y = np.reshape(Y,[198,100,100])
        for i,y in enumerate(Y):
            Y[i]=y.T
        Y = np.reshape(Y, [198, 10000])

        GT_file = 'PGMSU/dataset/JasperRidge2_end4.mat'
        S_gt = scio.loadmat(GT_file)['A']
        A_gt = scio.loadmat(GT_file)['M']
        S_gt = np.reshape(S_gt, (4, 100, 100))
        for i,A in enumerate(S_gt):
            S_gt[i]=A.T
        S_gt = np.reshape(S_gt, (4, 10000))

    elif case == 'cuprite':
        file = 'dataset/Cuprite/CupriteS1_R188.mat'
        data = scio.loadmat(file)
        Y = data['Y']
        SlectBands = data['SlectBands'].squeeze()
        nRow, nCol = data['nRow'][0][0], data['nCol'][0][0]

        GT_file = 'dataset/Cuprite/groundTruth_Cuprite_nEnd12.mat'
        A_gt = scio.loadmat(GT_file)['M'][SlectBands,:]
        # GT_file = 'dataset/Cuprite/AVIRIS_corrected (MoffettField).mat'

        Y = np.delete(Y, [0,1,135], axis=0)
        A_gt = np.delete(A_gt, [0,1,135], axis=0)
        if np.max(Y) > 1:
            Y = Y / np.max(Y)
    

    elif case == 'urban':
        file = 'dataset/urban/Urban_R162.mat'
        data = scio.loadmat(file)
        Y = data['Y']  # (C,w*h)
        nRow, nCol = data['nRow'][0][0], data['nCol'][0][0]

        GT_file = 'dataset/urban/end5_groundTruth.mat'
        S_gt = scio.loadmat(GT_file)['A']
        A_gt = scio.loadmat(GT_file)['M']
        if np.max(Y) > 1:
            Y = Y / np.max(Y)

    Y = torch.tensor(Y).float()
    A_gt = torch.tensor(A_gt).float()
    if 'S_gt' in locals():
        S_gt = torch.tensor(S_gt).float()
    else:
        S_gt = torch.ones((A_gt.shape[1], Y.shape[1]))

    return Y, A_gt, S_gt, nRow, nCol