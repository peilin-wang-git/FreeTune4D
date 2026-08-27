"""
*Preliminary* pytorch implementation.

data generators for voxelmorph
"""

import numpy as np
import sys
import os
from os. path import exists
import math
import random
import torch
import torch.nn.functional as F

def dvf_interp(dvf, vol_size):
    orig_shape = dvf.shape[2:]
    DVF_0 = F.interpolate(torch.unsqueeze(
        dvf[:, 0, :, :, :], 1), vol_size, mode='trilinear')
    DVF_1 = F.interpolate(torch.unsqueeze(
        dvf[:, 1, :, :, :], 1), vol_size, mode='trilinear')
    DVF_2 = F.interpolate(torch.unsqueeze(
        dvf[:, 2, :, :, :], 1), vol_size, mode='trilinear')
    DVF_0 = DVF_0 * (vol_size[0]/orig_shape[0])
    DVF_1 = DVF_1 * (vol_size[1]/orig_shape[1])
    DVF_2 = DVF_2 * (vol_size[2]/orig_shape[2])
    DVF = torch.cat([DVF_0, DVF_1, DVF_2], 1)
    return DVF

"""
Get a random number outside, and pick the corresponding data
Input: 
    random study index (list)
    list of paths (list)
    batch_size = 1
Output:
    [X, Y, DVF] in torch.Tensor        
"""
def gen_CrossValidation(path_list, study_index, batch_size = 1, vol_size = [128,128,64], device = 'cuda'):
    # Randomly choose a study
    X_CT_data = []
    Y_CT_data = []
    DVF_data = []
    # Load number of batch_size volumes
    for idx_s in study_index:
        # Check which group this study belongs to
        # Choose 2 phase volumes from the study, one for moving and
        # one for fixed
        files = os.listdir(path_list[idx_s])
        DVF_path = os.path.join(path_list[idx_s], files[0])
        DVF = np.load(DVF_path)
        DVF = np.transpose(DVF, [3,0,1,2])
        DVF = DVF[np.newaxis, ...]
        DVF = torch.from_numpy(DVF).to(device).float()
        DVF = dvf_interp(DVF, vol_size)
        DVF_data.append(DVF)
            
        CT_X_path = os.path.join(path_list[idx_s], files[2])
        X_CT = np.load(CT_X_path)
        X_CT = X_CT[np.newaxis, np.newaxis, ...]
        X_CT = torch.from_numpy(X_CT).to(device).float()
        X_CT = F.interpolate(X_CT, vol_size, mode='trilinear')
        X_CT_data.append(X_CT)
        
        CT_Y_path = os.path.join(path_list[idx_s], files[1])
        Y_CT = np.load(CT_Y_path)
        Y_CT = Y_CT[np.newaxis, np.newaxis, ...]
        Y_CT = torch.from_numpy(Y_CT).to(device).float()
        Y_CT = F.interpolate(Y_CT, vol_size, mode='trilinear')
        Y_CT_data.append(Y_CT)
    if batch_size > 1:
        return_X_CT = torch.cat(X_CT_data, 0)
        return_Y_CT = torch.cat(Y_CT_data, 0)
        return_DVF = torch.cat(DVF_data, 0)
    else:
        return_X_CT = X_CT_data[0]
        return_Y_CT = Y_CT_data[0]
        return_DVF = DVF_data[0]     
    yield ([return_X_CT, return_Y_CT, return_DVF])

"""
Get a random number outside, and pick the corresponding data
Input: 
    random study index (list)
    list of paths (list)
    batch_size = 1
Output:
    [X, Y, DVF] in torch.Tensor        
"""
def gen_CrossValidation_valid(studies, sty_index, batch_size=1):
    # Choose 2 phase volumes from the study, one for moving and
    X_CT_data = []
    Y_CT_data = []
    DVF_data = []
    # one for fixed
    files = os.listdir(studies[sty_index])
    DVF_path = os.path.join(studies[sty_index], files[0])
    DVF = np.load(DVF_path)
    DVF = np.transpose(DVF, [3,0,1,2])
    DVF = DVF[np.newaxis, ...]
    DVF_data.append(DVF)
    
    CT_X_path = os.path.join(studies[sty_index], files[2])
    X_CT = np.load(CT_X_path)
    X_CT = X_CT[np.newaxis, np.newaxis, ...]
    X_CT_data.append(X_CT)
    CT_Y_path = os.path.join(studies[sty_index], files[1])
    Y_CT = np.load(CT_Y_path)
    Y_CT = Y_CT[np.newaxis, np.newaxis, ...]
    Y_CT_data.append(Y_CT)
    
    return_X_CT = X_CT_data[0]
    return_Y_CT = Y_CT_data[0]
    return_DVF = DVF_data[0]     
    yield ([return_X_CT, return_Y_CT, return_DVF])

      
            
        
