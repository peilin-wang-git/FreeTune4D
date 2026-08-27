# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 20:40:22 2020

@author: Haonan Xiao

E-mail: hx42@duke.edu
"""


import os

# external imports
import numpy as np
import torch
from torch.optim import Adam

# internal imports
from model import cvpr2018_net, SpatialTransformer
from model_C2F import Unet
import torch.nn.functional as F
import datagenerators
import losses
from math import floor
import time

def train_v5_RTS(gpu,
          cwd,
          vol_size,
          nf_enc_coarse,
          nf_dec_coarse,
          nf_enc_fine,
          nf_dec_fine,
          data_folder,
          lr,
          n_epoch,
          params, 
          batch_size,
          n_save_epoch,
          model_dir,
          lr_schedule):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param lr: learning rate
    :param n_iter: number of training iterations
    :param data_loss: data_loss: 'mse' or 'ncc
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param params: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
                    [reg_param, grad_param, lung_param, DICE_param, top_param]
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param n_save_iter: Optional, default of 500. Determines how many epochs before saving model version.
    :param model_dir: the model directory to save to
    """
    reg_param = params[0]
    grad_param = params[1]
    RTS_param = params[2]
    coarse_param = params[3]
    fine_param = params[4]
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    
    model_coarse = Unet(nf_enc_coarse, nf_dec_coarse, 2)
    model_fine = Unet(nf_enc_fine, nf_dec_fine, 1)
    ST = SpatialTransformer(vol_size) 
    
    model_coarse.to(device)
    model_fine.to(device)
    ST.to(device)
    
    
    # Set optimizer and losses
    params = list(model_coarse.parameters()) + list(model_fine.parameters())
    opt = Adam(params, lr=lr)
#    opt_coarse = Adam(model_coarse.parameters(), lr=lr)
#    opt_fine = Adam(model_fine.parameters(), lr=lr)

#    mse_loss_fn = losses.mse_loss_ReallynoRTS
    cc_loss_fn = losses.mse_loss_ReallynoRTS
    reg_loss_fn = losses.regulation_loss
    grad_loss_fn = losses.gradient_loss

    # data generator
    studies = os.listdir(data_folder)
    train_example_gen = datagenerators.gen_CT_RTS_v5(cwd, studies, data_folder)

#    loss_c_out = np.array([])
#    mse_loss_c_out = np.array([])
    cc_loss_c_out = np.array([])
#    loss_f_out = np.array([])
#    mse_loss_f_out = np.array([])
    cc_loss_f_out = np.array([])
    reg_loss_out = np.array([])
    grad_loss_out = np.array([])
    RTS_loss_out = np.array([])

    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt = Adam(params, lr=lr_new)
        # Save model checkpoint
        if i % n_save_epoch == 0:
            save_file_name_coarse = os.path.join(model_dir, 'coarse%d.ckpt' % i)
            save_file_name_fine = os.path.join(model_dir, 'fine%d.ckpt' % i)
            torch.save(model_coarse.state_dict(), save_file_name_coarse)
            torch.save(model_fine.state_dict(), save_file_name_fine)
            
        cc_loss_c_accu = np.array([])
        cc_loss_f_accu = np.array([])
        reg_loss_accu = np.array([])
        grad_loss_accu = np.array([])
        RTS_loss_accu = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
#        moving_patch, fixed_patch = next(train_example_gen)
        for num_patch in range(100):
            moving_patch, fixed_patch, moving_RTS, fixed_RTS = next(train_example_gen)
            start = time.time()
            input_A = torch.from_numpy(moving_patch).to(device).float()
            input_A[input_A>-200] = 0
            # -1000 - -200 ?  
#            input_A = (input_A+1000)/1000
            # 0 - 0.2
            input_A = F.interpolate(input_A, vol_size, mode = 'trilinear')
            A_RTS = torch.from_numpy(moving_RTS).to(device).float()
            A_RTS = F.interpolate(A_RTS, vol_size, mode = 'trilinear')
#            A_RTS = torch.round(A_RTS)
            
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B[input_B>-200] = 0
#            input_B = (input_B+1000)/1000
            input_B = F.interpolate(input_B, vol_size, mode = 'trilinear')
            B_RTS = torch.from_numpy(fixed_RTS).to(device).float()
            B_RTS = F.interpolate(B_RTS, vol_size, mode = 'trilinear')       
#            B_RTS = torch.round(B_RTS)
            
            # Run the data through the model to produce warp and flow field
            # Coarse network x2
            flow_c, flow_c_up = model_coarse(input_A, input_B)
            warp_c = ST(input_A, flow_c_up)
#            mse_loss_c = mse_param * mse_loss_fn(warp_c, input_B)
            cc_loss_c = coarse_param * cc_loss_fn(input_B, warp_c)
#            loss_c = mse_loss_c + cc_loss_c
            
#             Fine network x1
            flow_f, flow_f_up = model_fine(warp_c, input_B)
            warp_f = ST(warp_c.detach(), flow_f_up)
#            mse_loss_f = mse_param * mse_loss_fn(warp_f, input_B)
            cc_loss_f = fine_param * cc_loss_fn(input_B, warp_f)
#            loss_f = mse_loss_f + cc_loss_f
            
            dvf = flow_c_up + flow_f_up
            reg_loss = reg_param * reg_loss_fn(dvf)
            grad_loss = grad_param * grad_loss_fn(dvf)
            
            warp_RTS = ST(A_RTS,dvf)
            RTS_loss = RTS_param * cc_loss_fn(warp_RTS,B_RTS)
            
            loss = cc_loss_c + cc_loss_f + reg_loss + grad_loss + RTS_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            
            # accu loss
            cc_loss_c_accu = np.append(cc_loss_c_accu, cc_loss_c.item())
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())
            cc_loss_f_accu = np.append(cc_loss_f_accu, cc_loss_f.item())
            RTS_loss_accu = np.append(RTS_loss_accu, RTS_loss.item())
            
            if (i == 0) & (num_patch == 0):
                print('saved\n')
                cc_loss_c_out = np.append(cc_loss_c_out, np.mean(cc_loss_c_accu))
                cc_loss_f_out = np.append(cc_loss_f_out, np.mean(cc_loss_f_accu))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))
                RTS_loss_out = np.append(RTS_loss_out, np.mean(RTS_loss_accu))

            end = time.time()
            print("\r n_iter: %d/100, time = %fs, lr: %e, RTS:%f, cc_c: %f, cc_f: %f, reg:%f, grad:%f" 
              % (num_patch + 1, end-start, lr_new, np.mean(RTS_loss_accu), np.mean(cc_loss_c_accu),
                 np.mean(cc_loss_f_accu), np.mean(reg_loss_accu), np.mean(grad_loss_accu),
                 ), end = '', flush=True)
        cc_loss_c_out = np.append(cc_loss_c_out, cc_loss_c_accu)
        cc_loss_f_out = np.append(cc_loss_f_out, cc_loss_f_accu)
        reg_loss_out = np.append(reg_loss_out, reg_loss_accu)
        grad_loss_out = np.append(grad_loss_out, grad_loss_accu)
        RTS_loss_out = np.append(RTS_loss_out, RTS_loss_accu)
    return model_coarse,model_fine,RTS_loss_out,cc_loss_c_out,cc_loss_f_out,reg_loss_out,grad_loss_out

def train_v5_RTS_single(gpu,
          cwd,
          vol_size,
          nf_enc_coarse,
          nf_dec_coarse,
          nf_enc_fine,
          nf_dec_fine,
          data_folder,
          lr,
          n_epoch,
          params, 
          batch_size,
          n_save_epoch,
          model_dir,
          lr_schedule):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param lr: learning rate
    :param n_iter: number of training iterations
    :param data_loss: data_loss: 'mse' or 'ncc
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param params: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
                    [reg_param, grad_param, lung_param, DICE_param, top_param]
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param n_save_iter: Optional, default of 500. Determines how many epochs before saving model version.
    :param model_dir: the model directory to save to
    """
    reg_param = params[0]
    grad_param = params[1]
    RTS_param = params[2]
    coarse_param = params[3]
    fine_param = params[4]
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    
    model_fine = Unet(nf_enc_fine, nf_dec_fine, 1)
    ST = SpatialTransformer(vol_size) 
    
    model_fine.to(device)
    ST.to(device)
    
    
    # Set optimizer and losses
    opt = Adam(model_fine.parameters(), lr=lr)
#    opt_coarse = Adam(model_coarse.parameters(), lr=lr)
#    opt_fine = Adam(model_fine.parameters(), lr=lr)

#    mse_loss_fn = losses.mse_loss_ReallynoRTS
    cc_loss_fn = losses.cc_loss_ReallynoRTS
    reg_loss_fn = losses.regulation_loss
    grad_loss_fn = losses.gradient_loss

    # data generator
    studies = os.listdir(data_folder)
    train_example_gen = datagenerators.gen_CT_RTS_v5(cwd, studies, data_folder)

#    loss_c_out = np.array([])
#    mse_loss_c_out = np.array([])
    cc_loss_c_out = np.array([])
#    loss_f_out = np.array([])
#    mse_loss_f_out = np.array([])
    cc_loss_f_out = np.array([])
    reg_loss_out = np.array([])
    grad_loss_out = np.array([])
    RTS_loss_out = np.array([])

    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt = Adam(model_fine.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_epoch == 0:
            save_file_name_fine = os.path.join(model_dir, 'fine%d.ckpt' % i)
            torch.save(model_fine.state_dict(), save_file_name_fine)
            
        cc_loss_c_accu = np.array([])
        cc_loss_f_accu = np.array([])
        reg_loss_accu = np.array([])
        grad_loss_accu = np.array([])
        RTS_loss_accu = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
#        moving_patch, fixed_patch = next(train_example_gen)
        for num_patch in range(100):
            moving_patch, fixed_patch, moving_RTS, fixed_RTS = next(train_example_gen)
            start = time.time()
            input_A = torch.from_numpy(moving_patch).to(device).float()
            # -1000 - -200 ?  
            input_A = (input_A+1000)/1000
            # 0 - 0.2
            input_A = F.interpolate(input_A, vol_size, mode = 'trilinear')
            A_RTS = torch.from_numpy(moving_RTS).to(device).float()
            A_RTS = F.interpolate(A_RTS, vol_size, mode = 'trilinear')
#            A_RTS = torch.round(A_RTS)
            
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B = (input_B+1000)/1000
            input_B = F.interpolate(input_B, vol_size, mode = 'trilinear')
            B_RTS = torch.from_numpy(fixed_RTS).to(device).float()
            B_RTS = F.interpolate(B_RTS, vol_size, mode = 'trilinear')       
#            B_RTS = torch.round(B_RTS)
            
            # Run the data through the model to produce warp and flow field
            # Coarse network x2 
#             Fine network x1
            flow_f, flow_f_up = model_fine(input_A, input_B)
            warp_f = ST(input_A, flow_f_up)
#            mse_loss_f = mse_param * mse_loss_fn(warp_f, input_B)
            cc_loss_f = fine_param * cc_loss_fn(input_B, warp_f)
#            loss_f = mse_loss_f + cc_loss_f
            
            dvf = flow_f_up
            reg_loss = reg_param * reg_loss_fn(dvf)
            grad_loss = grad_param * grad_loss_fn(dvf)
            
            warp_RTS = ST(A_RTS,dvf)
            RTS_loss = RTS_param * cc_loss_fn(warp_RTS,B_RTS)
            
            loss = cc_loss_f + reg_loss + grad_loss + RTS_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            
            # accu loss
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())
            cc_loss_f_accu = np.append(cc_loss_f_accu, cc_loss_f.item())
            RTS_loss_accu = np.append(RTS_loss_accu, RTS_loss.item())
            
            if (i == 0) & (num_patch == 0):
                print('saved\n')
                cc_loss_f_out = np.append(cc_loss_f_out, np.mean(cc_loss_f_accu))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))
                RTS_loss_out = np.append(RTS_loss_out, np.mean(RTS_loss_accu))

            end = time.time()
            print("\r n_iter: %d/100, time = %fs, lr: %e, RTS:%f, cc_f: %f, reg:%f, grad:%f" 
              % (num_patch + 1, end-start, lr_new, np.mean(RTS_loss_accu), 
                 np.mean(cc_loss_f_accu), np.mean(reg_loss_accu), np.mean(grad_loss_accu),
                 ), end = '', flush=True)
        cc_loss_f_out = np.append(cc_loss_f_out, cc_loss_f_accu)
        reg_loss_out = np.append(reg_loss_out, reg_loss_accu)
        grad_loss_out = np.append(grad_loss_out, grad_loss_accu)
        RTS_loss_out = np.append(RTS_loss_out, RTS_loss_accu)
    return model_fine,RTS_loss_out,cc_loss_f_out,reg_loss_out,grad_loss_out
