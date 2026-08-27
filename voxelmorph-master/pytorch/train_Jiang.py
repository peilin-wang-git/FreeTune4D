# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:11:48 2020

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
from model_C2F import Unet, JiangNet, JiangNet_Fine
import torch.nn.functional as F
import datagenerators
import losses
from math import floor
import time

def train(gpu,
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
    mse_param = params[2]
    coarse_param = params[3]
    fine_param = params[4]
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    
#    model_coarse = Unet(nf_enc_coarse, nf_dec_coarse, 2)
#    model_fine = Unet(nf_enc_fine, nf_dec_fine, 1)
    model_coarse = JiangNet_Fine(dim=3, scale=4, residual=False)
    model_fine = JiangNet_Fine(dim=3, scale=2, residual=True)
    ST = SpatialTransformer(vol_size) 
    
    model_coarse.to(device)
    model_fine.to(device)
    ST.to(device)
    
    
    # Set optimizer and losses
    opt_coarse = Adam(model_coarse.parameters(), lr=lr)
    opt_fine = Adam(model_fine.parameters(), lr=lr)

#    mse_loss_fn = losses.mse_loss_ReallynoRTS
    sim_loss_fn = losses.cc_loss_ReallynoRTS
    reg_loss_fn = losses.regulation_loss
    grad_loss_fn = losses.gradient_loss
    reg_xyloss_fn = losses.regulation_loss_xy
    grad_zloss_fn = losses.gradient_loss_z

    # data generator
    studies = os.listdir(data_folder)
    train_example_gen = datagenerators.gen_CT_RTS_v4(cwd, studies, data_folder, extreme = True)

#    loss_c_out = np.array([])
#    mse_loss_c_out = np.array([])
    cc_loss_c_out = np.array([])
#    loss_f_out = np.array([])
#    mse_loss_f_out = np.array([])
    cc_loss_f_out = np.array([])
    reg_loss_out = np.array([])
    grad_loss_out = np.array([])

    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt_coarse = Adam(model_coarse.parameters(), lr=lr_new)
        opt_fine = Adam(model_fine.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_epoch == 0:
            save_file_name_coarse = os.path.join(model_dir, 'coarse%d.ckpt' % i)
            save_file_name_fine = os.path.join(model_dir, 'fine%d.ckpt' % i)
            torch.save(model_coarse.state_dict(), save_file_name_coarse)
            torch.save(model_fine.state_dict(), save_file_name_fine)
#        loss_c_accu = np.array([])
#        mse_loss_c_accu = np.array([])
        cc_loss_c_accu = np.array([])
        
#        loss_f_accu = np.array([])
#        mse_loss_f_accu = np.array([])
        cc_loss_f_accu = np.array([])
        reg_loss_accu = np.array([])
        grad_loss_accu = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
#        moving_patch, fixed_patch = next(train_example_gen)
        for num_patch in range(100):
            moving_patch, fixed_patch = next(train_example_gen)
            start = time.time()
            input_A = torch.from_numpy(moving_patch).to(device).float()
            # -1000 - -200
            input_A = (input_A+1000)/4000
            # 0 - 0.2
            input_A = F.interpolate(input_A, vol_size, mode = 'trilinear')
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B = (input_B+1000)/4000
            input_B = F.interpolate(input_B, vol_size, mode = 'trilinear')
            
            # Run the data through the model to produce warp and flow field
            # Coarse network x2
            flow_c, flow_c_up = model_coarse(input_A, input_B)
            warp_c = ST(input_A, flow_c_up)
#            mse_loss_c = mse_param * mse_loss_fn(warp_c, input_B)
            sim_loss_c = coarse_param * sim_loss_fn(input_B, warp_c)
#            loss_c = mse_loss_c + cc_loss_c
            
#             Fine network x1
            flow_f, flow_f_up = model_fine(warp_c, input_B)
            warp_f = ST(warp_c.detach(), flow_f_up)
#            mse_loss_f = mse_param * mse_loss_fn(warp_f, input_B)
            sim_loss_f = fine_param * sim_loss_fn(input_B, warp_f)
#            loss_f = mse_loss_f + cc_loss_f
            
            dvf = flow_c_up + flow_f_up
            reg_loss = reg_param * reg_loss_fn(dvf)
            grad_loss = grad_param * grad_loss_fn(dvf)
            
            loss = sim_loss_c + sim_loss_f + reg_loss + grad_loss
            opt_coarse.zero_grad()
            opt_fine.zero_grad()
            loss.backward()
            opt_coarse.step()    
            opt_fine.step()

            
            # accu loss
#            loss_c_accu = np.append(loss_c_accu, loss_c.item())
#            mse_loss_c_accu = np.append(mse_loss_c_accu, mse_loss_c.item())
            cc_loss_c_accu = np.append(cc_loss_c_accu, sim_loss_c.item())
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())
#            loss_f_accu = np.append(loss_f_accu, loss_f.item())
#            mse_loss_f_accu = np.append(mse_loss_f_accu, mse_loss_f.item())
            cc_loss_f_accu = np.append(cc_loss_f_accu, sim_loss_f.item())

      
            if (i == 0) & (num_patch == 0):
                print('saved\n')
#                loss_c_out = np.append(loss_c_out, np.mean(loss_c_accu))
#                mse_loss_c_out = np.append(mse_loss_c_out, np.mean(mse_loss_c_accu))
                cc_loss_c_out = np.append(cc_loss_c_out, np.mean(cc_loss_c_accu))
#                loss_f_out = np.append(loss_f_out, np.mean(loss_f_accu))
#                mse_loss_f_out = np.append(mse_loss_f_out, np.mean(mse_loss_f_accu))
                cc_loss_f_out = np.append(cc_loss_f_out, np.mean(cc_loss_f_accu))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))

            end = time.time()
            print("\r n_iter: %d/100, time = %fs, lr: %e, sim_c: %f, sim_f: %f, reg:%f, grad:%f" 
              % (num_patch + 1, end-start, lr_new, np.mean(cc_loss_c_accu),
                 np.mean(cc_loss_f_accu), np.mean(reg_loss_accu), np.mean(grad_loss_accu),
                 ), end = '', flush=True)
#        loss_c_out = np.append(loss_c_out, loss_c_accu)
#        mse_loss_c_out = np.append(mse_loss_c_out, mse_loss_c_accu)
        cc_loss_c_out = np.append(cc_loss_c_out, cc_loss_c_accu)
#        loss_f_out = np.append(loss_f_out, loss_f_accu)
#        mse_loss_f_out = np.append(mse_loss_f_out, mse_loss_f_accu)
        cc_loss_f_out = np.append(cc_loss_f_out, cc_loss_f_accu)
        reg_loss_out = np.append(reg_loss_out, reg_loss_accu)
        grad_loss_out = np.append(grad_loss_out, grad_loss_accu)
    return model_coarse,model_fine,cc_loss_c_out,cc_loss_f_out,reg_loss_out,grad_loss_out

def train_v6(gpu,
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
    

    RTS_loss_fn = losses.mse_loss_ReallynoRTS
    sim_loss_fn = losses.mse_loss_ReallynoRTS
    reg_loss_fn = losses.regulation_loss
    grad_loss_fn = losses.gradient_loss

    # data generator
    studies = os.listdir(data_folder)
    train_example_gen = datagenerators.gen_CT_RTS_v5(cwd, studies, data_folder)

#    loss_c_out = np.array([])
    RTS_loss_c_out = np.array([])
    sim_loss_c_out = np.array([])
#    loss_f_out = np.array([])
    RTS_loss_f_out = np.array([])
    sim_loss_f_out = np.array([])
    reg_loss_out = np.array([])
    grad_loss_out = np.array([])

    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        params = list(model_coarse.parameters()) + list(model_fine.parameters())
        lr_new = lr * pow(0.5,num_reduced_factor)
#        opt_coarse = Adam(model_coarse.parameters(), lr=lr_new)
#        opt_fine = Adam(model_fine.parameters(), lr=lr_new)
        opt = Adam(params, lr=lr_new)
        # Save model checkpoint
        if i % n_save_epoch == 0:
            save_file_name_coarse = os.path.join(model_dir, 'coarse%d.ckpt' % i)
            save_file_name_fine = os.path.join(model_dir, 'fine%d.ckpt' % i)
            torch.save(model_coarse.state_dict(), save_file_name_coarse)
            torch.save(model_fine.state_dict(), save_file_name_fine)
#        loss_c_accu = np.array([])
        RTS_loss_c_accu = np.array([])
        sim_loss_c_accu = np.array([])
        
#        loss_f_accu = np.array([])
        RTS_loss_f_accu = np.array([])
        sim_loss_f_accu = np.array([])
        reg_loss_accu = np.array([])
        grad_loss_accu = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
#        moving_patch, fixed_patch = next(train_example_gen)
        for num_patch in range(100):
            moving_image, fixed_image, moving_RTS, fixed_RTS = next(train_example_gen)
            moving_image [moving_image>-200] = 0
            fixed_image [fixed_image>-200] = 0
            input_A = torch.from_numpy(moving_image).to(device).float()
            input_A = F.interpolate(input_A, vol_size, mode = 'trilinear')
            input_B = torch.from_numpy(fixed_image).to(device).float()
            input_B = F.interpolate(input_B, vol_size, mode = 'trilinear')
            A_RTS_input = torch.from_numpy(moving_RTS).to(device).float()
            A_RTS_input = F.interpolate(A_RTS_input, vol_size, mode = 'trilinear')
            B_RTS_input = torch.from_numpy(fixed_RTS).to(device).float()
            B_RTS_input = F.interpolate(B_RTS_input, vol_size, mode = 'trilinear')
            
            start = time.time()
            
            # Run the data through the model to produce warp and flow field
            # Coarse network x2
            flow_c, flow_c_up = model_coarse(input_A, input_B)
            warp_c = ST(input_A, flow_c_up)
            warp_RTS_c = ST(A_RTS_input, flow_c_up)
            RTS_loss_c = RTS_param * RTS_loss_fn(warp_RTS_c, B_RTS_input)
            sim_loss_c = coarse_param * sim_loss_fn(input_B, warp_c)
#            loss_c = mse_loss_c + cc_loss_c
            
#             Fine network x1
            flow_f, flow_f_up = model_fine(warp_c, input_B)
            warp_f = ST(warp_c.detach(), flow_f_up)
            warp_RTS_f = ST(warp_RTS_c.detach(), flow_c_up)
            RTS_loss_f = RTS_param * RTS_loss_fn(warp_RTS_f, B_RTS_input)
            sim_loss_f = fine_param * sim_loss_fn(input_B, warp_f)
#            loss_f = mse_loss_f + cc_loss_f
            
            dvf = flow_c_up + flow_f_up
            reg_loss = reg_param * reg_loss_fn(dvf)
            grad_loss = grad_param * grad_loss_fn(dvf)
            
            loss = sim_loss_c + sim_loss_f + RTS_loss_c + RTS_loss_f + reg_loss + grad_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            
            # accu loss
#            loss_c_accu = np.append(loss_c_accu, loss_c.item())
            RTS_loss_c_accu = np.append(RTS_loss_c_accu, RTS_loss_c.item())
            sim_loss_c_accu = np.append(sim_loss_c_accu, sim_loss_c.item())
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())
#            loss_f_accu = np.append(loss_f_accu, loss_f.item())
            RTS_loss_f_accu = np.append(RTS_loss_f_accu, RTS_loss_f.item())
            sim_loss_f_accu = np.append(sim_loss_f_accu, sim_loss_f.item())

      
            if (i == 0) & (num_patch == 0):
                print('saved\n')
#                loss_c_out = np.append(loss_c_out, np.mean(loss_c_accu))
                RTS_loss_c_out = np.append(RTS_loss_c_out, np.mean(RTS_loss_c_accu))
                sim_loss_c_out = np.append(sim_loss_c_out, np.mean(sim_loss_c_accu))
#                loss_f_out = np.append(loss_f_out, np.mean(loss_f_accu))
                RTS_loss_f_out = np.append(RTS_loss_f_out, np.mean(RTS_loss_f_accu))
                sim_loss_f_out = np.append(sim_loss_f_out, np.mean(sim_loss_f_accu))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))

            end = time.time()
            print("\r n_iter: %d/100, time = %fs, lr: %e, sim_c: %f, sim_f: %f, RTS_c:%f, RTS_f:%f, reg:%f, grad:%f" 
              % (num_patch + 1, end-start, lr_new, np.mean(sim_loss_c_accu),np.mean(sim_loss_f_accu),
                 np.mean(RTS_loss_c_accu),np.mean(RTS_loss_f_accu), np.mean(reg_loss_accu), np.mean(grad_loss_accu),
                 ), end = '', flush=True)
#       loss_c_out = np.append(loss_c_out, np.mean(loss_c_accu))
        RTS_loss_c_out = np.append(RTS_loss_c_out, np.mean(RTS_loss_c_accu))
        sim_loss_c_out = np.append(sim_loss_c_out, np.mean(sim_loss_c_accu))
#       loss_f_out = np.append(loss_f_out, np.mean(loss_f_accu))
        RTS_loss_f_out = np.append(RTS_loss_f_out, np.mean(RTS_loss_f_accu))
        sim_loss_f_out = np.append(sim_loss_f_out, np.mean(sim_loss_f_accu))
        reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
        grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))
    return model_coarse,model_fine,sim_loss_c_out,sim_loss_f_out,RTS_loss_c_out,RTS_loss_f_out,reg_loss_out,grad_loss_out


def retrain(gpu,
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
          lr_schedule,
          model_coarse,
          model_fine):
    
    reg_param = params[0]
    grad_param = params[1]
    mse_param = params[2]
    coarse_param = params[3]
    fine_param = params[4]
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    ST = SpatialTransformer(vol_size) 
    ST.to(device)
    
    
    # Set optimizer and losses
    opt_coarse = Adam(model_coarse.parameters(), lr=lr)
    opt_fine = Adam(model_fine.parameters(), lr=lr)

#    mse_loss_fn = losses.mse_loss_ReallynoRTS
    sim_loss_fn = losses.cc_loss_ReallynoRTS
    reg_loss_fn = losses.regulation_loss
    grad_loss_fn = losses.gradient_loss

    # data generator
    studies = os.listdir(data_folder)
    train_example_gen = datagenerators.gen_CT_RTS_v4(cwd, studies, data_folder, extreme = True)

#    loss_c_out = np.array([])
#    mse_loss_c_out = np.array([])
    cc_loss_c_out = np.array([])
#    loss_f_out = np.array([])
#    mse_loss_f_out = np.array([])
    cc_loss_f_out = np.array([])
    reg_loss_out = np.array([])
    grad_loss_out = np.array([])

    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt_coarse = Adam(model_coarse.parameters(), lr=lr_new)
        opt_fine = Adam(model_fine.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_epoch == 0:
            save_file_name_coarse = os.path.join(model_dir, 'coarse%d.ckpt' % i)
            save_file_name_fine = os.path.join(model_dir, 'fine%d.ckpt' % i)
            torch.save(model_coarse.state_dict(), save_file_name_coarse)
            torch.save(model_fine.state_dict(), save_file_name_fine)
#        loss_c_accu = np.array([])
#        mse_loss_c_accu = np.array([])
        cc_loss_c_accu = np.array([])
        
#        loss_f_accu = np.array([])
#        mse_loss_f_accu = np.array([])
        cc_loss_f_accu = np.array([])
        reg_loss_accu = np.array([])
        grad_loss_accu = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
#        moving_patch, fixed_patch = next(train_example_gen)
        for num_patch in range(100):
            moving_patch, fixed_patch = next(train_example_gen)
            start = time.time()
            input_A = torch.from_numpy(moving_patch).to(device).float()
            # -1000 - -200
            input_A = (input_A+1000)/4000
            # 0 - 0.2
            input_A = F.interpolate(input_A, vol_size, mode = 'trilinear')
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B = (input_B+1000)/4000
            input_B = F.interpolate(input_B, vol_size, mode = 'trilinear')
            
            # Run the data through the model to produce warp and flow field
            # Coarse network x2
            flow_c, flow_c_up = model_coarse(input_A, input_B)
            warp_c = ST(input_A, flow_c_up)
#            mse_loss_c = mse_param * mse_loss_fn(warp_c, input_B)
            sim_loss_c = coarse_param * sim_loss_fn(input_B, warp_c)
#            loss_c = mse_loss_c + cc_loss_c
            
#             Fine network x1
            flow_f, flow_f_up = model_fine(warp_c, input_B)
            warp_f = ST(warp_c, flow_f_up)
#            mse_loss_f = mse_param * mse_loss_fn(warp_f, input_B)
            sim_loss_f = fine_param * sim_loss_fn(input_B, warp_f)
#            loss_f = mse_loss_f + cc_loss_f
            
            dvf = flow_c_up + flow_f_up
            reg_loss = reg_param * reg_loss_fn(dvf)
            grad_loss = grad_param * grad_loss_fn(dvf)
            
            loss = sim_loss_c + sim_loss_f + reg_loss + grad_loss
            opt_coarse.zero_grad()
            opt_fine.zero_grad()
            loss.backward()
            opt_coarse.step()    
            opt_fine.step()

            
            # accu loss
#            loss_c_accu = np.append(loss_c_accu, loss_c.item())
#            mse_loss_c_accu = np.append(mse_loss_c_accu, mse_loss_c.item())
            cc_loss_c_accu = np.append(cc_loss_c_accu, sim_loss_c.item())
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())
#            loss_f_accu = np.append(loss_f_accu, loss_f.item())
#            mse_loss_f_accu = np.append(mse_loss_f_accu, mse_loss_f.item())
            cc_loss_f_accu = np.append(cc_loss_f_accu, sim_loss_f.item())

      
            if (i == 0) & (num_patch == 0):
                print('saved\n')
#                loss_c_out = np.append(loss_c_out, np.mean(loss_c_accu))
#                mse_loss_c_out = np.append(mse_loss_c_out, np.mean(mse_loss_c_accu))
                cc_loss_c_out = np.append(cc_loss_c_out, np.mean(cc_loss_c_accu))
#                loss_f_out = np.append(loss_f_out, np.mean(loss_f_accu))
#                mse_loss_f_out = np.append(mse_loss_f_out, np.mean(mse_loss_f_accu))
                cc_loss_f_out = np.append(cc_loss_f_out, np.mean(cc_loss_f_accu))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))

            end = time.time()
            print("\r n_iter: %d/100, time = %fs, lr: %e, sim_c: %f, sim_f: %f, reg:%f, grad:%f" 
              % (num_patch + 1, end-start, lr_new, np.mean(cc_loss_c_accu),
                 np.mean(cc_loss_f_accu), np.mean(reg_loss_accu), np.mean(grad_loss_accu),
                 ), end = '', flush=True)
#        loss_c_out = np.append(loss_c_out, loss_c_accu)
#        mse_loss_c_out = np.append(mse_loss_c_out, mse_loss_c_accu)
        cc_loss_c_out = np.append(cc_loss_c_out, cc_loss_c_accu)
#        loss_f_out = np.append(loss_f_out, loss_f_accu)
#        mse_loss_f_out = np.append(mse_loss_f_out, mse_loss_f_accu)
        cc_loss_f_out = np.append(cc_loss_f_out, cc_loss_f_accu)
        reg_loss_out = np.append(reg_loss_out, reg_loss_accu)
        grad_loss_out = np.append(grad_loss_out, grad_loss_accu)
    return model_coarse,model_fine,cc_loss_c_out,cc_loss_f_out,reg_loss_out,grad_loss_out

def retrain_cycle(gpu,
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
          lr_schedule,
          model_coarse,
          model_fine):
    
    reg_param = params[0]
    grad_param = params[1]
#    mse_param = params[2]
    coarse_param = params[3]
    fine_param = params[4]
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    ST = SpatialTransformer(vol_size) 
    ST.to(device)
    
    
    # Set optimizer and losses
    opt_coarse = Adam(model_coarse.parameters(), lr=lr)
    opt_fine = Adam(model_fine.parameters(), lr=lr)

    sim_loss_fn = losses.mse_loss_ReallynoRTS
    reg_loss_fn = losses.regulation_loss
    grad_loss_fn = losses.gradient_loss

    # data generator
    studies = os.listdir(data_folder)
    train_example_gen = datagenerators.gen_CT_RTS_v4(cwd, studies, data_folder, extreme = True)

    sim_loss_out = np.array([])
    reg_loss_out = np.array([])
    grad_loss_out = np.array([])

    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt_coarse = Adam(model_coarse.parameters(), lr=lr_new)
        opt_fine = Adam(model_fine.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_epoch == 0:
            save_file_name_coarse = os.path.join(model_dir, 'coarse%d.ckpt' % i)
            save_file_name_fine = os.path.join(model_dir, 'fine%d.ckpt' % i)
            torch.save(model_coarse.state_dict(), save_file_name_coarse)
            torch.save(model_fine.state_dict(), save_file_name_fine)
            
        sim_loss_accu = np.array([])
        reg_loss_accu = np.array([])
        grad_loss_accu = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
#        moving_patch, fixed_patch = next(train_example_gen)
        for num_patch in range(100):
            moving_patch, fixed_patch = next(train_example_gen)
            start = time.time()
            input_A = torch.from_numpy(moving_patch).to(device).float()
            input_A = F.interpolate(input_A, vol_size, mode = 'trilinear')
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B = F.interpolate(input_B, vol_size, mode = 'trilinear')
            
            # Run the data through the model to produce warp and flow field
            # A to B
                # coarse
            flow_c_A2B, flow_c_up_A2B = model_coarse(input_A, input_B)
            warp_c_A2B = ST(input_A, flow_c_up_A2B)
            sim_loss_c_A2B = coarse_param * sim_loss_fn(input_B, warp_c_A2B)
                # fine
            flow_f_A2B, flow_f_up_A2B = model_fine(warp_c_A2B, input_B)
            warp_f_A2B = ST(warp_c_A2B, flow_f_up_A2B)
            sim_loss_f_A2B = fine_param * sim_loss_fn(input_B, warp_f_A2B)        
            
                #loss
            dvf_A2B = flow_c_up_A2B + flow_f_up_A2B
            sim_loss_A2B = sim_loss_c_A2B + sim_loss_f_A2B
            reg_loss_A2B = reg_param * reg_loss_fn(dvf_A2B)
            grad_loss_A2B = grad_param * grad_loss_fn(dvf_A2B) 

            # sudo B to A
                # coarse
            input_B_sudo = warp_f_A2B
            flow_c_B2A, flow_c_up_B2A = model_coarse(input_B_sudo, input_A)
            warp_c_B2A = ST(input_B_sudo, flow_c_up_B2A)
            sim_loss_c_B2A = coarse_param * sim_loss_fn(input_A, warp_c_B2A)
                # fine
            flow_f_B2A, flow_f_up_B2A = model_fine(input_B_sudo, input_A)
            warp_f_B2A = ST(input_B_sudo, flow_f_up_B2A)
            sim_loss_f_B2A = fine_param * sim_loss_fn(input_A, warp_f_B2A)        
            
                #loss
            dvf_B2A = flow_c_up_B2A + flow_f_up_B2A
            sim_loss_B2A = sim_loss_c_B2A + sim_loss_f_B2A
            reg_loss_B2A = reg_param * reg_loss_fn(dvf_B2A)
            grad_loss_B2A = grad_param * grad_loss_fn(dvf_B2A)
            
            #sum up
            sim_loss = sim_loss_A2B + sim_loss_B2A
            reg_loss = reg_loss_A2B + reg_loss_B2A
            grad_loss = grad_loss_A2B + grad_loss_B2A
            loss = sim_loss + reg_loss + grad_loss
            
            opt_coarse.zero_grad()
            opt_fine.zero_grad()
            loss.backward()
            opt_coarse.step()    
            opt_fine.step()

            
            # accu loss
            sim_loss_accu = np.append(sim_loss_accu, sim_loss.item())
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())

            if (i == 0) & (num_patch == 0):
                print('saved\n')
                sim_loss_out = np.append(sim_loss_out, np.mean(sim_loss_accu))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))

            end = time.time()
            print("\r n_iter: %d/100, time = %fs, lr: %e, sim: %f, reg:%f, grad:%f" 
              % (num_patch + 1, end-start, lr_new, np.mean(sim_loss_accu),
                 np.mean(reg_loss_accu), np.mean(grad_loss_accu),
                 ), end = '', flush=True)
        sim_loss_out = np.append(sim_loss_out, sim_loss_accu)
        reg_loss_out = np.append(reg_loss_out, reg_loss_accu)
        grad_loss_out = np.append(grad_loss_out, grad_loss_accu)
    return model_coarse,model_fine,sim_loss_out,reg_loss_out,grad_loss_out

def train_single(gpu,
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
    mse_param = params[2]
    coarse_param = params[3]
    fine_param = params[4]
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    
    model_fine = JiangNet_Fine(dim=3, scale=1, residual=True)
    ST = SpatialTransformer(vol_size) 

    model_fine.to(device)
    ST.to(device)
    
    
    # Set optimizer and losses
    opt_fine = Adam(model_fine.parameters(), lr=lr)

#    mse_loss_fn = losses.mse_loss_ReallynoRTS
    sim_loss_fn = losses.cc_loss_ReallynoRTS
    reg_loss_fn = losses.regulation_loss
    grad_loss_fn = losses.gradient_loss

    # data generator
    studies = os.listdir(data_folder)
    train_example_gen = datagenerators.gen_CT_RTS_v4(cwd, studies, data_folder, extreme = True)

#    loss_c_out = np.array([])
#    mse_loss_c_out = np.array([])
#    loss_f_out = np.array([])
#    mse_loss_f_out = np.array([])
    cc_loss_f_out = np.array([])
    reg_loss_out = np.array([])
    grad_loss_out = np.array([])

    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt_fine = Adam(model_fine.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_epoch == 0:
            save_file_name_fine = os.path.join(model_dir, 'fine%d.ckpt' % i)
            torch.save(model_fine.state_dict(), save_file_name_fine)
#        loss_c_accu = np.array([])
#        mse_loss_c_accu = np.array([])
#        loss_f_accu = np.array([])
#        mse_loss_f_accu = np.array([])
        cc_loss_f_accu = np.array([])
        reg_loss_accu = np.array([])
        grad_loss_accu = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
#        moving_patch, fixed_patch = next(train_example_gen)
        for num_patch in range(100):
            moving_patch, fixed_patch = next(train_example_gen)
            start = time.time()
            input_A = torch.from_numpy(moving_patch).to(device).float()
            # -1000 - -200
#            input_A = (input_A+1000)/4000
            # 0 - 0.2
            input_A = F.interpolate(input_A, vol_size, mode = 'trilinear')
            input_B = torch.from_numpy(fixed_patch).to(device).float()
#            input_B = (input_B+1000)/4000
            input_B = F.interpolate(input_B, vol_size, mode = 'trilinear')
            
            # Run the data through the model to produce warp and flow field
            
#             Fine network x1
            flow_f, flow_f_up = model_fine(input_A, input_B)
            warp_f = ST(input_A, flow_f_up)
#            mse_loss_f = mse_param * mse_loss_fn(warp_f, input_B)
            sim_loss_f = fine_param * sim_loss_fn(input_B, warp_f)
#            loss_f = mse_loss_f + cc_loss_f
            
            dvf = flow_f_up
            reg_loss = reg_param * reg_loss_fn(dvf)
            grad_loss = grad_param * grad_loss_fn(dvf)
            
            loss = sim_loss_f + reg_loss + grad_loss
            opt_fine.zero_grad()
            loss.backward()  
            opt_fine.step()

            
            # accu loss
#            loss_c_accu = np.append(loss_c_accu, loss_c.item())
#            mse_loss_c_accu = np.append(mse_loss_c_accu, mse_loss_c.item())
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())
#            loss_f_accu = np.append(loss_f_accu, loss_f.item())
#            mse_loss_f_accu = np.append(mse_loss_f_accu, mse_loss_f.item())
            cc_loss_f_accu = np.append(cc_loss_f_accu, sim_loss_f.item())

      
            if (i == 0) & (num_patch == 0):
                print('saved\n')
#                loss_c_out = np.append(loss_c_out, np.mean(loss_c_accu))
#                mse_loss_c_out = np.append(mse_loss_c_out, np.mean(mse_loss_c_accu))
#                loss_f_out = np.append(loss_f_out, np.mean(loss_f_accu))
#                mse_loss_f_out = np.append(mse_loss_f_out, np.mean(mse_loss_f_accu))
                cc_loss_f_out = np.append(cc_loss_f_out, np.mean(cc_loss_f_accu))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))

            end = time.time()
            print("\r n_iter: %d/100, time = %fs, lr: %e, sim_f: %f, reg:%f, grad:%f" 
              % (num_patch + 1, end-start, lr_new, np.mean(cc_loss_f_accu), np.mean(reg_loss_accu), np.mean(grad_loss_accu),
                 ), end = '', flush=True)
#        loss_c_out = np.append(loss_c_out, loss_c_accu)
#        mse_loss_c_out = np.append(mse_loss_c_out, mse_loss_c_accu)
#        loss_f_out = np.append(loss_f_out, loss_f_accu)
#        mse_loss_f_out = np.append(mse_loss_f_out, mse_loss_f_accu)
        cc_loss_f_out = np.append(cc_loss_f_out, cc_loss_f_accu)
        reg_loss_out = np.append(reg_loss_out, reg_loss_accu)
        grad_loss_out = np.append(grad_loss_out, grad_loss_accu)
    return model_fine,cc_loss_f_out,reg_loss_out,grad_loss_out

def retrain_single(gpu,
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
          lr_schedule,
          model_fine):
    
    reg_param = params[0]
    grad_param = params[1]
    mse_param = params[2]
    coarse_param = params[3]
    fine_param = params[4]
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    ST = SpatialTransformer(vol_size) 
    ST.to(device)
    
    
    # Set optimizer and losses
    opt_fine = Adam(model_fine.parameters(), lr=lr)

#    mse_loss_fn = losses.mse_loss_ReallynoRTS
    sim_loss_fn = losses.mse_loss_ReallynoRTS
    reg_loss_fn = losses.regulation_loss
    grad_loss_fn = losses.gradient_loss

    # data generator
    studies = os.listdir(data_folder)
    train_example_gen = datagenerators.gen_CT_RTS_v4(cwd, studies, data_folder, extreme = True)

#    loss_c_out = np.array([])
#    mse_loss_c_out = np.array([])
    cc_loss_c_out = np.array([])
#    loss_f_out = np.array([])
#    mse_loss_f_out = np.array([])
    cc_loss_f_out = np.array([])
    reg_loss_out = np.array([])
    grad_loss_out = np.array([])

    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt_fine = Adam(model_fine.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_epoch == 0:
            save_file_name_fine = os.path.join(model_dir, 'fine%d.ckpt' % i)
            torch.save(model_fine.state_dict(), save_file_name_fine)
#        loss_c_accu = np.array([])
#        mse_loss_c_accu = np.array([])
        
#        loss_f_accu = np.array([])
#        mse_loss_f_accu = np.array([])
        cc_loss_f_accu = np.array([])
        reg_loss_accu = np.array([])
        grad_loss_accu = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
#        moving_patch, fixed_patch = next(train_example_gen)
        for num_patch in range(100):
            moving_patch, fixed_patch = next(train_example_gen)
            start = time.time()
            input_A = torch.from_numpy(moving_patch).to(device).float()
            # -1000 - -200
#            input_A = (input_A+1000)/4000
            # 0 - 0.2
            input_A = F.interpolate(input_A, vol_size, mode = 'trilinear')
            input_B = torch.from_numpy(fixed_patch).to(device).float()
#            input_B = (input_B+1000)/4000
            input_B = F.interpolate(input_B, vol_size, mode = 'trilinear')
            
            # Run the data through the model to produce warp and flow field
            # Coarse network x2
            flow_f, flow_f_up = model_fine(input_A, input_B)
            warp_f = ST(input_A, flow_f_up)
#            mse_loss_f = mse_param * mse_loss_fn(warp_f, input_B)
            sim_loss_f = fine_param * sim_loss_fn(input_B, warp_f)
#            loss_f = mse_loss_f + cc_loss_f
            
            dvf = flow_f_up
            reg_loss = reg_param * reg_loss_fn(dvf)
            grad_loss = grad_param * grad_loss_fn(dvf)
            
            loss = sim_loss_f + reg_loss + grad_loss
            opt_fine.zero_grad()
            loss.backward()  
            opt_fine.step()

            
            # accu loss
#            loss_c_accu = np.append(loss_c_accu, loss_c.item())
#            mse_loss_c_accu = np.append(mse_loss_c_accu, mse_loss_c.item())
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())
#            loss_f_accu = np.append(loss_f_accu, loss_f.item())
#            mse_loss_f_accu = np.append(mse_loss_f_accu, mse_loss_f.item())
            cc_loss_f_accu = np.append(cc_loss_f_accu, sim_loss_f.item())

      
            if (i == 0) & (num_patch == 0):
                print('saved\n')
#                loss_c_out = np.append(loss_c_out, np.mean(loss_c_accu))
#                mse_loss_c_out = np.append(mse_loss_c_out, np.mean(mse_loss_c_accu))
#                loss_f_out = np.append(loss_f_out, np.mean(loss_f_accu))
#                mse_loss_f_out = np.append(mse_loss_f_out, np.mean(mse_loss_f_accu))
                cc_loss_f_out = np.append(cc_loss_f_out, np.mean(cc_loss_f_accu))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))

            end = time.time()
            print("\r n_iter: %d/100, time = %fs, lr: %e, sim_f: %f, reg:%f, grad:%f" 
              % (num_patch + 1, end-start, lr_new, np.mean(cc_loss_f_accu), np.mean(reg_loss_accu), np.mean(grad_loss_accu),
                 ), end = '', flush=True)
#        loss_c_out = np.append(loss_c_out, loss_c_accu)
#        mse_loss_c_out = np.append(mse_loss_c_out, mse_loss_c_accu)
#        loss_f_out = np.append(loss_f_out, loss_f_accu)
#        mse_loss_f_out = np.append(mse_loss_f_out, mse_loss_f_accu)
        cc_loss_f_out = np.append(cc_loss_f_out, cc_loss_f_accu)
        reg_loss_out = np.append(reg_loss_out, reg_loss_accu)
        grad_loss_out = np.append(grad_loss_out, grad_loss_accu)
    return model_fine,cc_loss_f_out,reg_loss_out,grad_loss_out

def train_v5(gpu,
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
    mse_param = params[2]
    coarse_param = params[3]
    fine_param = params[4]
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    
#    model_coarse = Unet(nf_enc_coarse, nf_dec_coarse, 2)
#    model_fine = Unet(nf_enc_fine, nf_dec_fine, 1)
#    model_coarse = JiangNet(dim=3, scale=2, residual=False)
#    model_fine = JiangNet(dim=3, scale=1, residual=True)
    model_coarse = JiangNet_Fine(dim=3, scale=4, residual=False)
    model_fine = JiangNet_Fine(dim=3, scale=2, residual=True)
    ST = SpatialTransformer(vol_size) 
    
    model_coarse.to(device)
    model_fine.to(device)
    ST.to(device)
    
    
    # Set optimizer and losses
    params = list(model_coarse.parameters()) + list(model_fine.parameters())
    opt = Adam(params, lr=lr)

#    mse_loss_fn = losses.mse_loss_ReallynoRTS
    sim_loss_fn = losses.cc_loss_ReallynoRTS
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
#        loss_c_accu = np.array([])
#        mse_loss_c_accu = np.array([])
        cc_loss_c_accu = np.array([])
        
#        loss_f_accu = np.array([])
#        mse_loss_f_accu = np.array([])
        cc_loss_f_accu = np.array([])
        reg_loss_accu = np.array([])
        grad_loss_accu = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
#        moving_patch, fixed_patch = next(train_example_gen)
        for num_patch in range(100):
            moving_patch, fixed_patch, moving_RTS, fixed_RTS = next(train_example_gen)
            start = time.time()
            input_A = torch.from_numpy(moving_patch).to(device).float()
            input_A[input_A>-200] = 0
            # -1000 - -200
#            input_A = (input_A+1000)/4000
            # 0 - 0.2
            input_A = F.interpolate(input_A, vol_size, mode = 'trilinear')
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B[input_B>-200] = 0
#            input_B = (input_B+1000)/4000
            input_B = F.interpolate(input_B, vol_size, mode = 'trilinear')
            
            # Run the data through the model to produce warp and flow field
            # Coarse network x2
            flow_c, flow_c_up = model_coarse(input_A, input_B)
            warp_c = ST(input_A, flow_c_up)
#            mse_loss_c = mse_param * mse_loss_fn(warp_c, input_B)
            sim_loss_c = coarse_param * sim_loss_fn(input_B, warp_c)
#            loss_c = mse_loss_c + cc_loss_c
            
#             Fine network x1
            flow_f, flow_f_up = model_fine(warp_c, input_B)
            warp_f = ST(warp_c, flow_f_up)
#            mse_loss_f = mse_param * mse_loss_fn(warp_f, input_B)
            sim_loss_f = fine_param * sim_loss_fn(input_B, warp_f)
#            loss_f = mse_loss_f + cc_loss_f
            
            dvf = flow_c_up + flow_f_up
            reg_loss = reg_param * reg_loss_fn(dvf)
            grad_loss = grad_param * grad_loss_fn(dvf)
            
            loss = sim_loss_c + sim_loss_f + reg_loss + grad_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            
            # accu loss
#            loss_c_accu = np.append(loss_c_accu, loss_c.item())
#            mse_loss_c_accu = np.append(mse_loss_c_accu, mse_loss_c.item())
            cc_loss_c_accu = np.append(cc_loss_c_accu, sim_loss_c.item())
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())
#            loss_f_accu = np.append(loss_f_accu, loss_f.item())
#            mse_loss_f_accu = np.append(mse_loss_f_accu, mse_loss_f.item())
            cc_loss_f_accu = np.append(cc_loss_f_accu, sim_loss_f.item())

      
            if (i == 0) & (num_patch == 0):
                print('saved\n')
#                loss_c_out = np.append(loss_c_out, np.mean(loss_c_accu))
#                mse_loss_c_out = np.append(mse_loss_c_out, np.mean(mse_loss_c_accu))
                cc_loss_c_out = np.append(cc_loss_c_out, np.mean(cc_loss_c_accu))
#                loss_f_out = np.append(loss_f_out, np.mean(loss_f_accu))
#                mse_loss_f_out = np.append(mse_loss_f_out, np.mean(mse_loss_f_accu))
                cc_loss_f_out = np.append(cc_loss_f_out, np.mean(cc_loss_f_accu))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))

            end = time.time()
            print("\r n_iter: %d/100, time = %fs, lr: %e, sim_c: %f, sim_f: %f, reg:%f, grad:%f" 
              % (num_patch + 1, end-start, lr_new, np.mean(cc_loss_c_accu),
                 np.mean(cc_loss_f_accu), np.mean(reg_loss_accu), np.mean(grad_loss_accu),
                 ), end = '', flush=True)
#        loss_c_out = np.append(loss_c_out, loss_c_accu)
#        mse_loss_c_out = np.append(mse_loss_c_out, mse_loss_c_accu)
        cc_loss_c_out = np.append(cc_loss_c_out, cc_loss_c_accu)
#        loss_f_out = np.append(loss_f_out, loss_f_accu)
#        mse_loss_f_out = np.append(mse_loss_f_out, mse_loss_f_accu)
        cc_loss_f_out = np.append(cc_loss_f_out, cc_loss_f_accu)
        reg_loss_out = np.append(reg_loss_out, reg_loss_accu)
        grad_loss_out = np.append(grad_loss_out, grad_loss_accu)
    return model_coarse,model_fine,cc_loss_c_out,cc_loss_f_out,reg_loss_out,grad_loss_out

def train_v5_single(gpu,
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
    mse_param = params[2]
    coarse_param = params[3]
    fine_param = params[4]
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    
#    model_coarse = Unet(nf_enc_coarse, nf_dec_coarse, 2)
    model_fine = Unet(nf_enc_fine, nf_dec_fine, 1)
#    model_coarse = JiangNet(dim=3, scale=2, residual=False)
#    model_fine = JiangNet(dim=3, scale=1, residual=True)
    ST = SpatialTransformer(vol_size) 
    
#    model_coarse.to(device)
    model_fine.to(device)
    ST.to(device)
    
    
    # Set optimizer and losses
    opt = Adam(model_fine.parameters(), lr=lr)

#    mse_loss_fn = losses.mse_loss_ReallynoRTS
    sim_loss_fn = losses.mse_loss_ReallynoRTS
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

    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt = Adam(model_fine.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_epoch == 0:
            save_file_name_coarse = os.path.join(model_dir, 'coarse%d.ckpt' % i)
            save_file_name_fine = os.path.join(model_dir, 'fine%d.ckpt' % i)
            torch.save(model_fine.state_dict(), save_file_name_fine)
#        loss_c_accu = np.array([])
#        mse_loss_c_accu = np.array([])
        cc_loss_c_accu = np.array([])
        
#        loss_f_accu = np.array([])
#        mse_loss_f_accu = np.array([])
        cc_loss_f_accu = np.array([])
        reg_loss_accu = np.array([])
        grad_loss_accu = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
#        moving_patch, fixed_patch = next(train_example_gen)
        for num_patch in range(100):
            moving_patch, fixed_patch, moving_RTS, fixed_RTS = next(train_example_gen)
            start = time.time()
            input_A = torch.from_numpy(moving_patch).to(device).float()
            input_A[input_A>-200] = 0
            # -1000 - -200
#            input_A = (input_A+1000)/4000
            # 0 - 0.2
            input_A = F.interpolate(input_A, vol_size, mode = 'trilinear')
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B[input_B>-200] = 0
#            input_B = (input_B+1000)/4000
            input_B = F.interpolate(input_B, vol_size, mode = 'trilinear')
            
            # Run the data through the model to produce warp and flow field
            # Coarse network x2
#            flow_c, flow_c_up = model_coarse(input_A, input_B)
#            warp_c = ST(input_A, flow_c_up)
##            mse_loss_c = mse_param * mse_loss_fn(warp_c, input_B)
#            sim_loss_c = coarse_param * sim_loss_fn(input_B, warp_c)
#            loss_c = mse_loss_c + cc_loss_c
            
#             Fine network x1
            flow_f, flow_f_up = model_fine(input_A, input_B)
            warp_f = ST(input_A, flow_f_up)
#            mse_loss_f = mse_param * mse_loss_fn(warp_f, input_B)
            sim_loss_f = fine_param * sim_loss_fn(input_B, input_A)
#            loss_f = mse_loss_f + cc_loss_f
            
            dvf = flow_f_up
            reg_loss = reg_param * reg_loss_fn(dvf)
            grad_loss = grad_param * grad_loss_fn(dvf)
            
            loss = sim_loss_f + reg_loss + grad_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            
            # accu loss
#            loss_c_accu = np.append(loss_c_accu, loss_c.item())
#            mse_loss_c_accu = np.append(mse_loss_c_accu, mse_loss_c.item())
#            cc_loss_c_accu = np.append(cc_loss_c_accu, sim_loss_c.item())
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())
#            loss_f_accu = np.append(loss_f_accu, loss_f.item())
#            mse_loss_f_accu = np.append(mse_loss_f_accu, mse_loss_f.item())
            cc_loss_f_accu = np.append(cc_loss_f_accu, sim_loss_f.item())

      
            if (i == 0) & (num_patch == 0):
                print('saved\n')
#                loss_c_out = np.append(loss_c_out, np.mean(loss_c_accu))
#                mse_loss_c_out = np.append(mse_loss_c_out, np.mean(mse_loss_c_accu))
                cc_loss_c_out = np.append(cc_loss_c_out, np.mean(cc_loss_c_accu))
#                loss_f_out = np.append(loss_f_out, np.mean(loss_f_accu))
#                mse_loss_f_out = np.append(mse_loss_f_out, np.mean(mse_loss_f_accu))
                cc_loss_f_out = np.append(cc_loss_f_out, np.mean(cc_loss_f_accu))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))

            end = time.time()
            print("\r n_iter: %d/100, time = %fs, lr: %e, sim_c: %f, sim_f: %f, reg:%f, grad:%f" 
              % (num_patch + 1, end-start, lr_new, np.mean(cc_loss_c_accu),
                 np.mean(cc_loss_f_accu), np.mean(reg_loss_accu), np.mean(grad_loss_accu),
                 ), end = '', flush=True)
#        loss_c_out = np.append(loss_c_out, loss_c_accu)
#        mse_loss_c_out = np.append(mse_loss_c_out, mse_loss_c_accu)
        cc_loss_c_out = np.append(cc_loss_c_out, cc_loss_c_accu)
#        loss_f_out = np.append(loss_f_out, loss_f_accu)
#        mse_loss_f_out = np.append(mse_loss_f_out, mse_loss_f_accu)
        cc_loss_f_out = np.append(cc_loss_f_out, cc_loss_f_accu)
        reg_loss_out = np.append(reg_loss_out, reg_loss_accu)
        grad_loss_out = np.append(grad_loss_out, grad_loss_accu)
    return model_fine,cc_loss_c_out,cc_loss_f_out,reg_loss_out,grad_loss_out

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
    
    model_coarse = JiangNet_Fine(dim=3, scale=4, residual=False)
    model_fine = JiangNet_Fine(dim=3, scale=2, residual=True)
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
