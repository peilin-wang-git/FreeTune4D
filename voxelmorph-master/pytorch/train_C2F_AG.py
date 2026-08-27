# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 01:03:45 2020

@author: Haonan Xiao

E-mail: hx42@duke.edu
"""

import os

# external imports
import numpy as np
import torch
from torch.optim import Adam

# internal imports
from model_C2F import SpatialTransformer, G_coarse, G_fine, D_coarse, D_fine
import datagenerators
import losses
from math import floor
import time

def train(gpu,
          cwd,
          coarse_size,
          fine_size,
          studies,
          lr,
          n_iter,
          n_per_iter,
          data_loss,
          params, 
          batch_size,
          n_save_iter,
          model_dir):
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
    lung_param = params[2]
    DICE_param = params[3]
    GAN_param = params[4]
    mse_param = params[5]
    
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    Gen_coarse = G_coarse()
    Gen_fine = G_fine()
    Dis_coarse = D_coarse()
    Dis_fine = D_fine()
    ST_coarse = SpatialTransformer(coarse_size)
    ST_fine = SpatialTransformer(fine_size)
    
    Gen_coarse.to(device)
    Gen_fine.to(device)
    Dis_coarse.to(device)
    Dis_fine.to(device)
    ST_coarse.to(device)
    ST_fine.to(device)
    
    
    # Set optimizer and losses
    opt_Gen_coarse = Adam(Gen_coarse.parameters(), lr=lr)
    opt_Gen_fine = Adam(Gen_fine.parameters(), lr=lr)
    opt_Dis_coarse = Adam(Dis_coarse.parameters(), lr=lr)
    opt_Dis_fine = Adam(Dis_fine.parameters(), lr=lr)

    cc_loss_fn = losses.cc_loss_noRTS
    mse_loss_fn = losses.mse_loss_noRTS
    reg_loss_fn = losses.regulation_loss
    DICE_loss_fn = losses.DICE_loss
    grad_loss_fn = losses.gradient_loss
    BCE_loss_fn = torch.nn.BCELoss()

    # data generator
    data_folder = '4D-Lung-contour-npy-v2'
    train_example_gen = datagenerators.gen_CT_RTS(cwd, studies, data_folder)
    

    mse_loss_accu_c = np.array([])
    mse_loss_accu_f = np.array([])
    cc_loss_accu_c = np.array([])
    cc_loss_accu_f = np.array([])
    reg_loss_accu_c = np.array([])
    reg_loss_accu_f = np.array([])
    DICE_loss_accu_f = np.array([])
    DICE_loss_accu_c = np.array([])
    grad_loss_accu_f = np.array([])
    grad_loss_accu_c = np.array([])
    loss_coarse_accu_G = np.array([])
    loss_fine_accu_G = np.array([])     
    
    loss_coarse_G = np.array([])
    loss_fine_G = np.array([])    
    mse_loss_c = np.array([])
    mse_loss_f = np.array([])
    cc_loss_c = np.array([])
    cc_loss_f = np.array([])
    reg_loss_c = np.array([])
    reg_loss_f = np.array([])
    DICE_loss_c = np.array([])
    DICE_loss_f = np.array([])
    grad_loss_c = np.array([])
    grad_loss_f = np.array([])

    
    real_label = 1
    fake_label = 0
    low_limit = int(fine_size[0]/2)
    up_limit = int(coarse_size[0] - fine_size[0]/2)
    # Training loop.
    for i in range(n_iter):
        num_epoch = floor(i/100)+1
        num_reduced_factor = floor(num_epoch)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt_Gen_coarse = Adam(Gen_coarse.parameters(), lr=lr_new)
        opt_Gen_fine = Adam(Gen_fine.parameters(), lr=lr_new)
        opt_Dis_coarse = Adam(Dis_coarse.parameters(), lr=lr_new)
        opt_Dis_fine = Adam(Dis_fine.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_iter == 0:
            save_file_name_coarse_G = os.path.join(model_dir, 'Gen_coarse%d.ckpt' % i)
            save_file_name_fine_G = os.path.join(model_dir, 'Gen_fine%d.ckpt' % i)
            torch.save(Gen_coarse.state_dict(), save_file_name_coarse_G)
            torch.save(Gen_fine.state_dict(), save_file_name_fine_G)
            save_file_name_coarse_D = os.path.join(model_dir, 'Dis_coarse%d.ckpt' % i)
            save_file_name_fine_D = os.path.join(model_dir, 'Dis_fine%d.ckpt' % i)
            torch.save(Dis_coarse.state_dict(), save_file_name_coarse_D)
            torch.save(Dis_fine.state_dict(), save_file_name_fine_D)
            
        if i % 100 == 0:
            loss_coarse_G = np.array([])
            loss_fine_G = np.array([])
            mse_loss_accu_c = np.array([])
            mse_loss_accu_f = np.array([])
            cc_loss_accu_c = np.array([])
            cc_loss_accu_f = np.array([])
            reg_loss_accu_c = np.array([])
            reg_loss_accu_f = np.array([])
            DICE_loss_accu_f = np.array([])
            DICE_loss_accu_c = np.array([])
            grad_loss_accu_f = np.array([])
            grad_loss_accu_c = np.array([])
            loss_coarse_accu_G = np.array([])
            loss_fine_accu_G = np.array([])   
            print(' ')
            print('Epoch %d/%d' % (floor(i/100)+1,floor(n_iter/100)))
        # Generate the moving images and convert them to tensors.
        moving_image, fixed_image, moving_RTS, fixed_RTS = next(train_example_gen)
        patch_gen = datagenerators.gen_patch(moving_image, fixed_image, moving_RTS, fixed_RTS, coarse_size, batch_size)
        for num_patch in range(n_per_iter):
            start = time.time()
            moving_patch, fixed_patch, RS_M_patch, RS_F_patch = next(patch_gen)
            input_A = torch.from_numpy(moving_patch).to(device).float()
            input_A = input_A.permute(0, 4, 3, 2, 1)/1000
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B = input_B.permute(0, 4, 3, 2, 1)/1000
            input_A_RTS = torch.from_numpy(RS_M_patch).to(device).float()
            input_A_RTS = input_A_RTS.permute(1, 0, 4, 3, 2)
            input_B_RTS = torch.from_numpy(RS_F_patch).to(device).float()
            input_B_RTS = input_B_RTS.permute(1, 0, 4, 3, 2)
            body_contour = torch.unsqueeze(input_B_RTS[:,0,:,:,:],0)
            lung_contour = torch.unsqueeze(input_B_RTS[:,1,:,:,:],0)
            if (input_A_RTS.size()[1] > 2) & (input_B_RTS.size()[1] > 2):
                num_organs = input_A_RTS.size()[1]
                organ_contour_A = torch.unsqueeze(torch.sum(input_A_RTS[:,2:num_organs,:,:,:],1),1)
                organ_contour_B = torch.unsqueeze(torch.sum(input_B_RTS[:,2:num_organs,:,:,:],1),1)
            else:
                organ_contour_A = torch.zeros_like(lung_contour)
                organ_contour_B = torch.zeros_like(lung_contour) 
            
            # Run the data through the model to produce warp and flow field
            # Coarse network, Discriminator
            # Real
            Dis_coarse.zero_grad()
            label = torch.full((batch_size,), real_label, device=device)
            input_Dc_r = input_B[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
            output_Dc = Dis_coarse(input_Dc_r).view(-1)
            if (output_Dc<0) or (output_Dc>1):
                print(output_Dc)
            errD_real_coarse = GAN_param * BCE_loss_fn(output_Dc, label)
            errD_real_coarse.backward()
            Dc_r = output_Dc.mean().item()
            # Fake
            flow_coarse, flow_coarse_up = Gen_coarse(input_A, input_B)
            warp_vol_coarse = ST_coarse(input_A, flow_coarse_up)
            label.fill_(fake_label)
            input_Dc_f = warp_vol_coarse[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
            output_Dc = Dis_coarse(input_Dc_f.detach()).view(-1)
            if (output_Dc<0) or (output_Dc>1):
                print(output_Dc)
            errD_fake_coarse = GAN_param * BCE_loss_fn(output_Dc, label)
            errD_fake_coarse.backward()
            Dc_f = output_Dc.mean().item()
            errDc = errD_real_coarse + errD_fake_coarse
            opt_Dis_coarse.step()
            # Coarse network, Generator
            Gen_coarse.zero_grad()
            flow_coarse, flow_coarse_up = Gen_coarse(input_A, input_B)
            warp_vol_coarse = ST_coarse(input_A, flow_coarse_up)
            warp_RTS_coarse = ST_coarse(organ_contour_A ,flow_coarse_up)
            if torch.max(lung_contour) > 0:
                cc_loss_coarse = lung_param * cc_loss_fn(input_B, warp_vol_coarse, lung_contour) + cc_loss_fn(input_B, warp_vol_coarse, body_contour)
                mse_loss_coarse = mse_param*(lung_param * mse_loss_fn(input_B, warp_vol_coarse, lung_contour) + mse_loss_fn(input_B, warp_vol_coarse, body_contour))
            else:
                cc_loss_coarse = cc_loss_fn(input_B, warp_vol_coarse, body_contour)
                mse_loss_coarse = mse_param*(mse_loss_fn(input_B, warp_vol_coarse, body_contour))
            reg_loss_coarse = reg_param * reg_loss_fn(flow_coarse)
            DICE_loss_coarse = DICE_param * DICE_loss_fn(organ_contour_B, warp_RTS_coarse)
            grad_loss_coarse = grad_param * grad_loss_fn(flow_coarse)
            loss_coarse = mse_loss_coarse + cc_loss_coarse + reg_loss_coarse + grad_loss_coarse + DICE_loss_coarse
            loss_coarse.backward(retain_graph=True)
            loss_coarse_G = np.append(loss_coarse_G, loss_coarse.item())
            mse_loss_c = np.append(mse_loss_c, mse_loss_coarse.item())
            cc_loss_c = np.append(cc_loss_c, cc_loss_coarse.item())
            reg_loss_c = np.append(reg_loss_c, reg_loss_coarse.item())
            grad_loss_c = np.append(grad_loss_c, grad_loss_coarse.item())
            DICE_loss_c = np.append(DICE_loss_c, DICE_loss_coarse.item())
            opt_Gen_coarse.step()
            
            
            # Fine network 32x32x32 [:,:,16:48,16:48,16:48]
            input_B_fine = input_B[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
            input_vol_fine = warp_vol_coarse[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
            organ_contour_B_fine = organ_contour_B[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
            body_contour_fine = body_contour[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
            lung_contour_fine = lung_contour[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
            input_RTS_fine = warp_RTS_coarse[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
            # Discriminator
            # Real
            Dis_fine.zero_grad()
            label = torch.full((batch_size,), real_label, device=device)
            input_Df_r = input_B_fine[:,:,int(low_limit/2):int(up_limit/2),int(low_limit/2):int(up_limit/2),int(low_limit/2):int(up_limit/2)]
            output_Df = Dis_fine(input_Df_r).view(-1)
            if (output_Df<0) or (output_Df>1):
                print(output_Df)
            errD_real_fine = GAN_param * BCE_loss_fn(output_Df, label)
            errD_real_fine.backward()
            Df_r = output_Df.mean().item()            
            # Fake
            flow_fine, flow_fine_up = Gen_fine(input_vol_fine, input_B_fine)
            warp_vol_fine = ST_fine(input_vol_fine, flow_fine_up)
            label.fill_(fake_label)
            input_Df_f = warp_vol_fine[:,:,int(low_limit/2):int(up_limit/2),int(low_limit/2):int(up_limit/2),int(low_limit/2):int(up_limit/2)]
            output_Df = Dis_fine(input_Df_f.detach()).view(-1)
            if (output_Df<0) or (output_Df>1):
                print(output_Df)
            errD_fake_fine = GAN_param * BCE_loss_fn(output_Df, label)
            errD_fake_fine.backward()
            Df_f = output_Df.mean().item()
            errDf = errD_real_fine + errD_fake_fine
            opt_Dis_fine.step()            
            # Generator
            Gen_fine.zero_grad()
            flow_fine, flow_fine_up = Gen_fine(input_vol_fine, input_B_fine)
            warp_vol_fine = ST_fine(input_vol_fine, flow_fine_up)
            warp_RTS_fine = ST_fine(input_RTS_fine, input_B_fine)
            if torch.max(lung_contour_fine) > 0:
                cc_loss_fine = lung_param * cc_loss_fn(input_B_fine, warp_vol_fine, lung_contour_fine) + cc_loss_fn(input_B_fine, warp_vol_fine, body_contour_fine)
                mse_loss_fine = mse_param*(lung_param * mse_loss_fn(input_B_fine, warp_vol_fine, lung_contour_fine) + mse_loss_fn(input_B_fine, warp_vol_fine, body_contour_fine))
            else:
                cc_loss_fine = lung_param * cc_loss_fn(input_B_fine, warp_vol_fine, lung_contour_fine) + cc_loss_fn(input_B_fine, warp_vol_fine, body_contour_fine)
                mse_loss_fine = mse_param*(lung_param * mse_loss_fn(input_B_fine, warp_vol_fine, lung_contour_fine) + mse_loss_fn(input_B_fine, warp_vol_fine, body_contour_fine))
            reg_loss_fine = reg_param * reg_loss_fn(flow_fine)
            DICE_loss_fine = DICE_param * DICE_loss_fn(organ_contour_B_fine, warp_RTS_fine)
            grad_loss_fine = grad_param * grad_loss_fn(flow_fine)
            loss_fine = mse_loss_fine + cc_loss_fine + reg_loss_fine + grad_loss_fine + DICE_loss_fine            
            loss_fine.backward(retain_graph=True)
            loss_fine_G = np.append(loss_fine_G, loss_fine.item())
            mse_loss_f = np.append(mse_loss_f, mse_loss_fine.item())
            cc_loss_f = np.append(cc_loss_f, cc_loss_fine.item())
            reg_loss_f = np.append(reg_loss_f, reg_loss_fine.item())
            grad_loss_f = np.append(grad_loss_f, grad_loss_fine.item())
            DICE_loss_f = np.append(DICE_loss_f, DICE_loss_fine.item())
            opt_Gen_fine.step()
            
            
            if (i == 0) & (num_patch == 0):
                print('saved')
                loss_coarse_accu_G = np.append(loss_coarse_accu_G, np.mean(loss_coarse_G))
                loss_fine_accu_G = np.append(loss_fine_accu_G, np.mean(loss_fine_G))
                mse_loss_accu_c = np.append(mse_loss_accu_c, np.mean(mse_loss_c))
                mse_loss_accu_f = np.append(mse_loss_accu_f, np.mean(mse_loss_f))
                cc_loss_accu_c = np.append(cc_loss_accu_c, np.mean(cc_loss_c))
                cc_loss_accu_f = np.append(cc_loss_accu_f, np.mean(cc_loss_f))
                reg_loss_accu_c = np.append(reg_loss_accu_c, np.mean(reg_loss_c))
                reg_loss_accu_f = np.append(reg_loss_accu_f, np.mean(reg_loss_f))
                DICE_loss_accu_f = np.append(DICE_loss_accu_f, np.mean(DICE_loss_f))
                DICE_loss_accu_c = np.append(DICE_loss_accu_c, np.mean(DICE_loss_c))
                grad_loss_accu_c = np.append(grad_loss_accu_c, np.mean(grad_loss_c))
                grad_loss_accu_f = np.append(grad_loss_accu_f, np.mean(grad_loss_f))
            
            
            end = time.time()
            print("\r n_iter: %d/100, num_patch: %d/%d, time = %fs, lr: %e\
                  coarse:\
                  loss_c_G: %f, mse_c: %f, cc_c: %f, reg_c:%f, grad_c:%f, DICE_c:%f, Dc_r:%f, Dc_f:%f\
                  fine:\
                  loss_f_G: %f, mse_f: %f, cc_f: %f, reg_f:%f, grad_f:%f, DICE_f:%f, Df_r:%f, Df_f:%f." 
              % (i-100 * floor(i/100) + 1, num_patch + 1, n_per_iter, end-start, lr_new,
                 np.mean(loss_coarse_G), np.mean(mse_loss_c), np.mean(cc_loss_c),
                 np.mean(reg_loss_c), np.mean(grad_loss_c), np.mean(DICE_loss_c),
                 errD_real_coarse, errD_fake_coarse,
                 np.mean(loss_fine_G), np.mean(mse_loss_f), np.mean(cc_loss_f),
                 np.mean(reg_loss_f), np.mean(grad_loss_f), np.mean(DICE_loss_f),
                 errD_real_fine, errD_fake_fine
                 ), end = '', flush=True)
        loss_coarse_accu_G = np.append(loss_coarse_accu_G, np.mean(loss_coarse_G))
        loss_fine_accu_G = np.append(loss_fine_accu_G, np.mean(loss_fine_G))
        mse_loss_accu_c = np.append(mse_loss_accu_c, np.mean(mse_loss_c))
        mse_loss_accu_f = np.append(mse_loss_accu_f, np.mean(mse_loss_f))
        cc_loss_accu_c = np.append(cc_loss_accu_c, np.mean(cc_loss_c))
        cc_loss_accu_f = np.append(cc_loss_accu_f, np.mean(cc_loss_f))
        reg_loss_accu_c = np.append(reg_loss_accu_c, np.mean(reg_loss_c))
        reg_loss_accu_f = np.append(reg_loss_accu_f, np.mean(reg_loss_f))
        DICE_loss_accu_c = np.append(DICE_loss_accu_f, np.mean(DICE_loss_c))
        DICE_loss_accu_f = np.append(DICE_loss_accu_c, np.mean(DICE_loss_f))
        grad_loss_accu_c = np.append(grad_loss_accu_c, np.mean(grad_loss_c))
        grad_loss_accu_f = np.append(grad_loss_accu_f, np.mean(grad_loss_f))

    return Gen_coarse, Gen_fine, Dis_coarse, Dis_fine, loss_coarse_accu_G, loss_fine_accu_G, mse_loss_accu_c, mse_loss_accu_f,  cc_loss_accu_c,  cc_loss_accu_f, reg_loss_accu_c, reg_loss_accu_f, DICE_loss_accu_c, DICE_loss_accu_f,grad_loss_accu_c,grad_loss_accu_f
