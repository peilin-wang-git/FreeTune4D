 # -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 16:03:29 2020

@author: Haonan Xiao

E-mail: hx42@duke.edu
"""

import os

# external imports
import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
# internal imports
from model_Fu import SpatialTransformer, G_coarse, G_fine, D_coarse, D_fine, D_Unet
from model_C2F import Unet, JiangNet, JiangNet_Fine
import datagenerators
import losses
from math import floor
import time
import torch.nn as nn

def train_Fu_single(gpu,
          cwd,
          coarse_size,
          vol_size,
          data_folder,
          lr,
          n_epoch,
          params, 
          batch_size,
          num_para_patch,
          n_save_epoch,
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
    sim_param = params[0]
    GAN_param = params[1]
    l1_param = params[2]
    l2_param = params[3]
    RTS_param = params[4]
    
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    nf_enc = [32,64,128,256]
    nf_dec = [256,256,128,128,64,32]
    Gen = Unet(nf_enc, nf_dec, 1)
    Dis = D_Unet()
    ST = SpatialTransformer(coarse_size)
    
    Gen.to(device)
    Dis.to(device)
    ST.to(device)
    
    
    # Set optimizer and losses
    opt_Gen = Adam(Gen.parameters(), lr=lr)
    opt_Dis = Adam(Dis.parameters(), lr=lr)

    sim_loss_fn = losses.cc_loss_ReallynoRTS
    grad_loss_fn = losses.gradient_loss_v2
    GAN_loss_fn = torch.nn.BCELoss()

    # data generator
    studies = os.listdir(data_folder)
    train_example_gen = datagenerators.gen_CT_RTS_v5(cwd, studies, data_folder, batch_size=1)
    
    sim_loss_accu = np.array([])
    # sim_loss_accu_f = np.array([])
    # l1_loss_accu_f = np.array([])
    l1_loss_accu = np.array([])
    # l2_loss_accu_f = np.array([])
    l2_loss_accu = np.array([])
    # GAN_loss_accu_f = np.array([])
    GAN_loss_accu = np.array([])    
    loss_Gen_accu = np.array([])
    # loss_fine_accu_G = np.array([])
    
    sim_loss_out = np.array([])
    # sim_loss_f = np.array([])
    # RTS_loss_f = np.array([])
    # RTS_loss_c = np.array([])
    # l1_loss_f = np.array([])
    l1_loss_out = np.array([])
    # l2_loss_f = np.array([])
    l2_loss_out = np.array([])
    # GAN_loss_f = np.array([])
    GAN_loss_out = np.array([])    
    loss_Gen_out = np.array([])
    # loss_fine_G = np.array([])
    
    real_label = 1
    fake_label = 0
    # low_limit = int(fine_size[0]/2)
    # up_limit = int(coarse_size[0] - fine_size[0]/2)
    # Training loop.
    for i in range(n_epoch):
        # Save model checkpoint
        if i % n_save_epoch == 0:
            save_file_name_G = os.path.join(model_dir, 'Gen_%d.ckpt' % i)
            # save_file_name_fine_G = os.path.join(model_dir, 'Gen_fine%d.ckpt' % i)
            torch.save(Gen.state_dict(), save_file_name_G)
            # torch.save(Gen_fine.state_dict(), save_file_name_fine_G)
            save_file_name_D = os.path.join(model_dir, 'Dis_%d.ckpt' % i)
            # save_file_name_fine_D = os.path.join(model_dir, 'Dis_fine%d.ckpt' % i)
            torch.save(Dis.state_dict(), save_file_name_D)
            # torch.save(Dis_fine.state_dict(), save_file_name_fine_D)
            
        sim_loss_accu = np.array([])
        # sim_loss_accu_f = np.array([])
        # RTS_loss_accu_f = np.array([])
        # RTS_loss_accu_c = np.array([])
        # l1_loss_accu_f = np.array([])
        l1_loss_accu = np.array([])
        # l2_loss_accu_f = np.array([])
        l2_loss_accu = np.array([])
        # GAN_loss_accu_f = np.array([])
        GAN_loss_accu = np.array([])    
        loss_accu_G = np.array([])
        # loss_fine_accu_G = np.array([])  
        print('\nEpoch %d/%d' % (i+1,n_epoch))
        # Generate the moving images and convert them to tensors.
        for num_iter in range(100):
            moving_image, fixed_image, moving_RTS, fixed_RTS = next(train_example_gen)
            moving_input = torch.from_numpy(moving_image).to(device).float()
#            moving_input = F.interpolate(moving_input, vol_size, mode = 'trilinear')
            fixed_input = torch.from_numpy(fixed_image).to(device).float()
#            fixed_input = F.interpolate(fixed_input, vol_size, mode = 'trilinear')
            moving_RTS_input = torch.from_numpy(moving_RTS).to(device).float()
#            moving_RTS_input = F.interpolate(moving_RTS_input, vol_size, mode = 'trilinear')
            fixed_RTS_input = torch.from_numpy(fixed_RTS).to(device).float()
#            fixed_RTS_input = F.interpolate(fixed_RTS_input, vol_size, mode = 'trilinear')
            moving_input = F.interpolate(moving_input, size=[256,256,96], mode='trilinear')
            fixed_input = F.interpolate(fixed_input, size=[256,256,96], mode='trilinear')
            patch_gen = datagenerators.gen_patch_NoRTS(moving_input, fixed_input, moving_RTS_input, fixed_RTS_input, coarse_size, batch_size = num_para_patch)
                
            start = time.time()
            moving_patch, fixed_patch, RS_M_patch, RS_F_patch = next(patch_gen)
            moving_patch[moving_patch>-200] = 1
            # moving_patch = (moving_patch+1000)/4000
            fixed_patch[fixed_patch>-200] = 1
            # fixed_patch = (fixed_patch+1000)/4000
        
            # Run the data through the model to produce warp and flow field
            # Coarse network, Discriminator
            Dis.zero_grad()
            # Real
            label_r = torch.full((num_para_patch,), real_label, device=device)
            # input_Dc_r = fixed_patch[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
            output_D_r = Dis(fixed_patch)
            errD_real = GAN_param * GAN_loss_fn(output_D_r, label_r)
#               errD_real_coarse.backward()
            D_r = output_D_r.mean().item()
            # Fake
            flow, flow_up = Gen(moving_patch, fixed_patch)
            warp_vol = ST(moving_patch, flow_up)
            label_f = torch.full((num_para_patch,), fake_label, device=device)
            # input_Dc_f = warp_vol_coarse[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
            output_D_f = Dis(warp_vol.detach())
            errD_fake = GAN_param * GAN_loss_fn(output_D_f, label_f)
#                errD_fake_coarse.backward()
            D_f = output_D_f.mean().item()
            errD = errD_real + errD_fake
            opt_Dis.zero_grad()
            errD.backward()
            opt_Dis.step()
            GAN_loss_accu = np.append(GAN_loss_accu, errD.item())
                
            # Coarse network, Generator
            Gen.zero_grad()
            flow, flow_up = Gen(moving_patch, fixed_patch)
            warp_vol = ST(moving_patch, flow_up)
            # warp_RTS_coarse = ST(RS_M_patch ,flow_coarse_up)
               
            sim_loss = sim_param * sim_loss_fn(fixed_patch, warp_vol)
            l1_loss = l1_param * grad_loss_fn(flow, penalty = 'l2', order = '1')
            l2_loss = l2_param * grad_loss_fn(flow, penalty = 'l2', order = '2')
            # RTS_loss_coarse = RTS_param * sim_loss_fn(RS_F_patch, warp_RTS_coarse)
            loss_Gen = sim_loss + l1_loss + l2_loss
            opt_Gen.zero_grad()
            loss_Gen.backward()
            opt_Gen.step()
            
            loss_Gen_accu = np.append(loss_Gen_accu, loss_Gen.item())
            sim_loss_accu = np.append(sim_loss_accu, sim_loss.item())
            l1_loss_accu  = np.append(l1_loss_accu, l1_loss.item())
            l2_loss_accu = np.append(l2_loss_out, l2_loss.item())
            # RTS_loss = np.append(RTS_loss_c, RTS_loss_coarse.item())
            
            end = time.time()
            print('\r num_iter: %d/100, time = %fs, lr: %e, GAN:%f, sim: %f, l1:%f, l2:%f, D_r:%f, D_f: %f .'
              % (num_iter+1, end-start, lr,
                np.mean(GAN_loss_accu), np.mean(sim_loss_accu), np.mean(l1_loss_accu),
                np.mean(l2_loss_accu), D_r, D_f), end = '', flush=True)
            if (i == 0) & (num_iter == 0):
                print('saved')
                # loss_coarse_accu_G = np.append(loss_coarse_accu_G, np.mean(loss_coarse_G))
                # loss_fine_accu_G = np.append(loss_fine_accu_G, np.mean(loss_fine_G))
                GAN_loss_out = np.append(GAN_loss_out, GAN_loss_accu)
                sim_loss_out = np.append(sim_loss_out, sim_loss_accu)
                # sim_loss_accu_f = np.append(sim_loss_accu_f, np.mean(sim_loss_f))
                # RTS_loss_accu_f = np.append(RTS_loss_accu_f, np.mean(RTS_loss_f))
                # RTS_loss_accu_c = np.append(RTS_loss_accu_c, np.mean(RTS_loss_c))
                l1_loss_out = np.append(l1_loss_out, l1_loss_accu)
                # l1_loss_accu_f = np.append(l1_loss_accu_f, np.mean(l1_loss_f))
                l2_loss_out = np.append(l2_loss_out, l2_loss_accu)
                # l2_loss_accu_f = np.append(l2_loss_accu_f, np.mean(l2_loss_f)) 
        # loss_coarse_accu_G = np.append(loss_coarse_accu_G, np.mean(loss_coarse_G))
        # loss_fine_accu_G = np.append(loss_fine_accu_G, np.mean(loss_fine_G))
        GAN_loss_out = np.append(GAN_loss_out, GAN_loss_accu)
        sim_loss_out = np.append(sim_loss_out, sim_loss_accu)
        # sim_loss_accu_f = np.append(sim_loss_accu_f, np.mean(sim_loss_f))
        # RTS_loss_accu_f = np.append(RTS_loss_accu_f, np.mean(RTS_loss_f))
        # RTS_loss_accu_c = np.append(RTS_loss_accu_c, np.mean(RTS_loss_c))
        l1_loss_out = np.append(l1_loss_out, l1_loss_accu)
        # l1_loss_accu_f = np.append(l1_loss_accu_f, np.mean(l1_loss_f))
        l2_loss_out = np.append(l2_loss_out, l2_loss_accu)
        # l2_loss_accu_f = np.append(l2_loss_accu_f, np.mean(l2_loss_f)) 
        loss_combine = [GAN_loss_out, sim_loss_out, l1_loss_out, l2_loss_out]

    return Gen, Dis, loss_combine


def train_Fu(gpu,
          cwd,
          coarse_size,
          fine_size,
          vol_size,
          data_folder,
          lr,
          n_epoch,
          n_per_iter,
          params, 
          batch_size,
          num_para_patch,
          n_save_epoch,
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
    sim_param = params[0]
    GAN_param = params[1]
    l1_param = params[2]
    l2_param = params[3]
    RTS_param = params[4]
#    mse_param = params[5]
    
    
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

    sim_loss_fn = losses.cc_loss_ReallynoRTS
    grad_loss_fn = losses.gradient_loss_v2
    GAN_loss_fn = torch.nn.BCELoss()

    # data generator
    studies = os.listdir(data_folder)
    train_example_gen = datagenerators.gen_CT_RTS_v5(cwd, studies, data_folder, batch_size=1)
    
    sim_loss_accu_c = np.array([])
    sim_loss_accu_f = np.array([])
    RTS_loss_accu_f = np.array([])
    RTS_loss_accu_c = np.array([])
    l1_loss_accu_f = np.array([])
    l1_loss_accu_c = np.array([])
    l2_loss_accu_f = np.array([])
    l2_loss_accu_c = np.array([])
    GAN_loss_accu_f = np.array([])
    GAN_loss_accu_c = np.array([])    
    loss_coarse_accu_G = np.array([])
    loss_fine_accu_G = np.array([])
    
    sim_loss_c = np.array([])
    sim_loss_f = np.array([])
    RTS_loss_f = np.array([])
    RTS_loss_c = np.array([])
    l1_loss_f = np.array([])
    l1_loss_c = np.array([])
    l2_loss_f = np.array([])
    l2_loss_c = np.array([])
    GAN_loss_f = np.array([])
    GAN_loss_c = np.array([])    
    loss_coarse_G = np.array([])
    loss_fine_G = np.array([])

    
    real_label = 1
    fake_label = 0
    low_limit = int(fine_size[0]/2)
    up_limit = int(coarse_size[0] - fine_size[0]/2)
    # Training loop.
    for i in range(n_epoch):
        # Save model checkpoint
        if i % n_save_epoch == 0:
            save_file_name_coarse_G = os.path.join(model_dir, 'Gen_coarse%d.ckpt' % i)
            save_file_name_fine_G = os.path.join(model_dir, 'Gen_fine%d.ckpt' % i)
            torch.save(Gen_coarse.state_dict(), save_file_name_coarse_G)
            torch.save(Gen_fine.state_dict(), save_file_name_fine_G)
            save_file_name_coarse_D = os.path.join(model_dir, 'Dis_coarse%d.ckpt' % i)
            save_file_name_fine_D = os.path.join(model_dir, 'Dis_fine%d.ckpt' % i)
            torch.save(Dis_coarse.state_dict(), save_file_name_coarse_D)
            torch.save(Dis_fine.state_dict(), save_file_name_fine_D)
            
        if i % 100 == 0:
            sim_loss_accu_c = np.array([])
            sim_loss_accu_f = np.array([])
            RTS_loss_accu_f = np.array([])
            RTS_loss_accu_c = np.array([])
            l1_loss_accu_f = np.array([])
            l1_loss_accu_c = np.array([])
            l2_loss_accu_f = np.array([])
            l2_loss_accu_c = np.array([])
            GAN_loss_accu_f = np.array([])
            GAN_loss_accu_c = np.array([])    
            loss_coarse_accu_G = np.array([])
            loss_fine_accu_G = np.array([])  
            print('\nEpoch %d/%d' % (i+1,n_epoch))
        # Generate the moving images and convert them to tensors.
        for num_iter in range(100):
            moving_image, fixed_image, moving_RTS, fixed_RTS = next(train_example_gen)
            moving_input = torch.from_numpy(moving_image).to(device).float()
#            moving_input = F.interpolate(moving_input, vol_size, mode = 'trilinear')
            fixed_input = torch.from_numpy(fixed_image).to(device).float()
#            fixed_input = F.interpolate(fixed_input, vol_size, mode = 'trilinear')
            moving_RTS_input = torch.from_numpy(moving_RTS).to(device).float()
#            moving_RTS_input = F.interpolate(moving_RTS_input, vol_size, mode = 'trilinear')
            fixed_RTS_input = torch.from_numpy(fixed_RTS).to(device).float()
#            fixed_RTS_input = F.interpolate(fixed_RTS_input, vol_size, mode = 'trilinear')
            patch_gen = datagenerators.gen_patch_NoRTS(moving_input, fixed_input, moving_RTS_input, fixed_RTS_input, coarse_size, batch_size = num_para_patch)
            for num_patch in range(n_per_iter):
                start = time.time()
                moving_patch, fixed_patch, RS_M_patch, RS_F_patch = next(patch_gen)
#                moving_patch[moving_patch>-200] = 1
#                moving_patch = (moving_patch+1000)/4000
#                fixed_patch[fixed_patch>-200] = 1
#                fixed_patch = (fixed_patch+1000)/4000
            
                # Run the data through the model to produce warp and flow field
                # Coarse network, Discriminator
                Dis_coarse.zero_grad()
                # Real
                label_r = torch.full((num_para_patch,), real_label, device=device)
                input_Dc_r = fixed_patch[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
                output_Dc_r = Dis_coarse(input_Dc_r)
                errD_real_coarse = GAN_param * GAN_loss_fn(output_Dc_r, label_r)
#                errD_real_coarse.backward()
                Dc_r = output_Dc_r.mean().item()
                # Fake
                flow_coarse, flow_coarse_up = Gen_coarse(moving_patch, fixed_patch)
                warp_vol_coarse = ST_coarse(moving_patch, flow_coarse_up)
                label_f = torch.full((num_para_patch,), fake_label, device=device)
                input_Dc_f = warp_vol_coarse[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
                output_Dc_f = Dis_coarse(input_Dc_f.detach())
                errD_fake_coarse = GAN_param * GAN_loss_fn(output_Dc_f, label_f)
#                errD_fake_coarse.backward()
                Dc_f = output_Dc_f.mean().item()
                errDc = errD_real_coarse + errD_fake_coarse
                opt_Dis_coarse.zero_grad()
                errDc.backward()
                opt_Dis_coarse.step()
                GAN_loss_c = np.append(GAN_loss_c, errDc.item())
                
                # Coarse network, Generator
                Gen_coarse.zero_grad()
                flow_coarse, flow_coarse_up = Gen_coarse(moving_patch, fixed_patch)
                warp_vol_coarse = ST_coarse(moving_patch, flow_coarse_up)
                warp_RTS_coarse = ST_coarse(RS_M_patch ,flow_coarse_up)
                
                sim_loss_coarse = sim_param * sim_loss_fn(fixed_patch, warp_vol_coarse)
                l1_loss_coarse = l1_param * grad_loss_fn(flow_coarse, penalty = 'l2', order = '1')
                l2_loss_coarse = l2_param * grad_loss_fn(flow_coarse, penalty = 'l2', order = '2')
                RTS_loss_coarse = RTS_param * sim_loss_fn(RS_F_patch, warp_RTS_coarse)
                loss_coarse = sim_loss_coarse + l1_loss_coarse + l2_loss_coarse + RTS_loss_coarse
                opt_Gen_coarse.zero_grad()
                loss_coarse.backward()
                opt_Gen_coarse.step()
                
                loss_coarse_G = np.append(loss_coarse_G, loss_coarse.item())
                sim_loss_c = np.append(sim_loss_c, sim_loss_coarse.item())
                l1_loss_c = np.append(l1_loss_c, l1_loss_coarse.item())
                l2_loss_c = np.append(l2_loss_c, l2_loss_coarse.item())
                RTS_loss_c = np.append(RTS_loss_c, RTS_loss_coarse.item())
                
            
            
#                # Fine network 32x32x32
#                # Discriminator
#                Dis_fine.zero_grad()
#                warp_crop = warp_vol_coarse[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
#                fixed_crop = fixed_patch[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
#                warp_RTS_crop = warp_RTS_coarse[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
#                fixed_RTS_crop = RS_F_patch[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
#                # Real
#                input_Df_r = fixed_crop[:,:,int(low_limit/2):int(up_limit/2),int(low_limit/2):int(up_limit/2),int(low_limit/2):int(up_limit/2)]
#                output_Df_r = Dis_fine(input_Df_r).view(-1)
#                errD_real_fine = GAN_param * GAN_loss_fn(output_Df_r, label_r)
##                errD_real_fine.backward()
#                Df_r = output_Df_r.mean().item()            
#                # Fake
#                flow_fine, flow_fine_up = Gen_fine(warp_crop.detach(), fixed_crop)
#                warp_vol_fine = ST_fine(warp_crop.detach(), flow_fine_up)
#                input_Df_f = warp_vol_fine[:,:,int(low_limit/2):int(up_limit/2),int(low_limit/2):int(up_limit/2),int(low_limit/2):int(up_limit/2)]
#                output_Df_f = Dis_fine(input_Df_f.detach()).view(-1)
#                
#                if not (output_Df_f>=0 or output_Df_f <=1):
#                    print(output_Df_f.item())
#                    break
#                
#                errD_fake_fine = GAN_param * GAN_loss_fn(output_Df_f, label_f)
##                errD_fake_fine.backward()
#                Df_f = output_Df_f.mean().item()
#                errDf = errD_real_fine + errD_fake_fine
#                opt_Dis_fine.zero_grad()
#                errDf.backward()
#                opt_Dis_fine.step()
#                GAN_loss_f = np.append(GAN_loss_f, errDf.item())
#                
#                # Generator
#                Gen_fine.zero_grad()
#                flow_fine, flow_fine_up = Gen_fine(warp_crop.detach(), fixed_crop)
#                warp_vol_fine = ST_fine(warp_crop.detach(), flow_fine_up)
#                warp_RTS_fine = ST_fine(warp_RTS_crop.detach(), flow_fine_up)
#                
#                sim_loss_fine = sim_param * sim_loss_fn(fixed_crop, warp_vol_fine)
#                RTS_loss_fine = RTS_param * sim_loss_fn(fixed_RTS_crop, warp_RTS_fine)
#                l1_loss_fine = l1_param * grad_loss_fn(flow_fine, penalty = 'l2', order = '1')
#                l2_loss_fine = l2_param * grad_loss_fn(flow_fine, penalty = 'l2', order = '2')
#                loss_fine = sim_loss_fine + l1_loss_fine + l2_loss_fine + RTS_loss_fine            
#                
#                loss_fine_G = np.append(loss_fine_G, loss_fine.item())
#                sim_loss_f = np.append(sim_loss_f, sim_loss_fine.item())
#                l1_loss_f = np.append(l1_loss_f, l1_loss_fine.item())
#                l2_loss_f = np.append(l2_loss_f, l2_loss_fine.item())
#                RTS_loss_f = np.append(RTS_loss_f, RTS_loss_fine.item())
                
#                opt_Gen_fine.zero_grad()
#                loss_fine.backward()
#                opt_Gen_fine.step()

                end = time.time()
                print('\r Epoch: %d/%d, num_iter: %d/100, n_patch: %d/%d, time = %fs, lr: %e, coarse: loss_c_G: %f, GAN_c:%f, sim_c: %f, 1stG_c:%f, 2ndG_c:%f, RTS_c:%f, fine: loss_f_G: %f, GAN_f:%f, sim_f: %f, 1stG_f:%f, 2ndG_f:%f, RTS_f:%f.'
                      % (i-100 * floor(i/100) + 1, n_epoch, num_iter + 1, num_patch, n_per_iter, end-start, lr,
                         np.mean(loss_coarse_G), np.mean(GAN_loss_c), np.mean(sim_loss_c), np.mean(l1_loss_c),
                         np.mean(l2_loss_c), np.mean(RTS_loss_c),
                         np.mean(loss_fine_G), np.mean(GAN_loss_f), np.mean(sim_loss_f), np.mean(l1_loss_f),
                         np.mean(l2_loss_f), np.mean(RTS_loss_f)), end = '', flush=True)
                if (i == 0) & (num_patch == 0):
                    print('saved')
                    loss_coarse_accu_G = np.append(loss_coarse_accu_G, np.mean(loss_coarse_G))
                    loss_fine_accu_G = np.append(loss_fine_accu_G, np.mean(loss_fine_G))
                    GAN_loss_accu_c = np.append(GAN_loss_accu_c, np.mean(GAN_loss_c))
                    GAN_loss_accu_f = np.append(GAN_loss_accu_f, np.mean(GAN_loss_f))
                    sim_loss_accu_c = np.append(sim_loss_accu_c, np.mean(sim_loss_c))
                    sim_loss_accu_f = np.append(sim_loss_accu_f, np.mean(sim_loss_f))
                    RTS_loss_accu_f = np.append(RTS_loss_accu_f, np.mean(RTS_loss_f))
                    RTS_loss_accu_c = np.append(RTS_loss_accu_c, np.mean(RTS_loss_c))
                    l1_loss_accu_c = np.append(l1_loss_accu_c, np.mean(l1_loss_c))
                    l1_loss_accu_f = np.append(l1_loss_accu_f, np.mean(l1_loss_f))
                    l2_loss_accu_c = np.append(l2_loss_accu_c, np.mean(l2_loss_c))
                    l2_loss_accu_f = np.append(l2_loss_accu_f, np.mean(l2_loss_f))  
        loss_coarse_accu_G = np.append(loss_coarse_accu_G, np.mean(loss_coarse_G))
        loss_fine_accu_G = np.append(loss_fine_accu_G, np.mean(loss_fine_G))
        sim_loss_accu_c = np.append(sim_loss_accu_c, np.mean(sim_loss_c))
        sim_loss_accu_f = np.append(sim_loss_accu_f, np.mean(sim_loss_f))
        RTS_loss_accu_f = np.append(RTS_loss_accu_f, np.mean(RTS_loss_f))
        RTS_loss_accu_c = np.append(RTS_loss_accu_c, np.mean(RTS_loss_c))
        l1_loss_accu_c = np.append(l1_loss_accu_c, np.mean(l1_loss_c))
        l1_loss_accu_f = np.append(l1_loss_accu_f, np.mean(l1_loss_f))
        l2_loss_accu_c = np.append(l2_loss_accu_c, np.mean(l2_loss_c))
        l2_loss_accu_f = np.append(l2_loss_accu_f, np.mean(l2_loss_f))
        GAN_loss_accu_c = np.append(GAN_loss_accu_c, np.mean(GAN_loss_c))
        GAN_loss_accu_f = np.append(GAN_loss_accu_f, np.mean(GAN_loss_f))
        loss_combine = [loss_coarse_accu_G, loss_fine_accu_G, sim_loss_accu_c, \
                        sim_loss_accu_f, RTS_loss_accu_f, RTS_loss_accu_c, l1_loss_accu_c, \
                        l1_loss_accu_f, l2_loss_accu_c, l2_loss_accu_f, GAN_loss_accu_c, GAN_loss_accu_f]

    return Gen_coarse, Gen_fine, Dis_coarse, Dis_fine, loss_coarse_accu_G, loss_combine
