# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 00:18:01 2020

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
from model_C2F import voxelmorph_C, simple_fine, voxelmorph_F, voxelmorph_FF
from model_C2F_AG import AG_Unet
import torch.nn.functional as F
import datagenerators
import losses
from math import floor
import time

def train(gpu,
          cwd,
          coarse_size,
          fine_size,
          nf_enc_coarse,
          nf_dec_coarse,
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
    lung_param = params[2]
    DICE_param = params[3]
    GAN_param = params[4]
    mse_param = params[5]

    coarse = True
    fine = False
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    model_coarse = voxelmorph_F(coarse_size, nf_enc_coarse, nf_dec_coarse,fine)
#    model_coarse = AG_Unet(nf_enc_coarse, nf_dec_coarse,2)
    model_fine = simple_fine()
    ST_coarse = SpatialTransformer(coarse_size)
#    ST_fine = SpatialTransformer(fine_size)    
    
    model_coarse.to(device)
#    model_fine.to(device)
    ST_coarse.to(device)
#    ST_fine.to(device)
    
    
    # Set optimizer and losses
    opt_coarse = Adam(model_coarse.parameters(), lr=lr)
#    opt_fine = Adam(model_fine.parameters(), lr=lr)

    mse_loss_fn = losses.mse_loss_noRTS
#    cc_loss_fn = losses.cc_loss_noRTS
    reg_loss_fn = losses.regulation_loss
#    DICE_loss_fn = losses.DICE_loss
    grad_loss_fn = losses.gradient_loss

    # data generator
#    train_example_gen = datagenerators.example_gen_HX(cwd, studies)
    studies = os.listdir(data_folder)
    train_example_gen = datagenerators.gen_CT_RTS_v3_lung(cwd, studies, data_folder)

    loss_c_out = np.array([])
    mse_loss_c_out = np.array([])
    cc_loss_c_out = np.array([])
    reg_loss_c_out = np.array([])
    DICE_loss_c_out = np.array([])
    grad_loss_c_out = np.array([])
    loss_f_out = np.array([])
    mse_loss_f_out = np.array([])
    cc_loss_f_out = np.array([])
    reg_loss_f_out = np.array([])
    DICE_loss_f_out = np.array([])
    grad_loss_f_out = np.array([])

    # Training loop.
    for i in range(n_epoch):
        num_epoch = i
        num_reduced_factor = floor(num_epoch/lr_schedule)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt_coarse = Adam(model_coarse.parameters(), lr=lr_new)
#        opt_fine = Adam(model_fine.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_epoch == 0:
            save_file_name_coarse = os.path.join(model_dir, 'coarse%d.ckpt' % i)
            save_file_name_fine = os.path.join(model_dir, 'fine%d.ckpt' % i)
            torch.save(model_coarse.state_dict(), save_file_name_coarse)
            torch.save(model_fine.state_dict(), save_file_name_fine)
            
        loss_c_accu = np.array([])
        mse_loss_c_accu = np.array([])
        cc_loss_c_accu = np.array([])
        reg_loss_c_accu = np.array([])
        DICE_loss_c_accu = np.array([])
        grad_loss_c_accu = np.array([])
        loss_f_accu = np.array([])
        mse_loss_f_accu = np.array([])
        cc_loss_f_accu = np.array([])
        reg_loss_f_accu = np.array([])
        DICE_loss_f_accu = np.array([])
        grad_loss_f_accu = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
        # Generate the moving images and convert them to tensors.
        moving_patch, fixed_patch, RS_M_patch, RS_F_patch = next(train_example_gen)
        for num_patch in range(5):
            start = time.time()
            input_A = torch.from_numpy(moving_patch).to(device).float()
            input_A = input_A.permute(0, 4, 3, 2, 1)
            input_A = F.interpolate(input_A, coarse_size, mode = 'trilinear')
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B = input_B.permute(0, 4, 3, 2, 1)
            input_B = F.interpolate(input_B, coarse_size, mode = 'trilinear')
            input_A_RTS = torch.from_numpy(RS_M_patch).to(device).float()
            input_A_RTS = input_A_RTS.permute(1, 0, 4, 3, 2)
            input_A_RTS = F.interpolate(input_A_RTS, coarse_size, mode = 'trilinear')
            input_B_RTS = torch.from_numpy(RS_F_patch).to(device).float()
            input_B_RTS = input_B_RTS.permute(1, 0, 4, 3, 2)
            input_B_RTS = F.interpolate(input_B_RTS, coarse_size, mode = 'trilinear')
            body_contour = torch.unsqueeze(input_B_RTS[:,0,:,:,:],0)
            lung_contour = torch.unsqueeze(input_B_RTS[:,1,:,:,:],0)
#            if (input_A_RTS.size()[1] > 2) & (input_B_RTS.size()[1] > 2):
#                num_organs = input_A_RTS.size()[1]
#                organ_contour_A = torch.unsqueeze(torch.sum(input_A_RTS[:,2:num_organs,:,:,:],1),1)
#                organ_contour_B = torch.unsqueeze(torch.sum(input_B_RTS[:,2:num_organs,:,:,:],1),1)
#            else:
#                organ_contour_A = torch.zeros_like(lung_contour)
#                organ_contour_B = torch.zeros_like(lung_contour) 
            
            # Run the data through the model to produce warp and flow field
            # Coarse network
            opt_coarse.zero_grad()
            flow_c, flow_c_up = model_coarse(input_A, input_B)
            warp_c = ST_coarse(input_A, flow_c_up)
#            warp_RTS_c = ST_coarse(organ_contour_A ,flow_c_up)
            if torch.max(lung_contour) > 0:
                mse_loss_c = lung_param * mse_loss_fn(input_B, warp_c, lung_contour) + mse_loss_fn(input_B, warp_c, body_contour)
            else:
                mse_loss_c = mse_loss_fn(input_B, warp_c, body_contour)
            mse_loss_c = mse_param * mse_loss_c
#            cc_loss_c = cc_loss_fn(input_B, warp_c, body_contour)
            reg_loss_c = reg_param * reg_loss_fn(flow_c)
#            DICE_loss_c = DICE_param * DICE_loss_fn(organ_contour_B, warp_RTS_c)
            grad_loss_c = grad_param * grad_loss_fn(flow_c)
            loss_c = mse_loss_c + reg_loss_c + grad_loss_c
            loss_c.backward()
            opt_coarse.step()
            
            # Fine network 32x32x32 [:,:,16:48,16:48,16:48]
#            low_limit = int((coarse_size[0]-fine_size[0])/2)
#            up_limit = int(low_limit + fine_size[0])
##            low_limit = 0
##            up_limit = 192
#            input_A_fine = input_A[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
#            input_B_fine = input_B[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
#            warp_input_fine = warp_c.detach()[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
##            organ_contour_B_fine = organ_contour_B[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
#            body_contour_fine = body_contour[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
#            lung_contour_fine = lung_contour[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
##            warp_RTS_input_fine = warp_RTS_c[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
##            
#            opt_fine.zero_grad()
#            flow_f, flow_f_up = model_fine(warp_input_fine, input_B_fine)
#            warp_f = ST_fine(warp_input_fine, flow_f_up)
##            warp_RTS_f = ST_fine(warp_RTS_input_fine ,flow_f_up)
#            if torch.max(lung_contour) > 0:
#                mse_loss_f = lung_param * mse_loss_fn(input_B_fine, warp_f, lung_contour_fine) + mse_loss_fn(input_B_fine, warp_f, body_contour_fine)
#            else:
#                mse_loss_f = mse_loss_fn(input_B_fine, warp_f, body_contour_fine)
#            mse_loss_f = mse_param * mse_loss_f
##            cc_loss_f = cc_loss_fn(input_B_fine, warp_f, body_contour_fine)
#            reg_loss_f = reg_param * reg_loss_fn(flow_f)
##            DICE_loss_f = DICE_param * DICE_loss_fn(organ_contour_B_fine, warp_RTS_f)
#            grad_loss_f = grad_param * grad_loss_fn(flow_f)
#            loss_f = mse_loss_f + reg_loss_f + grad_loss_f         
#            loss_f.backward(retain_graph=True)
#            opt_fine.step()
            
            # accu loss
            loss_c_accu = np.append(loss_c_accu, loss_c.item())
            mse_loss_c_accu = np.append(mse_loss_c_accu, mse_loss_c.item())
#            cc_loss_c_accu = np.append(cc_loss_c_accu, cc_loss_c.item())
            grad_loss_c_accu = np.append(grad_loss_c_accu, grad_loss_c.item())
            reg_loss_c_accu = np.append(reg_loss_c_accu, reg_loss_c.item())
#            DICE_loss_c_accu = np.append(DICE_loss_c_accu, DICE_loss_c.item())
#            loss_f_accu = np.append(loss_f_accu, loss_f.item())
#            mse_loss_f_accu = np.append(mse_loss_f_accu, mse_loss_f.item())
#            cc_loss_f_accu = np.append(cc_loss_f_accu, cc_loss_f.item())
#            grad_loss_f_accu = np.append(grad_loss_f_accu, grad_loss_f.item())
#            reg_loss_f_accu = np.append(reg_loss_f_accu, reg_loss_f.item())
#            DICE_loss_f_accu = np.append(DICE_loss_f_accu, DICE_loss_f.item())
      
            if (i == 0) & (num_patch == 0):
                print('saved\n')
                loss_c_out = np.append(loss_c_out, np.mean(loss_c_accu))
                mse_loss_c_out = np.append(mse_loss_c_out, np.mean(mse_loss_c_accu))
                cc_loss_c_out = np.append(cc_loss_c_out, np.mean(cc_loss_c_accu))
                reg_loss_c_out = np.append(reg_loss_c_out, np.mean(reg_loss_c_accu))
                DICE_loss_c_out = np.append(DICE_loss_c_out, np.mean(DICE_loss_c_accu))
                grad_loss_c_out = np.append(grad_loss_c_out, np.mean(grad_loss_c_accu))
                loss_f_out = np.append(loss_f_out, np.mean(loss_f_accu))
                mse_loss_f_out = np.append(mse_loss_f_out, np.mean(mse_loss_f_accu))
                cc_loss_f_out = np.append(cc_loss_f_out, np.mean(cc_loss_f_accu))
                reg_loss_f_out = np.append(reg_loss_f_out, np.mean(reg_loss_f_accu))
                DICE_loss_f_out = np.append(DICE_loss_f_out, np.mean(DICE_loss_f_accu))
                grad_loss_f_out = np.append(grad_loss_f_out, np.mean(grad_loss_f_accu))

            end = time.time()
            print("\r n_iter: %d/5, time = %fs, lr: %e\
                  coarse:\
                  loss_c: %f, mse_c: %f, cc_c: %f, reg_c:%f, grad_c:%f, DICE_c:%f\
                  fine:\
                  loss_f: %f, mse_f: %f, cc_f: %f, reg_f:%f, grad_f:%f, DICE_f:%f." 
              % (num_patch + 1, end-start, lr_new,
                 np.mean(loss_c_accu), np.mean(mse_loss_c_accu), np.mean(cc_loss_c_accu),
                 np.mean(reg_loss_c_accu), np.mean(grad_loss_c_accu), np.mean(DICE_loss_c_accu),
                 np.mean(loss_f_accu), np.mean(mse_loss_f_accu), np.mean(cc_loss_f_accu),
                 np.mean(reg_loss_f_accu), np.mean(grad_loss_f_accu), np.mean(DICE_loss_f_accu)
                 ), end = '', flush=True)
        loss_c_out = np.append(loss_c_out, loss_c_accu)
        mse_loss_c_out = np.append(mse_loss_c_out, mse_loss_c_accu)
        cc_loss_c_out = np.append(cc_loss_c_out, cc_loss_c_accu)
        reg_loss_c_out = np.append(reg_loss_c_out, reg_loss_c_accu)
        DICE_loss_c_out = np.append(DICE_loss_c_out, DICE_loss_c_accu)
        grad_loss_c_out = np.append(grad_loss_c_out, grad_loss_c_accu)
        loss_f_out = np.append(loss_f_out, loss_f_accu)
        mse_loss_f_out = np.append(mse_loss_f_out, mse_loss_f_accu)
        cc_loss_f_out = np.append(cc_loss_f_out, cc_loss_f_accu)
        reg_loss_f_out = np.append(reg_loss_f_out, reg_loss_f_accu)
        DICE_loss_f_out = np.append(DICE_loss_f_out, DICE_loss_f_accu)
        grad_loss_f_out = np.append(grad_loss_f_out, grad_loss_f_accu)
    return model_coarse, model_fine, loss_c_out, mse_loss_c_out, cc_loss_c_out, DICE_loss_c_out, grad_loss_c_out, loss_f_out, mse_loss_f_out, cc_loss_f_out, DICE_loss_f_out, grad_loss_f_out
