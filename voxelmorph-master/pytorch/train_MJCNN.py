# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:47:11 2020

@author: Haonan Xiao

E-mail: hx42@duke.edu
"""

import os

# external imports
import numpy as np
import torch
from torch.optim import Adam, SGD

# internal imports
from model import cvpr2018_net, SpatialTransformer
from model_C2F import Unet, JiangNet_Fine
import torch.nn.functional as F
import datagenerators
import losses
from math import floor
import time

def train_initial_single(
        gpu,
        params_initial,
        scale,
        vol_size,
        ST_size,
        lr,
        lr_schedule,
        cwd,
        data_folder,
        n_save_epoch,
        n_epoch,
        model_dir
        ):
    sim_param = params_initial[0]
    grad_param = params_initial[1]
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    
    model_x4 = JiangNet_Fine(dim=3, scale=1, residual=True)
    ST = SpatialTransformer(ST_size) 

    model_x4.to(device)
    ST.to(device)
    
    # Set optimizer and losses
    opt = SGD(model_x4.parameters(), lr=lr)

#    mse_loss_fn = losses.mse_loss_ReallynoRTS
    sim_loss_fn = losses.mse_loss_ReallynoRTS
    cc_loss_fn = losses.cc_loss_ReallynoRTS
    grad_loss_fn = losses.gradient_loss
    reg_loss_fn = losses.regulation_loss

    # data generator
    studies = os.listdir(data_folder)
    train_example_gen = datagenerators.gen_CT_RTS_v4(cwd, studies, data_folder, extreme = True)

    sim_loss_out = np.array([])
    grad_loss_out = np.array([])
    cc_loss_out = np.array([])

    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt = Adam(model_x4.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_epoch == 0:
            save_file_name_fine = os.path.join(model_dir, 'model_%d_%d.ckpt' % (scale, i))
            torch.save(model_x4.state_dict(), save_file_name_fine)
        sim_loss_accu = np.array([])
        grad_loss_accu = np.array([])
        cc_loss_accu = np.array([])
        reg_loss_accu = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
#        moving_patch, fixed_patch = next(train_example_gen)
        for num_patch in range(100):
            moving_patch, fixed_patch = next(train_example_gen)
            start = time.time()
            input_A = torch.from_numpy(moving_patch).to(device).float()
            # -1000 - -200
            input_A = (input_A+1000)/1000
            # 0 - 0.2
            input_A = F.interpolate(input_A, vol_size, mode = 'trilinear')
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B = (input_B+1000)/1000
            input_B = F.interpolate(input_B, vol_size, mode = 'trilinear')
            
##            input_A_1 = input_A[:,:,:,:,0:32]
##            input_B_1 = input_B[:,:,:,:,0:32]
#            input_A = input_A[:,:,:,:,32:64]
#            input_B = input_B[:,:,:,:,32:64]            
##            input_A_3 = input_A[:,:,:,:,64:96]
##            input_B_3 = input_B[:,:,:,:,64:96]
#            
##            input_A_slab = torch.cat([input_A_1, input_A_2, input_A_3], 0)
##            input_B_slab = torch.cat([input_B_1, input_B_2, input_B_3], 0)
#            input_A_slab = input_A_2
#            input_B_slab = input_B_2
            
            # Run the data through the model to produce warp and flow field
            flow, flow_up = model_x4(input_A, input_B)
            flow_up = flow_up
            warp_f = ST(input_A, flow_up)
            
            sim_loss = sim_param * sim_loss_fn(input_B, warp_f)
            cc_loss = sim_param * cc_loss_fn(input_B,warp_f)
            dvf = flow_up
            grad_loss = grad_param * grad_loss_fn(dvf)
            reg_loss = 0.1*grad_param * reg_loss_fn(dvf)
            
            loss = sim_loss + grad_loss + cc_loss + reg_loss
            opt.zero_grad()
            loss.backward()  
            opt.step()

            
            # accu loss
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            sim_loss_accu = np.append(sim_loss_accu, sim_loss.item())
            cc_loss_accu = np.append(cc_loss_accu, cc_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())
      
            if (i == 0) & (num_patch == 0):
                print('saved\n')
#                loss_c_out = np.append(loss_c_out, np.mean(loss_c_accu))
#                mse_loss_c_out = np.append(mse_loss_c_out, np.mean(mse_loss_c_accu))
#                loss_f_out = np.append(loss_f_out, np.mean(loss_f_accu))
#                mse_loss_f_out = np.append(mse_loss_f_out, np.mean(mse_loss_f_accu))
                sim_loss_out = np.append(sim_loss_out, np.mean(sim_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))

            end = time.time()
            print("\r n_iter: %d/100, time = %fs, lr: %e, sim: %f, cc:%f, grad:%f, reg:%f" 
              % (num_patch + 1, end-start, lr_new, np.mean(sim_loss_accu),np.mean(cc_loss_accu), np.mean(grad_loss_accu), np.mean(reg_loss_accu)
                 ), end = '', flush=True)
        sim_loss_out = np.append(sim_loss_out, sim_loss_accu)
        grad_loss_out = np.append(grad_loss_out, grad_loss_accu)
        cc_loss_out = np.append(cc_loss_out, cc_loss_accu)
    return model_x4,sim_loss_out,cc_loss_out,grad_loss_out

def train_initial_4(
        gpu,
        params_initial,
        scale,
        vol_size,
        ST_size,
        lr,
        lr_schedule,
        cwd,
        data_folder,
        n_save_epoch,
        n_epoch,
        model_dir
        ):
    sim_param = params_initial[0]
    grad_param = params_initial[1]
    reg_param = params_initial[2]
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    
    model_x4 = JiangNet_Fine(dim=3, scale=4, residual=False)
    ST = SpatialTransformer(ST_size) 

    model_x4.to(device)
    ST.to(device)
    
    # Set optimizer and losses
    opt = Adam(model_x4.parameters(), lr=lr)

#    mse_loss_fn = losses.mse_loss_ReallynoRTS
    sim_loss_fn = losses.ncc_loss
    grad_loss_fn = losses.gradient_loss
    reg_loss_fn = losses.regulation_loss

    # data generator
    studies = os.listdir(data_folder)
    train_example_gen = datagenerators.gen_CT_RTS_v4(cwd, studies, data_folder, extreme = True)

    sim_loss_out = np.array([])
    grad_loss_out = np.array([])
    reg_loss_out = np.array([])

    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt = Adam(model_x4.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_epoch == 0:
            save_file_name_fine = os.path.join(model_dir, 'model_%d_%d.ckpt' % (scale, i))
            torch.save(model_x4.state_dict(), save_file_name_fine)
        sim_loss_accu = np.array([])
        grad_loss_accu = np.array([])
        reg_loss_accu = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
#        moving_patch, fixed_patch = next(train_example_gen)
        for num_patch in range(100):
            moving_patch, fixed_patch = next(train_example_gen)
            start = time.time()
            input_A = torch.from_numpy(moving_patch).to(device).float()
            # -1000 - -200
            input_A = (input_A+1000)/1000
            # 0 - 0.2
            input_A = F.interpolate(input_A, vol_size, mode = 'trilinear')
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B = (input_B+1000)/1000
            input_B = F.interpolate(input_B, vol_size, mode = 'trilinear')
        
            
            # Run the data through the model to produce warp and flow field
            flow, flow_up = model_x4(input_A, input_B)
            flow_up = flow_up * 4
            warp_x4 = ST(input_A, flow_up)
            
            sim_loss = sim_param * sim_loss_fn(input_B, warp_x4)
            dvf = flow_up
            grad_loss = grad_param * grad_loss_fn(dvf)
            reg_loss = reg_param * reg_loss_fn(dvf)
            
            loss = sim_loss + grad_loss + reg_loss
            opt.zero_grad()
            loss.backward()  
            opt.step()

            
            # accu loss
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            sim_loss_accu = np.append(sim_loss_accu, sim_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())

      
            if (i == 0) & (num_patch == 0):
                print('saved\n')
#                loss_c_out = np.append(loss_c_out, np.mean(loss_c_accu))
#                mse_loss_c_out = np.append(mse_loss_c_out, np.mean(mse_loss_c_accu))
#                loss_f_out = np.append(loss_f_out, np.mean(loss_f_accu))
#                mse_loss_f_out = np.append(mse_loss_f_out, np.mean(mse_loss_f_accu))
                sim_loss_out = np.append(sim_loss_out, np.mean(sim_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))
                reg_loss_out = np.append(reg_loss_out, reg_loss_accu)
            end = time.time()
            print("\r n_iter: %d/100, time = %fs, lr: %e, sim: %f, grad:%f, reg:%f." 
              % (num_patch + 1, end-start, lr_new, np.mean(sim_loss_accu),np.mean(grad_loss_accu),np.mean(reg_loss_accu)
                 ), end = '', flush=True)
        sim_loss_out = np.append(sim_loss_out, sim_loss_accu)
        grad_loss_out = np.append(grad_loss_out, grad_loss_accu)
        reg_loss_out = np.append(reg_loss_out, reg_loss_accu)
    return model_x4,sim_loss_out,grad_loss_out, reg_loss_accu

def train_initial_2(
        model_x4,
        gpu,
        params_initial,
        scale,
        vol_size,
        ST_size,
        lr,
        lr_schedule,
        cwd,
        data_folder,
        n_save_epoch,
        n_epoch,
        model_dir
        ):
    sim_param = params_initial[0]
    grad_param = params_initial[1]
    reg_param = params_initial[2]
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    model_x2 = JiangNet_Fine(dim=3, scale=2, residual=True)
    ST = SpatialTransformer(ST_size) 

    model_x2.to(device)
    ST.to(device)
    
    # Set optimizer and losses
    opt = Adam(model_x2.parameters(), lr=lr)

#    mse_loss_fn = losses.mse_loss_ReallynoRTS
    sim_loss_fn = losses.ncc_loss
    grad_loss_fn = losses.gradient_loss
    reg_loss_fn = losses.regulation_loss

    # data generator
    studies = os.listdir(data_folder)
    train_example_gen = datagenerators.gen_CT_RTS_v4(cwd, studies, data_folder, extreme = True)

    sim_loss_out = np.array([])
    grad_loss_out = np.array([])
    reg_loss_out = np.array([])
    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt = Adam(model_x2.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_epoch == 0:
            save_file_name_fine = os.path.join(model_dir, 'model_%d_%d.ckpt' % (scale, i))
            torch.save(model_x2.state_dict(), save_file_name_fine)
        sim_loss_accu = np.array([])
        grad_loss_accu = np.array([])
        reg_loss_accu = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
#        moving_patch, fixed_patch = next(train_example_gen)
        for num_patch in range(100):
            moving_patch, fixed_patch = next(train_example_gen)
            start = time.time()
            input_A = torch.from_numpy(moving_patch).to(device).float()
            # -1000 - -200
            input_A = (input_A+1000)/1000
            # 0 - 0.2
            input_A = F.interpolate(input_A, vol_size, mode = 'trilinear')
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B = (input_B+1000)/1000
            input_B = F.interpolate(input_B, vol_size, mode = 'trilinear')
            
            
            # Run the data through the model to produce warp and flow field
            with torch.no_grad():
                flow, flow_up = model_x4(input_A, input_B)
                flow_up_x4 = flow_up * 4
                warp_x4 = ST(input_A, flow_up_x4)
            flow, flow_up = model_x2(warp_x4.detach(), input_B)
            flow_up_x2 = flow_up * 2
            warp_x2 = ST(warp_x4, flow_up_x2)
            sim_loss = sim_param * sim_loss_fn(input_B, warp_x2)
            dvf = flow_up_x4 + flow_up_x2
            grad_loss = grad_param * grad_loss_fn(dvf)
            reg_loss = reg_param * reg_loss_fn(dvf)
            
            loss = sim_loss + grad_loss + reg_loss
            opt.zero_grad()
            loss.backward()  
            opt.step()
            
            # accu loss
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            sim_loss_accu = np.append(sim_loss_accu, sim_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())
      
            if (i == 0) & (num_patch == 0):
                print('saved\n')
#                loss_c_out = np.append(loss_c_out, np.mean(loss_c_accu))
#                mse_loss_c_out = np.append(mse_loss_c_out, np.mean(mse_loss_c_accu))
#                loss_f_out = np.append(loss_f_out, np.mean(loss_f_accu))
#                mse_loss_f_out = np.append(mse_loss_f_out, np.mean(mse_loss_f_accu))
                sim_loss_out = np.append(sim_loss_out, np.mean(sim_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))

            end = time.time()
            print("\r n_iter: %d/100, time = %fs, lr: %e, sim: %f, grad:%f, reg:%f." 
              % (num_patch + 1, end-start, lr_new, np.mean(sim_loss_accu),np.mean(grad_loss_accu),np.mean(reg_loss_accu)
                 ), end = '', flush=True)
        sim_loss_out = np.append(sim_loss_out, sim_loss_accu)
        grad_loss_out = np.append(grad_loss_out, grad_loss_accu)
        reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
    return model_x2,sim_loss_out,grad_loss_out,reg_loss_out

def train_initial_1(
        model_x4,
        model_x2,
        gpu,
        params_initial,
        scale,
        vol_size,
        ST_size,
        lr,
        lr_schedule,
        cwd,
        data_folder,
        n_save_epoch,
        n_epoch,
        model_dir
        ):
    sim_param = params_initial[0]
    grad_param = params_initial[1]
    reg_param = params_initial[2]
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    model_x1 = JiangNet_Fine(dim=3, scale=1, residual=True)
    ST = SpatialTransformer(ST_size) 

    model_x1.to(device)
    ST.to(device)
    
    # Set optimizer and losses
    opt = Adam(model_x1.parameters(), lr=lr)

#    mse_loss_fn = losses.mse_loss_ReallynoRTS
    sim_loss_fn = losses.ncc_loss
    grad_loss_fn = losses.gradient_loss
    reg_loss_fn = losses.regulation_loss

    # data generator
    studies = os.listdir(data_folder)
    train_example_gen = datagenerators.gen_CT_RTS_v4(cwd, studies, data_folder, extreme = True)

    sim_loss_out = np.array([])
    grad_loss_out = np.array([])
    reg_loss_out = np.array([])

    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt = Adam(model_x1.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_epoch == 0:
            save_file_name_fine = os.path.join(model_dir, 'model_%d_%d.ckpt' % (scale, i))
            torch.save(model_x1.state_dict(), save_file_name_fine)
        sim_loss_accu = np.array([])
        grad_loss_accu = np.array([])
        reg_loss_accu = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
#        moving_patch, fixed_patch = next(train_example_gen)
        for num_patch in range(100):
            moving_patch, fixed_patch = next(train_example_gen)
            start = time.time()
            input_A = torch.from_numpy(moving_patch).to(device).float()
            # -1000 - -200
            input_A = (input_A+1000)/1000
            # 0 - 0.2
            input_A = F.interpolate(input_A, vol_size, mode = 'trilinear')
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B = (input_B+1000)/1000
            input_B = F.interpolate(input_B, vol_size, mode = 'trilinear')
            
            
            # Run the data through the model to produce warp and flow field
            with torch.no_grad():
                flow, flow_up = model_x4(input_A, input_B)
                flow_up_x4 = flow_up * 4
                warp_x4 = ST(input_A, flow_up_x4)
                flow, flow_up = model_x2(warp_x4, input_B)
                flow_up_x2 = flow_up * 2
                warp_x2 = ST(warp_x4, flow_up_x2)
            
            flow, flow_up = model_x1(warp_x2.detach(), input_B)
            flow_up_x1 = flow_up * 1
            warp_x1 = ST(warp_x2.detach(), flow_up_x1)
            sim_loss = sim_param * sim_loss_fn(input_B, warp_x1)
            dvf = flow_up_x4 + flow_up_x2 + flow_up_x1
            grad_loss = grad_param * grad_loss_fn(dvf)
            reg_loss = reg_param * reg_loss_fn(dvf)
            
            loss = sim_loss + grad_loss + reg_loss
            opt.zero_grad()
            loss.backward()  
            opt.step()

            
            # accu loss
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            sim_loss_accu = np.append(sim_loss_accu, sim_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())

      
            if (i == 0) & (num_patch == 0):
                print('saved\n')
#                loss_c_out = np.append(loss_c_out, np.mean(loss_c_accu))
#                mse_loss_c_out = np.append(mse_loss_c_out, np.mean(mse_loss_c_accu))
#                loss_f_out = np.append(loss_f_out, np.mean(loss_f_accu))
#                mse_loss_f_out = np.append(mse_loss_f_out, np.mean(mse_loss_f_accu))
                sim_loss_out = np.append(sim_loss_out, np.mean(sim_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))

            end = time.time()
            print("\r n_iter: %d/100, time = %fs, lr: %e, sim: %f, grad:%f, reg:%f." 
              % (num_patch + 1, end-start, lr_new, np.mean(sim_loss_accu),np.mean(grad_loss_accu),np.mean(reg_loss_accu)
                 ), end = '', flush=True)
        sim_loss_out = np.append(sim_loss_out, sim_loss_accu)
        grad_loss_out = np.append(grad_loss_out, grad_loss_accu)
        reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
    return model_x1,sim_loss_out,grad_loss_out, reg_loss_out

def train_joint(
        gpu,
        model_x4,
        model_x2,
        model_x1,
        params_joint,
        lr,
        vol_size,
        ST_size,
        cwd,
        data_folder,
        lr_schedule,
        n_epoch,
        n_save_epoch,
        model_dir
        ):
    
    sim_param_x4 = params_joint[0]
    sim_param_x2 = params_joint[1]
    sim_param_x1 = params_joint[2]
    grad_param = params_joint[3]
    reg_param = params_joint[4]
    
    model_parameters = list(model_x4.parameters()) + list(model_x2.parameters()) + list(model_x1.parameters())
    opt = Adam(model_parameters, lr=lr)
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    
    ST = SpatialTransformer(ST_size) 
    ST.to(device)
    
    sim_loss_fn = losses.ncc_loss
    grad_loss_fn = losses.gradient_loss
    reg_loss_fn = losses.regulation_loss
    # data generator
    studies = os.listdir(data_folder)
    train_example_gen = datagenerators.gen_CT_RTS_v4(cwd, studies, data_folder, extreme = True)

    sim_loss_x4_out = np.array([])
    sim_loss_x2_out = np.array([])
    sim_loss_x1_out = np.array([])
    sim_loss_out = np.array([])
    grad_loss_out = np.array([])
    reg_loss_out = np.array([])
    
    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt = Adam(model_parameters, lr=lr_new)
        # Save model checkpoint
        if i % n_save_epoch == 0:
            save_file_name_fine = os.path.join(model_dir, 'model_4_%d.ckpt' %  i)
            torch.save(model_x4.state_dict(), save_file_name_fine)
            save_file_name_fine = os.path.join(model_dir, 'model_2_%d.ckpt' %  i)
            torch.save(model_x2.state_dict(), save_file_name_fine)
            save_file_name_fine = os.path.join(model_dir, 'model_1_%d.ckpt' %  i)
            torch.save(model_x1.state_dict(), save_file_name_fine)
        sim_loss_accu = np.array([])
        sim_loss_x4_accu = np.array([])
        sim_loss_x2_accu = np.array([])
        sim_loss_x1_accu = np.array([])
        grad_loss_accu = np.array([])
        reg_loss_accu = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
#        moving_patch, fixed_patch = next(train_example_gen)
        for num_patch in range(100):
            moving_patch, fixed_patch = next(train_example_gen)
            start = time.time()
            input_A = torch.from_numpy(moving_patch).to(device).float()
            # -1000 - -200
            input_A = (input_A+1000)/1000
            # 0 - 0.2
            input_A = F.interpolate(input_A, vol_size, mode = 'trilinear')
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B = (input_B+1000)/1000
            input_B = F.interpolate(input_B, vol_size, mode = 'trilinear')
            
            # Run the data through the model to produce warp and flow field
            # x4
            flow_x4, flow_up_x4 = model_x4(input_A, input_B)
            flow_up_x4 = flow_up_x4 * 4
            warp_x4 = ST(input_A, flow_up_x4)
            sim_loss_x4 = sim_param_x4 * sim_loss_fn(input_B, warp_x4)
            # x2
            flow_x2, flow_up_x2 = model_x2(warp_x4, input_B)
            flow_up_x2 = flow_up_x2 * 2
            warp_x2 = ST(warp_x4, flow_up_x2)
            sim_loss_x2 = sim_param_x2 * sim_loss_fn(input_B, warp_x2)
            #x1
            flow_x1, flow_up_x1 = model_x1(warp_x2, input_B)
            flow_up_x1 = flow_up_x1 * 1
            warp_x1 = ST(warp_x2, flow_up_x1)
            sim_loss_x1 = sim_param_x1 * sim_loss_fn(input_B, warp_x1)            
            
            dvf = flow_up_x4 + flow_up_x2 + flow_up_x1
            grad_loss = grad_param * grad_loss_fn(dvf)
            reg_loss = reg_param * reg_loss_fn(dvf)
            sim_loss = sim_loss_x4 + sim_loss_x2 + sim_loss_x1
            loss = sim_loss + grad_loss + reg_loss
            opt.zero_grad()
            loss.backward()  
            opt.step()

            
            # accu loss
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())
            sim_loss_accu = np.append(sim_loss_accu, sim_loss.item())
            sim_loss_x4_accu = np.append(sim_loss_x4_accu, sim_loss_x4.item())
            sim_loss_x2_accu = np.append(sim_loss_x2_accu, sim_loss_x2.item())
            sim_loss_x1_accu = np.append(sim_loss_x1_accu, sim_loss_x1.item())

      
            if (i == 0) & (num_patch == 0):
                print('saved\n')
                sim_loss_out = np.append(sim_loss_out, np.mean(sim_loss_accu))
                sim_loss_x4_out = np.append(sim_loss_x4_out, np.mean(sim_loss_x4_accu))
                sim_loss_x2_out = np.append(sim_loss_x2_out, np.mean(sim_loss_x2_accu))
                sim_loss_x1_out = np.append(sim_loss_x1_out, np.mean(sim_loss_x1_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))

            end = time.time()
            print("\r n_iter: %d/100, time = %fs, lr: %e, sim: %f, grad:%f, reg:%f." 
              % (num_patch + 1, end-start, lr_new, np.mean(sim_loss_accu),np.mean(grad_loss_accu),np.mean(reg_loss_accu)
                 ), end = '', flush=True)
        sim_loss_out = np.append(sim_loss_out, np.mean(sim_loss_accu))
        sim_loss_x4_out = np.append(sim_loss_x4_out, np.mean(sim_loss_x4_accu))
        sim_loss_x2_out = np.append(sim_loss_x2_out, np.mean(sim_loss_x2_accu))
        sim_loss_x1_out = np.append(sim_loss_x1_out, np.mean(sim_loss_x1_accu))
        grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))
        reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))

    return model_x4, model_x2, model_x1, sim_loss_x4_out, sim_loss_x2_out, sim_loss_x1_out, sim_loss_out, grad_loss_out, reg_loss_out
