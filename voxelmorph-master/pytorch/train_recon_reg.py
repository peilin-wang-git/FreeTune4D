# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:11:09 2022

@author: user
"""

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
import torch.nn as nn
from torch.optim import Adam
# internal imports
from model_recon_reg import D2R_Net, SpatialTransformer
import torch.nn.functional as F
import functions
import losses
from math import floor
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import random
import math


def train(model_parameters):
    
    vol_size, enc_nf, dec_nf, data_path, data_list, lr, lr_schedule, n_epoch, params, n_save_epoch, n_plot_epoch, model_dir, batch_size, base_path, split_ratio = model_parameters
    
    DVF_param = params[0]
    sim_param = params[1]
    reg_param = params[2]
    mae_param = params[3]
    gpu = '0'
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    
    model = D2R_Net(enc_nf, dec_nf)
    ST = SpatialTransformer(vol_size) 
    model.to(device)
    ST.to(device)
    
    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)
    mae_loss_fn = losses.mae_loss
    ncc_loss_fn = losses.ncc_loss
    mse_loss_fn = losses.mse_loss_ReallynoRTS
    reg_loss_fn = losses.gradient_loss
    
    # data generator
    example_gen = functions.datagenerator_Downsample_inverse(data_path, data_list, split_ratio, vol_size, device)
    num_train = example_gen.num_train
    num_valid = example_gen.num_valid

    DVF_loss_out = np.array([])
    sim_loss_out = np.array([])
    reg_loss_out = np.array([])
    mae_loss_out = np.array([])
    
    DVF_val_out = np.array([])
    mse_val_out = np.array([])
    ncc_val_out = np.array([])
    mae_val_out = np.array([])
    
    DVF_val_loss, mse_val_loss, ncc_val_loss, mae_val_loss = validation(model, example_gen, ST, 0, model_dir)
    print("\nValidation: DVF_loss: %f, mse_loss: %f, cc_loss: %f, mae_loss: %f." % 
          (np.mean(DVF_val_loss),np.mean(mse_val_loss), np.mean(ncc_val_loss), np.mean(mae_val_loss)))
    DVF_val_out = np.append(DVF_val_out, DVF_val_loss)
    mse_val_out = np.append(mse_val_out, mse_val_loss)
    ncc_val_out = np.append(ncc_val_out, ncc_val_loss)
    mae_val_out = np.append(mae_val_out, mae_val_loss)
    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.7, num_reduced_factor)
        opt = Adam(model.parameters(), lr=lr_new)
        DVF_loss_epoch = np.array([])
        sim_loss_epoch = np.array([])
        reg_loss_epoch = np.array([])
        mae_loss_epoch = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
        start = time.time()
        for num_image in range(num_train):
            input_mov, input_fix, input_fix_down, dvf_resize = example_gen.train_generator(num_image)
            fix_SR, flow = model(input_mov, input_fix_down)
            warp = ST(input_mov, flow)
            
            mae_loss = mae_param * mae_loss_fn(input_fix, fix_SR)
            DVF_loss = DVF_param * mse_loss_fn(flow, dvf_resize)
            sim_loss = sim_param * ncc_loss_fn(warp, input_fix)
            reg_loss = reg_param * reg_loss_fn(flow)                
            loss = DVF_loss + sim_loss + reg_loss + mae_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            # accu loss
            DVF_loss_epoch = np.append(DVF_loss_epoch, DVF_loss.item())
            sim_loss_epoch = np.append(sim_loss_epoch, sim_loss.item())
            reg_loss_epoch = np.append(reg_loss_epoch, reg_loss.item())
            mae_loss_epoch = np.append(mae_loss_epoch, mae_loss.item())
            end = time.time()
            print("\r n_image: %d/%d, time = %fs, lr: %e, DVF: %f, sim: %f, reg:%f, mae:%f." 
                  % (num_image+1, num_train, end-start, lr_new, np.mean(DVF_loss_epoch),
                     np.mean(sim_loss_epoch), np.mean(reg_loss_epoch), np.mean(mae_loss_epoch)),
                     end = '', flush=True)
        DVF_loss_out = np.append(DVF_loss_out, np.mean(DVF_loss_epoch))
        sim_loss_out = np.append(sim_loss_out, np.mean(sim_loss_epoch))
        reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_epoch))
        mae_loss_out = np.append(mae_loss_out, np.mean(mae_loss_epoch))
        DVF_val_loss, mse_val_loss, ncc_val_loss, mae_val_loss = validation(model,example_gen, ST, i, model_dir)
        DVF_val_out = np.append(DVF_val_out, DVF_val_loss)
        mse_val_out = np.append(mse_val_out, mse_val_loss)
        ncc_val_out = np.append(ncc_val_out, ncc_val_loss)
        mae_val_out = np.append(mae_val_out, mae_val_loss)
        #DVF_val_loss, sim_val_loss = validation(model,cwd,validation_folder,params,patch_shape,patch_size,stride,pad)
        print("\nValidation: DVF_loss: %f, mse_loss: %f, cc_loss: %f, mae_loss: %f." % 
              (np.mean(DVF_val_loss),np.mean(mse_val_loss), np.mean(ncc_val_loss), np.mean(mae_val_loss)))
        if i % n_save_epoch == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % i)
            torch.save(model.state_dict(), save_file_name)
            
        loss = [DVF_loss_out, sim_loss_out, reg_loss_out, mae_loss_out, DVF_val_out, mse_val_out, ncc_val_out, mae_val_out]
    return model, loss

def validation(model, example_gen, ST, i, model_dir, batch_size = 1):

    mse_loss_fn = losses.mse_loss_ReallynoRTS
    ncc_loss_fn = losses.ncc_loss
    mae_loss_fn = losses.mae_loss
    reg_loss_fn = losses.gradient_loss
    DVF_loss_val = np.array([])
    mse_loss_after = np.array([])
    ncc_loss_after = np.array([])
    mae_loss_after = np.array([])
    valid_iter = math.floor(example_gen.num_valid/batch_size)
    
    for sty_index in range(valid_iter):
        input_mov, input_fix, input_fix_down, dvf_resize = example_gen.valid_generator(sty_index, batch_size)
        # difference map and gradient map
        with torch.no_grad():
            fix_SR, flow = model(input_mov, input_fix_down)
            warp = ST(input_mov, flow)
        mae_loss = mae_loss_fn(input_fix, fix_SR)
        DVF_loss = mse_loss_fn(flow, dvf_resize)
        ncc_loss = ncc_loss_fn(warp, input_fix)
        mse_loss = mse_loss_fn(warp, input_fix)
        reg_loss = reg_loss_fn(flow)   
        DVF_loss_val = np.append(DVF_loss_val, DVF_loss.item())
        mse_loss_after = np.append(mse_loss_after, mse_loss.item())
        ncc_loss_after = np.append(ncc_loss_after, ncc_loss.item())
        mae_loss_after = np.append(mae_loss_after, mae_loss.item())
        

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(input_fix_down.cpu().numpy()[0,0,:,:,30], cmap='gray')
    plt.title(f"DS epoch {i}")          
    plt.axis('off')
    
    plt.subplot(1,3,2)
    plt.imshow(fix_SR.detach().cpu().numpy()[0,0,:,:,30], cmap='gray')
    plt.title(f"SR epoch {i}")         
    plt.axis('off')
    
    plt.subplot(1,3,3)
    plt.imshow(input_fix.cpu().numpy()[0,0,:,:,30], cmap='gray')
    plt.title(f"GT epoch {i}")         
    plt.axis('off')
    save_path = model_dir + f"epoch {i}.png"
    if os.path.isfile(save_path):
        save_path = model_dir + f"epoch {i}_1.png"
    else:
        pass
    plt.savefig(save_path, bbox_inches='tight')
        # accu loss
    return np.mean(DVF_loss_val),np.mean(mse_loss_after), np.mean(ncc_loss_after), np.mean(mae_loss_after)


def train_continue(model_path, loss_path, n_epoch_trained, model_parameters):
    
    vol_size, enc_nf, dec_nf, data_path, data_list, lr, lr_schedule, n_epoch, params, n_save_epoch, n_plot_epoch, model_dir, batch_size, base_path, split_ratio = model_parameters
    
    loss = np.load(loss_path,allow_pickle=True)
    
    DVF_loss_out, sim_loss_out, reg_loss_out, mae_loss_out, DVF_val_out, mse_val_out, ncc_val_out, mae_val_out = loss
    
    DVF_param = params[0]
    sim_param = params[1]
    reg_param = params[2]
    mae_param = params[3]
    gpu = '0'
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    
    model = D2R_Net(enc_nf, dec_nf)
    ST = SpatialTransformer(vol_size) 
    model.to(device)
    ST.to(device)
    model.load_state_dict(torch.load(model_path))
    
    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)
    mae_loss_fn = losses.mae_loss
    ncc_loss_fn = losses.ncc_loss
    mse_loss_fn = losses.mse_loss_ReallynoRTS
    reg_loss_fn = losses.gradient_loss
    
    # data generator
    example_gen = functions.datagenerator_Downsample_inverse(data_path, data_list, split_ratio, vol_size, device)
    num_train = example_gen.num_train
    num_valid = example_gen.num_valid
    
    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.8, num_reduced_factor)
        opt = Adam(model.parameters(), lr=lr_new)
        DVF_loss_epoch = np.array([])
        sim_loss_epoch = np.array([])
        reg_loss_epoch = np.array([])
        mae_loss_epoch = np.array([])
        print('\nEpoch %d/%d' % (i+1+n_epoch_trained,n_epoch+n_epoch_trained))
        start = time.time()
        for num_image in range(num_train):
            input_mov, input_fix, input_fix_down, dvf_resize = example_gen.train_generator(num_image)
            fix_SR, flow = model(input_mov, input_fix_down)
            warp = ST(input_mov, flow)
            
            mae_loss = mae_param * mae_loss_fn(input_fix, fix_SR)
            DVF_loss = DVF_param * mse_loss_fn(flow, dvf_resize)
            sim_loss = sim_param * ncc_loss_fn(warp, input_fix)
            reg_loss = reg_param * reg_loss_fn(flow)                
            loss = DVF_loss + sim_loss + reg_loss + mae_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            # accu loss
            DVF_loss_epoch = np.append(DVF_loss_epoch, DVF_loss.item())
            sim_loss_epoch = np.append(sim_loss_epoch, sim_loss.item())
            reg_loss_epoch = np.append(reg_loss_epoch, reg_loss.item())
            mae_loss_epoch = np.append(mae_loss_epoch, mae_loss.item())
            end = time.time()
            print("\r n_image: %d/%d, time = %fs, lr: %e, DVF: %f, sim: %f, reg:%f, mae:%f." 
                  % (num_image+1, num_train, end-start, lr_new, np.mean(DVF_loss_epoch),
                     np.mean(sim_loss_epoch), np.mean(reg_loss_epoch), np.mean(mae_loss_epoch)),
                     end = '', flush=True)
        DVF_loss_out = np.append(DVF_loss_out, np.mean(DVF_loss_epoch))
        sim_loss_out = np.append(sim_loss_out, np.mean(sim_loss_epoch))
        reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_epoch))
        mae_loss_out = np.append(mae_loss_out, np.mean(mae_loss_epoch))
        DVF_val_loss, mse_val_loss, ncc_val_loss, mae_val_loss = validation(model,example_gen, ST, i+n_epoch_trained, model_dir)
        DVF_val_out = np.append(DVF_val_out, DVF_val_loss)
        mse_val_out = np.append(mse_val_out, mse_val_loss)
        ncc_val_out = np.append(ncc_val_out, ncc_val_loss)
        mae_val_out = np.append(mae_val_out, mae_val_loss)
        print("\nValidation: DVF_loss: %f, mse_loss: %f, cc_loss: %f, mae_loss: %f." % 
              (np.mean(DVF_val_loss),np.mean(mse_val_loss), np.mean(ncc_val_loss), np.mean(mae_val_loss)))
        if i+n_epoch_trained % n_save_epoch == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % (i+n_epoch_trained))
            torch.save(model.state_dict(), save_file_name)
            
        loss = [DVF_loss_out, sim_loss_out, reg_loss_out, mae_loss_out, DVF_val_out, mse_val_out, ncc_val_out, mae_val_out]
    return model, loss

def train_parallel(model_parameters):
    
    vol_size, enc_nf, dec_nf, data_path, data_list, lr, lr_schedule, n_epoch, params, n_save_epoch, n_plot_epoch, model_dir, batch_size, base_path, split_ratio = model_parameters
    
    DVF_param = params[0]
    sim_param = params[1]
    reg_param = params[2]
    mae_param = params[3]
    gpu = '0,1'
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    model = D2R_Net(enc_nf, dec_nf)
    ST = SpatialTransformer(vol_size) 
    model.cuda()
    ST.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    
    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)
    mae_loss_fn = losses.mae_loss
    ncc_loss_fn = losses.ncc_loss
    mse_loss_fn = losses.mse_loss_ReallynoRTS
    reg_loss_fn = losses.gradient_loss
    
    # data generator
    example_gen = functions.datagenerator_Downsample_inverse_parallel(data_path, data_list, split_ratio, vol_size)
    num_train = example_gen.num_train
    num_valid = example_gen.num_valid

    DVF_loss_out = np.array([])
    sim_loss_out = np.array([])
    reg_loss_out = np.array([])
    mae_loss_out = np.array([])
    
    DVF_val_out = np.array([])
    mse_val_out = np.array([])
    ncc_val_out = np.array([])
    mae_val_out = np.array([])
    
    DVF_val_loss, mse_val_loss, ncc_val_loss, mae_val_loss = validation(model, example_gen, ST, 0, model_dir, batch_size=2)
    print("\nValidation: DVF_loss: %f, mse_loss: %f, cc_loss: %f, mae_loss: %f." % 
          (np.mean(DVF_val_loss),np.mean(mse_val_loss), np.mean(ncc_val_loss), np.mean(mae_val_loss)))
    DVF_val_out = np.append(DVF_val_out, DVF_val_loss)
    mse_val_out = np.append(mse_val_out, mse_val_loss)
    ncc_val_out = np.append(ncc_val_out, ncc_val_loss)
    mae_val_out = np.append(mae_val_out, mae_val_loss)
    
    train_iter = math.floor(example_gen.num_train/batch_size)
    
    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.7, num_reduced_factor)
        opt = Adam(model.parameters(), lr=lr_new)
        DVF_loss_epoch = np.array([])
        sim_loss_epoch = np.array([])
        reg_loss_epoch = np.array([])
        mae_loss_epoch = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
        start = time.time()
        for num_image in range(train_iter):
            input_mov, input_fix, input_fix_down, dvf_resize = example_gen.train_generator(num_image)
            fix_SR, flow = model(input_mov, input_fix_down)
            warp = ST(input_mov, flow)
            
            mae_loss = mae_param * mae_loss_fn(input_fix, fix_SR)
            DVF_loss = DVF_param * mse_loss_fn(flow, dvf_resize)
            sim_loss = sim_param * ncc_loss_fn(warp, input_fix)
            reg_loss = reg_param * reg_loss_fn(flow)                
            loss = DVF_loss + sim_loss + reg_loss + mae_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            # accu loss
            DVF_loss_epoch = np.append(DVF_loss_epoch, DVF_loss.item())
            sim_loss_epoch = np.append(sim_loss_epoch, sim_loss.item())
            reg_loss_epoch = np.append(reg_loss_epoch, reg_loss.item())
            mae_loss_epoch = np.append(mae_loss_epoch, mae_loss.item())
            end = time.time()
            print("\r n_image: %d/%d, time = %fs, lr: %e, DVF: %f, sim: %f, reg:%f, mae:%f." 
                  % (num_image+1, train_iter, end-start, lr_new, np.mean(DVF_loss_epoch),
                     np.mean(sim_loss_epoch), np.mean(reg_loss_epoch), np.mean(mae_loss_epoch)),
                     end = '', flush=True)
        DVF_loss_out = np.append(DVF_loss_out, np.mean(DVF_loss_epoch))
        sim_loss_out = np.append(sim_loss_out, np.mean(sim_loss_epoch))
        reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_epoch))
        mae_loss_out = np.append(mae_loss_out, np.mean(mae_loss_epoch))
        DVF_val_loss, mse_val_loss, ncc_val_loss, mae_val_loss = validation(model,example_gen, ST, i, model_dir, batch_size=2)
        DVF_val_out = np.append(DVF_val_out, DVF_val_loss)
        mse_val_out = np.append(mse_val_out, mse_val_loss)
        ncc_val_out = np.append(ncc_val_out, ncc_val_loss)
        mae_val_out = np.append(mae_val_out, mae_val_loss)
        #DVF_val_loss, sim_val_loss = validation(model,cwd,validation_folder,params,patch_shape,patch_size,stride,pad)
        print("\nValidation: DVF_loss: %f, mse_loss: %f, cc_loss: %f, mae_loss: %f." % 
              (np.mean(DVF_val_loss),np.mean(mse_val_loss), np.mean(ncc_val_loss), np.mean(mae_val_loss)))
        if i % n_save_epoch == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % (i))
            torch.save(model.state_dict(), save_file_name)
            loss = [DVF_loss_out, sim_loss_out, reg_loss_out, mae_loss_out, DVF_val_out, mse_val_out, ncc_val_out, mae_val_out]
            save_npy_name = os.path.join(model_dir, '%d.npy' % (i))
            np.save(save_npy_name, loss)
        loss = [DVF_loss_out, sim_loss_out, reg_loss_out, mae_loss_out, DVF_val_out, mse_val_out, ncc_val_out, mae_val_out]
    return model, loss


def train_continue_parallel(model_path, loss_path, n_epoch_trained, model_parameters):
    
    vol_size, enc_nf, dec_nf, data_path, data_list, lr, lr_schedule, n_epoch, params, n_save_epoch, n_plot_epoch, model_dir, batch_size, base_path, split_ratio = model_parameters
    
    loss = np.load(loss_path,allow_pickle=True)
    
    DVF_loss_out, sim_loss_out, reg_loss_out, mae_loss_out, DVF_val_out, mse_val_out, ncc_val_out, mae_val_out = loss
    
    DVF_param = params[0]
    sim_param = params[1]
    reg_param = params[2]
    mae_param = params[3]
    gpu = '0,1'
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    model = D2R_Net(enc_nf, dec_nf)
    ST = SpatialTransformer(vol_size) 
    model.cuda()
    ST.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(torch.load(model_path))
    
    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)
    mae_loss_fn = losses.mae_loss
    ncc_loss_fn = losses.ncc_loss
    mse_loss_fn = losses.mse_loss_ReallynoRTS
    reg_loss_fn = losses.gradient_loss
    
    # data generator
    example_gen = functions.datagenerator_Downsample_inverse_parallel(data_path, data_list, split_ratio, vol_size)
    num_train = example_gen.num_train
    num_valid = example_gen.num_valid

    train_iter = math.floor(example_gen.num_train/batch_size)

    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.8, num_reduced_factor)
        opt = Adam(model.parameters(), lr=lr_new)
        DVF_loss_epoch = np.array([])
        sim_loss_epoch = np.array([])
        reg_loss_epoch = np.array([])
        mae_loss_epoch = np.array([])
        print('\nEpoch %d/%d' % (i+1+n_epoch_trained,n_epoch+n_epoch_trained))
        start = time.time()
        for num_image in range(train_iter):
            input_mov, input_fix, input_fix_down, dvf_resize = example_gen.train_generator(num_image)
            fix_SR, flow = model(input_mov, input_fix_down)
            warp = ST(input_mov, flow)
            
            mae_loss = mae_param * mae_loss_fn(input_fix, fix_SR)
            DVF_loss = DVF_param * mse_loss_fn(flow, dvf_resize)
            sim_loss = sim_param * ncc_loss_fn(warp, input_fix)
            reg_loss = reg_param * reg_loss_fn(flow)                
            loss = DVF_loss + sim_loss + reg_loss + mae_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            # accu loss
            DVF_loss_epoch = np.append(DVF_loss_epoch, DVF_loss.item())
            sim_loss_epoch = np.append(sim_loss_epoch, sim_loss.item())
            reg_loss_epoch = np.append(reg_loss_epoch, reg_loss.item())
            mae_loss_epoch = np.append(mae_loss_epoch, mae_loss.item())
            end = time.time()
            print("\r n_image: %d/%d, time = %fs, lr: %e, DVF: %f, sim: %f, reg:%f, mae:%f." 
                  % (num_image+1, train_iter, end-start, lr_new, np.mean(DVF_loss_epoch),
                     np.mean(sim_loss_epoch), np.mean(reg_loss_epoch), np.mean(mae_loss_epoch)),
                     end = '', flush=True)
        DVF_loss_out = np.append(DVF_loss_out, np.mean(DVF_loss_epoch))
        sim_loss_out = np.append(sim_loss_out, np.mean(sim_loss_epoch))
        reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_epoch))
        mae_loss_out = np.append(mae_loss_out, np.mean(mae_loss_epoch))
        DVF_val_loss, mse_val_loss, ncc_val_loss, mae_val_loss = validation(model,example_gen, ST, i+n_epoch_trained, model_dir, batch_size=2)
        DVF_val_out = np.append(DVF_val_out, DVF_val_loss)
        mse_val_out = np.append(mse_val_out, mse_val_loss)
        ncc_val_out = np.append(ncc_val_out, ncc_val_loss)
        mae_val_out = np.append(mae_val_out, mae_val_loss)
        print("\nValidation: DVF_loss: %f, mse_loss: %f, cc_loss: %f, mae_loss: %f." % 
              (np.mean(DVF_val_loss),np.mean(mse_val_loss), np.mean(ncc_val_loss), np.mean(mae_val_loss)))
        if i+n_epoch_trained % n_save_epoch == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % (i+n_epoch_trained))
            torch.save(model.state_dict(), save_file_name)
            loss = [DVF_loss_out, sim_loss_out, reg_loss_out, mae_loss_out, DVF_val_out, mse_val_out, ncc_val_out, mae_val_out]
            save_npy_name = os.path.join(model_dir, '%d.npy' % (i+n_epoch_trained))
            np.save(save_npy_name, loss)
        loss = [DVF_loss_out, sim_loss_out, reg_loss_out, mae_loss_out, DVF_val_out, mse_val_out, ncc_val_out, mae_val_out]
    return model, loss

def fine_tune_UTSW(model_path, loss_path, model_parameters):
    
    vol_size, enc_nf, dec_nf, data_path, lr, lr_schedule, n_epoch, params, n_save_epoch, n_plot_epoch, model_dir, batch_size, base_path, split_ratio = model_parameters
    
    loss = np.load(loss_path,allow_pickle=True)
    
    DVF_loss_out, sim_loss_out, reg_loss_out, mae_loss_out, DVF_val_out, mse_val_out, ncc_val_out, mae_val_out = loss
    
    DVF_param = params[0]
    sim_param = params[1]
    reg_param = params[2]
    mae_param = params[3]
    gpu = '0,1'
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    model = D2R_Net(enc_nf, dec_nf)
    ST = SpatialTransformer(vol_size) 
    model.cuda()
    ST.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(torch.load(model_path))
    
    # freeze reg part
    for param in model.module.unet_model.parameters():
        param.requires_grad = False
    
    for param in model.module.flow.parameters():
        param.requires_grad = False        
    
    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)
    mae_loss_fn = losses.mae_loss
    ncc_loss_fn = losses.ncc_loss
    mse_loss_fn = losses.mse_loss_ReallynoRTS
    reg_loss_fn = losses.gradient_loss
    
    # data generator
    example_gen = functions.datagenerator_UTSW_finetune(data_path, split_ratio, vol_size)
    num_train = example_gen.num_train
    num_valid = example_gen.num_valid

    train_iter = math.floor(example_gen.num_train/batch_size)

    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.8, num_reduced_factor)
        opt = Adam(model.parameters(), lr=lr_new)
        DVF_loss_epoch = np.array([])
        sim_loss_epoch = np.array([])
        reg_loss_epoch = np.array([])
        mae_loss_epoch = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
        start = time.time()
        for num_image in range(train_iter):
            input_vol, input_vol_down = example_gen.train_generator(num_image)
            vol_SR, flow = model(input_vol, input_vol_down) # input_vol does not work actually
            
            mae_loss = mae_param * mae_loss_fn(input_vol, vol_SR)

            loss = mae_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            # accu loss
            mae_loss_epoch = np.append(mae_loss_epoch, mae_loss.item())
            end = time.time()
            print("\r n_image: %d/%d, time = %fs, lr: %e, mae:%f." 
                  % (num_image+1, train_iter, end-start, lr_new, np.mean(mae_loss_epoch)),
                     end = '', flush=True)
        mae_loss_out = np.append(mae_loss_out, np.mean(mae_loss_epoch))
        mae_val_loss = validation_UTSW(model,example_gen, ST, i, model_dir, batch_size=2)
        mae_val_out = np.append(mae_val_out, mae_val_loss)
        print("\nValidation: mae_loss: %f." % (np.mean(mae_val_loss)))
        if i % n_save_epoch == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % (i))
            torch.save(model.state_dict(), save_file_name)
            loss = [mae_loss_out, mae_val_out]
            save_npy_name = os.path.join(model_dir, '%d.npy' % (i))
            np.save(save_npy_name, loss)
        loss = [DVF_loss_out, sim_loss_out, reg_loss_out, mae_loss_out, DVF_val_out, mse_val_out, ncc_val_out, mae_val_out]
    return model, loss

def validation_UTSW(model, example_gen, ST, i, model_dir, batch_size = 1):

    mse_loss_fn = losses.mse_loss_ReallynoRTS
    ncc_loss_fn = losses.ncc_loss
    mae_loss_fn = losses.mae_loss
    reg_loss_fn = losses.gradient_loss
    mae_loss_after = np.array([])
    valid_iter = math.floor(example_gen.num_valid/batch_size)
    
    for sty_index in range(valid_iter):
        input_vol, input_vol_down = example_gen.valid_generator(sty_index, batch_size)
        # difference map and gradient map
        with torch.no_grad():
            vol_SR, flow = model(input_vol, input_vol_down)

        mae_loss = mae_loss_fn(input_vol, vol_SR)
        mae_loss_after = np.append(mae_loss_after, mae_loss.item())
        

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(input_vol_down.cpu().numpy()[0,0,:,:,30], cmap='gray')
    plt.title(f"DS epoch {i}")          
    plt.axis('off')
    
    plt.subplot(1,3,2)
    plt.imshow(vol_SR.detach().cpu().numpy()[0,0,:,:,30], cmap='gray')
    plt.title(f"SR epoch {i}")         
    plt.axis('off')
    
    plt.subplot(1,3,3)
    plt.imshow(input_vol.cpu().numpy()[0,0,:,:,30], cmap='gray')
    plt.title(f"GT epoch {i}")         
    plt.axis('off')
    save_path = model_dir + f"epoch {i}.png"
    if os.path.isfile(save_path):
        save_path = model_dir + f"fine_tune epoch {i}_1.png"
    else:
        pass
    plt.savefig(save_path, bbox_inches='tight')
        # accu loss
    return np.mean(mae_loss_after)

def finetune_reconreg(model_path, loss_path, model_parameters):
    """
    Finetune for UTSW data, both recon and reg part of D2R net. 
    """
    vol_size, enc_nf, dec_nf, data_path, lr, lr_schedule, n_epoch, params, n_save_epoch, n_plot_epoch, model_dir, batch_size, base_path, split_ratio = model_parameters
    
    loss = np.load(loss_path,allow_pickle=True)
    
    DVF_loss_out, sim_loss_out, reg_loss_out, mae_loss_out, DVF_val_out, mse_val_out, ncc_val_out, mae_val_out = loss
    
    DVF_param = params[0]
    sim_param = params[1]
    reg_param = params[2]
    mae_param = params[3]
    gpu = '0,1'
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    model = D2R_Net(enc_nf, dec_nf)
    ST = SpatialTransformer(vol_size) 
    model.cuda()
    ST.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(torch.load(model_path))
    
    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)
    mae_loss_fn = losses.mae_loss
    ncc_loss_fn = losses.ncc_loss
    mse_loss_fn = losses.mse_loss_ReallynoRTS
    reg_loss_fn = losses.gradient_loss
    
    # data generator
    example_gen = functions.datagenerator_UTSW_finetune_ReconReg(data_path, split_ratio, vol_size)
    num_train = example_gen.num_train
    num_valid = example_gen.num_valid

    train_iter = math.floor(example_gen.num_train/batch_size)

    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.8, num_reduced_factor)
        opt = Adam(model.parameters(), lr=lr_new)
        DVF_loss_epoch = np.array([])
        sim_loss_epoch = np.array([])
        reg_loss_epoch = np.array([])
        mae_loss_epoch = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
        start = time.time()
        for num_image in range(train_iter):
            input_mov, input_fix, input_fix_down, dvf_resize = example_gen.train_generator(num_image)
            fix_SR, flow = model(input_mov, input_fix_down)
            warp = ST(input_mov, flow)
            
            mae_loss = mae_param * mae_loss_fn(input_fix, fix_SR)
            DVF_loss = DVF_param * mse_loss_fn(flow, dvf_resize)
            sim_loss = sim_param * ncc_loss_fn(warp, input_fix)
            reg_loss = reg_param * reg_loss_fn(flow)                
            loss = DVF_loss + sim_loss + reg_loss + mae_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            # accu loss
            DVF_loss_epoch = np.append(DVF_loss_epoch, DVF_loss.item())
            sim_loss_epoch = np.append(sim_loss_epoch, sim_loss.item())
            reg_loss_epoch = np.append(reg_loss_epoch, reg_loss.item())
            mae_loss_epoch = np.append(mae_loss_epoch, mae_loss.item())
            end = time.time()
            print("\r n_image: %d/%d, time = %fs, lr: %e, DVF: %f, sim: %f, reg:%f, mae:%f." 
                  % (num_image+1, train_iter, end-start, lr_new, np.mean(DVF_loss_epoch),
                     np.mean(sim_loss_epoch), np.mean(reg_loss_epoch), np.mean(mae_loss_epoch)),
                     end = '', flush=True)
        DVF_loss_out = np.append(DVF_loss_out, np.mean(DVF_loss_epoch))
        sim_loss_out = np.append(sim_loss_out, np.mean(sim_loss_epoch))
        reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_epoch))
        mae_loss_out = np.append(mae_loss_out, np.mean(mae_loss_epoch))
        DVF_val_loss, mse_val_loss, ncc_val_loss, mae_val_loss = validation(model,example_gen, ST, i, model_dir, batch_size=2)
        DVF_val_out = np.append(DVF_val_out, DVF_val_loss)
        mse_val_out = np.append(mse_val_out, mse_val_loss)
        ncc_val_out = np.append(ncc_val_out, ncc_val_loss)
        mae_val_out = np.append(mae_val_out, mae_val_loss)
        print("\nValidation: DVF_loss: %f, mse_loss: %f, cc_loss: %f, mae_loss: %f." % 
              (np.mean(DVF_val_loss),np.mean(mse_val_loss), np.mean(ncc_val_loss), np.mean(mae_val_loss)))
        if i % n_save_epoch == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % (i))
            torch.save(model.state_dict(), save_file_name)
            loss = [DVF_loss_out, sim_loss_out, reg_loss_out, mae_loss_out, DVF_val_out, mse_val_out, ncc_val_out, mae_val_out]
            save_npy_name = os.path.join(model_dir, '%d.npy' % (i))
            np.save(save_npy_name, loss)
        loss = [DVF_loss_out, sim_loss_out, reg_loss_out, mae_loss_out, DVF_val_out, mse_val_out, ncc_val_out, mae_val_out]

    return model, loss