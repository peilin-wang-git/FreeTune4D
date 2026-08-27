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
from model import SpatialTransformer
from model_C2F import Unet
import torch.nn.functional as F
import functions
import losses
from math import floor
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import random

def train(model_parameters):
    
    vol_size, enc_nf, dec_nf, data_path, lr, lr_schedule, n_epoch, params, n_save_epoch, model_dir, batch_size, base_path, split_ratio = model_parameters
    
    DVF_param = params[0]
    sim_param = params[1]
    reg_param = params[2]
    gpu = '0'
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    
    model = Unet(enc_nf, dec_nf, 1)
    ST = SpatialTransformer(vol_size) 
    model.to(device)
    ST.to(device)
    
    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)
    ncc_loss_fn = losses.ncc_loss
    mse_loss_fn = losses.mse_loss_ReallynoRTS
    reg_loss_fn = losses.gradient_loss
    
    # data generator
    example_gen = functions.gen_CrossValidation
    data_list = os.listdir(data_path)
    random.shuffle(data_list)
    num_total = len(data_list)
    list_train = data_list[:round(num_total*split_ratio)]
    list_valid = data_list[round(num_total*split_ratio):]

    DVF_loss_out = np.array([])
    sim_loss_out = np.array([])
    reg_loss_out = np.array([])
    DVF_val_out = np.array([])
    mse_val_out = np.array([])
    ncc_val_out = np.array([])
    
    DVF_val_loss, mse_val_loss, ncc_val_loss = validation_CrossValidation(model, data_path, list_valid, ST)
    print("\nValidation: DVF_loss: %f, mse_loss: %f, cc_loss: %f." % (np.mean(DVF_val_loss),np.mean(mse_val_loss), np.mean(ncc_val_loss)))
    DVF_val_out = np.append(DVF_val_out, DVF_val_loss)
    mse_val_out = np.append(mse_val_out, mse_val_loss)
    ncc_val_out = np.append(ncc_val_out, ncc_val_loss)
    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.8,num_reduced_factor)
        opt = Adam(model.parameters(), lr=lr_new)
        DVF_loss_epoch = np.array([])
        sim_loss_epoch = np.array([])
        reg_loss_epoch = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
        start = time.time()
        for num_image in range(len(list_train)):
            input_mov, input_fix, input_fix_down, dvf_resize = example_gen(data_path, list_train, num_image)
            input_diff = input_mov - input_fix_down
            flow = model(input_mov, input_fix_down, input_diff)
            warp = ST(input_mov, flow)
            
            DVF_loss = DVF_param * mse_loss_fn(flow, dvf_resize)
            sim_loss = sim_param * ncc_loss_fn(warp, input_fix)
            reg_loss = reg_param * reg_loss_fn(flow)                
            loss = DVF_loss + sim_loss + reg_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            # accu loss
            DVF_loss_epoch = np.append(DVF_loss_epoch, DVF_loss.item())
            sim_loss_epoch = np.append(sim_loss_epoch, sim_loss.item())
            reg_loss_epoch = np.append(reg_loss_epoch, reg_loss.item())
            end = time.time()
            print("\r n_image: %d/%d, time = %fs, lr: %e, DVF: %f, sim: %f, reg:%f" 
                  % (num_image+1, len(list_train), end-start, lr_new, np.mean(DVF_loss_epoch),
                     np.mean(sim_loss_epoch), np.mean(reg_loss_epoch)), end = '', flush=True)
        DVF_loss_out = np.append(DVF_loss_out, np.mean(DVF_loss_epoch))
        sim_loss_out = np.append(sim_loss_out, np.mean(sim_loss_epoch))
        reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_epoch))
        DVF_val_loss, mse_val_loss, ncc_val_loss = validation_CrossValidation(model, data_path, list_valid, ST)
        DVF_val_out = np.append(DVF_val_out, DVF_val_loss)
        mse_val_out = np.append(mse_val_out, mse_val_loss)
        ncc_val_out = np.append(ncc_val_out, ncc_val_loss)
        #DVF_val_loss, sim_val_loss = validation(model,cwd,validation_folder,params,patch_shape,patch_size,stride,pad)
        print("\nValidation: DVF_loss: %f, mse_loss: %f, cc_loss: %f." % (np.mean(DVF_val_loss),np.mean(mse_val_loss), np.mean(ncc_val_loss)))
        print("\nLoss of this epoch: DVF_loss:%f, sim_loss:%f." %(np.mean(DVF_loss_epoch), np.mean(sim_loss_epoch)))
        if i % n_save_epoch == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % i)
            torch.save(model.state_dict(), save_file_name)
        loss = [DVF_loss_out, sim_loss_out, reg_loss_out, DVF_val_out, mse_val_out, ncc_val_out]
    return model, loss

def train_continue(model, loss, epoch_last, model_parameters):
    
    vol_size, enc_nf, dec_nf, data_list, lr, lr_schedule, n_epoch, params, n_save_epoch, model_dir, batch_size, base_path, split_ratio = model_parameters
    
    DVF_param = params[0]
    sim_param = params[1]
    reg_param = params[2]
    gpu = '0'
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    
    ST = SpatialTransformer(vol_size)
    ST.to(device)
    
    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)
    ncc_loss_fn = losses.ncc_loss
    mse_loss_fn = losses.mse_loss_ReallynoRTS
    
    # data generator
    example_gen = functions.datagenerator_SHZ(base_path, data_list, split_ratio, vol_size, device)
    num_train = example_gen.num_train
    num_valid = example_gen.num_valid
    
    DVF_loss_out, sim_loss_out, DVF_val_out, mse_val_out, ncc_val_out = loss
    
    DVF_val_loss, mse_val_loss, ncc_val_loss = validation_CrossValidation(model,example_gen, ST)
    print("\nValidation: DVF_loss: %f, mse_loss: %f, cc_loss: %f." % (np.mean(DVF_val_loss),np.mean(mse_val_loss), np.mean(ncc_val_loss)))
    DVF_val_out = np.append(DVF_val_out, DVF_val_loss)
    mse_val_out = np.append(mse_val_out, mse_val_loss)
    ncc_val_out = np.append(ncc_val_out, ncc_val_loss)
    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt = Adam(model.parameters(), lr=lr_new)
        DVF_loss_epoch = np.array([])
        sim_loss_epoch = np.array([])
        print('\nEpoch %d/%d' % (i+1+epoch_last,n_epoch+epoch_last))
        start = time.time()
        for num_image in range(num_train):
            input_mov, input_fix, dvf_resize = example_gen.train_generator(num_image)
            input_diff = input_mov - input_fix
            flow = model(input_mov, input_fix, input_diff)
            warp = ST(input_mov, flow)
            
            DVF_loss = DVF_param * mse_loss_fn(flow, dvf_resize)
            sim_loss = sim_param * ncc_loss_fn(warp, input_fix)                
            loss = DVF_loss + sim_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            # accu loss
            DVF_loss_epoch = np.append(DVF_loss_epoch, DVF_loss.item())
            sim_loss_epoch = np.append(sim_loss_epoch, sim_loss.item())
            end = time.time()
            print("\r n_image: %d/%d, time = %fs, lr: %e, DVF_loss: %f, sim_loss: %f" 
                  % (num_image+1, num_train, end-start, lr_new, np.mean(DVF_loss_epoch),
                     np.mean(sim_loss_epoch)), end = '', flush=True)
        DVF_loss_out = np.append(DVF_loss_out, np.mean(DVF_loss_epoch))
        sim_loss_out = np.append(sim_loss_out, np.mean(sim_loss_epoch))
        DVF_val_loss, mse_val_loss, ncc_val_loss = validation_CrossValidation(model,example_gen, ST)
        DVF_val_out = np.append(DVF_val_out, DVF_val_loss)
        mse_val_out = np.append(mse_val_out, mse_val_loss)
        ncc_val_out = np.append(ncc_val_out, ncc_val_loss)
        #DVF_val_loss, sim_val_loss = validation(model,cwd,validation_folder,params,patch_shape,patch_size,stride,pad)
        print("\nValidation: DVF_loss: %f, mse_loss: %f, cc_loss: %f." % (np.mean(DVF_val_loss),np.mean(mse_val_loss), np.mean(ncc_val_loss)))
        print("\nLoss of this epoch: DVF_loss:%f, sim_loss:%f." %(np.mean(DVF_loss_epoch), np.mean(sim_loss_epoch)))
        if (i+epoch_last) % n_save_epoch == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % (i+epoch_last))
            torch.save(model.state_dict(), save_file_name)
        loss = [DVF_loss_out, sim_loss_out, DVF_val_out, mse_val_out, ncc_val_out]
    return model, loss

def validation_CrossValidation(model, data_path, list_valid, ST):

    mse_loss_fn = losses.mse_loss_ReallynoRTS
    ncc_loss_fn = losses.ncc_loss
    DVF_loss_val = np.array([])
    mse_loss_after = np.array([])
    ncc_loss_after = np.array([])
    
    for num_image in range(len(list_valid)):
        input_mov, input_fix, input_fix_down, dvf_resize = functions.gen_CrossValidation(data_path, list_valid, num_image)
        input_diff = input_mov - input_fix_down
        # difference map and gradient map
        with torch.no_grad():
            flow = model(input_mov, input_fix_down, input_diff)
            warp = ST(input_mov, flow)
        DVF_loss = mse_loss_fn(flow, dvf_resize)
        mse_loss = mse_loss_fn(warp, input_fix)
        ncc_loss = ncc_loss_fn(warp, input_fix)
        DVF_loss_val = np.append(DVF_loss_val, DVF_loss.item())
        mse_loss_after = np.append(mse_loss_after, mse_loss.item())
        ncc_loss_after = np.append(ncc_loss_after, ncc_loss.item())           
        # accu loss
    return np.mean(DVF_loss_val),np.mean(mse_loss_after), np.mean(ncc_loss_after)

def dvf_interp(dvf, vol_size):
    orig_shape = dvf.shape[2:]
    DVF_0 = F.interpolate(torch.unsqueeze(dvf[:,0,:,:,:],1), vol_size, mode = 'trilinear')
    DVF_1 = F.interpolate(torch.unsqueeze(dvf[:,1,:,:,:],1), vol_size, mode = 'trilinear')
    DVF_2 = F.interpolate(torch.unsqueeze(dvf[:,2,:,:,:],1), vol_size, mode = 'trilinear')
    DVF_0 = DVF_0 * (vol_size[0]/orig_shape[0])
    DVF_1 = DVF_1 * (vol_size[1]/orig_shape[1])
    DVF_2 = DVF_2 * (vol_size[2]/orig_shape[2])
    DVF = torch.cat([DVF_0,DVF_1,DVF_2],1)
    return DVF

def train_RAM(model_parameters):
    
    vol_size, enc_nf, dec_nf, data_path, data_list, lr, lr_schedule, n_epoch, params, n_save_epoch, model_dir, batch_size, base_path, split_ratio = model_parameters
    
    DVF_param = params[0]
    sim_param = params[1]
    reg_param = params[2]
    gpu = '0'
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    
    model = Unet(enc_nf, dec_nf, 1)
    ST = SpatialTransformer(vol_size) 
    model.to(device)
    ST.to(device)
    
    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)
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
    DVF_val_out = np.array([])
    mse_val_out = np.array([])
    ncc_val_out = np.array([])
    
    DVF_val_loss, mse_val_loss, ncc_val_loss = validation_CrossValidation_RAM(model, example_gen, ST)
    print("\nValidation: DVF_loss: %f, mse_loss: %f, cc_loss: %f." % (np.mean(DVF_val_loss),np.mean(mse_val_loss), np.mean(ncc_val_loss)))
    DVF_val_out = np.append(DVF_val_out, DVF_val_loss)
    mse_val_out = np.append(mse_val_out, mse_val_loss)
    ncc_val_out = np.append(ncc_val_out, ncc_val_loss)
    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.8,num_reduced_factor)
        opt = Adam(model.parameters(), lr=lr_new)
        DVF_loss_epoch = np.array([])
        sim_loss_epoch = np.array([])
        reg_loss_epoch = np.array([])
        print('\nEpoch %d/%d' % (i+1,n_epoch))
        start = time.time()
        for num_image in range(num_train):
            input_mov, input_fix, input_fix_down, dvf_resize = example_gen.train_generator(num_image)
            input_diff = input_mov - input_fix
            flow = model(input_mov, input_fix, input_diff)
            warp = ST(input_mov, flow)
            
            DVF_loss = DVF_param * mse_loss_fn(flow, dvf_resize)
            sim_loss = sim_param * ncc_loss_fn(warp, input_fix)
            reg_loss = reg_param * reg_loss_fn(flow)                
            loss = DVF_loss + sim_loss + reg_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            # accu loss
            DVF_loss_epoch = np.append(DVF_loss_epoch, DVF_loss.item())
            sim_loss_epoch = np.append(sim_loss_epoch, sim_loss.item())
            reg_loss_epoch = np.append(reg_loss_epoch, reg_loss.item())
            end = time.time()
            print("\r n_image: %d/%d, time = %fs, lr: %e, DVF: %f, sim: %f, reg:%f" 
                  % (num_image+1, num_train, end-start, lr_new, np.mean(DVF_loss_epoch),
                     np.mean(sim_loss_epoch), np.mean(reg_loss_epoch)), end = '', flush=True)
        DVF_loss_out = np.append(DVF_loss_out, np.mean(DVF_loss_epoch))
        sim_loss_out = np.append(sim_loss_out, np.mean(sim_loss_epoch))
        reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_epoch))
        DVF_val_loss, mse_val_loss, ncc_val_loss = validation_CrossValidation_RAM(model,example_gen, ST)
        DVF_val_out = np.append(DVF_val_out, DVF_val_loss)
        mse_val_out = np.append(mse_val_out, mse_val_loss)
        ncc_val_out = np.append(ncc_val_out, ncc_val_loss)
        #DVF_val_loss, sim_val_loss = validation(model,cwd,validation_folder,params,patch_shape,patch_size,stride,pad)
        print("\nValidation: DVF_loss: %f, mse_loss: %f, cc_loss: %f." % (np.mean(DVF_val_loss),np.mean(mse_val_loss), np.mean(ncc_val_loss)))
        print("\nLoss of this epoch: DVF_loss:%f, sim_loss:%f." %(np.mean(DVF_loss_epoch), np.mean(sim_loss_epoch)))
        if i % n_save_epoch == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % i)
            torch.save(model.state_dict(), save_file_name)
        loss = [DVF_loss_out, sim_loss_out, reg_loss_out, DVF_val_out, mse_val_out, ncc_val_out]
    return model, loss

def validation_CrossValidation_RAM(model, example_gen, ST):

    mse_loss_fn = losses.mse_loss_ReallynoRTS
    ncc_loss_fn = losses.ncc_loss
    DVF_loss_val = np.array([])
    mse_loss_after = np.array([])
    ncc_loss_after = np.array([])
    
    for sty_index in range(example_gen.num_valid):
        input_mov, input_fix, input_fix_down, dvf_resize = example_gen.valid_generator(sty_index)
        input_diff = input_mov - input_fix
        # difference map and gradient map
        with torch.no_grad():
            flow = model(input_mov, input_fix, input_diff)
            warp = ST(input_mov, flow)
        DVF_loss = mse_loss_fn(flow, dvf_resize)
        mse_loss = mse_loss_fn(warp, input_fix)
        ncc_loss = ncc_loss_fn(warp, input_fix)
        DVF_loss_val = np.append(DVF_loss_val, DVF_loss.item())
        mse_loss_after = np.append(mse_loss_after, mse_loss.item())
        ncc_loss_after = np.append(ncc_loss_after, ncc_loss.item())           
        # accu loss
    return np.mean(DVF_loss_val),np.mean(mse_loss_after), np.mean(ncc_loss_after)
