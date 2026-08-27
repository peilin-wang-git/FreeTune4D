# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:08:47 2020

@author: user
"""
import os

# external imports
import numpy as np
import itertools
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.autograd import Variable
# internal imports
from CycleGAN import Unet, Dis_Unet, SpatialTransformer
from func import get_empty_array, array_append, array_append_mean, get_scheduler, update_learning_rate
import torch.nn.functional as F
import datagenerators
import losses
from math import floor
import time
import scipy.io as sio
from utils import ReplayBuffer

def train_cycle(gpu,
          cwd,
          volsize,
          nf_enc,
          nf_dec,
          data_folder,
          validation_folder,
          lr,
          n_epoch,
          params, 
          n_save_epoch,
          model_dir,
          lr_schedule):
    sim_param = params[0]
    GAN_param = params[1]
    id_param = params[2]
    cycle_param = params[3]
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    
    G_AB = Unet(nf_enc, nf_dec, volsize).to(device)
    G_BA = Unet(nf_enc, nf_dec, volsize).to(device)
    D_A = Dis_Unet(volsize).to(device)
    D_B = Dis_Unet(volsize).to(device)
    ST = SpatialTransformer(volsize).to(device)
    
    criterion_GAN = torch.nn.BCELoss().to(device)
    criterion_cycle = losses.ncc_loss
    criterion_identity = losses.ncc_loss
    criterion_sim = losses.ncc_loss
    
    torch.cuda.empty_cache()
    Tensor = torch.cuda.FloatTensor 
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    # data generator
    studies = os.listdir(data_folder)
    train_example_gen = datagenerators.gen_CT_dvf_v7(cwd, studies, data_folder)
    loss_all = get_empty_array('loss', 'G_identity', 'G_cycle','G_sim', 'D_A', 'D_B')
    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.5,num_reduced_factor)
        lr_G, lr_D_A, lr_D_B = lr_new[0], lr_new[1], lr_new[2]
        optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr_G, betas=(0.5, 0.999))
        optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr_D_A, betas=(0.5, 0.999))
        optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr_D_B, betas=(0.5, 0.999))
        print('\nEpoch %d/%d' % (i+1,n_epoch))
        losses_temp = get_empty_array('G_GAN', 'G_identity', 'G_cycle', 'G_sim', 'D_A', 'D_B')
        for num_image in range(100):
            # load data
            volmov, volfix, dvf = next(train_example_gen)
            volmov = torch.from_numpy(volmov).to(device).float()
            volfix = torch.from_numpy(volfix).to(device).float()
            dvf = torch.from_numpy(dvf).to(device).float()
            orig_shape = volmov.shape[2:]
            input_A = F.interpolate(volmov, size = volsize, mode='trilinear')
            input_B = F.interpolate(volfix, size = volsize, mode='trilinear')
            dvf_resize = dvf_interp(dvf, volsize)
            valid = Variable(Tensor(np.ones(1)), requires_grad = False)
            fake = Variable(Tensor(np.zeros(1)), requires_grad = False)
            
            ###---Train Generator---###
            G_AB.train()
            G_BA.train()
            # indentity loss
            loss_id_A = criterion_identity(G_BA(input_A, input_A)[1], input_A)
            loss_id_B = criterion_identity(G_AB(input_B, input_B)[1], input_B)
            loss_identity = id_param*0.5*(loss_id_A + loss_id_B)
            # GAN loss & sim loss
            fake_B = G_AB(input_A, input_B)[1]
            loss_GAN_AB = criterion_GAN(D_B(fake_B),valid)
            loss_sim_AB = criterion_sim(fake_B, input_B)
            fake_A = G_BA(input_B, input_A)[1]
            loss_GAN_BA = criterion_GAN(D_A(fake_B),valid)
            loss_sim_BA = criterion_sim(fake_A, input_A)
            loss_GAN = 0.5*GAN_param*(loss_GAN_AB+loss_GAN_BA)
            loss_sim = 0.5*sim_param * (loss_sim_BA + loss_sim_AB)
            # Cycle loss
            recov_A = G_BA(fake_B, input_A)[1]
            recov_B = G_AB(fake_A, input_B)[1]
            loss_cycle_A = criterion_cycle(recov_A, input_A)
            loss_cycle_B = criterion_cycle(recov_B, input_B)
            loss_cycle = cycle_param*0.5*(loss_cycle_A+loss_cycle_B)
            # total loss
            loss_G = loss_GAN + loss_identity + loss_cycle + loss_sim
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            
            ###---Train Discriminator A---###
            optimizer_D_A.zero_grad()
            loss_real = 0.5*criterion_GAN(D_A(input_A), valid)
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = 0.5*criterion_GAN(D_A(fake_A.detach()), fake)
            loss_D_A = loss_real+loss_fake
            loss_D_A.backward()
            optimizer_D_A.step()
            
            ###---Train Discriminator B---###      
            optimizer_D_B.zero_grad()
            loss_real = 0.5*criterion_GAN(D_B(input_B), valid)
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = 0.5*criterion_GAN(D_B(fake_B.detach()), fake)          
            loss_D_B = loss_real+loss_fake
            loss_D_B.backward()
            optimizer_D_B.step()    
            
            loss_D = 0.5*(loss_D_A+loss_D_B)
                    
            # accu loss
            losses_temp = array_append(losses_temp, G_GAN=loss_GAN,G_cycle=loss_cycle,G_identity=loss_identity,\
                                   D_A=loss_D_A,D_B=loss_D_B)
            print('\r G:{:.5f},D:{:.5f},G_adv:{:.5f},G_cycle:{:.5f},G_idt:{:.5f},G_sim:{:.5f},epoch:{:d})'\
              .format(loss_G.data.item(), loss_D.data.item(), loss_GAN.data.item(), loss_cycle.data.item(), loss_sim.data.item(), loss_identity.data.item(), i), end = '', flush=True)               
        loss_all = array_append_mean(loss_all, losses_temp)
        if i % n_save_epoch == 0:
            torch.save(G_AB.state_dict(), os.path.join(model_dir, 'G_AB_{:d}.pth'.format(i)))
            torch.save(G_BA.state_dict(), os.path.join(model_dir, 'G_BA_{:d}.pth'.format(i)))
    return G_AB, G_BA, D_A, D_B, loss_all

def train_single(gpu,
          cwd,
          volsize,
          nf_enc,
          nf_dec,
          data_folder,
          validation_folder,
          lr,
          n_epoch,
          params, 
          n_save_epoch,
          model_dir,
          lr_schedule):
    sim_param = params[0]
    reg_param = params[1]
    GAN_param = params[0]
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    
    G = Unet(nf_enc, nf_dec, volsize).to(device)
    D = Dis_Unet(volsize).to(device)
    ST = SpatialTransformer(volsize).to(device)
    
    criterion_GAN = torch.nn.BCELoss().to(device)
    criterion_sim = losses.ncc_loss
    criterion_reg = losses.gradient_loss
    
    torch.cuda.empty_cache()
    Tensor = torch.cuda.FloatTensor 
    fake_buffer = ReplayBuffer()
    # data generator
    studies = os.listdir(data_folder)
    train_example_gen = datagenerators.gen_CT_dvf_v7(cwd, studies, data_folder)
    loss_all = get_empty_array('G_GAN', 'G_sim', 'G_reg','G_iden', 'D')
    # Training loop.
    for i in range(n_epoch):
        num_reduced_factor = floor(i/lr_schedule)
        lr_new = lr * pow(0.5,num_reduced_factor)
        lr_G, lr_D = lr_new[0], lr_new[1]
        optimizer_G = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))
        print('\nEpoch %d/%d' % (i+1,n_epoch))
        losses_temp = get_empty_array('G_GAN', 'G_sim', 'G_reg','G_iden', 'D')
        for num_image in range(100):
            # load data
            volmov, volfix, dvf = next(train_example_gen)
            volmov = torch.from_numpy(volmov).to(device).float()
            volfix = torch.from_numpy(volfix).to(device).float()
            dvf = torch.from_numpy(dvf).to(device).float()
            orig_shape = volmov.shape[2:]
            input_A = F.interpolate(volmov, size = volsize, mode='trilinear')
            input_B = F.interpolate(volfix, size = volsize, mode='trilinear')
            dvf_resize = dvf_interp(dvf, volsize)
            valid = Variable(Tensor(np.ones(1)), requires_grad = False)
            fake = Variable(Tensor(np.zeros(1)), requires_grad = False)
            
            ###---Train Generator---###
            G.train()
            # indentity loss
            loss_id_A = criterion_identity(G_BA(input_A, input_A)[1], input_A)
            loss_id_B = criterion_identity(G_AB(input_B, input_B)[1], input_B)
            loss_identity = id_param*0.5*(loss_id_A + loss_id_B)
            # GAN loss & sim loss
            fake_B = G_AB(input_A, input_B)[1]
            loss_GAN_AB = criterion_GAN(D_B(fake_B),valid)
            loss_sim_AB = criterion_sim(fake_B, input_B)
            fake_A = G_BA(input_B, input_A)[1]
            loss_GAN_BA = criterion_GAN(D_A(fake_B),valid)
            loss_sim_BA = criterion_sim(fake_A, input_A)
            loss_GAN = 0.5*GAN_param*(loss_GAN_AB+loss_GAN_BA)
            loss_sim = 0.5*sim_param * (loss_sim_BA + loss_sim_AB)
            # Cycle loss
            recov_A = G_BA(fake_B, input_A)[1]
            recov_B = G_AB(fake_A, input_B)[1]
            loss_cycle_A = criterion_cycle(recov_A, input_A)
            loss_cycle_B = criterion_cycle(recov_B, input_B)
            loss_cycle = cycle_param*0.5*(loss_cycle_A+loss_cycle_B)
            # total loss
            loss_G = loss_GAN + loss_identity + loss_cycle + loss_sim
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            
            ###---Train Discriminator A---###
            optimizer_D_A.zero_grad()
            loss_real = 0.5*criterion_GAN(D_A(input_A), valid)
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = 0.5*criterion_GAN(D_A(fake_A.detach()), fake)
            loss_D_A = loss_real+loss_fake
            loss_D_A.backward()
            optimizer_D_A.step()
            
            ###---Train Discriminator B---###      
            optimizer_D_B.zero_grad()
            loss_real = 0.5*criterion_GAN(D_B(input_B), valid)
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = 0.5*criterion_GAN(D_B(fake_B.detach()), fake)          
            loss_D_B = loss_real+loss_fake
            loss_D_B.backward()
            optimizer_D_B.step()    
            
            loss_D = 0.5*(loss_D_A+loss_D_B)
                    
            # accu loss
            losses_temp = array_append(losses_temp, G_GAN=loss_GAN,G_cycle=loss_cycle,G_identity=loss_identity,\
                                   D_A=loss_D_A,D_B=loss_D_B)
            print('\r G:{:.5f},D:{:.5f},G_adv:{:.5f},G_cycle:{:.5f},G_idt:{:.5f},G_sim:{:.5f},epoch:{:d})'\
              .format(loss_G.data.item(), loss_D.data.item(), loss_GAN.data.item(), loss_cycle.data.item(), loss_sim.data.item(), loss_identity.data.item(), i), end = '', flush=True)               
        loss_all = array_append_mean(loss_all, losses_temp)
        if i % n_save_epoch == 0:
            torch.save(G_AB.state_dict(), os.path.join(model_dir, 'G_AB_{:d}.pth'.format(i)))
            torch.save(G_BA.state_dict(), os.path.join(model_dir, 'G_BA_{:d}.pth'.format(i)))
    return G_AB, G_BA, D_A, D_B, loss_all

def validation_resize(model,cwd,validation_folder,params,vol_size):
    device = "cuda"
    ST = SpatialTransformer(vol_size) 
    ST.to(device)
    DVF_param = params[0]
    sim_param = params[1]
    mse_loss_fn = losses.mse_loss_ReallynoRTS
    cc_loss_fn = losses.cc_loss_ReallynoRTS
    studies = os.listdir(validation_folder)
    DVF_loss_val = np.array([])
    mse_loss_before = np.array([])
    cc_loss_before = np.array([])
    mse_loss_after = np.array([])
    cc_loss_after = np.array([])
    for sty_index in range(len(studies)):
        valid_example_gen = datagenerators.gen_CT_dvf_v7_valid(cwd, studies, sty_index, validation_folder)
        volmov, volfix, dvf = next(valid_example_gen)
        volmov = torch.from_numpy(volmov).to(device).float()
        volfix = torch.from_numpy(volfix).to(device).float()
        dvf = torch.from_numpy(dvf).to(device).float()
        input_mov = F.interpolate(volmov, size = vol_size, mode='trilinear')
        input_fix = F.interpolate(volfix, size = vol_size, mode='trilinear')
        dvf_resize = dvf_interp(dvf, vol_size)
        input_diff = input_mov - input_fix
        # difference map and gradient map
        with torch.no_grad():
            flow_up = model(input_mov, input_fix, input_diff)
            warp = ST(input_mov, flow_up)
        
        DVF_loss = mse_loss_fn(flow_up, dvf_resize) # + 10 * DVF_param * mse_loss_fn(flow_up[:,2,:,:,:], dvf_resize[:,2,:,:,:])
        mse_loss_b = mse_loss_fn(input_mov, input_fix)
        mse_loss_a = mse_loss_fn(warp, input_fix)
        cc_loss_b = cc_loss_fn(input_mov, input_fix)
        cc_loss_a = cc_loss_fn(warp, input_fix)
        DVF_loss_val = np.append(DVF_loss_val, DVF_loss.item())
        mse_loss_before = np.append(mse_loss_before, mse_loss_b.item())
        mse_loss_after = np.append(mse_loss_after, mse_loss_a.item())
        cc_loss_before = np.append(cc_loss_before, cc_loss_b.item())
        cc_loss_after = np.append(cc_loss_after, cc_loss_a.item())       
        
    print("\nValidation is: DVF:%f, sim:%f." %(np.mean(DVF_loss_val), np.mean(sim_loss_val)))
    return 0

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