"""
*Preliminary* pytorch implementation.

VoxelMorph training.
"""


# python imports
import os

# external imports
import numpy as np
import torch
from torch.optim import Adam

# internal imports
from model import cvpr2018_net, cvpr2018_net_RTS, SpatialTransformer
from model_C2F import voxelmorph_coarse, voxelmorph_fine
from model_C2F import voxelmorph_C, voxelmorph_F, voxelmorph_FF
import datagenerators
import losses
from math import floor
import time

def train_C2F_new(gpu,
          cwd,
          vol_size,
          nf_enc,
          nf_dec,
          studies,
          lr,
          n_epoch,
          data_loss,
          params, 
          batch_size,
          n_save_iter,
          model_dir,
          data_folder,
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
    iden_param = params[4]
    cycle_param = params[5]
    
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    model_x4 = voxelmorph_C(vol_size, nf_enc, nf_dec, True)
    model_x2 = voxelmorph_F(vol_size, nf_enc, nf_dec, False)
    model_x1 = voxelmorph_FF(vol_size, nf_enc, nf_dec)
    ST = SpatialTransformer(vol_size)
    model_x4.to(device)
    model_x2.to(device)
    model_x1.to(device)
    ST.to(device)
    
    
    # Set optimizer and losses
    opt_x4 = Adam(model_x4.parameters(), lr=lr)
    opt_x2 = Adam(model_x2.parameters(), lr=lr)
    opt_x1 = Adam(model_x1.parameters(), lr=lr)

    sim_loss_fn = losses.cc_loss_noRTS if data_loss == "cc" else losses.mse_loss
#    diff_loss_fn = losses.diff_loss
    reg_loss_fn = losses.regulation_loss
    grad_loss_fn = losses.gradient_loss

    # data generator
#    train_example_gen = datagenerators.example_gen_HX(cwd, studies)
    train_example_gen = datagenerators.gen_CT_RTS_v3_ext(cwd, studies, data_folder)
    
    loss_accu_x4 = np.array([])
    recon_loss_accu_x4 = np.array([])
    reg_loss_accu_x4 = np.array([])
    grad_loss_accu_x4 = np.array([])
    
    loss_accu_x2 = np.array([])
    recon_loss_accu_x2 = np.array([])
    reg_loss_accu_x2 = np.array([])
    grad_loss_accu_x2 = np.array([])
    
    loss_accu_x1 = np.array([])
    recon_loss_accu_x1 = np.array([])
    reg_loss_accu_x1 = np.array([])
    grad_loss_accu_x1 = np.array([])
    
    loss_out = np.array([])
    recon_loss_out = np.array([])
    reg_loss_out = np.array([])
    grad_loss_out = np.array([])

    # Training loop.
    for i in range(n_epoch):
        num_epoch = i
        num_reduced_factor = floor(num_epoch/lr_schedule)
        lr_new = lr * pow(0.7,num_reduced_factor)
        opt_x4 = Adam(model_x4.parameters(), lr=lr_new)
        opt_x2 = Adam(model_x2.parameters(), lr=lr_new)
        opt_x1 = Adam(model_x1.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_iter == 0:
            save_file_name_x4 = os.path.join(model_dir, '%d_x4.ckpt' % i)
            torch.save(model_x4.state_dict(), save_file_name_x4)
            save_file_name_x2 = os.path.join(model_dir, '%d_x2.ckpt' % i)
            torch.save(model_x2.state_dict(), save_file_name_x2)
            save_file_name_x1 = os.path.join(model_dir, '%d_x1.ckpt' % i)
            torch.save(model_x1.state_dict(), save_file_name_x1)
        if i % 1 == 0:
            loss_accu_x4 = np.array([])
            recon_loss_accu_x4 = np.array([])
            reg_loss_accu_x4 = np.array([])
            grad_loss_accu_x4 = np.array([])
            loss_accu_x2 = np.array([])
            recon_loss_accu_x2 = np.array([])
            reg_loss_accu_x2 = np.array([])
            grad_loss_accu_x2 = np.array([])
            loss_accu_x1 = np.array([])
            recon_loss_accu_x1 = np.array([])
            reg_loss_accu_x1 = np.array([])
            grad_loss_accu_x1 = np.array([])
            print(' ')
            print('Epoch %d/%d' % (num_epoch+1, n_epoch))
        # Generate the moving images and convert them to tensors.
        moving_image, fixed_image, moving_RTS, fixed_RTS = next(train_example_gen)
        patch_gen = datagenerators.gen_patch(moving_image, fixed_image, moving_RTS, fixed_RTS, vol_size, batch_size)
        for num_patch in range(150):
            start = time.time()
            moving_patch, fixed_patch, RS_M_patch, RS_F_patch = next(patch_gen)
            input_A = torch.from_numpy(moving_patch).to(device).float()
            input_A = input_A.permute(0, 4, 3, 2, 1)
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B = input_B.permute(0, 4, 3, 2, 1)
            input_A_RTS = torch.from_numpy(RS_M_patch).to(device).float()
            input_A_RTS = input_A_RTS.permute(1, 0, 4, 3, 2)
            input_B_RTS = torch.from_numpy(RS_F_patch).to(device).float()
            input_B_RTS = input_B_RTS.permute(1, 0, 4, 3, 2)
            body_contour = torch.unsqueeze(input_B_RTS[:,0,:,:,:],0)
            lung_contour = torch.unsqueeze(input_B_RTS[:,1,:,:,:],0)
#            plt.imshow(input_A.cpu()[0,0,:,:,20])
#            plt.imshow(input_B.cpu()[0,0,:,:,20])
            
            # Run the data through the model to produce warp and flow field
            # x4
            flow_x4, flow_x4_up = model_x4(input_A, input_B)
            warp_x4 = ST(input_A ,flow_x4_up)
            if torch.max(lung_contour) > 0:
                recon_loss_x4 = 0.05 * lung_param * sim_loss_fn(input_B, warp_x4, body_contour)
            else:
                recon_loss_x4 = 0.05 * sim_loss_fn(input_B, warp_x4, body_contour)
            reg_loss_x4 = reg_param * reg_loss_fn(flow_x4_up)
            grad_loss_x4 = grad_param * grad_loss_fn(flow_x4_up)
            loss_x4 = recon_loss_x4 + reg_loss_x4 + grad_loss_x4
            
            # x2
            flow_x2, flow_x2_up = model_x2(warp_x4, input_B)
            warp_x2 = ST(warp_x4 ,flow_x2_up)
            if torch.max(lung_contour) > 0:
                recon_loss_x2 = 0.1 * lung_param * sim_loss_fn(input_B, warp_x2, body_contour)
            else:
                recon_loss_x2 = 0.1 * sim_loss_fn(input_B, warp_x2, body_contour)
            reg_loss_x2 = reg_param * reg_loss_fn(flow_x2_up)
            grad_loss_x2 = grad_param * grad_loss_fn(flow_x2_up)
            loss_x2 = recon_loss_x2 + reg_loss_x2 + grad_loss_x2
            
            # x1
            flow_x1, flow_x1_up = model_x1(warp_x2, input_B)
            warp_x1 = ST(warp_x2 ,flow_x1_up)
            if torch.max(lung_contour) > 0:
                recon_loss_x1 = 0.9 * lung_param * sim_loss_fn(input_B, warp_x1, body_contour)
            else:
                recon_loss_x1 = 0.9 * sim_loss_fn(input_B, warp_x1, body_contour)
            reg_loss_x1 = reg_param * reg_loss_fn(flow_x1_up)
            grad_loss_x1 = grad_param * grad_loss_fn(flow_x1_up)
            loss_x1 = recon_loss_x1 + reg_loss_x1 + grad_loss_x1            
            
            #sum loss
#            loss = loss_x4+loss_x2+loss_x1
#            recon_loss = recon_loss_x4 + recon_loss_x2 + recon_loss_x1
#            reg_loss = reg_loss_x4 + reg_loss_x2 + reg_loss_x1
#            grad_loss = grad_loss_x4 + grad_loss_x2 + grad_loss_x1
            
            # Print loss
            loss_accu_x4 = np.append(loss_accu_x4,loss_x4.item())
            loss_accu_x2 = np.append(loss_accu_x2,loss_x2.item())
            loss_accu_x1 = np.append(loss_accu_x1,loss_x1.item())
            recon_loss_accu_x4 = np.append(recon_loss_accu_x4, recon_loss_x4.item())
            recon_loss_accu_x2 = np.append(recon_loss_accu_x2, recon_loss_x2.item())
            recon_loss_accu_x1 = np.append(recon_loss_accu_x1, recon_loss_x1.item())
            grad_loss_accu_x4 = np.append(grad_loss_accu_x4, grad_loss_x4.item())
            grad_loss_accu_x2 = np.append(grad_loss_accu_x2, grad_loss_x2.item())
            grad_loss_accu_x1 = np.append(grad_loss_accu_x1, grad_loss_x1.item())
            reg_loss_accu_x4 = np.append(reg_loss_accu_x4, reg_loss_x4.item())
            reg_loss_accu_x2 = np.append(reg_loss_accu_x2, reg_loss_x2.item())
            reg_loss_accu_x1 = np.append(reg_loss_accu_x1, reg_loss_x1.item())
            
            if (i == 0) & (num_patch == 0):
                print('saved\n')
                loss_out = np.append(loss_out, np.mean(loss_accu_x4 + loss_accu_x2 + loss_accu_x1))
                recon_loss_out = np.append(recon_loss_out, np.mean(recon_loss_accu_x4+recon_loss_accu_x2+recon_loss_accu_x1))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu_x4+reg_loss_accu_x2+reg_loss_accu_x1))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu_x4+grad_loss_accu_x2+grad_loss_accu_x1))
            # Backwards and optimize
            opt_x4.zero_grad()
            loss_x4.backward(retain_graph=True)
            opt_x4.step()
            opt_x2.zero_grad()
            loss_x2.backward(retain_graph=True)
            opt_x2.step()
            opt_x1.zero_grad()
            loss_x1.backward(retain_graph=True)
            opt_x1.step()
            end = time.time()
            print("\r n_iter: %d/150, time = %fs, lr: %e\
                  x4:\
                  loss: %f, sim: %f, reg:%f, grad:%f\
                  x2:\
                  loss: %f, sim: %f, reg:%f, grad:%f\
                  x1:\
                  loss: %f, sim: %f, reg:%f, grad:%f." 
              % (num_patch + 1, end-start, lr_new,
                 np.mean(loss_accu_x4), np.mean(recon_loss_accu_x4), np.mean(reg_loss_accu_x4), np.mean(grad_loss_accu_x4),
                 np.mean(loss_accu_x2), np.mean(recon_loss_accu_x2), np.mean(reg_loss_accu_x2), np.mean(grad_loss_accu_x2),
                 np.mean(loss_accu_x1), np.mean(recon_loss_accu_x1), np.mean(reg_loss_accu_x1), np.mean(grad_loss_accu_x1),
                 ), end = '', flush=True)
        loss_out = np.append(loss_out, np.mean(loss_accu_x4 + loss_accu_x2 + loss_accu_x1))
        recon_loss_out = np.append(recon_loss_out, np.mean(recon_loss_accu_x4+recon_loss_accu_x2+recon_loss_accu_x1))
        reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu_x4+reg_loss_accu_x2+reg_loss_accu_x1))
        grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu_x4+grad_loss_accu_x2+grad_loss_accu_x1))
    return model_x4, model_x2, model_x1, loss_out, recon_loss_out, reg_loss_out, grad_loss_out


def train(gpu,
          cwd,
          vol_size,
          nf_enc,
          nf_dec,
          studies,
          lr,
          n_iter,
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
    iden_param = params[4]
    cycle_param = params[5]
    
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    model = cvpr2018_net(vol_size, nf_enc, nf_dec)
    ST = SpatialTransformer(vol_size)        
    model.to(device)
    ST.to(device)
    
    
    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)

    sim_loss_fn = losses.cc_loss_noRTS if data_loss == "cc" else losses.mse_loss_noRTS
#    diff_loss_fn = losses.diff_loss
    reg_loss_fn = losses.regulation_loss
    DICE_loss_fn = losses.DICE_loss
    grad_loss_fn = losses.gradient_loss
    cycle_loss_fn = losses.mae_loss

    # data generator
#    train_example_gen = datagenerators.example_gen_HX(cwd, studies)
    data_folder = '4D-Lung-contour-npy-v2'
    train_example_gen = datagenerators.gen_CT_RTS(cwd, studies, data_folder)
    
    loss_accu = np.array([])
    recon_loss_accu = np.array([])
    reg_loss_accu = np.array([])
    DICE_loss_accu = np.array([])
    grad_loss_accu = np.array([])
    iden_loss_accu = np.array([])
    cycle_loss_accu = np.array([])
    
    loss_out = np.array([])
    recon_loss_out = np.array([])
    reg_loss_out = np.array([])
    DICE_loss_out = np.array([])
    grad_loss_out = np.array([])
    iden_loss_out = np.array([])
    cycle_loss_out = np.array([])

    # Training loop.
    for i in range(n_iter):
        num_epoch = floor(i/100)+1
        num_reduced_factor = floor(num_epoch)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt = Adam(model.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_iter == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % i)
            torch.save(model.state_dict(), save_file_name)
            
        if i % 100 == 0:
            loss_accu = np.array([])
            recon_loss_accu = np.array([])
            reg_loss_accu = np.array([])
            DICE_loss_accu = np.array([])
            grad_loss_accu = np.array([])
            iden_loss_accu = np.array([])
            cycle_loss_accu = np.array([])
            print(' ')
            print('Epoch %d/%d' % (floor(i/100)+1,floor(n_iter/100)))
        # Generate the moving images and convert them to tensors.
        moving_image, fixed_image, moving_RTS, fixed_RTS = next(train_example_gen)
        patch_gen = datagenerators.gen_patch(moving_image, fixed_image, moving_RTS, fixed_RTS, vol_size, batch_size)
        for num_patch in range(50):
            start = time.time()
            moving_patch, fixed_patch, RS_M_patch, RS_F_patch = next(patch_gen)
            input_A = torch.from_numpy(moving_patch).to(device).float()
            input_A = input_A.permute(0, 4, 3, 2, 1)
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B = input_B.permute(0, 4, 3, 2, 1)
            input_A_RTS = torch.from_numpy(RS_M_patch).to(device).float()
            input_A_RTS = input_A_RTS.permute(1, 0, 4, 3, 2)
            input_B_RTS = torch.from_numpy(RS_F_patch).to(device).float()
            input_B_RTS = input_B_RTS.permute(1, 0, 4, 3, 2)
            body_contour = torch.unsqueeze(input_B_RTS[:,0,:,:,:],0)
            lung_contour = torch.unsqueeze(input_B_RTS[:,1,:,:,:],0)
            # organ contour exists?
            if (input_A_RTS.size()[1] > 2) & (input_B_RTS.size()[1] > 2):
                organ_contour_A = torch.unsqueeze(input_A_RTS[:,2,:,:,:],0)
                organ_contour_B = torch.unsqueeze(input_B_RTS[:,2,:,:,:],0)
            else:
                organ_contour_A = torch.zeros_like(lung_contour)
                organ_contour_B = torch.zeros_like(lung_contour)            
#            plt.imshow(input_A.cpu()[0,0,:,:,20])
#            plt.imshow(input_B.cpu()[0,0,:,:,20])
            
            # Run the data through the model to produce warp and flow field
            # Seduo B
            warp_A2sB, flow_A2sB = model(input_A, input_B)
            warp_RTS_A2sB = ST(organ_contour_A ,flow_A2sB)
            if torch.max(lung_contour) > 0:
                recon_loss_A2sB = lung_param * sim_loss_fn(input_B, warp_A2sB, body_contour)
            else:
                recon_loss_A2sB = sim_loss_fn(input_B, warp_A2sB, body_contour)
            reg_loss_A2sB = reg_param * reg_loss_fn(flow_A2sB)
            DICE_loss_A2sB = DICE_param * DICE_loss_fn(organ_contour_B, warp_RTS_A2sB)
            grad_loss_A2sB = grad_param * grad_loss_fn(flow_A2sB)
            loss_A2sB = recon_loss_A2sB + reg_loss_A2sB + grad_loss_A2sB + DICE_loss_A2sB
            
            # Seduo A
            warp_B2sA, flow_B2sA = model(input_B, input_A)
            warp_RTS_B2sA = ST(organ_contour_B, flow_B2sA)
            if torch.max(lung_contour) > 0:
                recon_loss_B2sA = lung_param * sim_loss_fn(input_A, warp_B2sA, body_contour)
            else:
                recon_loss_B2sA = sim_loss_fn(input_A, warp_B2sA, body_contour)
            reg_loss_B2sA = reg_param * reg_loss_fn(flow_B2sA)
            DICE_loss_B2sA = DICE_param * DICE_loss_fn(organ_contour_A, warp_RTS_B2sA)
            grad_loss_B2sA = grad_param * grad_loss_fn(flow_B2sA)
            loss_B2sA = recon_loss_B2sA + reg_loss_B2sA + grad_loss_B2sA + DICE_loss_B2sA
            
            # Seduo B back to Seduo Seduo A
            warp_sB2sA, flow_sB2sA = model(warp_A2sB, warp_B2sA)
            warp_RTS_sB2sA = ST(warp_RTS_A2sB, flow_sB2sA)
            if torch.max(lung_contour) > 0:
                recon_loss_sB2sA = sim_loss_fn(warp_sB2sA, warp_B2sA, body_contour)
            else:
                recon_loss_sB2sA = sim_loss_fn(warp_sB2sA, warp_B2sA, body_contour)
            reg_loss_sB2sA = reg_param * reg_loss_fn(flow_sB2sA)
            DICE_loss_sB2sA = DICE_param * DICE_loss_fn(warp_RTS_A2sB, warp_RTS_sB2sA)
            grad_loss_sB2sA = grad_param * grad_loss_fn(flow_sB2sA)
            loss_sB2sA = recon_loss_sB2sA + reg_loss_sB2sA + grad_loss_sB2sA + DICE_loss_sB2sA
                        
            # Seduo A back to Seduo Seduo B
            warp_sA2sB, flow_sA2sB = model(warp_B2sA, warp_A2sB)
            warp_RTS_sA2sB = ST(warp_RTS_B2sA, flow_sA2sB)
            if torch.max(lung_contour) > 0:
                recon_loss_sA2sB = lung_param * sim_loss_fn(warp_sA2sB, warp_A2sB, body_contour)
            else:
                recon_loss_sA2sB = sim_loss_fn(warp_sA2sB, warp_A2sB, body_contour)
            reg_loss_sA2sB = reg_param * reg_loss_fn(flow_sA2sB)
            DICE_loss_sA2sB = DICE_param * DICE_loss_fn(warp_RTS_A2sB, warp_RTS_sA2sB)
            grad_loss_sA2sB = grad_param * grad_loss_fn(flow_sA2sB)
            loss_sA2sB = recon_loss_sA2sB + reg_loss_sA2sB + grad_loss_sA2sB + DICE_loss_sA2sB
            
            # Cycle loss
            cycle_loss_a = cycle_param * cycle_loss_fn(input_A, warp_sB2sA)
            cycle_loss_b = cycle_param * cycle_loss_fn(input_B, warp_sA2sB)
            cycle_loss = cycle_loss_a + cycle_loss_b
            
            # Identity loss
            warp_A2A, flow_A2A = model(input_A, input_A)
            iden_loss_A = sim_loss_fn(warp_A2A, input_A, body_contour) 
            warp_B2B, flow_B2B = model(input_B, input_B)
            iden_loss_B = sim_loss_fn(warp_B2B, input_B, body_contour)
            iden_loss = iden_param * iden_loss_A + iden_param * iden_loss_B
            
            #sum loss
            loss = loss_A2sB + loss_B2sA + loss_sB2sA + loss_sA2sB + iden_loss + cycle_loss
            recon_loss = recon_loss_A2sB + recon_loss_B2sA + recon_loss_sB2sA + recon_loss_sA2sB
            reg_loss = reg_loss_A2sB + reg_loss_B2sA + reg_loss_sB2sA + reg_loss_sA2sB
            DICE_loss = DICE_loss_A2sB + DICE_loss_B2sA + DICE_loss_sB2sA + DICE_loss_sA2sB
            grad_loss = grad_loss_A2sB + grad_loss_B2sA + grad_loss_sB2sA + grad_loss_sA2sB
            
            # Print loss
            loss_accu = np.append(loss_accu,loss.item())
            recon_loss_accu = np.append(recon_loss_accu, recon_loss.item())
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())
            DICE_loss_accu = np.append(DICE_loss_accu, DICE_loss.item())
            iden_loss_accu = np.append(iden_loss_accu, iden_loss.item())
            cycle_loss_accu = np.append(cycle_loss_accu, cycle_loss.item())
            
            if (i == 0) & (num_patch == 0):
                print('saved\n')
                loss_out = np.append(loss_out, np.mean(loss_accu))
                recon_loss_out = np.append(recon_loss_out, np.mean(recon_loss_accu))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
                DICE_loss_out = np.append(DICE_loss_out, np.mean(DICE_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))
                iden_loss_out = np.append(iden_loss_out, np.mean(iden_loss_accu))
                cycle_loss_out = np.append(cycle_loss_out, np.mean(cycle_loss_accu))
            
            # Backwards and optimize
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
            
            end = time.time()
            print("\r n_iter: %d/100, num_patch: %d/200, time = %fs, lr: %f, Loss: %f, sim_loss: %f, reg_loss: %f, grad_loss: %f, DICE_loss: %f, iden_loss: %f, cycle_loss: %f" 
              % (i-100 * floor(i/100) + 1, num_patch + 1, end-start, lr_new,
                 np.mean(loss_accu), np.mean(recon_loss_accu), np.mean(reg_loss_accu), np.mean(grad_loss_accu),
                 np.mean(DICE_loss_accu), np.mean(iden_loss_accu), np.mean(cycle_loss_accu)), end = '', flush=True)
        loss_out = np.append(loss_out, np.mean(loss_accu))
        recon_loss_out = np.append(recon_loss_out, np.mean(recon_loss_accu))
        reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
        DICE_loss_out = np.append(DICE_loss_out, np.mean(DICE_loss_accu))
        grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))
        iden_loss_out = np.append(iden_loss_out, np.mean(iden_loss_accu))
        cycle_loss_out = np.append(cycle_loss_out, np.mean(cycle_loss_accu))
    return model, loss_out, recon_loss_out, reg_loss_out, DICE_loss_out, grad_loss_out, iden_loss_out, cycle_loss_out

def train_simple(gpu,
          cwd,
          vol_size,
          nf_enc,
          nf_dec,
          studies,
          lr,
          n_iter,
          data_loss,
          params, 
          batch_size,
          n_save_iter,
          model_dir,
          point_per_iter):
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
    iden_param = params[4]
    cycle_param = params[5]
    
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    model = cvpr2018_net(vol_size, nf_enc, nf_dec)
    ST = SpatialTransformer(vol_size)        
    model.to(device)
    ST.to(device)
    
    
    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)

    sim_loss_fn = losses.cc_loss_noRTS if data_loss == "cc" else losses.mse_loss
#    diff_loss_fn = losses.diff_loss
    reg_loss_fn = losses.regulation_loss
    DICE_loss_fn = losses.DICE_loss
    grad_loss_fn = losses.gradient_loss
    cycle_loss_fn = losses.mae_loss

    # data generator
#    train_example_gen = datagenerators.example_gen_HX(cwd, studies)
    data_folder = '4D-Lung-contour-npy-v3'
    train_example_gen = datagenerators.gen_CT_RTS_2_28(cwd, studies, data_folder)
    
    loss_accu = np.array([])
    recon_loss_accu = np.array([])
    reg_loss_accu = np.array([])
    DICE_loss_accu = np.array([])
    grad_loss_accu = np.array([])
    iden_loss_accu = np.array([])
    cycle_loss_accu = np.array([])
    
    loss_out = np.array([])
    recon_loss_out = np.array([])
    reg_loss_out = np.array([])
    DICE_loss_out = np.array([])
    grad_loss_out = np.array([])
    iden_loss_out = np.array([])
    cycle_loss_out = np.array([])

    # Training loop.
    for i in range(n_iter):
        num_epoch = floor(i/100)+1
        num_reduced_factor = floor(num_epoch)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt = Adam(model.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_iter == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % i)
            torch.save(model.state_dict(), save_file_name)
            
        if i % 100 == 0:
            loss_accu = np.array([])
            recon_loss_accu = np.array([])
            reg_loss_accu = np.array([])
            DICE_loss_accu = np.array([])
            grad_loss_accu = np.array([])
            iden_loss_accu = np.array([])
            cycle_loss_accu = np.array([])
            print(' ')
            print('Epoch %d/%d' % (floor(i/100)+1,floor(n_iter/100)))
        # Generate the moving images and convert them to tensors.
        moving_image, fixed_image, moving_RTS, fixed_RTS = next(train_example_gen)
        patch_gen = datagenerators.gen_patch(moving_image, fixed_image, moving_RTS, fixed_RTS, vol_size, batch_size)
        for num_patch in range(point_per_iter):
            start = time.time()
            moving_patch, fixed_patch, RS_M_patch, RS_F_patch = next(patch_gen)
            input_A = torch.from_numpy(moving_patch).to(device).float()
            input_A = input_A.permute(0, 4, 3, 2, 1)
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B = input_B.permute(0, 4, 3, 2, 1)
            input_A_RTS = torch.from_numpy(RS_M_patch).to(device).float()
            input_A_RTS = input_A_RTS.permute(1, 0, 4, 3, 2)
            input_B_RTS = torch.from_numpy(RS_F_patch).to(device).float()
            input_B_RTS = input_B_RTS.permute(1, 0, 4, 3, 2)
            body_contour = torch.unsqueeze(input_B_RTS[:,0,:,:,:],0)
            lung_contour = torch.unsqueeze(input_B_RTS[:,1,:,:,:],0)
            if (input_A_RTS.size()[1] > 2) & (input_B_RTS.size()[1] > 2):
                organ_contour_A = torch.unsqueeze(input_A_RTS[:,2,:,:,:],0)
                organ_contour_B = torch.unsqueeze(input_B_RTS[:,2,:,:,:],0)
            else:
                organ_contour_A = torch.zeros_like(lung_contour)
                organ_contour_B = torch.zeros_like(lung_contour) 
#            plt.imshow(input_A.cpu()[0,0,:,:,20])
#            plt.imshow(input_B.cpu()[0,0,:,:,20])
            
            # Run the data through the model to produce warp and flow field
            # Seduo B
            warp_A2sB, flow_A2sB = model(input_A, input_B)
            warp_RTS_A2sB = ST(organ_contour_A ,flow_A2sB)
            if torch.max(lung_contour) > 0:
                recon_loss_A2sB = lung_param * sim_loss_fn(input_B, warp_A2sB, body_contour)
            else:
                recon_loss_A2sB = sim_loss_fn(input_B, warp_A2sB, body_contour)
            reg_loss_A2sB = reg_param * reg_loss_fn(flow_A2sB)
            DICE_loss_A2sB = DICE_param * DICE_loss_fn(organ_contour_B, warp_RTS_A2sB)
            grad_loss_A2sB = grad_param * grad_loss_fn(flow_A2sB)
            loss_A2sB = recon_loss_A2sB + reg_loss_A2sB + grad_loss_A2sB + DICE_loss_A2sB
            
            # Seduo A
            warp_B2sA, flow_B2sA = model(input_B, input_A)
            warp_RTS_B2sA = ST(organ_contour_B, flow_B2sA)
            if torch.max(lung_contour) > 0:
                recon_loss_B2sA = lung_param * sim_loss_fn(input_A, warp_B2sA, body_contour)
            else:
                recon_loss_B2sA = sim_loss_fn(input_A, warp_B2sA, body_contour)
            reg_loss_B2sA = reg_param * reg_loss_fn(flow_B2sA)
            DICE_loss_B2sA = DICE_param * DICE_loss_fn(organ_contour_A, warp_RTS_B2sA)
            grad_loss_B2sA = grad_param * grad_loss_fn(flow_B2sA)
            loss_B2sA = recon_loss_B2sA + reg_loss_B2sA + grad_loss_B2sA + DICE_loss_B2sA
            
            # Seduo B back to Seduo Seduo A
            warp_sB2sA, flow_sB2sA = model(warp_A2sB, warp_B2sA)
                       
            # Seduo A back to Seduo Seduo B
            warp_sA2sB, flow_sA2sB = model(warp_B2sA, warp_A2sB)
           
            # Cycle loss
            cycle_loss_a = cycle_param * cycle_loss_fn(input_A, warp_sB2sA)
            cycle_loss_b = cycle_param * cycle_loss_fn(input_B, warp_sA2sB)
            cycle_loss = cycle_loss_a + cycle_loss_b
            
            # Identity loss
            warp_A2A, flow_A2A = model(input_A, input_A)
            iden_loss_A = sim_loss_fn(warp_A2A, input_A, body_contour) 
            warp_B2B, flow_B2B = model(input_B, input_B)
            iden_loss_B = sim_loss_fn(warp_B2B, input_B, body_contour)
            iden_loss = iden_param * iden_loss_A + iden_param * iden_loss_B
            
            #sum loss
            loss = loss_A2sB + loss_B2sA + iden_loss + cycle_loss
            recon_loss = recon_loss_A2sB + recon_loss_B2sA 
            reg_loss = reg_loss_A2sB + reg_loss_B2sA
            DICE_loss = DICE_loss_A2sB + DICE_loss_B2sA
            grad_loss = grad_loss_A2sB + grad_loss_B2sA
            
            # Print loss
            loss_accu = np.append(loss_accu,loss.item())
            recon_loss_accu = np.append(recon_loss_accu, recon_loss.item())
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())
            DICE_loss_accu = np.append(DICE_loss_accu, DICE_loss.item())
            iden_loss_accu = np.append(iden_loss_accu, iden_loss.item())
            cycle_loss_accu = np.append(cycle_loss_accu, cycle_loss.item())
            
            if (i == 0) & (num_patch == 0):
                print('saved\n')
                loss_out = np.append(loss_out, np.mean(loss_accu))
                recon_loss_out = np.append(recon_loss_out, np.mean(recon_loss_accu))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
                DICE_loss_out = np.append(DICE_loss_out, np.mean(DICE_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))
                iden_loss_out = np.append(iden_loss_out, np.mean(iden_loss_accu))
                cycle_loss_out = np.append(cycle_loss_out, np.mean(cycle_loss_accu))
            
            # Backwards and optimize
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
            
            end = time.time()
            print("\r n_iter: %d/100, num_patch: %d/500, time = %fs, lr: %f, Loss: %f, sim_loss: %f, reg_loss: %f, grad_loss: %f, DICE_loss: %f, iden_loss: %f, cycle_loss: %f" 
              % (i-100 * floor(i/100) + 1, num_patch + 1, end-start, lr_new,
                 np.mean(loss_accu), np.mean(recon_loss_accu), np.mean(reg_loss_accu), np.mean(grad_loss_accu),
                 np.mean(DICE_loss_accu), np.mean(iden_loss_accu), np.mean(cycle_loss_accu)), end = '', flush=True)
        loss_out = np.append(loss_out, np.mean(loss_accu))
        recon_loss_out = np.append(recon_loss_out, np.mean(recon_loss_accu))
        reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
        DICE_loss_out = np.append(DICE_loss_out, np.mean(DICE_loss_accu))
        grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))
        iden_loss_out = np.append(iden_loss_out, np.mean(iden_loss_accu))
        cycle_loss_out = np.append(cycle_loss_out, np.mean(cycle_loss_accu))
    return model, loss_out, recon_loss_out, reg_loss_out, DICE_loss_out, grad_loss_out, iden_loss_out, cycle_loss_out

def retrain_simple(gpu,
          cwd,
          vol_size,
          nf_enc,
          nf_dec,
          studies,
          lr,
          n_iter,
          data_loss,
          params, 
          batch_size,
          n_save_iter,
          model_dir,
          saved_weights,
          point_per_iter):
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
    iden_param = params[4]
    cycle_param = params[5]
    
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    model = cvpr2018_net(vol_size, nf_enc, nf_dec)
    ST = SpatialTransformer(vol_size)        
    model.to(device)
    ST.to(device)
    model.load_state_dict(torch.load(saved_weights))
    
    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)

    sim_loss_fn = losses.cc_loss_noRTS if data_loss == "cc" else losses.mse_loss
#    diff_loss_fn = losses.diff_loss
    reg_loss_fn = losses.regulation_loss
    DICE_loss_fn = losses.DICE_loss
    grad_loss_fn = losses.gradient_loss
    cycle_loss_fn = losses.mae_loss

    # data generator
#    train_example_gen = datagenerators.example_gen_HX(cwd, studies)
    data_folder = '4D-Lung-contour-npy-v2'
    train_example_gen = datagenerators.gen_CT_RTS(cwd, studies, data_folder)
    
    loss_accu = np.array([])
    recon_loss_accu = np.array([])
    reg_loss_accu = np.array([])
    DICE_loss_accu = np.array([])
    grad_loss_accu = np.array([])
    iden_loss_accu = np.array([])
    cycle_loss_accu = np.array([])
    
    loss_out = np.array([])
    recon_loss_out = np.array([])
    reg_loss_out = np.array([])
    DICE_loss_out = np.array([])
    grad_loss_out = np.array([])
    iden_loss_out = np.array([])
    cycle_loss_out = np.array([])

    # Training loop.
    for i in range(n_iter):
        num_epoch = floor(i/100)+1
        num_reduced_factor = floor(num_epoch/2)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt = Adam(model.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_iter == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % i)
            torch.save(model.state_dict(), save_file_name)
            
        if i % 100 == 0:
            loss_accu = np.array([])
            recon_loss_accu = np.array([])
            reg_loss_accu = np.array([])
            DICE_loss_accu = np.array([])
            grad_loss_accu = np.array([])
            iden_loss_accu = np.array([])
            cycle_loss_accu = np.array([])
            print(' ')
            print('Epoch %d/%d' % (floor(i/100)+1,floor(n_iter/100)))
        # Generate the moving images and convert them to tensors.
        moving_image, fixed_image, moving_RTS, fixed_RTS = next(train_example_gen)
        patch_gen = datagenerators.gen_patch(moving_image, fixed_image, moving_RTS, fixed_RTS, vol_size, batch_size)
        for num_patch in range(point_per_iter):
            start = time.time()
            moving_patch, fixed_patch, RS_M_patch, RS_F_patch = next(patch_gen)
            input_A = torch.from_numpy(moving_patch).to(device).float()
            input_A = input_A.permute(0, 4, 3, 2, 1)
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B = input_B.permute(0, 4, 3, 2, 1)
            input_A_RTS = torch.from_numpy(RS_M_patch).to(device).float()
            input_A_RTS = input_A_RTS.permute(1, 0, 4, 3, 2)
            input_B_RTS = torch.from_numpy(RS_F_patch).to(device).float()
            input_B_RTS = input_B_RTS.permute(1, 0, 4, 3, 2)
            body_contour = torch.unsqueeze(input_B_RTS[:,0,:,:,:],0)
            lung_contour = torch.unsqueeze(input_B_RTS[:,1,:,:,:],0)
            if (input_A_RTS.size()[1] > 2) & (input_B_RTS.size()[1] > 2):
                organ_contour_A = torch.unsqueeze(input_A_RTS[:,2,:,:,:],0)
                organ_contour_B = torch.unsqueeze(input_B_RTS[:,2,:,:,:],0)
            else:
                organ_contour_A = torch.zeros_like(lung_contour)
                organ_contour_B = torch.zeros_like(lung_contour) 
#            plt.imshow(input_A.cpu()[0,0,:,:,20])
#            plt.imshow(input_B.cpu()[0,0,:,:,20])
            
            # Run the data through the model to produce warp and flow field
            # Seduo B
            warp_A2sB, flow_A2sB = model(input_A, input_B)
            warp_RTS_A2sB = ST(organ_contour_A ,flow_A2sB)
            if torch.max(lung_contour) > 0:
                recon_loss_A2sB = lung_param * sim_loss_fn(input_B, warp_A2sB, body_contour)
            else:
                recon_loss_A2sB = sim_loss_fn(input_B, warp_A2sB, body_contour)
            reg_loss_A2sB = reg_param * reg_loss_fn(flow_A2sB)
            DICE_loss_A2sB = DICE_param * DICE_loss_fn(organ_contour_B, warp_RTS_A2sB)
            grad_loss_A2sB = grad_param * grad_loss_fn(flow_A2sB)
            loss_A2sB = recon_loss_A2sB + reg_loss_A2sB + grad_loss_A2sB + DICE_loss_A2sB
            
            # Seduo A
            warp_B2sA, flow_B2sA = model(input_B, input_A)
            warp_RTS_B2sA = ST(organ_contour_B, flow_B2sA)
            if torch.max(lung_contour) > 0:
                recon_loss_B2sA = lung_param * sim_loss_fn(input_A, warp_B2sA, body_contour)
            else:
                recon_loss_B2sA = sim_loss_fn(input_A, warp_B2sA, body_contour)
            reg_loss_B2sA = reg_param * reg_loss_fn(flow_B2sA)
            DICE_loss_B2sA = DICE_param * DICE_loss_fn(organ_contour_A, warp_RTS_B2sA)
            grad_loss_B2sA = grad_param * grad_loss_fn(flow_B2sA)
            loss_B2sA = recon_loss_B2sA + reg_loss_B2sA + grad_loss_B2sA + DICE_loss_B2sA
            
            # Seduo B back to Seduo Seduo A
            warp_sB2sA, flow_sB2sA = model(warp_A2sB, warp_B2sA)
                       
            # Seduo A back to Seduo Seduo B
            warp_sA2sB, flow_sA2sB = model(warp_B2sA, warp_A2sB)
           
            # Cycle loss
            cycle_loss_a = cycle_param * cycle_loss_fn(input_A, warp_sB2sA)
            cycle_loss_b = cycle_param * cycle_loss_fn(input_B, warp_sA2sB)
            cycle_loss = cycle_loss_a + cycle_loss_b
            
            # Identity loss
            warp_A2A, flow_A2A = model(input_A, input_A)
            iden_loss_A = sim_loss_fn(warp_A2A, input_A, body_contour) 
            warp_B2B, flow_B2B = model(input_B, input_B)
            iden_loss_B = sim_loss_fn(warp_B2B, input_B, body_contour)
            iden_loss = iden_param * iden_loss_A + iden_param * iden_loss_B
            
            #sum loss
            loss = loss_A2sB + loss_B2sA + iden_loss + cycle_loss
            recon_loss = recon_loss_A2sB + recon_loss_B2sA 
            reg_loss = reg_loss_A2sB + reg_loss_B2sA
            DICE_loss = DICE_loss_A2sB + DICE_loss_B2sA
            grad_loss = grad_loss_A2sB + grad_loss_B2sA
            
            # Print loss
            loss_accu = np.append(loss_accu,loss.item())
            recon_loss_accu = np.append(recon_loss_accu, recon_loss.item())
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())
            DICE_loss_accu = np.append(DICE_loss_accu, DICE_loss.item())
            iden_loss_accu = np.append(iden_loss_accu, iden_loss.item())
            cycle_loss_accu = np.append(cycle_loss_accu, cycle_loss.item())
            
            if (i == 0) & (num_patch == 0):
                print('saved\n')
                loss_out = np.append(loss_out, np.mean(loss_accu))
                recon_loss_out = np.append(recon_loss_out, np.mean(recon_loss_accu))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
                DICE_loss_out = np.append(DICE_loss_out, np.mean(DICE_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))
                iden_loss_out = np.append(iden_loss_out, np.mean(iden_loss_accu))
                cycle_loss_out = np.append(cycle_loss_out, np.mean(cycle_loss_accu))
            
            # Backwards and optimize
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
            
            end = time.time()
            print("\r n_iter: %d/100, num_patch: %d/%d, time = %fs, lr: %f, Loss: %f, sim_loss: %f, reg_loss: %f, grad_loss: %f, DICE_loss: %f, iden_loss: %f, cycle_loss: %f" 
              % (i-100 * floor(i/100) + 1, num_patch + 1,  point_per_iter, end-start, lr_new,
                 np.mean(loss_accu), np.mean(recon_loss_accu), np.mean(reg_loss_accu), np.mean(grad_loss_accu),
                 np.mean(DICE_loss_accu), np.mean(iden_loss_accu), np.mean(cycle_loss_accu)), end = '', flush=True)
        loss_out = np.append(loss_out, np.mean(loss_accu))
        recon_loss_out = np.append(recon_loss_out, np.mean(recon_loss_accu))
        reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
        DICE_loss_out = np.append(DICE_loss_out, np.mean(DICE_loss_accu))
        grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))
        iden_loss_out = np.append(iden_loss_out, np.mean(iden_loss_accu))
        cycle_loss_out = np.append(cycle_loss_out, np.mean(cycle_loss_accu))
    return model, loss_out, recon_loss_out, reg_loss_out, DICE_loss_out, grad_loss_out, iden_loss_out, cycle_loss_out

def train_simple_organs(gpu,
          cwd,
          vol_size,
          nf_enc,
          nf_dec,
          studies,
          lr,
          n_iter,
          data_loss,
          params, 
          batch_size,
          n_save_iter,
          model_dir,
          point_per_iter):
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
    iden_param = params[4]
    cycle_param = params[5]
    
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    model = cvpr2018_net(vol_size, nf_enc, nf_dec)
    ST = SpatialTransformer(vol_size)        
    model.to(device)
    ST.to(device)
    
    
    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)

    sim_loss_fn = losses.cc_loss_noRTS if data_loss == "cc" else losses.mse_loss
#    diff_loss_fn = losses.diff_loss
    reg_loss_fn = losses.regulation_loss
    DICE_loss_fn = losses.DICE_loss
    grad_loss_fn = losses.gradient_loss
    cycle_loss_fn = losses.mae_loss

    # data generator
#    train_example_gen = datagenerators.example_gen_HX(cwd, studies)
    data_folder = '4D-Lung-contour-npy-v3'
    train_example_gen = datagenerators.gen_CT_RTS_2_28(cwd, studies, data_folder)
    
    loss_accu = np.array([])
    recon_loss_accu = np.array([])
    reg_loss_accu = np.array([])
    DICE_loss_accu = np.array([])
    grad_loss_accu = np.array([])
    iden_loss_accu = np.array([])
    cycle_loss_accu = np.array([])
    
    loss_out = np.array([])
    recon_loss_out = np.array([])
    reg_loss_out = np.array([])
    DICE_loss_out = np.array([])
    grad_loss_out = np.array([])
    iden_loss_out = np.array([])
    cycle_loss_out = np.array([])

    # Training loop.
    for i in range(n_iter):
        num_epoch = floor(i/100)+1
        num_reduced_factor = floor(num_epoch)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt = Adam(model.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_iter == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % i)
            torch.save(model.state_dict(), save_file_name)
            
        if i % 100 == 0:
            loss_accu = np.array([])
            recon_loss_accu = np.array([])
            reg_loss_accu = np.array([])
            DICE_loss_accu = np.array([])
            grad_loss_accu = np.array([])
            iden_loss_accu = np.array([])
            cycle_loss_accu = np.array([])
            print(' ')
            print('Epoch %d/%d' % (floor(i/100)+1,floor(n_iter/100)))
        # Generate the moving images and convert them to tensors.
        moving_image, fixed_image, moving_RTS, fixed_RTS = next(train_example_gen)
        patch_gen = datagenerators.gen_patch(moving_image, fixed_image, moving_RTS, fixed_RTS, vol_size, batch_size)
        for num_patch in range(point_per_iter):
            start = time.time()
            moving_patch, fixed_patch, RS_M_patch, RS_F_patch = next(patch_gen)
            input_A = torch.from_numpy(moving_patch).to(device).float()
            input_A = input_A.permute(0, 4, 3, 2, 1)
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B = input_B.permute(0, 4, 3, 2, 1)
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
#            plt.imshow(input_A.cpu()[0,0,:,:,20])
#            plt.imshow(input_B.cpu()[0,0,:,:,20])
            
            # Run the data through the model to produce warp and flow field
            # Seduo B
            warp_A2sB, flow_A2sB = model(input_A, input_B)
            warp_RTS_A2sB = ST(organ_contour_A ,flow_A2sB)
            if torch.max(lung_contour) > 0:
                recon_loss_A2sB = lung_param * sim_loss_fn(input_B, warp_A2sB, body_contour)
            else:
                recon_loss_A2sB = sim_loss_fn(input_B, warp_A2sB, body_contour)
            reg_loss_A2sB = reg_param * reg_loss_fn(flow_A2sB)
            DICE_loss_A2sB = DICE_param * DICE_loss_fn(organ_contour_B, warp_RTS_A2sB)
            grad_loss_A2sB = grad_param * grad_loss_fn(flow_A2sB)
            loss_A2sB = recon_loss_A2sB + reg_loss_A2sB + grad_loss_A2sB + DICE_loss_A2sB
            
            # Seduo A
            warp_B2sA, flow_B2sA = model(input_B, input_A)
            warp_RTS_B2sA = ST(organ_contour_B, flow_B2sA)
            if torch.max(lung_contour) > 0:
                recon_loss_B2sA = lung_param * sim_loss_fn(input_A, warp_B2sA, body_contour)
            else:
                recon_loss_B2sA = sim_loss_fn(input_A, warp_B2sA, body_contour)
            reg_loss_B2sA = reg_param * reg_loss_fn(flow_B2sA)
            DICE_loss_B2sA = DICE_param * DICE_loss_fn(organ_contour_A, warp_RTS_B2sA)
            grad_loss_B2sA = grad_param * grad_loss_fn(flow_B2sA)
            loss_B2sA = recon_loss_B2sA + reg_loss_B2sA + grad_loss_B2sA + DICE_loss_B2sA
            
            # Seduo B back to Seduo Seduo A
            warp_sB2sA, flow_sB2sA = model(warp_A2sB, warp_B2sA)
                       
            # Seduo A back to Seduo Seduo B
            warp_sA2sB, flow_sA2sB = model(warp_B2sA, warp_A2sB)
           
            # Cycle loss
            cycle_loss_a = cycle_param * cycle_loss_fn(input_A, warp_sB2sA)
            cycle_loss_b = cycle_param * cycle_loss_fn(input_B, warp_sA2sB)
            cycle_loss = cycle_loss_a + cycle_loss_b
            
            # Identity loss
            warp_A2A, flow_A2A = model(input_A, input_A)
            iden_loss_A = sim_loss_fn(warp_A2A, input_A, body_contour) 
            warp_B2B, flow_B2B = model(input_B, input_B)
            iden_loss_B = sim_loss_fn(warp_B2B, input_B, body_contour)
            iden_loss = iden_param * iden_loss_A + iden_param * iden_loss_B
            
            #sum loss
            loss = loss_A2sB + loss_B2sA + iden_loss + cycle_loss
            recon_loss = recon_loss_A2sB + recon_loss_B2sA 
            reg_loss = reg_loss_A2sB + reg_loss_B2sA
            DICE_loss = DICE_loss_A2sB + DICE_loss_B2sA
            grad_loss = grad_loss_A2sB + grad_loss_B2sA
            
            # Print loss
            loss_accu = np.append(loss_accu,loss.item())
            recon_loss_accu = np.append(recon_loss_accu, recon_loss.item())
            grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
            reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())
            DICE_loss_accu = np.append(DICE_loss_accu, DICE_loss.item())
            iden_loss_accu = np.append(iden_loss_accu, iden_loss.item())
            cycle_loss_accu = np.append(cycle_loss_accu, cycle_loss.item())
            
            if (i == 0) & (num_patch == 0):
                print('saved\n')
                loss_out = np.append(loss_out, np.mean(loss_accu))
                recon_loss_out = np.append(recon_loss_out, np.mean(recon_loss_accu))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
                DICE_loss_out = np.append(DICE_loss_out, np.mean(DICE_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))
                iden_loss_out = np.append(iden_loss_out, np.mean(iden_loss_accu))
                cycle_loss_out = np.append(cycle_loss_out, np.mean(cycle_loss_accu))
            
            # Backwards and optimize
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
            
            end = time.time()
            print("\r n_iter: %d/100, num_patch: %d/500, time = %fs, lr: %f, Loss: %f, sim_loss: %f, reg_loss: %f, grad_loss: %f, DICE_loss: %f, iden_loss: %f, cycle_loss: %f" 
              % (i-100 * floor(i/100) + 1, num_patch + 1, end-start, lr_new,
                 np.mean(loss_accu), np.mean(recon_loss_accu), np.mean(reg_loss_accu), np.mean(grad_loss_accu),
                 np.mean(DICE_loss_accu), np.mean(iden_loss_accu), np.mean(cycle_loss_accu)), end = '', flush=True)
        loss_out = np.append(loss_out, np.mean(loss_accu))
        recon_loss_out = np.append(recon_loss_out, np.mean(recon_loss_accu))
        reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
        DICE_loss_out = np.append(DICE_loss_out, np.mean(DICE_loss_accu))
        grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))
        iden_loss_out = np.append(iden_loss_out, np.mean(iden_loss_accu))
        cycle_loss_out = np.append(cycle_loss_out, np.mean(cycle_loss_accu))
    return model, loss_out, recon_loss_out, reg_loss_out, DICE_loss_out, grad_loss_out, iden_loss_out, cycle_loss_out

def train_C2F(gpu,
          cwd,
          coarse_size,
          fine_size,
          nf_enc_coarse,
          nf_dec_coarse,
          nf_enc_fine,
          nf_dec_fine,
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
    iden_param = params[4]
#    cycle_param = params[5]
    coarse = True
    fine = False
    
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    model_coarse = voxelmorph_coarse(coarse_size, nf_enc_coarse, nf_dec_coarse, coarse)
    model_fine = voxelmorph_fine(fine_size, nf_enc_fine, nf_dec_fine, fine)
    ST_coarse = SpatialTransformer(coarse_size)
    ST_fine = SpatialTransformer(fine_size)    
    
    model_coarse.to(device)
    model_fine.to(device)
    ST_coarse.to(device)
    ST_fine.to(device)
    
    
    # Set optimizer and losses
    opt_coarse = Adam(model_coarse.parameters(), lr=lr)
    opt_fine = Adam(model_fine.parameters(), lr=lr)

    sim_loss_fn = losses.cc_loss_noRTS if data_loss == "cc" else losses.mse_loss_noRTS
#    diff_loss_fn = losses.diff_loss
    reg_loss_fn = losses.regulation_loss
    DICE_loss_fn = losses.DICE_loss
    grad_loss_fn = losses.gradient_loss
#    cycle_loss_fn = losses.mae_loss

    # data generator
#    train_example_gen = datagenerators.example_gen_HX(cwd, studies)
    data_folder = '4D-Lung-contour-npy-v3'
    train_example_gen = datagenerators.gen_CT_RTS_2_28(cwd, studies, data_folder)
    
    loss_accu = np.array([])
    recon_loss_accu = np.array([])
    reg_loss_accu = np.array([])
    DICE_loss_accu = np.array([])
    grad_loss_accu = np.array([])
    iden_loss_accu = np.array([])
#    cycle_loss_accu = np.array([])
    
    loss_out = np.array([])
    recon_loss_out = np.array([])
    reg_loss_out = np.array([])
    DICE_loss_out = np.array([])
    grad_loss_out = np.array([])
    iden_loss_out = np.array([])
#    cycle_loss_out = np.array([])

    # Training loop.
    for i in range(n_iter):
        num_epoch = floor(i/100)+1
        num_reduced_factor = floor(num_epoch/5)
        lr_new = lr * pow(0.5,num_reduced_factor-1)
        opt_coarse = Adam(model_coarse.parameters(), lr=lr_new)
        opt_fine = Adam(model_fine.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_iter == 0:
            save_file_name_coarse = os.path.join(model_dir, 'coarse%d.ckpt' % i)
            save_file_name_fine = os.path.join(model_dir, 'fine%d.ckpt' % i)
            torch.save(model_coarse.state_dict(), save_file_name_coarse)
            torch.save(model_fine.state_dict(), save_file_name_fine)
            
        if i % 100 == 0:
            loss_accu = np.array([])
            recon_loss_accu = np.array([])
            reg_loss_accu = np.array([])
            DICE_loss_accu = np.array([])
            grad_loss_accu = np.array([])
            iden_loss_accu = np.array([])
#            cycle_loss_accu = np.array([])
            print(' ')
            print('Epoch %d/%d' % (floor(i/100)+1,floor(n_iter/100)))
        # Generate the moving images and convert them to tensors.
        moving_image, fixed_image, moving_RTS, fixed_RTS = next(train_example_gen)
        patch_gen = datagenerators.gen_patch(moving_image, fixed_image, moving_RTS, fixed_RTS, coarse_size, batch_size)
        for num_patch in range(n_per_iter):
            start = time.time()
            moving_patch, fixed_patch, RS_M_patch, RS_F_patch = next(patch_gen)
            input_A = torch.from_numpy(moving_patch).to(device).float()
            input_A = input_A.permute(0, 4, 3, 2, 1)
            input_B = torch.from_numpy(fixed_patch).to(device).float()
            input_B = input_B.permute(0, 4, 3, 2, 1)
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
#            plt.imshow(input_A.cpu()[0,0,:,:,20])
#            plt.imshow(input_B.cpu()[0,0,:,:,20])
            
            # Run the data through the model to produce warp and flow field
            # Coarse network
            # Seduo B
            warp_A2sB, flow_A2sB, flow_A2sB_up = model_coarse(input_A, input_B)
            warp_RTS_A2sB = ST_coarse(organ_contour_A ,flow_A2sB_up)
            if torch.max(lung_contour) > 0:
                recon_loss_A2sB = lung_param * sim_loss_fn(input_B, warp_A2sB, lung_contour) + sim_loss_fn(input_B, warp_A2sB, body_contour)
            else:
                recon_loss_A2sB = sim_loss_fn(input_B, warp_A2sB, body_contour)
            reg_loss_A2sB = reg_param * reg_loss_fn(flow_A2sB)
            DICE_loss_A2sB = DICE_param * DICE_loss_fn(organ_contour_B, warp_RTS_A2sB)
            grad_loss_A2sB = grad_param * grad_loss_fn(flow_A2sB)
            loss_A2sB = recon_loss_A2sB + reg_loss_A2sB + grad_loss_A2sB + DICE_loss_A2sB
            
#            # Seduo A
#            warp_B2sA, flow_B2sA, flow_B2sA_up = model_coarse(input_B, input_A)
#            warp_RTS_B2sA = ST_coarse(organ_contour_B, flow_B2sA_up)
#            if torch.max(lung_contour) > 0:
#                recon_loss_B2sA = lung_param * sim_loss_fn(input_A, warp_B2sA, body_contour)
#            else:
#                recon_loss_B2sA = sim_loss_fn(input_A, warp_B2sA, body_contour)
#            reg_loss_B2sA = reg_param * reg_loss_fn(flow_B2sA)
#            DICE_loss_B2sA = DICE_param * DICE_loss_fn(organ_contour_A, warp_RTS_B2sA)
#            grad_loss_B2sA = grad_param * grad_loss_fn(flow_B2sA)
#            loss_B2sA = recon_loss_B2sA + reg_loss_B2sA + grad_loss_B2sA + DICE_loss_B2sA
            
            # Fine network 32x32x32 [:,:,16:48,16:48,16:48]
            low_limit = int(fine_size[0]/2)
            up_limit = int(coarse_size[0] - fine_size[0]/2)
            input_A_fine = input_A[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
            input_B_fine = input_B[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
            warp_sB_fine = warp_A2sB[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
            organ_contour_B_fine = organ_contour_B[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
            body_contour_fine = body_contour[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
            lung_contour_fine = lung_contour[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
            warp_RTS_sB_fine = warp_RTS_A2sB[:,:,low_limit:up_limit,low_limit:up_limit,low_limit:up_limit]
            
            warp_A2sB_fine, flow_A2sB_fine, flow_A2sB_up_fine = model_fine(warp_sB_fine, input_B_fine)
            warp_RTS_A2sB_fine = ST_fine(warp_RTS_sB_fine ,flow_A2sB_up_fine)
            if torch.max(lung_contour) > 0:
                recon_loss_A2sB_fine = lung_param * sim_loss_fn(input_B_fine, warp_A2sB_fine, lung_contour_fine) + sim_loss_fn(input_B_fine, warp_A2sB_fine, body_contour_fine)
            else:
                recon_loss_A2sB_fine = sim_loss_fn(input_B_fine, warp_A2sB_fine, body_contour_fine)            
            reg_loss_A2sB_fine = reg_param * reg_loss_fn(flow_A2sB_fine)
            DICE_loss_A2sB_fine = DICE_param * DICE_loss_fn(organ_contour_B_fine, warp_RTS_A2sB_fine)
            grad_loss_A2sB_fine = grad_param * grad_loss_fn(flow_A2sB_fine)
            loss_A2sB_fine = recon_loss_A2sB_fine + reg_loss_A2sB_fine + grad_loss_A2sB_fine + DICE_loss_A2sB_fine            
            
#            # Seduo B back to Seduo Seduo A
#            warp_sB2sA, flow_sB2sA = model_coarse(warp_A2sB, warp_B2sA)
#                       
#            # Seduo A back to Seduo Seduo B
#            warp_sA2sB, flow_sA2sB = model_coarse(warp_B2sA, warp_A2sB)
#           
#            # Cycle loss
#            cycle_loss_a = cycle_param * cycle_loss_fn(input_A, warp_sB2sA)
#            cycle_loss_b = cycle_param * cycle_loss_fn(input_B, warp_sA2sB)
#            cycle_loss = cycle_loss_a + cycle_loss_b
            
            # Identity loss
#            warp_A2A, flow_A2A = model_coarse(input_A, input_A)
#            iden_loss_A = sim_loss_fn(warp_A2A, input_A, body_contour) 
#            warp_B2B, flow_B2B = model_coarse(input_B, input_B)
#            iden_loss_B = sim_loss_fn(warp_B2B, input_B, body_contour)
#            iden_loss = iden_param * iden_loss_A + iden_param * iden_loss_B
            warp_A2A, flow_A2A, flow_A2A_up = model_coarse(input_A, input_A)
            iden_loss_A = iden_param * sim_loss_fn(warp_A2A, input_A, body_contour_fine)
            warp_A2A_fine, flow_A2A_fine, flow_A2A_up_fine = model_fine(input_A_fine, input_A_fine)
            iden_loss_A_fine = iden_param * sim_loss_fn(warp_A2A_fine, input_A_fine, body_contour_fine)
            
            #sum loss
#            loss = loss_A2sB + loss_B2sA + iden_loss + cycle_loss
#            recon_loss = recon_loss_A2sB + recon_loss_B2sA 
#            reg_loss = reg_loss_A2sB + reg_loss_B2sA
#            DICE_loss = DICE_loss_A2sB + DICE_loss_B2sA
#            grad_loss = grad_loss_A2sB + grad_loss_B2sA
            loss_coarse = loss_A2sB + iden_loss_A
            loss_fine = loss_A2sB_fine + iden_loss_A_fine
            recon_loss_coarse = recon_loss_A2sB
            recon_loss_fine = recon_loss_A2sB_fine 
            reg_loss_coarse = reg_loss_A2sB
            reg_loss_fine = reg_loss_A2sB_fine
            DICE_loss_coarse = DICE_loss_A2sB
            DICE_loss_fine = DICE_loss_A2sB_fine
            grad_loss_coarse = grad_loss_A2sB
            grad_loss_fine = grad_loss_A2sB_fine
            iden_loss_coarse = iden_loss_A
            iden_loss_fine = iden_loss_A_fine
            
            # Print loss
            loss_accu = np.append(loss_accu,loss_coarse.item()+loss_fine.item())
            loss_recon_accu = np.append(recon_loss_accu, recon_loss_coarse.item()+recon_loss_fine.item())
            loss_grad_accu = np.append(grad_loss_accu, grad_loss_coarse.item()+grad_loss_fine.item())
            loss_reg_accu = np.append(reg_loss_accu, reg_loss_coarse.item()+reg_loss_fine.item())
            loss_DICE_accu = np.append(DICE_loss_accu, DICE_loss_coarse.item()+DICE_loss_fine.item())
            loss_iden_accu = np.append(iden_loss_accu, iden_loss_coarse.item()+iden_loss_fine.item())
#            cycle_loss_accu = np.append(cycle_loss_accu, cycle_loss.item())
            
            if (i == 0) & (num_patch == 0):
                print('saved\n')
                loss_out = np.append(loss_out, np.mean(loss_accu))
                recon_loss_out = np.append(recon_loss_out, np.mean(recon_loss_accu))
                reg_loss_out = np.append(reg_loss_out, np.mean(reg_loss_accu))
                DICE_loss_out = np.append(DICE_loss_out, np.mean(DICE_loss_accu))
                grad_loss_out = np.append(grad_loss_out, np.mean(grad_loss_accu))
                iden_loss_out = np.append(iden_loss_out, np.mean(iden_loss_accu))
#                cycle_loss_out = np.append(cycle_loss_out, np.mean(cycle_loss_accu))
            
            # Backwards and optimize
            opt_coarse.zero_grad()
            opt_fine.zero_grad()
            loss_coarse.backward(retain_graph=True)
            loss_fine.backward(retain_graph=True)
            opt_coarse.step()
            opt_fine.step()
            
            end = time.time()
            print("\r n_iter: %d/100, num_patch: %d/%d, time = %fs, lr: %e, Loss: %f, sim_loss: %f, reg_loss: %f, grad_loss: %f, DICE_loss: %f, iden_loss: %f" 
              % (i-100 * floor(i/100) + 1, num_patch + 1, n_per_iter, end-start, lr_new,
                 np.mean(loss_accu), np.mean(loss_recon_accu), np.mean(loss_reg_accu), np.mean(loss_grad_accu),
                 np.mean(loss_DICE_accu), np.mean(loss_iden_accu)), end = '', flush=True)
        loss_out = np.append(loss_out, np.mean(loss_accu))
        recon_loss_out = np.append(recon_loss_out, np.mean(loss_recon_accu))
        reg_loss_out = np.append(reg_loss_out, np.mean(loss_reg_accu))
        DICE_loss_out = np.append(DICE_loss_out, np.mean(loss_DICE_accu))
        grad_loss_out = np.append(grad_loss_out, np.mean(loss_grad_accu))
        iden_loss_out = np.append(iden_loss_out, np.mean(loss_iden_accu))
#        cycle_loss_out = np.append(cycle_loss_out, np.mean(cycle_loss_accu))
    return model_coarse, model_fine, loss_out, recon_loss_out, reg_loss_out, DICE_loss_out, grad_loss_out, iden_loss_out
