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
import datagenerators
import losses
from math import floor
import time



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
#    top_param = params[4]
    
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    model = cvpr2018_net(vol_size, nf_enc, nf_dec)
    ST = SpatialTransformer(vol_size)        
    model.to(device)
    ST.to(device)
    
    
    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)

    sim_loss_fn = losses.cc_loss if data_loss == "cc" else losses.mse_loss
#    diff_loss_fn = losses.diff_loss
    reg_loss_fn = losses.regulation_loss
    DICE_loss_fn = losses.DICE_loss
    grad_loss_fn = losses.gradient_loss

    # data generator
#    train_example_gen = datagenerators.example_gen_HX(cwd, studies)
    train_example_gen = datagenerators.gen_CT_RTS(cwd, studies)
    
    loss_accu = np.array([])
    lung_loss_accu = np.array([])
    recon_loss_accu = np.array([])
#    diff_loss_accu = np.array([])
    reg_loss_accu = np.array([])
    DICE_loss_accu = np.array([])
    grad_loss_accu = np.array([])

    # Training loop.
    for i in range(n_iter):
        start = time.time()
        num_epoch = floor(i/100)+1
        num_reduced_factor = floor(num_epoch/100)
        lr_new = lr * pow(0.5,num_reduced_factor)
        opt = Adam(model.parameters(), lr=lr_new)
        # Save model checkpoint
        if i % n_save_iter == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % i)
            torch.save(model.state_dict(), save_file_name)
            
        if i % 100 == 0:
            loss_accu = np.array([])
            lung_loss_accu = np.array([])
            recon_loss_accu = np.array([])
#           diff_loss_accu = np.array([])
            reg_loss_accu = np.array([])
            DICE_loss_accu = np.array([])
            print(' ')
            print('Epoch %d/%d' % (floor(i/100)+1,floor(n_iter/100)))
        # Generate the moving images and convert them to tensors.
        moving_image, fixed_image, moving_RTS, fixed_RTS = next(train_example_gen)
#        moving_image = moving_image/2000 + 0.5
#        fixed_image = fixed_image/2000 + 0.5
#        moving_image, fixed_image = next(train_example_gen)
        input_moving = torch.from_numpy(moving_image).to(device).float()
#        input_moving = input_moving.permute(0, 4, 1, 2, 3)
        input_moving = input_moving.permute(0, 4, 3, 2, 1)
        input_fixed = torch.from_numpy(fixed_image).to(device).float()
        input_fixed = input_fixed.permute(0, 4, 3, 2, 1)
        input_moving_RTS = torch.from_numpy(moving_RTS).to(device).float()
        input_moving_RTS = input_moving_RTS.permute(1, 0, 4, 3, 2)
        input_fixed_RTS = torch.from_numpy(fixed_RTS).to(device).float()
        input_fixed_RTS = input_fixed_RTS.permute(1, 0, 4, 3, 2)
        body_contour = torch.unsqueeze(input_fixed_RTS[:,0,:,:,:],0)
        lung_contour = torch.unsqueeze(input_fixed_RTS[:,1,:,:,:],0)
        organ_contour_M = torch.unsqueeze(input_moving_RTS[:,2,:,:,:],0)
        organ_contour_F = torch.unsqueeze(input_fixed_RTS[:,2,:,:,:],0)
        
        # Run the data through the model to produce warp and flow field
        warp, flow = model(input_moving, input_fixed)
#        warp, flow, warp_RTS = model(input_moving, input_moving_RTS, input_fixed)
        
        # Calculate loss
        warp_RTS = ST(organ_contour_M ,flow)
        recon_loss = sim_loss_fn(input_fixed, warp, body_contour)        
        lung_loss = sim_loss_fn(input_fixed, warp, lung_contour)
#        diff_loss = diff_loss_fn(flow)
        reg_loss = reg_loss_fn(flow)
        DICE_loss = DICE_loss_fn(organ_contour_F, warp_RTS)
        grad_loss = grad_loss_fn(flow)
        
        loss = recon_loss + reg_param * reg_loss + grad_param * grad_loss + lung_param * lung_loss + DICE_param * DICE_loss
        
        # Print loss
        loss_accu = np.append(loss_accu,loss.item())
        recon_loss_accu = np.append(recon_loss_accu, recon_loss.item())
        grad_loss_accu = np.append(grad_loss_accu, grad_loss.item())
        reg_loss_accu = np.append(reg_loss_accu, reg_loss.item())
        DICE_loss_accu = np.append(DICE_loss_accu, DICE_loss.item())
        lung_loss_accu = np.append(lung_loss_accu, lung_loss.item())
        
        # Backwards and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        end = time.time()
        print("\r n_iter: %d/100, time = %fs, lr: %f, Loss: %f, sim_loss: %f, reg_loss: %f, grad_loss: %f, DICE_loss: %f, lung_loss: %f" 
              % (i-100 * floor(i/100) + 1, end-start, lr_new,
                 np.mean(loss_accu), np.mean(recon_loss_accu), reg_param * np.mean(reg_loss_accu), grad_param * np.mean(grad_loss_accu),
                 np.mean(DICE_loss_accu) * DICE_param, np.mean(lung_loss_accu) * lung_param), end = '', flush=True)
    return model

#
#if __name__ == "__main__":
#    with warnings.catch_warnings():
#        warnings.filterwarnings("ignore", category=DeprecationWarning)
#
#    parser = ArgumentParser()
#
#    parser.add_argument("--gpu",
#                        type=str,
#                        default='0',
#                        help="gpu id")
#
#    parser.add_argument("--cwd",
#                        type=str,
#                        help="current path")
#
#    parser.add_argument("--vol_size",
#                        type=tuple,
#                        help="vol_size")
#    
#    parser.add_argument("--nf_enc",
#                        type=list,
#                        help="number of filters in encoder")
#    
#    parser.add_argument("--nf_dec",
#                        type=list,
#                        help="number of filters in decoder")    
#
#    parser.add_argument("--studies",
#                        type=list,
#                        help="folders of scans")   
#
#    parser.add_argument("--lr",
#                        type=float,
#                        dest="lr",
#                        default=1e-4,
#                        help="learning rate")
#
#    parser.add_argument("--n_iter",
#                        type=int,
#                        dest="n_iter",
#                        default=150000,
#                        help="number of iterations")
#
#    parser.add_argument("--data_loss",
#                        type=str,
#                        dest="data_loss",
#                        default='ncc',
#                        help="data_loss: mse of ncc")
#
#
#    parser.add_argument("--lambda", 
#                        type=float,
#                        dest="reg_param", 
#                        default=0.01,  # recommend 1.0 for ncc, 0.01 for mse
#                        help="regularization parameter")
#
#    parser.add_argument("--batch_size", 
#                        type=int,
#                        dest="batch_size", 
#                        default=1,
#                        help="batch_size")
#
#    parser.add_argument("--n_save_iter", 
#                        type=int,
#                        dest="n_save_iter", 
#                        default=500,
#                        help="frequency of model saves")
#
#    parser.add_argument("--model_dir", 
#                        type=str,
#                        dest="model_dir", 
#                        default='./models-PyTorch/',
#                        help="models folder")
#
#
#    train(**vars(parser.parse_args()))
#
