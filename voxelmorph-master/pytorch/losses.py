"""
*Preliminary* pytorch implementation.

Losses for VoxelMorph
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :]) 
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :]) 
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1]) 

    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0

def gradient_loss_v2(s, penalty='l2', order = '1'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :]) 
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :]) 
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1]) 
    if(order == '2'):
        dy = torch.abs(dy[:, :, 1:, :, :] - dy[:, :, :-1, :, :]) 
        dx = torch.abs(dx[:, :, :, 1:, :] - dx[:, :, :, :-1, :]) 
        dz = torch.abs(dz[:, :, :, :, 1:] - dz[:, :, :, :, :-1]) 
    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0

def gradient_loss_z(s, penalty='l2'):
    s_z = s[:,2:,:,:,:]
    dy = torch.abs(s_z[:, :, 1:, :, :] - s_z[:, :, :-1, :, :]) 
    dx = torch.abs(s_z[:, :, :, 1:, :] - s_z[:, :, :, :-1, :]) 
    dz = torch.abs(s_z[:, :, :, :, 1:] - s_z[:, :, :, :, :-1]) 
    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz
    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0

def regulation_loss_xy(flow):
    flow_xy = flow[:,:2,:,:,:]
    return torch.mean(flow_xy*flow_xy)

def mse_loss_ReallynoRTS(x, y):
    mse = torch.mean( (x - y) ** 2 )        
    return mse   

def mse_loss_ReallynoRTS_mask(x, y, mask):
    total = torch.sum(mask)
    if x.shape[1] == 1:
        mse = (torch.sum( (x[mask>0] - y[mask>0]) ** 2 ))/total
    else:
        for i in range(x.shape[1]):
            x_slice = torch.unsqueeze(x[:,i,:,:,:],0)
            y_slice = torch.unsqueeze(y[:,i,:,:,:],0)
            x_slice[mask==0] = 0
            y_slice[mask==0] = 0
            x[:,i,:,:,:] = x_slice.clone()
            y[:,i,:,:,:] = y_slice.clone()
        mse = (torch.sum((x - y)**2))/total
    return mse

def cc_loss_ReallynoRTS(x,y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    top = torch.mean((x - x_mean) * (y - y_mean))
    bot = torch.std(x) * torch.std(y)
    cc = top/bot
    return 1-cc 

def diff_loss(flow):
    flow_u = flow[0,0,:,:,:]
    flow_u_diffx = flow_u[1:,:,:] - flow_u[:-1,:,:]
    flow_u_diffy = flow_u[:,1:,:] - flow_u[:,:-1,:]
    flow_u_diffz = flow_u[:,:,1:] - flow_u[:,:,:-1]
    loss_u = torch.mean(flow_u_diffx * flow_u_diffx) + torch.mean(flow_u_diffy * flow_u_diffy) + torch.mean(flow_u_diffz * flow_u_diffz)
    
    flow_v = flow[0,1,:,:,:]
    flow_v_diffx = flow_v[1:,:,:] - flow_v[:-1,:,:]
    flow_v_diffy = flow_v[:,1:,:] - flow_v[:,:-1,:]
    flow_v_diffz = flow_v[:,:,1:] - flow_v[:,:,:-1]
    loss_v = torch.mean(flow_v_diffx * flow_v_diffx) + torch.mean(flow_v_diffy * flow_v_diffy) + torch.mean(flow_v_diffz * flow_v_diffz)  
    
    flow_w = flow[0,2,:,:,:]
    flow_w_diffx = flow_w[1:,:,:] - flow_w[:-1,:,:]
    flow_w_diffy = flow_w[:,1:,:] - flow_w[:,:-1,:]
    flow_w_diffz = flow_w[:,:,1:] - flow_w[:,:,:-1]
    loss_w = torch.mean(flow_w_diffx * flow_w_diffx) + torch.mean(flow_w_diffy * flow_w_diffy) + torch.mean(flow_w_diffz * flow_w_diffz)      
    
    loss = loss_u + loss_v + loss_w
    
    return loss

def regulation_loss(flow):
    return torch.mean(flow*flow)
    

def mse_loss_noRTS(x, y, RTS):
    mse = torch.mean( (x - y) ** 2 )        
    return mse

def mae_loss(x,y):
    mae = torch.mean(torch.abs(x-y))
    return mae

#def mse_loss(x, y):
#    mse = torch.mean( ((x - y) ) ** 2 )        
#    return mse

def DICE_loss(y_true, y_pred):
    laplace_smooth = 1
    intersection = torch.sum(y_pred * y_true)
    DICE = torch.div((2 * intersection + laplace_smooth),(torch.sum(y_true) + torch.sum(y_pred) + laplace_smooth))
    loss = 1-DICE
    return loss

def JSC_loss(y_true, y_pred):
    laplace_smooth = 1
    intersection = torch.sum(y_pred * y_true)
    union = y_true + y_pred
    union[union > 0.5] = 1
    union[union < 0.5] = 0
    loss = torch.div((intersection + laplace_smooth), (torch.sum(union)+1))
    return 1 - loss

def Distance_loss(RTS1, RTS2):
    RTS1_crop = RTS1[0,0,:]
    RTS2_crop = RTS2[0,0,:]
    x_1, y_1, z_1 = torch.where(RTS1_crop>0.5)
    x_2, y_2, z_2 = torch.where(RTS2_crop>0.5)
    center_1 = [torch.mean(x_1.float()),torch.mean(y_1.float()),torch.mean(z_1.float())]
    center_2 = [torch.mean(x_2.float()),torch.mean(y_2.float()),torch.mean(z_2.float())]
    dis = torch.abs(torch.Tensor([center_1[0]-center_2[0],center_1[1]-center_2[1],center_1[2]-center_2[2]]))
    dis[2] = dis[2] * 2.5
    return torch.norm(dis)

def cc_loss(x,y,RTS):
    x = x * RTS
    y = y * RTS
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    top = torch.mean((x - x_mean) * (y - y_mean)) + 1
    bot = torch.std(x) * torch.std(y) + 1
    cc = top/bot
    return (1 - cc)

def cc_loss_noRTS(x,y,RTS):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    top = torch.mean((x - x_mean) * (y - y_mean))
    bot = torch.std(x) * torch.std(y)
    cc = top/bot
    return (1 - cc)

def Negcc_loss_ReallynoRTS(x,y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    top = torch.mean((x - x_mean) * (y - y_mean))
    bot = torch.std(x) * torch.std(y)
    cc = top/bot
    return -cc

def cc_test(x,y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    top = torch.mean((x - x_mean) * (y - y_mean))
    bot = torch.std(x) * torch.std(y)
    cc = top/bot
    return cc

def ncc_loss(I, J, win=None):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    conv_fn = getattr(F, 'conv%dd' % ndims)
    I2 = I*I
    J2 = J*J
    IJ = I*J

    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0]/2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1,1)
        padding = (pad_no, pad_no)
    else:
        stride = (1,1,1)
        padding = (pad_no, pad_no, pad_no)
    
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    # cc = cross*cross / (I_var*J_var + 1e-5)
    cc = (cross + 1e-5)**2 / ((I_var + 1e-5)*(J_var + 1e-5))

    return -torch.mean(cc)



def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross