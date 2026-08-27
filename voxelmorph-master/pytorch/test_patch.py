# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:43:14 2020

@author: Haonan Xiao

E-mail: hx42@duke.edu
"""

import os
import numpy as np
import torch
# internal imports
from model import SpatialTransformer
from math import floor
import scipy.io as sio
import time


def test_DIRLab(model, cwd, datafolder, save_folder):
    stride = 48
    device = "cuda"
    studies = os.listdir(os.path.join(cwd, datafolder))
    patch_size = (64,64,64)
    for i in range(len(studies)):
         files = os.listdir(os.path.join(cwd, datafolder, studies[i]))
         moving_vol = sio.loadmat(os.path.join(cwd, datafolder, studies[i], files[0]))[str(files[0][0:-4])]-1000
         image_size = np.shape(moving_vol)
         moving_patch = image2patch(moving_vol, patch_size, stride)
         num_patch = moving_patch.shape[0]
         for j in range(len(files)):
             if j == 0:
                 continue
             fixed_vol = sio.loadmat(os.path.join(cwd, datafolder, studies[i], files[j]))[str(files[j][0:-4])]-1000
             fixed_patch = image2patch(fixed_vol, patch_size, stride)
             DVF_patch = np.zeros((num_patch, 64, 64, 64, 3))
             WARP_patch = np.zeros((num_patch, 64, 64, 64))
             for k in range(num_patch):
                 with torch.no_grad():
                     input_moving = moving_patch[k]
                     input_moving = input_moving[np.newaxis, np.newaxis, ...]
                     input_moving = torch.from_numpy(input_moving).to(device).float()
                     input_fixed = fixed_patch[k]
                     input_fixed = input_fixed[np.newaxis, np.newaxis, ...]
                     input_fixed = torch.from_numpy(input_fixed).to(device).float()
                     WARP_out, DVF_out = model(input_moving, input_fixed)
                     WARP_out = torch.squeeze(WARP_out)
                     WARP_patch[k] = WARP_out.cpu().numpy()
                     DVF_out= torch.squeeze(DVF_out)
                     DVF_out = DVF_out.permute(1,2,3,0)
                     DVF_patch[k] = DVF_out.cpu().numpy()
             WARP = patch2image(WARP_patch, image_size, stride)
             DVF = patch2DVF(DVF_patch, image_size, stride)
             WARP_name = files[0][0:-4] + '_to_' + files[j][0:-4] + '_volume' + '.mat'
             DVF_name = files[0][0:-4] + '_to_' + files[j][0:-4] + '_DVF' + '.mat'
             if not os.path.exists(os.path.join(cwd, save_folder, studies[i])):
                 os.mkdir(os.path.join(cwd, save_folder, studies[i]))
             sio.savemat(os.path.join(cwd, save_folder, studies[i], WARP_name), {WARP_name[0:-4]:WARP})
             sio.savemat(os.path.join(cwd, save_folder, studies[i], DVF_name), {DVF_name[0:-4]:DVF})

def test(model, cwd, datafolder, save_folder):
    stride = 48
    device = "cuda"
    studies = os.listdir(os.path.join(cwd, datafolder))
    patch_size = (64,64,64)
    for i in range(len(studies)):
         files = os.listdir(os.path.join(cwd, datafolder, studies[i]))
         moving_vol = np.load(os.path.join(cwd, datafolder, studies[i], 'volume_1..npy'))
         moving_vol = np.transpose(moving_vol, (2,1,0))
         image_size = np.shape(moving_vol)
         moving_patch = image2patch(moving_vol, patch_size, stride)
         num_patch = moving_patch.shape[0]
         for j in range(len(files)):
             if files[j][0] == 'R':
                 continue
             if files[j] == 'volume_1..npy':
                 continue
             fixed_vol = np.load(os.path.join(cwd, datafolder, studies[i], files[j]))
             fixed_vol = np.transpose(fixed_vol, (2,1,0))
             fixed_patch = image2patch(fixed_vol, patch_size, stride)
             DVF_patch = np.zeros((num_patch, 64, 64, 64, 3))
             WARP_patch = np.zeros((num_patch, 64, 64, 64))
             for k in range(num_patch):
                 with torch.no_grad():
                     input_moving = moving_patch[k]
                     input_moving = input_moving[np.newaxis, np.newaxis, ...]
                     input_moving = torch.from_numpy(input_moving).to(device).float()
                     input_fixed = fixed_patch[k]
                     input_fixed = input_fixed[np.newaxis, np.newaxis, ...]
                     input_fixed = torch.from_numpy(input_fixed).to(device).float()
                     WARP_out, DVF_out = model(input_moving, input_fixed)
                     WARP_out = torch.squeeze(WARP_out)
                     WARP_patch[k] = WARP_out.cpu().numpy()
                     DVF_out= torch.squeeze(DVF_out)
                     DVF_out = DVF_out.permute(1,2,3,0)
                     DVF_patch[k] = DVF_out.cpu().numpy()
             WARP = patch2image(WARP_patch, image_size, stride)
             DVF = patch2DVF(DVF_patch, image_size, stride)
             WARP_name = 'volume_1 to' + files[j][0:-4] + '_volume'
             DVF_name = 'volume_1 to' + files[j][0:-4] + '_DVF'
             if not os.path.exists(os.path.join(cwd, save_folder, studies[i])):
                 os.mkdir(os.path.join(cwd, save_folder, studies[i]))
             np.save(os.path.join(cwd, save_folder, studies[i], WARP_name), WARP)
             np.save(os.path.join(cwd, save_folder, studies[i], DVF_name), DVF)
                 

def image2patch(image,patch_size,stride):
    """
    image:需要切分为图像块的图像
    patch_size:图像块的尺寸，如:(10,10)
    stride:切分图像块时移动过得步长，如:5
    """
    imhigh,imwidth,imdepth = image.shape
    ## 构建图像块的索引
    range_z = np.arange(0, imdepth - patch_size[2],stride)
    range_y = np.arange(0,imhigh - patch_size[0],stride)
    range_x = np.arange(0,imwidth - patch_size[1],stride)
    
    if range_y[-1] != imhigh - patch_size[0]:
        range_y = np.append(range_y,imhigh - patch_size[0])
    if range_x[-1] != imwidth - patch_size[1]:
        range_x = np.append(range_x,imwidth - patch_size[1])
    if range_z[-1] != imdepth - patch_size[2]:
        range_z = np.append(range_z, imdepth - patch_size[2])
    sz = len(range_z) * len(range_y) * len(range_x)  ## 图像块的数量
    res = np.zeros((sz,patch_size[0],patch_size[1],patch_size[2]))
    index = 0
    for y in range_y:
        for x in range_x:
            for z in range_z:
                patch = image[y:y+patch_size[0],x:x+patch_size[1],z:z+patch_size[2]]
                res[index] = patch
                index = index + 1   
    return res  
    
## 定义函数  图像转化为图像块的逆变换
def patch2image(coldata,imsize,stride):
    """
    coldata: 使用image2cols得到的数据
    imsize:原始图像的宽和高，如(321, 481)
    stride:图像切分时的步长，如10
    """
    patch_size = coldata.shape[1:4]
    res = np.zeros((imsize[0],imsize[1],imsize[2]))
    w = np.zeros(((imsize[0],imsize[1],imsize[2])))
    range_z = np.arange(0,imsize[2] - patch_size[2],stride)
    range_y = np.arange(0,imsize[0] - patch_size[0],stride)
    range_x = np.arange(0,imsize[1] - patch_size[1],stride)
    if range_y[-1] != imsize[0] - patch_size[0]:
        range_y = np.append(range_y,imsize[0] - patch_size[0])
    if range_x[-1] != imsize[1] - patch_size[1]:
        range_x = np.append(range_x,imsize[1] - patch_size[1])
    if range_z[-1] != imsize[2] - patch_size[2]:
        range_z = np.append(range_z,imsize[2] - patch_size[2])
    index = 0
    for y in range_y:
        for x in range_x:
            for z in range_z:
                res[y:y+patch_size[0],x:x+patch_size[1],z:z+patch_size[2]] = res[y:y+patch_size[0],x:x+patch_size[1],z:z+patch_size[2]] + coldata[index]
                w[y:y+patch_size[0],x:x+patch_size[1],z:z+patch_size[2]] = w[y:y+patch_size[0],x:x+patch_size[1],z:z+patch_size[2]] + 1
                index = index + 1
    return res / w

def patch2DVF(coldata,imsize,stride):
    """
    coldata: 使用image2cols得到的数据
    imsize:原始图像的宽和高，如(321, 481)
    stride:图像切分时的步长，如10
    """
    patch_size = coldata.shape[1:4]
    res = np.zeros((imsize[0],imsize[1],imsize[2],3))
    w = np.zeros((imsize[0],imsize[1],imsize[2],3))
    range_z = np.arange(0,imsize[2] - patch_size[2],stride)
    range_y = np.arange(0,imsize[0] - patch_size[0],stride)
    range_x = np.arange(0,imsize[1] - patch_size[1],stride)
    if range_y[-1] != imsize[0] - patch_size[0]:
        range_y = np.append(range_y,imsize[0] - patch_size[0])
    if range_x[-1] != imsize[1] - patch_size[1]:
        range_x = np.append(range_x,imsize[1] - patch_size[1])
    if range_z[-1] != imsize[2] - patch_size[2]:
        range_z = np.append(range_z,imsize[2] - patch_size[2])
    index = 0
    for y in range_y:
        for x in range_x:
            for z in range_z:
                res[y:y+patch_size[0],x:x+patch_size[1],z:z+patch_size[2]] = res[y:y+patch_size[0],x:x+patch_size[1],z:z+patch_size[2]] + coldata[index]
                w[y:y+patch_size[0],x:x+patch_size[1],z:z+patch_size[2]] = w[y:y+patch_size[0],x:x+patch_size[1],z:z+patch_size[2]] + 1
                index = index + 1
    return res / w
    
    
             