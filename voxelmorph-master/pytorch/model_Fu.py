# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 16:01:26 2020

@author: Haonan Xiao

E-mail: hx42@duke.edu
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal

class G_fine_new(nn.Module):
    def __init__(self):
        super(G_fine,self).__init__()
        self.conv11 = conv_block_bn(dim=3, in_channels=2, out_channels=24, ksize=3, stride=1, padding=1)
        self.conv12 = conv_block_bn(dim=3, in_channels=24, out_channels=48, ksize=5, stride=2, padding=2)
        self.conv21 = conv_block_bn(dim=3, in_channels=48, out_channels=48, ksize=3, stride=1, padding=1)
        self.conv22 = conv_block_bn(dim=3, in_channels=48, out_channels=24, ksize=3, stride=1, padding=1)
        self.conv23 = conv_block_bn(dim=3, in_channels=24, out_channels=3, ksize=3, stride=1, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear')
    def forward(self,src,tgt):
        x = torch.cat([src, tgt], dim=1)
        # 2x96x96x96
        out = self.conv11(x, True)
        # 24x96x96x96
        out = self.conv12(out, True)
        # 48x48x48x48
        out = self.conv21(out, True)
        # 48x48x48x48
        out = self.conv21(out, True)
        # 48x48x48x48
        out = self.conv22(out, True)
        # 24x48x48x48
        out = self.conv23(out, False)
        # 3x48x48x48
        out_up = self.up(out)
        # 3x96x96x96
        return out_up

class G_fine(nn.Module):
    def __init__(self):
        super(G_fine,self).__init__()
        self.conv1 = conv_block_bn(dim=3, in_channels=2, out_channels=24, ksize=3, stride=1, padding=0)
        self.conv2 = conv_block_bn(dim=3, in_channels=24, out_channels=48, ksize=5, stride=1, padding=0)
        self.conv3 = conv_block_bn(dim=3, in_channels=48, out_channels=48, ksize=5, stride=1, padding=0)
        self.conv4 = conv_block_bn(dim=3, in_channels=48, out_channels=24, ksize=3, stride=1, padding=0)
        self.conv5 = conv_block_bn(dim=3, in_channels=24, out_channels=3, ksize=3, stride=1, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear')
    def forward(self,src,tgt):
        x = torch.cat([src, tgt], dim=1)
        # 2x32x32x32
        out = self.conv1(x, True)
        # 24x30x30x30
        out = self.conv2(out, True)
        # 48x26x26x26
        out = self.conv3(out, True)
        # 48x22x22x22
        out = self.conv3(out, True)
        # 48x18x18x18
        out = self.conv4(out, True)
        # 24x16x16x16
        out = self.conv5(out, False)
        # 3x16x16x16
        out_up = self.up(out)
        # 3x32x32x32
        return out, out_up

class D_Unet(nn.Module):
    def __init__(self):
        super(D_Unet,self).__init__()
        self.input = conv_block_bn(dim=3, in_channels=1, out_channels=32,ksize=3, stride=1, padding=1)
        self.conv11 = conv_block_bn(dim=3, in_channels=32, out_channels=32,ksize=3, stride=1, padding=1)
        self.conv12 = conv_block_bn(dim=3, in_channels=32, out_channels=32,ksize=3, stride=2, padding=1)
        self.conv21 = conv_block_bn(dim=3, in_channels=32, out_channels=48,ksize=3, stride=1, padding=1)
        self.conv22 = conv_block_bn(dim=3, in_channels=48, out_channels=48,ksize=3, stride=2, padding=1)
        self.conv31 = conv_block_bn(dim=3, in_channels=48, out_channels=64,ksize=3, stride=1, padding=1)
        self.conv32 = conv_block_bn(dim=3, in_channels=64, out_channels=64,ksize=3, stride=2, padding=1)   
        self.conv41 = conv_block_bn(dim=3, in_channels=64, out_channels=96,ksize=3, stride=1, padding=1)
        self.conv42 = conv_block_bn(dim=3, in_channels=96, out_channels=96,ksize=3, stride=2, padding=1)   
        self.conv51 = conv_block_bn(dim=3, in_channels=96, out_channels=96,ksize=3, stride=1, padding=1)
        self.conv52 = conv_block_bn(dim=3, in_channels=96, out_channels=96,ksize=3, stride=2, padding=1)    
        self.conv53 = conv_block_bn(dim=3, in_channels=96, out_channels=32,ksize=1, stride=1, padding=0)    
        self.full = nn.Linear(6*6*2*32, 1)
        self.drop = nn.Dropout(0.5)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        out = self.input(x, True)
        # 128x128x64x32
        # 192x192x64
        out = self.conv11(out, True)
        out = self.conv12(out, True)
        # 64x64x32x32
        # 96x96x32
        out = self.conv21(out, True)
        out = self.conv22(out, True)
        # 32x32x16x32
        # 48x48x16
        out = self.conv31(out, True)
        out = self.conv32(out, True)
        # 16x16x8x32
        # 24x24x8
        out = self.conv41(out, True)
        out = self.conv42(out, True)
        # 8x8x4x32
        # 12x12x4
        out = self.conv51(out, True)
        out = self.conv52(out, True)
        out = self.conv53(out, True)
        # 4x4x2x32
        # 6x6x2
        out = out.view(out.size(0), -1)
        out = self.drop(out)
        out = self.full(out)
        out = self.sig(out)
        return out
    
class D_fine(nn.Module):
    def __init__(self):
        super(D_fine,self).__init__()
        self.conv1 = conv_block_bn(dim=3, in_channels=1, out_channels=16, ksize=5, stride=1, padding=0)
        self.conv2 = conv_block_bn(dim=3, in_channels=16, out_channels=32, ksize=3, stride=1, padding=0)
        self.conv3 = conv_block_bn(dim=3, in_channels=32, out_channels=64, ksize=2, stride=1, padding=0)
        self.conv4 = conv_block_bn(dim=3, in_channels=64, out_channels=1, ksize=1, stride=1, padding=0)
        self.maxp = nn.MaxPool3d(2)
        self.sig = nn.Sigmoid()        
    def forward(self,x):
        # x 1x1x16x16x16
        out = self.conv1(x, True)
        # 16x12x12x12
        out = self.maxp(out)
        # 16x6x6x6
        out = self.conv2(out, True)
        # 32x4x4x4
        out = self.maxp(out)
        # 32x2x2x2
        out = self.conv3(out, False)
        # 64x1x1x1
        out = self.conv4(out, False)
        # 1x1x1x1
        out = self.sig(out)
        return out
    
class G_coarse(nn.Module):
    """
    input 64x64x64
    output 8x8x8
    """
    def __init__(self):
        super(G_coarse, self).__init__()
        self.input_1 = conv_block_bn(dim=3, in_channels=2, out_channels=16, ksize=3, stride=1, padding=2, dilation=2)
        self.input_2 = conv_block_bn(dim=3, in_channels=2, out_channels=16, ksize=3, stride=1, padding=4, dilation=4)
        self.input_3 = conv_block_bn(dim=3, in_channels=2, out_channels=16, ksize=3, stride=1, padding=6, dilation=6)
        self.input_4 = conv_block_bn(dim=3, in_channels=2, out_channels=16, ksize=3, stride=1, padding=1, dilation=1)
        self.conv11 = conv_block_bn(dim=3, in_channels=64, out_channels=48, ksize=3, stride=1, padding=1, dilation=1)
        self.conv12 = conv_block_bn(dim=3, in_channels=48, out_channels=48, ksize=3, stride=1, padding=1, dilation=1)
        self.atten1 = Attention_block(48,96,48)
        self.atten2 = Attention_block(96,192,96)
        self.conv21 = conv_block_bn(dim=3, in_channels=48, out_channels=96, ksize=3, stride=1, padding=1, dilation=1)
        self.conv22 = conv_block_bn(dim=3, in_channels=96, out_channels=96, ksize=3, stride=1, padding=1, dilation=1)
        self.conv23 = conv_block_bn(dim=3, in_channels=192, out_channels=96, ksize=3, stride=1, padding=1, dilation=1)
        self.conv31 = conv_block_bn(dim=3, in_channels=96, out_channels=192, ksize=3, stride=1, padding=1, dilation=1)
        self.conv32 = conv_block_bn(dim=3, in_channels=192, out_channels=192, ksize=3, stride=1, padding=1, dilation=1)
        self.conv33 = conv_block_bn(dim=3, in_channels=384, out_channels=192, ksize=3, stride=1, padding=1, dilation=1)
        self.conv41 = conv_block_bn(dim=3, in_channels=192, out_channels=256, ksize=3, stride=1, padding=1, dilation=1)
        self.conv42 = conv_block_bn(dim=3, in_channels=256, out_channels=3, ksize=3, stride=1, padding=1, dilation=1)
        self.maxp = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=8, mode='trilinear')
    def forward(self,src, tgt):
        x = torch.cat([src, tgt], dim=1)
        # 1x2x64x64x64
        dia_1 = self.input_1(x, True)
        dia_2 = self.input_2(x, True)
        dia_3 = self.input_3(x, True)
        dia_4 = self.input_4(x, True)
#        dia_4 = self.input_4(x)
        x = torch.cat([dia_1, dia_2, dia_3, dia_4], dim=1)
        # 1x(3*16)x64x64x64
        x = self.conv11(x, True)
        # 1x48x64x64x64
        x = self.conv12(x, True)
        # 1x48x64x64x64, g
        g = x
        x = self.maxp(x)
        # 1x48x32x32x32
        x = self.conv21(x, True)
        # 1x96x32x32x32
        alpha = self.atten1(g,x)
        # 1x96x32x32x32
        x = self.conv22(x, True)
        # 1x96x32x32x32
        x = torch.cat([alpha,x], dim=1)
        # 1x(96x2)x32x32x32
        x = self.conv23(x, True)
        # 1x96x32x32x32
        g = x
        x = self.maxp(x)
        # 1x96x16x16x16
        x = self.conv31(x, True)
        # 1x192x16x16x16
        alpha = self.atten2(g,x)
        # 1x192x16x16x16
        x = self.conv32(x, True)
        # 1x192x16x16x16
        x = torch.cat([alpha,x], dim=1)
        # 1x(192x2)x16x16x16
        x = self.conv33(x, True)
        # 1x192x16x16x16
        x = self.maxp(x)
        # 1x192x8x8x8
        x = self.conv41(x, True)
        # 1x256x8x8x8
        x = self.conv42(x, False)
        # 1x3x8x8x8
        x_up = self.upsample(x)
        # 1x3x64x64x64
        return x, x_up

class G_coarse_simple(nn.Module):
    """
    input 64x64x64
    output 8x8x8
    """
    def __init__(self):
        super(G_coarse_simple, self).__init__()
        self.input_1 = conv_block_bn(dim=3, in_channels=2, out_channels=16, ksize=3, stride=1, padding=2, dilation=2)
        self.input_2 = conv_block_bn(dim=3, in_channels=2, out_channels=16, ksize=3, stride=1, padding=4, dilation=4)
        self.input_3 = conv_block_bn(dim=3, in_channels=2, out_channels=16, ksize=3, stride=1, padding=6, dilation=6)
        self.input_4 = conv_block_bn(dim=3, in_channels=2, out_channels=16, ksize=3, stride=1, padding=1, dilation=1)
        self.conv11 = conv_block_bn(dim=3, in_channels=48, out_channels=32, ksize=3, stride=1, padding=1, dilation=1)
        self.conv12 = conv_block_bn(dim=3, in_channels=32, out_channels=32, ksize=3, stride=1, padding=1, dilation=1)
        self.atten1 = Attention_block(32,64,32)
        self.atten2 = Attention_block(96,192,96)
        self.conv21 = conv_block_bn(dim=3, in_channels=48, out_channels=48, ksize=3, stride=1, padding=1, dilation=1)
        self.conv22 = conv_block_bn(dim=3, in_channels=48, out_channels=48, ksize=3, stride=1, padding=1, dilation=1)
        self.conv23 = conv_block_bn(dim=3, in_channels=80, out_channels=64, ksize=3, stride=1, padding=1, dilation=1)
        self.conv31 = conv_block_bn(dim=3, in_channels=64, out_channels=64, ksize=3, stride=1, padding=1, dilation=1)
        self.conv32 = conv_block_bn(dim=3, in_channels=64, out_channels=64, ksize=3, stride=1, padding=1, dilation=1)
        self.conv33 = conv_block_bn(dim=3, in_channels=128, out_channels=96, ksize=3, stride=1, padding=1, dilation=1)
        self.conv41 = conv_block_bn(dim=3, in_channels=96, out_channels=128, ksize=3, stride=1, padding=1, dilation=1)
        self.conv42 = conv_block_bn(dim=3, in_channels=128, out_channels=3, ksize=3, stride=1, padding=1, dilation=1)
        self.maxp = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=8, mode='trilinear')
    def forward(self,src, tgt):
        x = torch.cat([src, tgt], dim=1)
        # 1x2x256x256x96
        dia_1 = self.input_1(x, True)
        dia_2 = self.input_2(x, True)
        dia_3 = self.input_3(x, True)
#        dia_4 = self.input_4(x)
        x = torch.cat([dia_1, dia_2, dia_3], dim=1)
        # 1x(3*16)x256x256x96
        x = self.conv11(x, True)
        # 1x32x256x256x96
        x = self.conv12(x, True)
        # 1x32x256x256x96
        g = x
        x = self.maxp(x)
        # 1x32x128x128x48
        x = self.conv21(x, True)
        # 1x48x128x128x48
        alpha = self.atten1(g,x)
        # 1x32x128x128x48
        x = self.conv22(x, True)
        # 1x48x128x128x48
        x = torch.cat([alpha,x], dim=1)
        # 1x(48+32)x128x128x48
        x = self.conv23(x, True)
        # 1x64x128x128x48
        g = x
        x = self.maxp(x)
        # 1x64x64x64x24
        x = self.conv31(x, True)
        # 1x64x64x64x24
        alpha = self.atten2(g,x)
        # 1x64x64x64x24
        x = self.conv32(x, True)
        # 1x64x64x64x24
        x = torch.cat([alpha,x], dim=1)
        # 1x(64x2)x64x64x24
        x = self.conv33(x, True)
        # 1x96x64x64x24
        x = self.maxp(x)
        # 1x96x32x32x12
        x = self.conv41(x, True)
        # 1x128x32x32x12
        x = self.conv42(x, False)
        # 1x3x32x32x12
        x_up = self.upsample(x)
        # 1x3x256x256x96
        return x_up
    
class D_coarse_new(nn.Module):
    def __init__(self):
        super(D_coarse_new,self).__init__()
        self.conv11 = conv_block_bn(dim=3, in_channels=1, out_channels=16, ksize=3, stride=1, padding=1)
        self.conv12 = conv_block_bn(dim=3, in_channels=16, out_channels=16, ksize=5, stride=2, padding=2)
        self.conv21 = conv_block_bn(dim=3, in_channels=16, out_channels=32, ksize=3, stride=1, padding=1)
        self.conv22 = conv_block_bn(dim=3, in_channels=32, out_channels=32, ksize=5, stride=2, padding=2)
        self.conv31 = conv_block_bn(dim=3, in_channels=32, out_channels=64, ksize=3, stride=1, padding=1)
        self.conv32 = conv_block_bn(dim=3, in_channels=64, out_channels=64, ksize=5, stride=2, padding=2)
        self.conv41 = conv_block_bn(dim=3, in_channels=64, out_channels=128, ksize=3, stride=1, padding=1)
        self.conv42 = conv_block_bn(dim=3, in_channels=128, out_channels=128, ksize=5, stride=2, padding=2)
        self.conv5 = conv_block_bn(dim=3, in_channels=128, out_channels=1, ksize=6, stride=1, padding=0)
        self.maxp = nn.MaxPool3d(2)
        self.sig = nn.Sigmoid()
        
    def forward(self,x):
        # x 96x96x96
        out = self.conv11(x, True)
        out = self.conv12(out, True)
        # 16x48x48x48
        out = self.conv21(out, True)
        out = self.conv22(out, True)
        # 32x24x24x24
        out = self.conv31(out, True)
        out = self.conv32(out, True)
        # 64x12x12x12
        out = self.conv41(out, True)
        out = self.conv42(out, True)
        # 128x6x6x6
        out = self.conv5(out, False)
        # 1x1x1x1
        out = self.sig(out)
        return out

class D_coarse(nn.Module):
    def __init__(self):
        super(D_coarse,self).__init__()
        self.conv1 = conv_block_bn(dim=3, in_channels=1, out_channels=16, ksize=5, stride=1, padding=0)
        self.conv2 = conv_block_bn(dim=3, in_channels=16, out_channels=32, ksize=3, stride=1, padding=0)
        self.conv3 = conv_block_bn(dim=3, in_channels=32, out_channels=64, ksize=3, stride=1, padding=0)
        self.conv4 = conv_block_bn(dim=3, in_channels=64, out_channels=128, ksize=2, stride=1, padding=0)
        self.conv5 = conv_block_bn(dim=3, in_channels=128, out_channels=1, ksize=1, stride=1, padding=0)
        self.maxp = nn.MaxPool3d(2)
        self.sig = nn.Sigmoid()
        
    def forward(self,x):
        # x 32x32x32
        out = self.conv1(x, True)
        # 16x28x28x28
        out = self.maxp(out)
        # 16x14x14x14
        out = self.conv2(out, True)
        # 32x12x12x12
        out = self.maxp(out)
        # 32x6x6x6
        out = self.conv3(out, True)
        # 64x4x4x4
        out = self.maxp(out)
        # 64x2x2x2
        out = self.conv4(out, False)
        # 128x1x1x1
        out = self.conv5(out, False)
        out = self.sig(out)
        return out

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.downsample = nn.MaxPool3d(2)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        g1 = self.downsample(g1)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.sigmoid(self.psi(psi))
        return x*psi
    
class conv_block_bn(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """
    def __init__(self, dim, in_channels, out_channels, ksize, stride=1, padding=1, dilation=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block_bn, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        self.main = conv_fn(in_channels, out_channels, ksize, stride, padding, dilation)
        self.activation = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x, withbn):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        if withbn:
            out = self.bn(out)
        else:
            out = out
        return out
    
class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [ torch.arange(0, s) for s in size ] 
        grids = torch.meshgrid(vectors) 
        grid  = torch.stack(grids) # y, x, z
        grid  = torch.unsqueeze(grid, 0)  #add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):   
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow 

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1) 
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1) 
            new_locs = new_locs[..., [2,1,0]]

        return nnf.grid_sample(src, new_locs, mode=self.mode)
    