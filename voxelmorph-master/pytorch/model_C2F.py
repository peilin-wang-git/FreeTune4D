# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:16:27 2020

@author: Haonan Xiao

E-mail: hx42@duke.edu
"""
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal

class Unet_GF_twice(nn.Module):
    def __init__(self, dim):
        super(Unet_GF_twice,self).__init__()
        self.max = nn.MaxPool3d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.input = conv_block_bn(dim=3, in_channels=3, out_channels=64, ksize=3, stride=1, padding=1, dilation = 1)
        self.conv_layer111 = conv_block_bn(dim=3, in_channels=64, out_channels=64, ksize=3, stride=1, padding=1)
        self.conv_layer112 = conv_block_bn(dim=3, in_channels=64, out_channels=64, ksize=3, stride=1, padding=1)
        self.conv_layer12 = conv_block_bn(dim=3, in_channels=64, out_channels=64, ksize=3, stride=1, padding=1)
        self.conv_layer131 = conv_block_bn(dim=3, in_channels=128+64, out_channels=64, ksize=3, stride=1, padding=1)
        self.conv_layer132 = conv_block_bn(dim=3, in_channels=64, out_channels=64, ksize=3, stride=1, padding=1)
        
        self.conv_layer211 = conv_block_bn(dim=3, in_channels=64, out_channels=128, ksize=3, stride=1, padding=1)
        self.conv_layer212 = conv_block_bn(dim=3, in_channels=128, out_channels=128, ksize=3, stride=1, padding=1)
        self.conv_layer22 = conv_block_bn(dim=3, in_channels=128, out_channels=128, ksize=3, stride=1, padding=1)
        self.conv_layer231 = conv_block_bn(dim=3, in_channels=128+256, out_channels=128, ksize=3, stride=1, padding=1)
        self.conv_layer232 = conv_block_bn(dim=3, in_channels=128, out_channels=128, ksize=3, stride=1, padding=1)
        
        self.conv_layer31 = conv_block_bn(dim=3, in_channels=128, out_channels=128, ksize=3, stride=1, padding=1)
        self.conv_layer32 = conv_block_bn(dim=3, in_channels=128, out_channels=256, ksize=3, stride=1, padding=1)
        
        self.flow = nn.Conv3d(64, dim, kernel_size=3, padding=1)
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
    def forward(self, src, tgt, diff):
        x = torch.cat([src, tgt, diff], dim=1)
        x = self.input(x, False)
        # x_layer1 = torch.cat([x1,x2,x3], dim=1)
        # out size: 192x(64x64x64)
        x_layer1 = self.conv_layer111(x, False)
        x_layer1_pool = self.conv_layer112(x_layer1, False)
        x_layer1 = self.conv_layer12(x_layer1_pool, False) #1
        x_layer1 = self.conv_layer12(x_layer1, False) #2
        x_layer1 = self.conv_layer12(x_layer1, False) #3
        x_layer1 = self.conv_layer12(x_layer1, False) #4
        x_layer1 = self.conv_layer12(x_layer1, False) #5 
        x_layer1 = self.conv_layer12(x_layer1, False) #6
        # out size: 64x(64x64x64)
        x_layer2 = self.max(x_layer1_pool)
        x_layer2 = self.conv_layer211(x_layer2, False)
        x_layer2_pool = self.conv_layer212(x_layer2, False)
        x_layer2 = self.conv_layer22(x_layer2_pool, False)
        # out size: 128x(32x32x32)
        x_layer3 = self.max(x_layer2_pool)
        x_layer3 = self.conv_layer31(x_layer3, False)
        x_layer3 = self.conv_layer32(x_layer3, False)
        # out size: 256x(16x16x16)
        x_layer3_up = self.up(x_layer3)
        x_layer2_cat = torch.cat([x_layer3_up, x_layer2], dim=1)
        x_layer2 = self.conv_layer231(x_layer2_cat, False)
        x_layer2 = self.conv_layer232(x_layer2, False)
        x_layer2_up = self.up(x_layer2)
        x_layer1_cat = torch.cat([x_layer2_up, x_layer1], dim=1)
        x_layer1 = self.conv_layer131(x_layer1_cat, False)
        x_layer1 = self.conv_layer132(x_layer1, False)
        dvf = self.flow(x_layer1)
        return dvf

class Unet_GF(nn.Module):
    def __init__(self, dim):
        super(Unet_GF,self).__init__()
        self.max = nn.MaxPool3d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.input = conv_block_bn(dim=3, in_channels=3, out_channels=32, ksize=3, stride=1, padding=1, dilation = 1)
        self.conv_layer111 = conv_block_bn(dim=3, in_channels=32, out_channels=32, ksize=3, stride=1, padding=1)
        self.conv_layer112 = conv_block_bn(dim=3, in_channels=32, out_channels=32, ksize=3, stride=1, padding=1)
        self.conv_layer12 = conv_block_bn(dim=3, in_channels=32, out_channels=32, ksize=3, stride=1, padding=1)
        self.conv_layer131 = conv_block_bn(dim=3, in_channels=64 + 32, out_channels=32, ksize=3, stride=1, padding=1)
        self.conv_layer132 = conv_block_bn(dim=3, in_channels=32, out_channels=32, ksize=3, stride=1, padding=1)
        
        self.conv_layer211 = conv_block_bn(dim=3, in_channels=32, out_channels=64, ksize=3, stride=1, padding=1)
        self.conv_layer212 = conv_block_bn(dim=3, in_channels=64, out_channels=64, ksize=3, stride=1, padding=1)
        self.conv_layer22 = conv_block_bn(dim=3, in_channels=64, out_channels=64, ksize=3, stride=1, padding=1)
        self.conv_layer231 = conv_block_bn(dim=3, in_channels=64 + 128, out_channels=64, ksize=3, stride=1, padding=1)
        self.conv_layer232 = conv_block_bn(dim=3, in_channels=64, out_channels=64, ksize=3, stride=1, padding=1)
        
        self.conv_layer31 = conv_block_bn(dim=3, in_channels=64, out_channels=64, ksize=3, stride=1, padding=1)
        self.conv_layer32 = conv_block_bn(dim=3, in_channels=64, out_channels=128, ksize=3, stride=1, padding=1)
        
        self.flow = nn.Conv3d(32, dim, kernel_size=3, padding=1)      
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
    def forward(self, src, tgt, diff):
        x = torch.cat([src, tgt, diff], dim=1)
        x = self.input(x, False)
        # x_layer1 = torch.cat([x1,x2,x3], dim=1)
        # out size: 192x(64x64x64)
        x_layer1 = self.conv_layer111(x, False)
        x_layer1_pool = self.conv_layer112(x_layer1, False)
        x_layer1 = self.conv_layer12(x_layer1_pool, False) #1
        x_layer1 = self.conv_layer12(x_layer1, False) #2
        x_layer1 = self.conv_layer12(x_layer1, False) #3
        x_layer1 = self.conv_layer12(x_layer1, False) #4
        x_layer1 = self.conv_layer12(x_layer1, False) #5 
        x_layer1 = self.conv_layer12(x_layer1, False) #6
        # out size: 64x(64x64x64)
        x_layer2 = self.max(x_layer1_pool)
        x_layer2 = self.conv_layer211(x_layer2, False)
        x_layer2_pool = self.conv_layer212(x_layer2, False)
        x_layer2 = self.conv_layer22(x_layer2_pool, False)
        # out size: 128x(32x32x32)
        x_layer3 = self.max(x_layer2_pool)
        x_layer3 = self.conv_layer31(x_layer3, False)
        x_layer3 = self.conv_layer32(x_layer3, False)
        # out size: 256x(16x16x16)
        x_layer3_up = self.up(x_layer3)
        x_layer2_cat = torch.cat([x_layer3_up, x_layer2], dim=1)
        x_layer2 = self.conv_layer231(x_layer2_cat, False)
        x_layer2 = self.conv_layer232(x_layer2, False)
        x_layer2_up = self.up(x_layer2)
        x_layer1_cat = torch.cat([x_layer2_up, x_layer1], dim=1)
        x_layer1 = self.conv_layer131(x_layer1_cat, False)
        x_layer1 = self.conv_layer132(x_layer1, False)
        dvf = self.flow(x_layer1)
        return dvf

class JiangNet(nn.Module):
    def __init__(self, dim, scale, residual):
        super(JiangNet,self).__init__()
        self.down = nn.MaxPool3d(scale)
        self.up = nn.Upsample(scale_factor=scale, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=4, mode='nearest')
        if residual:
            self.conv1 = conv_block_bn(dim=3, in_channels=2, out_channels=16, ksize=3, stride=1, padding=1)
            self.conv2 = conv_block_bn(dim=3, in_channels=16, out_channels=32, ksize=5, stride=2, padding=2)
            self.conv3 = conv_block_bn(dim=3, in_channels=32, out_channels=32, ksize=3, stride=1, padding=1)
            self.conv4 = conv_block_bn(dim=3, in_channels=32, out_channels=48, ksize=5, stride=2, padding=2)
            self.conv5 = conv_block_bn(dim=3, in_channels=48, out_channels=48, ksize=3, stride=1, padding=1)
            self.conv6 = conv_block_bn(dim=3, in_channels=48, out_channels=48, ksize=3, stride=2, padding=1)
            self.conv7 = conv_block_bn(dim=3, in_channels=96, out_channels=32, ksize=3, stride=1, padding=1)
            self.conv8 = conv_block_bn(dim=3, in_channels=32, out_channels=16, ksize=3, stride=1, padding=1)
            self.conv9 = nn.Conv3d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = conv_block_bn(dim=3, in_channels=2, out_channels=8, ksize=3, stride=1, padding=1)
            self.conv2 = conv_block_bn(dim=3, in_channels=8, out_channels=24, ksize=5, stride=2, padding=2)
            self.conv3 = conv_block_bn(dim=3, in_channels=24, out_channels=24, ksize=3, stride=1, padding=1)
            self.conv4 = conv_block_bn(dim=3, in_channels=24, out_channels=24, ksize=5, stride=2, padding=2)
            self.conv5 = conv_block_bn(dim=3, in_channels=24, out_channels=24, ksize=3, stride=1, padding=1)
            self.conv6 = conv_block_bn(dim=3, in_channels=24, out_channels=24, ksize=3, stride=2, padding=1)
            self.conv7 = conv_block_bn(dim=3, in_channels=56, out_channels=8, ksize=3, stride=1, padding=1)
            self.conv8 = conv_block_bn(dim=3, in_channels=8, out_channels=8, ksize=3, stride=1, padding=1)
            self.conv9 = nn.Conv3d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1)
    def forward(self,src,tgt):
        x = torch.cat([src, tgt], dim=1)
        x = self.down(x)
        x1 = self.conv1(x,False)
        x2 = self.conv2(x1,False)
        x3 = self.conv3(x2,False)
        x4 = self.conv4(x3,False)
        x5 = self.conv5(x4,False)
        x6 = self.conv6(x5,False)
        x6_up = self.up4(x6)
        x6_upup = self.up2(torch.cat([x2, x6_up], dim=1))
        x7 = self.conv7(torch.cat([x1, x6_upup], dim=1),False)
        x8 = self.conv8(x7,False)
        flow = self.conv9(x8)
        flow_up = self.up(flow)
        return flow, flow_up

class JiangNet_Fine(nn.Module):
    def __init__(self, dim, scale, residual):
        super(JiangNet_Fine,self).__init__()
        self.down = nn.MaxPool3d(scale)
        self.up = nn.Upsample(scale_factor=scale, mode='trilinear')
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=4, mode='nearest')
        if residual:
            self.conv1 = conv_block_bn(dim=3, in_channels=3, out_channels=16, ksize=3, stride=1, padding=1)
            self.conv2 = conv_block_bn(dim=3, in_channels=16, out_channels=32, ksize=5, stride=2, padding=2)
            self.conv3 = conv_block_bn(dim=3, in_channels=32, out_channels=32, ksize=3, stride=1, padding=1)
            self.conv4 = conv_block_bn(dim=3, in_channels=32, out_channels=64, ksize=5, stride=2, padding=2)
            self.conv5 = conv_block_bn(dim=3, in_channels=64, out_channels=64, ksize=3, stride=1, padding=1)
            self.conv6 = conv_block_bn(dim=3, in_channels=64, out_channels=64, ksize=3, stride=2, padding=1)
            self.conv7 = conv_block_bn(dim=3, in_channels=112, out_channels=32, ksize=3, stride=1, padding=1)
            self.conv8 = conv_block_bn(dim=3, in_channels=32, out_channels=16, ksize=3, stride=1, padding=1)
            self.conv9 = nn.Conv3d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = conv_block_bn(dim=3, in_channels=3, out_channels=8, ksize=3, stride=1, padding=1)
            self.conv2 = conv_block_bn(dim=3, in_channels=8, out_channels=24, ksize=5, stride=2, padding=2)
            self.conv3 = conv_block_bn(dim=3, in_channels=24, out_channels=24, ksize=3, stride=1, padding=1)
            self.conv4 = conv_block_bn(dim=3, in_channels=24, out_channels=24, ksize=5, stride=2, padding=2)
            self.conv5 = conv_block_bn(dim=3, in_channels=24, out_channels=24, ksize=3, stride=1, padding=1)
            self.conv6 = conv_block_bn(dim=3, in_channels=24, out_channels=24, ksize=3, stride=2, padding=1)
            self.conv7 = conv_block_bn(dim=3, in_channels=56, out_channels=8, ksize=3, stride=1, padding=1)
            self.conv8 = conv_block_bn(dim=3, in_channels=8, out_channels=8, ksize=3, stride=1, padding=1)
            self.conv9 = nn.Conv3d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1)
    def forward(self,src,tgt, diff):
        x = torch.cat([src, tgt, diff], dim=1)
        x = self.down(x)
        x1 = self.conv1(x,False)
        x2 = self.conv2(x1,False)
        x3 = self.conv3(x2,False)
        x4 = self.conv4(x3,False)
        x5 = self.conv5(x4,False)
        x6 = self.conv6(x5,False)
        x6_up = self.up4(x6)
        x6_upup = self.up2(torch.cat([x2, x6_up], dim=1))
        x7 = self.conv7(torch.cat([x1, x6_upup], dim=1),False)
        x8 = self.conv8(x7,False)
        flow = self.conv9(x8)
        flow_up = self.up(flow)
        return flow_up

class unet_core(nn.Module):
    """
    [unet_core] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """
    def __init__(self, dim, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate UNet model
            :param dim: dimension of the image passed into the net
            :param enc_nf: the number of features maps in each layer of encoding stage
            :param dec_nf: the number of features maps in each layer of decoding stage
            :param full_size: boolean value representing whether full amount of decoding 
                            layers
        """
        super(unet_core, self).__init__()
        #(self, dim, in_channels, out_channels, stride=1)
        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7

        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 3 if i == 0 else enc_nf[i-1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(conv_block(dim, dec_nf[0] + enc_nf[-2], dec_nf[1]))  # 2
        self.dec.append(conv_block(dim, dec_nf[1] + enc_nf[-3], dec_nf[2]))  # 3
        self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(conv_block(dim, dec_nf[3], dec_nf[4]))  # 5

        if self.full_size:
            self.dec.append(conv_block(dim, dec_nf[4] + 3, dec_nf[5], 1))

#        if self.vm2:
#            self.vm2_conv = conv_block(dim, dec_nf[5], dec_nf[6]) 
 
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        for l in self.enc:
            x_enc.append(l(x_enc[-1]))

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
             y = self.dec[i](y)
             y = self.upsample(y)
             y = torch.cat([y, x_enc[-(i+2)]], dim=1)

        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)

        # Upsample to full res, concatenate and conv
        if self.full_size:
             y = self.upsample(y)
             y = torch.cat([y, x_enc[0]], dim=1)
             y = self.dec[5](y)

        # Extra conv for vm2
#        if self.vm2:
#             y = self.vm2_conv(y)

        return y

class unet_core_dila(nn.Module):
    """
    [unet_core] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """
    def __init__(self, dim, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate UNet model
            :param dim: dimension of the image passed into the net
            :param enc_nf: the number of features maps in each layer of encoding stage
            :param dec_nf: the number of features maps in each layer of decoding stage
            :param full_size: boolean value representing whether full amount of decoding 
                            layers
        """
        super(unet_core_dila, self).__init__()
        #(self, dim, in_channels, out_channels, stride=1)
        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7

        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 56 if i == 0 else enc_nf[i-1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(conv_block(dim, dec_nf[0] + enc_nf[-2], dec_nf[1]))  # 2
        self.dec.append(conv_block(dim, dec_nf[1] + enc_nf[-3], dec_nf[2]))  # 3
        self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(conv_block(dim, dec_nf[3], dec_nf[4]))  # 5

        if self.full_size:
            self.dec.append(conv_block(dim, dec_nf[4] + 56, dec_nf[5], 1))

#        if self.vm2:
#            self.vm2_conv = conv_block(dim, dec_nf[5], dec_nf[6]) 
 
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        for l in self.enc:
            x_enc.append(l(x_enc[-1]))

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
             y = self.dec[i](y)
             y = self.upsample(y)
             y = torch.cat([y, x_enc[-(i+2)]], dim=1)

        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)

        # Upsample to full res, concatenate and conv
        if self.full_size:
             y = self.upsample(y)
             y = torch.cat([y, x_enc[0]], dim=1)
             y = self.dec[5](y)

        # Extra conv for vm2
#        if self.vm2:
#             y = self.vm2_conv(y)

        return y

class voxelmorph_FF(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for 
    the 2018 implementation of voxelmorph.
    """
    def __init__(self, vol_size, enc_nf, dec_nf):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(voxelmorph_FF, self).__init__()

        dim = len(vol_size)

        self.unet_model = unet_core(dim, enc_nf, dec_nf)

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)      

        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, src, tgt):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        x = torch.cat([src, tgt], dim=1)
        x = self.unet_model(x)
        flow = self.flow(x)
        flow_up = flow
        return flow, flow_up

class unet_half(nn.Module):
    """
    [unet_core] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """
    def __init__(self, dim, enc_nf, dec_nf, coarse=True):
        """
        Instiatiate UNet model
            :param dim: dimension of the image passed into the net
            :param enc_nf: the number of features maps in each layer of encoding stage
            :param dec_nf: the number of features maps in each layer of decoding stage
            :param full_size: boolean value representing whether full amount of decoding 
                            layers
        """
        super(unet_half, self).__init__()
        #(self, dim, in_channels, out_channels, stride=1)

        # Encoder functions
        self.coarse = coarse
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i-1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(conv_block(dim, dec_nf[0] * 2, dec_nf[1]))  # 2
        self.dec.append(conv_block(dim, dec_nf[1] * 2, dec_nf[2]))  # 3
        self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(conv_block(dim, dec_nf[3], dec_nf[4]))  # 5
        if self.coarse:
            self.dec.append(conv_block(dim, dec_nf[4], dec_nf[5], 2))
            self.dec.append(conv_block(dim, dec_nf[5], dec_nf[6], 2))
 
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        for l in self.enc:
            x_enc.append(l(x_enc[-1]))

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
             y = self.dec[i](y)
             y = self.upsample(y)
             y = torch.cat([y, x_enc[-(i+2)]], dim=1)

        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)
        if self.coarse:
            y = self.dec[5](y)
            y = self.dec[6](y)
        return y

class unet_half_new(nn.Module):
    """
    [unet_core] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """
    def __init__(self, dim, enc_nf, dec_nf, coarse=True):
        """
        Instiatiate UNet model
            :param dim: dimension of the image passed into the net
            :param enc_nf: the number of features maps in each layer of encoding stage
            :param dec_nf: the number of features maps in each layer of decoding stage
            :param full_size: boolean value representing whether full amount of decoding 
                            layers
        """
        super(unet_half_new, self).__init__()
        #(self, dim, in_channels, out_channels, stride=1)

        # Encoder functions
        self.coarse = coarse
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i-1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(conv_block(dim, dec_nf[0] * 2, dec_nf[1]))  # 2
        self.dec.append(conv_block(dim, dec_nf[1] * 2, dec_nf[2]))  # 3
        self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(conv_block(dim, dec_nf[3], dec_nf[4]))  # 5
        if self.coarse:
            self.dec.append(conv_block(dim, dec_nf[4], dec_nf[5], 1))
            self.dec.append(conv_block(dim, dec_nf[5], dec_nf[6], 2))
 
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        for l in self.enc:
            x_enc.append(l(x_enc[-1]))

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
             y = self.dec[i](y)
             y = self.upsample(y)
             y = torch.cat([y, x_enc[-(i+2)]], dim=1)

        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)
        if self.coarse:
            y = self.dec[5](y)
            y = self.dec[6](y)
        return y

class Unet_dila(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for 
    the 2018 implementation of voxelmorph.
    """
    def __init__(self, enc_nf, dec_nf, down_factor):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(Unet_dila, self).__init__()

        dim = 3
        self.dila1 = nn.Conv3d(3, 32, 3,padding=1,dilation=1)
        self.dila2 = nn.Conv3d(3, 16, 3,padding=2,dilation=2)
        self.dila5 = nn.Conv3d(3, 8, 3,padding=5,dilation=5)
        self.unet_model = unet_core_dila(dim, enc_nf, dec_nf)
        self.down = nn.MaxPool3d(down_factor)
        self.upsample = nn.Upsample(scale_factor=down_factor, mode='trilinear')
        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)      
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, src, tgt, diff):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        x = torch.cat([src, tgt, diff], dim=1)
        x = self.down(x)
        dila1 = self.dila1(x)
        dila2 = self.dila2(x)
        dila5 = self.dila5(x)
        x = torch.cat([dila1, dila2, dila5],dim=1)
        x = self.unet_model(x)
        flow = self.flow(x)
        flow_up = self.upsample(flow)
        return flow, flow_up

class Unet(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for 
    the 2018 implementation of voxelmorph.
    """
    def __init__(self, enc_nf, dec_nf, down_factor):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(Unet, self).__init__()

        dim = 3
        self.unet_model = unet_core(dim, enc_nf, dec_nf)
        self.down = nn.MaxPool3d(down_factor)
        self.upsample = nn.Upsample(scale_factor=down_factor, mode='trilinear')
        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)      
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, src, tgt, diff):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        x = torch.cat([src, tgt, diff], dim=1)
        x = self.down(x)
        x = self.unet_model(x)
        flow = self.flow(x)
        flow_up = self.upsample(flow)
        return flow_up

class Dis_Unet(nn.Module):
    def __init__(self):
        super(Dis_Unet,self).__init__()
        self.conv1 = nn.Conv3d(3,32,3, stride=1,padding=1)
        self.conv11 = nn.Conv3d(32,32,3, stride=2,padding=1)
        self.conv2 = nn.Conv3d(32,64, 3, stride=1,padding=1)
        self.conv21 = nn.Conv3d(64,64,3, stride=2,padding=1)
        self.conv3 = nn.Conv3d(64,128,3, stride=1,padding=1)
        self.conv31 = nn.Conv3d(128,128,3,stride=2,padding=1)
        self.conv4 = nn.Conv3d(128,256,3,stride=1,padding=1)
        self.conv41 = nn.Conv3d(256,256,3,stride=2,padding=1)
        self.conv5 = nn.Conv3d(256,256,3,stride=1,padding=1)
        self.conv51 = nn.Conv3d(256,256,3,stride=2,padding=1)
        self.conv6 = nn.Conv3d(256,512,3,stride=1,padding=0)
        self.conv7 = nn.Conv3d(512,512,[4,4,1],stride=1,padding=0)
        self.linear = nn.Linear(512,1)
        self.drop = nn.Dropout(0.5)
        self.sig = nn.Sigmoid()
    def forward(self, dvf):
        output = self.conv1(dvf)
        output = self.conv11(output)
        # 96 96 48
        output = self.conv2(output)
        output = self.conv21(output)
        # 48 48 24
        output = self.conv3(output)
        output = self.conv31(output)
        # 24 24 12
        output = self.conv4(output)
        output = self.conv41(output)
        # 12 12 6
        output = self.conv5(output)
        output = self.conv51(output)
        # 6 6 3
        output = self.conv6(output)
        # 4 4 1
        output = self.conv7(output)
        # 1 1 1
        output = torch.flatten(output)
        output = self.drop(output)
        output = self.linear(output)
        output = self.sig(output)
        return output

class voxelmorph_C(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for 
    the 2018 implementation of voxelmorph.
    """
    def __init__(self, vol_size, enc_nf, dec_nf, coarse):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(voxelmorph_C, self).__init__()

        dim = len(vol_size)
        self.coarse = coarse
        self.unet_model = unet_half_new(dim, enc_nf, dec_nf, coarse)

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)      
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.upsample = nn.Upsample(scale_factor=4, mode='trilinear')

    def forward(self, src, tgt):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        x = torch.cat([src, tgt], dim=1)
        x = self.unet_model(x)
        flow = self.flow(x)
        flow_up = self.upsample(flow)
        return flow, flow_up
    
class simple_fine(nn.Module):
    def __init__(self):
        super(simple_fine,self).__init__()
        self.conv1 = conv_block_bn(dim=3, in_channels=2, out_channels=8, ksize=3, stride=1, padding=1)
        self.conv2 = conv_block_bn(dim=3, in_channels=8, out_channels=16, ksize=3, stride=1, padding=1)
        self.conv5 = conv_block_bn(dim=3, in_channels=16, out_channels=3, ksize=3, stride=1, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear')
        self.maxp = nn.MaxPool3d(2)
    def forward(self,src,tgt):
        x = torch.cat([src, tgt], dim=1)
        out = self.conv1(x, True)
        out = self.maxp(out)
        out = self.conv2(out, True)
        out = self.conv5(out, False)
        out_up = self.up(out)
        return out, out_up    

class voxelmorph_F(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for 
    the 2018 implementation of voxelmorph.
    """
    def __init__(self, vol_size, enc_nf, dec_nf, coarse):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(voxelmorph_F, self).__init__()

        dim = len(vol_size)

        self.unet_model = unet_half_new(dim, enc_nf, dec_nf, coarse)

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)      

        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

    def forward(self, src, tgt):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        x = torch.cat([src, tgt], dim=1)
        x = self.unet_model(x)
        flow = self.flow(x)
        flow_up = self.upsample(flow)
        return flow, flow_up
    
    
class voxelmorph_coarse(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for 
    the 2018 implementation of voxelmorph.
    """
    def __init__(self, vol_size, enc_nf, dec_nf, coarse):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(voxelmorph_coarse, self).__init__()

        dim = len(vol_size)
        self.coarse = coarse
        self.unet_model = unet_half(dim, enc_nf, dec_nf, coarse)

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)      
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.spatial_transform = SpatialTransformer(vol_size)
        self.upsample = nn.Upsample(scale_factor=8, mode='trilinear')

    def forward(self, src, tgt):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        x = torch.cat([src, tgt], dim=1)
        x = self.unet_model(x)
        flow = self.flow(x)
        flow_up = self.upsample(flow)
        y = self.spatial_transform(src, flow_up)
        return y, flow, flow_up


class voxelmorph_fine(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for 
    the 2018 implementation of voxelmorph.
    """
    def __init__(self, vol_size, enc_nf, dec_nf, coarse):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(voxelmorph_fine, self).__init__()

        dim = len(vol_size)

        self.unet_model = unet_half(dim, enc_nf, dec_nf, coarse)

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)      

        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')
        self.spatial_transform = SpatialTransformer(vol_size)


    def forward(self, src, tgt):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        x = torch.cat([src, tgt], dim=1)
        x = self.unet_model(x)
        flow = self.flow(x)
        flow_up = self.upsample(flow)
        y = self.spatial_transform(src, flow_up)
        return y, flow, flow_up

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


class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """
    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')

        self.main = conv_fn(in_channels, out_channels, ksize, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        return out
    

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
        self.conv1 = conv_block_bn(dim=3, in_channels=48, out_channels=48, ksize=3, stride=1, padding=1, dilation=1)
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
#        dia_4 = self.input_4(x)
        x = torch.cat([dia_1, dia_2, dia_3], dim=1)
        # 1x(3*16)x64x64x64
        x = self.conv1(x, True)
        # 1x48x64x64x64
        x = self.conv1(x, True)
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
        x = self.conv42(x, True)
        # 1x3x8x8x8
        x_up = self.upsample(x)
        # 1x3x64x64x64
        return x, x_up
    
    
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