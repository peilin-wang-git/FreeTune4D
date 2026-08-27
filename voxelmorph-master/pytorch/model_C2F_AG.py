# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:56:38 2020

@author: Haonan Xiao

E-mail: hx42@duke.edu
"""
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal


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

class AG_Unet(nn.Module):
    def __init__(self, enc_nf, dec_nf, down_factor):
        super(AG_Unet,self).__init__()
        self.maxp = nn.MaxPool3d(down_factor)
        self.upsample = nn.Upsample(scale_factor=down_factor, mode='trilinear')
        self.up = nn.Upsample(scale_factor=2, mode='trilinear')
        self.enc0 = conv_block_bn(dim=3, in_channels=2, out_channels=enc_nf[0], ksize=3, stride=2, padding=1)
        self.enc1 = conv_block_bn(dim=3, in_channels=enc_nf[0], out_channels=enc_nf[1], ksize=3, stride=2, padding=1)
        self.enc2 = conv_block_bn(dim=3, in_channels=enc_nf[1], out_channels=enc_nf[2], ksize=3, stride=2, padding=1)
        self.enc3 = conv_block_bn(dim=3, in_channels=enc_nf[2], out_channels=enc_nf[3], ksize=3, stride=2, padding=1)
        self.dec0 = conv_block_bn(dim=3, in_channels=enc_nf[3], out_channels=dec_nf[0], ksize=3, stride=1, padding=1)
        self.dec1 = conv_block_bn(dim=3, in_channels=enc_nf[2]+dec_nf[0], out_channels=dec_nf[1], ksize=3, stride=1, padding=1)
        self.dec2 = conv_block_bn(dim=3, in_channels=enc_nf[1]+dec_nf[1], out_channels=dec_nf[2], ksize=3, stride=1, padding=1)
        self.dec3 = conv_block_bn(dim=3, in_channels=enc_nf[0]+dec_nf[2], out_channels=dec_nf[3], ksize=3, stride=1, padding=1)
        self.dec4 = conv_block_bn(dim=3, in_channels=2+dec_nf[3], out_channels=dec_nf[4], ksize=3, stride=1, padding=1)
        self.atten0 = Attention_block(dec_nf[0],enc_nf[2],16)
        self.atten1 = Attention_block(dec_nf[1],enc_nf[1],16)
        self.atten2 = Attention_block(dec_nf[2],enc_nf[0],16)
        self.atten3 = Attention_block(dec_nf[3],2,16)
        self.flow = nn.Conv3d(dec_nf[4], 3, kernel_size=3, padding=1)       
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1) 
        x = self.maxp(x) # 1
        enc0 = self.enc0(x, True) # 1/2 * enc_nf[0]
        enc1 = self.enc1(enc0, True) # 1/4 * enc_nf[1]
        enc2 = self.enc2(enc1, True) # 1/8 * enc_nf[2]
        enc3 = self.enc3(enc2, True) # 1/16 * enc_nf[3]
        dec0 = self.dec0(enc3, True) # 1/16 * dec_nf[0]
        alpha0 = self.atten0(dec0, enc2) # 1/8 * dec_nf[0]
        dec1 = self.up(dec0) # 1/8 * dec_nf[0]
        dec1 = self.dec1(torch.cat([alpha0, dec1], dim=1), True) # 1/8 * dec_nf[1]
        alpha1 = self.atten1(dec1, enc1) # 1/8 * dec_nf[1]
        dec2 = self.up(dec1) # 1/4 * dec[1]
        dec2 = self.dec2(torch.cat([alpha1, dec2], dim=1), True) # 1/4 * dec_nf[2]
        alpha2 = self.atten2(dec2, enc0) # 1/4 * dec_nf[2]
        dec3 = self.up(dec2) # 1/2 * dec_nf[2]
        dec3 = self.dec3(torch.cat([alpha2, dec3], dim=1), True) # 1/2 * dec_nf[3]
        alpha3 = self.atten3(dec3, x) # 1 * dec_nf[3]
        dec4 = self.up(dec3) # 1 * dec_nf[3]
        dec4 = self.dec4(torch.cat([alpha3, dec4], dim=1), True) # 1 * dec_nf[4]
        flow = self.flow(dec4)
        flow = self.upsample(flow)
        return flow, flow

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
        g1 = self.W_g(g) # 1/2 * int
        x1 = self.W_x(x) # 1 * int
        x1 = self.downsample(x1) # 1/2 * int
        psi = self.relu(g1+x1) # 1/2 * int
        psi = self.sigmoid(self.psi(psi)) # 1/2 * 1
        psi = self.upsample(psi) # 1 * 1
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