# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:17:04 2022

@author: user
"""
#%% VoxelMorph
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal

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

class Unet(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for 
    the 2018 implementation of voxelmorph.
    """
    def __init__(self, enc_nf, dec_nf):
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
        x = self.unet_model(x)
        flow = self.flow(x)
        return flow

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


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
    
#%% DenseNet
class _DenseLayer(nn.Sequential):
    """Origninated from https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py"""
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features)),
        self.add_module('elu', nn.ELU(inplace=True)),
        self.add_module('conv', nn.Conv3d(num_input_features, growth_rate, 
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        # Concatenation 
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = nnf.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    """Origninated from https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, 
                                growth_rate=growth_rate, 
                                bn_size=bn_size, 
                                drop_rate=drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class Generator(nn.Module):
    """Origninated from Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """

    def __init__(self, growth_rate=16, block_config=(4, 4, 4, 4),
                 bn_size=2, drop_rate=0):

        super(Generator, self).__init__()
        # First convolution
        self.conv0 = nn.Conv3d(1, 2*growth_rate, kernel_size=3, padding=1, bias=False)

        # Each denseblock
        num_features = 2 * growth_rate  #2k
        num_features_cat = num_features
        self.block0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features_cat += block_config[0]* growth_rate + num_features
        self.comp0 = nn.Conv3d(num_features_cat, num_features,
                               kernel_size=1, stride=1, bias=False)
        
        self.block1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features_cat += block_config[1]* growth_rate + num_features
        self.comp1 = nn.Conv3d(num_features_cat, num_features,
                               kernel_size=1, stride=1, bias=False)        
        
        self.block2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features_cat += block_config[2]* growth_rate + num_features
        self.comp2 = nn.Conv3d(num_features_cat, num_features,
                               kernel_size=1, stride=1, bias=False)   
  
        self.block3 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features_cat += block_config[3]* growth_rate + num_features
        self.recon = nn.Conv3d(num_features_cat, 1,
                               kernel_size=1, stride=1, bias=False)      

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv0(x)
        out = self.block0(x)
        features = torch.cat([x,out],1)
        out = self.comp0(features)
        
        out = self.block1(out)
        features = torch.cat([features,out],1)
        out = self.comp1(features)
    
        out = self.block2(out)
        features = torch.cat([features,out],1)
        out = self.comp2(features)
    
        out = self.block3(out)
        features = torch.cat([features,out],1)
        out = self.recon(features)
        return out
    
class Discriminator(nn.Module):
    """Origninated from SRGAN paper, see `"Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" <https://arxiv.org/abs/1609.04802>`_ 
    Args:
        ngpu (int) - how many GPU you use.
        cube_size (int) - the size of one patch (eg. 64 means a cubic patch with size: 64x64x64), this is exact the size of the model input.
    """
    def __init__(self, ngpu, cube_size=64):
        super(Discriminator, self).__init__()
        num_features = 64
        self.gpu = ngpu
        self.main = nn.Sequential(
            nn.Conv3d(1, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(1),

            nn.Conv3d(num_features, num_features, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([num_features,cube_size//2,cube_size//2,cube_size//2]),
            nn.LeakyReLU(1),

            nn.Conv3d(num_features, 2*num_features, kernel_size=3, padding=1),
            nn.LayerNorm([2*num_features,cube_size//2,cube_size//2,cube_size//2]),
            nn.LeakyReLU(1),

            nn.Conv3d(2*num_features, 2*num_features, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([2*num_features,cube_size//4,cube_size//4,cube_size//4]),
            nn.LeakyReLU(1),

            nn.Conv3d(2*num_features, 4*num_features, kernel_size=3, padding=1),
            nn.LayerNorm([4*num_features,cube_size//4,cube_size//4,cube_size//4]),
            nn.LeakyReLU(1),

            nn.Conv3d(4*num_features, 4*num_features, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([4*num_features,cube_size//8,cube_size//8,cube_size//8]),
            nn.LeakyReLU(1),

            nn.Conv3d(4*num_features, 8*num_features, kernel_size=3, padding=1),
            nn.LayerNorm([8*num_features,cube_size//8,cube_size//8,cube_size//8]),
            nn.LeakyReLU(1),

            nn.Conv3d(8*num_features, 8*num_features, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([8*num_features,cube_size//16,cube_size//16,cube_size//16]),
            nn.LeakyReLU(1),

            # different from the original SRGAN, we replaced the FC layers by global averaging pooling and convolution layers.
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(8*num_features, 16*num_features, kernel_size=1),
            nn.LeakyReLU(1),
            nn.Conv3d(16*num_features, 1, kernel_size=1)
        )

    def forward(self, x):
        out = self.main(x)
        return out.view(out.size()[0])
    
#%% Combination
class D2R_Net(nn.Module):
    """
    A network for joint image quality improvment and deformable image registration.
    """
    def __init__(self, enc_nf, dec_nf):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(D2R_Net, self).__init__()
        dim = 3
        self.unet_model = unet_core(dim, enc_nf, dec_nf)
        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)      
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.SR = Generator(growth_rate=16)

    def forward(self, src, tgt_down):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        tgt = self.SR(tgt_down)
        diff = src - tgt
        x = torch.cat([src, tgt, diff], dim=1)
        x = self.unet_model(x)
        flow = self.flow(x)
        return tgt, flow



