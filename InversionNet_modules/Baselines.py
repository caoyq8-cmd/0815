from collections import OrderedDict
from math import ceil

import torch.nn as nn
import torch.nn.functional as F

NORM_LAYERS = {'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm}

class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResizeBlock(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest', norm='bn'):
        super(ResizeBlock, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class InversionNet(nn.Module):
    def __init__(self, dim1=64, dim2=128, dim3=256, dim4=512, dim5=1024, sample_spatial=1.0, **kwargs):
        super(InversionNet, self).__init__()
        self.convblock1 = ConvBlock(2, dim1, kernel_size=5, stride=2, padding=2)
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=5, padding=2)
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=5, stride=2, padding=2)
        self.convblock3_1 = ConvBlock(dim2, dim3, kernel_size=5, padding=2)
        self.convblock3_2 = ConvBlock(dim3, dim3, kernel_size=5, stride=2, padding=2)
        self.convblock4_1 = ConvBlock(dim3, dim4, kernel_size=5, padding=2)
        self.convblock4_2 = ConvBlock(dim4, dim4, kernel_size=5, stride=2, padding=2)
        self.convblock5_1 = ConvBlock(dim4, dim5, stride=2)
        self.convblock5_2 = ConvBlock(dim5, dim5)
        self.convblock6 = ConvBlock(dim5, dim5, stride=2)

        # 解码器
        self.decoder5 = UNetDecoderBlock(dim5, dim4) 
        self.decoder4 = UNetDecoderBlock(dim4, dim3) 
        self.decoder3 = UNetDecoderBlock(dim3, dim2) 
        self.decoder2 = UNetDecoderBlock(dim2, dim1) 
        self.decoder1 = UNetDecoderBlock(dim1, 32) 
        # 输出层 
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):

        # Encoder Part
        x = self.convblock1(x) 
        x = self.convblock2_1(x) 
        x = self.convblock2_2(x) 
        x = self.convblock3_1(x) 
        x = self.convblock3_2(x) 
        x = self.convblock4_1(x) 
        x = self.convblock4_2(x) 
        x = self.convblock5_1(x) 
        x = self.convblock5_2(x)  
        
         # 解码器前向传播 
        x = self.decoder5(x)
        x = self.decoder4(x)
        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.decoder1(x)
        # 输出层
        x = self.final_conv(x)

        return x

class InversionNet_modified_2(nn.Module):
    def __init__(self, dim1=64, dim2=128, dim3=256, sample_spatial=1.0, **kwargs):
        super(InversionNet_modified_2, self).__init__()
        self.convblock1 = ConvBlock(2, dim1, kernel_size=5, stride=2, padding=2)
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=5, padding=2)
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=5, stride=2, padding=2)
        self.convblock3_1 = ConvBlock(dim2, dim3, stride=2)
        self.convblock3_2 = ConvBlock(dim3, dim3)

        # 解码器
        self.decoder5 = UNetDecoderBlock(dim3, dim2) # 1 * 1 -> 2 * 2
        self.decoder4 = UNetDecoderBlock(dim2, dim1) # 2 * 16 -> 32 * 32
        self.decoder3 = UNetDecoderBlock(dim1, 32) # 32 * 32 -> 64 * 64
        self.decoder2 = UNetDecoderBlock(32, 16) # 64 * 64 -> 128 * 128
        self.decoder1 = UNetDecoderBlock(16, 8) # 64 * 64 -> 128 * 128

        # 输出层 
        self.final_conv = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):

        # Encoder Part
        x = self.convblock1(x)  # (None, 32, 500, 70)
        x = self.convblock2_1(x)  # (None, 64, 250, 70)
        x = self.convblock2_2(x)  # (None, 64, 250, 70)
        x = self.convblock3_1(x)  # (None, 128, 63, 70)
        x = self.convblock3_2(x)  # (None, 128, 63, 70)
        
        # 解码器前向传播 
        x = self.decoder5(x)
        x = self.decoder4(x)
        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.decoder1(x)

        # 输出层
        x = self.final_conv(x)

        return x