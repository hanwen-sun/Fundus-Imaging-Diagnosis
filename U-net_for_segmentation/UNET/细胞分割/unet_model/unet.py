# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/12/9 下午12:58

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    unet的编码器中，每一个level都会有两层卷积和Relu
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.double_conv(x)

class downsample(nn.Module):
    """
    下采样  maxpool + DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(downsample,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),#feature map 减半
            DoubleConv(in_channels,out_channels),
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class upsample(nn.Module):
    """
    upsample,  使用双线性插值或者反卷积
    """
    def __init__(self, in_channels,out_channels,bilinear = True):
        super(upsample,self).__init__()
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                        align_corners=True)
        else:
            self.upsample = nn.ConvTranspose2d(in_channels//2, out_channels//2,
                                               kernel_size=2,stride=2)
        self.conv = DoubleConv(in_channels,out_channels)
    def forward(self,x1,x2):
        """
        :param x1: decoder feature
        :param x2: encoder feature
        :return:
        """
        x1 = self.upsample(x1)

        diff_y = torch.tensor([x2.size()[2] - x1.size()[2]])
        diff_x = torch.tensor([x2.size()[3] - x1.size()[3]])

        #将x1与x2的特征图对齐后concat
        x1 = F.pad(x1, [diff_x//2,diff_x - diff_x//2,
                   diff_y//2,diff_y - diff_y // 2])
        x = torch.cat([x2,x1],dim=1)
        return self.conv(x)

class output_conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(output_conv, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
    def forward(self,x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self,n_channels,n_classes,bilinear = True):
        """
        :param n_channels: input channel
        :param n_classes: segmentation classes
        :param bilinear: upsample tpye
        """
        super(UNET,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.init = DoubleConv(n_channels,64)
        self.downsample1 = downsample(64,128)
        self.downsample2 = downsample(128,256)
        self.downsample3 = downsample(256,512)
        self.downsample4 = downsample(512,512)
        self.upsample1 = upsample(1024,256,bilinear)
        self.upsample2 = upsample(512,128,bilinear)
        self.upsample3 = upsample(256,64,bilinear)
        self.upsample4 = upsample(128,64,bilinear)
        self.outconv = output_conv(64,n_classes)
    def forward(self,x):
        x1 = self.init(x)
        x2 = self.downsample1(x1)
        x3 = self.downsample2(x2)
        x4 = self.downsample3(x3)
        x5 = self.downsample4(x4)

        x = self.upsample1(x5,x4)
        x = self.upsample2(x, x3)
        x = self.upsample3(x,x2)
        x = self.upsample4(x, x1)

        res = self.outconv(x)
        return res



