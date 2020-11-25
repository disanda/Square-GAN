#第4版, D2E , Equal-learningRate

import torch
from torch import nn
import torch.nn.utils.spectral_norm as spectral_norm
import math

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def get_para_GByte(parameter_number):
     x=parameter_number['Total']*8/1024/1024/1024
     y=parameter_number['Total']*8/1024/1024/1024
     return {'Total_GB': x, 'Trainable_BG': y}

class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None
    def forward(self, input):
        out = F.conv2d(input,self.weight * self.scale,bias=self.bias,stride=self.stride,padding=self.padding)
        return out


class Generator(nn.Module):
    def __init__(self, input_dim=128, output_channels=3, image_size=128, scale=16, another_times=0):
        super().__init__()
        layers = []
        bias_flag = False
        up_times = 5

        # 1: 1x1 -> 4x4
        layers.append(nn.ConvTranspose2d(512, 512, kernel_size=4,stride=1,padding=0,bias=bias_flag))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU())

        layers.append(nn.ConvTranspose2d(512, 512, kernel_size=1,stride=1,padding=0,bias=bias_flag))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU())

        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> 32*32 -> 64 -> 128
        while up_times>0:
            layers.append(nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1 ,bias=bias_flag))
            layers.append(nn.BatchNorm2d(512))
            layers.append(nn.ReLU())
            layers.append(nn.ConvTranspose2d(512, 512, kernel_size=1, stride=1, padding=0 ,bias=bias_flag))
            layers.append(nn.BatchNorm2d(512))
            layers.append(nn.ReLU())
            up_times = up_times - 1


        # 3:end 
        layers.append(nn.ConvTranspose2d(512,256, kernel_size=4,stride=2, padding=1, bias=bias_flag))
        layers.append(nn.Tanh())


        layers.append(nn.ConvTranspose2d(256,3,kernel_size=1, bias=bias_flag))
        layers.append(nn.Tanh())


        # all
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.net(z)
        return x


class Discriminator_SpectrualNorm(nn.Module):
    def __init__(self, input_dim=128, input_channels=3, image_size=128, scale=16, another_times=0):
        super().__init__()
        layers=[]
        up_times = 5
        bias_flag = False

        # 1:
        layers.append(spectral_norm(nn.EqualConv2d(3, 256, kernel_size=3,tride=1,padding=0,bias=bias_flag)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(spectral_norm(nn.EqualConv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 2: 64*64 > 4*4
        while up_times>0:
            layers.append(spectral_norm(nn.EqualConv2d(512, 512, kernel_size=3,tride=1,padding=0, bias=bias_flag)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(spectral_norm(nn.EqualConv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            up_times = up_times - 1

        # 3: 4*4 > 1*1
        layers.append(spectral_norm(nn.EqualConv2d(512, 512, kernel_size=3,tride=1,padding=0, bias=bias_flag)))
        layers.append(nn.EqualConv2d(512, 512, kernel_size=4, stride=1, padding=0))
        #layers.append(nn.Sigmoid())

        # all:
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        y = y.mean()
        return y # [1,1,1,1]


