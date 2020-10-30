import torch
from torch import nn
import torch.nn.utils.spectral_norm as spectral_norm
import math

class G_2(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        bias_flag = False

        # 1: k:1x1 
        layers.append(nn.ConvTranspose2d(3, 64, kernel_size=1,stride=1,padding=0,bias=bias_flag))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())

        # 2: upsamplings, k:4x4
        layers.append(nn.ConvTranspose2d(64,3,kernel_size=4, stride=2, padding=1, bias=bias_flag))
        layers.append(nn.Tanh())

        # all
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.net(z)
        return x

class D_2(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        bias_flag = False

        # 1:
        layers.append(spectral_norm(nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0, bias=bias_flag)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 2: 4*4 > 1*1
        layers.append(nn.Conv2d(64, 3, kernel_size=4, stride=2, padding=1))

        # all:
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y # [1,1,1,1]