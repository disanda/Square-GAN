# 自己写的一套DCGAN网络，可以通过图像分辨率调整网络层数
# input_dim 对应G的《输入维度》，scale表示《输入维度》对应《网络中间层维度（起点）》的放大倍数


import torch
from torch import nn
import torch.nn.utils.spectral_norm as spectral_norm


class Generator(nn.Module):
    def __init__(self, input_dim=128, output_channels=3, scale=16):
        super().__init__()
        layers = []
        x = input_dim*scale # 这里对应输入维度，表示《输入维度》对应《网络中间层维度（起点）》的放大倍数
        bias_flag = False

        # 1: 1x1 -> 4x4
        layers.append(nn.ConvTranspose2d(input_dim, x, kernel_size=4,stride=1,padding=0,bias=bias_flag))
        layers.append(nn.BatchNorm2d(x))
        layers.append(nn.ReLU())

        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> 32*32
        while x>input_dim:
            layers.append(nn.ConvTranspose2d(x, x//2, kernel_size=4, stride=2, padding=1 ,bias=bias_flag))
            layers.append(nn.BatchNorm2d(x//2))
            layers.append(nn.ReLU())
            x = x//2

        # 3:end 
        layers.append(nn.ConvTranspose2d(x,output_channels,kernel_size=4, stride=2, padding=1, bias=bias_flag))
        layers.append(nn.Tanh())

        # all
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.net(z)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim=128, input_channels=3, scale=16):
        super().__init__()
        layers=[]
        x = 1
        bias_flag = False

        # 1:
        layers.append(nn.Conv2d(input_channels, input_dim*x, kernel_size=4, stride=2, padding=1, bias=bias_flag))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        x = x*2

        # 2: 64*64 > 4*4
        while x<scale:  
            layers.append(nn.Conv2d(input_dim*x, input_dim*x*2, kernel_size=4, stride=2, padding=1, bias=bias_flag))
            layers.append(nn.BatchNorm2d(input_dim*x*2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            x = x * 2

        # 3: 4*4 > 1*1
        layers.append(nn.Conv2d(input_dim*x, 1, kernel_size=4, stride=1, padding=0))
        #layers.append(nn.Sigmoid())

        # all:
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y # [1,1,1,1]

class Discriminator_SpectrualNorm(nn.Module):
    def __init__(self, input_channels=3, feature_maps=64):
        super().__init__()
        layers=[]
        x1 = feature_maps
        x2 = 1
        bias_flag = False

        # 1:
        layers.append(spectral_norm(nn.Conv2d(input_channels, feature_maps, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        x1 = x1 // 2

        # 2: 64*64 > 4*4
        while x1>4:  
            layers.append(spectral_norm(nn.Conv2d(feature_maps*x2, feature_maps*x2*2, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            x1 = x1 // 2
            x2 = x2 * 2

        # 3: 4*4 > 1*1
        layers.append(nn.Conv2d(feature_maps*x2, 1, kernel_size=4, stride=1, padding=0))
        #layers.append(nn.Sigmoid())

        # all:
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y # [1,1,1,1]

class Encoder(nn.Module):
    def __init__(self, input_dim=128, input_channels=3, image_size=64):
        super().__init__()
        layers = []
        x = image_size//8

        # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
        d = dim
        layers.append(nn.Conv2d(input_channels, d, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))
        for i in range(n_downsamplings - 1):
            d_last = d
            d = d*2
            layers.append(conv_norm_lrelu(d_last, d, kernel_size=4, stride=2, padding=1))

        self.net = nn.Sequential(*layers)

        # 2: encoder:4*4*dim
        #layers.append(nn.Conv2d(d, 1, kernel_size=4, stride=1, padding=0))
        in_fc = int(input_dim*d*4*4) 
        layers_2 = []
        layers_2.append( spectral_norm(nn.Linear(in_fc,in_fc//d,bias=False)))
        layers_2.append( spectral_norm(nn.Linear(in_fc,in_fc//16,bias=False)))
        self.fc = nn.Sequential(*layers_2)

    def forward(self, x):
        y = self.net(x)
        y = y.view(-1,dim*4*4)
        y = self.fc(y)
        return y # [-1,dim] 
