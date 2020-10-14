# ----------自己写的一套DCGAN网络，可以通过图像分辨率调整网络参数，包括输入维度，中间维度.--------

# 1. 以及不同分辨率的对应不同上采样数(即网络的层数)，默认第一次上采样，像素从 1->4。 之后每一次上采样，像素增加一倍(这里应该是长宽都增加一倍).
# 2. input_dim 对应G的《输入维度》，image_size表示《生成图片对应的像素》, first_hidden_dim对应《网络中间层维度》(中间层起点的维度)
# 3. scale是input_dim放大的倍数，用于决定中间隐藏层起始时的size, 其和input_dim共同决定网络参数的规模.
# 4. 如果希望G的输入维度 和 D的输出维度对称, 则 first_hidden_dim = input_dim * scale

import torch
from torch import nn
import torch.nn.utils.spectral_norm as spectral_norm
import math

class Generator(nn.Module):
    def __init__(self, input_dim=128, output_channels=3, image_size=128, scale=16):
        super().__init__()
        layers = []
        up_times = math.log(image_size,2)- 3 # 减去前两次 1->2->4， 及最后一次， 方便中间写循环
        first_hidden_dim = input_dim*scale # 这里对应输入维度，表示《输入维度》对应《网络中间层维度（起点）》的放大倍数
        bias_flag = False

        # 1: 1x1 -> 4x4
        layers.append(nn.ConvTranspose2d(input_dim, first_hidden_dim, kernel_size=4,stride=1,padding=0,bias=bias_flag))
        layers.append(nn.BatchNorm2d(first_hidden_dim))
        layers.append(nn.ReLU())

        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> 32*32
        hidden_dim = first_hidden_dim
        while up_times>0:
            layers.append(nn.ConvTranspose2d(hidden_dim, hidden_dim//2, kernel_size=4, stride=2, padding=1 ,bias=bias_flag))
            layers.append(nn.BatchNorm2d(hidden_dim//2))
            layers.append(nn.ReLU())
            up_times = up_times - 1
            hidden_dim = hidden_dim // 2

        # 3:end 
        layers.append(nn.ConvTranspose2d(hidden_dim,output_channels,kernel_size=4, stride=2, padding=1, bias=bias_flag))
        layers.append(nn.Tanh())

        # all
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.net(z)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim=128, input_channels=3, image_size=128, scale=16):
        super().__init__()
        layers=[]
        up_times = math.log(image_size,2)- 3
        first_hidden_dim = input_dim*scale//16 # 默认为input_dim 
        bias_flag = False

        # 1:
        layers.append(nn.Conv2d(input_channels, first_hidden_dim, kernel_size=4, stride=2, padding=1, bias=bias_flag))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 2: 64*64 > 4*4
        hidden_dim = first_hidden_dim
        while up_times>0:  
            layers.append(nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1, bias=bias_flag))
            layers.append(nn.BatchNorm2d(hidden_dim*2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            hidden_dim = hidden_dim * 2
            up_times = up_times - 1

        # 3: 4*4 > 1*1
        layers.append(nn.Conv2d(hidden_dim, 1, kernel_size=4, stride=1, padding=0))
        #layers.append(nn.Sigmoid())

        # all:
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y # [1,1,1,1]

class Discriminator_SpectrualNorm(nn.Module):
    def __init__(self, input_dim=128, input_channels=3, image_size=128, scale=16):
        super().__init__()
        layers=[]
        up_times = math.log(image_size,2)- 3
        first_hidden_dim = input_dim * scale // 16 # 默认为input_dim 
        bias_flag = False

        # 1:
        layers.append(spectral_norm(nn.Conv2d(input_channels, first_hidden_dim, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 2: 64*64 > 4*4
        hidden_dim = first_hidden_dim
        while up_times>0:  
            layers.append(spectral_norm(nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            hidden_dim = hidden_dim * 2
            up_times = up_times - 1

        # 3: 4*4 > 1*1
        layers.append(nn.Conv2d(hidden_dim, 1, kernel_size=4, stride=1, padding=0))
        #layers.append(nn.Sigmoid())

        # all:
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y # [1,1,1,1]

#需要参照上述改动重写

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
