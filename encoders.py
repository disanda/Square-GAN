import torch
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self, *args, **keyword_args):
        super().__init__()
    def forward(self, x):
        return x

def _get_norm_layer_2d(norm):
    if norm == 'none':
        return Identity
    elif norm == 'batch_norm':
        return nn.BatchNorm2d
    elif norm == 'instance_norm':
        return functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm == 'layer_norm':
        return lambda num_features: nn.GroupNorm(1, num_features)
    else:
        raise NotImplementedError

# z:[128,-1,1,1]  , y/x = [-1,1,64,64]

# class D(nn.Module):
#     def __init__(self, nc, ndf):
#         super().__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),             # input is (nc) x 64 x 64
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),                   # state size. (ndf) x 32 x 32
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),                           # state size. (ndf*2) x 16 x 16
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),                                       # state size. (ndf*4) x 8 x 8
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),             # state size. (ndf*8) x 4 x 4
#             nn.Sigmoid()
#         )
#     def forward(self, input):
#         return self.main(input)

#encoder_v1: linear+lrelu  y/x = [-1,1,64,64] z:[128,-1,1,1]  , 
class encoder_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
        nn.Linear(4096, 1024),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(1024, 256),
        nn.BatchNorm1d(256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 128),
        )
    def forward(self, x):
        #print(x.shape)
        y = x.view(-1,4096)
        y = self.main(y)
        y = y.view(-1,128,1,1)
        return y


#encoder_v2: conv+lrelu
class encoder_v2(nn.Module):
    def __init__(self,nc=1,ndf=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),             # input is nc->128 | 64 --> 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),                   # 128->128  | 32 --> 16
            nn.BatchNorm2d(ndf ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, 2, 4, 2, 1, bias=False),                           # 128 --> 2  | 16 --> 8 | [-1,2,8,8]
        )
    def forward(self, x):
        y = self.conv(x)
        y = y.view(-1,128,1,1)
        return y

#改编自DCGAN
class encoder_v3(nn.Module):
    def __init__(self,
                 input_channels=1,
                 dim=128,
                 n_downsamplings=4,
                 norm='batch_norm'):
        super().__init__()

        Norm = _get_norm_layer_2d(norm)

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False or Norm == Identity),
                Norm(out_dim),
                nn.LeakyReLU(0.2)
            )
        layers = []
        # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
        d = dim
        layers.append(nn.Conv2d(input_channels, d, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))
        for i in range(n_downsamplings - 1):
            d_last = d
            d = min(dim * 2 ** (i + 1), dim * 8)
            layers.append(conv_norm_lrelu(d_last, d, kernel_size=4, stride=2, padding=1))
        # 2: logit
        layers.append(nn.Conv2d(d, 128, kernel_size=4, stride=1, padding=0)) #只改动这一层
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        y = self.net(x)
        #y = torch.transpose(y, 1, 0) #[-1,128,1,1]->[128,-1,1,1] 对齐G
        return y 

#encoder_v4: 1conv+1lrelu
class encoder_v4(nn.Module):
    def __init__(self,nc=1,ndf=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nc, 1, 4, 2, 1, bias=False),             # input is nc->2 | 64 --> 32
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(1024,128)
    def forward(self, x):
        y = self.conv(x)
        y = y.view(-1,1024)
        y = self.fc(y)
        y = y.view(-1,128,1,1)
        return y

# 1 fc
class encoder_v5(nn.Module):
    def __init__(self,nc=1,ndf=128):
        super().__init__()
        self.fc = nn.Linear(4096,128)
    def forward(self, x):
        y = x.view(-1,4096)
        y = self.fc(y)
        y = y.view(-1,128,1,1)
        return y



