import functools
import numpy as np
import tensorboardX
import torch
import tqdm
import argparse
import os
import yaml
import torchvision

import data
import networks
import loss_func
import g_penal


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

# command line
parser = argparse.ArgumentParser(description='the training args')
parser.add_argument('--dataset', default='fashion_mnist')#choices=['cifar10', 'fashion_mnist', 'mnist', 'celeba', 'anime', 'custom'])
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta_1', type=float, default=0.5)
parser.add_argument('--n_d', type=int, default=1)  # # d updates per g update
parser.add_argument('--z_dim', type=int, default=128)
parser.add_argument('--adversarial_loss_mode', default='gan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
parser.add_argument('--gradient_penalty_mode', default='none', choices=['none', '1-gp', '0-gp', 'lp'])
parser.add_argument('--gradient_penalty_sample_mode', default='line', choices=['line', 'real', 'fake', 'dragan'])
parser.add_argument('--gradient_penalty_weight', type=float, default=10.0)
parser.add_argument('--experiment_name', default='none')
parser.add_argument('--gradient_penalty_d_norm', default='layer_norm', choices=['instance_norm', 'layer_norm'])  # !!!
parser.add_argument('--img_size',type=int,default=64)
args = parser.parse_args()

# output_dir
if args.experiment_name == 'none':
    args.experiment_name = '%s_%s' % (args.dataset, args.adversarial_loss_mode)
    if args.gradient_penalty_mode != 'none':
        args.experiment_name += '_%s_%s' % (args.gradient_penalty_mode, args.gradient_penalty_sample_mode)

output_dir = os.path.join('output', args.experiment_name)

if not os.path.exists('output'):
    os.mkdir('output')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# save settings

with open(os.path.join(output_dir, 'settings.yml'), "w", encoding="utf-8") as f:
    yaml.dump(args, f)


# others
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

# setup dataset

if args.dataset in ['cifar10', 'fashion_mnist', 'mnist']:  # 3: 32x32  4:64:64
    data_loader, shape = data.make_dataset(args.dataset, args.batch_size,args.img_size,pin_memory=use_gpu)
    n_G_upsamplings = n_D_downsamplings = 4




# ==============================================================================
# =                                   model                                    =
# ==============================================================================

# setup the normalization function for discriminator
if args.gradient_penalty_mode == 'none':
    d_norm = 'batch_norm'
else:  # cannot use batch normalization with gradient penalty
    d_norm = args.gradient_penalty_d_norm

# networks
G = networks.ConvGenerator(args.z_dim, shape[-1], n_upsamplings=n_G_upsamplings).to(device)
D = networks.ConvDiscriminator(shape[-1], n_downsamplings=n_D_downsamplings, norm=d_norm).to(device)
print(G)
print(D)

# adversarial_loss_functions
d_loss_fn, g_loss_fn = loss_func.get_adversarial_losses_fn(args.adversarial_loss_mode)

# optimizer
G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

def train_G():
    G.train()
    D.train()

    z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)
    x_fake = G(z)

    x_fake_d_logit = D(x_fake)
    G_loss = g_loss_fn(x_fake_d_logit)

    G.zero_grad()
    G_loss.backward()
    G_optimizer.step()

    return {'g_loss': G_loss}


def train_D(x_real):
    G.train()
    D.train()

    z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)
    x_fake = G(z).detach()

    x_real_d_logit = D(x_real)
    x_fake_d_logit = D(x_fake)

    x_real_d_loss, x_fake_d_loss = d_loss_fn(x_real_d_logit, x_fake_d_logit)
    gp = g_penal.gradient_penalty(functools.partial(D), x_real, x_fake, gp_mode=args.gradient_penalty_mode, sample_mode=args.gradient_penalty_sample_mode)

    D_loss = (x_real_d_loss + x_fake_d_loss) + gp * args.gradient_penalty_weight

    D.zero_grad()
    D_loss.backward()
    D_optimizer.step()

    return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}


@torch.no_grad()
def sample(z):
    G.eval()
    return G(z)


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

ckpt_dir = os.path.join(output_dir, 'checkpoints')
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

# try:
#     ckpt_path = os.path.join(ckpt_dir, 'xxx.ckpt')
#     ckpt=torch.load(ckpt_path)
#     ep, it_d, it_g = ckpt['ep'], ckpt['it_d'], ckpt['it_g']
#     D.load_state_dict(ckpt['D'])
#     G.load_state_dict(ckpt['G'])
#     D_optimizer.load_state_dict(ckpt['D_optimizer'])
#     G_optimizer.load_state_dict(ckpt['G_optimizer'])
# except:
#     ep, it_d, it_g = 0, 0, 0


# sample
sample_dir = os.path.join(output_dir, 'samples_training')
if not os.path.exists(sample_dir):
	os.mkdir(sample_dir)

# main loop
writer = tensorboardX.SummaryWriter(os.path.join(output_dir, 'summaries'))
z = torch.randn(64, args.z_dim, 1, 1).to(device)  # a fixed noise for sampling

for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
    it_d, it_g = 0, 0
    for x_real,flag in tqdm.tqdm(data_loader, desc='Inner Epoch Loop'):
        #print(x_real.shape)
        x_real = x_real.to(device)
        D_loss_dict = train_D(x_real)
        it_d += 1
        for k, v in D_loss_dict.items():
            writer.add_scalar('D/%s' % k, v.data.cpu().numpy(), global_step=it_d)

        if it_d % args.n_d == 0:
            G_loss_dict = train_G()
            it_g += 1
            for k, v in G_loss_dict.items():
                writer.add_scalar('G/%s' % k, v.data.cpu().numpy(), global_step=it_g)
        # sample
        if it_g % 100 == 0:
            x_fake = (sample(z)+1)/2
            torchvision.utils.save_image(x_fake,sample_dir+'/ep%d_%d.jpg'%(ep,it_g), nrow=8)
    # save checkpoint
    if (ep+1)%5==0:
        torch.save(G.state_dict(), ckpt_dir+'/Epoch_G_(%d).pth' % ep)
        torch.save(D.state_dict(), ckpt_dir+'/Epoch_D_(%d).pth' % ep)
