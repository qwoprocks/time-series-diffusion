# Portions of code in this file were created with reference to:
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# CMU 16-726 Learning-Based Image Synthesis / Spring 2024, Assignment 3

import torch
from torch import einsum
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torch_frft.frft_module import frft
from torch_frft.dfrft_module import dfrft, dfrftmtx

import math
import random
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import numpy as np

import cv2
from diffusion_utils import *

# change this according to the training and testing files!
# Variable does not autopopulate from opts, but needs to be same as opts.denoising_steps
TIMESTEPS = 500


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        self.rnn_layer = nn.Sequential(
            nn.LayerNorm(2036),
            nn.LSTM(input_size=2036, hidden_size=256, num_layers=2, batch_first=True),
        )
        self.rnn_proj = nn.Sequential(
            nn.Conv1d(55, 1, 1),
            nn.Linear(256, 512)
        )

        self.frft_param = nn.Parameter(torch.tensor(1.25, dtype=torch.float32), requires_grad=True)
        self.frft_proj = nn.Sequential(
            nn.Conv1d(55, 1, 1),
            nn.Linear(2036, 512)
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings
        time_dim = dim * 4
        self.time_layer = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == (len(in_out) - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                Attention(dim_in),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))
        
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        self.mid_attn = Attention(mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                Attention(dim_out),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        self.out_dim = default(out_dim, channels)

        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        self.low_rank_adapter_conv = nn.Conv1d(55, 1, 1)
        self.low_rank_adapter_lin = nn.Linear(2036, time_dim)


    def forward(self, x, time, x_self_cond=None, time_cond=None):
        # forward pass

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        time_embed = self.time_layer(time)
        if time_cond is not None:
            time_cond_lora = self.low_rank_adapter_conv(time_cond)
            time_cond_lora = time_cond_lora.squeeze(dim=1)
            time_cond_lora = self.low_rank_adapter_lin(time_cond_lora)
            time_embed += time_cond_lora

            x_rnn_cond, (_, _) = self.rnn_layer(time_cond)
            x_rnn_cond = self.rnn_proj(x_rnn_cond).squeeze(dim=1)
            time_embed += x_rnn_cond

            x_frft_cond = dfrft(time_cond, self.frft_param, dim=1)
            x_frft_cond = self.frft_proj(x_frft_cond.real).squeeze(dim=1)
            time_embed += x_frft_cond

        x = self.init_conv(x)
        r = x.clone()

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time_embed)
            h.append(x)

            x = block2(x, time_embed)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, time_embed)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, time_embed)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, time_embed)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, time_embed)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, time_embed)
        return self.final_conv(x)
    

def beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

timesteps = TIMESTEPS

# define beta schedule
betas = beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]

    out = a.gather(-1, t.cpu())

    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

image_size = 128

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1", time_cond=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start, t, noise) # get noisy image

    x_self_cond = None
    if denoise_model.self_condition and random.random() < 0.5:
        with torch.no_grad():
            x_self_cond = denoise_model(x_noisy, t)
            x_self_cond.detach_()

    predicted_noise = denoise_model(x_noisy, t, x_self_cond, time_cond) # denoise the generated noisy image

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise) # define l1 loss
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise) # define l2 loss
    elif loss_type == "huber":
        loss = F.huber_loss(noise, predicted_noise) # define huber loss
    else:
        raise NotImplementedError()

    return loss


@torch.no_grad()
def p_sample(model, x, t, t_index, time_cond):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, time_cond=time_cond) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)

        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def p_sample_loop(model, shape, time_cond):
    device = next(model.parameters()).device

    b = shape[0]

    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, time_cond)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3, time_cond=None):
    return p_sample_loop(model, shape=(batch_size, channels, image_size[0], image_size[1]), time_cond=time_cond)
