import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)



class ConvNextBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mult=2,
        time_embedding_dim=None,
        norm=True,
        group=8,
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_embedding_dim, in_channels))
            if time_embedding_dim
            else None
        )

        self.in_conv = nn.Conv2d(
            in_channels, in_channels, 7, padding=3, groups=in_channels
        )

        self.block = nn.Sequential(
            nn.GroupNorm(1, in_channels) if norm else nn.Identity(),
            nn.Conv2d(in_channels, out_channels * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, out_channels * mult),
            nn.Conv2d(out_channels * mult, out_channels, 3, padding=1),
        )

        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_embedding=None):
        h = self.in_conv(x)
        if self.mlp is not None and time_embedding is not None:
            assert self.mlp is not None, "MLP is None"
            h = h + rearrange(self.mlp(time_embedding), "b c -> b c 1 1")
        h = self.block(h)
        return h + self.residual_conv(x)
    

class DownSample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        self.net = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            nn.Conv2d(dim * 4, default(dim_out, dim), 1),
        )

    def forward(self, x):
        return self.net(x)


class UpsampleC(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(dim, dim_out or dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)
    
class TwoResUNet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        sinusoidal_pos_emb_theta=10000,
        convnext_block_groups=8,
    ):
        super().__init__()
        self.channels = channels
        input_channels = channels
        self.init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, self.init_dim, 7, padding=3)

        dims = [self.init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)

        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(
                            in_channels=dim_in,
                            out_channels=dim_in,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        ConvNextBlock(
                            in_channels=dim_in,
                            out_channels=dim_in,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        DownSample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_embedding_dim=time_dim)
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_embedding_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            is_first = ind == 0

            self.ups.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(
                            in_channels=dim_out + dim_in,
                            out_channels=dim_out,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        ConvNextBlock(
                            in_channels=dim_out + dim_in,
                            out_channels=dim_out,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        UpsampleC(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                    ]
                )
            )

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ConvNextBlock(dim * 2, dim, time_embedding_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time,  x_self_cond = None):
        # b, _, h, w = x.shape

        if x_self_cond is not None:
            x = torch.cat((x_self_cond, x), dim = 1)
            
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        unet_stack = []
        for down1, down2, downsample in self.downs:
            x = down1(x, t)
            unet_stack.append(x)
            x = down2(x, t)
            unet_stack.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for up1, up2, upsample in self.ups:
            x = torch.cat((x, unet_stack.pop()), dim=1)
            x = up1(x, t)
            x = torch.cat((x, unet_stack.pop()), dim=1)
            x = up2(x, t)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)

        return self.final_conv(x)

class BlockAttention(nn.Module):
    def __init__(self, gate_in_channel, residual_in_channel, scale_factor):
        super().__init__()
        self.gate_conv = nn.Conv2d(gate_in_channel, gate_in_channel, kernel_size=1, stride=1)
        self.residual_conv = nn.Conv2d(residual_in_channel, gate_in_channel, kernel_size=1, stride=1)
        self.in_conv = nn.Conv2d(gate_in_channel, 1, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        in_attention = self.relu(self.gate_conv(g) + self.residual_conv(x))
        in_attention = self.in_conv(in_attention)
        in_attention = self.sigmoid(in_attention)
        return in_attention * x

class Forward_diffussion_process():
    def __init__(self, args, config, model,  optimizer, scheduler, loss, eta=0):
        self.start = config.beta_start
        self.end = config.beta_end
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.timesteps = config.timesteps
        self.device = config.device
        self.loss_fn = loss
        self.eta = 0.
        self._set_alphas_betas(args)
        self._set_post_variance()
        self.model = model

    def _set_schedule(self, args, timesteps):
        if args.diff_schedule == "linear":
            return self.linear_beta_schedule(timesteps)
        elif args.diff_schedule == "quadratic":
            return self.quadratic_beta_schedule(timesteps)
        elif args.diff_schedule == "sigmoid":
            return self.sigmoid_beta_schedule(timesteps)
        elif args.diff_schedule == "cosine":
            return self.cosine_beta_schedule(timesteps)
    
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = torch.gather(a, index=t, dim=0)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).float()
    # pre calculate values of beta, underroot alpha_bar and 1-alpha_bar for all timesteps

    def linear_beta_schedule(self, timesteps):
        return torch.linspace(self.start, self.end, timesteps, 
                            dtype=torch.float32,
                            device=self.device)
    
    def quadratic_beta_schedule(self, timesteps):
        return torch.linspace(self.start**0.5, self.end**0.5, timesteps,
                            dtype=torch.float32,
                            device=self.device) ** 2
    
    def sigmoid_beta_schedule(self, timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
        """
        sigmoid schedule
        proposed in https://arxiv.org/abs/2212.11972 - Figure 8
        better for images > 64x64, when used during training
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alpha_bars = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alpha_bars = alpha_bars / alpha_bars[0]
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def cosine_beta_schedule(self, timesteps, s = 0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def _set_alphas_betas(self, args):
        self.betas = self._set_schedule(args, timesteps=self.timesteps).to(self.device) #diffusion rates
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(self.device)   # alpha_bar  = alpha1 * alpha2 * ... * alphat
        self.signal_rate = torch.sqrt(self.alpha_bars).to(self.device) 
        self.noise_rate = torch.sqrt(1 - self.alpha_bars).to(self.device)

    # forward diffusion (using the nice property)
    def forward_process(self, x_start, t, noise=None):
        r""" Adds noise to a batch of original images at time-step t.
        
        :param x_start: Input Image Tensor
        :param noise: Random Noise Tensor sampled from Normal Dist N(0, 1)
        :param t: timestep of the forward process of shape -> (B, )
        
        Note: time-step t may differ for each image inside the batch.
        
        """
        if noise is None:
            noise = torch.randn_like(x_start, dtype=torch.float32).to(self.device)

        sqrt_alphas_cumprod_t = self._extract(self.signal_rate, t,
                                        x_start.shape).to(self.device)  # Read as, extract values at index t from sqrt_alpha_bars and reshape it to match x_start.shape
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.noise_rate, t, x_start.shape
        ).to(self.device)

        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t.float().to(self.device)


    def get_noisy_image(self, x_start, t):
        # sample a new noisy image
        x_noisy = self.forward_process(x_start, t=t)  # This is a tensor

        # # turn back into PIL image
        # reverse_transform = get_tensor_to_image_transform()
        # noisy_image = reverse_transform(x_noisy.squeeze())

        return x_noisy
    
    def _set_post_variance(self):
        self.coeff_1 = torch.sqrt(1.0 / self.alphas).to(self.device).to(self.device)
        self.alphas_cumprod_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0).to(self.device)
        self.coeff_2 = (self.coeff_1 * (1.0 - self.alphas) / torch.sqrt(1.0 - self.alpha_bars)).to(self.device)
        self.posterior_variance = (self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alpha_bars)).to(self.device) #equation 7
    
    @torch.no_grad()
    def p_sample_ddpm(self, model, x_t, t:int, x_cond=None, clip=True):

        r""" Sample x_(t-1) given x_t and noise predicted
             by model.
             
             :param xt: Image tensor at timestep t of shape -> B x C x H x W
             :param noise_pred: Noise Predicted by model of shape -> B x C x H x W
             :param t: Current time step

        """
        b = x_t.shape[0]

        batched_times = torch.full((b,), t, device=self.device, dtype=torch.long)
        epsilon_theta = model(x_t, batched_times, x_cond)

        mean = self._extract(self.coeff_1, batched_times, x_t.shape) * x_t - self._extract(self.coeff_2, batched_times, x_t.shape) * epsilon_theta
        # var is a constant
        var = self._extract(self.posterior_variance, batched_times, x_t.shape)

        z = torch.randn_like(x_t) if t > 0 else 0
        x_t_minus_one = mean + torch.sqrt(var) * z

        if clip is True:
            x_t_minus_one = torch.clip(x_t_minus_one, -1, 1)
        return x_t_minus_one
    
    @torch.no_grad()
    def p_sample_ddim(self, model, x_t, t:int, x_cond=None, clip=True):

        r""" Sample x_(t-1) given x_t and noise predicted
             by model.
             
             :param xt: Image tensor at timestep t of shape -> B x C x H x W
             :param noise_pred: Noise Predicted by model of shape -> B x C x H x W
             :param t: Current time step

        """
        epsilon_t = torch.randn_like(x_t)

        b = x_t.shape[0]
        prev_time_step = (t-1)

        batched_times = torch.full((b,), t, device=self.device, dtype=torch.long)
        batched_prev_t =  torch.full((b,), prev_time_step, device=self.device, dtype=torch.long)

        # get current and previous alpha_cumprod
        alpha_t = self._extract(self.alpha_bars, batched_times, x_t.shape)
        alpha_t_prev = self._extract(self.alpha_bars, batched_prev_t, x_t.shape)

        epsilon_theta_t = model(x_t, batched_times, x_cond)
        sigma_t = self.eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        
        # Original Image Prediction at timestep t
        x_t_minus_one = (
                torch.sqrt(alpha_t_prev / alpha_t) * x_t +
                (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                    (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
                sigma_t * epsilon_t
        )
        if clip is True:
            x_t_minus_one = torch.clip(x_t_minus_one, -1, 1)
        return x_t_minus_one

        # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def p_sample(self, args, model, x_start, time_steps, x_cond):
        if args.diff_sample == "ddpm":
            return self.p_sample_ddpm( model, x_start, time_steps, x_cond)
        elif args.diff_sample =="ddim":
            return self.p_sample_ddim( model, x_start, time_steps, x_cond)
        
    @torch.no_grad()
    def reverse_diffusion(self, args, model, dataloader, shape, timesteps, time_steps_list):
        import random
        idx = random.randint(0, len(dataloader.data)-1)
        x_cond = torch.from_numpy(dataloader.data[idx]).unsqueeze(0).to(self.device)
        y_true = torch.from_numpy(dataloader.labels[idx]).unsqueeze(0).to(self.device)
         # generate Gaussian noise
        z_t = torch.randn(shape, device=self.device, dtype=torch.float32)
        imgs = [z_t]
        x_start = None
        with tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps) as sampling_steps:
            for t in sampling_steps:
                x_start = z_t if x_start is None else x_start
                x_start = self.p_sample(args, model, x_start, time_steps_list[t], x_cond)
                imgs.append(x_start)
    
        return imgs[-1], y_true

    @torch.no_grad()
    def p_sample_loop(self, args, model, dataloader, shape, timesteps, samples):
        
        if args.diff_sample == "ddim":
            a = self.timesteps // timesteps
            time_steps = np.asarray(list(range(0, self.timesteps, a))) + 1
        
        elif args.diff_sample == "ddpm":
            timesteps = self.timesteps
            time_steps = np.asarray(list(range(0, self.timesteps, 1)))

        imgs = torch.empty(shape)
        y = torch.empty(shape)
        img_shape = (1, *shape[1:])
        for s in range(samples):
            img,  y_true = self.reverse_diffusion(args, model, dataloader, img_shape, timesteps, time_steps)
            imgs[s] = img.squeeze()
            y[s] = y_true.squeeze()
        return imgs, y 

    @torch.no_grad()
    def sample(self, args, model, dataloader, shape, timesteps=300, samples=1):
        return self.p_sample_loop(args, model, dataloader, shape, timesteps, samples)


'''

The network is built up as follows:

-> first, a convolutional layer is applied on the batch of noisy images, and position embeddings are computed for the noise levels
-> next, a sequence of downsampling stages are applied. Each downsampling stage consists of
    2 ResNet blocks + groupnorm + attention + residual connection + a downsample operation
-> at the middle of the network, again ResNet blocks are applied, interleaved with attention
-> next, a sequence of upsampling stages are applied. Each upsampling stage consists of
    2 ResNet blocks + groupnorm + attention + residual connection + an upsample operation
-> finally, a ResNet block followed by a convolutional layer is applied.
'''
from functools import partial
from inspect import isfunction
import torch
from torch import nn
from einops import reduce, rearrange
from einops.layers.torch import Rearrange
from torch import nn, einsum
from torch.nn import Module

class PreNorm(nn.Module):
    def __init__(self, channels, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        x = self.norm(x)
        x = self.fn(x)

        return x


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# conv layer with weight standardisation incorporated
class WeightStandardizedConv2d(nn.Conv2d):
    # no need to override init, we will just override forward

    def forward(self, x):
        # eps is added to denominator to prevent division by zero
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=True))

        weight = (weight - mean) / (var + eps).rsqrt()

        x = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )

# Most basic block with conv + groupNorm + Silu activation
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


# The residual block
class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
    

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
    
def getSequnceOfDimensions(init_dim, dim, dim_mults):
    dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
    in_out = list(zip(dims[:-1], dims[1:]))
    return in_out, dims

class UNET(nn.Module):
    def __init__(self,
                 dim,
                 init_dim=None,
                 out_dim=None,      # channels of final output image
                 dim_mults=(1, 2, 4, 8),
                 channels=3,
                 resnet_block_groups=4, ):
        super().__init__()

        self.channels = channels  # channels in input image
        input_channels = channels

        init_dim = default(init_dim, dim)

        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        in_out, dims = getSequnceOfDimensions(init_dim, dim, dim_mults)
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)  # depth of U-net

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),  # input x; apply group norm, then attention to x, then add x to it.
                        Downsample(dim_in, dim_out) if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )
        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):

        if x_self_cond is not None:
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []  # To store bridge connections

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)  # concat along dim = 1 i.e. channels dimension
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)


        x = self.final_res_block(x, t)
        return self.final_conv(x)