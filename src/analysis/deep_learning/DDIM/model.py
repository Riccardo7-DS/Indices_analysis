import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt

image_size = 64
num_frames = 4

# sampling

min_signal_rate = 0.015
max_signal_rate = 0.95

# architecture

embedding_dims = 64 # 32
embedding_max_frequency = 1000.0
#widths = [32, 64, 96, 128]
widths = [64, 128, 256, 384]
block_depth = 2

# optimization

batch_size =  2
ema = 0.999

# Assuming you have a similar setup.py for configurations
# from setup import *

class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dims, embedding_max_frequency):
        super(SinusoidalEmbedding, self).__init__()
        self.embedding_dims = embedding_dims
        self.embedding_max_frequency = embedding_max_frequency
        self.embedding_min_frequency = 1.0

    def forward(self, x):
        frequencies = torch.exp(
            torch.linspace(
                math.log(self.embedding_min_frequency),
                math.log(self.embedding_max_frequency),
                self.embedding_dims // 2,
            )
        ).to(x.device)

        angular_speeds = 2.0 * math.pi * frequencies
        embeddings = torch.cat([torch.sin(angular_speeds * x), torch.cos(angular_speeds * x)], dim=3)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, width):
        super(ResidualBlock, self).__init__()
        self.width = width

    def forward(self, x):
        input_width = x.shape[3]
        if input_width == self.width:
            residual = x
        else:
            residual = nn.Conv2d(input_width, self.width, kernel_size=1).to(x.device)(x)

        x = nn.LayerNorm(x.size()[1:], elementwise_affine=True).to(x.device)(x)
        x = nn.Conv2d(input_width, self.width, kernel_size=3, padding="same")(x)
        x = nn.SiLU()(x)
        x = nn.Conv2d(self.width, self.width, kernel_size=3, padding="same")(x)
        x = x + residual
        return x

class DownBlock(nn.Module):
    def __init__(self, width, block_depth):
        super(DownBlock, self).__init__()
        self.width = width
        self.block_depth = block_depth

    def forward(self, x, skips):
        for _ in range(self.block_depth):
            x = ResidualBlock(self.width)(x)
            skips.append(x)
        x = nn.AvgPool2d(kernel_size=2)(x)
        return x, skips

class UpBlock(nn.Module):
    def __init__(self, width, block_depth):
        super(UpBlock, self).__init__()
        self.width = width
        self.block_depth = block_depth

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        for _ in range(self.block_depth):
            x = torch.cat([x, skips.pop()], dim=1)
            x = ResidualBlock(self.width)(x)
        return x

class UNet(nn.Module):
    def __init__(self, image_size, input_frames, output_frames, widths, block_depth):
        super(UNet, self).__init__()
        self.noisy_images_conv = nn.Conv2d(input_frames + output_frames, widths[0], kernel_size=1)
        self.embedding = SinusoidalEmbedding(embedding_dims=widths[0], embedding_max_frequency=image_size)
        self.embedding_up = nn.Upsample(size=image_size, mode='nearest')

        self.down_blocks = nn.ModuleList()
        for width in widths[:-1]:
            self.down_blocks.append(DownBlock(width, block_depth))

        self.mid_blocks = nn.ModuleList([ResidualBlock(widths[-1]) for _ in range(block_depth)])

        self.up_blocks = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.up_blocks.append(UpBlock(width, block_depth))

        self.final_conv = nn.Conv2d(widths[0], output_frames, kernel_size=1)

    def forward(self, noisy_images, noise_variances):
        e = self.embedding(noise_variances)
        e = self.embedding_up(e)
        x = self.noisy_images_conv(noisy_images)
        x = torch.cat([x, e], dim=1)

        skips = []
        for down_block in self.down_blocks:
            x, skips = down_block(x, skips)

        for mid_block in self.mid_blocks:
            x = mid_block(x)

        for up_block in self.up_blocks:
            x = up_block(x, skips)

        x = self.final_conv(x)
        return x

class DiffusionModel(nn.Module):
    def __init__(self, image_size, input_frames, output_frames, widths, block_depth):
        super(DiffusionModel, self).__init__()
        # self.normalizer = nn.LayerNorm(image_size)
        self.network = UNet(image_size, input_frames, output_frames, widths, block_depth)
        self.ema_network = UNet(image_size, input_frames, output_frames, widths, block_depth)  # Assuming a deep copy here
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.ema = 0.999

    def diffusion_schedule(self, diffusion_times):
        start_angle = torch.acos(torch.tensor(max_signal_rate))
        end_angle = torch.acos(torch.tensor(min_signal_rate))
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        signal_rates = torch.cos(diffusion_angles)
        noise_rates = torch.sin(diffusion_angles)
        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        network = self.network if training else self.ema_network
        pred_noises = network(noisy_images, noise_rates**2)
        pred_images = (noisy_images[:,:,:,-self.output_frames:] - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        past = initial_noise[:,:,:,:-self.output_frames]
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images
            diffusion_times = torch.ones((num_images, 1, 1, 1)).to(initial_noise.device) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            next_noisy_frames = next_signal_rates * pred_images + next_noise_rates * pred_noises
            next_noisy_images = torch.cat([past, next_noisy_frames], dim=-1)
        return pred_images

    def generate(self, images, diffusion_steps):
        initial_noise = torch.randn((images.shape[0], images.shape[1], images.shape[2], self.output_frames)).to(images.device)
        images[:,:,:,-self.output_frames:] = initial_noise
        generated_images = self.reverse_diffusion(images, diffusion_steps)
        return self.denormalize(torch.cat([images[:,:,:,:-self.output_frames], generated_images], dim=-1))

    def denormalize(self, images):
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return torch.clamp(images, 0.0, 1.0)

    def train_step(self, images, optimizer, loss_fn, batch_size, image_size, max_signal_rate, min_signal_rate):
        # images = self.normalizer(images)
        noises = torch.randn((batch_size, image_size, 
                              image_size, self.output_frames)).to(images.device)
        diffusion_times = torch.rand((batch_size, 1, 1, 1)).to(images.device)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        target = images[:,:,:,-self.output_frames:]
        noisy_images = signal_rates * target + noise_rates * noises
        noise_two = torch.cat([images[:,:,:,:-self.output_frames], noisy_images], dim=-1)

        optimizer.zero_grad()
        pred_noises, pred_images = self.denoise(noise_two, noise_rates, signal_rates, training=True)
        noise_loss = loss_fn(noises, pred_noises)
        image_loss = loss_fn(target, pred_images)
        noise_loss.backward()
        optimizer.step()

        for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
            ema_param.data = self.ema * ema_param.data + (1.0 - self.ema) * param.data

        return noise_loss.item(), image_loss.item()

    def test_step(self, images, batch_size, image_size, loss_fn):
        # images = self.normalizer(images)
        noises = torch.randn((batch_size, image_size, image_size, self.output_frames)).to(images.device)
        diffusion_times = torch.rand((batch_size, 1, 1, 1)).to(images.device)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        target = images[:,:,:,-self.output_frames:]
        noisy_images = signal_rates * target + noise_rates * noises
        noise_two = torch.cat([images[:,:,:,:-self.output_frames], noisy_images], dim=-1)
        pred_noises, pred_images = self.denoise(noise_two, noise_rates, signal_rates, training=False)
        noise_loss = loss_fn(noises, pred_noises)
        image_loss = loss_fn(target, pred_images)
        return noise_loss.item(), image_loss
