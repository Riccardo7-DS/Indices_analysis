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
        embeddings = torch.cat([torch.sin(angular_speeds * x), 
                                torch.cos(angular_speeds * x)], dim=3)
        return embeddings.permute(0, 3, 1, 2)

class ResidualBlock(nn.Module):
    def __init__(self, width):
        super(ResidualBlock, self).__init__()
        self.width = width

    def forward(self, x):
        input_width = x.shape[1]  ### Channels
        if input_width == self.width:
            residual = x
        else:
            residual = nn.Conv2d(input_width, self.width, kernel_size=1).to(x.device)(x)

        x = nn.LayerNorm(x.size()[1:], elementwise_affine=True).to(x.device)(x)
        x = nn.Conv2d(input_width, self.width, kernel_size=3, padding="same").to(x.device)(x)
        x = nn.SiLU()(x)
        x = nn.Conv2d(self.width, self.width, kernel_size=3, padding="same").to(x.device)(x)
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
        skip = skips.pop()
        
        # Ensuring the spatial dimensions match
        if x.shape[2:] != skip.shape[2:]:
            # Cropping the skip connection if necessary
            height_diff = skip.shape[2] - x.shape[2]
            width_diff = skip.shape[3] - x.shape[3]

            if height_diff != 0:
                skip = skip[:, :, :-height_diff, :] if height_diff > 0 else F.pad(skip, (0, 0, 0, -height_diff))
            if width_diff != 0:
                skip = skip[:, :, :, :-width_diff] if width_diff > 0 else F.pad(skip, (0, -width_diff, 0, 0))

        x = torch.cat([x, skip], dim=1)
        for _ in range(self.block_depth):
            x = ResidualBlock(self.width)(x)
        return x

class UNet(nn.Module):
    def __init__(self, config, input_frames, output_frames, widths, block_depth):
        super(UNet, self).__init__()

        self.noisy_images_conv = nn.Conv2d(input_frames+output_frames, config.widths[0], kernel_size=1)

        self.down_blocks = nn.ModuleList()
        for width in widths[:-1]:
            self.down_blocks.append(DownBlock(width, block_depth))

        self.mid_blocks = nn.ModuleList([ResidualBlock(widths[-1]) for _ in range(block_depth)])

        self.up_blocks = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.up_blocks.append(UpBlock(width, block_depth))

        self.upsample = nn.Upsample(size=config.input_size, mode='bilinear', align_corners=True)

        self.final_conv = nn.Conv2d(widths[0], output_frames, kernel_size=1)
        nn.init.zeros_(self.final_conv.weight)

    def forward(self, images, embedded_variance):
        x = self.noisy_images_conv(images)
        x = torch.cat([x, embedded_variance], dim=1)

        skips = []
        for down_block in self.down_blocks:
            x, skips = down_block(x, skips)

        for mid_block in self.mid_blocks:
            x = mid_block(x)

        for up_block in self.up_blocks:
            x = up_block(x, skips)

        x = self.upsample(x)

        x = self.final_conv(x)
        return x
        


class Section(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Section, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding='same'),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.process(x)
    
class UNET_rect(nn.Module):
    def __init__(self, input_frames, output_frames, config):
        super(UNET_rect, self).__init__()
        # Contraction
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = Section(in_channels=input_frames+output_frames+config.widths[0], out_channels=64, kernel_size=3) #config.widths[0]
        self.down2 = Section(in_channels=64, out_channels=128, kernel_size=3)
        self.down3 = Section(in_channels=128, out_channels=256, kernel_size=3)
        self.down4 = Section(in_channels=256, out_channels=512, kernel_size=3)
        self.down5 = Section(in_channels=512, out_channels=1024, kernel_size=3)
        # Expansion
        self.up_conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up1 = Section(in_channels=1024, out_channels=512, kernel_size=3)
        self.up2 = Section(in_channels=512, out_channels=256, kernel_size=3)
        self.up3 = Section(in_channels=256, out_channels=128, kernel_size=3)
        self.up4 = Section(in_channels=128, out_channels=64, kernel_size=3)
        self.output = self.final_conv = nn.Conv2d(64, output_frames, kernel_size=1, padding='same')
        
    def forward(self, x, embedded_variance):
        x = torch.cat([x, embedded_variance], dim=1)

        skip_connections = []
        # CONTRACTION
        # down 1
        x = self.down1(x)
        skip_connections.append(x)
        x = self.pool(x)
        
        # down 2
        x = self.down2(x)
        skip_connections.append(x)
        x = self.pool(x)
        # down 3
        x = self.down3(x)
        skip_connections.append(x)
        x = self.pool(x)
        # down 4
        x = self.down4(x)
        skip_connections.append(x)
        x = self.pool(x)
        # down 5
        x = self.down5(x)
        
        # EXPANSION
        # up1
        x = self.up_conv1(x)
        y = skip_connections[3]
        x = TF.resize(x, y.shape[2:])
        y_new = torch.cat((y, x), dim=1)
        x = self.up1(y_new)
        # up2
        x = self.up_conv2(x)
        y = skip_connections[2]
        # resize skip commention
        x = TF.resize(x, y.shape[2:])
        y_new = torch.cat((y, x), dim=1)
        x = self.up2(y_new)
        # up3
        x = self.up_conv3(x)
        y = skip_connections[1]
        x = TF.resize(x, y.shape[2:])
        y_new = torch.cat((y, x), dim=1)
        x = self.up3(y_new)
        # up4 
        x = self.up_conv4(x)
        y = skip_connections[0]
        x = TF.resize(x, y.shape[2:])
        y_new = torch.cat((y, x), dim=1)
        x = self.up4(y_new)
        
        x = self.output(x)
        return x

class DiffusionModel(nn.Module):
    def __init__(self, config, input_frames):
        super(DiffusionModel, self).__init__()
        self.image_size = config.input_size
        self.input_frames = input_frames
        self.output_frames = config.num_frames_output
        self.normalizer = nn.LayerNorm([*self.image_size])
        # self.network = UNet(config, self.input_frames, 
        #                     self.output_frames, config.widths, config.block_depth)
        self.network = UNET_rect(self.input_frames, self.output_frames, config)
        self.ema_network = UNet(config, self.input_frames, self.output_frames, 
                                config.widths, config.block_depth)  # Assuming a deep copy here
        
        self.ema = 0.999
        self.max_signal_rate = config.max_signal_rate
        self.min_signal_rate = config.min_signal_rate
        self.batch_size = config.batch_size
        self.device = config.device
        
        self.embedding = SinusoidalEmbedding(embedding_dims=config.widths[0], embedding_max_frequency=config.embedding_max_frequency) 
        self.upsample = nn.Upsample(size=config.input_size, mode='nearest')
        
    def diffusion_schedule(self, diffusion_times):
        import torch.nn.functional as F
        start_angle = torch.acos(torch.tensor(self.max_signal_rate))
        end_angle = torch.acos(torch.tensor(self.min_signal_rate))
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        signal_rates = torch.cos(diffusion_angles)
        noise_rates = torch.sin(diffusion_angles)
        return noise_rates, signal_rates
    
    def _embed_variance(self, noise_rates):
        return self.upsample(self.embedding(noise_rates**2))

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        embedded_variance = self._embed_variance(noise_rates)
    
        network = self.network if training else self.ema_network
        pred_noises = network(noisy_images, embedded_variance)
        pred_images = (noisy_images[:,-self.output_frames:,:,:] - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        past = initial_noise[:,:-self.output_frames,:,:]
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

    def generate(self, images, diffusion_steps, normalize=False):
        initial_noise = torch.randn((images.shape[0], images.shape[1], 
                                     images.shape[2], self.output_frames)).to(images.device)
        images[:,:,:,-self.output_frames:] = initial_noise
        generated_images = self.reverse_diffusion(images, diffusion_steps)
        cat_images = torch.cat([images[:,:,:,:-self.output_frames], generated_images], dim=-1)
        if normalize is True:
            return self.denormalize(cat_images)
        else:
            return cat_images

    def denormalize(self, images):
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return torch.clamp(images, 0.0, 1.0)
    

class TrainDiffusion:
    def __init__(self, model:DiffusionModel, optimizer, loss_fn):
        self.optimizer = optimizer
        self.loss = loss_fn
        self.model = model

    def train_step(self, train_loader, epoch, writer, plot:bool=False):
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            
            noises = torch.randn((self.model.batch_size, self.model.output_frames, 
                                  images.shape[2], images.shape[3]))\
                                .to(self.model.device)
            diffusion_times = torch.rand((self.model.batch_size, 1, 1, 1))\
                                .to(self.model.device)
            noise_rates, signal_rates = self.model.diffusion_schedule(diffusion_times)
            
            target = targets.squeeze(2)

            if plot is True:
                logger.info("Plotting target data")
                self.plot_data(target.squeeze(), 0, 1)

            if plot is True:
                logger.info("Plotting input data")
                self.plot_data(images, 0, images.shape[1])
            
            noisy_images = signal_rates * target + noise_rates * noises
            noise_two = torch.cat([images, noisy_images], dim=1)

            self.optimizer.zero_grad()
            
            pred_noises, pred_images = self.model.denoise(noise_two, noise_rates, 
                                                    signal_rates, training=True)
            
            noise_loss = self.loss(noises, pred_noises)
            image_loss = self.loss(target, pred_images)

            
            noise_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    writer.add_scalar(f'Gradient/{name}', param.grad.norm(), epoch * len(train_loader) + batch_idx)
                    writer.add_scalar(f'Parameter/{name}', param.norm(), epoch * len(train_loader) + batch_idx)

            self.optimizer.step()
            # for ema_param, param in zip(self.model.ema_network.parameters(), 
            #                             self.model.network.parameters()):
            #     ema_param.data = self.model.ema * ema_param.data + (1.0 - self.model.ema) * param.data
            return noise_loss.item(), image_loss.item()
        
    def test_step(self, test_loader):
        for idx, (images, targets) in enumerate(test_loader):
            with torch.no_grad():
                # images = self.normalizer(inputs)
                target = targets.squeeze(2)
                noises = torch.randn((self.model.batch_size, self.model.output_frames, 
                                      images.shape[2], images.shape[3]))\
                                    .to(self.model.device)
                diffusion_times = torch.rand((self.model.batch_size, 1, 1, 1))\
                                    .to(self.model.device)
                noise_rates, signal_rates = self.model.diffusion_schedule(diffusion_times)

                noisy_images = signal_rates * target + noise_rates * noises
                noise_two = torch.cat([images, noisy_images], dim=1)
                pred_noises, pred_images = self.model.denoise(noise_two, noise_rates, 
                                                        signal_rates, training=False)


                noise_loss = self.loss(noises, pred_noises)
                image_loss = self.loss(target, pred_images)

            return noise_loss.item(), image_loss.item()
        
    def plot_data(self, tensor, idx:int, range_images:int, seconds:int=10):
        import matplotlib.pyplot as plt
        import numpy as np


        # Select a specific slice of the first dimension to plot
        # For example, we select the first slice (tensor[0, :, :, :])
        tensor_slice = tensor[idx].cpu().detach().float()

        # Create a figure
        fig, axes = plt.subplots(5, 7, figsize=(20, 15))  # 5 rows and 7 columns of subplots
        axes = axes.flatten()  # Flatten the 2D array of axes to 1D

        if range_images > 1:
            for i in range(range_images):
                ax = axes[i]
                cax = ax.imshow(tensor_slice[i], aspect='auto')
                ax.set_title(f'Slice {i}')
                fig.colorbar(cax, ax=ax)
        else:
            for i in range(range_images):
                ax = axes[i]
                cax = ax.imshow(tensor_slice, aspect='auto')
                ax.set_title(f'Slice {i}')
                fig.colorbar(cax, ax=ax)


        # Remove any unused subplots (if there are any)
        for j in range(range_images, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.pause(seconds)
        plt.close()



class Forward_diffussion_process():
    def __init__(self, args, config, model, timesteps=500, eta=0):
        self.start = config.beta_start
        self.end = config.beta_end
        self.timesteps = timesteps
        self.device = config.device
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
        self.alpha_bars = alpha_bars / alpha_bars[0]
        betas = 1 - (self.alpha_bars[1:] / self.alpha_bars[:-1])
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
    def p_sample_loop(self, args, model, dataloader, shape, timesteps):
        import random
        if args.diff_sample == "ddim":
            a = self.timesteps // timesteps
            time_steps = np.asarray(list(range(0, self.timesteps, a))) + 1
            time_steps_prev = np.concatenate([[0], time_steps[:-1]])
        
        elif args.diff_sample == "ddpm":
            timesteps = self.timesteps
            time_steps = np.asarray(list(range(0, self.timesteps, 1)))

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
                x_start = self.p_sample(args, model, x_start, time_steps[t], x_cond)
                imgs.append(x_start)
        
        return imgs[-1], y_true

    @torch.no_grad()
    def sample(self, args, model, dataloader, shape, timesteps=300):
        return self.p_sample_loop(args, model, dataloader, shape, timesteps)


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