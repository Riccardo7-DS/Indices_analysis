from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import numpy as np
import time

import torch
import torch.nn as nn

class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.num_features = num_features
        self.conv = self._make_layer(in_channels+num_features, num_features*4,
                                       kernel_size, padding, stride)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, inputs):
        '''

        :param inputs: (B, S, C, H, W)
        :param hidden_state: (hx: (B, S, C, H, W), cx: (B, S, C, H, W))
        :return:
        '''
        outputs = []
        B, S, C, H, W = inputs.shape
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        for t in range(S):
            combined = torch.cat([inputs[:, t], # (B, C, H, W)
                                  hx], dim=1)
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        return torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous() # (S, B, C, H, W) -> (B, S, C, H, W)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = []
        for idx, params in enumerate(config.encoder):
            setattr(self, params[0]+'_'+str(idx), self._make_layer(*params))
            self.layers.append(params[0]+'_'+str(idx))

    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size, padding, stride):
        layers = []
        if type == 'conv':
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
            elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
        elif type == 'convlstm':
            layers.append(ConvLSTMBlock(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''
        :param x: (B, S, C, H, W)
        :return:
        '''
        outputs = [x]
        for layer in self.layers:
            if 'conv_' in layer:
                B, S, C, H, W = x.shape
                x = x.view(B*S, C, H, W)
            x = getattr(self, layer)(x)
            if 'conv_' in layer: x = x.view(B, S, x.shape[1], x.shape[2], x.shape[3])
            if 'convlstm' in layer: outputs.append(x)
        return outputs

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = []
        self.config = config
        for idx, params in enumerate(config.decoder):
            setattr(self, params[0]+'_'+str(idx), self._make_layer(*params))
            self.layers.append(params[0]+'_'+str(idx))

    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size, padding, stride):
        layers = []
        if type == 'conv':
            if self.config.decoder_3d is True:
                layers.append(nn.Conv3d(in_ch, out_ch, kernel_size=(1,3,3), padding=(0,1,1)))
                layers.append(nn.BatchNorm3d(out_ch))
            else:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
                layers.append(nn.BatchNorm2d(out_ch))
            if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
            elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
            elif activation == 'sigmoid': layers.append(nn.Sigmoid())
        elif type == 'convlstm':
            layers.append(ConvLSTMBlock(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride))
        elif type == 'deconv':
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
            elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
            elif activation == "linear": layers.append(nn.Linear(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, encoder_outputs):
        '''
        :param x: (B, S, C, H, W)
        :return:
        '''
        idx = len(encoder_outputs)-1
        for layer in self.layers:
            if layer.startswith('conv_'):
                if self.config.decoder_3d is True:
                    x = encoder_outputs[idx]
                    B, S, C, H, W = x.shape
                    x = x.permute(0, 2, 1, 3, 4)
                    x = getattr(self, layer)(x)
                    x = x.permute(0, 2, 1, 3, 4)
                else:
                    x = encoder_outputs[idx]
                    B, S, C, H, W = x.shape
                    x = x.view(B*S, C, H, W)
                    x = getattr(self, layer)(x)
                    x = x.view(B, S, x.shape[1], x.shape[2], x.shape[3])

            elif 'deconv_' in layer: #'
                x = encoder_outputs[idx]
                B, S, C, H, W = x.shape
                x = x.view(B*S, C, H, W)
                x = getattr(self, layer)(x)
                x = x.view(B, S, x.shape[1], x.shape[2], x.shape[3])

            elif 'convlstm' in layer:
                idx -= 1
                x = torch.cat([encoder_outputs[idx], x], dim=2)
                x = getattr(self, layer)(x)
                encoder_outputs[idx] = x
        return x

class ConvLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_loop(config, args, logger, epoch, model, train_loader, criterion, 
               optimizer, scaler=None, mask=None, draw_scatter:bool=False):
    from analysis.deep_learning.GWNET.pipeline_gwnet import masked_mse_loss, mask_mape, mask_rmse, MetricsRecorder, create_paths

    output_dir, log_path, img_path, checkpoint_dir= create_paths(args)
    
    #from torcheval.metrics import R2Score
    model.train()
    epoch_records = {'loss': [], "mape":[], "rmse":[], "r2":[]}
    #metric = R2Score()

    if draw_scatter is True:
        nbins= 200
        bin0 = np.linspace(0, config.max_value, nbins+1)
        n = np.zeros((nbins,nbins))

    num_batchs = len(train_loader)

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs = inputs.float().to(config.device)
        targets = torch.squeeze(targets.float().to(config.device))
        outputs = torch.squeeze(model(inputs))

        # num_dimensions = outputs.dim()

        # if num_dimensions==4:
        #     outputs = outputs[:,-1,:,:]
        # elif num_dimensions == 3:
        #     outputs = outputs[-1, :, :]

        if draw_scatter is True:
            img_pred = outputs.detach().cpu().numpy()
            img_real = targets.detach().cpu().numpy()
            if args.normalize is True:
                img_pred = scaler.inverse_transform(img_pred)
                img_real = scaler.inverse_transform(img_real)
            h, xed, yed = evaluate_hist2d(img_real, img_pred, nbins)
            n = n+h

        #print("Output shape:", outputs.shape)
        if config.masked_loss is False:
            losses = criterion(outputs, targets)
        else:
            if mask is None:
                raise ValueError("Please provide a mask for loss computation")
            else:
                mask = mask.float().to(config.device)
                losses = masked_mse_loss(criterion, outputs, targets, mask)

        mape = mask_mape(outputs,targets, mask).item()
        rmse = mask_rmse(outputs,targets, mask).item()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_records['loss'].append(losses.item())
        epoch_records["rmse"].append(rmse)
        epoch_records["mape"].append(mape)
        if batch_idx and batch_idx % config.display == 0:
            logger.info('EP:{:03d}\tBI:{:05d}/{:05d}\tLoss:{:.6f}({:.6f})'\
                        .format(epoch, batch_idx, num_batchs,
                                epoch_records['loss'][-1], np.mean(epoch_records['loss'])))

    if draw_scatter is True:
        plot_scatter_hist(n,  bin0, img_path)

    return epoch_records


def evaluate_hist2d(real_img, pred_img, nbins):
    mdata=np.isnan(real_img)==0
    h, xed,yed=np.histogram2d(real_img[mdata], 
                pred_img[mdata], bins=nbins, density=None, weights=None)
    
    return h, xed, yed
    
def plot_scatter_hist(n, bin0, path=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import os

    n[n<=0]=np.nan
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    #ax = axes.ravel()
    #a0=ax[0].imshow(np.log10(n),origin='lower')
    a0=ax.pcolor(bin0,bin0,n,norm=LogNorm(vmin=1, vmax=np.nanmax(n)))
    plt.colorbar(a0)
    #plt.show()
    #plt.pause(3)
    #plt.close()
    if path is not None:
        name = "scatterplot.png"
        plt.savefig(os.path.join(path,name))


def valid_loop(config, args, logger, epoch, model, valid_loader, criterion, 
               scaler=None, mask=None, draw_scatter:bool=False):
    
    from analysis.deep_learning.GWNET.pipeline_gwnet import masked_mse_loss, mask_mape, mask_rmse, masked_mse, MetricsRecorder, create_paths
    output_dir, log_path, img_path, checkpoint_dir= create_paths(args)

    model.eval()
    epoch_records = {'loss': [], "mape":[], "rmse":[]}

    if draw_scatter is True:
        nbins= 200
        bin0 = np.linspace(0, config.max_value, nbins+1)
        n = np.zeros((nbins,nbins))

    num_batchs = len(valid_loader)
    for batch_idx, (inputs, targets) in enumerate(valid_loader):
        with torch.no_grad():
            inputs = inputs.float().to(config.device)
            #print(inputs.shape)
            targets = torch.squeeze(targets.float().to(config.device))
            outputs = torch.squeeze(model(inputs))

            # num_dimensions = outputs.dim()
            # if num_dimensions==4:
            #     outputs = outputs[:,-1,:,:]
            # elif num_dimensions == 3:
            #     outputs = outputs[-1, :, :]

            if draw_scatter is True:
                img_pred = outputs.cpu().detach().numpy().flatten()
                img_real = targets.cpu().detach().numpy().flatten()
                if args.normalize is True:
                    img_pred = scaler.inverse_transform(img_pred)
                    img_real = scaler.inverse_transform(img_real)
                h, xed, yed = evaluate_hist2d(img_real, img_pred, nbins)
                n = n+h

            if config.masked_loss is False:
                losses = criterion(outputs, targets)
            else:
                if mask is None:
                    raise ValueError("Please provide a mask for loss computation")
                else:
                    mask = mask.float().to(config.device)
                    losses = masked_mse_loss(criterion, outputs, targets, mask)

            
            mape = mask_mape(outputs,targets, mask).item()
            rmse = mask_rmse(outputs,targets, mask).item()

            epoch_records['loss'].append(losses.item())
            epoch_records["rmse"].append(rmse)
            epoch_records["mape"].append(mape)
            if batch_idx and batch_idx % config.display == 0:
                logger.info('[V] EP:{:03d}\tBI:{:05d}/{:05d}\tLoss:{:.6f}({:.6f})'\
                            .format(epoch, batch_idx, num_batchs,
                                    epoch_records['loss'][-1], np.mean(epoch_records['loss'])))
    if draw_scatter is True:
        plot_scatter_hist(n,  bin0, img_path)
    
    return epoch_records


def build_logging(config):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(config.log_dir, 
                                              time.strftime("%Y%d%m_%H%M") + '.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging