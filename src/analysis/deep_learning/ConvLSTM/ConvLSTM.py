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
        for idx, params in enumerate(config.decoder):
            setattr(self, params[0]+'_'+str(idx), self._make_layer(*params))
            self.layers.append(params[0]+'_'+str(idx))

    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size, padding, stride):
        layers = []
        if type == 'conv':
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
        return nn.Sequential(*layers)

    def forward(self, encoder_outputs):
        '''
        :param x: (B, S, C, H, W)
        :return:
        '''
        idx = len(encoder_outputs)-1
        for layer in self.layers:
            if 'conv_' in layer or 'deconv_' in layer: #'
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


def train_loop(config, logger, epoch, model, train_loader, criterion, optimizer):
    from analysis.deep_learning.GWNET.pipeline_gwnet import masked_mae, masked_mape, masked_rmse, masked_mse, MetricsRecorder
    #from torcheval.metrics import R2Score
    model.train()
    epoch_records = {'loss': [], "mape":[], "rmse":[], "r2":[]}
    #metric = R2Score()
    num_batchs = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(inputs.max())
        inputs = inputs.float().to(config.device)
        targets = targets.float().to(config.device)
        outputs = model(inputs)
        #metric.update(inputs, outputs)
        print("Output shape:", outputs.shape)
        losses = criterion(outputs, targets)
        mape = masked_mape(outputs,targets,0.0).item()
        rmse = masked_rmse(outputs,targets,0.0).item()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_records['loss'].append(losses.item())
        epoch_records["rmse"].append(rmse)
        epoch_records["mape"].append(mape)
        if batch_idx and batch_idx % config.display == 0:
            logger.info('EP:{:03d}\tBI:{:05d}/{:05d}\tLoss:{:.6f}({:.6f})'.format(epoch, batch_idx, num_batchs,
                                                                                epoch_records['loss'][-1], np.mean(epoch_records['loss'])))
    return epoch_records

def valid_loop(config, logger, epoch, model, valid_loader, criterion):
    from analysis.deep_learning.GWNET.pipeline_gwnet import masked_mae, masked_mape, masked_rmse, masked_mse, MetricsRecorder

    model.eval()
    epoch_records = {'loss': [], "mape":[], "rmse":[]}
    num_batchs = len(valid_loader)
    for batch_idx, (inputs, targets) in enumerate(valid_loader):
        with torch.no_grad():
            inputs = inputs.float().to(config.device)
            targets = targets.float().to(config.device)
            outputs = model(inputs)
            mape = masked_mape(outputs,targets,0.0).item()
            rmse = masked_rmse(outputs,targets,0.0).item()
            losses = criterion(outputs, targets)
            epoch_records['loss'].append(losses.item())
            epoch_records["rmse"].append(rmse)
            epoch_records["mape"].append(mape)
            if batch_idx and batch_idx % config.display == 0:
                logger.info('[V] EP:{:03d}\tBI:{:05d}/{:05d}\tLoss:{:.6f}({:.6f})'.format(epoch, batch_idx, num_batchs,
                                                                                    epoch_records['loss'][-1], np.mean(epoch_records['loss'])))
    return epoch_records



def build_logging(config):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(config.log_dir, time.strftime("%Y%d%m_%H%M") + '.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging