# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import yaml
import torch
from definitions import ROOT_DIR

class ConfigDDIM:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 64
    input_size = (83, 77)
    num_frames = 1
    output_channels = 1
    num_frames_output = 1

    scheduler_patience = 10
    scheduler_factor = 0.1

    # sampling

    min_signal_rate = 0.015
    max_signal_rate = 0.95

    # architecture

    embedding_dims = 64 # 32
    embedding_max_frequency = 1000.0
    widths = [32, 64, 96, 128]
    # widths = [64, 128, 256, 384]
    block_depth = 2

    # optimization

    batch_size =  128
    ema = 0.999
    learning_rate = 1e-3
    epochs = 500
    patience = 100

    include_lag = True
    
    data_dir = os.path.join(ROOT_DIR, "..", 'data')
    output_dir = os.path.join(ROOT_DIR,  "..", 'output')
    log_dir = os.path.join(output_dir, 'log')
    model_dir = os.path.join(output_dir, 'model')

    autoencoder_path = output_dir + "/dime/days_15/features_90/autoencoder/checkpoints/checkpoint_epoch_63.pth.tar"
    

    for path in [data_dir, output_dir, log_dir, model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

class ConfigConvLSTM:

    gpus = [0, ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    masked_loss = True

    num_frames_output = 1 ### means 1 frame as output
    output_channels = 1
    include_lag = True
    num_samples = 9 #### the number of channels to use

    epochs = 100
    patience = 20
    learning_rate = 1e-3
    batch_size= 8

    null_value = -1
    max_value = 1

    scheduler_patience = 3
    scheduler_factor = 0.7

    image_size = (64, 64)
    input_size = (64, 64)
    

    # Parameters specific to ConvLSTM
    # dim = 64
    layers = [32, 32, 32]

    # (type, activation, in_ch, out_ch, kernel_size, padding, stride)
    encoder = [('conv', 'leaky', num_samples, 16, 3, 1, 2),
             ('convlstm', '', 16, 16, 3, 1, 1),
             ('conv', 'leaky', 16, 32, 3, 1, 2),
             ('convlstm', '', 32, 32, 3, 1, 1),
             ('conv', 'leaky', 32, 64, 3, 1, 2),
             ('convlstm', '', 64, 64, 3, 1, 1)]
    decoder = [('deconv', 'leaky', 64, 32, 4, 1, 2),
               ('convlstm', '', 64, 32, 3, 1, 1),
               ('deconv', 'leaky', 32, 16, 4, 1, 2),
               ('convlstm', '', 32, 16, 3, 1, 1),
               ('deconv', 'leaky', 16, 16, 4, 1, 2),
               ('convlstm', '', 16+num_samples, 16, 3, 1, 1),
               ('conv', 'sigmoid', 16, 1, 1, 0, 1)]

    data_dir = os.path.join(ROOT_DIR, "..", 'data')
    output_dir = os.path.join(ROOT_DIR,  "..", 'output')
    log_dir = os.path.join(output_dir, 'log')
    model_dir = os.path.join(output_dir, 'model')

    for path in [data_dir, output_dir, log_dir, model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


class ConfigGWNET():

    num_frames_output = 1 ### means 1 frame as output
    output_channels = 1
    include_lag = True
    num_samples = 9 #### the number of channels to use

    epochs = 300
    patience = 20
    learning_rate = 1e-3
    batch_size= 4
    dim = 64

    null_value = -1
    max_value = 1

    masked_loss = False

    ### Parameters specific to GWNET
    weight_decay = 0.0001
    in_dim = 10
    out_dim = 1
    dropout = 0.3
    nhid = 16
    print_every = 100

    scheduler_patience = 3
    scheduler_factor = 0.7
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_dir = os.path.join(ROOT_DIR,  "..",'data')
    output_dir = os.path.join(ROOT_DIR, "..", 'output')
    model_dir = os.path.join(output_dir, 'model')
    adj_path = os.path.join(output_dir,  "adjacency_matrix")
    log_dir = os.path.join(output_dir, 'log')

    for path in [data_dir, output_dir, adj_path, log_dir, model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

config_convlstm_1 = ConfigConvLSTM()
config_gwnet = ConfigGWNET()
config_ddim = ConfigDDIM()