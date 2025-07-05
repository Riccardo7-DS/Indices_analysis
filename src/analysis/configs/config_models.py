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


class BaseConfig():
    def __init__(self):
        self.data_dir = os.path.join(ROOT_DIR,  "..",'data')
        self.output_dir = os.path.join(ROOT_DIR, "..", 'output')
        self.model_dir = os.path.join(self.output_dir, 'model')
        self.log_dir = os.path.join(self.output_dir, 'log')
        for path in [self.data_dir, self.output_dir, self.log_dir, self.model_dir]:
            if not os.path.exists(path):
                os.makedirs(path)


class ConfigAutoDime(BaseConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 1e-3
    squared = True
    batch_size = 8
    include_lag = True
    image_size = 64
    input_size = (83, 77)
    output_channels = 1
    num_frames_output = 1
    patience = 20
    epochs = 200
    sampling_steps = 50

class ConfigDDIM(BaseConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 64
    input_size = (83, 77)
    num_frames = 1
    output_channels = 1
    num_frames_output = 1
    # num_ensambles = 10

    save_and_sample_every = 10
    update_after_step = 1300 # 130*10
    sigma_rel = 0.15

    ema_decay = 0.999
    ema_update_every = 10

    scheduler_patience = 3
    scheduler_factor = 0.7

    timesteps = 1000
    sampling_steps = 50

    # sampling

    min_signal_rate = 0.015
    max_signal_rate = 0.95

    beta_start = 0.0001
    beta_end = 0.02

    # architecture

    embedding_dims = 64 # 32
    embedding_max_frequency = 1000.0
    # widths = [32, 64, 96, 128]
    widths = [64, 128, 256, 384]
    block_depth = 2

    # optimization

    batch_size = 16
    learning_rate = 1e-4
    epochs = 500
    patience = 20
    min_patience = 60

    include_lag = True
    squared = True

class ConfigConvLSTM(BaseConfig):

    gpus = [0, ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    masked_loss = True

    num_frames_output = 1 ### means 1 frame as output
    output_channels = 1
    include_lag = True
    num_samples = 9 #### the number of channels to use

    epochs = 100
    patience = 10
    learning_rate = 1e-3
    batch_size= 52#16 
    null_value = -1
    max_value = 1

    scheduler_patience = 3
    scheduler_factor = 0.7

    image_size = (64, 64)
    input_size = (64, 64)

    weight_decay = 0.0001
    

    # Parameters specific to ConvLSTM
    # dim = 64
    layers = [64, 64, 64]

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


class ConfigGWNET(BaseConfig):

    num_frames_output = 1 ### means 1 frame as output
    output_channels = 1
    include_lag = True
    num_samples = 9 #### the number of channels to use

    epochs = 300
    patience = 10
    learning_rate = 1e-4
    batch_size= 16
    dim = 64

    null_value = -1
    max_value = 1

    masked_loss = False

    ### Parameters specific to GWNET
    weight_decay = 0.0001
    in_dim = 10
    out_dim = 1
    dropout = 0.3
    nhid = 32
    blocks = 6
    layers = 4
    print_every = 100

    scheduler_patience = 3
    scheduler_factor = 0.7
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConfigWNET(BaseConfig):

    num_frames_output = 1 ### means 1 frame as output
    output_channels = 1
    include_lag = True
    num_samples = 9 #### the number of channels to use

    epochs = 200
    patience = 10
    learning_rate = 0.1
    batch_size= 8
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
    blocks = 6
    layers = 4
    print_every = 100

    scheduler_patience = 3
    scheduler_factor = 0.7
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

config_convlstm_1 = ConfigConvLSTM()
config_gwnet = ConfigGWNET()
config_ddim = ConfigDDIM()
config_wnet = ConfigWNET()
config_autodime = ConfigAutoDime()