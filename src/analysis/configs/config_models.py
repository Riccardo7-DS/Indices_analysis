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

class ConfigConvLSTM:

    gpus = [0, ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    masked_loss = True
    model_name = "checkpoint_epoch_99"

    # num_frames_input = 60
    num_frames_output = 1 ### means 1 frame as output
    output_channels = 1
    include_lag = True
    num_samples = 9 #### the number of channels to use

    epochs = 300
    patience = 20
    learning_rate = 1e-3
    batch_size= 8

    image_size = (64, 64)
    input_size = (64, 64)

    # Parameters specific to ConvLSTM
    dim = 64


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
    batch_size= 8
    dim = 64

    masked_loss = False

    ### Parameters specific to GWNET
    weight_decay = 0.0001
    in_dim = 10
    dropout = 0.3
    nhid = 32
    print_every = 50
    num_nodes = 500

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