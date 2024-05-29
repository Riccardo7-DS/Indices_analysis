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
root_dir = os.path.join(os.getcwd(), '.')
print(root_dir)

class Config:

    gpus = [0, ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #if torch.cuda.is_available():
    #    num_workers = 8 * len(gpus)
    #    train_batch_size = 64
    #    valid_batch_size = 2 * train_batch_size
    #    test_batch_size = 2 * train_batch_size
    #else:
    num_workers = 0
    train_batch_size = 32
    valid_batch_size = 2 * train_batch_size
    test_batch_size = 2 * train_batch_size
    data_file = 'datas/train-images-idx3-ubyte.gz'
    masked_loss = True
    model_name = "checkpoint_epoch_99"

    num_frames_input = 90
    num_frames_output = 1
    output_channels = 1

    image_size = (64, 64)
    input_size = (64, 64)
    step_length = 4 #### the jump in the future
    num_samples = 9 #### the number of channels to use
    num_objects = [3]
    display = 100
    draw = 10
    rnn_dropout = None
    cnn_dropout = None
    decoder_3d = False
    epochs = 100
    patience = 10
    learning_rate = 1e-4
    batch_size= 8

    max_value = 1

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

    data_dir = os.path.join(root_dir, 'data')
    output_dir = os.path.join(root_dir, 'output')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checkpoint_dir =  os.path.join(output_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_dir = os.path.join(output_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_dir = os.path.join(output_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    cache_dir = os.path.join(output_dir, 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

config = Config()