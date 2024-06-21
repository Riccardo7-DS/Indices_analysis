import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import sampler
import numpy as np
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
from analysis import CustomConvLSTMDataset
# from analysis.deep_learning.DDIM.model import DiffusionModel
from analysis.configs.config_models import config_convlstm_1 as model_config
from torch.nn import MSELoss
from utils.function_clns import CNN_split, config
import argparse
import os
import sys
import pyproj
parser = argparse.ArgumentParser()
parser.add_argument('-f')

### Convlstm parameters
parser.add_argument('--model',type=str,default="DIME",help='DL model training')
parser.add_argument('--step_length',type=int,default=1)
parser.add_argument('--feature_days',type=int,default=60)

parser.add_argument("--normalize", type=bool, default=True, help="Input data normalization")
parser.add_argument("--scatterplot", type=bool, default=True, help="Whether to visualize scatterplot")
args = parser.parse_args()

data_dir = model_config.data_dir+"/data_convlstm"
data = np.load(os.path.join(data_dir, "data.npy"))
target = np.load(os.path.join(data_dir, "target.npy"))
mask = torch.tensor(np.load(os.path.join(data_dir, "mask.npy")))

################################# Initialize datasets #############################
train_data, val_data, train_label, val_label, \
    test_valid, test_label = CNN_split(data, target, split_percentage=config["MODELS"]["split"])
# create a CustomDataset object using the reshaped input data
train_dataset = CustomConvLSTMDataset(model_config, args, 
                                      train_data, train_label)

val_dataset = CustomConvLSTMDataset(model_config, args, 
                                    val_data, val_label)
        
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class Autoencoder(nn.Module):
    """Makes the main denoising auto

    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape=20):
        super(Autoencoder, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=1, stride=1), ## N, C, L --> N, C, L_out
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=0), ## N, C, L --> N, C, L_out
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=0), ## N, C, L --> N, C, L_out
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(256),
            nn.ELU(),
            Flatten(),
            nn.Linear(256*5, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 20),
            nn.ELU())
        
        self.decode = nn.Sequential(
            nn.Linear(20, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, enc_shape),
            nn.ELU(),
        )
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

encoder = Autoencoder(in_shape=60, enc_shape=1).to(model_config.device)
criterion = MSELoss()
optimizer = torch.optim.Adam(encoder.parameters())

# number of epochs to train the model
n_epochs = 20

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for idx, (data, target) in enumerate(train_dataset):
        # _ stands in for labels, here
        images = torch.from_numpy(data[-1, :, :, :]).to(model_config.device)
        target = torch.from_numpy(target).to(model_config.device)

        images = images.view(64 * 64, 1, 60)
        # flatten images
        # images = images.view(images.size(0), -1)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = encoder(images).view(1, 1, 64, 64)
        # calculate the loss
        loss = criterion(outputs, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*images.size(0)
            
    # print avg training statistics 
    train_loss = train_loss/len(train_dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))

