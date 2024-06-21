import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from analysis import CustomConvLSTMDataset
# from analysis.deep_learning.DDIM.model import DiffusionModel
from analysis.configs.config_models import config_convlstm_1 as model_config
from torch.nn import MSELoss
from utils.function_clns import CNN_split, config, init_logging
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
import os
import logging
from analysis import masked_mape, masked_rmse
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('-f')

logger = init_logging()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info("The selected device is {}".format(device))

### Convlstm parameters
parser.add_argument('--model',type=str,default="DIME",help='DL model training')
parser.add_argument('--step_length',type=int,default=15)
parser.add_argument('--feature_days',type=int,default=180)

parser.add_argument("--normalize", type=bool, default=True, help="Input data normalization")
parser.add_argument("--scatterplot", type=bool, default=True, help="Whether to visualize scatterplot")
args = parser.parse_args()

data_dir = model_config.data_dir+"/data_convlstm"
data = np.load(os.path.join(data_dir, "data.npy"))
target = np.load(os.path.join(data_dir, "target.npy"))
mask = torch.tensor(np.load(os.path.join(data_dir, "mask.npy")))
decoder_output = 1

def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

def corrcoef(x):
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

def tensor_corr(prediction, label):
    x = label.squeeze().view(-1)
    y = prediction.squeeze().view(-1)
    return pearsonr(x, y)


################################# Initialize datasets #############################
train_data, val_data, train_label, val_label, \
    test_valid, test_label = CNN_split(data, target, split_percentage=config["MODELS"]["split"])
# create a CustomDataset object using the reshaped input data
train_dataset = CustomConvLSTMDataset(model_config, args, 
                                      train_data, train_label)

val_dataset = CustomConvLSTMDataset(model_config, args, 
                                    val_data, val_label)

train_dataloader = DataLoader(train_dataset, 
                                  batch_size=model_config.batch_size, shuffle=True)
        
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
            nn.Linear(256*20, 128),
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

encoder = Autoencoder(in_shape=args.feature_days, 
                      enc_shape=decoder_output).to(model_config.device)
criterion = MSELoss()
optimizer = torch.optim.Adam(encoder.parameters())

# number of epochs to train the model
n_epochs = 20

train_loss_records = []

for epoch in range(1, n_epochs + 1):
    # monitor training loss
    ######################################
    epoch_records = {'loss': [], "mape": [], "rmse": [], "corr": []}
    
    for batch_idx, (data, target) in enumerate(train_dataloader):
        # _ stands in for labels, here
        images = data[:, -1, :, :, :].to(model_config.device)
        target = target.to(model_config.device).squeeze()

        imag = images.reshape(images.size(0) * 64 * 64, 1, args.feature_days)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = encoder(imag).view(images.size(0), 1, 64, 64).squeeze()
        # calculate the loss
        loss = criterion(outputs, target)
        mape = masked_mape(outputs, target).item()
        rmse = masked_rmse(outputs, target).item()
        corr = tensor_corr(outputs, target).item()
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        epoch_records['loss'].append(loss.item())
        epoch_records["rmse"].append(rmse)
        epoch_records["mape"].append(mape)
        epoch_records["corr"].append(corr)
        
        # # Print metrics every 100 steps
        # if (batch_idx + 1) % 500 == 0:
        #     log = 'Epoch: {:03d}, Step: {:04d}, Loss: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}, CORR: {:.4f}'
        #     logger.info(log.format(epoch, batch_idx + 1,
        #                            np.mean(epoch_records['loss']),
        #                            np.mean(epoch_records['mape']),
        #                            np.mean(epoch_records['rmse']),
        #                            np.mean(epoch_records['corr'])))
            
        #     # Clear epoch_records for the next set of 100 steps
        #     epoch_records = {'loss': [], "mape": [], "rmse": [], "corr": []}
    
    # Log metrics for the entire epoch
    log = 'Epoch: {:03d}, Loss: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}, CORR: {:.4f}'
    logger.info(log.format(epoch, 
                           np.mean(epoch_records['loss']),
                           np.mean(epoch_records['mape']),
                           np.mean(epoch_records['rmse']),
                           np.mean(epoch_records['corr'])))
    
    train_loss_records.append(np.mean(epoch_records['loss']))
    
    plt.plot(range(epoch), train_loss_records, label='train')
    plt.legend()
    plt.close()

