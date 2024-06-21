 
import torch
import numpy as np
import matplotlib.pyplot as plt
from analysis import CustomConvLSTMDataset
# from analysis.deep_learning.DDIM.model import DiffusionModel
from analysis.deep_learning.DDIM.autoencoder import Autoencoder
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
parser.add_argument('--step_length',type=int,default=15)
parser.add_argument('--feature_days',type=int,default=60)

parser.add_argument("--normalize", type=bool, default=True, help="Input data normalization")
parser.add_argument("--scatterplot", type=bool, default=True, help="Whether to visualize scatterplot")
args = parser.parse_args()

data_dir = model_config.data_dir+"/data_convlstm"
data = np.load(os.path.join(data_dir, "data.npy"))
target = np.load(os.path.join(data_dir, "target.npy"))
mask = torch.tensor(np.load(os.path.join(data_dir, "mask.npy")))

class DataGenerator(CustomConvLSTMDataset):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __getitem__(self, index):
        
        result_x = (self.batch_size, *self.data[0].shape)
        result_y = (self.batch_size, *self.labels[0].shape)
        
        for i in range(self.batch_size):
            np.empty(self.shape_train, dtype=np.float32)
            random = np.random.randint(0,(self.num_timesteps-self.learning_window - self.steps_head - self.output_window)) 
            X = self.data[random]
            y = self.labels[random]
            result_x[i] = X
            result_y[i] = y
        return X, y
    


################################# Initialize datasets #############################
train_data, val_data, train_label, val_label, \
    test_valid, test_label = CNN_split(data, target, split_percentage=config["MODELS"]["split"])
# create a CustomDataset object using the reshaped input data
train_dataset = CustomConvLSTMDataset(model_config, args, 
                                      train_data, train_label)

val_dataset = CustomConvLSTMDataset(model_config, args, 
                                    val_data, val_label)
        

encoder = Autoencoder(in_shape=60, enc_shape=5).double().to(model_config.device)
loss = MSELoss()
optimizer = torch.optim.Adam(encoder.parameters())

def train(model, error, optimizer, n_epochs, x):
    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        output = model(x)
        loss = error(output, x)
        loss.backward()
        optimizer.step()
        
        if epoch % int(0.1*n_epochs) == 0:
            print(f'epoch {epoch} \t Loss: {loss.item():.4g}')

train(encoder, loss, optimizer, 5000, train_dataset.data)


sys.exit(0)

# generator definition 

train_generator = DataGenerator(train_dataset, model_config.batch_size)
val_generator = DataGenerator(val_dataset,model_config.batch_size)

################################# Parameters #############################

learning_rate = 1e-5
epochs = 30
widths = [64, 128, 256, 384]
block_depth = 2
image_size = (64, 64)
loss = MSELoss().to(model_config.device)

# diffusion model 

model = DiffusionModel(image_size, 9, 3, widths, block_depth)

optimizer= torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

# pixelwise mean absolute error is used as loss


