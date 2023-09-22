import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import xarray as xr
import re
from typing import Union
import torch.nn.functional as F


class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y



class CustomConvLSTMDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        X = np.expand_dims(self.data[index, :, :, :], axis=1)
        y = np.expand_dims(self.targets[index, :, :, :], axis=1)
        return X, y


class MyDataset(Dataset):
    """Subclass of PyTorch's Dataset
    """
    def __init__(self, data, transform=None):
        self.data_size = data.shape[0]
        self.input = data
        self.transform = transform

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = self.input[idx,:]
        sample = torch.tensor(sample,dtype=torch.float)
        if self.transform:
            sample = self.transform(sample) 
        sample = F.normalize(sample, p=1, dim=-1)
        return sample