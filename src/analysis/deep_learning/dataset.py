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
    def __init__(self, config, data, labels):
        self.data = data
        self.labels = labels
        self.image_size = config.image_size
        self.input_size = config.input_size
        self.steps_head = config.step_length
        self.num_timesteps = data.shape[1]
        self.learning_window = config.num_frames_input
        self.num_samples = config.num_samples
        self.output_window = config.num_frames_output
        self.num_frames = config.num_frames_input + config.num_frames_output
        self._generate_traing_data()
        #print('Loaded {} samples ({})'.format(self.__len__(), split))

    def _generate_traing_data(self):

        train_data_processed = np.zeros((self.num_timesteps - self.learning_window - self.steps_head - self.output_window, 
                                         self.num_samples,
                                         self.learning_window, *self.input_size), dtype=np.float32)
        
        label_data_processed = np.zeros((self.num_timesteps - self.learning_window - self.steps_head - self.output_window, 
                                         self.num_samples, 
                                         self.output_window, *self.input_size), dtype=np.float32)

        current_idx = 0
        
        if self.num_samples == 1:
            while current_idx + self.steps_head + self.learning_window + self.output_window <  self.num_timesteps:
                train_data_processed[current_idx, 0, :, :, :] = self.data[0, current_idx : current_idx + self.learning_window, :, :]
                label_data_processed[current_idx, 0, :, :, :] = self.labels[0, current_idx + self.learning_window+ self.steps_head : current_idx + self.learning_window+ self.steps_head + self.output_window, :, :]
                current_idx +=1
        
        elif self.num_samples ==2:
            while current_idx + self.steps_head + self.learning_window + self.output_window <  self.num_timesteps:
                train_data_processed[current_idx, 0, :, :, :] = self.data[0, current_idx : current_idx + self.learning_window, :, :]
                train_data_processed[current_idx, 1, :, :, :] = self.labels[0, current_idx : current_idx + self.learning_window, :, :]
                label_data_processed[current_idx, 0, :, :, :] = self.labels[0, current_idx + self.learning_window+ self.steps_head : current_idx + self.learning_window+ self.steps_head + self.output_window, :, :]
                current_idx +=1
            
        else:
            raise NotImplementedError(f"Not implemented a model with {self.num_samples} channels")

        self.data = train_data_processed.swapaxes(1,2)
        self.labels = label_data_processed.swapaxes(1,2)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        #X = np.expand_dims(self.data[index, :, :, :], axis=1)
        #y = np.expand_dims(self.targets[index, :, :, :], axis=1)
        X = self.data[index, :, :, :, :]
        y = self.labels[index,:,  :, :, :]
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