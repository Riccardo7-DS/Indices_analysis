import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from geotorchai.utility.exceptions import InvalidParametersException
from geotorchai.utility._download_utils import _download_cdsapi_files
import xarray as xr
import re
from typing import Union

class PrecipDataset(Dataset):
    '''
    This dataset is based on https://github.com/jwwthu/DL4Traffic/tree/main/TaxiBJ21
    Parameters
    ..........
    root (String) - Path to the dataset if it is already downloaded. If not downloaded, it will be downloaded in the given path.
    history_length (Int) - Length of history data in sequence of each sample
    prediction_length (Int) - Length of prediction data in sequence of each sample
    years (List, Optional) - Dataset will be downloaded for the given years
    '''


    def __init__(self, root, length_spi:Union[int, None], history_length, prediction_length):
        super().__init__()

        assert length_spi in [30, 60, 90, 180]

        if length_spi != None:
            self.name = [f for f in os.listdir(root) if f.endswith('.nc') and length_spi in f][0]
            _abbrev = re.search('(.*)(spi_gamma_\d+)(.nc)', self.name)
            self.freq = length_spi
            #self.name = os.path.basename(os.path.normpath(self.path))
            self.abbrev = _abbrev.group(2)
            self.product = _abbrev.group(1)
            self.product_type = 'radiometer'

        else:
            raise NotImplementedError('The non-SPI products need to be implemented')

        data_dir = os.path.join(root, self.name)

        arr = xr.open_dataset(data_dir)
        self.full_data = arr[self.abbrev].values

        self.dims = self.full_data.dims
        

        self.timesteps = self.full_data.shape[self.dims.index('time')]
        self.grid_height = self.full_data.shape[self.dims.index('lat')]
        self.grid_width = self.full_data.shape[self.dims.index('lon')]

        self.full_data = self.full_data.reshape((self.timesteps, 1, self.grid_height, self.grid_width))

        self._generate_sequence_data(history_length, prediction_length)



    ## This method returns the total number of timesteps in the generated dataset
    def get_timesteps(self):
        return self.timesteps



    ## This method returns the height of the grid in the generated dataset
    def get_grid_height(self):
        return self.grid_height



    ## This method returns the width of the grid in the generated dataset
    def get_grid_width(self):
        return self.grid_width



    def _generate_sequence_data(self, history_length, prediction_length):
        self.X_data = []
        self.Y_data = []
        total_length = self.full_data.shape[self.dims.index('time')]
        for end_idx in range(history_length + prediction_length, total_length):
            predict_frames = self.full_data[end_idx-prediction_length:end_idx]
            history_frames = self.full_data[end_idx-prediction_length-history_length:end_idx-prediction_length]
            self.X_data.append(history_frames)
            self.Y_data.append(predict_frames)
        self.X_data = np.stack(self.X_data)
        self.Y_data = np.stack(self.Y_data)
        


    def __len__(self) -> int:
        return len(self.Y_data)


    def __getitem__(self, index: int):
        sample = {"x_data": self.X_data[index], "y_data": self.Y_data[index]}
        return sample
