import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import xarray as xr
import re
from typing import Union

class PrecipDataset(Dataset):
    '''
    This dataset is based on https://github.com/jwwthu/DL4Traffic/tree/main/TaxiBJ21
    Parameters
    ..........
    root (String) - Path to the dataset if it is already downloaded. If not downloaded, it will be downloaded in the given path.
    length_spi - the latency of the SPI product
    history_length (Int) - Length of history data in sequence of each sample
    prediction_length (Int) - Length of prediction data in sequence of each sample
    years (List, Optional) - Dataset will be downloaded for the given years
    '''


    def __init__(self, root, length_spi:Union[int, None], history_length, prediction_length):
        super().__init__()

        assert length_spi in [30, 60, 90, 180]

        if length_spi != None:
            self.name = [f for f in os.listdir(root) if f.endswith('.nc') and str(length_spi) and ('gamma') in f][0]
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
        arr = arr.transpose('time','lat','lon')
        self.full_data = arr[self.abbrev].values

        self.timesteps = self.full_data.shape[0]
        self.grid_height = self.full_data.shape[1]
        self.grid_width = self.full_data.shape[2]

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
        total_length = self.full_data.shape[0]
        for end_idx in range(history_length + prediction_length, total_length):
            predict_frames = self.full_data[end_idx-prediction_length:end_idx]
            history_frames = self.full_data[end_idx-prediction_length-history_length:end_idx-prediction_length]
            self.X_data.append(history_frames)
            self.Y_data.append(predict_frames)
        print(self.X_data)
        self.X_data = np.stack(self.X_data)
        self.Y_data = np.stack(self.Y_data)
        


    def __len__(self) -> int:
        return len(self.Y_data)


    def __getitem__(self, index: int):
        sample = {"x_data": self.X_data[index], "y_data": self.Y_data[index]}
        return sample


import glob
import tensorflow as tf

def load_nc_dir_with_generator(dir_):
    def gen():
        for file in glob.glob(os.path.join(dir_, "*.nc")):
            ds = xr.open_dataset(file, engine='netcdf4')
            yield {key: tf.convert_to_tensor(val) for key, val in ds.items()}


    sample = next(iter(gen()))

    return tf.data.Dataset.from_generator(
        gen,
        output_signature={
            key: tf.TensorSpec(val.shape, dtype=val.dtype)
            for key, val in sample.items()
        }
    )


def load_nc_dir_with_map_and_xarray(dir_):
    def open_path(path_tensor: tf.Tensor):
        ds = xr.open_dataset(path_tensor.numpy().decode())
        return tf.convert_to_tensor(ds["a"])
    return tf.data.Dataset.list_files(os.path.join(dir_, "*.nc")).map(
        lambda path: tf.py_function(open_path, [path], Tout=tf.float64),
        )

def load_nc_dir_cached_to_tfrecord(dir_):
    """Save data to tfRecord, open it, and deserialize
    
    Note that tfrecords are not that complicated! The simply store some
    bytes, and you can serialize data into those bytes however you find
    convenient. In this case, I serialie with `tf.io.serialize_tensor` and 
    deserialize with `tf.io.parse_tensor`. No need for `tf.train.Example` or any
    of the other complexities mentioned in the official tutorial.

    """
    generator_tfds = load_nc_dir_with_generator(dir_)
    writer = tf.data.experimental.TFRecordWriter("local.tfrecord")
    writer.write(generator_tfds.map(lambda x: tf.io.serialize_tensor(x["a"])))

    return tf.data.TFRecordDataset("local.tfrecord").map(
        lambda x: tf.io.parse_tensor(x, tf.float64))


def load_nc_dir_after_save(dir_):
    generator_tfds = load_nc_dir_with_generator(dir_)
    tf.data.experimental.save(generator_tfds, "local_ds")
    return tf.data.experimental.load("local_ds")

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