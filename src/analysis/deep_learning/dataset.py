import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from typing import Union
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)

"""
Custom Datasets classes for Deep Learning models
"""

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

class NumpyBatchDataLoader(Dataset):
    def __init__(self, batches_dir, shuffle:bool=False):
        self.shuffle = shuffle
        self.batches_dir = batches_dir
        self.batch_files = sorted([f for f in os.listdir(batches_dir) if f.endswith('.npy')])
        
        if shuffle:
            np.random.shuffle(self.batch_files)

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, idx):
        batch_file = os.path.join(self.batches_dir, self.batch_files[idx])
        batch_dict = np.load(batch_file, allow_pickle=True).item()
        
        data_batch = batch_dict['data']
        label_batch = batch_dict['label']
        
        tensor_data = torch.tensor(data_batch, dtype=torch.float32), torch.tensor(label_batch, dtype=torch.float32)
        return DataLoader(tensor_data, batch_size=1)

class CustomConvLSTMDataset(Dataset):
    """
    Class for the ConvLSTM model, converting features and instances to pytorch tensors
    """
    def __init__(self, config:dict, args:dict, 
            data:xr.DataArray, 
            labels: xr.DataArray, 
            save_files:bool=False,
            filename = "dataset"):
        
        self.batch_size = config.batch_size
        self.num_timesteps = data.shape[0]
        self.num_channels = data.shape[1]
        self.foldername = filename

        if (len(labels.shape) == 3) and (len(data.shape)==4):
            labels = np.expand_dims(labels, 1)

        self.data = np.swapaxes(data, 1,0)
        self.labels = np.swapaxes(labels, 1,0)

        self.lag = config.include_lag
        self.input_size = config.input_size if args.model in ["CONVLSTM","DIME","AUTO_DIME"] \
            else data.shape[-1] if args.model=="GWNET"  or args.model=="WNET" else None

        self.steps_head = args.step_length
        self.learning_window = args.feature_days
        
        self.output_channels = config.output_channels
        self.output_window = config.num_frames_output
        self.save_path = config.data_dir
        self._generate_traing_data(args)

        if save_files is True:
            self._save_files(self.data, self.labels)

    def _generate_traing_data(self, args):

        logger.debug("Adding one channel to features to account for lagged data" if self.lag else "No lag channel added")
        tot_channels = self.num_channels + 1 if self.lag else self.num_channels

        self.available_timesteps = self.num_timesteps - self.learning_window - self.steps_head - self.output_window
        shape_train = (self.available_timesteps, tot_channels, self.learning_window, self.data.shape[2], self.data.shape[3]) if isinstance(self.input_size, tuple) else \
                      (self.available_timesteps, tot_channels, self.learning_window, self.input_size)
        shape_label = (self.available_timesteps, self.output_channels, self.output_window, self.labels.shape[2], self.labels.shape[3]) if isinstance(self.input_size, tuple) else \
                      (self.available_timesteps, self.output_channels, self.output_window, self.input_size) 

        train_data_processed = np.empty(shape_train, dtype=np.float32)
        label_data_processed = np.empty(shape_label, dtype=np.float32)

        logger.debug(f"Empty training matrix has shape {train_data_processed.shape}")
        logger.debug(f"Empty instance matrix has shape {label_data_processed.shape}")

        def populate_arrays(args, train_data_processed, label_data_processed):

            current_idx = self.learning_window
            while current_idx < self.num_timesteps - self.steps_head - self.output_window:
                start_idx = current_idx - self.learning_window
                end_idx = current_idx

                if self.num_channels == 1:
                    train_data_processed[start_idx, 0] = self.data[0, start_idx:end_idx]
                    label_data_processed[start_idx, 0, :] = \
                        self.labels[0, current_idx + self.steps_head : current_idx + self.steps_head + self.output_window]

                elif self.num_channels > 1:
                    for chnl in range(self.num_channels):
                        train_data_processed[start_idx, chnl ] = self.data[chnl, start_idx:end_idx]

                    if self.lag:
                        train_data_processed[start_idx, -1] = self.labels[0, start_idx:end_idx]

                    label_data_processed[start_idx, 0 ] = \
                        self.labels[0, current_idx + self.steps_head : current_idx + self.steps_head + self.output_window]

                current_idx += 1
            
            if args.model == "CONVLSTM" or args.model=="DIME":
                train_data_processed = train_data_processed.swapaxes(1,2)
                label_data_processed = label_data_processed.swapaxes(1,2)
            
            return train_data_processed, label_data_processed
        
        
        self.data, self.labels = populate_arrays(args, train_data_processed, label_data_processed)

    def __len__(self):
        return self.data.shape[0] #- self.learning_window - self.steps_head - self.output_window

    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]
        return X, y

    def _save_files(self, data, labels):
        from tqdm.auto import tqdm
        dest_path = self.save_path + "/data_convlstm" + f"/{self.foldername}"
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        num_batches = self.data.shape[0] // self.batch_size
        for idx in tqdm(range(num_batches)):
            data_batch = data[idx*self.batch_size:(idx+1)*self.batch_size]
            label_batch = labels[idx*self.batch_size:(idx+1)*self.batch_size]
            batch_dict = {'data': data_batch, 'label': label_batch}
            np.save(os.path.join(dest_path, f'batch_{idx}.npy'), batch_dict)
        
        if self.data.shape[0] % self.batch_size != 0:
            data_batch = data[num_batches*self.batch_size:]
            label_batch = labels[num_batches*self.batch_size:]
            batch_dict = {'data': data_batch, 'label': label_batch}
            np.save(os.path.join(dest_path, f'batch_{num_batches}.npy'), batch_dict)

class DataGenerator(CustomConvLSTMDataset):
    def __init__(self, config, args, data, labels, autoencoder):
        super().__init__(config, args, data, labels)

        self.device = config.device
        self.time_list = self._add_time_list(data)

        filepath = os.path.join(config.data_dir, "autoencoder_output")
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        file = os.path.join(filepath, "vae_features.npy")

        if os.path.exists(file):
            self.data = self._load_auto_output(file)
        else:
            vae_output = self._reduce_data_vae(autoencoder.to(self.device))
            extra_features = self.data[:, -1, :-1]
            data = np.concatenate([extra_features, vae_output], axis=1)
            self._export_auto_output(data, file)
            self.data = data

    def _load_auto_output(self, data_dir):
        logger.info("Loading stored VAE output...")
        with open(data_dir, "rb") as f:
            return np.load(f)

    def _export_auto_output(self, data, data_dir):
        logger.info("Saving VAE output...")
        with open(data_dir, "wb") as f:
            np.save(f, data)


    def _add_time_list(self, data):
        from analysis import date_to_sinusoidal_embedding
        from utils.function_clns import config
        import pandas as pd
        from utils.xarray_functions import _output_dates
        from datetime import datetime, timedelta
        num_steps = data.shape[0]
        min_date = config["DEFAULT"]["date_start"]
        max_date = datetime.strftime(pd.to_datetime(min_date) + timedelta(days = num_steps -1), "%Y-%m-%d")
        expected_dates = _output_dates("P1D", min_date, max_date)
        dates = [datetime.strftime(i,"%Y-%m-%d") for i in expected_dates]
        
        if len(dates) != num_steps:
            logger.info(f"The tensor input has missing dates")
            logger.info(f"Dates should be {len(dates)} instead of {num_steps}")
        return dates

    def _apply_batched_autoencoder(self, x, autoencoder, batch_size):
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm.auto import tqdm
        # Create DataLoader
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                                shuffle=False)

        # Apply the encoder to each batch
        encoded_batches = []
        for batch in tqdm(dataloader):
            inputs = batch[0]
            with torch.no_grad():  # Disable gradient computation for inference
                encoded = autoencoder.encoder(inputs)
            encoded_batches.append(encoded)

        # Concatenate all encoded batches
        return torch.cat(encoded_batches)

    def _reduce_data_vae(self, autoencoder):
        imag_past = torch.from_numpy(self.data[:,:,-1]).to(self.device)
        b, t, h, w = imag_past.size()
        imag_past = imag_past.permute(0, 2, 3, 1)
        imag  =  imag_past.reshape(b*h*w, t)
        x = torch.unsqueeze(imag, 1)
        reduced_data = self._apply_autoencoder(x, autoencoder)
        return reduced_data.view(b, h, w, 20).permute(0, 3, 1, 2) + 1 

    def _apply_autoencoder(self, x, autoencoder):
        reduced_data = self._apply_batched_autoencoder(x, autoencoder, 256)\
            .detach().float().cpu()
        return reduced_data

    def old__getitem__(self, index):
        from analysis import date_to_sinusoidal_embedding
        data = np.empty((self.batch_size, *self.data.shape[1:]))
        pred = np.empty((self.batch_size, *self.labels.shape[1:]))
        time_array = np.empty((self.batch_size, *self.data.shape[2:]))

        for i in range(self.batch_size):
            random = np.random.randint(0, self.available_timesteps)
            X = self.data[random]
            y = self.labels[random]
            time = self.time_list[random]
            time_embeddings = date_to_sinusoidal_embedding(time, *self.data.shape[2:])
            pred[i] = y
            data[i] = X
            time_array[i] = (time_embeddings + 1)/2

        time_array = np.expand_dims(time_array, 1)
        data = np.concatenate([time_array, data], axis = 1)

        data = torch.from_numpy(data).float().to(self.device)
        pred = torch.from_numpy(pred).float().to(self.device)
        return data, pred
    
    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]
        return X, y


class MyDataset(Dataset):
    """
    Subclass of PyTorch's Dataset
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
    

"""
Dataset classes for training
"""

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, config, logger, patience=None, verbose=False,):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        if patience is None:
            self.patience = config.patience
        else:
            self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.logger = logger

    def __call__(self, val_loss, model, epoch, save_path):
    
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_path)
            
        elif score < self.best_score:
            self.counter += 1
            self.logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )            
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, save_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.logger.info(
                f'Validation loss change: ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        torch.save(
            model, save_path + "/" +
            "checkpoint_epoch_{}.pth.tar".format(epoch, val_loss))
        self.val_loss_min = val_loss