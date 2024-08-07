import os
import numpy as np
import torch
from torch.utils.data import Dataset
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



class CustomConvLSTMDataset(Dataset):
    """
    Class for the ConvLSTM model, converting features and instances to pytorch tensors
    """
    def __init__(self, config:dict, args:dict, 
            data:xr.DataArray, 
            labels: xr.DataArray, 
            save_files:bool=False,
            filename = "dataset"):
        
        self.num_timesteps = data.shape[0]
        self.num_channels = data.shape[1]

        if (len(labels.shape) == 3) and (len(data.shape)==4):
            labels = np.expand_dims(labels, 1)

        self.data = np.swapaxes(data, 1,0)
        self.labels= np.swapaxes(labels, 1,0)
        # else:              
        #     self.labels = labels
        self.lag = config.include_lag
        # self.image_size = config.image_size
        self.input_size = config.input_size if args.model in ["CONVLSTM","DIME","AUTO_DIME"] else data.shape[-1] if args.model=="GWNET"  or args.model=="WNET" else None

        self.steps_head = args.step_length
        self.learning_window = args.feature_days
        
        self.output_channels = config.output_channels
        self.output_window = config.num_frames_output
        # self.num_frames = config.num_frames_input + config.num_frames_output
        self.save_path = config.output_dir
        self._generate_traing_data(args)
        
        # if save_files is True:
        #     self.filename = filename
        #     self._save_files(self.data, self.labels)

        #print('Loaded {} samples ({})'.format(self.__len__(), split))

    
    def _generate_traing_data(self, args):

        logger.debug("Adding one channel to features to account for lagged data" if self.lag else "No lag channel added")
        tot_channels = self.num_channels + 1 if self.lag else self.num_channels

        self.available_timesteps = self.num_timesteps - self.learning_window - self.steps_head - self.output_window
        shape_train = (self.available_timesteps, tot_channels, self.learning_window, *self.input_size) if isinstance(self.input_size, tuple) else \
                      (self.available_timesteps, tot_channels, self.learning_window, self.input_size)
        shape_label = (self.available_timesteps, self.output_channels, self.output_window, *self.input_size) if isinstance(self.input_size, tuple) else \
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
        import pickle
        for idx in range(data.shape[0]):
            img_x = data[idx] 
            img_y = labels[idx]
            data_dict = {"data": img_x, "label": img_y}

            dest_path = self.save_path + "/data_convlstm" + f"/{self.filename}"
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            pfilename= os.path.join(dest_path,  f"{idx:06d}"+".pkl")
            with open(pfilename, 'wb') as file:
                pickle.dump(data_dict, file)


class DataGenerator(CustomConvLSTMDataset):
    def __init__(self, config, args, data, labels, batch_size, autoencoder):
        super().__init__(config, args, data, labels)
        self.batch_size = batch_size
        self.device = config.device
        x = self._prepare_for_diffusion(config, args, self.data)
        reduced_data = self._apply_autoencoder(x, autoencoder)
        output = self._postrocess(config, reduced_data, self.data)
        self.data = np.concatenate([self.data[:, -1, :-1], output], axis=1)

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

    def _postrocess(self, config, output, data):
        return output.view(data.shape[0], config.image_size , 
                           config.image_size , 20).transpose(1, 3)

    def _prepare_for_diffusion(self, config, args, data):
        imag_past = data.transpose(0, 2, 1, 3, 4)[:,-1]
        imag_past  =  imag_past.reshape(data.shape[0] * config.image_size * config.image_size, 
                              args.feature_days)
        return np.expand_dims(imag_past, 1) 

    def _apply_autoencoder(self, x, autoencoder):
        x = torch.from_numpy(x).to(self.device)
        reduced_data = self._apply_batched_autoencoder(x, autoencoder, 256)\
            .detach().float().cpu()
        return reduced_data

    def __getitem__(self, index):
        data = np.empty((self.batch_size, *self.data.shape[1:]))
        pred = np.empty((self.batch_size, *self.labels.shape[1:]))
        for i in range(self.batch_size):
            random = np.random.randint(0, self.available_timesteps)
            X = self.data[random]
            y = self.labels[random]
            pred[i] = y
            data[i] = X

        data = torch.from_numpy(data).float().to(self.device)
        pred = torch.from_numpy(pred).float().to(self.device)
        return data, pred

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