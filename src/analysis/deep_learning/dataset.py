import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from typing import Union
import torch.nn.functional as F
import glob
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import logging
from datetime import datetime, timedelta
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
    def __init__(self, 
            config:dict, 
            args:dict, 
            data:xr.DataArray, 
            labels: xr.DataArray, 
            start_date: str = None,
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
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None  # <- Convert to datetime
        # self.end_date = datetime.strptime(start_date + timedelta(days=), "%Y-%m-%d") if start_date else None  # <- Convert to datetime

        
        self.output_channels = config.output_channels
        self.output_window = config.num_frames_output
        self.save_path = config.data_dir
        self._generate_traing_data(args)

        if save_files is True:
            self._save_files(self.data, self.labels)

    def _generate_traing_data(self, args):

        logger.debug("Adding one channel to features to account for lagged data" \
                      if self.lag else "No lag channel added")
        tot_channels = self.num_channels + 1 if self.lag else self.num_channels

        if args.model == "DIME":
            self.available_timesteps = self.num_timesteps - args.auto_days - self.steps_head 
        else:
            self.available_timesteps = self.num_timesteps - self.learning_window - self.steps_head
        
        shape_train = (self.available_timesteps, tot_channels, self.learning_window, self.data.shape[2], self.data.shape[3]) if isinstance(self.input_size, tuple) else \
                      (self.available_timesteps, tot_channels, self.learning_window, self.input_size)
        shape_label = (self.available_timesteps, self.output_channels, self.output_window, self.labels.shape[2], self.labels.shape[3]) if isinstance(self.input_size, tuple) else \
                      (self.available_timesteps, self.output_channels, self.output_window, self.input_size) 

        train_data_processed = np.empty(shape_train, dtype=np.float32)
        label_data_processed = np.empty(shape_label, dtype=np.float32)

        logger.debug(f"Empty training matrix has shape {train_data_processed.shape}")
        logger.debug(f"Empty instance matrix has shape {label_data_processed.shape}")

        def populate_arrays(args, train_data_processed, label_data_processed):
            sample_idx = 0

            if args.model == "DIME":
                current_idx = args.auto_days
                start_idx = current_idx
            else:
                current_idx = self.learning_window
                start_idx = 0
            
            while current_idx + self.steps_head + self.output_window <= self.num_timesteps:
                if current_idx + self.steps_head + self.output_window == self.num_timesteps:
                     logger.info(f"Populating training data with {self.available_timesteps} days of data")
                
                if args.model == "DIME":
                    end_idx = current_idx + self.output_window
                else:
                    end_idx = current_idx

                for chnl in range(self.num_channels):
                    train_data_processed[sample_idx, chnl] = self.data[chnl, start_idx:end_idx]

                if self.lag:
                    train_data_processed[sample_idx, -1] = self.labels[0, start_idx:end_idx]

                label_data_processed[sample_idx, 0] = \
                    self.labels[0, current_idx + self.steps_head : current_idx + self.steps_head + self.output_window]

                current_idx += 1
                sample_idx += 1
                start_idx += 1

            if args.model in ["CONVLSTM", "DIME"]:
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
    
    def get_sample_dates(self, index:int, model:str) -> dict:
        """
        Returns a dictionary with the start and end date of both the input data and label window.
        """
        if self.start_date is None:
            raise ValueError("start_date was not provided during initialization.")
        
        # Compute key time indices
        data_start_idx = index

        if model == "DIME":
            data_end_idx = data_start_idx
        else:
            data_end_idx = self.learning_window + index  # last index of input window

        label_start_idx = data_end_idx + self.steps_head - 1
        label_end_idx = label_start_idx + self.output_window

        # Convert to dates
        data_start_date = self.start_date + timedelta(days=data_start_idx)
        data_end_date = self.start_date + timedelta(days=data_end_idx)
        
        label_start_date = self.start_date + timedelta(days=label_start_idx)
        label_end_date = self.start_date + timedelta(days=label_end_idx - 1)

        return {
            "data_window": {
                "start": data_start_date.strftime("%Y-%m-%d"),
                "end": data_end_date.strftime("%Y-%m-%d")
            },
            "label_window": {
                "start": label_start_date.strftime("%Y-%m-%d"),
                "end": label_end_date.strftime("%Y-%m-%d")
            }
        }

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
    def __init__(self, 
                config:dict, 
                args:dict, 
                data:np.array, 
                labels:np.array,
                start_date: str = None,
                autoencoder: Union[torch.nn.Module, None]=None,
                # past_data: Union[np.ndarray, None]= None,
                data_split:Union[str, None]=None):
        super().__init__(config, args, data, labels, start_date) 

        self.device = config.device
        self.time_list = self._add_time_list(data)
        self.autoencoder = autoencoder.to(self.device)
        self.config = config

        if args.conditioning == "climate":
            self.data = self.data[:, -1, :-1]

        elif args.conditioning != "none":

            if data_split is not None:
                
                filepath = os.path.join(config.data_dir, f"autoencoder_output")
                
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                file = os.path.join(filepath, f"vae_features_{data_split}_{args.step_length}.npy")

                if (os.path.exists(file)):
                    self.data = self._load_auto_output(file)
                    self.data, self.labels = self._align_auto_datasets(self.data, 
                                                                       self.labels)
                else:
                    autodata = self._collect_autoencoder_data(args, config, data, labels)
                    self.data = self._autoencoder_pipeline(args, 
                                                           autodata=autodata.data[:self.data.shape[0], :, -1],
                                                           file = file, 
                                                           export=True)
                if args.conditioning == "autoenc":
                    auto_channels = args.auto_days // 5
                    if self.data.shape[1]!= auto_channels:
                        logger.warning("Found different number of channels in data "
                                       "compared to autoencoder setting. Removing "
                                       "climate varaibles...")  
                        self.data = self.data[:, -auto_channels:]

                elif args.conditioning == "soil":
                    self.data = self.data[:, 4:]
                    logger.warning("Soil data only used as conditioning information. "
                                      "Removing climate variables...")
            else:
                logger.info("Avoiding using local autoencoder output and generating"
                            " new one")
                autodata = self._collect_autoencoder_data(args, config, data, labels)
                self.data = self._autoencoder_pipeline(args, 
                    autodata=autodata.data[:self.data.shape[0], :, -1],
                    file = None, 
                    export=False)
                
        if config.squared is True:
            self.data = self.data[:, :, :64, :64]
            self.labels = self.labels[:, :64, :64]
        
        logger.info(f"Data shape is {self.data.shape}")
        logger.info(f"Target shape is {self.labels.shape}")

    def _autoencoder_pipeline(self, 
            args, 
            autodata=None, 
            file=None, 
            export=True
        ):
        # calculate the reduced time series for all the n days up to t-1
        vae_output = self._reduce_data_vae(autodata=autodata,
            output_shape=args.auto_days//5) 
        
        extra_features, self.labels = self._align_auto_datasets(self.data, self.labels, vae_output)
        
        if args.conditioning == "all":
            data = np.concatenate([extra_features, vae_output], axis=1)
        elif args.conditioning == "autoenc":
            data = vae_output
        if export:
            self._export_auto_output(data, file)
        return data
    
    def _align_auto_datasets(self, data, labels, vae_data = None):
        if vae_data is not None:

            n_vae = vae_data.shape[0]
            n_samples = data.shape[0]

            if n_vae != n_samples:
                logger.warning(f"VAE output has different shape than data"
                            f"...proceeding with subsetting samples {n_samples} --> {n_vae}")    

                extra_features = data[-n_vae:, -1, :-1]
                labels = labels[-n_vae:]
        
            else:
                extra_features = data[:, -1, :-1]  

            return extra_features, labels
        
        else:
            n_data= data.shape[0]
            n_labels = labels.shape[0]

            if n_data != n_labels:
                logger.warning(f"VAE output has different shape than data"
                            f"...proceeding with subsetting samples {n_labels} --> {n_data}")
            labels = labels[-n_data:]
            return data, labels

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
    
    def _reshape_in_chunks(imag_past, batch_size):
        from tqdm.auto import tqdm
        b, t, h, w = imag_past.size()
        imag_past = imag_past.permute(0, 2, 3, 1)  # (b, h, w, t)
        reshaped = []

        # Iterate batchwise over the last dimension (t) with progress bar
        for batch_start in tqdm(range(0, t, batch_size), desc="Processing batches along t"):
            batch_end = min(batch_start + batch_size, t)
            batch = imag_past[..., batch_start:batch_end]  # Slice along t
            reshaped.append(batch.reshape(b * h * w, -1))  # Reshape the batch

        # Concatenate the results along the last dimension
        imag = torch.cat(reshaped, dim=1)
        return torch.unsqueeze(imag, dim=1)

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
            inputs = batch[0].to(self.device)
            with torch.no_grad():  # Disable gradient computation for inference
                encoded = autoencoder.encoder(inputs)
            encoded_batches.append(encoded)

        # Concatenate all encoded batches
        return torch.cat(encoded_batches)
    

    def _collect_autoencoder_data(self, args, model_config, data, target):
        original_features = args.feature_days
        original_step = args.step_length 
        args.feature_days = 180
        args.step_length = 0
        dataset = CustomConvLSTMDataset(model_config, args, data, target)
        args.feature_days = original_features
        args.step_length = original_step
        return dataset

    def _reduce_data_vae(self, output_shape, autodata=None):
        if autodata is None:
            logger.warning("No data provided to compute autoencoder output. "
                          f"Using the default data with shape {self.data.shape}"
                          f" taking the last channel (third dimension)")
            autodata = self.data[:, :, -1]

        from tqdm.auto import tqdm
        # if len(self.data.shape) == 5:
        #     self.data = np.swapaxes(self.data, 1, 2)
        imag_past = torch.from_numpy(autodata)
        b, t, h, w = imag_past.size()
        imag_past = imag_past.permute(0, 2, 3, 1)

        batch_size = int(512/1)

        outputs = []
        
        for batch_start in tqdm(range(0, b, batch_size), desc="Applying autoencoder in batches"):
            batch_end = min(batch_start + batch_size, b)
            temp_b = batch_end - batch_start
            batch = imag_past[batch_start:batch_end]  # Slice along t
            s = batch.reshape(temp_b*h*w, -1) 
            x = torch.unsqueeze(s, 1)
            reduced_data = self._apply_autoencoder(x, self.autoencoder)
            outputs.append(reduced_data)

        outputs = np.concatenate(outputs, 0)
        outputs = np.reshape(outputs, (b, h, w, output_shape))  # Reshape to (b, h, w, output_shape)
        outputs = np.transpose(outputs, (0, 3, 1, 2))  
        outputs = outputs + 1 ### to deal with normalization from -1,0 to 0,1
        return outputs

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
import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading
import logging

logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, config, patience=None, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience if patience is not None else config.patience
        self.min_patience =  getattr(config, "min_patience", 0)
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.current_checkpoint = None

    def __call__(self, val_loss, model_dict, epoch, save_path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if (self.counter >= self.patience) & (epoch > self.min_patience):
                self.early_stop = True
                self._cleanup_checkpoints(save_path)
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model_dict, epoch, save_path)
            self.counter = 0
    
    def find_checkpoints(self, directory, pattern):
        """Finds files matching the given pattern."""
        try:
            directories = [
                os.path.join(directory, entry) 
                for entry in os.listdir(directory)
            ]
            files = [
                os.path.join(directory, file) 
                for directory in directories 
                for file in os.listdir(directory) 
                if pattern in file
            ]
            return files
        except Exception as e:
            logger.error(f"Error while scanning directory: {e}")
            return []
        
    def save_checkpoint(self, val_loss, model_dict, epoch, save_path, n_save=3):
        """Saves model when validation loss decreases and removes older checkpoints."""
        
        checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch}")
        os.makedirs(checkpoint_path, exist_ok=True)

        if self.verbose:
            logger.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). "
                f"Saving model trained with {epoch} epochs..."
            )
        self.val_loss_min = val_loss
        
        
        for key, value in model_dict.items():
            if key == "epoch":
                continue
            temp_save_path = os.path.join(checkpoint_path, f"{key}_epoch_{model_dict['epoch']}.pth")
            torch.save(value, temp_save_path)

        metadata = {
            'epoch': model_dict['epoch'],
            'components': {
                key: os.path.join(checkpoint_path, f"{key}_epoch_{model_dict['epoch']}.pth")
                for key in model_dict if key != 'epoch'
            }
        }
        dest_path = os.path.join(checkpoint_path, f"metadata_epoch_{model_dict['epoch']}.pth")
        self.current_checkpoint = dest_path
        torch.save(metadata, dest_path)
        
        # Cleanup old checkpoints
        # self._cleanup_checkpoints(save_path, n_save)

    def remove_file_with_timeout(self, file, timeout=5):
        """
        Attempts to remove a file with a timeout.   
        Args:
            file (str): Path to the file to be removed.
            timeout (int): Maximum time to wait in seconds. 
        Returns:
            bool: True if the file was removed successfully, False otherwise.
        """
        success = []    
        def remove():
            try:
                os.remove(file)
                success.append(True)
            except Exception as e:
                logger.error(f"Failed to remove {file}: {e}")
                success.append(False)   

        thread = threading.Thread(target=remove)
        thread.start()
        thread.join(timeout)    
        if thread.is_alive():
            logger.error(f"Timeout reached while trying to remove {file}")
            thread.join()  # Cleanup the thread
            return False    
        return success[0] if success else False


    def _cleanup_checkpoints(self, checkpoint_dir, n_save=5):
        """Keeps only the n most recent checkpoints in the directory."""
        # Find all metadata files (primary references for checkpoints)
        metadata_files = self.find_checkpoints(checkpoint_dir, "metadata_epoch_")

        # Exclude the current checkpoint's metadata file
        metadata_files = [
            metadata for metadata in metadata_files if metadata != self.current_checkpoint
        ]
        logger.info(f"Found metadata files: {[os.path.basename(f) for f in metadata_files]}")

        if len(metadata_files) > n_save:
            # Sort metadata files by creation time (oldest first)
            metadata_files.sort(key=os.path.getctime)

            # Remove older checkpoints and associated files
            for metadata in metadata_files[:-n_save]:
                # Load metadata to find related component files
                try:
                    metadata_content = torch.load(metadata)
                    related_files = list(metadata_content['components'].values()) + [metadata]
                    for file in related_files:
                        if os.path.exists(file):
                            if self.remove_file_with_timeout(file):
                                logger.info(f"Removed: {os.path.basename(file)}")
                            else:
                                logger.error(f"Could not remove: {os.path.basename(file)}")

                    parent_dir = os.path.dirname(metadata)
                    if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                        os.rmdir(parent_dir)
                        logger.info(f"Removed empty directory: {parent_dir}")
                            
                except Exception as e:
                    logger.error(f"Error while cleaning up checkpoint {metadata}: {e}")


                    