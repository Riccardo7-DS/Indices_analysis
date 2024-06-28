from utils.function_clns import config, prepare, get_lat_lon_window, check_xarray_dataset, check_timeformat_arrays
import xarray as xr
import os
import numpy as np
from scipy.sparse import linalg

import scipy.sparse as sp
import torch
import time
import argparse
import torch.optim as optim
import torch.nn as nn
from scipy.spatial.distance import cdist
from geopy.distance import geodesic
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import logging
logger = logging.getLogger(__name__)


def init_tb(log_path):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=log_path)
    print("Tensorboard:  tensorboard --logdir="+ log_path)
    return writer

def create_runtime_paths(args:dict, spi:bool=False):
    from definitions import ROOT_DIR
    from utils.function_clns import config

    ### Create all the paths
    if args.model == "GWNET":
        output_dir = os.path.join(ROOT_DIR, "..",
                    f"output/gwnet/days_{args.step_length}/features_{args.feature_days}")
    elif args.model == "CONVLSTM": 
        output_dir = os.path.join(ROOT_DIR, "..",
                    f"output/convlstm/days_{args.step_length}/features_{args.feature_days}")
    elif args.model == "WNET":
        output_dir = os.path.join(ROOT_DIR, "..", 
                    f"output/wnet/days_{args.step_length}/features_{args.feature_days}")
    elif args.model == "DIME":
        output_dir = os.path.join(ROOT_DIR, "..", 
                    f"output/dime/days_{args.step_length}/features_{args.feature_days}")
    elif args.model == "AUTO_DIME":
        output_dir = os.path.join(ROOT_DIR, "..", 
                    f"output/dime/days_{args.step_length}/features_{args.feature_days}/autoencoder")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    img_path = os.path.join(output_dir,  f"images")
    checkp_path = os.path.join(output_dir,  f"checkpoints")
    log_path = os.path.join(output_dir,  "logs")

    for sub_path in [img_path, checkp_path, log_path]:
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
    
    return output_dir, log_path, img_path, checkp_path


def get_dataloader(args, dataset:xr.DataArray, 
                   ds:xr.DataArray, check_matrix:bool=False):
    
    from utils.function_clns import config

    x_df = dataset.to_dataframe()

    # if args.spi is False:
    #     x_df = x_df.swaplevel(1,2)
    # else:
    #     x_df = x_df.swaplevel(0,2)

    for col in ["spatial_ref","crs"]:
        if col in x_df:
            x_df.drop(columns={col}, inplace=True)
    var_target = dataset.name
    x_df = x_df.dropna(subset={var_target})
    x_df = x_df.sort_values(["lat", "lon","time"],ascending=False)

    data_x_unstack = x_df.unstack(["lat","lon"])
    print(data_x_unstack.isnull().sum())
    print(data_x_unstack.isnull().sum().sum())
    #x_unstack = data_x_unstack.to_numpy()
    num_samples, num_nodes = data_x_unstack.shape
    x_unstack = np.expand_dims(data_x_unstack, axis=-1)
    logger.info("The features have dimensions: {}", x_unstack.shape)

    y_df = ds.to_dataframe()
    for col in ["spatial_ref","crs"]:
        if col in y_df:
            y_df.drop(columns={col}, inplace=True)
    y_df = y_df.dropna(subset={"ndvi"})
    y_df = y_df.sort_values(["lat", "lon","time"],ascending=False)
    y_df = y_df.reset_index().set_index(["time","lon","lat"])
    y_df = y_df[y_df.index.isin(x_df.index)]

    data_y_unstack = y_df.unstack(["lat","lon"])
    print(data_y_unstack.isnull().sum())
    print(data_y_unstack.isnull().sum().sum())
    y_unstack = data_y_unstack.to_numpy()
    y_unstack = np.expand_dims(y_unstack, axis=-1)

    logger.info("The instance have dimensions: {}", y_unstack.shape)
    st_df = x_df.reset_index()[["lon","lat"]].drop_duplicates()

    dest_path = os.path.join(os.path.join(args.output_dir,  "adjacency_matrix"), \
        f"{args.precp_product}_{args.dim}_adj_dist.pkl")

    if os.path.isfile(dest_path) and check_matrix is True:
        logger.info("Using previously created adjacency matrix")
    else:
        adj_dist = generate_adj_dist(st_df)
        with open(dest_path, 'wb') as f:
            pickle.dump(adj_dist, f, protocol=2)
        logger.info(f"Created new adjacency matrix {config['GWNET']['precp_product']}_{config['GWNET']['dim']}")

    seq_length_x = seq_length_y = args.forecast
    y_start = 1

    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(y_start, (seq_length_y + 1), 1))

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(x_unstack[t + x_offsets, ...])
        y.append(y_unstack[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    logger.info("x shape: {}", x.shape) 
    logger.info("y shape: {}", y.shape)

    num_test = round(num_samples * config["GWNET"]["test_split"])
    num_train = round(num_samples * config["GWNET"]["train_split"])
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        logger.info("{cat} x: {x_shape}, y: {y_shape}".format(cat = cat, x_shape=_x.shape, y_shape=_y.shape))
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

    batch_size = config["GWNET"]["batch_size"]
    dataloader = load_dataset(args.output_dir, batch_size, batch_size, batch_size)
    return dataloader, num_nodes, data_y_unstack.iloc[num_train: num_test+num_train,:]


def check_shape_dataloaders(train_dataloader, val_dataloader):
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs = inputs.float()
        targets = targets.float()
        logger.info("Logging data info for train dataloader: "
            "Input shape: %s, Target shape: %s, "
            "Input max: %s, Input min: %s, "
            "Target max: %s, Target min: %s", 
            inputs.shape, targets.shape, 
            inputs.max().item(), inputs.min().item(),
            targets.max().item(), targets.min().item())
    
    
    for batch_idx, (inputs, targets) in enumerate(val_dataloader):
        inputs = inputs.float()
        targets = targets.float()
        logger.info("Logging data info for validation dataloader: "
            "Input shape: %s, Target shape: %s, "
            "Input max: %s, Input min: %s, "
            "Target max: %s, Target min: %s", 
            inputs.shape, targets.shape, 
            inputs.max().item(), inputs.min().item(),
            targets.max().item(), targets.min().item())

def prepare_array_wgcnet(data):
    
    if len(data.shape)==3:
        logger.info("Adding one channel dimension")
        data = np.expand_dims(data, 1)

    # Get the dimensions of the input array
    T, C, lat, lon = data.shape

    # Reshape the array to combine the lat and lon dimensions into one
    data_reshaped = data.reshape(T, C, lat * lon) # T, C, V

    # data_lag = transform_array(data_reshaped, lag)
    return data_reshaped

def print_lr_change(old_lr, scheduler):
    lr = scheduler.get_last_lr()[0]
    if lr != old_lr:
        logger.info(f"Learning rate changed: {old_lr} --> {lr}")
    return lr

def train_loop(config, args, model, train_loader, criterion, 
               optimizer, scaler=None, mask=None, draw_scatter:bool=False):
    from analysis.deep_learning.utils_gwnet import masked_mse_loss, masked_rmse, masked_mape, mask_mape, mask_rmse, MetricsRecorder, create_runtime_paths

    _, _, img_path, _ = create_runtime_paths(args)
    
    #from torcheval.metrics import R2Score
    model.train()
    epoch_records = {'loss': [], "mape":[], "rmse":[], "lr":[]}
    #metric = R2Score()

    if draw_scatter is True:
        nbins= 200
        bin0 = np.linspace(0, config.max_value, nbins+1)
        n = np.zeros((nbins,nbins))

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs = inputs.float().to(config.device)
        targets = torch.squeeze(targets.float().to(config.device))
        outputs = torch.squeeze(model(inputs))

        if draw_scatter is True:
            img_pred = outputs.detach().cpu().numpy()
            img_real = targets.detach().cpu().numpy()
            if args.normalize is True:
                img_pred = scaler.inverse_transform(img_pred)
                img_real = scaler.inverse_transform(img_real)
            h, xed, yed = evaluate_hist2d(img_real, img_pred, nbins)
            n = n+h

        if config.masked_loss is False:
            losses = criterion(outputs, targets)
            mape = masked_mape(outputs,targets).item()
            rmse = masked_rmse(outputs,targets).item()
        else:
            if mask is None:
                raise ValueError("Please provide a mask for loss computation")
            else:
                mask = mask.float().to(config.device)
                losses = masked_mse_loss(criterion, outputs, targets, mask)
                mape = mask_mape(outputs,targets, mask).item()
                rmse = mask_rmse(outputs,targets, mask).item()
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_records['loss'].append(losses.item())
        epoch_records["rmse"].append(rmse)
        epoch_records["mape"].append(mape)
        

    if draw_scatter is True:
        plot_scatter_hist(n,  bin0, img_path)

    return epoch_records


def valid_loop(config, args, model, valid_loader, criterion, scheduler,
               scaler=None, mask=None, draw_scatter:bool=False):
    
    from analysis.deep_learning.utils_gwnet import masked_mse_loss, masked_mape, masked_rmse, mask_mape, mask_rmse, masked_mse, MetricsRecorder, create_runtime_paths
    _, _, img_path, _ = create_runtime_paths(args)

    model.eval()
    epoch_records = {'loss': [], "mape":[], "rmse":[], "lr":[]}

    if draw_scatter is True:
        nbins= 200
        bin0 = np.linspace(0, config.max_value, nbins+1)
        n = np.zeros((nbins,nbins))

    num_batchs = len(valid_loader)
    for batch_idx, (inputs, targets) in enumerate(valid_loader):
        with torch.no_grad():
            inputs = inputs.float().to(config.device)
            #print(inputs.shape)
            targets = torch.squeeze(targets.float().to(config.device))
            outputs = torch.squeeze(model(inputs))

            if draw_scatter is True:
                img_pred = outputs.cpu().detach().numpy().flatten()
                img_real = targets.cpu().detach().numpy().flatten()
                if args.normalize is True:
                    img_pred = scaler.inverse_transform(img_pred)
                    img_real = scaler.inverse_transform(img_real)
                h, xed, yed = evaluate_hist2d(img_real, img_pred, nbins)
                n = n+h

            if config.masked_loss is False:
                losses = criterion(outputs, targets)
                mape = masked_mape(outputs,targets).item()
                rmse = masked_rmse(outputs,targets).item()
            else:
                if mask is None:
                    raise ValueError("Please provide a mask for loss computation")
                else:
                    mask = mask.float().to(config.device)
                    losses = masked_mse_loss(criterion, outputs, targets, mask)
                    mape = mask_mape(outputs,targets, mask).item()
                    rmse = mask_rmse(outputs,targets, mask).item()


            epoch_records['loss'].append(losses.item())
            epoch_records["rmse"].append(rmse)
            epoch_records["mape"].append(mape)

    ##### Step learning rate
    mean_loss = sum(epoch_records['loss']) / len(epoch_records['loss'])
    scheduler.step(mean_loss)
    learning_rate = scheduler.get_last_lr()[0]
    epoch_records["lr"].append(learning_rate)
            
    if draw_scatter is True:
        plot_scatter_hist(n,  bin0, img_path)
    
    return epoch_records


def evaluate_hist2d(real_img, pred_img, nbins):
    mdata=np.isnan(real_img)==0
    h, xed,yed=np.histogram2d(real_img[mdata], 
                pred_img[mdata], bins=nbins, density=None, weights=None)
    
    return h, xed, yed
    
def plot_scatter_hist(n, bin0, path=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import os
    n[n<=0]=np.nan
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    #ax = axes.ravel()
    #a0=ax[0].imshow(np.log10(n),origin='lower')
    a0=ax.pcolor(bin0,bin0,n,norm=LogNorm(vmin=1, vmax=np.nanmax(n)))
    plt.colorbar(a0)
    if path is not None:
        name = "scatterplot.png"
        plt.savefig(os.path.join(path,name))
    plt.show(block=False)
    plt.pause(2)
    plt.close()

def persistence_baseline(train_label):
    batch_rmse = []
    for idx in range(train_label.shape[1]- 1):
        target = train_label[:, idx+1, :, :]
        pred = train_label[:, idx, :, :]
        rmse = np.sqrt(np.mean((pred - target)**2))
        batch_rmse.append(rmse)
    logger.info("The baseline is a RMSE of {}".format(np.mean(batch_rmse)))


def get_train_valid_loader(model_config, args, data, labels, train_split:float=0.7, 
                           add_lag:bool=True):
    from scipy.io import loadmat
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    from analysis import CustomConvLSTMDataset
    from utils.function_clns import CNN_split

    data = prepare_array_wgcnet(data)
    labels = prepare_array_wgcnet(labels)
    

    train_data, val_data, train_label, val_label, \
        test_valid, test_label = CNN_split(data, labels, split_percentage=train_split)
    
    train_dataset = CustomConvLSTMDataset(model_config, args, 
                                          train_data, train_label, 
                                          save_files=True, filename="train_ds")
    
    val_dataset = CustomConvLSTMDataset(model_config, args, 
                                        val_data, val_label, 
                                        save_files=True, filename="val_ds")
    
    # create a DataLoader object that uses the dataset
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=model_config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=model_config.batch_size, shuffle=False)

    # dataset = CustomConvLSTMDataset(model_config, args, data, labels)

    # x = torch.tensor(dataset.data) #N, C, T, V (vertices e.g. lat-lon)
    # y = torch.tensor(dataset.labels)

    # train_val_dataset = TensorDataset(x, y)

    # num_train = len(train_val_dataset)
    # indices = list(range(num_train))
    # split = int(np.floor(0.1 * num_train))

    # np.random.seed(123)
    # np.random.shuffle(indices)

    # train_idx, valid_idx = indices[split:], indices[:split]
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    # train_loader = DataLoader(train_val_dataset, 
    #                           batch_size=model_config.batch_size, 
    #                           sampler=train_sampler)
    # valid_loader = DataLoader(train_val_dataset, 
    #                           batch_size=model_config.batch_size, 
    #                           sampler=valid_sampler)

    return train_dataloader, val_dataloader, train_dataset



class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data, null_value=None):
        transf = (data * self.std) + self.mean
        if null_value is not None:
            return transf.where(data != null_value, null_value)
        else:
            return transf
    

class StandardNormalizer():
    """
    Standard the input
    """
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        norm_data = (data - self.min) / (self.max - self.min)
        return norm_data

    def inverse_transform(self, norm_data, null_value=None):
        transf = (norm_data * (self.max - self.min)) + self.min 
        if null_value is not None:
            return transf.where(norm_data != null_value, null_value)
        else:
            return transf

def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=np.nanmean(data['x_train'][..., 0]), std=np.nanstd(data['x_train'][..., 0]))
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    adj_mx = load_pickle(pkl_filename)
    logger.info("Loading adjacency matrix...")
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj

"""
Metrics with null values
"""

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


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
    x = label.squeeze().reshape(-1)
    y = prediction.squeeze().reshape(-1)
    return pearsonr(x, y)

"""
Metrics with custom mask
"""

def mask_mae(preds, labels, mask):
    loss = torch.abs(preds-labels)
    mask = mask.float()
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def mask_rmse(preds, labels, mask):
    mse = mask_mse(preds=preds, labels=labels, mask=mask)
    return torch.sqrt(mse)

def mask_mape(preds, labels, mask):
    loss = torch.abs((preds-labels)/labels)
    mask = mask.float()
    loss = (loss * mask).sum() 
    #loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    non_zero_elements = mask.sum()
    mape_loss_val = loss / non_zero_elements
    return mape_loss_val

def mask_mse(preds, labels, mask):
    loss = (preds-labels)**2
    mask = mask.float()
    loss = (loss * mask).sum() 
    #loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    non_zero_elements = mask.sum()
    mse_loss_val = loss / non_zero_elements
    return mse_loss_val

def masked_mse_loss(criterion, preds, labels, mask):
    loss = criterion(preds, labels)
    mask = mask.float()
    loss = (loss * mask).sum() # gives \sigma_euclidean over unmasked elements
    non_zero_elements = mask.sum()
    mse_loss_val = loss / non_zero_elements
    return mse_loss_val


def save_figures(args:dict, 
                 epoch:int, 
                 path, 
                 metrics_recorder:dict):

    epochs = list(range(1, epoch + 1))

    plt.figure()
    plt.plot(epochs, metrics_recorder.train_mape, label='train MAPE')
    plt.plot(epochs, metrics_recorder.val_mape, label='validation MAPE')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE')
    plt.legend()
    plt.title('Mean Absolute Percentage Error (MAPE) vs. Epoch')
    plt.savefig(os.path.join(path, 'mape_vs_epoch.png'))
    plt.close()

    # Plot RMSE
    plt.figure()
    plt.plot(epochs, metrics_recorder.train_rmse, label='train RMSE')
    plt.plot(epochs, metrics_recorder.val_rmse,  label='validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('Root Mean Squared Error (RMSE) vs. Epoch')
    plt.savefig(os.path.join(path, 'rmse_vs_epoch.png'))
    plt.close()

    # Plot Loss
    plt.figure()
    plt.plot(epochs, metrics_recorder.train_loss , label='train Loss')
    plt.plot(epochs,metrics_recorder.val_loss  , label='validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epoch')
    plt.savefig(os.path.join(path, 'loss_vs_epoch.png'))
    plt.close()
    # logger.info(f"Figures saved for {args.precp_product} {args.forecast} days forecast")
    

class MetricsRecorder:
    def __init__(self):
        self.train_mape = []
        self.train_rmse = []
        self.train_loss = []
        self.val_mape = []
        self.val_rmse = []
        self.val_loss = []
        self.lr = []
        self.epoch = []

    def add_train_metrics(self, metrics, epoch):
        self.train_mape.append(np.mean(metrics['mape']))
        self.train_rmse.append(np.mean(metrics["rmse"]))
        self.train_loss.append(np.mean(metrics["loss"]))
        self.epoch.append(epoch)

    def add_val_metrics(self, metrics):
        self.val_mape.append(np.mean(metrics['mape']))
        self.val_rmse.append(np.mean(metrics["rmse"]))
        self.val_loss.append(np.mean(metrics["loss"]))
        self.lr.append(metrics["lr"][-1])

def update_tensorboard_scalars(writer, recorder:MetricsRecorder):
    writer.add_scalars('loss', {'train': recorder.train_loss[-1]}, recorder.epoch[-1])
    #writer.add_scalars('LossTR', {'trainD': train.lossd[-1]}, train.epoch[-1])
    writer.add_scalars('loss', {'valid': recorder.val_loss[-1]}, recorder.epoch[-1])
    #writer.add_scalars('LossVAL', {'validD': valid.lossd[-1]}, valid.epoch[-1])
    
    writer.add_scalars('rmse', {'train': recorder.train_rmse[-1]}, recorder.epoch[-1])
    writer.add_scalars('rmse', {'valid': recorder.val_rmse[-1]}, recorder.epoch[-1])
    writer.add_scalars('mape', {'train': recorder.train_mape[-1]}, recorder.epoch[-1])
    writer.add_scalars('mape', {'valid': recorder.val_mape[-1]}, recorder.epoch[-1])
    writer.add_scalars('lr', {'learning rate': recorder.lr[-1]}, recorder.epoch[-1])


class trainer():
    def __init__(self, model_config, scaler, args,
                 num_nodes:int, supports, checkpoint_path=None):
        
        from analysis import gwnet

        self.device = model_config.device
        self.model = gwnet(self.device, num_nodes, model_config.dropout, supports=supports, 
                           gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, 
                           aptinit=args.aptinit, in_dim=model_config.in_dim, 
                           out_dim=model_config.out_dim, 
                           residual_channels=model_config.nhid, 
                           dilation_channels=model_config.nhid,
                           skip_channels=model_config.nhid * 8,
                           end_channels=model_config.nhid * 16)
        
        self.learning_rate = model_config.learning_rate
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.learning_rate,
                                    weight_decay=model_config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer, factor=model_config.scheduler_factor, patience=model_config.scheduler_patience
        )
        self.loss = masked_mse
        self.scaler = scaler
        self.clip = 5

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['lr_sched'])
            self.checkp_epoch = checkpoint['epoch']
            logger.info(f"Resuming training from epoch {self.checkp_epoch}")

    def train(self, input, real_val):
        epoch_records = {'loss': [], "mape":[], "rmse":[], "lr":[]}
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        # output = output.transpose(2,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        # Check gradients for NaN or Infinity
        #for name, param in self.model.named_parameters():
        #    if param.grad is not None:
        #        print("Gradient Stats:", name, param.grad.mean().item(), param.grad.max().item(), param.grad.min().item())
        mape = masked_mape(predict,real).item()
        rmse = masked_rmse(predict,real).item()

        return loss, mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
       
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        # output = output.transpose(2,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real)
        mape = masked_mape(predict,real).item()
        rmse = masked_rmse(predict,real).item()
        return loss, mape, rmse
    
    def test(self, test_loader):
        self.model.eval()
        predictions = []
        targets = []
        test_loss = 0
        with torch.no_grad():
            for iter, (x, y) in enumerate(test_loader.get_iterator()):
                testx = torch.Tensor(x).to(self.device)
                testx = testx.transpose(1,3)
                y = torch.Tensor(y).to(self.device)
                y = y.transpose(1, 3)
                input = nn.functional.pad(testx, (1, 0, 0, 0))
                output = self.model(input).transpose(1, 3)
                real = torch.unsqueeze(y[:, 0, :, :], dim=1)
                loss = self.loss(output, real)
                test_loss += loss
                predictions.append(output) #[:,0,:,:].squeeze()
                targets.append(real)
            return (test_loss / test_loader.num_batch).item(), predictions, targets
        
    def schedule_learning_rate(self, mean_loss):
        self.scheduler.step(mean_loss)
        new_lr = print_lr_change(self.learning_rate, self.scheduler)
        self.learning_rate = new_lr

def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse

def gwnet_train_loop(model_config, engine, train_dl):
    epoch_records = {'loss': [], "mape":[], "rmse":[], "lr":[]}
    for iter, (x, y) in enumerate(train_dl):
        trainx = torch.Tensor(x).to(model_config.device)
        trainx= trainx.transpose(3, 2)
        trainy = torch.Tensor(y).to(model_config.device)
        trainy = trainy.transpose(3, 2)
        loss, mape, rmse = engine.train(trainx, trainy[:, 0,:,:])
        
        epoch_records["loss"].append(loss.item())
        epoch_records["rmse"].append(rmse)
        epoch_records["mape"].append(mape)
    return epoch_records

def gwnet_val_loop(model_config, engine, val_dl):
    epoch_records = {'loss': [], "mape":[], "rmse":[], "lr":[]}
    for iter, (x, y) in enumerate(val_dl):
        trainx = torch.Tensor(x).to(model_config.device)
        trainx= trainx.transpose(2, 3)
        trainy = torch.Tensor(y).to(model_config.device)
        trainy = trainy.transpose(2, 3)
        loss, rmse, mape = engine.eval(trainx, trainy[:, 0,:,:])

        epoch_records["loss"].append(loss.item())
        epoch_records["rmse"].append(rmse)
        epoch_records["mape"].append(mape)

    ##### Step learning rate
    mean_loss = sum(epoch_records['loss']) / len(epoch_records['loss'])
    engine.scheduler.step(mean_loss)
    engine.learning_rate = engine.scheduler.get_last_lr()[0]
    epoch_records["lr"].append(engine.learning_rate)

    return epoch_records

def generate_adj_dist(df, normalized_k=0.05,):
    coord = df[['lat', 'lon']].values
    dist_mx = cdist(coord, coord,
                   lambda u, v: geodesic(u, v).kilometers)
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))

    adj_mx[adj_mx < normalized_k] = 0
    return adj_mx

def generate_adj_matrix(args:dict, save_plot:bool=True):
    logger.info("Generating new adjacency matrix...")
    import seaborn as sns
    from analysis.configs.config_models import config_gwnet
    from utils.function_clns import config, crop_image_left

    output_dir, log_path, img_path, checkp_path = create_runtime_paths(args)

    path = os.path.join(config["DEFAULT"]["basepath"], "hydro_vars.zarr")
    data = xr.open_zarr(path).isel(time=0)["total_precipitation"]

    dim = config["GWNET"]["dim"]
    logger.debug(f"The adjacency matrix has coordinates {dim} x  {dim}")
    adj_path = config_gwnet.adj_path

    idx_lat, lat_max, idx_lon, lon_min = crop_image_left(data, dim)
    sub_dataset = data.sel(lat=slice(lat_max, idx_lat), 
                                          lon=slice(lon_min, idx_lon))

    df = sub_dataset.to_dataframe()

    adj_dist = generate_adj_dist(df.reset_index())
    with open(os.path.join(adj_path, "adj_dist.pkl"), 'wb') as f:
            pickle.dump(adj_dist, f, protocol=2)
    if save_plot is True:
        df = df.reset_index()
        df["lat_lon"] = df["lat"].astype(str) + ", " + df["lon"].astype(str)
        mask = np.zeros_like(adj_dist)
        mask[np.triu_indices_from(mask)] = True
        ax_labels = [df.iloc[i].lat_lon for i in range(adj_dist.shape[0])]
    
        sns.heatmap(adj_dist, mask=mask, cmap="YlGnBu", xticklabels=ax_labels, yticklabels=ax_labels)\
            .get_figure().savefig(os.path.join(img_path, "adj_dist.png"))

    

if __name__=="__main__":
    # import time
    # # get the start time

    start = time.time()
    from utils.function_clns import config
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('-f')

    parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
    parser.add_argument('--gcn_bool',default=True,help='whether to add graph convolution layer')
    parser.add_argument('--aptonly',action='store_true',help='whether   adaptive adj')
    parser.add_argument('--addaptadj',default=True,help='whether add adaptive adj')
    parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
    parser.add_argument('--aptinit',action=None)

    # parser.add_argument('--expid',type=int,default=1,help='experiment id')
    # parser.add_argument("--country", type=list, default=["Kenya", "Ethiopia","Somalia"], help="Location for dataset")
    # parser.add_argument("--region",type=list, default=None, help="region location for dataset")
    parser.add_argument("--model", type=str, default="WNET", help="DL model training")
    parser.add_argument('--step_length',type=int,default=15)
    parser.add_argument('--feature_days',type=int,default=30)
    parser.add_argument('--scatterplot',type=bool,default=True)
    parser.add_argument("--normalize", type=bool, default=True, help="Input data normalization")

    args = parser.parse_args()
    # main(args)

    # checkpoint = "src/output/gwnet/days_15/features_60/checkpoints/checkpoint_epoch_261.pth.tar"
    pipeline_wavenet(args)
    end = time.time()
    total_time = end - start
    print("\n The script took "+ time.strftime("%H%M:%S", \
                                                    time.gmtime(total_time)) + "to run")