from analysis.deep_learning.dataset import MyDataset
from utils.function_clns import load_config, prepare, get_lat_lon_window, subsetting_pipeline, check_xarray_dataset, check_timeformat_arrays, crop_image_right
import xarray as xr
import os
import numpy as np
from scipy.sparse import linalg

import scipy.sparse as sp
from analysis.deep_learning.GWNET.gwnet import gwnet
import torch
import time
import argparse
import torch.optim as optim
import torch.nn as nn
from scipy.spatial.distance import cdist
from geopy.distance import geodesic
import pickle
import matplotlib.pyplot as plt
from loguru import logger
import sys

CONFIG_PATH = "config.yaml"

def generate_adj_dist(df, normalized_k=0.05,):
    coord = df[['lat', 'lon']].values
    dist_mx = cdist(coord, coord,
                   lambda u, v: geodesic(u, v).kilometers)
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))

    adj_mx[adj_mx < normalized_k] = 0
    return adj_mx

def create_paths(args:dict, path:str, spi:bool=False):
    ### Create all the paths
    output_dir = os.path.join(path,  "graph_net")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    adj_path = os.path.join(output_dir,  "adjacency_matrix")
    if not os.path.exists(adj_path):
        os.makedirs(adj_path)

    log_path = os.path.join(output_dir,  "logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if args.spi==False:
        img_path = os.path.join(output_dir,  f"images_results/forecast_{args.forecast}")
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        checkp_path = os.path.join(output_dir,  f"checkpoints/forecast_{args.forecast}")
        if not os.path.exists(checkp_path):
            os.makedirs(checkp_path)
    
    else:
        img_path = os.path.join(output_dir,  f"images_results/forecast_{args.precp_product}_SPI_{args.latency}")
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        checkp_path = os.path.join(output_dir,  f"checkpoints/forecast_{args.precp_product}_SPI_{args.latency}")
        if not os.path.exists(checkp_path):
            os.makedirs(checkp_path)
    
    return output_dir, log_path


def data_preparation(args, CONFIG_PATH:str, precp_dataset:str="ERA5", ndvi_dataset:str='ndvi_smoothed_w2s.nc'):
    
    from utils.function_clns import crop_image_left

    config = load_config(CONFIG_PATH)

    config_directories = [config['SPI']['IMERG']['path'], config['SPI']['GPCC']['path'], 
                          config['SPI']['CHIRPS']['path'], config['SPI']['ERA5']['path'], config['SPI']['MSWEP']['path'] ]
    config_dir_precp = [config['PRECIP']['IMERG']['path'],config['PRECIP']['GPCC']['path'], config['PRECIP']['CHIRPS']['path'], 
                        config['PRECIP']['ERA5']['path'],  config['PRECIP']['TAMSTAT']['path'],config['PRECIP']['MSWEP']['path'],
                        config["PRECIP"]["ERA5_land"]["path"]]
    
    list_precp_prods = ["ERA5", "GPCC","CHIRPS","SPI_ERA5", "SPI_GPCC","SPI_CHIRPS","ERA5_land"]
    
    if precp_dataset not in list_precp_prods:
        raise ValueError(f"Precipitation product must be one of {list_precp_prods}")
    
    if "SPI" in precp_dataset:
        precp_dataset = precp_dataset.replace("SPI_","")
        path = [f for f in config_directories if precp_dataset in f][0]
        late = args.latency
        filename = "spi_gamma_{}".format(late)
        file = [f for f in os.listdir(path) if filename in f][0]
    else:
        path = [f for f in config_dir_precp if precp_dataset in f][0]
        file = f"{precp_dataset}_merged.nc"

    ### create all the paths
    output_dir, log_path = create_paths(args, path)
    args.output_dir = output_dir

    ### specify all the logging
    logger.remove()
    logger.add(sys.stderr, format = "{time:YYYY-MM-DD at HH:mm:ss} | <lvl>{level}</lvl> {level.icon} | <lvl>{message}</lvl>", colorize = True)
    if args.spi is False:
        logger_name = os.path.join(log_path, f"log_{precp_dataset}_{args.forecast}.log")
    else:
        logger_name = os.path.join(log_path, f"log_{precp_dataset}_spi_{args.latency}.log")
    if os.path.exists(logger_name): 
        os.remove(logger_name)
    logger.add(logger_name, format = "{time:YYYY-MM-DD at HH:mm:ss} | <lvl>{level}</lvl> {level.icon} | <lvl>{message}</lvl>", colorize = True)

    if args.spi is False:
        logger.info(f"Starting NDVI prediction with product {args.precp_product} with {args.forecast} days of features...")
    else:
        logger.info(f"Starting NDVI prediction with product {args.precp_product} {filename} with {args.forecast} days of features...")

    # Open the precipitation to use for reprojection file with xarray
    #path_oth = config['PRECIP']["ERA5"]['path']
    #file_oth = [f for f in os.listdir(path_oth) if f.endswith(".nc") and "merged" in f ][0]
    #era5_ds = prepare(subsetting_pipeline(CONFIG_PATH, xr.open_dataset(os.path.join(path_oth, file_oth)),countries=None, regions=args.location))
    #var_era5 = [var for var in era5_ds.data_vars][0]

    # Open the precipitation file with xarray
    precp_ds = prepare(subsetting_pipeline(CONFIG_PATH, xr.open_dataset(os.path.join(path, file)),countries=args.country, regions=args.region ))\
        .rio.write_crs(4326, inplace=True)

    #precp_ds = prepare(xr.open_dataset(os.path.join(path, file))).sel(lon=slice(33.099998474121094, 42.900001525878906), lat=slice(10.300000190734863, 3.5999999046325684, ))

    var_target = [var for var in precp_ds.data_vars][0]
    precp_ds[var_target] = precp_ds[var_target].astype(np.float32)
    precp_ds[var_target].rio.write_nodata("nan", inplace=True)
    #precp_ds[var_target] = precp_ds[var_target].rio.reproject_match(era5_ds[var_era5]).rename({'x':'lon','y':'lat'})

    logger.info("The {p} raster has spatial dimensions: {r}".format(p = precp_dataset, r= precp_ds.rio.resolution()))
    time_end = config['PRECIP'][precp_dataset]['date_end']
    time_start = config['PRECIP'][precp_dataset]['date_start']

    # Open the vegetation file with xarray
    dataset = prepare(subsetting_pipeline(CONFIG_PATH, xr.open_dataset(os.path.join(config['NDVI']['ndvi_path'], ndvi_dataset)),countries=args.country, regions=args.region))#.rio.write_crs(4326, inplace=True)
    #dataset = prepare(xr.open_dataset(os.path.join(config['NDVI']['ndvi_path'], ndvi_dataset))).sel(lon=slice(33.099998474121094, 42.900001525878906), lat=slice(3.5999999046325684, 10.300000190734863))
    dataset["ndvi"] = dataset["ndvi"].transpose("time","lat","lon")
    dataset["ndvi"] = dataset["ndvi"].astype(np.float32)
    dataset["ndvi"].rio.write_nodata(np.nan, inplace=True)
    dataset = dataset.sel(time=slice(time_start,time_end))[["time","lat","lon","ndvi"]]
    logger.info("MSG NDVI dataset resolution: {}", dataset.rio.resolution())
    logger.info("{p} precipitation dataset resolution: {r}".format(p=precp_dataset, r=precp_ds.rio.resolution()))

    if args.convlstm is False:
        print("Selecting data for GCNN WaveNet")
        try:
            idx_lat, lat_max, idx_lon, lon_min = get_lat_lon_window(precp_ds, args.dim)
            sub_precp = prepare(precp_ds).sel(time=slice(time_start,time_end))\
                .sel(lat=slice(lat_max, idx_lat), lon=slice(lon_min, idx_lon))
        except IndexError:
            logger.error("The dataset {} is out of bounds when using a subset, using original product".format(args.precp_product))
            sub_precp = prepare(precp_ds).sel(time=slice(time_start,time_end))
            args.dim = max(len(sub_precp["lat"]),len(sub_precp["lon"]))

    else:
        print("Selecting data for ConvLSTM")
        idx_lat, lat_max, idx_lon, lon_min = crop_image_left(precp_ds, args.dim)
        sub_precp = prepare(precp_ds).sel(time=slice(time_start,time_end))\
            .sel(lat=slice(lat_max, idx_lat), lon=slice(lon_min, idx_lon))


    ds = dataset["ndvi"].rio.reproject_match(sub_precp[var_target]).rename({'x':'lon','y':'lat'})
    
    if args.convlstm is True:
        return sub_precp, ds
    
    else:
        sub_precp, ds = check_timeformat_arrays(sub_precp[var_target], ds)
        sub_precp = sub_precp.where(ds.notnull())
        return sub_precp, ds


def get_dataloader(args, CONFIG_PATH:str, sub_precp:xr.DataArray, ds:xr.DataArray, check_matrix:bool=False):
    config = load_config(CONFIG_PATH)

    x_df = sub_precp.to_dataframe()

    if args.spi is False:
        x_df = x_df.swaplevel(1,2)
    else:
        x_df = x_df.swaplevel(0,2)
    for col in ["spatial_ref","crs"]:
        if col in x_df:
            x_df.drop(columns={col}, inplace=True)
    var_target = sub_precp.name
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

    dest_path = os.path.join(os.path.join(args.output_dir,  "adjacency_matrix"), f"{args.precp_product}_{args.dim}_adj_dist.pkl")

    if os.path.isfile(dest_path) and check_matrix is True:
        logger.info("Using previously created adjacency matrix")
    else:
        adj_dist = generate_adj_dist(st_df)
        with open(dest_path, 'wb') as f:
            pickle.dump(adj_dist, f, protocol=2)
        logger.info(f"Created new adjacency matrix {args.precp_product}_{args.dim}")

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

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

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

def save_figures(args:dict, epoch:int, train_loss:list, train_mape:list, train_rmse:list, test_loss:list, test_mape:list, test_rmse:list):
    epochs = list(range(1, epoch + 1))
    if args.spi is False:
        output = os.path.join(args.output_dir,  f"images_results/forecast_{args.forecast}")
    else:
        output = os.path.join(args.output_dir,  f"images_results/forecast_{args.precp_product}_SPI_{args.latency}")

    # Plot MAPE
    plt.figure()
    plt.plot(epochs, train_mape, label='Train MAPE')
    plt.plot(epochs, test_mape, label='Validation MAPE')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE')
    plt.legend()
    plt.title('Mean Absolute Percentage Error (MAPE) vs. Epoch')
    plt.savefig(os.path.join(output, 'mape_vs_epoch.png'))
    plt.close()

    # Plot RMSE
    plt.figure()
    plt.plot(epochs, train_rmse, label='Train RMSE')
    plt.plot(epochs, test_rmse, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('Root Mean Squared Error (RMSE) vs. Epoch')
    plt.savefig(os.path.join(output, 'rmse_vs_epoch.png'))
    plt.close()

    # Plot Loss
    plt.figure()
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epoch')
    plt.savefig(os.path.join(output, 'loss_vs_epoch.png'))
    plt.close()
    logger.info(f"Figures saved for {args.precp_product} {args.forecast} days forecast")
    

class MetricsRecorder:
    def __init__(self):
        self.train_mape = []
        self.train_rmse = []
        self.train_loss = []
        self.val_mape = []
        self.val_rmse = []
        self.val_loss = []

    def add_train_metrics(self, mape, rmse, loss):
        self.train_mape.append(mape)
        self.train_rmse.append(rmse)
        self.train_loss.append(loss)

    def add_val_metrics(self, mape, rmse, loss):
        self.val_mape.append(mape)
        self.val_rmse.append(rmse)
        self.val_loss.append(loss)


class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = masked_mae
        self.scaler = scaler
        self.clip = 5
        self.device = device

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        # Check gradients for NaN or Infinity
        #for name, param in self.model.named_parameters():
        #    if param.grad is not None:
        #        print("Gradient Stats:", name, param.grad.mean().item(), param.grad.max().item(), param.grad.min().item())
        mape = masked_mape(predict,real,0.0).item()
        rmse = masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = masked_mape(predict,real,0.0).item()
        rmse = masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
    
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

def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse

def build_model(args):
    device = torch.device(args.device)
    adj_mx = load_adj(args.adjdata,args.adjtype)
    dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None


    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit)
    return engine, scaler, dataloader, adj_mx

def main(args, CONFIG_PATH):
    config = load_config(CONFIG_PATH)
    sub_precp, ds =  data_preparation(args, CONFIG_PATH, precp_dataset=args.precp_product)
    logger.info("Prepared data from {s} to {e}".format(s=sub_precp["time"].values[0], e=sub_precp["time"].values[-1]))

    print("Checking precipitation dataset...")
    check_xarray_dataset(args, sub_precp, save=True)
    print("Checking vegetation dataset...")
    check_xarray_dataset(args, ds, save=True)

    dataloader, num_nodes, x_df = get_dataloader(args, CONFIG_PATH, sub_precp, ds, check_matrix=True)
    epochs = config["GWNET"]["epochs"]
    device = torch.device(args.device)
    adj_path = os.path.join(os.path.join(args.output_dir,  "adjacency_matrix"), f"{args.precp_product}_{args.dim}_adj_dist.pkl")
    adj_mx = load_adj(adj_path,  args.adjtype)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    metrics_recorder = MetricsRecorder()

    dictionary = {'predFile': args.output_dir + '/predicted_data/data/' + "pred_{p}_{f}".format(f=str(args.precp_product), 
                                                                                          p=str(args.forecast)) + '.pkl',
                  'targetFile': args.output_dir + '/predicted_data/data/' + "target_{p}_{f}".format(f=str(args.precp_product), 
                                                                                          p=str(args.forecast)) + '.pkl',}

    if args.spi is True:
        checkp_path = os.path.join(args.output_dir,  f"checkpoints/forecast_{args.precp_product}_SPI_{args.latency}")
    else:
        checkp_path = os.path.join(args.output_dir,  f"checkpoints/forecast_{args.forecast}")

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = trainer(scaler, args.in_dim, args.seq_length, num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit)


    logger.info("Starting training...",flush=True)
    
    his_loss =[]
    val_time = []
    train_time = []
    for i in range(1,epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                logger.info(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []


        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        logger.info(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        metrics_recorder.add_train_metrics(mtrain_mape, mtrain_rmse, mtrain_loss)


        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)
        metrics_recorder.add_val_metrics(mvalid_mape, mvalid_rmse, mvalid_loss)

        save_figures(args=args, epoch=i, train_loss=metrics_recorder.train_loss, 
                    train_mape=metrics_recorder.train_mape, train_rmse=metrics_recorder.train_rmse, 
                    test_loss=metrics_recorder.val_loss, test_rmse=metrics_recorder.val_loss, 
                    test_mape=metrics_recorder.val_mape)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        logger.info(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), checkp_path+"/checkpoints_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
    logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(checkp_path +"/checkpoints_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))

    print(engine.model)

    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    test_loss, predictions, targets = engine.test(dataloader["test_loader"])

    yhat = torch.cat(predictions,dim=0).squeeze()
    yhat = yhat[:realy.size(0),...]

    output = open(dictionary['predFile'], 'wb')
    pickle.dump(yhat.cpu().detach().numpy(), output)
    output.close()

    target = open(dictionary['targetFile'], 'wb')
    pickle.dump(realy.cpu().detach().numpy(), target)
    target.close()

    logger.success("Training finished")
    logger.info("The valid loss on best model is {}".format(str(round(his_loss[bestid],4))))

    amae = []
    amape = []
    armse = []
    for i in range(args.seq_length):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = metric(pred,real) 
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        logger.info(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    logger.info(log.format(args.forecast, np.mean(amae),np.mean(amape),np.mean(armse)))
    torch.save(engine.model.state_dict(), checkp_path +"/checkpoints_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")


if __name__=="__main__":
    import time
    # get the start time
    product = "ERA5_land"
    start = time.time()
    config = load_config(CONFIG_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument('--device',type=str,default='cuda',help='')
    parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
    parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
    parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
    parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
    parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
    parser.add_argument('--nhid',type=int,default=32,help='')
    parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
    parser.add_argument('--batch_size',type=int,default=config["GWNET"]["batch_size"],help='batch size')
    parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
    parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
    parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
    parser.add_argument('--print_every',type=int,default=50,help='Steps before printing')
    parser.add_argument('--expid',type=int,default=1,help='experiment id')
    parser.add_argument('--latency',type=int,default=90,help='days used to accumulate precipitation for SPI')
    parser.add_argument('--spi',type=bool,default=False,help='if dataset is SPI')
    parser.add_argument('--precp_product',type=str,default=product,help='precipitation product')
    parser.add_argument('--forecast',type=int,default=12,help='days used to perform forecast')
    parser.add_argument('--seq_length',type=int,default=12,help='')
    parser.add_argument("--country", type=list, default=["Kenya", "Ethiopia","Somalia"], help="Location for dataset")
    parser.add_argument("--region",type=list, default=None, help="region location for dataset") #"Oromia","SNNPR","Gambela" 
    parser.add_argument("--dim", type=int, default= config["GWNET"]["pixels"], help="")
    parser.add_argument("--convlstm", type=bool, default= False, help="")
    

    args = parser.parse_args()
    main(args, CONFIG_PATH)
    end = time.time()
    total_time = end - start
    print("\n The script took "+ time.strftime("%H%M:%S", \
                                                    time.gmtime(total_time)) + "to run")