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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    img_path = os.path.join(output_dir,  f"images")
    checkp_path = os.path.join(output_dir,  f"checkpoints")
    log_path = os.path.join(output_dir,  "logs")

    for sub_path in [img_path, checkp_path, log_path]:
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
    
    return output_dir, log_path, img_path, checkp_path


def data_preparation(args:dict, 
                     precp_dataset:str="ERA5", 
                     ndvi_dataset:str='ndvi_smoothed_w2s.nc'):
    
    from utils.function_clns import config, crop_image_left
    from analysis.configs.config_models import config as model_config

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

    if args.pipeline == "CONVLSTM":
        from analysis.configs.config_models import config as model_config
        logger.info(f"Starting NDVI prediction with product {config['CONVLSTM']['precp_product']} with {model_config.num_frames_input} days of features and {args.step_length} in the future...")
    else:
        logger.info(f"Starting NDVI prediction with product {config['GWNET']['precp_product']} with {args.forecast} days of features...")

    # Open the precipitation file with xarray
    precp_ds = prepare(xr.open_dataset(os.path.join(path, file)),countries=args.country, 
                        regions=args.region )\
                        .rio.write_crs(4326, inplace=True)

    var_target = [var for var in precp_ds.data_vars][0]
    precp_ds[var_target] = precp_ds[var_target].astype(np.float32)
    precp_ds[var_target].rio.write_nodata("nan", inplace=True)

    logger.info("The {p} raster has spatial dimensions: {r}"
                .format(p = precp_dataset, r= precp_ds.rio.resolution()))
    time_end = config['PRECIP'][precp_dataset]['date_end']
    time_start = config['PRECIP'][precp_dataset]['date_start']

    # Open the vegetation file with xarray
    dataset = prepare(xr.open_dataset(os.path.join(config['NDVI']['ndvi_path'], \
                        ndvi_dataset)),countries=args.country,regions=args.region)

    dataset["ndvi"] = dataset["ndvi"].transpose("time","lat","lon")
    dataset["ndvi"] = dataset["ndvi"].astype(np.float32)
    dataset["ndvi"].rio.write_nodata(np.nan, inplace=True)
    dataset = dataset.sel(time=slice(time_start,time_end))[["time","lat","lon","ndvi"]]
    
    logger.info("MSG NDVI dataset resolution: {}", dataset.rio.resolution())
    logger.info("{p} precipitation dataset resolution: {r}"
                .format(p=precp_dataset, r=precp_ds.rio.resolution()))

    ##### Normalization
    if args.normalize is True:

        ndvi_scaler = StandardScaler(mean=np.nanmean(dataset["ndvi"]), 
                                           std=np.nanstd(dataset["ndvi"]))
        
        precp_scaler = StandardScaler(mean=np.nanmean(precp_ds[var_target]), 
                                           std=np.nanstd(precp_ds[var_target]))
        
        dataset["ndvi"] = ndvi_scaler.transform(dataset["ndvi"])
        
        precp_ds[var_target] = precp_scaler.transform(precp_ds[var_target])
    
    else:
        ndvi_scaler = None
        
    if args.model == "GWNET":
        print("Selecting data for GCNN WaveNet")
        try:
            idx_lat, lat_max, idx_lon, lon_min = get_lat_lon_window(precp_ds, config['GWNET']['dim'])
            sub_precp = prepare(precp_ds).sel(time=slice(time_start,time_end))\
                .sel(lat=slice(lat_max, idx_lat), lon=slice(lon_min, idx_lon))
        except IndexError:
            logger.error("The dataset {} is out of bounds when using a subset, using original product"\
                         .format(args['GWNET']['precp_product']))
            sub_precp = prepare(precp_ds).sel(time=slice(time_start,time_end))
            args.dim = max(len(sub_precp["lat"]),len(sub_precp["lon"]))

    else:
        print("Selecting data for ConvLSTM")
        idx_lat, lat_max, idx_lon, lon_min = crop_image_left(precp_ds, config["CONVLSTM"]["dim"])
        sub_precp = prepare(precp_ds).sel(time=slice(time_start,time_end))\
            .sel(lat=slice(lat_max, idx_lat), lon=slice(lon_min, idx_lon))


    ds = dataset["ndvi"].rio.reproject_match(sub_precp[var_target]).rename({'x':'lon','y':'lat'})

    if args.pipeline=="CONVLSTM":
        return sub_precp, ds, ndvi_scaler
    
    else:
        sub_precp, ds = check_timeformat_arrays(sub_precp[var_target], ds)
        sub_precp = sub_precp.where(ds.notnull())
        return sub_precp, ds


def prepare_array_wgcnet(data):
    
    if len(data.shape)==3:
        logger.info("Adding one channel dimension")
        data = np.expand_dims(data, 1)

    # Get the dimensions of the input array
    T, C, lat, lon = data.shape

    # Reshape the array to combine the lat and lon dimensions into one
    data_reshaped = data.reshape(T, C, lat * lon).transpose(1, 0, 2) # C, T, V

    # data_lag = transform_array(data_reshaped, lag)
    return data_reshaped

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


def train_WeatherGCNet(args, checkpoint_path=None):
    import torch.nn.functional as F
    import numpy as np
    from definitions import ROOT_DIR
    from analysis import WGCNModel, train_loop, valid_loop, EarlyStopping
    from utils.function_clns import init_logging
    import torch
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    import os
    from analysis.configs.config_models import config_gwnet 

    data_dir = config_gwnet.data_dir+"/data_convlstm"
    data = np.load(os.path.join(data_dir, "data.npy"))
    target = np.load(os.path.join(data_dir, "target.npy"))
    with open(os.path.join(ROOT_DIR,"../data/ndvi_scaler.pickle"), "rb") as handle:
            ndvi_scaler = pickle.loads(handle.read())

    ################################# Module level logging #############################
    _, log_path, img_path, checkpoint_dir = create_runtime_paths(args)
    init_logging("training_gwnet", verbose=False, log_file=os.path.join(log_path, 
                                                      f"gwnet_days_{args.step_length}"
                                                      f"features_{args.feature_days}.log"))
    logger = logging.getLogger("training_gwnet")

    train_dl, valid_dl, dataset = get_train_valid_loader(config_gwnet, 
                                                         args, 
                                                         data, 
                                                         target)
    model = WGCNModel(args, config_gwnet, dataset.data).to(device)

    metrics_recorder = MetricsRecorder()
    train_records, valid_records, test_records = [], [], []
    rmse_train, rmse_valid, rmse_test = [], [], []
    mape_train, mape_valid, mape_test = [], [], []

    early_stopping = EarlyStopping(config_gwnet, logger, verbose=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_gwnet.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )

    loss_func = F.l1_loss
    loss_func_2 = F.mse_loss

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_sched'])
        checkp_epoch = checkpoint['epoch']
        logger.info(f"Resuming training from epoch {checkp_epoch}")

    start_epoch = 0 if checkpoint_path is None else checkp_epoch

    for epoch in range(start_epoch, config_gwnet.epochs):
        epoch_records = train_loop(config_gwnet, args, epoch, model, train_dl, loss_func, 
                            optimizer, scaler=ndvi_scaler, mask=None, draw_scatter=False)
        
        train_records.append(np.mean(epoch_records['loss']))
        rmse_train.append(np.mean(epoch_records['rmse']))
        mape_train.append(np.mean(epoch_records['mape']))

        metrics_recorder.add_train_metrics(np.mean(epoch_records['mape']), 
                                           np.mean(epoch_records['rmse']), 
                                           np.mean(epoch_records['loss']))
        
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
        logger.info(log.format(epoch, np.mean(epoch_records['loss']), 
                               np.mean(epoch_records['mape']), 
                               np.mean(epoch_records['rmse'])))
        
        mean_loss = sum(epoch_records['loss']) / len(epoch_records['loss'])
        scheduler.step(mean_loss)
        
        epoch_records = valid_loop(config_gwnet, args,  epoch, model, valid_dl, loss_func, 
                                   scaler=ndvi_scaler, mask=None, draw_scatter=args.scatterplot)
        
        valid_records.append(np.mean(epoch_records['loss']))
        rmse_valid.append(np.mean(epoch_records['rmse']))
        mape_valid.append(np.mean(epoch_records['mape']))

        log = 'Epoch: {:03d}, Val Loss: {:.4f}, Val MAPE: {:.4f}, Val RMSE: {:.4f}'
        logger.info(log.format(epoch, np.mean(epoch_records['loss']), 
                               np.mean(epoch_records['mape']), 
                               np.mean(epoch_records['rmse'])))
        
        metrics_recorder.add_val_metrics(np.mean(epoch_records['mape']), 
                                         np.mean(epoch_records['rmse']), 
                                         np.mean(epoch_records['loss']))
        
        model_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "lr_sched": scheduler.state_dict()
        }

        early_stopping( np.mean(epoch_records['loss']), 
                       model_dict, epoch, checkpoint_dir)
        
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

        plt.plot(range(epoch - start_epoch + 1), train_records, label='train')
        plt.plot(range(epoch - start_epoch + 1), valid_records, label='valid')
        plt.legend()
        plt.savefig(os.path.join(img_path, f'learning_curve_feat_'
                                 f'{args.feature_days}.png'))
        plt.close()

# # # # # # # # # #


def get_train_valid_loader(model_config, args, data, labels, add_lag:bool=True):
    from scipy.io import loadmat
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    from analysis import CustomConvLSTMDataset

    data = prepare_array_wgcnet(data)
    labels = prepare_array_wgcnet(labels)

    dataset = CustomConvLSTMDataset(model_config, args, data, labels)

    x = torch.tensor(dataset.data) #N, C, T, V (vertices e.g. lat-lon)
    y = torch.tensor(dataset.labels)

    train_val_dataset = TensorDataset(x, y)

    num_train = len(train_val_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))

    np.random.seed(123)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_val_dataset, 
                              batch_size=model_config.batch_size, 
                              sampler=train_sampler)
    valid_loader = DataLoader(train_val_dataset, 
                              batch_size=model_config.batch_size, 
                              sampler=valid_sampler)

    return train_loader, valid_loader, dataset



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

    def inverse_transform(self, norm_data):
        return (norm_data * (self.max - self.min)) + self.min

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
    plt.plot(epochs, metrics_recorder.train_mape, label='Train MAPE')
    plt.plot(epochs, metrics_recorder.val_mape, label='Validation MAPE')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE')
    plt.legend()
    plt.title('Mean Absolute Percentage Error (MAPE) vs. Epoch')
    plt.savefig(os.path.join(path, 'mape_vs_epoch.png'))
    plt.close()

    # Plot RMSE
    plt.figure()
    plt.plot(epochs, metrics_recorder.train_rmse, label='Train RMSE')
    plt.plot(epochs, metrics_recorder.val_rmse,  label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('Root Mean Squared Error (RMSE) vs. Epoch')
    plt.savefig(os.path.join(path, 'rmse_vs_epoch.png'))
    plt.close()

    # Plot Loss
    plt.figure()
    plt.plot(epochs, metrics_recorder.train_loss , label='Train Loss')
    plt.plot(epochs,metrics_recorder.val_loss  , label='Validation Loss')
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

    def add_train_metrics(self, mape, rmse, loss):
        self.train_mape.append(mape)
        self.train_rmse.append(rmse)
        self.train_loss.append(loss)

    def add_val_metrics(self, mape, rmse, loss):
        self.val_mape.append(mape)
        self.val_rmse.append(rmse)
        self.val_loss.append(loss)


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
        
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=model_config.learning_rate,
                                    weight_decay=model_config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer, factor=0.7, patience=3, verbose=False
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
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        # output = output.transpose(2,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, -1)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        # Check gradients for NaN or Infinity
        #for name, param in self.model.named_parameters():
        #    if param.grad is not None:
        #        print("Gradient Stats:", name, param.grad.mean().item(), param.grad.max().item(), param.grad.min().item())
        mape = masked_mape(predict,real, -1).item()
        rmse = masked_rmse(predict,real,-1).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        # output = output.transpose(2,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, -1)
        mape = masked_mape(predict,real,-1).item()
        rmse = masked_rmse(predict,real,-1).item()
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

def build_model(args, config):
    device = torch.device(config.GWNET.device)
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


    engine = trainer(scaler, config.GWNET.in_dim, config.GWNET.seq_length, args.num_nodes, config.GWNET.nhid, config.GWNET.dropout,
                         config.GWNET.learning_rate, config.GWNET.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit)
    return engine, scaler, dataloader, adj_mx

def gwnet_train_loop(model_config, engine, x, y):
    trainx = torch.Tensor(x).to(model_config.device)
    trainx= trainx.transpose(3, 2)
    trainy = torch.Tensor(y).to(model_config.device)
    trainy = trainy.transpose(3, 2)
    metrics = engine.train(trainx, trainy[:,0,:,:])
    return metrics

def gwnet_val_loop(model_config, engine, x, y):
    trainx = torch.Tensor(x).to(model_config.device)
    trainx= trainx.transpose(2, 3)
    trainy = torch.Tensor(y).to(model_config.device)
    trainy = trainy.transpose(2, 3)
    metrics = engine.eval(trainx, trainy[:,0,:,:])
    return metrics

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


def pipeline_wavenet(args, checkpoint_path):
    import pickle
    from analysis import EarlyStopping
    from analysis.configs.config_models import config_gwnet
    from definitions import ROOT_DIR
    import sys
    from utils.function_clns import init_logging

    _, log_path, img_path, checkpoint_dir = create_runtime_paths(args)
    init_logging("training_wavenet", verbose=True, log_file=os.path.join(log_path, 
                                                      f"wavenet_days_{args.step_length}"
                                                      f"features_{args.feature_days}.log"))
    logger = logging.getLogger("training_wavenet")

    logger.info(f"Starting training WaveNet model for {args.step_length} days in the future"
                f" with {args.feature_days} days of features")

    device = config_gwnet.device
    data_dir = config_gwnet.data_dir+"/data_convlstm"
    data = np.load(os.path.join(data_dir, "data.npy"))
    target = np.load(os.path.join(data_dir, "target.npy"))
    adj_mx_path = os.path.join(config_gwnet.adj_path, "adj_dist.pkl")
    with open(os.path.join(config_gwnet.data_dir, "ndvi_scaler.pickle"), "rb") as handle:
            scaler = pickle.loads(handle.read())

    if not os.path.exists(adj_mx_path):
        generate_adj_matrix(args)

    adj_mx = load_adj(adj_mx_path, args.adjtype)
    early_stopping = EarlyStopping(config_gwnet, logger, verbose=True)
        

    train_dl, valid_dl, dataset = get_train_valid_loader(config_gwnet, 
                                                         args, 
                                                         data, 
                                                         target)
    num_nodes = dataset.data.shape[-1]
                                                   
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    metrics_recorder = MetricsRecorder()

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = trainer(config_gwnet, scaler, args, 
                     num_nodes, supports, checkpoint_path)
    
    start_epoch = 0 if checkpoint_path is None else engine.checkp_epoch

    logger.info("Starting training...")
    
    his_loss, val_time, train_time = [],[],[]

    for epoch in range(start_epoch+1, config_gwnet.epochs+1):
        train_loss, train_mape, train_rmse = [],[],[]
        valid_loss, valid_mape, valid_rmse = [],[],[]

        t1 = time.time()
        # train_dl.shuffle()
        for iter, (x, y) in enumerate(train_dl):
            metrics = gwnet_train_loop(config_gwnet, engine, x, y)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            # if iter % config_gwnet.print_every == 0 :
            #     log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            #     logger.info(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]))
            if iter % config_gwnet.print_every == 0:
                logger.info(f"learning rate {engine.scheduler.get_last_lr()}")
        t2 = time.time()
        train_time.append(t2-t1)
        
        ###validation

        s1 = time.time()
        for iter, (x, y) in enumerate(valid_dl):
            metrics = gwnet_val_loop(config_gwnet, engine, x, y)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        
        s2 = time.time()
        # log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        # logger.info(log.format(epoch,(s2-s1)))
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

        save_figures(args=args, epoch=epoch-start_epoch, path=img_path, metrics_recorder=metrics_recorder)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}'
        logger.info(log.format(epoch, mtrain_loss, mtrain_rmse, mvalid_loss, mvalid_rmse))
        
        
        model_dict = {
            'epoch': epoch,
            'state_dict': engine.model.state_dict(),
            'optimizer': engine.optimizer.state_dict(),
            "lr_sched": engine.scheduler.state_dict()
        }

        mean_loss = sum(valid_loss) / len(valid_loss)
        engine.scheduler.step(mean_loss)

        early_stopping(mvalid_loss, model_dict, epoch, checkpoint_dir)
        
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    sys.exit(0)

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
  



def old_main(args, config):
    sub_precp, ds, _ =  data_preparation(args, precp_dataset=config.GWNET.precp_product)
    logger.info("Prepared data from {s} to {e}".format(s=sub_precp["time"].values[0], e=sub_precp["time"].values[-1]))

    print("Checking precipitation dataset...")
    check_xarray_dataset(args, sub_precp, save=True)
    print("Checking vegetation dataset...")
    check_xarray_dataset(args, ds, save=True)

    dataloader, num_nodes, x_df = get_dataloader(args, sub_precp, ds, check_matrix=True)
    epochs = config.GWNET.epochs
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