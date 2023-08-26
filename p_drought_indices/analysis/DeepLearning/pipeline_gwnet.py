from p_drought_indices.analysis.DeepLearning.dataset import MyDataset
from p_drought_indices.functions.function_clns import load_config, prepare, get_lat_lon_window, subsetting_pipeline, check_xarray_dataset
import xarray as xr
import os
import numpy as np
from scipy.sparse import linalg

import scipy.sparse as sp
from p_drought_indices.analysis.DeepLearning.gwnet import gwnet
import torch
import time
import argparse
import torch.optim as optim
import torch.nn as nn
from scipy.spatial.distance import cdist
from geopy.distance import geodesic
import pickle
import matplotlib.pyplot as plt

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

def data_preparation(CONFIG_PATH:str, precp_dataset:str="ERA5"):
    config = load_config(CONFIG_PATH)
    device = "cpu"

    config_directories = [config['SPI']['IMERG']['path'], config['SPI']['GPCC']['path'], config['SPI']['CHIRPS']['path'], config['SPI']['ERA5']['path'], config['SPI']['MSWEP']['path'] ]
    config_dir_precp = [config['PRECIP']['IMERG']['path'],config['PRECIP']['GPCC']['path'], config['PRECIP']['CHIRPS']['path'], config['PRECIP']['ERA5']['path'],  config['PRECIP']['TAMSTAT']['path'],config['PRECIP']['MSWEP']['path']]
    output_dir = os.path.join(config["DEFAULT"]["data"],  "graph_net")

    # Open the NetCDF file with xarray
    prod = precp_dataset
    path = config['SPI']['ERA5']['path']
    file = "era5_land_merged.nc" #f"ERA5_spi_gamma_{late}.nc"
    precp_ds = prepare(subsetting_pipeline(CONFIG_PATH, xr.open_dataset(os.path.join(path, file))))
    #precp_ds = precp_ds.reindex(lat=precp_ds['lat'][::-1])
    var_target = [var for var in precp_ds.data_vars][0] #"spi_gamma_{}".format(late)
    print(f"The {prod} raster has spatial dimensions:", precp_ds.rio.resolution())

    time_end = config['DEFAULT']['date_end']
    time_start = config['DEFAULT']['date_start']

    dim = config["GWNET"]["pixels"]

    dataset = prepare(xr.open_dataset(os.path.join(config['NDVI']['ndvi_path'], 'smoothed_ndvi_1.nc'))).sel(time=slice(time_start,time_end))[["time","lat","lon","ndvi"]]
    print("NDVI dataset resolution:", dataset.rio.resolution())
    print("Precipitation dataset resolution", precp_ds.rio.resolution())

    idx_lat, lat_max, idx_lon, lon_min = get_lat_lon_window(precp_ds, dim)
    sub_precp = prepare(precp_ds).sel(time=slice(time_start,time_end))\
        .sel(lat=slice(lat_max, idx_lat), lon=slice(lon_min, idx_lon))
    ds = dataset["ndvi"].rio.reproject_match(sub_precp[var_target]).rename({'x':'lon','y':'lat'})
    sub_precp = sub_precp[var_target].where(ds.notnull())
    return sub_precp, ds


def get_dataloader(CONFIG_PATH:str, sub_precp:xr.DataArray, ds:xr.DataArray):
    config = load_config(CONFIG_PATH)
    output_dir = os.path.join(config["DEFAULT"]["data"],  "graph_net")
    x_df = sub_precp.to_dataframe()
    x_df = x_df.swaplevel(1,2)
    x_df = x_df.dropna(subset={"tp"}).drop(columns={"spatial_ref"})
    x_df = x_df.sort_values(["lat", "lon","time"],ascending=False)

    data_x_unstack = x_df.unstack(["lat","lon"])
    #x_unstack = data_x_unstack.to_numpy()
    num_samples, num_nodes = data_x_unstack.shape
    x_unstack = np.expand_dims(data_x_unstack, axis=-1)
    print("The features have dimensions:", x_unstack.shape)

    y_df = ds.to_dataframe()
    y_df = y_df.dropna(subset={"ndvi"}).drop(columns={"spatial_ref"})
    y_df = y_df.sort_values(["lat", "lon","time"],ascending=False)
    y_df = y_df.reset_index().set_index(["time","lon","lat"])
    y_df = y_df[y_df.index.isin(x_df.index)]

    data_y_unstack = y_df.unstack(["lat","lon"])
    y_unstack = data_y_unstack.to_numpy()
    y_unstack = np.expand_dims(y_unstack, axis=-1)
    print("The instance have dimensions:", y_unstack.shape)

    st_df = x_df.reset_index()[["lon","lat"]].drop_duplicates()

    adj_dist = generate_adj_dist(st_df)
    with open(os.path.join(config["DEFAULT"]["data"], "graph_net/adj_dist.pkl"), 'wb') as f:
        pickle.dump(adj_dist, f, protocol=2)

    seq_length_x = seq_length_y = 12
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

    print("x shape: ", x.shape, ", y shape: ", y.shape)

    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

    batch_size = config["GWNET"]["batch_size"]
    dataloader = load_dataset(output_dir, batch_size, batch_size, batch_size)
    return dataloader, num_nodes

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

def save_figures(config:dict, epoch:int, train_loss:list, train_mape:list, train_rmse:list, test_loss:list, test_mape:list, test_rmse:list):
    epochs = list(range(1, epoch + 1))
    output = os.path.join(config["DEFAULT"]["output"], "gwnet_metrics")
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

def main(config):
    sub_precp, ds =  data_preparation(CONFIG_PATH)
    print("Checking precipitation dataset...")
    check_xarray_dataset(sub_precp)
    print("Checking vegetation dataset")
    check_xarray_dataset(ds)
    dataloader, num_nodes = get_dataloader(CONFIG_PATH, sub_precp, ds)
    epochs = config["GWNET"]["epochs"]
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    adj_path = os.path.join(config["DEFAULT"]["data"], "graph_net/adj_dist.pkl")
    adj_mx = load_adj(adj_path,  args.adjtype)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    metrics_recorder = MetricsRecorder()

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None



    engine = trainer(scaler, args.in_dim, args.seq_length, num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit)


    print("start training...",flush=True)
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
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
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
        print(log.format(i,(s2-s1)))
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

        save_figures(config=config, epoch=i, train_loss=metrics_recorder.train_loss, 
                    train_mape=metrics_recorder.train_mape, train_rmse=metrics_recorder.train_rmse, 
                    test_loss=metrics_recorder.val_loss, test_rmse=metrics_recorder.val_loss, 
                    test_mape=metrics_recorder.val_mape)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        #torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))


    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))


    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    #torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")


if __name__=="__main__":
    config = load_config(CONFIG_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument('--device',type=str,default='cpu',help='')
    #parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
    parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
    parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
    parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
    parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
    parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
    parser.add_argument('--seq_length',type=int,default=12,help='')
    parser.add_argument('--nhid',type=int,default=32,help='')
    parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
    #parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
    parser.add_argument('--batch_size',type=int,default=64,help='batch size')
    parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
    parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
    parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
    #parser.add_argument('--epochs',type=int,default=100,help='')
    parser.add_argument('--print_every',type=int,default=50,help='')
    #parser.add_argument('--seed',type=int,default=99,help='random seed')
    parser.add_argument('--save',type=str,default='notebooks/output',help='save path')
    parser.add_argument('--expid',type=int,default=1,help='experiment id')

    args = parser.parse_args()
    main(config)