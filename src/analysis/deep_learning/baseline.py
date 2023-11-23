from analysis.deep_learning.GWNET.pipeline_gwnet import data_preparation 
import pickle
import os
import matplotlib.pyplot as plt
from utils.function_clns import load_config, prepare, CNN_split, interpolate_prepare
import numpy as np
from analysis.deep_learning.dataset import CustomDataset
from torch.utils.data import DataLoader
from loguru import logger
import xarray as xr
import numpy as np
import torch
import os
from utils.function_clns import load_config, subsetting_pipeline, interpolate_prepare, prepare, CNN_split, CNN_preprocessing, get_lat_lon_window
import numpy as np
import torch
from torch.utils.data import Dataset
from analysis.deep_learning.dataset import CustomConvLSTMDataset
from torch.utils.data import DataLoader
import pickle
import argparse
from tqdm.auto import tqdm
from typing import Union

if __name__=="__main__":
    product = "ERA5"
    CONFIG_PATH = "config.yaml"
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
    #parser.add_argument("--location", type=list, default=["Amhara"], help="Location for dataset")
    parser.add_argument("--dim", type=int, default= config["CONVLSTM"]["pixels"], help="")
    parser.add_argument("--convlstm", type=bool, default= True, help="")
    parser.add_argument("--country", type=list, default=["Kenya","Somalia","Ethiopia"], help="Location for dataset")
    parser.add_argument("--region", type=list, default=None, help="Location for dataset")
    parser.add_argument("--normalize", type=bool, default=True, help="Input data normalization")
    args = parser.parse_args()
    sub_precp, ds = data_preparation(args, CONFIG_PATH, precp_dataset=args.precp_product, ndvi_dataset="ndvi_smoothed_w2s.nc")

    mask = torch.tensor(np.array(xr.where(ds.isel(time=0).notnull(), 1, 0)))

    print("Visualizing dataset before imputation...")
    #sub_precp = sub_precp.to_dataset()
    data, target = interpolate_prepare(args, sub_precp, ds, interpolate=True)
    train_split = 0.7

    #### training parameters
    train_data, val_data, train_label, val_label, test_valid, test_label = CNN_split(data, target, 
                                                               split_percentage=train_split)
    
    batch_rmse = []
    for idx in range(train_label.shape[1]- 1):
        target = train_label[:, idx+1, :, :]
        pred = train_label[:, idx, :, :]
        rmse = np.sqrt(np.mean((pred - target)**2))
        batch_rmse.append(rmse)
    
    print("The baseline is a RMSE of {}".format(np.mean(batch_rmse)))

        


