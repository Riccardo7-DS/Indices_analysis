from analysis.deep_learning.dataset import MyDataset
from utils.function_clns import load_config, prepare, get_lat_lon_window, subsetting_pipeline, check_xarray_dataset
import xarray as xr
import os
import numpy as np
from scipy.sparse import linalg

import scipy.sparse as sp
from analysis.deep_learning.GWNET.pipeline_gwnet import main
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
import time

if __name__=="__main__":

    logger.remove()
    CONFIG_PATH = "config.yaml"
    # get the start time
    start = time.time()
    config = load_config(CONFIG_PATH)
    parser = argparse.ArgumentParser(conflict_handler="resolve")
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

    #for product in ["CHIRPS","GPCC","ERA5"]:
    #    parser.add_argument('--precp_product',type=str,default=product,help='precipitation product')
    #    for days in range(5, 15, 5):
    #        parser.add_argument('--forecast',type=int,default=days,help='days used to perform forecast')
    #        parser.add_argument('--seq_length',type=int,default=days,help='')
    #        args = parser.parse_args()
    #        main(args, CONFIG_PATH)
    #        torch.cuda.empty_cache()
    
    for product in [ "SPI_GPCC","SPI_ERA5","SPI_CHIRPS"]:
        parser.add_argument('--forecast',type=int,default=12,help='days used to perform forecast')
        parser.add_argument('--precp_product',type=str,default=product,help='precipitation product')
        parser.add_argument('--seq_length',type=int,default=12,help='')
        parser.add_argument('--spi',type=bool,default=True,help='if dataset is SPI')

        for late in [30, 60, 90, 180]:
            parser.add_argument('--latency',type=int,default=late,help='days used to accumulate precipitation for SPI')
            args = parser.parse_args()
            main(args, CONFIG_PATH)
            torch.cuda.empty_cache()
    

    end = time.time()
    total_time = end - start
    print("\n The script took "+ time.strftime("%H%M:%S", \
                                                    time.gmtime(total_time)) + "to run")
    