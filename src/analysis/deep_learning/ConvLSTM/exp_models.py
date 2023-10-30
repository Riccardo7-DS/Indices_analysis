if __name__=="__main__":
    from p_drought_indices.analysis.DeepLearning.GWNET.pipeline_gwnet import data_preparation
    import pickle
    import os
    import matplotlib.pyplot as plt
    from p_drought_indices.functions.function_clns import load_config, prepare, CNN_split, interpolate_prepare
    import numpy as np
    from p_drought_indices.analysis.DeepLearning.dataset import CustomDataset
    from torch.utils.data import DataLoader
    from loguru import logger
    import argparse
    from p_drought_indices.analysis.DeepLearning.ConvLSTM.pipeline_convlstm import training_lstm

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
    parser.add_argument('--batch_size',type=int,default=config["CONVLSTM"]["batch_size"],help='batch size')
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

    args = parser.parse_args()
    sub_precp, ds = data_preparation(args, CONFIG_PATH, precp_dataset=args.precp_product, ndvi_dataset="ndvi_smoothed_w2s.nc")
    
    from p_drought_indices.functions.function_clns import check_xarray_dataset
    print("Visualizing dataset before imputation...")
    #sub_precp = sub_precp.to_dataset()
    data, target = interpolate_prepare(args, sub_precp, ds)
    train_split = 0.8
    training_lstm(CONFIG_PATH, data, target, train_split = train_split)