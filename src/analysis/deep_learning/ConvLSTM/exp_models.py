if __name__=="__main__":
    from analysis.deep_learning.GWNET.pipeline_gwnet import data_preparation, create_paths
    import argparse
    from utils.function_clns import config, prepare, CNN_split, interpolate_prepare
    import numpy as np
    from analysis.deep_learning.ConvLSTM.pipeline_convlstm import training_convlstm
    import torch
    import xarray as xr
    import os
    import sys
    from loguru import logger

    product = "ERA5"
    parser = argparse.ArgumentParser(description='test', conflict_handler="resolve")
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
    
    parser.add_argument("--pipeline", type=str, default= "CONVLSTM", help="")
    parser.add_argument("--country", type=list, default=["Kenya","Somalia","Ethiopia"], help="Location for dataset")
    parser.add_argument("--region", type=list, default=None, help="Location for dataset")
    parser.add_argument("--normalize", type=bool, default=False, help="Input data normalization")
    parser.add_argument("--scatterplot", type=bool, default=False, help="Whether to visualize scatterplot")
    args = parser.parse_args()

    for day in range(4, 8):

        parser.add_argument('--step_length',type=int,default=day,help='days in the future')
        args = parser.parse_args()

        ### create all the paths
        _, log_path, _, _ = create_paths(args)
        ### specify all the logging
        logger.remove()
        logger.add(sys.stderr, format = "{time:YYYY-MM-DD at HH:mm:ss} | <lvl>{level}</lvl> {level.icon} | <lvl>{message}</lvl>", colorize = True)
        logger_name = os.path.join(log_path, f"log_{args.step_length}.log")

        if os.path.exists(logger_name): 
            os.remove(logger_name)
        logger.add(logger_name, format = "{time:YYYY-MM-DD at HH:mm:ss} | <lvl>{level}</lvl> {level.icon} | <lvl>{message}</lvl>", colorize = True)

        ### Pipeline
        sub_precp, ds, ndvi_scaler = data_preparation(args, 
                                                      precp_dataset=config[args.pipeline]['precp_product'], 
                                                      ndvi_dataset="ndvi_smoothed_w2s.nc")

        from ancillary_vars.esa_landuse import drop_water_bodies_esa_downsample
        mask_ds = drop_water_bodies_esa_downsample(ds.isel(time=0))
        mask = torch.tensor(np.array(xr.where(mask_ds.notnull(), 1, 0)))

        logger.info("Visualizing dataset before imputation...")
        data, target = interpolate_prepare(args, sub_precp, ds, interpolate=True)
        train_split = 0.7
        training_convlstm(args, logger, data, target, mask=mask, train_split = train_split, 
                      ndvi_scaler=ndvi_scaler)

