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

def spi_ndvi_convlstm(CONFIG_PATH, time_start, time_end):
    config_file = load_config(CONFIG_PATH=CONFIG_PATH)

    # Open the NetCDF file with xarray
    dataset = prepare(xr.open_dataset(os.path.join(config_file['NDVI']['ndvi_path'], 'ndvi_smoothed_w2s.nc'))).sel(time=slice(time_start,time_end))[["time","lat","lon","ndvi"]]

    prod = "ERA5"
    late = 90

    path = config_file['PRECIP']['ERA5']['path']
    file = "ERA5_merged.nc" #"era5_land_merged.nc" #f"ERA5_spi_gamma_{late}.nc"
    precp_ds = prepare(subsetting_pipeline(CONFIG_PATH, xr.open_dataset(os.path.join(path, file))))
    var_target = [var for var in precp_ds.data_vars][0] #"spi_gamma_{}".format(late)
    print(f"The {prod} raster has spatial dimensions:", precp_ds.rio.resolution())

    #### training parameters
    train_split = 0.8
    dim=64
    preprocess_type="None"

    ### Load dataset
    file_path = os.path.join(config_file["DEFAULT"]["data"],'preprocessed_data.pkl')
    if os.path.exists(file_path):
        print("The file exists. Proceeding with the analysis")
        with open(file_path, 'rb') as file:
            train_data, test_data, train_label, test_label = pickle.load(file)
    else:
        print("The file does not exist. Proceeding with preprocessing")
        idx_lat, lat_max, idx_lon, lon_min = get_lat_lon_window(precp_ds, dim)
        sub_precp = prepare(precp_ds).sel(time=slice(time_start,time_end))\
            .sel(lat=slice(lat_max, idx_lat), lon=slice(lon_min, idx_lon))
        ds = dataset["ndvi"].rio.reproject_match(sub_precp[var_target])#.rename({'x':'lon','y':'lat'})
        

        train_data, test_data, train_label, test_label = CNN_preprocessing(ds, sub_precp, var_origin="ndvi", var_target=var_target, preprocess_type=preprocess_type,  split=train_split)
        # Save the image data using pickle
        with open(file_path, 'wb') as file:
            pickle.dump((train_data, test_data, train_label, test_label), file)
        print("Data written to pickle file")
    
    return sub_precp, ds


def training_lstm(CONFIG_PATH:str, data:np.array, target:np.array,
                  mask:Union[None, np.array]=None, train_split:float = 0.7):
    import numpy as np
    #from configs.config_3x3_32_3x3_64_3x3_128 import config
    from configs.config_3x3_16_3x3_32_3x3_64 import config
    from analysis.deep_learning.GWNET.pipeline_gwnet import MetricsRecorder, masked_mse_loss
    from torch.nn import MSELoss
    import matplotlib.pyplot as plt
    from analysis.deep_learning.ConvLSTM.ConvLSTM import train_loop, valid_loop, build_logging# ConvLSTM
    import numpy as np
    from utils.function_clns import load_config
    from analysis.deep_learning.dataset import EarlyStopping

    #### training parameters
    train_data, val_data, train_label, val_label, test_valid, test_label = CNN_split(data, target, 
                                                               split_percentage=train_split)
    
    batch_size = config.batch_size
    early_stopping = EarlyStopping(config, verbose=True)

    # create a CustomDataset object using the reshaped input data
    train_dataset = CustomConvLSTMDataset(config, train_data, train_label)
    val_dataset = CustomConvLSTMDataset(config, val_data, val_label)
    
    # create a DataLoader object that uses the dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    ### check shape of data
    
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs = inputs.float()
        targets = targets.float()
        print(inputs.shape, targets.shape, inputs.max(), inputs.min())
    
    
    for batch_idx, (inputs, targets) in enumerate(val_dataloader):
        inputs = inputs.float()
        targets = targets.float()
        print(inputs.shape, targets.shape, inputs.max(), inputs.min())


    #### Start training
    
    name = '3x3_16_3x3_32_3x3_64'
    #name = "3x3_32_3x3_64_3x3_128"

    ### parrameters for early stopping 
    # Define best_score, counter, and patience for early stopping:
    
    logger = build_logging(config)
    #from analysis.deep_learning.ConvLSTM.ConvLSTM import ConvLSTM
    from analysis.deep_learning.ConvLSTM.clstm import ConvLSTM
    model = ConvLSTM(config, config.num_samples, [64, 64, 128],  (3,3), 3, True, True, False).to(config.device)
    metrics_recorder = MetricsRecorder()

    #criterion = CrossEntropyLoss().to(config.device)
    if config.masked_loss is False:
        criterion = MSELoss().to(config.device)
    else:
        criterion = MSELoss(reduction='none').to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_records, valid_records, test_records = [], [], []
    rmse_train, rmse_valid, rmse_test = [], [], []
    mape_train, mape_valid, mape_test = [], [], []
    for epoch in tqdm(range(config.epochs)):
        epoch_records = train_loop(config, logger, epoch, model, train_dataloader, criterion, optimizer, mask=mask)
        
        train_records.append(np.mean(epoch_records['loss']))
        rmse_train.append(np.mean(epoch_records['rmse']))
        mape_train.append(np.mean(epoch_records['mape']))

        metrics_recorder.add_train_metrics(np.mean(epoch_records['mape']), np.mean(epoch_records['rmse']), np.mean(epoch_records['loss']))
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
        logger.info(log.format(epoch, np.mean(epoch_records['loss']), np.mean(epoch_records['mape']), np.mean(epoch_records['rmse'])))
        
        epoch_records = valid_loop(config, logger, epoch, model, val_dataloader, criterion, mask)
        valid_records.append(np.mean(epoch_records['loss']))
        rmse_valid.append(np.mean(epoch_records['rmse']))
        mape_valid.append(np.mean(epoch_records['mape']))
        log = 'Epoch: {:03d}, Val Loss: {:.4f}, Val MAPE: {:.4f}, Val RMSE: {:.4f}'
        logger.info(log.format(epoch, np.mean(epoch_records['loss']), np.mean(epoch_records['mape']), np.mean(epoch_records['rmse'])))
        metrics_recorder.add_val_metrics(np.mean(epoch_records['mape']), np.mean(epoch_records['rmse']), np.mean(epoch_records['loss']))

        model_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        early_stopping( np.mean(epoch_records['loss']), model_dict, epoch, config.checkpoint_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # if best_score is None:
        #     best_score = epoch_records['loss']
        # else:
        #     # Check if val_loss improves or not.
        #     if epoch_records['loss'] < best_score:
        #         # val_loss improves, we update the latest best_score, 
        #         # and save the current model
        #         best_score = epoch_records['loss']
        #         torch.save(model.state_dict(), os.path.join(config.checkpoint_dir,"convlstm_model.pth"))
        #         logger.info("Saved new pytorch model!")
        #     else:
        #         # val_loss does not improve, we increase the counter, 
        #         # stop training if it exceeds the amount of patience
        #         #counter += 1
        #         if epoch >= patience: #counter >= patience:
        #             break
        plt.plot(range(epoch + 1), train_records, label='train')
        plt.plot(range(epoch + 1), valid_records, label='valid')
        plt.legend()
        plt.savefig(os.path.join(config.output_dir, '{}.png'.format(name)))
        plt.close()

if __name__=="__main__":
    from analysis.deep_learning.GWNET.pipeline_gwnet import data_preparation 
    import pickle
    import os
    import matplotlib.pyplot as plt
    from utils.function_clns import load_config, prepare, CNN_split, interpolate_prepare
    import numpy as np
    from analysis.deep_learning.dataset import CustomDataset
    from torch.utils.data import DataLoader
    from loguru import logger

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
    
    from ancillary_vars.esa_landuse import drop_water_bodies_esa_downsample
    mask_ds = drop_water_bodies_esa_downsample(CONFIG_PATH, ds.isel(time=0))
    mask = torch.tensor(np.array(xr.where(mask_ds.notnull(), 1, 0)))

    print("Visualizing dataset before imputation...")
    #sub_precp = sub_precp.to_dataset()
    data, target = interpolate_prepare(args, sub_precp, ds, interpolate=True)
    train_split = 0.7
    training_lstm(CONFIG_PATH, data, target, mask=mask, train_split = train_split)