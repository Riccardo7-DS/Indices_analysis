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
from analysis.deep_learning.GWNET.pipeline_gwnet import StandardScaler

def spi_ndvi_convlstm(CONFIG_PATH, time_start, time_end):
    config_file = load_config(CONFIG_PATH=CONFIG_PATH)

    # Open the NetCDF file with xarray
    dataset = prepare(xr.open_dataset(os.path.join(config_file['NDVI']['ndvi_path'], 
                                                   'ndvi_smoothed_w2s.nc')))\
        .sel(time=slice(time_start,time_end))[["time","lat","lon","ndvi"]]

    prod = "ERA5"
    late = 90

    path = config_file['PRECIP']['ERA5']['path']
    file = "ERA5_merged.nc" #"era5_land_merged.nc" #f"ERA5_spi_gamma_{late}.nc"
    precp_ds = prepare(subsetting_pipeline(CONFIG_PATH, 
                                           xr.open_dataset(os.path.join(path, file))))
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
        

        train_data, test_data, train_label, test_label = CNN_preprocessing(ds, sub_precp, var_origin="ndvi", 
                                                                           var_target=var_target, 
                                                                           preprocess_type=preprocess_type,  
                                                                           split=train_split)
        # Save the image data using pickle
        with open(file_path, 'wb') as file:
            pickle.dump((train_data, test_data, train_label, test_label), file)
        print("Data written to pickle file")
    
    return sub_precp, ds


def training_convlstm(args, logger, data:np.array, target:np.array, ndvi_scaler:StandardScaler,
                  mask:Union[None, np.array]=None, train_split:float = 0.7):
                  
    import numpy as np
    #from configs.config_3x3_32_3x3_64_3x3_128 import config
    from analysis.configs.config_3x3_16_3x3_32_3x3_64 import config
    from analysis.deep_learning.GWNET.pipeline_gwnet import MetricsRecorder, create_paths
    from torch.nn import MSELoss
    import matplotlib.pyplot as plt
    from analysis.deep_learning.ConvLSTM.clstm_unet import train_loop, valid_loop, build_logging# ConvLSTM
    import numpy as np
    from utils.function_clns import load_config
    from analysis.deep_learning.dataset import EarlyStopping

    output_dir, log_path, img_path, checkpoint_dir= create_paths(args)

    #### training parameters
    train_data, val_data, train_label, val_label, test_valid, test_label = CNN_split(data, target, 
                                                               split_percentage=train_split)
    
    batch_size = config.batch_size
    early_stopping = EarlyStopping(config, verbose=True)

    # create a CustomDataset object using the reshaped input data
    train_dataset = CustomConvLSTMDataset(config, args, train_data, train_label)
    val_dataset = CustomConvLSTMDataset(config, args, val_data, val_label)
    
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
    
    #name = '3x3_16_3x3_32_3x3_64'
    #name = "3x3_32_3x3_64_3x3_128"
    
#    logger = build_logging(config)

    from analysis.deep_learning.ConvLSTM.clstm import ConvLSTM

    model = ConvLSTM(config.num_samples, 
                     [32, 32, 32],  
                     (3,3), 3, True, True, False).to(config.device)
    #model = ConvLSTM(config).to(config.device)
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

        epoch_records = train_loop(config, args, logger, epoch, model, train_dataloader, criterion, 
                                   optimizer, ndvi_scaler, mask=mask, draw_scatter=False)
        
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
        
        epoch_records = valid_loop(config, args, logger,  epoch, model, val_dataloader, criterion, 
                                   ndvi_scaler, mask, draw_scatter=args.scatterplot)
        
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
            'optimizer': optimizer.state_dict()
        }

        early_stopping( np.mean(epoch_records['loss']), 
                       model_dict, epoch, checkpoint_dir)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

        plt.plot(range(epoch + 1), train_records, label='train')
        plt.plot(range(epoch + 1), valid_records, label='valid')
        plt.legend()
        plt.savefig(os.path.join(img_path, f'learning_curve_feat_{config.num_frames_input}.png'))
        plt.close()

    def convlstm_pipeline(args:dict, 
                          train_split:float = 0.7,
                          use_water_mask:bool =True,
                          precipitation_only: bool = True):
        
        from ancillary.esa_landuse import drop_water_bodies_esa_downsample
        from precipitation.preprocessing.preprocess_data import PrecipDataPreparation
        from utils.function_clns import config, interpolate_prepare
        import numpy as np
        from loguru import logger
        import analysis.configs.config_3x3_16_3x3_32_3x3_64 as model_config
        from utils.function_clns import config

        data_dir = model_config.output_dir+"data_convlstm"

        if len(os.listdir(data_dir)) == 0:
            logger.info("No data found, proceeding with the creation of the training dataset.")

            if precipitation_only is False:
                from ancillary.hydro_data import InputHydroVariables
                import warnings
                warnings.filterwarnings('ignore')

                Era5variables = ["potential_evaporation", "evaporation",
                             "2m_temperature","total_precipitation"]

                X_data = InputHydroVariables(Era5variables,
                            config["DEFAULT"]["start_date"],
                            config["DEFAULT"]["end_date"]
                )
                
                ancillary_vars = X_data.data
            
            else:
                Era5variables = ["total_precipitation"]
                
                precp_data = PrecipDataPreparation(
                    args, 
                    precp_dataset=config["CONVLSTM"]['precp_product'], 
                    ndvi_dataset="ndvi_smoothed_w2s.nc",
                    model = "CONVLSTM"
                )

            if use_water_mask is True:
                logger.info("Loading water bodies mask...")
                mask_ds = drop_water_bodies_esa_downsample(
                    ndvi_ds.isel(time=0))

                mask = torch.tensor(np.array(
                    xr.where(mask_ds.notnull(), 1, 0)))
            else:
                mask = None

            logger.info("Checking dataset before imputation...")

            data, target = interpolate_prepare(
                args, 
                sub_precp, 
                ndvi_ds, 
                interpolate=True
            )
        
        else:
            logger.info("Training data found. Proceeding with loading...")

        
        training_convlstm(data, 
            target, 
            mask=mask, 
            train_split = train_split, 
            ndvi_scaler = ndvi_scaler
        )
    

if __name__=="__main__":

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
    
    parser.add_argument("--country", type=list, default=["Kenya","Somalia","Ethiopia"], help="Location for dataset")
    parser.add_argument("--region", type=list, default=None, help="Location for dataset")
    parser.add_argument("--normalize", type=bool, default=False, help="Input data normalization")
    parser.add_argument("--scatterplot", type=bool, default=False, help="Whether to visualize scatterplot")

    args = parser.parse_args()

    convlstm_pipeline(args)