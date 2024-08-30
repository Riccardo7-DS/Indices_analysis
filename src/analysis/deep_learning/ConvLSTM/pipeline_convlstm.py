import xarray as xr
import numpy as np
import torch
import os
from utils.function_clns import init_logging, CNN_split
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
import argparse
from tqdm.auto import tqdm
from typing import Union
import logging
from analysis import StandardScaler, StandardNormalizer


def training_convlstm(
                    args:dict,
                    data:np.array, 
                    target:np.array, 
                    ndvi_scaler:Union[StandardNormalizer, StandardScaler],
                    checkpoint_path:str,
                    mask:Union[None, torch.Tensor]=None,
                    dump_batches:bool = False):
                  
    import numpy as np
    from analysis.configs.config_models import config_convlstm_1 as model_config
    from utils.function_clns import config
    from analysis import MetricsRecorder, NumpyBatchDataLoader, check_shape_dataloaders, init_tb, update_tensorboard_scalars, ConvLSTM, create_runtime_paths, train_loop, valid_loop, EarlyStopping, CustomConvLSTMDataset
    from torch.nn import MSELoss
    import matplotlib.pyplot as plt
    import numpy as np

    def process_datasets(save:bool=False):
        logger.info(f"Generating dumped tensor and saving option set to {save}")
        ################################# Initialize datasets #############################
        train_data, val_data, train_label, val_label, \
            test_valid, test_label = CNN_split(data, target, 
                                               split_percentage=config["MODELS"]["split"])

        # create a CustomDataset object using the reshaped input data
        train_dataset = CustomConvLSTMDataset(model_config, args, 
                                              train_data, train_label, 
                                              save_files=save, 
                                              filename=f"train_ds_{model_config.batch_size}")

        val_dataset = CustomConvLSTMDataset(model_config, args, 
                                            val_data, val_label, 
                                            save_files=save, 
                                            filename=f"val_ds_{model_config.batch_size}")
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=model_config.batch_size, 
                                      shuffle=True)
        val_dataloader = DataLoader(val_dataset, 
                                    batch_size=model_config.batch_size, 
                                    shuffle=False)
        return train_dataloader, val_dataloader

    ################################# Module level logging #############################
    _, log_path, img_path, checkpoint_dir = create_runtime_paths(args)
    train_load_path = os.path.join(model_config.data_dir,f"data_convlstm",f"train_ds_{model_config.batch_size}")
    val_load_path = os.path.join(model_config.data_dir,f"data_convlstm",f"val_ds_{model_config.batch_size}")

    for path in [train_load_path, val_load_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    logger = init_logging(log_file=os.path.join(log_path, 
                                                      f"convlstm_days_{args.step_length}"
                                                      f"features_{args.feature_days}.log"))
    writer = init_tb(log_path)

    ####################################################################################

    if (dump_batches is True) & (len(os.listdir(train_load_path))==0):
        train_dataloader, val_dataloader = process_datasets(save=True)
        ################################# Check data shape #############################
        check_shape_dataloaders(train_dataloader, val_dataloader)
    elif dump_batches is False:
        train_dataloader, val_dataloader = process_datasets()
    else:
        logger.info(f"Loading dumped tensor")
        train_dataloader = NumpyBatchDataLoader(train_load_path,shuffle=True)
        val_dataloader = NumpyBatchDataLoader(val_load_path,shuffle=False)

    ############################ Start training parameters #########################

    tot_channels = data.shape[1]
    model = ConvLSTM(tot_channels+1 if model_config.include_lag is True else tot_channels, 
                     model_config.layers,  
                     (3,3), 3, True, True, False).to(model_config.device)
    metrics_recorder = MetricsRecorder()

    if model_config.masked_loss is False:
        criterion = MSELoss().to(model_config.device)
    else:
        criterion = MSELoss(reduction='none').to(model_config.device)

    learning_rate = model_config.learning_rate
    early_stopping = EarlyStopping(model_config, logger, verbose=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=model_config.scheduler_factor, 
        patience=model_config.scheduler_patience, 
    )

    train_loss_records, valid_loss_records, test_records = [], [], []

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_sched'])
        checkp_epoch = checkpoint['epoch']
        logger.info(f"Resuming training from epoch {checkp_epoch}")
    
    start_epoch = 0 if checkpoint_path is None else checkp_epoch

    #################################  Training  ################################## 

    for epoch in tqdm(range(start_epoch, model_config.epochs)):

        train_records = train_loop(model_config, args, model, train_dataloader, 
                                   criterion, optimizer, ndvi_scaler, mask=mask, 
                                   draw_scatter=False)
        
        train_loss_records.append(np.mean(train_records['loss']))
        metrics_recorder.add_train_metrics(train_records, epoch)
        
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
        logger.info(log.format(epoch, np.mean(train_records['loss']), 
                               np.mean(train_records['mape']), 
                               np.mean(train_records['rmse'])))
        
        ################################  Validation  ################################  
        
        val_records = valid_loop(model_config, args, model, val_dataloader, 
                                   criterion, scheduler, ndvi_scaler, mask, 
                                   draw_scatter=args.scatterplot)
        
        valid_loss_records.append(np.mean(val_records['loss']))

        log = 'Epoch: {:03d}, Val Loss: {:.4f}, Val MAPE: {:.4f}, Val RMSE: {:.4f}'
        logger.info(log.format(epoch, np.mean(val_records['loss']), 
                               np.mean(val_records['mape']), 
                               np.mean(val_records['rmse'])))
        
        metrics_recorder.add_val_metrics(val_records)

        ################################  Early stopping  ########################### 

        model_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "lr_sched": scheduler.state_dict()
        }

        update_tensorboard_scalars(writer, metrics_recorder)

        early_stopping(np.mean(val_records['loss']), 
                       model_dict, epoch, checkpoint_dir)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

        plt.plot(range(epoch - start_epoch + 1), train_loss_records, label='train')
        plt.plot(range(epoch - start_epoch + 1), valid_loss_records, label='valid')
        plt.legend()
        plt.savefig(os.path.join(img_path, f'learning_curve_feat_'
                                 f'{args.step_length}.png'))
        plt.close()



   

def pipeline_convlstm(args:dict,
                      use_water_mask:bool = True,
                      precipitation_only: bool = True,
                      load_zarr_features:bool = False,
                      load_local_precipitation:bool=True,
                      interpolate:bool =False,
                      checkpoint_path:str=None):
    
    from ancillary.esa_landuse import drop_water_bodies_copernicus
    from analysis.configs.config_models import config_convlstm_1 as model_config
    from precipitation.preprocessing.preprocess_data import PrecipDataPreparation
    from utils.function_clns import config, interpolate_prepare
    import numpy as np
    import pickle
    from definitions import ROOT_DIR

    def save_prepare_data(HydroData, save:bool=False):
        ndvi_scaler = HydroData.ndvi_scaler
        if use_water_mask is True:
            logger.info("Loading water bodies mask...")
            waterMask = drop_water_bodies_copernicus(HydroData.ndvi_ds).isel(time=0)
            # HydroData.hydro_data = HydroData.hydro_data.where(HydroData.ndvi_ds.notnull())
            mask = torch.tensor(np.array(xr.where(waterMask.notnull(), 1, 0)))
        else:
            mask = None

        logger.info("Checking dataset before imputation...")
        data, target = interpolate_prepare(
            HydroData.precp_ds, 
            HydroData.ndvi_ds, 
            interpolate=False,
            convert_to_float=False,
        )
        if save is True:
            for d, name in zip([data, target, mask],  ['data', 'target', 'mask']):
                np.save(os.path.join(data_dir, f"{name}.npy"), d)

            logger.debug("Saving NDVI scaler to file...")
            dump_obj = pickle.dumps(HydroData.ndvi_scaler, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(ROOT_DIR,"../data/ndvi_scaler.pickle"), "wb") as handle:
                handle.write(dump_obj)
        return data, target, ndvi_scaler, mask

    logger = init_logging(log_file=os.path.join(model_config.log_dir, 
                        "pipeline_convlstm.log"), verbose=False)
    
    data_dir = model_config.data_dir+"/data_convlstm"

    logger.info(f"Starting training CONVLSTM model for {args.step_length} days in the future"
                f" with {args.feature_days} days of features")

    if os.path.exists(data_dir) is False:
        logger.info("Created new path for data")
        os.makedirs(data_dir)

    if (len(os.listdir(data_dir)) == 0) & (load_zarr_features is False):
        #### Case 1: data existent but explicitly no load zarr
        logger.info(f"No numpy raw data found, proceeding with the creation"
                    f" of the training dataset.")

        if precipitation_only is False:
            import warnings
            warnings.filterwarnings('ignore')
            Era5variables = ["potential_evaporation", "evaporation",
                         "2m_temperature","total_precipitation"]
            HydroData = PrecipDataPreparation(
                args,
                variables=Era5variables,
                precipitation_data="ERA5_land",
                load_local_precp = load_local_precipitation,
                interpolate = interpolate
            )
        else:
            Era5variables = ["total_precipitation"]
            HydroData = PrecipDataPreparation(
                args, 
                precp_dataset=config["CONVLSTM"]['precp_product'],
                variables=Era5variables,
                load_local_precp=load_local_precipitation,
                interpolate = interpolate
            )
        data, target, ndvi_scaler, mask = save_prepare_data(HydroData, save=True)

    elif load_zarr_features is True:
        #### Case 2: explicitly load zarr
        HydroData = PrecipDataPreparation(
                args,
                variables=None,
                precipitation_data="ERA5_land",
                load_local_precp = False,
                load_zarr_features=load_zarr_features,
                interpolate = args.interpolate
            )
        data, target, ndvi_scaler, mask = save_prepare_data(HydroData, save=True)

    else:
        #### Case 3: explicitly no load zarr and data identified
        from definitions import ROOT_DIR
        logger.info("Training data found. Proceeding with loading...")
        data = np.load(os.path.join(data_dir, "data.npy"))
        target = np.load(os.path.join(data_dir, "target.npy"))
        mask = torch.tensor(np.load(os.path.join(data_dir, "mask.npy")))
        with open(os.path.join(ROOT_DIR,"../data/ndvi_scaler.pickle"), "rb") as handle:
            ndvi_scaler = pickle.loads(handle.read())
    
    training_convlstm(args,
                      data, 
                      target, 
                      mask=mask, 
                      ndvi_scaler = ndvi_scaler,
                      checkpoint_path=checkpoint_path,
                      dump_batches=True
    )

if __name__=="__main__":
    import pyproj
    parser = argparse.ArgumentParser()
    import gc
    import torch
    import os

    gc.collect()
    torch.cuda.empty_cache()
    parser.add_argument('-f')
    
    ### Convlstm parameters
    parser.add_argument('--model',type=str,default="CONVLSTM",help='DL model training')
    parser.add_argument('--step_length',type=int,default=60)
    parser.add_argument('--feature_days',type=int,default=90)
    parser.add_argument('--crop_area',type=bool,default=False)
    parser.add_argument('--fillna',type=bool,default=True)
    parser.add_argument("--normalize", type=bool, default=True, help="Input data normalization")
    parser.add_argument("--interpolate", type=bool, default=False, help="Input data interpolation over time")

    parser.add_argument("--country", type=list, default=["Kenya","Somalia","Ethiopia"], help="Location for dataset")
    parser.add_argument("--region", type=list, default=None, help="Location for dataset")
    parser.add_argument("--scatterplot", type=bool, default=True, help="Whether to visualize scatterplot")

    args = parser.parse_args()
    os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()
    # path = "output/convlstm/days_30/features_90/checkpoints/checkpoint_epoch_3.pth.tar"
    pipeline_convlstm(args, load_zarr_features=False, 
                      load_local_precipitation=False, 
                      precipitation_only=False)
    