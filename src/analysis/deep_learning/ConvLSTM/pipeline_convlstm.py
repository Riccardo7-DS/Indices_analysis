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
from analysis import StandardScaler, StandardNormalizer, print_lr_change
import matplotlib
# matplotlib.use('Agg') 

def training_convlstm(args:dict,
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
    from torch.nn import MSELoss, DataParallel
    import matplotlib.pyplot as plt
    import numpy as np

    def process_datasets(save:bool=False):
        logger.info(f"Generating dumped tensor and saving option set to {save}")
        ################################# Initialize datasets #############################
        train_data, val_data, train_label, val_label, \
            test_data, test_label = CNN_split(data, target, 
                                            split_percentage=config["MODELS"]["split"],
                                            val_split=0.333)
        
        # create a CustomDataset object using the reshaped input data
        train_dataset = CustomConvLSTMDataset(model_config, args, 
            train_data, train_label, 
            save_files=save, 
            filename=f"train_ds_{model_config.batch_size}"
        )
        logger.info("Generated train dataloader")

        val_dataset = CustomConvLSTMDataset(model_config, args, 
            val_data, val_label, 
            save_files=save, 
            filename=f"val_ds_{model_config.batch_size}"
        )
        logger.info("Generated val dataloader")

        test_dataset = CustomConvLSTMDataset(model_config, args, 
            test_data, test_label, 
            save_files=save, 
            filename=f"test_ds_{model_config.batch_size}"
        )

        logger.info("Generated test dataloader")

        train_dataloader = DataLoader(train_dataset, 
            batch_size=model_config.batch_size, 
            shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, 
            batch_size=model_config.batch_size, 
            shuffle=False
        )

        test_dataloader = DataLoader(test_dataset, 
            batch_size=model_config.batch_size, 
            shuffle=False
        )
        return train_dataloader, val_dataloader, test_dataloader

    ################################# Module level logging #############################
    _, log_path, img_path, checkpoint_dir = create_runtime_paths(args)
    train_load_path = os.path.join(model_config.data_dir,f"data_convlstm",f"train_ds_{model_config.batch_size}")
    val_load_path = os.path.join(model_config.data_dir,f"data_convlstm",f"val_ds_{model_config.batch_size}")
    test_load_path = os.path.join(model_config.data_dir,f"data_convlstm",f"test_ds_{model_config.batch_size}")

    for path in [train_load_path, val_load_path, test_load_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    if args.loglevel:
        verbose = True
    else:
        verbose= False

    logger = init_logging(log_file=os.path.join(log_path, 
        f"convlstm_days_{args.step_length}"
        f"features_{args.feature_days}.log"),
        verbose=verbose)
    
    # writer = init_tb(log_path)

    ####################################################################################

    if (dump_batches is True) & (len(os.listdir(train_load_path))==0):
        train_dataloader, val_dataloader, test_dataloader = process_datasets(save=True)
        ################################# Check data shape #############################
        if logging.getLevelName(logger.level) == "DEBUG":
            check_shape_dataloaders(train_dataloader, val_dataloader)
    elif dump_batches is False:
        train_dataloader, val_dataloader, test_dataloader = process_datasets()
    else:
        logger.info(f"Loading dumped tensor")
        train_dataloader = NumpyBatchDataLoader(train_load_path,shuffle=True)
        val_dataloader = NumpyBatchDataLoader(val_load_path,shuffle=False)
        test_dataloader = NumpyBatchDataLoader(test_load_path,shuffle=False)

    ############################ Start training parameters #########################

    tot_channels = data.shape[1]
    model = ConvLSTM(tot_channels+1 if model_config.include_lag is True else tot_channels, 
        model_config.layers,  
        (3,3), 
        3, 
        True, 
        True, 
        False).to(model_config.device)
    
    model = DataParallel(model)
    
    metrics_recorder = MetricsRecorder()

    if model_config.masked_loss is False:
        criterion = MSELoss().to(model_config.device)
    else:
        criterion = MSELoss(reduction='none').to(model_config.device)

    learning_rate = model_config.learning_rate
    early_stopping = EarlyStopping(model_config, verbose=True)
    optimizer = torch.optim.Adam(model.parameters(), 
        lr=learning_rate,
        weight_decay=model_config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=model_config.scheduler_factor, 
        patience=model_config.scheduler_patience, 
    )

    train_loss_records, valid_loss_records, test_loss_records = [], [], []

    try:
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_sched'])
            checkp_epoch = checkpoint['epoch']
            logger.info(f"Resuming training from epoch {checkp_epoch}")
        
        start_epoch = 0 if checkpoint_path is None else checkp_epoch

    except IsADirectoryError:
        from analysis import load_checkp_metadata
        checkpoint_path = checkpoint_path.removesuffix(".pth.tar")
        model, optimizer, scheduler, checkp_epoch, _ = load_checkp_metadata(checkpoint_path, model, optimizer, scheduler, None) 
    

    #################################  Training  ################################## 

    if args.mode == "train":

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

            new_lr = print_lr_change(learning_rate, scheduler)
            learning_rate = new_lr
            early_stopping(np.mean(val_records['loss']), 
                           model_dict, epoch, checkpoint_dir)

            # update_tensorboard_scalars(writer, metrics_recorder)
            learning_rate = print_lr_change(learning_rate, scheduler)

            plt.plot(range(epoch - start_epoch + 1), train_loss_records, label='train')
            plt.plot(range(epoch - start_epoch + 1), valid_loss_records, label='valid')
            plt.legend()
            plt.savefig(os.path.join(img_path, f'learning_curve_feat_'
                                     f'{args.step_length}.png'))
            plt.close()

            if early_stopping.early_stop:
                logger.info("Early stopping")
                break
        
    elif args.mode == "test":
        from analysis import test_loop, masked_custom_loss, tensor_ssim, mask_rmse, masked_custom_loss
        from utils.function_clns import bias_correction
        

        mask = mask.float().to(model_config.device)

        test_records, y_pred, y_true = test_loop(model_config, 
            args, 
            model, 
            test_dataloader, 
            criterion, 
            ndvi_scaler, 
            mask, 
            args.scatterplot
        )

        # Expand the mask to match the dimensions of the tensor [S, W, H]
        expanded_mask = mask.bool().unsqueeze(0).expand(y_pred.size())

        y_corr, _, _ = bias_correction(y_true.detach().cpu(), 
                                       y_pred.detach().cpu())
        
        y_corr = y_corr.to(model_config.device)

        # Set values where the mask is True to NaN
        pred_masked = y_pred.masked_fill(expanded_mask, float(-1.0))
        true_masked = y_true.masked_fill(expanded_mask, float(-1.0))
        y_corr_masked = y_corr.masked_fill(expanded_mask, float(-1.0))

        ssim_metric = tensor_ssim(pred_masked, true_masked, range=2.0)
        ssim_metric_bias = tensor_ssim(y_corr_masked, true_masked, range=2.0)

        rmse = mask_rmse(y_pred, y_true, mask)
        losses = masked_custom_loss(criterion, y_pred, y_true, mask)
        
        rmse_bias = mask_rmse(y_corr, y_true, mask)
        losses_bias = masked_custom_loss(criterion, y_corr, y_true, mask)

        log = "Test MSE: {:.4f},Test SSIM: {:.4f}, Test RMSE: {:.4f}, " \
              "Bias Test MSE: {:.4f},Bias Test SSIM: {:.4f}, Bias Test RMSE: {:.4f}," \
              "Batch RMSE: {:.4f}, Batch MAPE: {:.4f}, Batch MSE: {:.4f} " \
        
        logger.info(log.format(losses, 
                               ssim_metric, 
                               rmse,
                               losses_bias, 
                               ssim_metric_bias, 
                               rmse_bias,
                               np.mean(test_records["rmse"]),
                               np.mean(test_records["mape"]),
                               np.mean(test_records["loss"])))
        
        if args.save_output is True:
            out_path = os.path.join(img_path, "output_data")
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            for d, name in zip([y_corr, y_true, mask], ['pred_data_dr', 'true_data_dr', 'mask_dr']):
                np.save(os.path.join(out_path, f"{name}.npy"), d.detach().cpu())


def pipeline_convlstm(args:dict,
    use_water_mask:bool = True,
    precipitation_only: bool = True,
    load_zarr_features:bool = False,
    load_local_precipitation:bool=True,
    interpolate:bool =False,
    checkpoint_path:str=None):
    
    from analysis import pipeline_hydro_vars
    from analysis.configs.config_models import config_convlstm_1 as model_config
    from utils.function_clns import find_checkpoint_path

    data, target, mask, ndvi_scaler = pipeline_hydro_vars(args,
        model_config,
        "data_gnn_drought",
        use_water_mask,
        precipitation_only,
        load_zarr_features,
        load_local_precipitation,
        interpolate
    )

    if checkpoint_path is None and args.mode == "test":
        checkpoint_path = find_checkpoint_path(model_config, args, True)


    data[np.isnan(data)] = -1
    target[np.isnan(target)] = -1

    training_convlstm(args,
        data, 
        target, 
        mask=mask, 
        ndvi_scaler = ndvi_scaler,
        checkpoint_path=checkpoint_path,
        dump_batches=False
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
    parser.add_argument('--model', default=os.environ.get('model', "GWNET"))
    parser.add_argument('--mode', default=os.environ.get('mode', "train"))

    parser.add_argument('--loglevel', default=os.environ.get('loglevel',False))

    parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
    parser.add_argument('--gcn_bool',default=True,help='whether to add graph convolution layer')
    parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
    parser.add_argument('--addaptadj', default=True, help='whether add adaptive adj')
    parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')

    parser.add_argument("--country", type=list, default=["Kenya","Somalia","Ethiopia", "Djibouti"], help="Location for dataset")
    parser.add_argument("--region", type=list, default=None, help="Location for dataset")

    parser.add_argument('--step_length',type=int,default=os.environ.get('step_length', 15),help='days in the future')
    parser.add_argument('--feature_days',type=int,default=os.environ.get('feature_days', 90))

    parser.add_argument('--fillna',type=bool,default=False)
    parser.add_argument("--interpolate", type=bool, default=False, help="Input data interpolation over time")
    parser.add_argument("--normalize", type=bool, default=True, help="normalization")
    parser.add_argument("--scatterplot", type=bool, default=True, help="scatterplot")
    parser.add_argument('--crop_area',type=bool,default=False)
    parser.add_argument('--plotheatmap', default=False, help="Save adjacency matrix heatmap")


    parser.add_argument('--checkpoint',type=int,default=os.environ.get('checkpoint', 0))
    args = parser.parse_args()
    os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()
    # path = "output/convlstm/days_30/features_90/checkpoints/checkpoint_epoch_3.pth.tar"

    pipeline_convlstm(args,
        precipitation_only=False
    )
    