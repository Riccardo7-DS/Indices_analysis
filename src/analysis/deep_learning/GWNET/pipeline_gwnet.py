import logging
import numpy as np
import os
import torch
import time
import matplotlib.pyplot as plt
import pickle
from analysis import EarlyStopping, save_figures, create_runtime_paths, print_lr_change, MetricsRecorder, generate_adj_matrix, load_adj, get_train_valid_loader
import logging
logger = logging.getLogger(__name__)

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

    def inverse_transform(self, norm_data, null_value=None):
        transf = (norm_data * (self.max - self.min)) + self.min 
        if null_value is not None:
            mask = norm_data == null_value
            transf[mask] = null_value
            return transf
        else:
            return transf


def _load_data_and_scaler(data_dir, data, target, ndvi_scaler, mask):
    """Load data/target and scaler if not provided, then apply mask.

    Returns: (data, target, scaler, combined_mask)
    """
    import pickle
    from analysis import mask_gnn_pixels

    if data is None:
        data = np.load(os.path.join(data_dir, "data.npy"))
        target = np.load(os.path.join(data_dir, "target.npy"))

    if ndvi_scaler is None:
        with open(os.path.join(data_dir, "ndvi_scaler.pickle"), "rb") as handle:
            scaler = pickle.loads(handle.read())
    else:
        scaler = ndvi_scaler

    data, target, combined_mask = mask_gnn_pixels(data, target, mask)
    return data, target, scaler, combined_mask


def _prepare_adj_and_supports(adj_mx_path, data, combined_mask, args, device):
    """Ensure adjacency file exists (rank 0 creates it), load it, and convert to tensors on `device`.

    Returns: (adj_mx, supports)
    """
    from analysis import generate_adj_matrix, load_adj
    import torch.distributed as dist

    if args.ddp and args.local_rank == 0:
        if not os.path.exists(adj_mx_path):
            generate_adj_matrix(data, combined_mask, save_dir=os.path.dirname(adj_mx_path), save_plot=False)

    if args.ddp:
        dist.barrier()

    adj_mx = load_adj(adj_mx_path, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    return adj_mx, supports
  
def training_weatherGCNet(args, 
                          dataname,
                          data,
                          target, 
                          mask, 
                          ndvi_scaler=None, 
                          checkpoint_path=None):
    
    import torch.nn.functional as F
    import numpy as np
    from definitions import ROOT_DIR
    from analysis import WGCNModel, train_loop, valid_loop, EarlyStopping, update_tensorboard_scalars, mask_gnn_pixels
    from utils.function_clns import init_logging, config
    from analysis import check_shape_dataloaders, init_tb
    import torch
    from torch.nn import DataParallel
    from analysis.configs.config_models import config_wnet as model_config
    import os
    
    _, log_path, img_path, checkpoint_dir = create_runtime_paths(args)
    data_dir = os.path.join(model_config.data_dir, dataname)
    device = model_config.device

    logger = init_logging(log_file=os.path.join(log_path, 
                            f"pipeline_{(args.model).lower()}_"
                            f"features_{args.feature_days}.log"), verbose=False)
    # writer = init_tb(log_path)

    if data is None:
        logger.info(f"Starting training WeatherGCNet model for {args.step_length}"
                    f" days in the future with {args.feature_days} days of features")
        data = np.load(os.path.join(data_dir, "data.npy"))
        target = np.load(os.path.join(data_dir, "target.npy"))
    
    if ndvi_scaler is None:
        with open(os.path.join(data_dir, "ndvi_scaler.pickle"), "rb") as handle:
            scaler = pickle.loads(handle.read())
    else:
        scaler = ndvi_scaler

    data, target, combined_mask = mask_gnn_pixels(data, target, mask)

    ################################  Adjacency matrix  ################################ 
    
    train_dl, valid_dl, test_dl, dataset = get_train_valid_loader(model_config, 
        args, 
        data, 
        target,
        combined_mask,
        config["MODELS"]["split"])
    
    if logging.getLevelName(logger.level) == "DEBUG":
        check_shape_dataloaders(train_dl, valid_dl)

    model = WGCNModel(args, model_config, dataset.data).to(device)
    # model = DataParallel(model)

    metrics_recorder = MetricsRecorder()
    train_records, valid_records = [], []
    rmse_train, rmse_valid, = [], []
    mape_train, mape_valid = [], []

    loss_func = torch.nn.MSELoss()
    learning_rate = model_config.learning_rate

    early_stopping = EarlyStopping(model_config, verbose=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=model_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=model_config.scheduler_factor, 
        patience=model_config.scheduler_patience)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_sched'])
        checkp_epoch = checkpoint['epoch']
        logger.info(f"Resuming training from epoch {checkp_epoch}")

    start_epoch = 0 if checkpoint_path is None else checkp_epoch

    if args.mode == "train":

        for epoch in range(start_epoch, model_config.epochs):
            epoch_records = train_loop(model_config, args, model, train_dl, loss_func, 
                                optimizer, scaler=ndvi_scaler, draw_scatter=False)

            train_records.append(np.mean(epoch_records['loss']))
            rmse_train.append(np.mean(epoch_records['rmse']))
            mape_train.append(np.mean(epoch_records['mape']))

            metrics_recorder.add_train_metrics(epoch_records, epoch)

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            logger.info(log.format(epoch, np.mean(epoch_records['loss']), 
                                   np.mean(epoch_records['mape']), 
                                   np.mean(epoch_records['rmse'])))


            epoch_records = valid_loop(model_config, args,  model, valid_dl, loss_func, scheduler,
                                       scaler=ndvi_scaler, mask=None, draw_scatter=args.scatterplot)

            valid_records.append(np.mean(epoch_records['loss']))
            rmse_valid.append(np.mean(epoch_records['rmse']))
            mape_valid.append(np.mean(epoch_records['mape']))

            log = 'Epoch: {:03d}, Val Loss: {:.4f}, Val MAPE: {:.4f}, Val RMSE: {:.4f}'
            logger.info(log.format(epoch, np.mean(epoch_records['loss']), 
                                   np.mean(epoch_records['mape']), 
                                   np.mean(epoch_records['rmse'])))

            metrics_recorder.add_val_metrics(epoch_records)

            model_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "lr_sched": scheduler.state_dict()
            }

            mean_loss = sum(epoch_records['loss']) / len(epoch_records['loss'])
            scheduler.step(mean_loss)
            new_lr = print_lr_change(learning_rate, scheduler)
            learning_rate = new_lr
            early_stopping( np.mean(epoch_records['loss']), 
                           model_dict, epoch, checkpoint_dir)

            # update_tensorboard_scalars(writer, metrics_recorder)
            learning_rate = print_lr_change(learning_rate, scheduler)

            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

            plt.plot(range(epoch - start_epoch + 1), train_records, label='train')
            plt.plot(range(epoch - start_epoch + 1), valid_records, label='valid')
            plt.legend()
            plt.savefig(os.path.join(img_path, f'learning_curve_feat_'
                                     f'{args.feature_days}.png'))
            plt.close()
        
    elif args.mode == "test":
        logger.info("Starting evaluation")
        from analysis import test_loop, tensor_ssim, reverse_reshape_gnn, CustomMetrics
        test_records, y_pred, y_true = test_loop(model_config,
            args, 
            model, 
            test_dl, 
            loss_func, 
            ndvi_scaler,
            mask = combined_mask,
            draw_scatter=args.scatterplot
        )
        
        mean_loss = sum(test_records['loss']) / len(test_records['loss'])
        mean_rmse = sum(test_records['rmse']) / len(test_records['rmse'])
        mean_mape = sum(test_records['mape']) / len(test_records['mape'])

        corr_output = reverse_reshape_gnn(target, y_pred.detach().cpu().numpy(), combined_mask)
        corr_real = reverse_reshape_gnn(target, y_true.detach().cpu().numpy(), combined_mask)

        y_pred_null = np.where(np.isnan(corr_output), -1, corr_output)
        y_true_null = np.where(np.isnan(corr_real), -1, corr_real)
        ssim_metric = tensor_ssim(y_pred_null, y_true_null, range=2.0)

        metric_list = ["rmse", "mse"]

        test_metrics = CustomMetrics(
            corr_output, 
            corr_real, 
            metric_list, 
            combined_mask, 
            True
        )
        rmse = test_metrics.losses[0]
        losses = test_metrics.losses[1]

        log = 'The prediction for {} days ahead: Batch Test loss: {:.4f}, ' \
            'Batch Test MAPE: {:.4f}, Batch Test RMSE: {:.4f}, Test SSIM: {:.4f}' \
            'Test RMSE: {:.4f}, Test MSE:{:.4f}'
        
        logger.info(log.format(args.step_length,
                            mean_loss, 
                            mean_mape, 
                            mean_rmse, 
                            ssim_metric,
                            rmse,
                            losses))
        
        if args.save_output is True:
            out_path = os.path.join(img_path, "output_data")
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            for d, name in zip([corr_output, corr_real, mask], ['pred_data_dr', 'true_data_dr', 'mask_dr']):
                np.save(os.path.join(out_path, f"{name}.npy"), d)


# # # # # # # # # #

def training_wavenet(args,
                    dataname:str="data_gnn", 
                    data=None, 
                    target=None, 
                    ndvi_scaler=None,
                    mask=None,
                    checkpoint_path=None):
    import pickle
    from analysis.configs.config_models import config_gwnet as model_config
    from analysis import check_shape_dataloaders, GWNETtrainer
    import torch.distributed as dist
    from utils.function_clns import init_logging, config

    _, log_path, img_path, checkpoint_dir = create_runtime_paths(args)
    data_dir = os.path.join(model_config.data_dir, dataname)
    # device = model_config.device

    if args.ddp and args.local_rank != 0:
        logger = init_logging(verbose=False)
    else:
        logger = init_logging(log_file=os.path.join(log_path, 
                                f"pipeline_{(args.model).lower()}_"
                                f"features_{args.feature_days}.log"), 
                                verbose=args.loglevel)
    
    ################################  Adjacency matrix and data ################################ 

    adj_mx_path = os.path.join(data_dir, "adj_dist.pkl")

    # writer = init_tb(log_path)

    # Load data/scaler/mask and prepare adjacency/supports in helper functions
    data, target, scaler, combined_mask = _load_data_and_scaler(data_dir, data, target, ndvi_scaler, mask)

    # device for this rank
    device = torch.device(f'cuda:{args.local_rank}') if args.ddp else model_config.device

    adj_mx, supports = _prepare_adj_and_supports(adj_mx_path, data, combined_mask, args, device)

    # choose initial adjacency
    adjinit = None if args.randomadj else supports[0]
    if args.aptonly:
        supports = None

    # create dataloaders (DistributedSampler expected inside)
    train_dl, valid_dl, test_dl, dataset = get_train_valid_loader(
        model_config,
        args,
        data,
        target,
        combined_mask,
        config["MODELS"]["split"],
    )

    if logging.getLevelName(logger.level) == "DEBUG":
        check_shape_dataloaders(train_dl, valid_dl)

    num_nodes = dataset.data.shape[-1]

    loss = torch.nn.MSELoss()

    # Now create the engine using the prepared supports and adjinit
    engine = GWNETtrainer(
        args,
        model_config,
        scaler,
        loss,
        num_nodes,
        supports,
        adjinit,
        checkpoint_path,
    )

    metrics_recorder = MetricsRecorder()
    
    start_epoch = 0 if checkpoint_path is None else engine.checkp_epoch
    early_stopping = EarlyStopping(model_config, verbose=True)

    ################################  Training  ################################ 
    
    if args.mode == "train":
        logger.info("Starting training...")

        for epoch in range(start_epoch+1, model_config.epochs+1):

            train_records = engine.gwnet_train_loop(model_config, train_dl)

            dist.barrier()

            metrics_recorder.add_train_metrics(train_records, epoch)

        ################################  Validation ###############################

            val_records = engine.gwnet_val_loop(model_config, valid_dl)
            metrics_recorder.add_val_metrics(val_records)

            mtrain_loss = np.mean(train_records["loss"])
            mtrain_rmse = np.mean(train_records["rmse"])  
            mvalid_loss = np.mean(val_records["loss"])
            mvalid_rmse = np.mean(val_records["rmse"])


            if args.local_rank == 0:

                save_figures(epoch=epoch-start_epoch, 
                             path=img_path, 
                             metrics_recorder=metrics_recorder)

                log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, ' \
                      'Valid Loss: {:.4f}, Valid RMSE: {:.4f}'
                logger.info(log.format(epoch, mtrain_loss, mtrain_rmse, mvalid_loss, mvalid_rmse))

                model_dict = {
                    'epoch': epoch,
                    'state_dict': engine.model.state_dict(),
                    'optimizer': engine.optimizer.state_dict(),
                    "lr_sched": engine.scheduler.state_dict()
                }

                early_stopping(mvalid_loss, model_dict, epoch, checkpoint_dir)
                # update_tensorboard_scalars(writer, metrics_recorder)

                if early_stopping.early_stop:
                    logger.info("Early stopping")
                    if args.plotheatmap is True:
                        from analysis import draw_adj_heatmap
                        draw_adj_heatmap(engine, img_path)
                    break

    elif args.mode== "test":
        from analysis import reverse_reshape_gnn,tensor_ssim
        logger.info("Starting evaluation on independent dataset")
        test_records, outputs, y_real = engine.gwnet_test_loop(args,
            model_config, 
            test_dl, 
            ndvi_scaler,
            draw_scatter=args.scatterplot)
        
        mean_loss = sum(test_records['loss']) / len(test_records['loss'])
        mean_rmse = sum(test_records['rmse']) / len(test_records['rmse'])
        mean_mape = sum(test_records['mape']) / len(test_records['mape'])

        corr_output = reverse_reshape_gnn(target, outputs.detach().cpu().numpy(), combined_mask)
        corr_real = reverse_reshape_gnn(target, y_real.detach().cpu().numpy(), combined_mask)

        # Replace NaNs with -1 using torch.where
        y_pred_null = np.where(np.isnan(corr_output), -1, corr_output)
        y_true_null = np.where(np.isnan(corr_real), -1, corr_real)
        ssim_metric = tensor_ssim(y_pred_null, y_true_null, range=2.0)

        log = 'The prediction for {} days ahead: Test loss: {:.4f}, ' \
            'Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test SSIM: {:.4f}'
        logger.info(log.format(args.step_length, 
            mean_loss, 
            mean_mape, 
            mean_rmse, 
            ssim_metric)
        )
        
        if args.save_output is True:
            out_path = os.path.join(img_path, "output_data")
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            for d, name in zip([corr_output, corr_real, mask], ['pred_data_dr', 'true_data_dr', 'mask_dr']):
                np.save(os.path.join(out_path, f"{name}.npy"), d)

def pipeline_gnn(args:dict,
                use_water_mask:bool = True,
                precipitation_only: bool = True,
                load_zarr_features:bool = False,
                load_local_precipitation:bool=True,
                interpolate:bool =False,
                checkpoint_path:str=None,
                add_extra_data:bool=False):
    
    from analysis import pipeline_hydro_vars
    from analysis.configs.config_models import config_gwnet as model_config
    from utils.function_clns import find_checkpoint_path
    
    rawdata_name = "data_gnn_full"

    data, target, mask, ndvi_scaler = pipeline_hydro_vars(args,
        model_config,
        rawdata_name,
        use_water_mask,
        precipitation_only,
        load_zarr_features,
        load_local_precipitation,
        interpolate
    )

    if add_extra_data:
        logger.info("Adding extra data for 2021-22")
        extra_data  = np.load(os.path.join(model_config.data_dir, "data_gnn_drought/data.npy"))[-365*2:]
        extra_target =  np.load(os.path.join(model_config.data_dir, "data_gnn_drought/target.npy"))[-365*2:]

        extra_data[np.isnan(extra_data)] = -1
        extra_target[np.isnan(extra_target)] = -1

        target = np.concatenate([target, extra_target], 0)
        data = np.concatenate([data, extra_data], 0)

    if checkpoint_path is None and args.mode == "test":
        checkpoint_path = find_checkpoint_path(model_config, args, True)

    if args.model == "GWNET":
        training_wavenet(args,
            dataname=rawdata_name,
            data=data, 
            target=target, 
            mask=mask, 
            ndvi_scaler = ndvi_scaler,
            checkpoint_path=checkpoint_path
        )

    elif args.model == "WNET":
        training_weatherGCNet(args,
            dataname=rawdata_name,
            data=data,
            target=target,
            mask=mask,
            ndvi_scaler=ndvi_scaler,
            checkpoint_path=checkpoint_path)


if __name__=="__main__":
    import argparse
    import pyproj
    from analysis.configs.config_models import config_gwnet as model_config
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')

    parser.add_argument("--model", type=str, default="GWNET", help="")

    parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
    parser.add_argument('--gcn_bool',default=True,help='whether to add graph convolution layer')
    parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
    parser.add_argument('--addaptadj', default=True, help='whether add adaptive adj')
    parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')

    parser.add_argument("--country", type=list, default=["Kenya","Somalia","Ethiopia", "Djibouti"], help="Location for dataset")
    parser.add_argument("--region", type=list, default=None, help="Location for dataset")

    parser.add_argument('--step_length',type=int,default=15,help='days in the future')
    parser.add_argument('--feature_days',type=int,default=90)

    parser.add_argument('--fillna',type=bool,default=False)
    parser.add_argument("--interpolate", type=bool, default=False, help="Input data interpolation over time")
    parser.add_argument("--normalize", type=bool, default=True, help="normalization")
    parser.add_argument("--scatterplot", type=bool, default=True, help="scatterplot")
    parser.add_argument('--crop_area',type=bool,default=False)

    args = parser.parse_args()
    os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()
    checkpoint = model_config.output_dir + "/gwnet/days_15/features_90/checkpoints/checkpoint_epoch_3.pth.tar"
    pipeline_gnn(args,
        use_water_mask=True,
        load_local_precipitation=True,
        precipitation_only=False,
        checkpoint_path=checkpoint)