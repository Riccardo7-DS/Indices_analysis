import os
import numpy as np
import torch
from torch.nn import MSELoss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from utils.function_clns import init_logging, find_checkpoint_path, bias_correction
from utils.xarray_functions import ndvi_colormap
from analysis import (
    load_stored_data, DataGenerator, UNET, load_checkpoint, load_checkp_metadata,
    custom_subset_data, create_runtime_paths, autoencoder_wrapper,
    CustomConvLSTMDataset, TwoResUNet,
    Forward_diffussion_process, mask_mse, compute_image_loss_plot,
    tensor_ssim, CustomMetrics
)
from ema_pytorch import EMA, PostHocEMA
import argparse
import logging
logger = logging.getLogger(__name__)

def diffusion_arguments():
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument('-f')
    parser.add_argument('--model', type=str, default="DIME", help='DL model training')
    parser.add_argument('--step_length', type=int, default=os.getenv("step_length", 15))
    parser.add_argument('--conditioning', type=str, choices=["none", "all", "autoenc", "climate"], default=os.getenv("conditioning", "all"))
    parser.add_argument('--attention', type=bool, default=os.getenv("attention", False), help='U-NET architecture w/o attention')
    parser.add_argument('--auto_train', type=bool, default=os.getenv("auto_train", False))
    parser.add_argument('--save_output', type=bool, default=os.getenv("save_output", True))
    parser.add_argument('--auto_days', type=int, default=os.getenv("auto_days", 180))
    parser.add_argument('--feature_days', type=int, default=os.getenv("feature_days", 1))
    parser.add_argument('--auto_ep', type=int, default=os.getenv("auto_ep", 80))
    parser.add_argument('--gen_samples', type=int, default=os.getenv("gen_samples", 0))
    parser.add_argument('--diff_schedule', type=str, default=os.getenv("diff_schedule", "sigmoid"))
    parser.add_argument('--diff_sample', type=str, default=os.getenv("diff_sample", "ddim"))
    parser.add_argument('--epoch', type=int, default=os.getenv("epoch", 0), help="diffusion model trained epochs")
    parser.add_argument("--normalize", type=bool, default=True, help="Input data normalization")
    parser.add_argument("--scatterplot", type=bool, default=True, help="Whether to visualize scatterplot")
    parser.add_argument('--mode', type=str, default=os.getenv("mode", "generate"))
    parser.add_argument("--num_ensambles", type=int, default=os.getenv("num_ensambles", 1), help="if making ensamble predictions")
    parser.add_argument("--ema", type=str,choices=["none", "ema", "posthoc"], default=os.getenv("ema", "none"), help="if using ema")
    parser.add_argument("--sigma_rel", type=float, default=os.getenv("sigma_rel", 0.15), help="sigma value for ema")
    parser.add_argument("--eta", type=float, default=os.getenv("eta", 0.), help="eta value for ddim sampling")

    args = parser.parse_args()
    return args

def diffusion_evaluation(rank, world_size, args):

    from analysis.configs.config_models import config_ddim as model_config, config_autodime as auto_config

    autoencoder = autoencoder_wrapper(args, auto_config, generate_output=False)
    start, end = "2019-01-01", "2022-12-31"
    data_name = "data_gnn_drought"
    _, log_path, img_path, checkpoint_dir = create_runtime_paths(args)

    checkpoint_path = (find_checkpoint_path(model_config, args, True) if args.epoch == 0 else
        os.path.join(checkpoint_dir, f"checkpoint_epoch_{args.epoch}"))
    
    if rank == 0 or rank is None:
        logger = init_logging(log_file=os.path.join(log_path, 
            f"dime_days_{args.step_length}_"  \
            f"features_{args.feature_days}"
            f"{args.mode}.log")
    )

    data, target, scaler, mask = custom_subset_data(args, model_config, data_name, start, end, True, True)

    datagenerator = DataGenerator(model_config, args, data, target, start_date=start, autoencoder=autoencoder, data_split="test") # data_split="ema_test") #past_data=dataset_auto.data[:, :, -1]
    
    if is_torchrun():
        sampler = DistributedSampler(datagenerator, num_replicas=world_size, rank=rank, shuffle=False)
        dataloader = DataLoader(datagenerator, sampler=sampler, batch_size=model_config.batch_size, shuffle=False,  num_workers=6)
    else:
        dataloader = DataLoader(datagenerator, shuffle=False, batch_size=model_config.batch_size)

    input_channels = datagenerator.data.shape[1] if args.conditioning != "none" else 0

    model_class = TwoResUNet if args.attention else UNET
    model = model_class(dim=model_config.widths[0]*2, 
        channels=input_channels + 1, 
        dim_mults=(1, 2, 4, 8, 16),
        out_dim=model_config.output_channels).to(model_config.device)
    
    if rank is None:
        from torch.nn.parallel import DataParallel
        model = DataParallel(model)

    if args.ema == "ema":
        ema = EMA(model, 
            beta = model_config.ema_decay, 
            update_every= model_config.ema_update_every,
            update_after_step = model_config.update_after_step
        ).to(model_config.device)

    elif args.ema == "posthoc":
        post_dir = os.path.join(checkpoint_dir, "posthoc_checkpoints")
        ema = PostHocEMA(model, 
            sigma_rels = (0.05, 0.28), 
            update_every= model_config.ema_update_every,
            checkpoint_every_num_steps = 130,
            checkpoint_folder=post_dir)
    else:
        ema = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=model_config.scheduler_factor, patience=model_config.scheduler_patience)
    loss_fn = MSELoss()

    try:
        model, optimizer, scheduler, start_epoch, ema = load_checkp_metadata(checkpoint_path, model, optimizer, scheduler, ema)    
    except IsADirectoryError as e:
        logger.error(e)
        model, optimizer, scheduler, start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    if args.ema == "ema":
        model = ema.ema_model.eval()
    elif args.ema == "posthoc":
        model = ema.synthesize_ema_model(sigma_rel=args.sigma_rel)
        model.eval()

    if args.num_ensambles > 1:
        from analysis import ModelEnsamble, cleanup
        model = ModelEnsamble(args, model, local_rank=rank)

    fdp = Forward_diffussion_process(args, model_config, model, optimizer, scheduler, loss_fn, ema)
    logger.info(f"Starting generating from diffusion model with {args.diff_schedule} schedule " 
                f"and sampling technique {args.diff_sample} with model trained for {start_epoch} epochs")
    if args.mode == "test_model":  
        return model
    
    y_pred, y_true, ens = fdp.diffusion_sampling(args, model_config, model, dataloader, scaler, samples=args.gen_samples)

    if args.normalize:
        y_pred = scaler.inverse_transform(y_pred)
        y_true = scaler.inverse_transform(y_true)

    y_pred = torch.clamp(y_pred, -1, 1)
    y_true = torch.clamp(y_true, -1, 1)
    mask = torch.from_numpy(mask).to(model_config.device)

    # y_corr, _, _ = bias_correction(y_true.detach().cpu(), y_pred.detach().cpu())
    # y_corr = y_corr.to(model_config.device)   

    y_pred_null = torch.where(torch.isnan(y_pred), torch.tensor(-1.0), y_pred)
    y_true_null = torch.where(torch.isnan(y_true), torch.tensor(-1.0), y_true)

    ssim_metric = tensor_ssim(y_pred_null, y_true_null, range=2.0)
    metric_list = ["rmse", "mse"]

    test_metrics = CustomMetrics(y_pred, y_true, metric_list, mask, True)
    rmse, losses = test_metrics.losses[0], test_metrics.losses[1]

    if (rank == 0 or rank is None) and args.save_output is True:
        log = 'The prediction for {} days ahead: Test SSIM: {:.4f}, Test RMSE: {:.4f}, Test MSE: {:.4f}'
        logger.info(log.format(args.step_length, ssim_metric, rmse, losses))
        out_path = os.path.join(img_path, "output_data")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for d, name in zip([y_pred, y_true, mask, ens], ['pred_data_dr', 'true_data_dr', 'mask_dr', 'ens_dr']):
            np.save(os.path.join(out_path, f"{name}.npy"), d.detach().cpu())

    if rank is not None:
        cleanup()

def is_torchrun():
    return 'LOCAL_RANK' in os.environ or 'RANK' in os.environ

def main():
    args = diffusion_arguments()
    if is_torchrun():
        world_size = torch.cuda.device_count()
        local_rank = int(os.environ['LOCAL_RANK'])
        from analysis import setup
        setup(local_rank, world_size)
    else:
        local_rank = None
        world_size = None
    try:    
        diffusion_evaluation(local_rank, world_size, args)
    except Exception as e:
        logger.error("An error occurred during diffusion evaluation.", exc_info=True)
        
        if is_torchrun():
            logger.info("Destroying process group due to an exception.")
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
    