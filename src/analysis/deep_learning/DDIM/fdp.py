from analysis import Forward_diffussion_process, TwoResUNet, UNET, plot_noisy_images, load_stored_data, DataGenerator, tensor_corr, UNET, create_runtime_paths, init_tb, EarlyStopping
from analysis import diffusion_train_loop,load_checkp_metadata, compute_image_loss_plot, autoencoder_wrapper
from analysis.configs.config_models import config_ddim as model_config
from torch.nn import L1Loss, MSELoss, DataParallel
from ema_pytorch import EMA, PostHocEMA
import numpy as np
import os
import argparse 
from analysis import load_checkpoint
import torch
import matplotlib.pyplot as plt
import logging
from torch.utils.data import DataLoader
from utils.function_clns import config, CNN_split, init_logging
from utils.xarray_functions import ndvi_colormap
from timm.utils import ModelEmaV3
import matplotlib.pyplot as plt
import torch
from datetime import datetime

import matplotlib
import gc
matplotlib.use('Agg')
gc.collect()
torch.cuda.empty_cache()
cmap = ndvi_colormap("sequential")

parser = argparse.ArgumentParser(conflict_handler="resolve")
parser.add_argument('-f')

### Autoencoder parameters
parser.add_argument('--model',type=str,default="AUTO_DIME",help='DL model training')
parser.add_argument('--step_length',type=int,default=os.getenv("step_length", 15))

parser.add_argument('--attention',type=bool,default=os.getenv("attention", False),help='U-NET architecture w/o attention')
parser.add_argument('--auto_train',type=bool,default=os.getenv("auto_train", False))
parser.add_argument('--auto_days',type=int,default=os.getenv("auto_days", 180))

parser.add_argument('--feature_days',type=int,default=os.getenv("feature_days", 1))
parser.add_argument('--auto_ep',type=int,default=os.getenv("auto_ep", 80))
parser.add_argument('--gen_samples',type=int,default=os.getenv("gen_samples", 1))

### diffusion parameters
parser.add_argument('--diff_schedule',type=str,default=os.getenv("diff_schedule", "sigmoid"))
parser.add_argument('--diff_sample', type=str, default=os.getenv("diff_sample", "ddim"))
parser.add_argument('--epoch',type=int,default=os.getenv("epoch", 0), help="diffusion model trained epochs")
parser.add_argument('--conditioning', type=str, choices=["none", "all", "autoenc","climate"], default=os.getenv("conditioning",'all'))
parser.add_argument("--normalize", type=bool, default=True, help="Input data normalization")
parser.add_argument("--scatterplot", type=bool, default=True, help="Whether to visualize scatterplot")
parser.add_argument('--mode', type=str, default=os.getenv("mode", "generate"))
parser.add_argument("--ensamble", type=bool, default=os.getenv("ensamble", False), help="if making ensamble predictions")
parser.add_argument("--ema", type=str,choices=["none", "ema", "posthoc"], default=os.getenv("ema", "none"), help="if using ema")

args = parser.parse_args()

data, target, scaler, mask = load_stored_data(model_config)

################################# Autoencoder #############################

autoencoder = autoencoder_wrapper(args, model_config, data, target, generate_output=False)

################################# Initialize datasets #############################

### Diffusion parameters
parser.add_argument('--model',type=str,default="DIME",help='DL model training')
args = parser.parse_args()

_, log_path, img_path, checkpoint_dir = create_runtime_paths(args)
checkpoint_path  = checkpoint_dir + f"/checkpoint_epoch_{args.epoch}"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

logger = init_logging(log_file=os.path.join(
    log_path,
    f"dime_days_{args.step_length}_"
    f"features_{args.feature_days}_"
    f"{args.mode}_{timestamp}.log"
))
# writer = init_tb(log_path)

train_data, val_data, train_label, val_label, \
    test_data, test_label = CNN_split(data, target, 
    split_percentage =0.75) #0.9375,
    #val_split=0) 

# create a CustomDataset object using the reshaped input data
datagenrator_train = DataGenerator(model_config, 
    args,
    train_data, 
    train_label, 
    autoencoder=autoencoder, 
    start_date="2005-01-01",
    data_split=f"train_vae"
)

dataloader_train = DataLoader(datagenrator_train, 
    shuffle=True, 
    batch_size=model_config.batch_size)

# if test_data is not None:
#     datagenrator_test= DataGenerator(model_config, 
#         args,
#         test_data, 
#         test_label, 
#         autoencoder=autoencoder, 
#         data_split=f"test"
#     )

#     dataloader_test = DataLoader(datagenrator_test, 
#         shuffle=False, 
#         batch_size=model_config.batch_size
#     )

########################### Models and training functions #######################

input_channels = datagenrator_train.data.shape[1] if args.conditioning != "none" else 0

if args.attention:
    model = TwoResUNet(dim=model_config.widths[0]*2, 
        channels=input_channels+1,
        dim_mults=(1, 2, 4, 8, 16),
        out_dim=model_config.output_channels).to(model_config.device)
    weight_decay = 1e-2
else:
    model = UNET(dim=model_config.widths[0]*2, 
        channels=input_channels+1,
        dim_mults=(1, 2, 4, 8, 16),
        out_dim=model_config.output_channels).to(model_config.device)
    weight_decay = 1e-4

model = DataParallel(model)

if args.ema == "ema":
    ema = EMA(model, 
        beta = model_config.ema_decay, 
        update_every= model_config.ema_update_every,
        update_after_step = model_config.update_after_step
        ).to(model_config.device)
    
elif args.ema == "posthoc":
     post_dir = os.path.join(checkpoint_dir, "posthoc_checkpoints")
     os.makedirs(post_dir, exist_ok=True) 
     ema = PostHocEMA(model, 
        sigma_rels = (0.05, 0.28), 
        update_every= model_config.ema_update_every,
        checkpoint_every_num_steps = 130,
        checkpoint_folder=post_dir)
else:
    ema = None

optimizer = torch.optim.AdamW(model.parameters(), 
    lr=model_config.learning_rate, 
    weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=model_config.scheduler_factor, 
    patience=model_config.scheduler_patience
)

if args.epoch > 0:
    model, optimizer, scheduler, start_epoch, ema = load_checkp_metadata(checkpoint_path, 
        model, 
        optimizer, 
        scheduler,
        ema
    )
else:
    start_epoch = 0

loss_fn = MSELoss()
fdp = Forward_diffussion_process(args, 
    model_config, 
    model,  
    optimizer, 
    scheduler, 
    loss_fn,
    ema
)

################### test image ############################

x_features, img = next(iter(dataloader_train))
image = img[0].to(model_config.device)
# img = fdp.get_noisy_image(image.to(model_config.device), t)

plot_noisy_images(image, [fdp.get_noisy_image(image, torch.tensor([t])\
                          .to(model_config.device)) for t in [0, 20, 50, 75, 100, 150, 199]],
                          **{"cmap": cmap})

###########################################################

if args.mode == "train":
    logger.info(f"Starting training diffusion model with {args.diff_schedule} schedule " 
            f"with conditioning option {args.conditioning} and attention {args.attention}"  
            f" sampling technique {args.diff_sample} at epoch {start_epoch}") 
    diffusion_train_loop(args, 
        model_config, 
        fdp, 
        dataloader_train, 
        checkpoint_dir, 
        start_epoch
    )
    
elif args.mode == "generate":
    logger.info(f"Starting generating from diffusion model with {args.diff_schedule} schedule " 
           f"and sampling technique {args.diff_sample} with model trained for {start_epoch} epochs") 
    sample_image, y_true, std = fdp.diffusion_sampling(args, 
        model_config, 
        model, 
        dataloader_test, 
        samples=args.gen_samples
    )
    if args.normalize is True:
        sample_image = scaler.inverse_transform(sample_image)
        y_true = scaler.inverse_transform(y_true)   

    sample_image = torch.clamp(sample_image, -1, 1)
    y_true = torch.clamp(y_true, -1, 1)
    mask = torch.from_numpy(mask).to(model_config.device)

    out_path = os.path.join(img_path, "output_data")

    if not os.path.exists(out_path):
         os.makedirs(out_path)

    for d, name in zip([sample_image, y_true, mask],  ['pred_data', 'true_data', 'mask']):
                np.save(os.path.join(out_path, f"{name}.npy"), d.detach().cpu())

    # spat_loss = MSELoss(reduction="none")
    from analysis import mask_mbe, mask_mse
    compute_image_loss_plot(sample_image, y_true, mask_mse, mask, True, img_path, cmap)

else:
    raise ValueError(f"Specified {args.mode} must be \"train\" or \"generate\"")
