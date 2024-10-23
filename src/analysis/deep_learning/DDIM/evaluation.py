from analysis import load_stored_data
from analysis.configs.config_models import config_ddim as model_config, config_autodime as auto_config
import pandas as pd
from utils.function_clns import init_logging
from utils.xarray_functions import ndvi_colormap
import torch
from torch.nn import MSELoss
import os
import numpy as np
from analysis import DataGenerator, load_checkpoint, custom_subset_data, create_runtime_paths, autoencoder_wrapper, diffusion_sampling, CustomConvLSTMDataset, TwoResUNet, Forward_diffussion_process
from torch.utils.data import DataLoader
import argparse 
parser = argparse.ArgumentParser(conflict_handler="resolve")
parser.add_argument('-f')
cmap = ndvi_colormap("diverging")

### Autoencoder parameters
parser.add_argument('--model',type=str,default="AUTO_DIME",help='DL model training')
parser.add_argument('--step_length',type=int,default=os.getenv("step_length", 15))

parser.add_argument('--conditioning', type=str, choices=[None, "all", "autoencoder"], default='all')
parser.add_argument('--attention',type=bool,default=False,help='U-NET architecture w/o attention')
parser.add_argument('--auto_train',type=bool,default=os.getenv("auto_train", False))
parser.add_argument('--auto_days',type=int,default=os.getenv("auto_days", 180))

parser.add_argument('--feature_days',type=int,default=os.getenv("feature_days", 90))
parser.add_argument('--auto_ep',type=int,default=os.getenv("auto_ep", 80))
parser.add_argument('--gen_sample',type=int,default=os.getenv("gen_sample", 2))

### diffusion parameters
parser.add_argument('--diff_schedule',type=str,default=os.getenv("diff_schedule", "cosine"))
parser.add_argument('--diff_sample',type=str,default="ddpm")
parser.add_argument('--epoch',type=int,default=os.getenv("epoch", 0), help="diffusion model trained epochs")

parser.add_argument("--normalize", type=bool, default=True, help="Input data normalization")
parser.add_argument("--scatterplot", type=bool, default=True, help="Whether to visualize scatterplot")
parser.add_argument('--mode', type=str, default=os.getenv("mode", "generate"))

args = parser.parse_args()

################################# Autoencoder #############################

autoencoder = autoencoder_wrapper(args, 
    auto_config, 
    generate_output=False
)
##########################################################################

parser.add_argument('--model',type=str,default="DIME",help='DL model training')
args = parser.parse_args()

start = "2020-01-01"
end = "2022-12-31"

data_name = "data_gnn_drought"
_, log_path, img_path, checkpoint_dir = create_runtime_paths(args)
checkpoint_path  = checkpoint_dir + f"/checkpoint_epoch_{args.epoch}.pth.tar"

logger = init_logging(log_file=os.path.join(log_path, 
                            f"dime_days_{args.step_length}"
                            f"features_{args.feature_days}.log"))

data, target, scaler, mask = custom_subset_data(args, 
    model_config, 
    data_name, 
    start, 
    end,
    True, 
    True
)

print(data.shape)
print(target.shape)


args.feature_days = 180
dataset_auto = CustomConvLSTMDataset(model_config, 
    args, 
    data, 
    target
)
args.feature_days = 90
datagenerator = DataGenerator(model_config, 
    args, 
    data,
    target, 
    autoencoder, 
    past_data=dataset_auto.data[:, :, -1]
)

dataloader = DataLoader(datagenerator, 
    shuffle=True, 
    batch_size=model_config.batch_size
)

model = TwoResUNet(dim=model_config.widths[0]*2, 
    channels=datagenerator.data.shape[1]+1,
    dim_mults=(1, 2, 4, 8, 16),
    out_dim=model_config.output_channels).to(model_config.device)

optimizer = torch.optim.AdamW(model.parameters(), 
    lr=model_config.learning_rate, 
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=model_config.scheduler_factor, 
    patience=model_config.scheduler_patience
)
loss_fn = MSELoss()

if args.epoch > 0:
    model, optimizer, scheduler, start_epoch = load_checkpoint(checkpoint_path, 
        model, 
        optimizer, 
        scheduler
    )
else:
    start_epoch = 0


fdp = Forward_diffussion_process(args, 
    model_config, 
    model,  
    optimizer, 
    scheduler, 
    loss_fn
)

logger.info(f"Starting generating from diffusion model with {args.diff_schedule} schedule " 
       f"and sampling technique {args.diff_sample} with model trained for {start_epoch} epochs")

sample_image, y_true = diffusion_sampling(args, 
    model_config, 
    fdp, 
    dataloader, 
    samples=args.gen_sample,
    random_enabled=False
)
if args.normalize is True:
    sample_image = scaler.inverse_transform(sample_image)
    y_true = scaler.inverse_transform(y_true)   

sample_image = torch.clamp(sample_image, -1, 1)
y_true = torch.clamp(y_true, -1, 1)
mask = torch.from_numpy(mask).to(model_config.device)

from analysis import mask_mbe, mask_mse, compute_image_loss_plot
compute_image_loss_plot(sample_image, 
    y_true, 
    mask_mse, 
    mask, 
    True, 
    img_path, 
    cmap
)

out_path = os.path.join(img_path, "output_data")
for d, name in zip([sample_image, y_true, mask],  ['pred_data_dr', 'true_data_dr', 'mask_dr']):
    np.save(os.path.join(out_path, f"{name}.npy"), d.detach().cpu)