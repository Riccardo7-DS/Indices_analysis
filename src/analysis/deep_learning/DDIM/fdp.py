from analysis import Forward_diffussion_process, TwoResUNet, plot_noisy_images, load_stored_data, DataGenerator, tensor_corr, UNET, create_runtime_paths, init_tb, EarlyStopping
from analysis import diffusion_train_loop, diffusion_sampling, compute_image_loss_plot
from analysis.configs.config_models import config_ddim as model_config
from torch.nn import L1Loss, MSELoss
import numpy as np
import os
import argparse
from analysis import load_autoencoder, load_checkpoint
import torch
import matplotlib.pyplot as plt
import logging
from torch.utils.data import DataLoader
from utils.function_clns import config, CNN_split, init_logging
from utils.xarray_functions import ndvi_colormap

cmap = ndvi_colormap("diverging")

parser = argparse.ArgumentParser()
parser.add_argument('-f')
### Convlstm parameters
parser.add_argument('--model',type=str,default="DIME",help='DL model training')
parser.add_argument('--step_length',type=int,default=1)
parser.add_argument('--diff_schedule',type=str,default=os.getenv("diff_schedule", "linear"))
parser.add_argument('--diff_sample',type=str,default="ddpm")


parser.add_argument('--feature_days',type=int,default=90)
parser.add_argument('--auto_ep',type=int,default=100)
parser.add_argument('--epoch',type=int,default=os.getenv("epoch", 0))
parser.add_argument('--gen_sample',type=int,default=os.getenv("gen_sample", 2))

parser.add_argument("--normalize", type=bool, default=True, help="Input data normalization")
parser.add_argument("--scatterplot", type=bool, default=True, help="Whether to visualize scatterplot")
parser.add_argument('--mode', type=str, default=os.getenv("mode", "train"))

args = parser.parse_args()

_, log_path, img_path, checkpoint_dir = create_runtime_paths(args)
checkpoint_path  = checkpoint_dir + f"/checkpoint_epoch_{args.epoch}.pth.tar"

logger = init_logging(log_file=os.path.join(log_path, 
                                                      f"dime_days_{args.step_length}"
                                                      f"features_{args.feature_days}.log"))
writer = init_tb(log_path)

data, target = load_stored_data(model_config)

################################# Initialize datasets #############################
train_data, val_data, train_label, val_label, \
    test_valid, test_label = CNN_split(data, target, 
                                       split_percentage=config["MODELS"]["split"])


autoencoder_path = model_config.output_dir + \
    f"/dime/days_{args.step_length}" \
    f"/features_{args.feature_days}/autoencoder/checkpoints" \
    f"/checkpoint_epoch_{args.auto_ep}.pth.tar"

autoencoder = load_autoencoder(autoencoder_path)
# create a CustomDataset object using the reshaped input data
datagenrator_train = DataGenerator(model_config, args,
                            train_data, train_label, 
                            autoencoder)

dataloader = DataLoader(datagenrator_train, 
                        shuffle=True, 
                        batch_size=model_config.batch_size)

########################### Models and training functions #######################

# model = UNET(dim=datagenrator_train.data.shape[-1], 
#             channels=datagenrator_train.data.shape[1]+1,
#             dim_mults=(1, 2, 4),
#             out_dim=model_config.output_channels).to(model_config.device)

model = TwoResUNet(dim=model_config.widths[0], 
            channels=datagenrator_train.data.shape[1]+1,
            dim_mults=(1, 2, 4, 6),
            out_dim=model_config.output_channels).to(model_config.device)

optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=model_config.learning_rate, 
                              weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=model_config.scheduler_factor, 
    patience=model_config.scheduler_patience
)

if args.epoch > 0:
    model, optimizer, scheduler, start_epoch = load_checkpoint(checkpoint_path, 
                                                               model, 
                                                               optimizer, 
                                                               scheduler)
else:
    start_epoch = 0

loss_fn = MSELoss()
fdp = Forward_diffussion_process(args, model_config, 
                                 model,  optimizer, 
                                 scheduler, loss_fn)

################### test image ############################

x_features, img = next(iter(dataloader))
image = img[0].to(model_config.device)
# img = fdp.get_noisy_image(image.to(model_config.device), t)

plot_noisy_images(image, [fdp.get_noisy_image(image, torch.tensor([t])\
                          .to(model_config.device)) for t in [0, 20, 50, 75, 100, 150, 199]],
                          **{"cmap": cmap})

###########################################################

if args.mode == "train":
    logger.info(f"Starting training diffusion model with {args.diff_schedule} schedule " 
           f"and sampling technique {args.diff_sample} at epoch {start_epoch}") 
    diffusion_train_loop(args, 
                         model_config, fdp, 
                         dataloader, datagenrator_train,
                         writer, checkpoint_dir, 
                         start_epoch)
    
elif args.mode == "generate":
    logger.info(f"Starting generating from diffusion model with {args.diff_schedule} schedule " 
           f"and sampling technique {args.diff_sample} with model trained for {start_epoch} epochs") 
    sample_image, y_true = diffusion_sampling(args, model_config, fdp, 
                                              datagenrator_train, samples=args.gen_sample)
    compute_image_loss_plot(sample_image, y_true, loss_fn, cmap)

else:
    raise ValueError(f"Specified {args.mode} must be \"train\" or \"generate\"")
