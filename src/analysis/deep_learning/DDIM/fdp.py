from analysis import Forward_diffussion_process, TwoResUNet, plot_noisy_images, load_stored_data, DataGenerator, tensor_corr, UNET, create_runtime_paths, init_tb, EarlyStopping
from analysis import diffusion_train_loop, diffusion_sampling, compute_image_loss_plot, autoencoder_wrapper
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
import matplotlib.pyplot as plt
import torch
import gc
gc.collect()
torch.cuda.empty_cache()
cmap = ndvi_colormap("sequential")

parser = argparse.ArgumentParser(conflict_handler="resolve")
parser.add_argument('-f')

### Autoencoder parameters
parser.add_argument('--model',type=str,default="AUTO_DIME",help='DL model training')
parser.add_argument('--step_length',type=int,default=os.getenv("step_length", 15))

parser.add_argument('--auto_train',type=bool,default=os.getenv("auto_train", False))
parser.add_argument('--auto_days',type=int,default=os.getenv("auto_days", 180))

parser.add_argument('--feature_days',type=int,default=os.getenv("feature_days", 90))
parser.add_argument('--auto_ep',type=int,default=os.getenv("auto_ep", 180))
parser.add_argument('--gen_sample',type=int,default=os.getenv("gen_sample", 2))

### diffusion parameters
parser.add_argument('--diff_schedule',type=str,default=os.getenv("diff_schedule", "cosine"))
parser.add_argument('--diff_sample',type=str,default="ddpm")
parser.add_argument('--epoch',type=int,default=os.getenv("epoch", 0), help="diffusion model trained epochs")

parser.add_argument("--normalize", type=bool, default=True, help="Input data normalization")
parser.add_argument("--scatterplot", type=bool, default=True, help="Whether to visualize scatterplot")
parser.add_argument('--mode', type=str, default=os.getenv("mode", "generate"))

args = parser.parse_args()

data, target, scaler, mask = load_stored_data(model_config)

################################# Autoencoder #############################

autoencoder = autoencoder_wrapper(args, model_config, data, target)

################################# Initialize datasets #############################

### Diffusion parameters
parser.add_argument('--model',type=str,default="DIME",help='DL model training')
args = parser.parse_args()

_, log_path, img_path, checkpoint_dir = create_runtime_paths(args)
checkpoint_path  = checkpoint_dir + f"/checkpoint_epoch_{args.epoch}.pth.tar"

logger = init_logging(log_file=os.path.join(log_path, 
                            f"dime_days_{args.step_length}"
                            f"features_{args.feature_days}.log"))
writer = init_tb(log_path)

train_data, val_data, train_label, val_label, \
    test_data, test_label = CNN_split(data, target, 
    split_percentage=config["MODELS"]["split"])

#from analysis import plot_first_n_images

# plot_first_n_images(train_data, 9)
# plt.show()
# plt.close()

# create a CustomDataset object using the reshaped input data
datagenrator_train = DataGenerator(model_config, args,
                            train_data, train_label, 
                            autoencoder, data_split="train")

dataloader_train = DataLoader(datagenrator_train, 
                        shuffle=True, 
                        batch_size=model_config.batch_size)

datagenrator_test= DataGenerator(model_config, args,
                            test_data, test_label, 
                            autoencoder, data_split="test")

dataloader_test = DataLoader(datagenrator_test, 
                        shuffle=True, 
                        batch_size=model_config.batch_size)

########################### Models and training functions #######################

# model = UNET(dim=datagenrator_train.data.shape[-1], 
#             channels=datagenrator_train.data.shape[1]+1,
#             dim_mults=(1, 2, 4),
#             out_dim=model_config.output_channels).to(model_config.device)

model = TwoResUNet(dim=model_config.widths[0]*2, 
            channels=datagenrator_train.data.shape[1]+1,
            dim_mults=(1, 2, 4, 8, 16),
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

x_features, img = next(iter(dataloader_train))
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
                         dataloader_train,
                         writer, checkpoint_dir, 
                         start_epoch)
    
elif args.mode == "generate":
    logger.info(f"Starting generating from diffusion model with {args.diff_schedule} schedule " 
           f"and sampling technique {args.diff_sample} with model trained for {start_epoch} epochs") 
    sample_image, y_true = diffusion_sampling(args, model_config, fdp, 
                                              dataloader_test, 
                                              samples=args.gen_sample,
                                              random_enabled=True)
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
