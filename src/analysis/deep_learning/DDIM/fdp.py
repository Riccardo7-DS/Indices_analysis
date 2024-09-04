from analysis import Forward_diffussion_process, DataGenerator, tensor_corr, UNET, create_runtime_paths, init_tb, EarlyStopping
from analysis.configs.config_models import config_ddim as model_config
from torch.nn import L1Loss, MSELoss
import numpy as np
import os
import argparse
from analysis import load_autoencoder
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
parser.add_argument('--diff_schedule',type=str,default="linear")
parser.add_argument('--diff_sample',type=str,default="ddpm")

parser.add_argument('--feature_days',type=int,default=90)
parser.add_argument('--auto_ep',type=int,default=100)

parser.add_argument("--normalize", type=bool, default=True, help="Input data normalization")
parser.add_argument("--scatterplot", type=bool, default=True, help="Whether to visualize scatterplot")
args = parser.parse_args()

_, log_path, img_path, checkpoint_dir = create_runtime_paths(args)
logger = init_logging(log_file=os.path.join(log_path, 
                                                      f"dime_days_{args.step_length}"
                                                      f"features_{args.feature_days}.log"))
writer = init_tb(log_path)

data_dir = model_config.data_dir+"/data_convlstm"
data = np.load(os.path.join(data_dir, "data.npy"))
target = np.load(os.path.join(data_dir, "target.npy"))

checkpoint_path  = model_config.output_dir + "/dime/days_1/features_90/checkpoints/checkpoint_epoch_141.pth.tar"

squared = True
if squared is True:
    data = data[:, :, :64, :64]
    target = target[:, :64, :64]

################################# Initialize datasets #############################
train_data, val_data, train_label, val_label, \
    test_valid, test_label = CNN_split(data, target, 
                                       split_percentage=config["MODELS"]["split"])

autoencoder_path = model_config.output_dir + f"/dime/days_{args.step_length}" \
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

timesteps = 1000
input_frames = datagenrator_train.data.shape[1]

model = UNET(dim=datagenrator_train.data.shape[-1], 
            channels=input_frames+1,
            dim_mults=(1, 2, 4),
            out_dim=model_config.output_channels).to(model_config.device)

optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=model_config.learning_rate, 
                              weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=model_config.scheduler_factor, 
    patience=model_config.scheduler_patience
)

early_stopping = EarlyStopping(model_config, logger, verbose=True)

if checkpoint_path is not None:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['lr_sched'])
    checkp_epoch = checkpoint['epoch']
    logger.info(f"Resuming training from epoch {checkp_epoch}")

start_epoch = 0 if checkpoint_path is None else checkp_epoch

fdp = Forward_diffussion_process(args, model_config, model,  timesteps)

################### test image ############################

t = torch.tensor([50]).to(model_config.device)
x_features, img = next(iter(dataloader))
image = img[0].to(model_config.device)
# img = fdp.get_noisy_image(image.to(model_config.device), t)

def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200,200), 
                            nrows=num_rows, 
                            ncols=num_cols, squeeze=False)
    
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img.squeeze().detach().cpu()), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.pause(5)
    plt.close()

plot([fdp.get_noisy_image(image, torch.tensor([t])\
                          .to(model_config.device)) for t in [0, 20, 50, 75, 100, 150, 199]],
                          **{"cmap": cmap})

###########################################################

loss_fn = L1Loss()
save_and_sample_every = 10

def saveImage(image, image_size, epoch, step, folder):
    import torchvision.transforms as T

    transform=T.ToPILImage()

    image_array = image[0][0, 0].reshape(image_size, image_size)
    image = transform(image_array)

    # Define the full path including the directory and file name
    save_path = os.path.join(folder, f"epoch_{epoch}_step_{step}.png")

    # Save the image as a PNG file in the specified directory
    image.save(save_path)

def train_loop(model_config, fdp, dataloader, timesteps, writer):
    noise_loss_records = []
    corr_records, r_records = [], []
    
    results_folder = model_config.output_dir + "/dime/temp_images_fdp"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for epoch in range(start_epoch, model_config.epochs):
        for step, (batch, target) in enumerate(dataloader):
            t = torch.randint(0, timesteps, (batch.shape[0],)).long().to(model_config.device)  # The integers are sampled uniformly from the range [0, timesteps),
            batch = batch.to(model_config.device)
            target = target.squeeze(1).to(model_config.device)

            noise = torch.randn_like(target, dtype=torch.float32)
            # Sampling noise from a gaussian distribution for every training example in batch
            noisy_images = fdp.forward_process(target, t, noise)
            optimizer.zero_grad()

            # Forward pass
            predicted_noise = fdp.model(noisy_images, t, batch)
            noise_loss = loss_fn(predicted_noise, noise)
            corr = tensor_corr(predicted_noise, noise)
            rsq = 1 - noise_loss

            # image_loss = fdp.get_images_denoised(target, t, predicted_noise)
            noise_loss.backward()
            optimizer.step()

        if epoch != 0 and epoch % save_and_sample_every == 0:
            # logger.info(f"Loss: {noise_loss}")
            shape = (1, 1, model_config.image_size, model_config.image_size)
            sample_im = fdp.sample(args, fdp.model, datagenrator_train, shape, timesteps)
            saveImage(sample_im, model_config.image_size, epoch, step, results_folder)

        log = 'Epoch: {:03d}, Noise Loss: {:.4f}, Noise correlation: {:.4f}'
        logger.info(log.format(epoch, np.mean(noise_loss.item()),
                               np.mean(corr.item())))

        noise_loss_records.append(noise_loss.item())
        corr_records.append(corr.item())
        r_records.append(rsq.item())
        writer.add_scalar('Loss/train', np.mean(noise_loss.item()), epoch)  #Logging average loss per epoch to tensorboard
        writer.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Correlation", np.mean(corr.item()), epoch)
        writer.add_scalar("R squared", np.mean(rsq.item()), epoch)

        model_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "lr_sched": scheduler.state_dict()
        }

        early_stopping(np.mean(noise_loss.item()), 
                        model_dict, epoch, checkpoint_dir)

        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
        
        # image_loss_records.append(np.mean(image_loss.item()))
        mean_loss = sum(noise_loss_records) / len(noise_loss_records)
        scheduler.step(mean_loss)
        plt.plot(range(epoch - start_epoch + 1), noise_loss_records, label='noise loss')
        # plt.plot(range(epoch + 1), image_loss_records, label='image loss')
        plt.legend()
        plt.savefig(os.path.join(results_folder, f'learning_curve_feat_'
                                 f'{args.step_length}.png'))
        plt.close()

def sampling(model_config, fdp, dataloader, timesteps, samples=1):
    shape = (samples, 1, model_config.image_size, model_config.image_size)
    sample_im = fdp.sample(args, fdp.model, dataloader, shape, timesteps)
    return sample_im

def compute_image_loss_plot(sample_image, y_true, cmap):
    test_loss = loss_fn(sample_image.squeeze(), y_true.squeeze())
    logger.info(f"The mean loss on the test data is {round(np.mean(test_loss.item()), 3)}")


    fig, axs = plt.subplots(1, 2, figsize=(10, 5))


    c1 = axs[0].imshow(sample_image.detach().cpu().numpy().squeeze() , cmap=cmap)
    axs[0].set_title('Predicted Image')

    # Plot the second image
    c2 = axs[1].imshow(y_true.detach().cpu().numpy().squeeze(), cmap=cmap)
    axs[1].set_title('True Image')

    fig.colorbar(c2, ax=axs[1]) 
    plt.pause(10)

    plt.close()

sample_image, y_true = sampling(model_config, fdp, datagenrator_train, timesteps=500, samples=16)
compute_image_loss_plot(sample_image, y_true, cmap)