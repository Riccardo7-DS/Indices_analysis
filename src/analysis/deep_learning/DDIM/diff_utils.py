import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.nn import MSELoss
from analysis import tensor_corr, EarlyStopping
import logging
import pickle
import pandas as pd
logger = logging.getLogger(__name__)

# Define a custom unpickler that will remap the old module to the new one
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Check if the module is the old one and remap it to the new one
        if module == 'analysis.deep_learning.utils_gwnet':
            module = 'analysis.deep_learning.utils_models'
        return super().find_class(module, name)

# Now use this custom unpickler to load the pickle file
def load_with_custom_unpickler(file_path):
    with open(file_path, "rb") as handle:
        return CustomUnpickler(handle).load()

def load_stored_data(model_config, data_name = "data_convlstm_full" ):
    import pickle
    data_dir = os.path.join(model_config.data_dir, data_name)
    data = np.load(os.path.join(data_dir, "data.npy"))
    target = np.load(os.path.join(data_dir, "target.npy"))
    mask = np.load(os.path.join(data_dir, "mask.npy"))
    path = os.path.join(data_dir, "ndvi_scaler.pickle")
    scaler = load_with_custom_unpickler(path)
    
    squared = model_config.squared
    if squared is True:
        data = data[:, :, :64, :64]
        target = target[:, :64, :64]
        mask = mask[:64, :64]
    return data, target, scaler, mask

def custom_subset_data(
    args,
    model_config, 
    data_name="data_convlstm", 
    start=None, 
    end=None,
    fillna=False,
    autoencoder = True):

    def shift_data_channels(data):
        era_5_data = data[:,:4]
        sm_data = data[:,-4:]
        precp_data = data[:, 4]
        return np.concatenate([np.expand_dims(precp_data, 1), era_5_data, sm_data], axis=1)
    
    from datetime import timedelta
    from utils.function_clns import config

    data, target, scaler, mask = load_stored_data(model_config, data_name)

    data = shift_data_channels(data)

    start_data = config["DEFAULT"]["date_start"]
    end_data = config["DEFAULT"]["date_end"]

    start_pd = pd.to_datetime(start_data, format='%Y-%m-%d')
    end_pd = pd.to_datetime(end_data, format='%Y-%m-%d')
    date_range = pd.date_range(start_pd, end_pd)

    extra_days =  args.auto_days + model_config.num_frames_output + model_config.output_channels if autoencoder else 0

    new_start = pd.to_datetime( start, format='%Y-%m-%d') - timedelta(days=extra_days)
    new_end = pd.to_datetime( end, format='%Y-%m-%d')
    i = date_range.get_loc(new_start)
    e = date_range.get_loc(new_end)

    if fillna is True:
        data = np.nan_to_num(data, nan=-1)
        target = np.nan_to_num(target, nan=-1)
    
    return data[i:e], target[i:e], scaler, mask

def plot_noisy_images(image, imgs, with_orig=False, row_title=None, **imshow_kwargs):
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
    plt.pause(10)
    plt.close()


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_sched'])
        checkp_epoch = checkpoint['epoch']
        logger.info(f"Resuming training from epoch {checkp_epoch}")

    start_epoch = 0 if checkpoint_path is None else checkp_epoch

    return model, optimizer, scheduler, start_epoch

def saveImage(image, image_size, epoch, step, folder):
    import torchvision.transforms as T

    transform=T.ToPILImage()

    image_array = image[0][0].reshape(image_size, image_size)
    image = transform(image_array)

    # Define the full path including the directory and file name
    save_path = os.path.join(folder, f"epoch_{epoch}_step_{step}.png")

    # Save the image as a PNG file in the specified directory
    image.save(save_path)

def diffusion_train_loop(args, 
                         model_config, fdp, 
                         dataloader, 
                         writer, checkpoint_dir, 
                         start_epoch=0):

    early_stopping = EarlyStopping(model_config, verbose=True)

    noise_loss_records = []
    corr_records, r_records = [], []
    
    results_folder = model_config.output_dir + "/dime/temp_images_fdp"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for epoch in range(start_epoch, model_config.epochs):
        for step, (batch, target) in enumerate(dataloader):
            t = torch.randint(0, model_config.timesteps, (batch.shape[0],)).long().to(model_config.device)  # The integers are sampled uniformly from the range [0, timesteps),
            batch = batch.to(model_config.device)
            target = target.squeeze(1).to(model_config.device)

            noise = torch.randn_like(target, dtype=torch.float32)
            # Sampling noise from a gaussian distribution for every training example in batch
            noisy_images = fdp.forward_process(target, t, noise)
            fdp.optimizer.zero_grad()

            if args.conditioning == "none":
                batch = None
            # Forward pass
            predicted_noise = fdp.model(noisy_images, t, batch)
            noise_loss = fdp.loss_fn(predicted_noise, noise)
            corr = tensor_corr(predicted_noise, noise)
            rsq = 1 - noise_loss

            # image_loss = fdp.get_images_denoised(target, t, predicted_noise)
            noise_loss.backward()
            fdp.optimizer.step()

        if epoch != 0 and epoch % model_config.save_and_sample_every == 0:
            # logger.info(f"Loss: {noise_loss}")
            img_shape = (1, 1, model_config.image_size, model_config.image_size)
            sample_im = fdp.p_sample_loop(args, 
                fdp.model, 
                dataloader, 
                img_shape, 
                model_config.timesteps, 
                samples=1,
                random_enabled=True
            )
            saveImage(sample_im, model_config.image_size, epoch, step, results_folder)

        log = 'Epoch: {:03d}, Noise Loss: {:.4f}, Noise correlation: {:.4f}'
        logger.info(log.format(epoch, np.mean(noise_loss.item()),
                               np.mean(corr.item())))

        noise_loss_records.append(noise_loss.item())
        corr_records.append(corr.item())
        r_records.append(rsq.item())
        writer.add_scalar('Loss/train', np.mean(noise_loss.item()), epoch)  #Logging average loss per epoch to tensorboard
        writer.add_scalar("Learning rate", fdp.optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Correlation", np.mean(corr.item()), epoch)
        writer.add_scalar("R squared", np.mean(rsq.item()), epoch)

        model_dict = {
            'epoch': epoch,
            'state_dict': fdp.model.state_dict(),
            'optimizer': fdp.optimizer.state_dict(),
            "lr_sched": fdp.scheduler.state_dict()
        }

        early_stopping(np.mean(noise_loss.item()), 
                        model_dict, epoch, checkpoint_dir)

        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
        
        # image_loss_records.append(np.mean(image_loss.item()))
        mean_loss = sum(noise_loss_records) / len(noise_loss_records)
        fdp.scheduler.step(mean_loss)
        plt.plot(range(epoch - start_epoch + 1), noise_loss_records, label='noise loss')
        # plt.plot(range(epoch + 1), image_loss_records, label='image loss')
        plt.legend()
        plt.savefig(os.path.join(results_folder, f'learning_curve_feat_'
                                 f'{args.step_length}.png'))
        plt.close()

def diffusion_sampling(args, model_config, fdp, dataloader, samples=1, random_enabled=False):
    img_shape = (model_config.batch_size, 1, model_config.image_size, model_config.image_size)
    sample_im = fdp.p_sample_loop(args, 
        fdp.model, 
        dataloader, 
        img_shape, 
        model_config.timesteps, 
        samples,
        random_enabled 
    )
    return sample_im

def compute_image_loss_plot(sample_image, 
                            y_true, 
                            loss_fn, 
                            mask=None, 
                            draw_scatter:bool=False,
                            img_path:str=None,
                            cmap="RdYlGn", 
                            plot_loss:bool=True):
    from analysis.configs.config_models import config_convlstm_1 as model_config
    from analysis import evaluate_hist2d, plot_scatter_hist

    if isinstance(loss_fn, torch.nn.Module):
        logger.info("Shifting numpy arrays to gpu")
        sample_image = torch.from_numpy(sample_image).to(model_config.device)
        y_true = torch.from_numpy(y_true).to(model_config.device)
        if mask is not None:
            mask = torch.from_numpy(mask).to(model_config.device)

    else:
        logger.info("Shifting tensors to cpu")
        sample_image = sample_image.detach().cpu().numpy().squeeze() if isinstance(sample_image, torch.Tensor) else sample_image
        y_true = y_true.detach().cpu().numpy().squeeze() if isinstance(y_true, torch.Tensor) else y_true
        if mask is not None:
            mask = mask.detach().cpu().numpy().squeeze() if isinstance(mask, torch.Tensor) else mask

    
    if img_path is None:
        from definitions import ROOT_DIR
        img_path = os.path.join(ROOT_DIR, "../output")
           
    if isinstance(loss_fn, torch.nn.Module):
        from analysis import masked_custom_loss
        test_loss = masked_custom_loss(loss_fn, 
            sample_image, 
            y_true,
            mask
        )
    else:
        test_loss = loss_fn(sample_image, 
                            y_true,
                            mask)
    final_error = round(np.nanmean(test_loss.item()), 5)
    logger.info(f"The mean loss on the test data is {final_error}")

    if plot_loss is True:
        if isinstance(loss_fn, torch.nn.Module):
            img_loss = masked_custom_loss(loss_fn,
                sample_image, 
                y_true, 
                mask, 
                return_value=False)
        else:
            img_loss = loss_fn(sample_image, 
                y_true, 
                mask, 
                return_value=False)
            
        image_masked = img_loss.detach().cpu() if isinstance(img_loss, torch.Tensor) else img_loss
        vmax = image_masked.max()*0.5
        plt.imshow(image_masked, cmap ="binary", vmax=vmax)
        plt.title("MSE error map")
        plt.colorbar()
        plt.savefig(os.path.join(img_path,"mse_error_map.png")) 
        plt.pause(10)
        plt.close()

    if isinstance(sample_image, torch.Tensor):
        logger.info("Shifting tensors to cpu")
        sample_image = sample_image.detach().cpu().numpy().squeeze() if isinstance(sample_image, torch.Tensor) else sample_image
        y_true = y_true.detach().cpu().numpy().squeeze() if isinstance(y_true, torch.Tensor) else y_true
        if mask is not None:
            mask = mask.detach().cpu().numpy().squeeze() if isinstance(mask, torch.Tensor) else mask

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    if plot_loss is True:
        if len(sample_image.shape) > 2:
            idx = np.random.randint(0, sample_image.shape[0])
            pred_sample = sample_image[idx]
            y_true_sample = y_true[idx]

        c1 = axs[0].imshow(pred_sample, cmap=cmap)
        axs[0].set_title('Predicted Image')

        # Plot the second image
        c2 = axs[1].imshow(y_true_sample, cmap=cmap)
        axs[1].set_title('True Image')

        fig.colorbar(c2, ax=axs[1]) 
        plt.pause(10)
        plt.close()

    if draw_scatter is True:
        nbins= 200
        bin0 = np.linspace(0, 1, nbins+1)
        n = np.zeros((nbins,nbins))
        h, xed, yed = evaluate_hist2d(y_true, sample_image, nbins)
        n = n+h
        plot_scatter_hist(n,  bin0, img_path)

    return final_error
            
        