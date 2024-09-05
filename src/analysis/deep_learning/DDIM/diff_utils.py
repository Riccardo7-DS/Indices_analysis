import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.nn import MSELoss
from analysis import tensor_corr, EarlyStopping
import logging
logger = logging.getLogger(__name__)


def load_stored_data(model_config, squared:bool = True):
    data_dir = model_config.data_dir+"/data_convlstm"
    data = np.load(os.path.join(data_dir, "data.npy"))
    target = np.load(os.path.join(data_dir, "target.npy"))
    
    squared = True
    if squared is True:
        data = data[:, :, :64, :64]
        target = target[:, :64, :64]
    return data, target

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
    plt.pause(5)
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

    image_array = image[0][0, 0].reshape(image_size, image_size)
    image = transform(image_array)

    # Define the full path including the directory and file name
    save_path = os.path.join(folder, f"epoch_{epoch}_step_{step}.png")

    # Save the image as a PNG file in the specified directory
    image.save(save_path)

def diffusion_train_loop(args, 
                         model_config, fdp, 
                         dataloader, datagenrator_train,  
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
            shape = (1, 1, model_config.image_size, model_config.image_size)
            sample_im = fdp.sample(args, fdp.model, datagenrator_train, shape, model_config.timesteps)
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

def diffusion_sampling(args, model_config, fdp, dataloader, samples=1):
    shape = (samples, 1, model_config.image_size, model_config.image_size)
    sample_im = fdp.sample(args, fdp.model, dataloader, shape, model_config.timesteps, samples)
    return sample_im

def compute_image_loss_plot(sample_image, y_true, loss_fn, cmap, plot_loss:bool=True):
    
    test_loss = loss_fn(sample_image.squeeze(), y_true.squeeze())
    logger.info(f"The mean loss on the test data is {round(np.mean(test_loss.item()), 3)}")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    if len(sample_image.squeeze().shape) > 2:
        sample_image = sample_image[0]
        y_true = y_true[0]
        
    c1 = axs[0].imshow(sample_image.detach().cpu().numpy().squeeze() , cmap=cmap)
    axs[0].set_title('Predicted Image')

    # Plot the second image
    c2 = axs[1].imshow(y_true.detach().cpu().numpy().squeeze(), cmap=cmap)
    axs[1].set_title('True Image')

    fig.colorbar(c2, ax=axs[1]) 
    plt.pause(10)
    plt.close()

    if plot_loss is True:
        spat_loss = MSELoss(reduction="none")
        img_loss = spat_loss(sample_image.squeeze(), y_true.squeeze())
        plt.imshow(img_loss.squeeze(), vmax=0.1)
        plt.title("MSE error map")
        plt.colorbar()
        plt.pause(10)
        plt.close()