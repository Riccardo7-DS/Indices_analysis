import os
from analysis import create_runtime_paths, init_tb, DiffusionModel, TrainDiffusion, DataGenerator
from utils.function_clns import init_logging
import torch.nn.functional as F
from tqdm.auto import tqdm
import torch
from torch.nn import MSELoss
from utils.function_clns import CNN_split, config, init_logging
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
import os
import numpy as np
from analysis import masked_mape, masked_rmse, tensor_corr
parser = argparse.ArgumentParser()
parser.add_argument('-f')

### Convlstm parameters
parser.add_argument('--model',type=str,default="DIME",help='DL model training')
parser.add_argument('--step_length',type=int,default=1)
parser.add_argument('--feature_days',type=int,default=90)
parser.add_argument('--auto_ep',type=int,default=100)

parser.add_argument("--normalize", type=bool, default=True, help="Input data normalization")
parser.add_argument("--scatterplot", type=bool, default=True, help="Whether to visualize scatterplot")
args = parser.parse_args()

def training_ddim(args):
    from analysis.configs.config_models import config_ddim as model_config
    from torch.nn import L1Loss, MSELoss
    from analysis import load_autoencoder

    _, log_path, img_path, checkpoint_dir = create_runtime_paths(args)
    logger = init_logging(log_file=os.path.join(log_path, 
                                                      f"dime_days_{args.step_length}"
                                                      f"features_{args.feature_days}.log"))
    writer = init_tb(log_path)

    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)


    data_dir = model_config.data_dir+"/data_convlstm"
    data = np.load(os.path.join(data_dir, "data.npy"))
    target = np.load(os.path.join(data_dir, "target.npy"))

    ################################# Initialize datasets #############################
    train_data, val_data, train_label, val_label, \
        test_valid, test_label = CNN_split(data, target, 
                                           split_percentage=config["MODELS"]["split"])
    
    autoencoder_path = model_config.output_dir + f"/dime/days_{args.step_length}" \
        f"/features_{args.feature_days}/autoencoder/checkpoints" \
        f"/checkpoint_epoch_{args.auto_ep}.pth.tar"
    
    autoencoder = load_autoencoder(autoencoder_path)
    # create a CustomDataset object using the reshaped input data
    train_generator_dataset = DataGenerator(model_config, args, 
                                          train_data, train_label, 
                                          model_config.batch_size, autoencoder)

    input_frames = train_generator_dataset.data.shape[1] + 1 #  add one for the time embedding channel
    model = DiffusionModel(model_config, input_frames).to(model_config.device)
    # model.apply(initialize_weights)

    optimizer = torch.optim.AdamW(model.network.parameters(), 
                                  lr=model_config.learning_rate, 
                                  weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=model_config.scheduler_factor, 
        patience=model_config.scheduler_patience, 
    )
    loss = MSELoss()
    trainer = TrainDiffusion(model, optimizer, loss)
    
    start_epoch  = 0
    noise_loss_records, image_loss_records = [], []

    for epoch in tqdm(range(start_epoch, model_config.epochs)):
        noise_loss, image_loss = trainer.train_step(train_generator_dataset, 
                                                    epoch, writer,
                                                    plot=True)

        log = 'Epoch: {:03d}, Noise Loss: {:.4f}, Image Loss: {:.4f}'
        logger.info(log.format(epoch, np.mean(noise_loss), 
                               np.mean(image_loss)))
        
        noise_loss_records.append(np.mean(noise_loss))
        image_loss_records.append(np.mean(image_loss))

        mean_loss = sum(noise_loss_records) / len(noise_loss_records)
        scheduler.step(mean_loss)

        plt.plot(range(epoch - start_epoch + 1), noise_loss_records, label='noise loss')
        plt.plot(range(epoch - start_epoch + 1), image_loss_records, label='image loss')
        plt.legend()
        plt.savefig(os.path.join(img_path, f'learning_curve_feat_'
                                 f'{args.step_length}.png'))
        plt.close()

training_ddim(args)