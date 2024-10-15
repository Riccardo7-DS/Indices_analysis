import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from analysis import CustomConvLSTMDataset, create_runtime_paths, plot_first_n_images
# from analysis.deep_learning.DDIM.model import DiffusionModel
from torch.nn import MSELoss
from utils.function_clns import CNN_split, config, init_logging
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
import os
from analysis import masked_mape, masked_rmse, tensor_corr, EarlyStopping, DataGenerator
        
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class TimeEncoder(nn.Module):
    def __init__(self, enc_shape):
        super(TimeEncoder, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, padding=1, stride=1),  # Adjust kernel size based on input size
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=1, padding=0),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=0),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=0),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(512),
            nn.ELU(),          
            nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=0),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(1024),
            nn.ELU(),         
            nn.Conv1d(1024, 2048, kernel_size=1, stride=1, padding=0),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(2048),
            nn.ELU(),
            Flatten(),
            nn.Linear(2048, 1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, enc_shape),
            nn.ELU()
        )
    def forward(self, x):
        return self.sequence.forward(x)

class TimeDecoder(nn.Module):
    def __init__(self, output_shape, enc_shape):
        super(TimeDecoder, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(enc_shape, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, output_shape),
            nn.ELU()
        )

    def forward(self, x):
        return self.sequence.forward(x)
    
class TimeAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(TimeAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    

def pipeline_autoencoder(args, 
                        data=None, 
                        target=None, 
                        checkpoint_path = None,
                        output_shape:int = 20,
                        dataset:str=None):
    
    from analysis.configs.config_models import config_autodime as model_config

    if args.auto_days != args.feature_days:
        args.feature_days = args.auto_days

    if data is None:
        data_dir = model_config.data_dir+"/data_convlstm_full"
        data = np.load(os.path.join(data_dir, "data.npy"))
        target = np.load(os.path.join(data_dir, "target.npy"))

    _, log_path, img_path, checkpoint_dir = create_runtime_paths(args)
    logger = init_logging(os.path.join(log_path, "autoencoder.log"))

    ################################# Initialize datasets #############################
    train_data, val_data, train_label, val_label, \
        test_data, test_label = CNN_split(data, target, 
                                           split_percentage=config["MODELS"]["split"])
    # create a CustomDataset object using the reshaped input data
    train_dataset = CustomConvLSTMDataset(model_config, args, 
                                          train_data, train_label)
    train_dataloader = DataLoader(train_dataset,
                            batch_size=model_config.batch_size, shuffle=True)
    
    encoder = TimeEncoder(output_shape).to(model_config.device)
    decoder = TimeDecoder(args.feature_days, output_shape).to(model_config.device)
    autoencoder = TimeAutoencoder(encoder, decoder).to(model_config.device)
    criterion = MSELoss()
    optimizer = torch.optim.AdamW(autoencoder.parameters(), 
                                  lr=model_config.learning_rate,
                                  weight_decay=1e-3,  betas=(0.5, 0.9))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.25
            )
    early_stopping = EarlyStopping(model_config, verbose=True)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        autoencoder.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_sched'])
        checkp_epoch = checkpoint['epoch']
        logger.info(f"Resuming training from epoch {checkp_epoch}")

    # number of epochs to train the model
    n_epochs = model_config.epochs

    train_loss_records = []

    start_epoch = 0 if checkpoint_path is None else checkp_epoch 

    if args.auto_train is True:

        logger.info(f"Generating new autoencoder with {args.feature_days} days of features")

        for epoch in range(start_epoch, n_epochs + 1):
            # monitor training loss
            ######################################
            epoch_records = {'loss': [], "mape": [], "rmse": [], "corr": []}

            for batch_idx, (data, target) in enumerate(train_dataloader):
                # _ stands in for labels, here
                images = data[:, -1, :, :, :].to(model_config.device)
                b, t, h, w = images.size() 
                imag = images.permute(0, 2, 3, 1)
                imag  =  imag.reshape(b*h*w, t)
                imag = torch.unsqueeze(imag, 1) 
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = autoencoder(imag)
                outputs = outputs.view(b, h, w, t).permute(0, 3, 1, 2)
                # calculate the loss
                loss = criterion(outputs, images)

                mape = masked_mape(outputs, images).item()
                rmse = masked_rmse(outputs, images).item()
                corr = tensor_corr(outputs, images).item()
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update running training loss
                epoch_records['loss'].append(loss.item())
                epoch_records["rmse"].append(rmse)
                epoch_records["mape"].append(mape)
                epoch_records["corr"].append(corr)

            # Log metrics for the entire epoch
            model_dict = {
                    'epoch': epoch,
                    'state_dict': autoencoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "lr_sched": scheduler.state_dict()
                }

            log = 'Epoch: {:03d}, Loss: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}, CORR: {:.4f}'
            logger.info(log.format(epoch, 
                                   np.mean(epoch_records['loss']),
                                   np.mean(epoch_records['mape']),
                                   np.mean(epoch_records['rmse']),
                                   np.mean(epoch_records['corr'])))

            early_stopping(np.mean(epoch_records['loss']), 
                               model_dict, epoch, checkpoint_dir)

            if early_stopping.early_stop:
                logger.info("Early stopping")
                return epoch

            mean_loss = sum(epoch_records['loss']) / len(epoch_records['loss'])
            scheduler.step(mean_loss)

            train_loss_records.append(np.mean(epoch_records['loss']))

            plt.plot(range(epoch-start_epoch+1), train_loss_records, label='train')
            plot_first_n_images(outputs, 10, True, "predicted_img", img_path)
            plot_first_n_images(images, 10, True, "true_img", img_path)
            plt.legend()
            plt.savefig(os.path.join(img_path, f'learning_curve_feat_'
                                         f'{args.step_length}.png'))

        return epoch
    
    elif args.auto_train is False:
        
        logger.info(f"Generating autoencoder output with {args.feature_days} days of features")
        dataset = "train" if dataset is None else dataset
        auto_epoch = None
        from analysis import default, load_autoencoder
        autoencoder_path = model_config.output_dir + \
            f"/dime/autoencoder/checkpoints" \
            "/checkpoint_epoch_{}.pth.tar"

        auto_epoch = default(auto_epoch, args.auto_ep)

        autoencoder = load_autoencoder(autoencoder_path.format(auto_epoch), 
            feature_days=args.auto_days,
            output_shape=args.auto_days//5
        )
        DataGenerator(model_config, 
            args, 
            locals()[f"{dataset}_data"], 
            locals()[f"{dataset}_label"], 
            autoencoder, 
            data_split=dataset
        )

if __name__== "__main__":
    import pyproj
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    
    ### Convlstm parameters
    parser.add_argument('--model',type=str,default="AUTO_DIME",help='DL model training')
    parser.add_argument('--step_length',type=int,default=15)
    parser.add_argument('--feature_days',type=int,default=180)
    parser.add_argument('--auto_days',type=int,default=os.getenv("feature_days", 180))

    parser.add_argument("--auto_train", type=bool, default=False)
    parser.add_argument("--auto_ep", type=bool, default=244)
    
    parser.add_argument("--country", type=list, default=["Kenya","Somalia","Ethiopia","Djibouti"], help="Location for dataset")
    parser.add_argument("--region", type=list, default=None, help="Location for dataset")
    parser.add_argument("--normalize", type=bool, default=True, help="Input data normalization")
    parser.add_argument("--scatterplot", type=bool, default=True, help="Whether to visualize scatterplot")

    parser.add_argument("--mode", type=str, default=os.getenv("mode", "test"))

    args = parser.parse_args()
    os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()
    pipeline_autoencoder(args)