import xarray as xr
import numpy as np
import torch
from torchvision import transforms
import os
from p_drought_indices.functions.function_clns import load_config, interpolate_prepare, prepare, CNN_split, CNN_preprocessing, get_lat_lon_window
import numpy as np
import torch
from torch.utils.data import Dataset
from p_drought_indices.analysis.DeepLearning.dataset import CustomDataset
from torch.utils.data import DataLoader
import pickle


def spi_ndvi_convlstm(CONFIG_PATH, time_start, time_end):
    config_file = load_config(CONFIG_PATH=CONFIG_PATH)

    # Open the NetCDF file with xarray
    dataset = prepare(xr.open_dataset(os.path.join(config_file['NDVI']['ndvi_path'], 'smoothed_ndvi_1.nc'))).sel(time=slice(time_start,time_end))[["time","lat","lon","ndvi"]]

    prod = "ERA5"
    late = 90

    path = config_file['SPI']['ERA5']['path']
    file = "era5_land_merged.nc" #f"ERA5_spi_gamma_{late}.nc"
    precp_ds = prepare(xr.open_dataset(os.path.join(path, file)))
    var_target = [var for var in precp_ds.data_vars][0] #"spi_gamma_{}".format(late)
    print(f"The {prod} raster has spatial dimensions:", precp_ds.rio.resolution())

    #### training parameters
    train_split = 0.8
    dim=64
    preprocess_type="None"

    ### Load dataset
    file_path = os.path.join(config_file["DEFAULT"]["data"],'preprocessed_data.pkl')
    if os.path.exists(file_path):
        print("The file exists. Proceeding with the analysis")
        with open(file_path, 'rb') as file:
            train_data, test_data, train_label, test_label = pickle.load(file)
    else:
        print("The file does not exist. Proceeding with preprocessing")
        idx_lat, lat_max, idx_lon, lon_min = get_lat_lon_window(precp_ds, dim)
        sub_precp = prepare(precp_ds).sel(time=slice(time_start,time_end))\
            .sel(lat=slice(lat_max, idx_lat), lon=slice(lon_min, idx_lon))
        ds = dataset["ndvi"].rio.reproject_match(sub_precp[var_target])#.rename({'x':'lon','y':'lat'})
        

        train_data, test_data, train_label, test_label = CNN_preprocessing(ds, sub_precp, var_origin="ndvi", var_target=var_target, preprocess_type=preprocess_type,  split=train_split)
        # Save the image data using pickle
        with open(file_path, 'wb') as file:
            pickle.dump((train_data, test_data, train_label, test_label), file)
        print("Data written to pickle file")
    
    return sub_precp, ds


def training_lstm(CONFIG_PATH:str, data:np.array, target:np.array, train_split:float = 0.8):
    import numpy as np
    from p_drought_indices.configs.config_3x3_16_3x3_32_3x3_64 import config
    from torch.nn import MSELoss
    import matplotlib.pyplot as plt
    from p_drought_indices.analysis.DeepLearning.ConvLSTM import ConvLSTM, train_loop, valid_loop, build_logging
    import numpy as np
    from p_drought_indices.functions.function_clns import load_config

    #### training parameters
    train_data, test_data, train_label, test_label = CNN_split(data, target, 
                                                               split_percentage=train_split)
    config_file = load_config(CONFIG_PATH=CONFIG_PATH)
    batch_size = config_file["CONVLSTM"]["batch_size"]

    # create a CustomDataset object using the reshaped input data
    train_dataset = CustomDataset(train_data, train_label)
    test_dataset = CustomDataset(test_data, test_label)
    
    # create a DataLoader object that uses the dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    
    ### check shape of data
    
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs = inputs.float()
        targets = targets.float()
        print(inputs.shape, targets.shape, inputs.max(), inputs.min())
    
    
    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
        inputs = inputs.float()
        targets = targets.float()
        print(inputs.shape, targets.shape, inputs.max(), inputs.min())


    #### Start training
    
    name = '3x3_16_3x3_32_3x3_64'

    ### parrameters for early stopping 
    # Define best_score, counter, and patience for early stopping:
    best_score = None
    counter = 0
    patience = 50
    checkpoint_path = os.path.join(config.output_dir,"checkpoints")
    
    logger = build_logging(config)
    model = ConvLSTM(config).to(config.device)
    #criterion = CrossEntropyLoss().to(config.device)
    #criterion = torch.nn.MSELoss().to(config.device)
    criterion = MSELoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_records, valid_records, test_records = [], [], []
    for epoch in range(config.epochs):
        epoch_records = train_loop(config, logger, epoch, model, train_dataloader, criterion, optimizer)
        train_records.append(np.mean(epoch_records['loss']))
        epoch_records = valid_loop(config, logger, epoch, model, test_dataloader, criterion)
        valid_records.append(np.mean(epoch_records['loss']))
        if best_score is None:
            best_score = epoch_records['loss']
        else:
            # Check if val_loss improves or not.
            if epoch_records['loss'] < best_score:
                # val_loss improves, we update the latest best_score, 
                # and save the current model
                best_score = epoch_records['loss']
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                torch.save({'state_dict':model.state_dict()}, checkpoint_path)
            else:
                # val_loss does not improve, we increase the counter, 
                # stop training if it exceeds the amount of patience
                counter += 1
                if counter >= patience:
                    break
        plt.plot(range(epoch + 1), train_records, label='train')
        plt.plot(range(epoch + 1), valid_records, label='valid')
        plt.legend()
        plt.savefig(os.path.join(config.output_dir, '{}.png'.format(name)))
        plt.close()

if __name__=="__main__":
    from p_drought_indices.analysis.DeepLearning.pipeline_gwnet import data_preparation 
    import pickle
    import os
    import matplotlib.pyplot as plt
    from p_drought_indices.functions.function_clns import load_config, prepare, CNN_split, interpolate_prepare
    import numpy as np
    from p_drought_indices.analysis.DeepLearning.dataset import CustomDataset
    from torch.utils.data import DataLoader

    CONFIG_PATH = "config.yaml"
    sub_precp, ds = data_preparation(CONFIG_PATH)
    sub_precp = sub_precp.to_dataset()
    data, target = interpolate_prepare(sub_precp, ds)
    train_split = 0.8
    training_lstm(CONFIG_PATH, data, target, train_split = train_split)