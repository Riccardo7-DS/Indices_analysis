from utils.function_clns import config, get_tensor_memory, get_vram, get_ram
import xarray as xr
import os
import numpy as np
from scipy.sparse import linalg

import scipy.sparse as sp
import torch
import time
import argparse
import torch.optim as optim
import torch.nn as nn
from scipy.spatial.distance import cdist
from geopy.distance import geodesic
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def init_tb(log_path):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=log_path)
    print("Tensorboard:  tensorboard --logdir="+ log_path)
    return writer


def remove_file_with_timeout(file, timeout=5):
    """
    Attempts to remove a file with a timeout.
    
    Args:
        file (str): Path to the file to be removed.
        timeout (int): Maximum time to wait in seconds.
    
    Returns:
        bool: True if the file was removed successfully, False otherwise.
    """
    success = []

    def remove():
        try:
            os.remove(file)
            success.append(True)
        except Exception as e:
            logger.error(f"Failed to remove {file}: {e}")
            success.append(False)

    thread = threading.Thread(target=remove)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        logger.error(f"Timeout reached while trying to remove {file}")
        thread.join()  # Cleanup the thread
        return False

    return success[0] if success else False

def create_runtime_paths(args:dict):
    from definitions import ROOT_DIR

    base_dir = os.path.join(ROOT_DIR, "..", f"output")

    ### Create all the paths
    if args.model == "GWNET":
        output_dir = os.path.join(base_dir,
                    f"gwnet/days_{args.step_length}/features_{args.feature_days}")
    elif args.model == "CONVLSTM": 
        output_dir = os.path.join(base_dir,
                    f"convlstm/days_{args.step_length}/features_{args.feature_days}")
    elif args.model == "WNET":
        output_dir = os.path.join(base_dir,
                    f"wnet/days_{args.step_length}/features_{args.feature_days}")
    elif args.model == "DIME":
        specific_path = f"days_{args.step_length}/features_{args.feature_days}"
        if args.attention:
            model_subpath =  "dime_attn"
        else:
            model_subpath =  "dime"

        if args.conditioning != "all":
            output_dir = os.path.join(base_dir, model_subpath, f"{args.conditioning}", 
                                      specific_path)
        else:
            output_dir = os.path.join(base_dir, model_subpath, specific_path)
        
            
            
    elif args.model == "AUTO_DIME":
        output_dir = os.path.join(base_dir, f"dime/autoencoder")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    img_path = os.path.join(output_dir,  f"images")
    checkp_path = os.path.join(output_dir,  f"checkpoints")
    log_path = os.path.join(output_dir,  "logs")

    for sub_path in [img_path, checkp_path, log_path]:
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
    
    return output_dir, log_path, img_path, checkp_path

def load_autoencoder(checkpoint_path,feature_days=90, output_shape=20):
    from analysis.configs.config_models import config_convlstm_1 as model_config
    import torch
    from analysis import TimeEncoder, TimeDecoder, TimeAutoencoder 
    encoder = TimeEncoder(output_shape).to(model_config.device)
    decoder = TimeDecoder(feature_days, output_shape).to(model_config.device)
    autoencoder = TimeAutoencoder(encoder, decoder).to(model_config.device)
    checkpoint = torch.load(checkpoint_path)
    autoencoder.load_state_dict(checkpoint['state_dict'])
    logger.info(f"Loading autoencoder trained up to epoch {checkpoint['epoch']}")
    return autoencoder


def autoencoder_wrapper(args, 
    model_config, 
    data=None, 
    target=None, 
    generate_output:bool=True):
    
    from analysis import default, pipeline_autoencoder
    auto_epoch = None

    autoencoder_path = model_config.output_dir + \
        f"/dime/autoencoder/checkpoints" \
        "/checkpoint_epoch_{}.pth.tar"

    if generate_output is True:
        if args.auto_train is True:
            if args.auto_ep > 0 :
                checkp_path = autoencoder_path.format(args.auto_ep)
                auto_epoch = pipeline_autoencoder(args, 
                                           output_shape=args.auto_days//5,
                                           checkpoint_path=checkp_path)
            else:

                auto_epoch = pipeline_autoencoder(args, data, target,
                                           output_shape=args.auto_days//5)
        else:
            dest_path = model_config.data_dir + "/autoencoder_output"
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            if len(os.listdir(dest_path)) == 0 :
                for dataset in ["train", "test"]:
                    pipeline_autoencoder(args, data, target,
                                        output_shape=args.auto_days//5, 
                                        dataset=dataset)

    auto_epoch = default(auto_epoch, args.auto_ep)

    autoencoder = load_autoencoder(autoencoder_path.format(auto_epoch), 
                                   feature_days=args.auto_days,
                                   output_shape=args.auto_days//5,)
    return autoencoder

def mask_gnn_pixels(data, target, mask):
    
    sm_data = data[0, -4:]

    water_mask = np.ma.masked_where(mask==0, mask).mask
    null_mask = np.any(np.isnan(target), axis=0)
    null_sm_mask = np.any(np.isnan(sm_data), axis=0)

    combined_mask =  ~water_mask & ~null_mask & ~null_sm_mask #1 is good pixel, 0 no

    broadcast_data_combined = np.broadcast_to(combined_mask, data.shape)
    broadcast_target_combined = np.broadcast_to(combined_mask, target.shape)

    data = np.ma.masked_where(broadcast_data_combined==False, data)
    target = np.ma.masked_where(broadcast_target_combined==False, target)

    return data, target, combined_mask
    

def plot_first_n_images(tensor, n:int, save:bool=False, name:str=None, img_path=None):
    from utils.xarray_functions import ndvi_colormap
    from utils.function_clns import config
    cmap = ndvi_colormap("diverging")

    if img_path is None:
        img_path = os.path.join(config["DEFAULT"]["image_path"])
    if name is None:
        name = "temp_img.png"
    else:
        name = name + ".png"

    # Ensure n does not exceed the time dimension
    if torch.is_tensor(tensor):
        tensor = tensor.cpu().detach().numpy()

    n = min(n, tensor.shape[1])
    
    # Calculate the grid size for subplots
    rows = int(np.ceil(n / 5))  # 5 images per row as an example, you can adjust this
    cols = min(n, 5)  # up to 5 columns
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    
    # Flatten axes for easy iteration if rows and cols are more than 1
    if rows > 1:
        axes = axes.flatten()
    
    for i in range(n):
        ax = axes[i] if rows > 1 else axes[i % cols]
        ax.imshow(tensor[0, i, :, :], cmap=cmap)
        ax.axis('off')
        ax.set_title(f"Time {i}")
    
    # Hide any remaining subplots if n < rows * cols
    for i in range(n, rows * cols):
        fig.delaxes(axes[i] if rows > 1 else axes[i % cols])
    
    plt.tight_layout()
    if save is True:
        plt.savefig(os.path.join(img_path, name))
    else:
        plt.pause(10)
    plt.close()


def get_dataloader(args, dataset:xr.DataArray, 
                   ds:xr.DataArray, check_matrix:bool=False):
    
    from utils.function_clns import config

    x_df = dataset.to_dataframe()

    # if args.spi is False:
    #     x_df = x_df.swaplevel(1,2)
    # else:
    #     x_df = x_df.swaplevel(0,2)

    for col in ["spatial_ref","crs"]:
        if col in x_df:
            x_df.drop(columns={col}, inplace=True)
    var_target = dataset.name
    x_df = x_df.dropna(subset={var_target})
    x_df = x_df.sort_values(["lat", "lon","time"],ascending=False)

    data_x_unstack = x_df.unstack(["lat","lon"])
    print(data_x_unstack.isnull().sum())
    print(data_x_unstack.isnull().sum().sum())
    #x_unstack = data_x_unstack.to_numpy()
    num_samples, num_nodes = data_x_unstack.shape
    x_unstack = np.expand_dims(data_x_unstack, axis=-1)
    logger.info("The features have dimensions: {}", x_unstack.shape)

    y_df = ds.to_dataframe()
    for col in ["spatial_ref","crs"]:
        if col in y_df:
            y_df.drop(columns={col}, inplace=True)
    y_df = y_df.dropna(subset={"ndvi"})
    y_df = y_df.sort_values(["lat", "lon","time"],ascending=False)
    y_df = y_df.reset_index().set_index(["time","lon","lat"])
    y_df = y_df[y_df.index.isin(x_df.index)]

    data_y_unstack = y_df.unstack(["lat","lon"])
    print(data_y_unstack.isnull().sum())
    print(data_y_unstack.isnull().sum().sum())
    y_unstack = data_y_unstack.to_numpy()
    y_unstack = np.expand_dims(y_unstack, axis=-1)

    logger.info("The instance have dimensions: {}", y_unstack.shape)
    st_df = x_df.reset_index()[["lon","lat"]].drop_duplicates()

    dest_path = os.path.join(os.path.join(args.output_dir,  "adjacency_matrix"), \
        f"{args.precp_product}_{args.dim}_adj_dist.pkl")

    if os.path.isfile(dest_path) and check_matrix is True:
        logger.info("Using previously created adjacency matrix")
    else:
        adj_dist = generate_adj_dist(st_df)
        with open(dest_path, 'wb') as f:
            pickle.dump(adj_dist, f, protocol=2)
        logger.info(f"Created new adjacency matrix {config['GWNET']['precp_product']}_{config['GWNET']['dim']}")

    seq_length_x = seq_length_y = args.forecast
    y_start = 1

    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(y_start, (seq_length_y + 1), 1))

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(x_unstack[t + x_offsets, ...])
        y.append(y_unstack[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    logger.info("x shape: {}", x.shape) 
    logger.info("y shape: {}", y.shape)

    num_test = round(num_samples * config["GWNET"]["test_split"])
    num_train = round(num_samples * config["GWNET"]["train_split"])
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        logger.info("{cat} x: {x_shape}, y: {y_shape}".format(cat = cat, x_shape=_x.shape, y_shape=_y.shape))
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

    batch_size = config["GWNET"]["batch_size"]
    dataloader = load_dataset(args.output_dir, batch_size, batch_size, batch_size)
    return dataloader, num_nodes, data_y_unstack.iloc[num_train: num_test+num_train,:]


def check_shape_dataloaders(train_dataloader, val_dataloader):
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs = inputs.float()
        targets = targets.float()
        logger.info("Logging data info for train dataloader: "
            "Input shape: %s, Target shape: %s, "
            "Input max: %s, Input min: %s, "
            "Target max: %s, Target min: %s", 
            inputs.shape, targets.shape, 
            inputs.max().item(), inputs.min().item(),
            targets.max().item(), targets.min().item())
    
    
    for batch_idx, (inputs, targets) in enumerate(val_dataloader):
        inputs = inputs.float()
        targets = targets.float()
        logger.info("Logging data info for validation dataloader: "
            "Input shape: %s, Target shape: %s, "
            "Input max: %s, Input min: %s, "
            "Target max: %s, Target min: %s", 
            inputs.shape, targets.shape, 
            inputs.max().item(), inputs.min().item(),
            targets.max().item(), targets.min().item())


def date_to_sinusoidal_embedding(date_string, h, w):
    from datetime import datetime
    # Parsing the date string
    date_time_obj = datetime.strptime(date_string, '%Y-%m-%d')
    
    # Extracting components
    month = (date_time_obj.month - 1) / 11
    day = (date_time_obj.day - 1) / 30
    
    # Creating sinusoidal embeddings
    embedding = []
    for value in [month, day]:
        embedding.append(np.sin(2 * np.pi * value))
        embedding.append(np.cos(2 * np.pi * value))
        
    result = np.array(embedding)
    
    # Reshape the result to be compatible with tiling
    result = result.reshape((1, 4))
    
    # Calculate the number of times to tile to get the final shape of (83, 77)
    num_tiles = (h, w // 4 + (w % 4 != 0))  # Add one more tile if there's a remainder

    tiled_result = np.tile(result, num_tiles)
    
    # Trim the result to the exact shape (83, 77)
    final_result = tiled_result[:, :w]
    
    return final_result

def prepare_array_wgcnet(data, mask=None):

    # Add one channel dimension if it's missing
    if len(data.shape) == 3:
        logger.info("Adding one channel dimension")
        data = np.expand_dims(data, 1)
    
    T, C, W, H = data.shape  # Original dimensions

    # Reshape the array to combine the lat and lon dimensions into one (T, C, V)
    data_reshaped = data.reshape(T, C, W * H)
    
    if mask is not None:
        # Create a mask to identify non-null images
        mask = np.all(np.isnan(data_reshaped), axis=(0,1))  # Shape will be (T, C)

        # Apply the mask to filter out null images
        non_null_data = data_reshaped[:, :, ~mask]
        # Reshape mask back to (T, C, W, H)
        data_reshaped = non_null_data.reshape(T, C, non_null_data.shape[-1])

    return data_reshaped


def reverse_reshape_gnn(original_data, pred_data, mask):
    
    T = pred_data.shape[0]

    if len(original_data.shape) == 3:
        _, W, H = original_data.shape  # Original dimensions
        C = 1
        
    elif len(original_data.shape) == 4:
        _, C, W, H = original_data.shape 
    # Create an empty array with the original shape filled with NaNs
    reconstructed_data = np.full((T, C, W * H), np.nan)

    if len(reconstructed_data.shape) != len(pred_data.shape):
        pred_data = np.squeeze(pred_data)
        reconstructed_data = np.squeeze(reconstructed_data)
    mask = np.reshape(mask, W*H)

    # Reinsert the non-null data into the non-masked locations
    reconstructed_data[..., mask] = pred_data

    reconstructed_data = np.reshape(reconstructed_data, (T, C, W, H))
    return np.squeeze(reconstructed_data)

def print_lr_change(old_lr, scheduler):
    lr = scheduler.get_last_lr()[0]
    if lr != old_lr:
        logger.info(f"Learning rate changed: {old_lr} --> {lr}")
    return lr


def pipeline_hydro_vars(args:dict,
                        model_config,
                        rawdata_name:str="data_convlstm",
                        use_water_mask:bool = True,
                        precipitation_only: bool = True,
                        load_zarr_features:bool = False,
                        load_local_precipitation:bool=True,
                        interpolate:bool =False,
                        checkpoint_path:str=None):
    
    from ancillary.esa_landuse import drop_water_bodies_copernicus
    from precipitation.preprocessing.preprocess_data import PrecipDataPreparation
    from utils.function_clns import config, interpolate_prepare, init_logging
    import numpy as np
    import pickle
    from definitions import ROOT_DIR

    def save_prepare_data(HydroData, save:bool=False):
        ndvi_scaler = HydroData.ndvi_scaler
        if use_water_mask is True:
            logger.info("Loading water bodies mask...")
            waterMask = drop_water_bodies_copernicus(HydroData.ndvi_ds).isel(time=0)
            # HydroData.hydro_data = HydroData.hydro_data.where(HydroData.ndvi_ds.notnull())
            mask = torch.tensor(np.array(xr.where(waterMask.notnull(), 1, 0)))
        else:
            mask = None

        logger.info("Preparating datasets as numpy arrays...")
        data, target = interpolate_prepare(
            HydroData.precp_ds, 
            HydroData.ndvi_ds, 
            interpolate=False,
            convert_to_float=False,
        )
        if save is True:
            for d, name in zip([data, target, mask],  ['data', 'target', 'mask']):
                np.save(os.path.join(data_dir, f"{name}.npy"), d)

            logger.debug("Saving NDVI scaler to file...")
            dump_obj = pickle.dumps(HydroData.ndvi_scaler, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(data_dir,"ndvi_scaler.pickle"), "wb") as handle:
                handle.write(dump_obj)
        return data, target, ndvi_scaler, mask

    
    _, log_path, _, _ = create_runtime_paths(args)

    logger = init_logging(log_file=os.path.join(log_path, 
                        f"pipeline_{(args.model).lower()}_"
                        f"features_{args.feature_days}.log"), verbose=False)
    
    data_dir = os.path.join(model_config.data_dir, rawdata_name)

    logger.info(f"Starting {args.mode} {args.model} model for {args.step_length}" \
                f"days in the future" \
                f" with {args.feature_days} days of features")

    if os.path.exists(data_dir) is False:
        logger.info("Created new path for data")
        os.makedirs(data_dir)

    if (len(os.listdir(data_dir)) == 0) & (load_zarr_features is False):
        #### Case 1: data existent but explicitly no load zarr
        logger.info(f"No numpy raw data found, proceeding with the creation"
                    f" of the training dataset.")

        if precipitation_only is False:
            import warnings
            warnings.filterwarnings('ignore')
            Era5variables = ["potential_evaporation", "evaporation",
                         "2m_temperature","total_precipitation"]
            HydroData = PrecipDataPreparation(
                args,
                variables=Era5variables,
                precipitation_data="ERA5",
                load_local_precp = load_local_precipitation,
                interpolate = interpolate
            )
        else:
            Era5variables = ["total_precipitation"]
            HydroData = PrecipDataPreparation(
                args, 
                precp_dataset=config["CONVLSTM"]['precp_product'],
                variables=Era5variables,
                load_local_precp=load_local_precipitation,
                interpolate = interpolate
            )
        data, target, ndvi_scaler, mask = save_prepare_data(HydroData, save=True)

    elif load_zarr_features is True:
        #### Case 2: explicitly load zarr with filled data
        HydroData = PrecipDataPreparation(
                args,
                variables=None,
                precipitation_data="ERA5",
                load_local_precp = False,
                load_zarr_features=load_zarr_features,
                interpolate = args.interpolate
            )
        data, target, ndvi_scaler, mask = save_prepare_data(HydroData, save=True)

    else:
        #### Case 3: explicitly no load zarr and data identified
        logger.info("Training data found. Proceeding with loading...")
        data = np.load(os.path.join(data_dir, "data.npy"))
        target = np.load(os.path.join(data_dir, "target.npy"))
        mask = torch.tensor(np.load(os.path.join(data_dir, "mask.npy")))
        with open(os.path.join(data_dir,"ndvi_scaler.pickle"), "rb") as handle:
            ndvi_scaler = pickle.loads(handle.read())

    return data, target, mask, ndvi_scaler

def train_loop(config, 
               args, 
               model, 
               train_loader, 
               criterion, 
               optimizer, 
               scaler=None, 
               mask=None, 
               draw_scatter:bool=False):

    _, _, img_path, _ = create_runtime_paths(args)
    
    #from torcheval.metrics import R2Score
    model.train()
    epoch_records = {'loss': [], "mape":[], "rmse":[], "lr":[]}
    #metric = R2Score()

    if draw_scatter is True:
        nbins= 200
        bin0 = np.linspace(0, config.max_value, nbins+1)
        n = np.zeros((nbins,nbins))

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs = inputs.float().to(config.device)
        # logger.info("Feature tensor GPU memory: {:.4f} GB".format(get_tensor_memory(inputs)/ (1024**3)))
        targets = torch.squeeze(targets.float().to(config.device))
        # logger.info("Label tensor GPU memory: {:.4f} GB".format(get_tensor_memory(targets)/ (1024**3)))
        # logger.info(get_ram())
        # logger.info(get_vram())
        outputs = torch.squeeze(model(inputs))

        if draw_scatter is True:
            img_pred = outputs.detach().cpu().numpy()
            img_real = targets.detach().cpu().numpy()
            if args.normalize is True:
                img_pred = scaler.inverse_transform(img_pred)
                img_real = scaler.inverse_transform(img_real)
            h, xed, yed = evaluate_hist2d(img_real, img_pred, nbins)
            n = n+h

        if config.masked_loss is False:
            losses = criterion(outputs, targets)
            mape = masked_mape(outputs,targets).item()
            rmse = masked_rmse(outputs,targets).item()
        else:
            if mask is None:
                raise ValueError("Please provide a mask for loss computation")
            else:
                mask = mask.float().to(config.device)
                losses =  masked_custom_loss(criterion, outputs, targets, mask)
                mape = mask_mape(outputs,targets, mask).item()
                rmse = mask_rmse(outputs,targets, mask).item()
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_records['loss'].append(losses.item())
        epoch_records["rmse"].append(rmse)
        epoch_records["mape"].append(mape)
        

    if draw_scatter is True:
        plot_scatter_hist(n,  bin0, img_path)

    return epoch_records


def valid_loop(config, args, model, valid_loader, criterion, scheduler,
               scaler=None, mask=None, draw_scatter:bool=False):
    
    _, _, img_path, _ = create_runtime_paths(args)

    model.eval()
    epoch_records = {'loss': [], "mape":[], "rmse":[], "lr":[]}

    if draw_scatter is True:
        nbins= 200
        bin0 = np.linspace(0, config.max_value, nbins+1)
        n = np.zeros((nbins,nbins))

    num_batchs = len(valid_loader)
    for batch_idx, (inputs, targets) in enumerate(valid_loader):
        with torch.no_grad():
            inputs = inputs.squeeze(0).float().to(config.device)
            # logger.info("Feature tensor GPU memory: {:.2f} GB".format(get_tensor_memory(inputs)/ (1024**3)))
            targets = torch.squeeze(targets.float().to(config.device))
            # logger.info("Label tensor GPU memory: {:.2f} GB".format(get_tensor_memory(targets)/ (1024**3)))
            outputs = torch.squeeze(model(inputs))

            if draw_scatter is True:
                img_pred = outputs.cpu().detach().numpy().flatten()
                img_real = targets.cpu().detach().numpy().flatten()
                if args.normalize is True:
                    img_pred = scaler.inverse_transform(img_pred)
                    img_real = scaler.inverse_transform(img_real)
                h, xed, yed = evaluate_hist2d(img_real, img_pred, nbins)
                n = n+h

            if config.masked_loss is False:
                losses = criterion(outputs, targets)
                mape = masked_mape(outputs,targets).item()
                rmse = masked_rmse(outputs,targets).item()
            else:
                if mask is None:
                    raise ValueError("Please provide a mask for loss computation")
                else:
                    mask = mask.float().to(config.device)
                    losses =  masked_custom_loss(criterion, outputs, targets, mask)
                    mape = mask_mape(outputs,targets, mask).item()
                    rmse = mask_rmse(outputs,targets, mask).item()


            epoch_records['loss'].append(losses.item())
            epoch_records["rmse"].append(rmse)
            epoch_records["mape"].append(mape)

    ##### Step learning rate
    mean_loss = sum(epoch_records['loss']) / len(epoch_records['loss'])
    scheduler.step(mean_loss)
    learning_rate = scheduler.get_last_lr()[0]
    epoch_records["lr"].append(learning_rate)
            
    if draw_scatter is True:
        plot_scatter_hist(n,  bin0, img_path)
    
    return epoch_records


def test_loop(config, 
    args, 
    model, 
    valid_loader, 
    criterion,
    scaler=None, 
    mask=None, 
    draw_scatter:bool=False):
    
    from tqdm.auto import tqdm
    
    _, _, img_path, _ = create_runtime_paths(args)

    model.eval()
    epoch_records = {'loss': [], "mape":[], "rmse":[], "lr":[]}
    predictions, outputs_real = [], []

    if draw_scatter is True:
        nbins= 200
        bin0 = np.linspace(0, config.max_value, nbins+1)
        n = np.zeros((nbins,nbins))

    num_batchs = len(valid_loader)
    for batch_idx, (inputs, targets) in enumerate(tqdm(valid_loader, desc="Testing", unit="batch")):
        with torch.no_grad():
            inputs = inputs.squeeze(0).float().to(config.device)
            # logger.info("Feature tensor GPU memory: {:.2f} GB".format(get_tensor_memory(inputs)/ (1024**3)))
            targets = torch.squeeze(targets.float().to(config.device))
            # logger.info("Label tensor GPU memory: {:.2f} GB".format(get_tensor_memory(targets)/ (1024**3)))
            outputs = torch.squeeze(model(inputs))

            if len(outputs.shape) == 2:
                outputs = outputs.unsqueeze(0)
            
            if len(targets.shape) == 2:
                targets = targets.unsqueeze(0)

            if args.normalize is True:  
                outputs = scaler.inverse_transform(outputs)
                targets = scaler.inverse_transform(targets)

            if draw_scatter is True:
                img_pred = outputs.cpu().detach().numpy().flatten()
                img_real = targets.cpu().detach().numpy().flatten()
                h, xed, yed = evaluate_hist2d(img_real, img_pred, nbins)
                n = n+h

            if config.masked_loss is False:
                losses = criterion(outputs, targets).item()
                mape = masked_mape(outputs,targets).item()
                rmse = masked_rmse(outputs,targets).item()
            else:
                if mask is None:
                    raise ValueError("Please provide a mask for loss computation")
                else:
                    mask = mask.float().to(config.device)
                    losses = masked_custom_loss(criterion, outputs, targets, mask).item()
                    mape = mask_mape(outputs,targets, mask).item()
                    rmse = mask_rmse(outputs,targets, mask).item()


            epoch_records['loss'].append(losses)
            epoch_records["rmse"].append(rmse)
            epoch_records["mape"].append(mape)
            
            outputs_real.append(outputs)
            predictions.append(targets)

    if draw_scatter is True:
        plot_scatter_hist(n,  bin0, img_path)

    predictions = torch.cat(predictions, 0)
    outputs_real = torch.cat(outputs_real, 0)
    
    return epoch_records, predictions, outputs_real


def evaluate_hist2d(real_img, pred_img, nbins):
    mdata=np.isnan(real_img)==0
    h, xed,yed=np.histogram2d(real_img[mdata], 
                pred_img[mdata], bins=nbins, density=None, weights=None)
    
    return h, xed, yed
    
def plot_scatter_hist(n, bin0, path=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import os
    n[n<=0]=np.nan
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    #ax = axes.ravel()
    #a0=ax[0].imshow(np.log10(n),origin='lower')
    a0=ax.pcolor(bin0,bin0,n,norm=LogNorm(vmin=1, vmax=np.nanmax(n)))
    plt.colorbar(a0)
    ax.plot(bin0, bin0, '--', color='black', label='y=x')
    if path is not None:
        name = "scatterplot.png"
        plt.savefig(os.path.join(path,name))
    plt.show(block=False)
    plt.pause(2)
    plt.close()

def persistence_baseline(train_label):
    batch_rmse = []
    for idx in range(train_label.shape[1]- 1):
        target = train_label[:, idx+1, :, :]
        pred = train_label[:, idx, :, :]
        rmse = np.sqrt(np.mean((pred - target)**2))
        batch_rmse.append(rmse)
    logger.info("The baseline is a RMSE of {}".format(np.mean(batch_rmse)))


def compute_metric_results(results:dict, 
                            metric_list:list, 
                            mask:np.ndarray, 
                            days:list= [10, 15, 30]):

    metrics_results = {}
    # Iterate through each model
    for model_name, model_data in results.items():
        metrics_results[model_name] = {}
        # Iterate through each prediction horizon
        for horizon in days:
            y_key = f"y_{horizon}"
            y_pred_key = f"y_pred_{horizon}"
            # mask_key = f"mask_{horizon}"

            # Extract the relevant arrays
            y_true = model_data[y_key]
            y_pred = model_data[y_pred_key]
            # mask = model_data.get(mask_key, None)  # Optional if no mask

            # Calculate SSIM
            ssim = tensor_ssim(y_true, y_pred)

            # Calculate RMSE using CustomMetrics
            metric_list = ["rmse"]
            custom_metrics = CustomMetrics(y_pred, y_true, metric_list, mask, True)
            rmse = custom_metrics.losses[0]

            # Store the metrics
            metrics_results[model_name][horizon] = {
                "rmse": rmse,
                "ssim": ssim
            }
    return metrics_results

def get_train_valid_loader(model_config, 
                           args, 
                           data, 
                           labels,
                           mask, 
                           train_split:float=0.7, 
                           add_lag:bool=True):
    from scipy.io import loadmat
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    from analysis import CustomConvLSTMDataset
    from utils.function_clns import CNN_split

    data = prepare_array_wgcnet(data, mask)
    labels = prepare_array_wgcnet(labels, mask)
    

    train_data, val_data, train_label, val_label, \
        test_valid, test_label = CNN_split(data, 
                                           labels, 
                                           split_percentage=train_split,
                                        #    val_split=0.5)
                                           val_split=0.333)
    
    train_dataset = CustomConvLSTMDataset(model_config, args, 
                                          train_data, train_label, 
                                          save_files=False)
    logger.info("Generated train dataloader")
    
    val_dataset = CustomConvLSTMDataset(model_config, args, 
                                        val_data, val_label, 
                                        save_files=False)
    
    logger.info("Generated validation dataloader")
    
    test_dataset = CustomConvLSTMDataset(model_config, args, 
                                        test_valid, test_label, 
                                        save_files=False)
    
    logger.info("Generated test dataloader")
    
    # create a DataLoader object that uses the dataset
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=model_config.batch_size, 
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=model_config.batch_size, 
                                shuffle=False)
    
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=model_config.batch_size, 
                                shuffle=False)
    
    

    return train_dataloader, val_dataloader, test_dataloader, train_dataset



class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data, null_value=None):
        transf = (data * self.std) + self.mean
        if null_value is not None:
            return transf.where(data != null_value, null_value)
        else:
            return transf
    

class StandardNormalizer():
    """
    Standard the input
    """
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        norm_data = (data - self.min) / (self.max - self.min)
        return norm_data

    def inverse_transform(self, norm_data, null_value=None):
        transf = (norm_data * (self.max - self.min)) + self.min 
        if null_value is not None:
            return transf.where(norm_data != null_value, null_value)
        else:
            return transf

def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=np.nanmean(data['x_train'][..., 0]), std=np.nanstd(data['x_train'][..., 0]))
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    adj_mx = load_pickle(pkl_filename)
    logger.info("Loading adjacency matrix...")
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj

"""
Metrics with null values
"""

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

from ignite.engine import Engine
from ignite.metrics import SSIM

def tensor_ssim(preds, labels, range=1.0):

    def process_function(engine, batch):
        if isinstance(batch[0], np.ndarray):
            x = torch.from_numpy(batch[0])
        else:
            x = batch[0]

        if isinstance(batch[1], np.ndarray):
            y = torch.from_numpy(batch[1])
        else:
            y = batch[1]
        
        if len(x.shape)==3:
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)

        return {'y_pred': x, 'y_true': y}

    def output_transform(output):
        # `output` variable is returned by above `process_function`
        y_pred = output['y_pred']
        y = output['y_true']
        return y_pred, y

    engine = Engine(process_function)

    metric = SSIM(output_transform=output_transform, data_range=range)
    metric.attach(engine, 'ssim')

    state = engine.run([[preds, labels]])
    return state.metrics['ssim']

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds, labels, null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    # normalize the mask by the mean
    mask /=  torch.mean(mask)
    # Replace any NaNs in the mask with zeros
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # Calculate the percentage error (with small epsilon to avoid division by zero)
    epsilon = 1e-10
    loss = torch.abs(preds - labels) / (labels + epsilon)
    # Apply the mask to the loss
    loss = loss * mask
    # Replace any NaNs in the loss with zeros
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # Return the mean of the masked loss
    return torch.mean(loss)


def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

def corrcoef(x):
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

def tensor_corr(prediction, label):
    x = label.squeeze().reshape(-1)
    y = prediction.squeeze().reshape(-1)
    return pearsonr(x, y)

"""
Metrics with custom mask
"""

def mask_mae(preds, labels, mask, return_value=True):
    loss = torch.abs(preds-labels)
    if isinstance(mask, torch.Tensor):
        full_mask = torch.broadcast_to(mask, loss.shape)
        null_loss = (loss * full_mask)
        null_loss = torch.where(torch.isnan(mask), torch.tensor(0.0), null_loss)
    elif isinstance(mask, np.ndarray):
        full_mask = np.broadcast_to(mask, loss.shape)
        null_loss = (loss * full_mask)
        null_loss = np.where(np.isnan(mask), 0, null_loss)
    elif not mask:
        null_loss = loss
        full_mask = torch.ones(loss.shape)

    if return_value:
        non_zero_elements = full_mask.sum()
        return null_loss.sum() / non_zero_elements
    else:
        if len(loss.shape)>2:
            return loss.mean(0)
        else:
            return loss

def mask_rmse(preds, labels, mask=None):
    mse = mask_mse(preds=preds, labels=labels, mask=mask)
    return torch.sqrt(mse)

from typing import Union

class CustomMetrics():
    def __init__(self,
                 preds:Union[torch.tensor, np.ndarray],
                 labels:Union[torch.tensor, np.ndarray], 
                 metric_lists:Union[list, str],
                 mask = Union[None, torch.tensor, np.ndarray],
                 masked:bool=False):
        
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        if masked:
            self._count_masked_pixels(mask)

        self.mask = mask
        if masked and mask is None:
            logger.error(RuntimeError("No provided mask but chosen masked loss option"))

        self.losses, self.metrics = self._get_metrics(metric_lists, preds, labels, masked)

    def _get_metrics(self, metric_list, preds, labels, masked):
        if isinstance(metric_list, list):
            results = []
            metrics = []
            for metric in metric_list:
                res, m = self._apply_metric(metric, preds, labels)
                if masked:
                    res = self._metric_masking(res, self.mask)
                results.append(res)
                metrics.append(m)
            return results, metrics
        
        elif isinstance(metric_list, str):
            results, m = self._apply_metric(metric_list, preds, labels)
            if masked:
                results = self._metric_masking(results, self.mask)
            return [results], [m]
    
    def _apply_metric(self, metric, preds, labels):
        if metric == "rmse":
            loss = torch.sqrt((preds-labels)**2)
        elif metric == "bias":
            loss = labels - preds
        
        elif metric == "mse":
            loss = (preds-labels)**2

        elif metric == "mae":
            loss = torch.abs(preds-labels)

        elif metric == "mape":
            loss = torch.abs((preds-labels)/labels)
        else:
            logger.warning(f"Metric {metric} not recognized")
            loss = None

        return loss, metric
    
    def _count_masked_pixels(self, mask):
        w, h = mask.shape
        good_pixels = mask.sum()
        tot_pixels = w*h
        logger.info(f"{(1-good_pixels/tot_pixels):.2%} of the pixels are masked in the loss computation")
        

    def _metric_masking(self, loss, mask, return_value=True):
        if isinstance(loss, np.ndarray):
            loss = torch.from_numpy(loss)

        full_mask = torch.broadcast_to(mask, loss.shape)
        null_loss = (loss * full_mask)
        null_loss = torch.where(torch.isnan(null_loss), torch.tensor(0.0), null_loss)

        if return_value:
            non_zero_elements = full_mask.sum()
            return (null_loss.sum() / non_zero_elements).item()
        else:
            if len(loss.shape)>2:
                return loss.mean(0)
            else:
                return loss

def mask_mape(preds, labels, mask, return_value=True):
    loss = torch.abs((preds-labels)/labels)
    if isinstance(mask, torch.Tensor):
        full_mask = torch.broadcast_to(mask, loss.shape)
        null_loss = (loss * full_mask)
        null_loss = torch.where(torch.isnan(mask), torch.tensor(0.0), null_loss)
    elif isinstance(mask, np.ndarray):
        full_mask = np.broadcast_to(mask, loss.shape)
        null_loss = (loss * full_mask)
        null_loss = np.where(np.isnan(mask), 0, null_loss)
    elif not mask:
        full_mask = torch.ones(loss.shape)

    if return_value:
        non_zero_elements = full_mask.sum()
        return null_loss.sum() / non_zero_elements
    else:
        if len(loss.shape)>2:
            return loss.mean(0)
        else:
            return loss

def mask_mse(preds, labels, mask, return_value=True):
    loss = (preds-labels)**2
    if isinstance(mask, torch.Tensor):
        full_mask = torch.broadcast_to(mask, loss.shape)
        null_loss = (loss * full_mask)
        null_loss = torch.where(torch.isnan(mask), torch.tensor(0.0), null_loss)
    elif isinstance(mask, np.ndarray):
        full_mask = np.broadcast_to(mask, loss.shape)
        null_loss = (loss * full_mask)
        null_loss = np.where(np.isnan(mask), 0, null_loss)
    elif not mask:
        null_loss = loss
        full_mask = torch.ones(loss.shape)

    if return_value:
        non_zero_elements = full_mask.sum()
        return null_loss.sum() / non_zero_elements
    else:
        if len(loss.shape)>2:
            return loss.mean(0)
        else:
            return loss
            

def masked_custom_loss(criterion, preds, labels, mask=None, return_value=True):
    loss = criterion(preds, labels)
    if isinstance(mask, torch.Tensor):
        full_mask = torch.broadcast_to(mask, loss.shape)
        null_loss = (loss * full_mask)
        null_loss = torch.where(torch.isnan(mask), torch.tensor(0.0), null_loss)
    elif isinstance(mask, np.ndarray):
        full_mask = np.broadcast_to(mask, loss.shape)
        null_loss = (loss * full_mask)
        null_loss = np.where(np.isnan(mask), 0, null_loss)
    elif not mask:
        null_loss = loss
        full_mask = torch.ones(loss.shape)

    if return_value:
        non_zero_elements = full_mask.sum()
        return null_loss.sum() / non_zero_elements
    else:
        if len(loss.shape)>2:
            return loss.mean(0)
        else:
            return loss
        
def mask_mbe(preds, labels, mask, return_value=True):
    loss = (preds-labels)
    if isinstance(mask, torch.Tensor):
        full_mask = torch.broadcast_to(mask, loss.shape)
    elif isinstance(mask, np.ndarray):
        full_mask = np.broadcast_to(mask, loss.shape)
    null_loss = (loss * full_mask)
    if return_value:
        non_zero_elements = full_mask.sum()
        return null_loss.sum() / non_zero_elements
    else:
        if len(loss.shape)>2:
            return loss.mean(0)
        else:
            return loss

def save_figures(
                 epoch:int, 
                 path, 
                 metrics_recorder:dict):

    epochs = list(range(1, epoch + 1))

    plt.figure()
    plt.plot(epochs, metrics_recorder.train_mape, label='train MAPE')
    plt.plot(epochs, metrics_recorder.val_mape, label='validation MAPE')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE')
    plt.legend()
    plt.title('Mean Absolute Percentage Error (MAPE) vs. Epoch')
    plt.savefig(os.path.join(path, 'mape_vs_epoch.png'))
    plt.close()

    # Plot RMSE
    plt.figure()
    plt.plot(epochs, metrics_recorder.train_rmse, label='train RMSE')
    plt.plot(epochs, metrics_recorder.val_rmse,  label='validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('Root Mean Squared Error (RMSE) vs. Epoch')
    plt.savefig(os.path.join(path, 'rmse_vs_epoch.png'))
    plt.close()

    # Plot Loss
    plt.figure()
    plt.plot(epochs, metrics_recorder.train_loss , label='train Loss')
    plt.plot(epochs,metrics_recorder.val_loss  , label='validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epoch')
    plt.savefig(os.path.join(path, 'loss_vs_epoch.png'))
    plt.close()
    # logger.info(f"Figures saved for {args.precp_product} {args.forecast} days forecast")
    

class MetricsRecorder:
    def __init__(self):
        self.train_mape = []
        self.train_rmse = []
        self.train_loss = []
        self.val_mape = []
        self.val_rmse = []
        self.val_loss = []
        self.lr = []
        self.epoch = []

    def add_train_metrics(self, metrics, epoch):
        self.train_mape.append(np.mean(metrics['mape']))
        self.train_rmse.append(np.mean(metrics["rmse"]))
        self.train_loss.append(np.mean(metrics["loss"]))
        self.epoch.append(epoch)

    def add_val_metrics(self, metrics):
        self.val_mape.append(np.mean(metrics['mape']))
        self.val_rmse.append(np.mean(metrics["rmse"]))
        self.val_loss.append(np.mean(metrics["loss"]))
        self.lr.append(metrics["lr"][-1])

def update_tensorboard_scalars(writer, recorder:MetricsRecorder):
    writer.add_scalars('loss', {'train': recorder.train_loss[-1]}, recorder.epoch[-1])
    #writer.add_scalars('LossTR', {'trainD': train.lossd[-1]}, train.epoch[-1])
    writer.add_scalars('loss', {'valid': recorder.val_loss[-1]}, recorder.epoch[-1])
    #writer.add_scalars('LossVAL', {'validD': valid.lossd[-1]}, valid.epoch[-1])
    
    writer.add_scalars('rmse', {'train': recorder.train_rmse[-1]}, recorder.epoch[-1])
    writer.add_scalars('rmse', {'valid': recorder.val_rmse[-1]}, recorder.epoch[-1])
    writer.add_scalars('mape', {'train': recorder.train_mape[-1]}, recorder.epoch[-1])
    writer.add_scalars('mape', {'valid': recorder.val_mape[-1]}, recorder.epoch[-1])
    writer.add_scalars('lr', {'learning rate': recorder.lr[-1]}, recorder.epoch[-1])


class GWNETtrainer():
    def __init__(self, 
                 args, 
                 model_config, 
                 scaler, 
                 loss,
                 num_nodes:int, 
                 supports,
                 adjinit, 
                 checkpoint_path=None):
        
        from analysis import gwnet
        from torch.nn import DataParallel

        self.device = model_config.device
        self.model = gwnet(self.device, 
            num_nodes, 
            model_config.dropout, 
            supports=supports, 
            gcn_bool=args.gcn_bool, 
            addaptadj=args.addaptadj, 
            aptinit=adjinit, 
            in_dim=model_config.in_dim, 
            out_dim=model_config.out_dim, 
            residual_channels=model_config.nhid, 
            dilation_channels=model_config.nhid,
            skip_channels=model_config.nhid * 8,
            end_channels=model_config.nhid * 16,
            blocks=model_config.blocks,
            layers=model_config.layers
        )

        self.model = DataParallel(self.model)
        
        self.learning_rate = model_config.learning_rate
        self.model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                    lr=self.learning_rate,
                                    weight_decay=model_config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            factor=model_config.scheduler_factor, 
            patience=model_config.scheduler_patience
        )
        self.loss = loss
        self.scaler = scaler
        self.clip = 5

    
        if checkpoint_path is not None:
            try:
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['lr_sched'])
                self.checkp_epoch = checkpoint['epoch']
                logger.info(f"Resuming training from epoch {self.checkp_epoch}")

            except FileNotFoundError:
                from analysis import load_checkp_metadata
                checkpoint_path = checkpoint_path.removesuffix(".pth.tar")
                self.model, self.optimizer, self.scheduler, self.checkp_epoch, _ = load_checkp_metadata(checkpoint_path, self.model, self.optimizer, self.scheduler, None)   

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        real = torch.unsqueeze(real_val,dim=1)
        # predict = self.scaler.inverse_transform(output)

        loss = self.loss(output, real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        # Check gradients for NaN or Infinity
        #for name, param in self.model.named_parameters():
        #    if param.grad is not None:
        #        print("Gradient Stats:", name, param.grad.mean().item(), param.grad.max().item(), param.grad.min().item())
        mape = masked_mape(output,real).item()
        rmse = masked_rmse(output,real).item()

        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
       
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        real = torch.unsqueeze(real_val,dim=1)
        # predict = self.scaler.inverse_transform(output)
        loss = self.loss(output, real)
        mape = masked_mape(output,real).item()
        rmse = masked_rmse(output,real).item()
        return loss.item(), mape, rmse
    
    def test(self, 
            args, 
            input, 
            real_val, 
            scaler, 
            all_metrics=True, 
            n:int=None,
            draw_scatter:bool=False
        ):

        self.model.eval()
        with torch.no_grad():
            input = nn.functional.pad(input, (1, 0, 0, 0))
            output = self.model(input)
            real = torch.unsqueeze(real_val, dim=1)

            if args.normalize:
                output_scaled = scaler.inverse_transform(output)
                real_scaled = scaler.inverse_transform(real)

            if args.scatterplot is True:
                img_pred = output_scaled.cpu().detach().numpy().flatten()
                img_real = real_scaled.cpu().detach().numpy().flatten()
                h, xed, yed = evaluate_hist2d(img_real, img_pred, nbins= 200)
                n = n+h

            loss = self.loss(output_scaled, real_scaled)
            if all_metrics is True:
                mape = masked_mape(output_scaled,real_scaled).item()
                rmse = masked_rmse(output_scaled,real_scaled).item()

            return loss.item(), mape, rmse, output_scaled, real_scaled, n
        
    def schedule_learning_rate(self, mean_loss):
        self.scheduler.step(mean_loss)
        new_lr = print_lr_change(self.learning_rate, self.scheduler)
        self.learning_rate = new_lr

    def metric(self, pred, real):
        mae = masked_mae(pred,real, 0.0).item()
        mape = masked_mape(pred,real, 0.0).item()
        rmse = masked_rmse(pred,real, 0.0).item()
        return mae,mape,rmse

    def gwnet_train_loop(self, model_config, engine, train_dl):
        epoch_records = {'loss': [], "mape":[], "rmse":[], "lr":[]}
        for iter, (x, y) in enumerate(train_dl):
            trainx = torch.Tensor(x).to(model_config.device)
            trainx= trainx.transpose(3, 2)
            trainy = torch.Tensor(y).to(model_config.device)
            trainy = trainy.transpose(3, 2)
            loss, mape, rmse = self.train(trainx, trainy[:, 0,:,:])

            epoch_records["loss"].append(loss)
            epoch_records["rmse"].append(rmse)
            epoch_records["mape"].append(mape)
        return epoch_records

    def gwnet_val_loop(self, model_config, engine, val_dl):
        epoch_records = {'loss': [], "mape":[], "rmse":[], "lr":[]}
        for iter, (x, y) in enumerate(val_dl):
            trainx = torch.Tensor(x).to(model_config.device)
            trainx= trainx.transpose(2, 3)
            trainy = torch.Tensor(y).to(model_config.device)
            trainy = trainy.transpose(2, 3)
            loss, rmse, mape = self.eval(trainx, trainy[:, 0,:,:])

            epoch_records["loss"].append(loss)
            epoch_records["rmse"].append(rmse)
            epoch_records["mape"].append(mape)

        ##### Step learning rate
        mean_loss = sum(epoch_records['loss']) / len(epoch_records['loss'])
        engine.scheduler.step(mean_loss)
        engine.learning_rate = engine.scheduler.get_last_lr()[0]
        epoch_records["lr"].append(engine.learning_rate)

        return epoch_records
    
    def gwnet_test_loop(self, 
            args, 
            config, 
            test_dl, 
            scaler, 
            draw_scatter:bool=False
        ):
        
        from tqdm.auto import tqdm
        _, _, img_path, _ = create_runtime_paths(args)

        epoch_records = {'loss': [], "mape":[], "rmse":[]}

        if draw_scatter is True:
            nbins= 200
            bin0 = np.linspace(0, config.max_value, nbins+1)
            n = np.zeros((nbins,nbins))
        else: n = 0
        
        outputs = []
        target = []
        for iter, (x, y) in enumerate(tqdm(test_dl, desc="Testing", unit="batch")):
            trainx = torch.Tensor(x).to(config.device)
            trainx= trainx.transpose(2, 3)
            trainy = torch.Tensor(y).to(config.device)
            trainy = trainy.transpose(2, 3)
            loss, mape, rmse, output, real, n = self.test(args, 
                trainx, 
                trainy[:, 0,:,:], 
                scaler=scaler,
                n=n, 
                draw_scatter=draw_scatter
            )

            output = torch.clamp(output, -1, 1)
            real = torch.clamp(real, -1, 1)
            outputs.append(output)
            target.append(real)
           
            epoch_records["loss"].append(loss)
            epoch_records["rmse"].append(rmse)
            epoch_records["mape"].append(mape)

        if img_path is not None and draw_scatter is True:
            plot_scatter_hist(n,  bin0, img_path)

        target = torch.cat(target,dim=0)
        outputs = torch.cat(outputs,dim=0)

        return epoch_records, outputs, target

def draw_adj_heatmap(engine, img_path):
    import seaborn as sns
    from torch.functional import F
    import pandas as pd
    adp = F.softmax(F.relu(torch.mm(engine.model.nodevec1, engine.model.nodevec2)), dim=1)
    device = torch.device('cpu')
    adp.to(device)
    adp = adp.cpu().detach().numpy()
    adp = adp*(1/np.max(adp))
    df = pd.DataFrame(adp)
    sns.heatmap(df, cmap="RdYlBu")
    plt.savefig(os.path.join(img_path,"emb.pdf"))
    logger.info("Adjacency matrix input saved in {}".format(img_path))

def generate_adj_dist(df, normalized_k=0.05,):
    coord = df[['lat', 'lon']].values
    dist_mx = cdist(coord, coord,
                   lambda u, v: geodesic(u, v).kilometers)
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))

    adj_mx[adj_mx < normalized_k] = 0
    return adj_mx

def generate_adj_matrix(dataset=None,
                        mask = None, 
                        save_dir = None,
                        save_plot:bool=True, 
                        load_zarr:bool=False):
    
    logger.info("Generating new adjacency matrix...")
    import seaborn as sns
    from analysis.configs.config_models import config_gwnet
    from utils.function_clns import config, crop_image_left
    import pandas as pd

    img_path = config_gwnet.data_dir

    if save_dir is None:
        save_dir = config_gwnet.data_dir

    if load_zarr is True:
        path = os.path.join(config["DEFAULT"]["basepath"], "hydro_vars.zarr")
        data = xr.open_zarr(path).isel(time=0)["total_precipitation"]
        dim = config["GWNET"]["dim"]
        logger.debug(f"The adjacency matrix has coordinates {dim} x  {dim}")
        
        idx_lat, lat_max, idx_lon, lon_min = crop_image_left(data, dim)
        sub_dataset = data.sel(lat=slice(lat_max, idx_lat), 
                                              lon=slice(lon_min, idx_lon))

        df = sub_dataset.to_dataframe()
        adj_dist = generate_adj_dist(df.reset_index())
    
    else:
        from utils.function_clns import extract_grid
        logger.debug(f"The adjacency matrix has coordinates "
                     f"{dataset.shape[-2]} x  {dataset.shape[-1]}")
        grid = extract_grid((dataset.shape[-2], dataset.shape[-1]), 
                          return_pandas=False)
        if mask is not None:
            mask_flatten = mask.flatten()
            grid_filtered = grid[~mask_flatten]
            df = pd.DataFrame(grid_filtered, columns=["lon","lat"])

        adj_dist = generate_adj_dist(df)
    
    with open(os.path.join(save_dir, "adj_dist.pkl"), 'wb') as f:
            pickle.dump(adj_dist, f, protocol=2)

    if save_plot is True:
        df = df.reset_index()
        df["lat_lon"] = df["lat"].astype(str) + ", " + df["lon"].astype(str)
        mask = np.zeros_like(adj_dist)
        mask[np.triu_indices_from(mask)] = True
        ax_labels = [df.iloc[i].lat_lon for i in range(adj_dist.shape[0])]
    
        sns.heatmap(adj_dist, mask=mask, cmap="YlGnBu", xticklabels=ax_labels, yticklabels=ax_labels)\
            .get_figure().savefig(os.path.join(img_path, "adj_dist.png"))



if __name__=="__main__":
    # import time
    # # get the start time

    start = time.time()
    from utils.function_clns import config
    import os
    from analysis import pipeline_wavenet

    parser = argparse.ArgumentParser()
    parser.add_argument('-f')

    parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
    parser.add_argument('--gcn_bool',default=True,help='whether to add graph convolution layer')
    parser.add_argument('--aptonly',action='store_true',help='whether   adaptive adj')
    parser.add_argument('--addaptadj',default=True,help='whether add adaptive adj')
    parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
    parser.add_argument('--aptinit',action=None)

    # parser.add_argument('--expid',type=int,default=1,help='experiment id')
    # parser.add_argument("--country", type=list, default=["Kenya", "Ethiopia","Somalia"], help="Location for dataset")
    # parser.add_argument("--region",type=list, default=None, help="region location for dataset")
    parser.add_argument("--model", type=str, default="WNET", help="DL model training")
    parser.add_argument('--step_length',type=int,default=15)
    parser.add_argument('--feature_days',type=int,default=30)
    parser.add_argument('--scatterplot',type=bool,default=True)
    parser.add_argument("--normalize", type=bool, default=True, help="Input data normalization")

    args = parser.parse_args()
    # main(args)

    # checkpoint = "src/output/gwnet/days_15/features_60/checkpoints/checkpoint_epoch_261.pth.tar"
    pipeline_wavenet(args,
                    use_water_mask=True,
                    load_local_precipitation=True,
                    precipitation_only=False)
    end = time.time()
    total_time = end - start
    print("\n The script took "+ time.strftime("%H%M:%S", \
                                                    time.gmtime(total_time)) + "to run")