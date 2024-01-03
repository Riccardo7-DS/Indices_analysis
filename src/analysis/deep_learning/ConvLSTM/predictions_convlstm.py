if __name__=="__main__":
    from analysis.deep_learning.GWNET.pipeline_gwnet import data_preparation, create_paths
    import pickle
    import os
    import matplotlib.pyplot as plt
    from utils.function_clns import config as config_file, prepare, CNN_split, interpolate_prepare
    import numpy as np
    from analysis.deep_learning.dataset import CustomConvLSTMDataset
    from torch.utils.data import DataLoader
    import argparse
    import torch
    from analysis.deep_learning.ConvLSTM.clstm_unet import train_loop, valid_loop, build_logging
    from torch.nn import MSELoss
    from torchvision.transforms import transforms 
    from loguru import logger
    import sys
    import xarray as xr
    from configs.config_3x3_16_3x3_32_3x3_64 import config
    from analysis.deep_learning.ConvLSTM.clstm import ConvLSTM
    import time

    logger.remove()
    logger.add(sys.stderr, format = "{time:YYYY-MM-DD at HH:mm:ss} | <lvl>{level}</lvl> {level.icon} | <lvl>{message}</lvl>", colorize = True)
    
    product = "ERA5"
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument('--device',type=str,default='cuda',help='')
    parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
    parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
    parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
    parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
    parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
    parser.add_argument('--nhid',type=int,default=32,help='')
    parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
    parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
    parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
    parser.add_argument('--print_every',type=int,default=50,help='Steps before printing')
    parser.add_argument('--expid',type=int,default=1,help='experiment id')
    parser.add_argument('--spi',type=bool,default=False,help='if dataset is SPI')

    #parser.add_argument("--location", type=list, default=["Amhara"], help="Location for dataset")
    parser.add_argument("--pipeline", type=str, default="CONVLSTM", help="")
    parser.add_argument("--country", type=list, default=["Kenya","Somalia","Ethiopia"], help="Location for dataset")
    parser.add_argument("--region", type=list, default=None, help="Location for dataset")
    parser.add_argument("--normalize", type=bool, default=False, help="normalization")
    parser.add_argument("--scatterplot", type=bool, default=True, help="scatterplot")
    parser.add_argument('--step_length',type=int,default=1,help='days in the future')

    args = parser.parse_args()
    logger.info("Starting preparing data...")

    output_dir, log_path, img_path, checkpoint_dir = create_paths(args)

    sub_precp, ds, ndvi_scaler = data_preparation(args, 
                                                  precp_dataset=config_file[args.pipeline]['precp_product'],
                                                  ndvi_dataset="ndvi_smoothed_w2s.nc")
    
    mask = torch.tensor(np.array(xr.where(ds.isel(time=0).notnull(), 1, 0)))
    
    logger.info("Starting interpolation...")
    #sub_precp = sub_precp.to_dataset()
    data, target = interpolate_prepare(args, sub_precp, ds)
    

    #model = ConvLSTM(config).to(config.device)

    
    model = ConvLSTM(config.num_samples, 
                     [32, 32, 32],  
                     (3,3), 3, True, True, False).to(config.device)

    if config.masked_loss is False:
        criterion = MSELoss().to(config.device)
    else:
        criterion = MSELoss(reduction='none').to(config.device)

    checkpoint = [f for f in os.listdir(checkpoint_dir) if config.model_name in f][0]

    path =  os.path.join(checkpoint_dir, checkpoint) 
    print("The file was last modified at", time.ctime(os.path.getmtime(path)))

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    logger.info("Correctly loaded model")

    train_split = 0.7

    #### training parameters
    logger.info("Splitting data for training...")
    train_data, val_data, train_label, val_label, test_data, test_label = CNN_split(data, target, 
                                                               split_percentage=train_split)
    
    print("First step shape training data...", train_data.shape)

    print("train data:", train_data.shape)
    print("val data:", val_data.shape)
    print("test data:", test_data.shape)
    
    print("train label:", train_label.shape)
    print("val label:", val_label.shape)
    print("test label:", test_label.shape)

    batch_size = config.batch_size
    # create a CustomDataset object using the reshaped input data
    train_dataset = CustomConvLSTMDataset(config, args, train_data, train_label)
    val_dataset = CustomConvLSTMDataset(config, args, val_data, val_label)
    test_dataset = CustomConvLSTMDataset(config, args, test_data, test_label)
    
    # create a DataLoader object that uses the dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False)

    plot = False
    evaluate = True
    pics = 5
    samples = 5
    dataloader = test_dataloader

    prediction_matrix = np.zeros((len(dataloader.dataset),
                                  *config.input_size),dtype=np.float32)
    
    print("Prediction matrix shape", prediction_matrix.shape)

    if evaluate is True:
        from analysis.deep_learning.ConvLSTM.clstm_unet import valid_loop
        from analysis.deep_learning.GWNET.pipeline_gwnet import MetricsRecorder

        metrics_recorder = MetricsRecorder()

        valid_records =[]
        rmse_valid = []
        mape_valid = []
        epoch = 1

        epoch_records = valid_loop(config, args, logger, epoch, model, dataloader, criterion,
                                   ndvi_scaler, mask, draw_scatter=True)
        valid_records.append(np.mean(epoch_records['loss']))
        rmse_valid.append(np.mean(epoch_records['rmse']))
        mape_valid.append(np.mean(epoch_records['mape']))
        log = 'Epoch: {:03d}, Test Loss: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        logger.info(log.format(epoch, np.mean(epoch_records['loss']), np.mean(epoch_records['mape']), np.mean(epoch_records['rmse'])))
        metrics_recorder.add_val_metrics(np.mean(epoch_records['mape']), np.mean(epoch_records['rmse']), np.mean(epoch_records['loss']))

        
        current_idx = 0
        for inputs, targets in dataloader:
            inputs = inputs.float().to(config.device)
            targets = targets.float().to(config.device)
            outputs = torch.squeeze(model(inputs)).detach().cpu().numpy()
            # num_dimensions = outputs.ndim
            # if num_dimensions==4:
            #     outputs = outputs[:,0,:,:]
            # elif num_dimensions==3:
            #     outputs = outputs[0,:,:]
            prediction_matrix[current_idx: current_idx +outputs.shape[0], :, :] = outputs
            current_idx +=outputs.shape[0]
    

    if plot is True:
        from utils.ndvi_functions import ndvi_colormap
        cmap_ndvi, norm =ndvi_colormap()

        fig, axes = plt.subplots(1, pics, figsize=(5*5, 5))
        for n in range(pics):
            img = target[:, :, n]
            axes[n].imshow(img, cmap="RdYlGn")

        plt.show()

        fig, axes = plt.subplots(1, pics, figsize=(5*5, 5))
        for n in range(pics):
            img = train_label[ 0, n,  :, :] 
            axes[n].imshow(img, cmap="RdYlGn")

        plt.show()

        print("Plotting vegetation from train set")
        fig, axes = plt.subplots(1, pics, figsize=(5*5, 5))
        for n in range(pics):
            img = val_dataset.labels[n, 0 ,0, :, :]
            axes[n].imshow(img, cmap="RdYlGn")

        plt.show()

        
    def get_subset_datarray(ds, prediction_matrix, train_split:float, test_split:float, dataset:str="test"):
        import xarray as xr
        from utils.function_clns import config as config_file
        n_samples = ds.sizes["time"] #data.shape[-1]
        train_samples =  int(round(train_split* n_samples, 0))
        test_samples = int(round(test_split* n_samples, 0))
        val_samples = n_samples - train_samples - test_samples

        lat = ds["lat"].values
        lon = ds["lon"].values
        if dataset == "train":
            time = ds.isel(time=slice(0, train_samples-(config.num_frames_input + config.step_length + config.num_frames_output)))["time"].values 
        
        elif dataset == "val":
            time = ds.isel(time=slice(train_samples, train_samples + val_samples-(config.num_frames_input + config.step_length + config.num_frames_output)))["time"].values 
        
        elif dataset == "test":
            time = ds.isel(time=slice(train_samples + val_samples, n_samples -(config.num_frames_input + config.step_length + config.num_frames_output)))["time"].values 

        else:
            raise NotImplementedError("You must chose between train, test and val sets!")
        
        da = xr.DataArray(prediction_matrix, 
                        coords={'time': time, 'lat': lat, 'lon': lon},
                        dims=['time', 'lat', 'lon'],
                        name="ndvi").to_netcdf(os.path.join(config_file["DEFAULT"]["output"],"predicted_ndvi_test.nc"))
        return da
    
    get_subset_datarray(ds, prediction_matrix, train_split, test_split=0.1, dataset="test")