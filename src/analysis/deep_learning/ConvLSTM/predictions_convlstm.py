if __name__=="__main__":
    from analysis.deep_learning.GWNET.pipeline_gwnet import data_preparation 
    import pickle
    import os
    import matplotlib.pyplot as plt
    from utils.function_clns import load_config, prepare, CNN_split, interpolate_prepare
    import numpy as np
    from analysis.deep_learning.dataset import CustomConvLSTMDataset
    from torch.utils.data import DataLoader
    import argparse
    import torch
    from analysis.deep_learning.ConvLSTM.ConvLSTM import train_loop, valid_loop, build_logging
    from analysis.deep_learning.ConvLSTM.cvlstm import ConvLSTM
    from torch.nn import MSELoss
    from torchvision.transforms import transforms 
    from loguru import logger
    import sys

    logger.remove()
    logger.add(sys.stderr, format = "{time:YYYY-MM-DD at HH:mm:ss} | <lvl>{level}</lvl> {level.icon} | <lvl>{message}</lvl>", colorize = True)
    
    product = "ERA5"
    CONFIG_PATH = "config.yaml"
    config_file = load_config(CONFIG_PATH)
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
    parser.add_argument('--batch_size',type=int,default=config_file["CONVLSTM"]["batch_size"],help='batch size')
    parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
    parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
    parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
    parser.add_argument('--print_every',type=int,default=50,help='Steps before printing')
    parser.add_argument('--expid',type=int,default=1,help='experiment id')
    parser.add_argument('--latency',type=int,default=90,help='days used to accumulate precipitation for SPI')
    parser.add_argument('--spi',type=bool,default=False,help='if dataset is SPI')
    parser.add_argument('--precp_product',type=str,default=product,help='precipitation product')
    parser.add_argument('--forecast',type=int,default=12,help='days used to perform forecast')
    parser.add_argument('--seq_length',type=int,default=12,help='')
    #parser.add_argument("--location", type=list, default=["Amhara"], help="Location for dataset")
    parser.add_argument("--dim", type=int, default= config_file["CONVLSTM"]["pixels"], help="")
    parser.add_argument("--convlstm", type=bool, default= True, help="")
    parser.add_argument("--country", type=list, default=["Kenya","Somalia","Ethiopia"], help="Location for dataset")
    parser.add_argument("--region", type=list, default=None, help="Location for dataset")

    args = parser.parse_args()
    logger.info("Starting preparing data...")
    sub_precp, ds = data_preparation(args, CONFIG_PATH, precp_dataset=args.precp_product, ndvi_dataset="ndvi_smoothed_w2s.nc")

    logger.info("Starting interpolation...")
    #sub_precp = sub_precp.to_dataset()
    data, target = interpolate_prepare(args, sub_precp, ds)
    
    from configs.config_3x3_16_3x3_32_3x3_64 import config


    path = os.path.join(config_file["DEFAULT"]["output"], "checkpoints\convlstm_model.pt")
    

    #logger = build_logging(config)
    model = ConvLSTM(config).to(config.device)
    criterion = MSELoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    model.load_state_dict(state['state_dict'])
    model.eval()
    logger.info("Correctly loaded model")


    train_split = 0.8
    #### training parameters
    logger.info("Splitting data for training...")
    train_data, test_data, train_label, test_label = CNN_split(data, target, 
                                                               split_percentage=train_split)
    
    print("First step shape training data...", train_data.shape)

    batch_size = config_file["CONVLSTM"]["batch_size"]
    # create a CustomDataset object using the reshaped input data
    train_dataset = CustomConvLSTMDataset(config, train_data, train_label)
    test_dataset = CustomConvLSTMDataset(config, test_data, test_label)

    print("train shape", train_dataset)

    # create a DataLoader object that uses the dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    pics = 5
    samples = 5
    
    
    fig, axes = plt.subplots(1, pics, figsize=(5*5, 5))
    for n in range(pics):
        img = target[:, :, n]
        axes[n].imshow(img, cmap="RdYlGn")
    
    plt.show()

    fig, axes = plt.subplots(1, pics, figsize=(5*5, 5))
    for n in range(pics):
        img = train_label[ 0, n,  :, :] 
        axes[n].imshow(img, cmap="RdYlGn")

    # Show the plot
    plt.show()

    # print("Plotting precipitation from train set")
    # fig, axes = plt.subplots(samples, pics, figsize=(5*5, 5))
    # for x in range(samples):
    #     for n in range(pics, 0, -1):
    #         img = test_dataset.labels[x, -n, 0, :, :]
    #         axes[x, n-pics].imshow(img, cmap= "RdBu")

    # plt.show()

    print("Plotting vegetation from train set")
    fig, axes = plt.subplots(1, pics, figsize=(5*5, 5))
    for n in range(pics):
        img = train_dataset.labels[n, 0 ,0, :, :]
        axes[n].imshow(img, cmap="RdYlGn")
    
    plt.show()
    

    from utils.ndvi_functions import ndvi_colormap
    cmap_ndvi, norm =ndvi_colormap()
    plot= True
    prediction_matrix = np.zeros((len(train_dataloader.dataset),
                                  *config.input_size),dtype=np.float32)
    
    print(prediction_matrix.shape)

    predictions = [] 
    with torch.no_grad():
        current_idx = 0
        for inputs, targets in train_dataloader:
            inputs = inputs.float().to(config.device)
            targets = targets.float().to(config.device)
            outputs = torch.squeeze(model(inputs)).detach().cpu().numpy()[:, 0, :, :]
            prediction_matrix[current_idx: current_idx +outputs.shape[0], :, :] = outputs
            current_idx +=outputs.shape[0]
            #predictions = np.append(predictions, outputs)
            #inputs = inputs.reshape(inputs.shape[1], inputs.shape[2], inputs.shape[0])
            print(inputs.shape, targets.shape, inputs.max(), inputs.min())

            if plot is True:
                #images = torch.cat([inputs, targets], dim=1)
                fig, axes = plt.subplots(1, pics, figsize=(inputs.shape[1]*5, 5))
                for n in range(pics):
                    img = outputs[n, :, :]
                    axes[n].imshow(img,  cmap=cmap_ndvi, vmax=1, vmin=-0.4)
                # Show the plot
                plt.show()

            # Save the matrix to a .npy file
        np.save(os.path.join(config_file["DEFAULT"]["output"],"matrix.npy"), prediction_matrix)

        import xarray as xr

        n_samples = data.shape[-1]
        train_samples = int(round(train_split*n_samples, 0))

        lat = ds["lat"].values
        lon = ds["lon"].values
        time = ds.isel(time=slice(0, train_samples-(config.num_frames_input + config.step_length + config.num_frames_output)))["time"].values 

        da = xr.DataArray(prediction_matrix, 
                        coords={'time': time, 'lat': lat, 'lon': lon},
                        dims=['time', 'lat', 'lon'],
                        name="ndvi").to_netcdf(os.path.join(config_file["DEFAULT"]["output"],"predicted_ndvi.nc"))
