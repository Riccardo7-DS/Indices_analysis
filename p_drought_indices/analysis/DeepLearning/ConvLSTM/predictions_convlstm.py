if __name__=="__main__":
    from p_drought_indices.analysis.DeepLearning.pipeline_gwnet import data_preparation 
    import pickle
    import os
    import matplotlib.pyplot as plt
    from p_drought_indices.functions.function_clns import load_config, prepare, CNN_split, interpolate_prepare
    import numpy as np
    from p_drought_indices.analysis.DeepLearning.dataset import CustomDataset
    from torch.utils.data import DataLoader
    import argparse
    import torch
    from p_drought_indices.analysis.DeepLearning.ConvLSTM.ConvLSTM import ConvLSTM, train_loop, valid_loop, build_logging
    from torch.nn import MSELoss
    from p_drought_indices.configs.config_3x3_16_3x3_32_3x3_64 import config
    from torchvision.transforms import transforms 
    from loguru import logger
    import sys

    logger.remove()
    logger.add(sys.stderr, format = "{time:YYYY-MM-DD at HH:mm:ss} | <lvl>{level}</lvl> {level.icon} | <lvl>{message}</lvl>", colorize = True)
    
    product = "ERA5_land"
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

    args = parser.parse_args()
    logger.info("Starting preparing data...")
    sub_precp, ds = data_preparation(args, CONFIG_PATH, precp_dataset=args.precp_product)

    logger.info("Starting interpolation...")
    #sub_precp = sub_precp.to_dataset()
    data, target = interpolate_prepare(args, sub_precp, ds)

    print("DATA SHAPE: ",data.shape)
    
    from p_drought_indices.configs.config_3x3_16_3x3_32_3x3_64 import config


    path = os.path.join(config_file["DEFAULT"]["output"], "checkpoints\convlstm_model_1.pt")
    
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

    import numpy as np
    from torch.nn import MSELoss
    import matplotlib.pyplot as plt
    from p_drought_indices.analysis.DeepLearning.ConvLSTM.ConvLSTM import ConvLSTM, train_loop, valid_loop, build_logging
    import numpy as np
    from p_drought_indices.functions.function_clns import load_config, CNN_split

    train_split = 0.8
    #### training parameters
    logger.info("Splitting data for training...")
    train_data, test_data, train_label, test_label = CNN_split(data, target, 
                                                               split_percentage=train_split)
    
    print("First step shape training data...", train_data.shape)
    config_file = load_config(CONFIG_PATH=CONFIG_PATH)
    batch_size = config_file["CONVLSTM"]["batch_size"]
    # create a CustomDataset object using the reshaped input data
    train_dataset = CustomDataset(train_data, train_label)
    test_dataset = CustomDataset(test_data, test_label)

    # create a DataLoader object that uses the dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

<<<<<<< HEAD

    logger.info("Checking data with images....")
    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
=======
    pics = 5


    fig, axes = plt.subplots(1, pics, figsize=(5*5, 5))
    for n in range(pics):
        img = data[:, :, n]
        axes[n].imshow(img, cmap="RdYlGn")
    
    plt.show()

    fig, axes = plt.subplots(1, pics, figsize=(5*5, 5))
    for n in range(pics):
        img = train_data[0,n, 0, :, :] 
        axes[n].imshow(img, cmap="RdYlGn")

    # Show the plot
    plt.show()

    fig, axes = plt.subplots(1, pics, figsize=(5*5, 5))
    for n in range(pics):
        img = train_dataset.data[0,n, 0, :, :] 
        axes[n].imshow(img, cmap="RdYlGn")
    
    plt.show()

    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs = inputs.float().to(config.device)
        targets = targets.float().to(config.device)

        inputs = torch.squeeze(inputs)
        inputs = inputs.reshape(inputs.shape[1], inputs.shape[2], inputs.shape[0])
        
        print(inputs.shape, targets.shape, inputs.max(), inputs.min())
        #images = torch.cat([inputs, targets], dim=1)
        fig, axes = plt.subplots(1, pics, figsize=(inputs.shape[1]*5, 5))
        for n in range(pics):
            img = inputs[ :, :, n].numpy()
            axes[n].imshow(img, cmap="RdYlGn")
    # Show the plot
    plt.show()

    import sys 
    sys.exit(0)

    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
>>>>>>> 18d558d5358923a158e46388d0339843807f4a25
        with torch.no_grad():
            inputs = inputs.float().to(config.device)
            targets = targets.float().to(config.device)
            print(inputs.shape, targets.shape, inputs.max(), inputs.min())
            # Forward pass
            outputs = model(inputs)

            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.title('Input Frame')
            print(inputs[0, 4000, 0].detach().cpu().numpy())
            img = inputs[0, 4000, 0].detach().cpu().numpy()
            plt.imshow(img)
            plt.show()

            plt.subplot(122)
            plt.title('Predicted Frame')
            print(outputs[0, 4000, 0].detach().cpu().numpy())
            img = outputs[0, 4000, 0].detach().cpu().numpy()
            plt.imshow(img)
            plt.show()