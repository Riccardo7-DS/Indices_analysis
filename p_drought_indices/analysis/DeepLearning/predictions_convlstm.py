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
    
    product = "ERA5_land"
    CONFIG_PATH = "config.yaml"
    config = load_config(CONFIG_PATH)
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
    parser.add_argument('--batch_size',type=int,default=config["CONVLSTM"]["batch_size"],help='batch size')
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
    parser.add_argument("--dim", type=int, default= config["CONVLSTM"]["pixels"], help="")
    parser.add_argument("--convlstm", type=bool, default= True, help="")

    args = parser.parse_args()
    sub_precp, ds = data_preparation(args, CONFIG_PATH, precp_dataset=args.precp_product)

    #sub_precp = sub_precp.to_dataset()
    data, target = interpolate_prepare(args, sub_precp, ds)

    path = os.path.join(config["DEFAULT"]["output"], "checkpoints\convlstm_model.pt")
    model = torch.load(path)
    model.eval()

    import numpy as np
    from p_drought_indices.configs.config_3x3_16_3x3_32_3x3_64 import config
    from torch.nn import MSELoss
    import matplotlib.pyplot as plt
    from p_drought_indices.analysis.DeepLearning.ConvLSTM import ConvLSTM, train_loop, valid_loop, build_logging
    import numpy as np
    from p_drought_indices.functions.function_clns import load_config, CNN_split

    train_split = 0.8
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

    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
        with torch.no_grad():
            inputs = inputs.float().to(config.device)
            targets = targets.float().to(config.device)
            # Forward pass
            outputs = model(inputs)

            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.title('Input Frame')
            plt.imshow(inputs[0].cpu().numpy(), cmap='gray')

            plt.subplot(122)
            plt.title('Predicted Frame')
            plt.imshow(outputs[0].cpu().numpy(), cmap='gray')
            plt.show()