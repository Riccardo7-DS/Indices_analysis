
import argparse
from p_drought_indices.analysis.DeepLearning.dataset import MyDataset
from p_drought_indices.functions.function_clns import load_config, prepare, get_lat_lon_window, subsetting_pipeline, check_xarray_dataset, check_timeformat_arrays
import xarray as xr
import os
import numpy as np
import torch


def get_predicted_dataset(pred, dates, sub_cols):
    df = pd.DataFrame(pred.cpu())
    df.index = list(set(dates))
    df.columns = sub_cols
    new_df = df.stack().reset_index()
    #new_df["lat"] = new_df.iloc[:,1].apply(lambda x: x[1:-1].split(",")[0]).astype(np.float32)
    #new_df["lon"] = new_df.iloc[:,1].apply(lambda x: x[1:-1].split(",")[1]).astype(np.float32)
    new_df["lat"] = new_df.iloc[:,1].apply(lambda x: x[0]).astype(np.float32)
    new_df["lon"] = new_df.iloc[:,1].apply(lambda x: x[1]).astype(np.float32)
    new_df.columns=["time", "latlon","ndvi","lat","lon"]
    data = new_df[["time","lat","lon", "ndvi"]]
    data = data.sort_values(by=["time","lat","lon"])
    data.set_index(["time","lat","lon"], inplace=True)
    ds = data.to_xarray()
    return ds


if __name__=="__main__":
    CONFIG_PATH = "config.yaml"
    import time
    product = "ERA5_land"
    start = time.time()
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
    parser.add_argument('--batch_size',type=int,default=config["GWNET"]["batch_size"],help='batch size')
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
    parser.add_argument("--region", type=list, default=["Amhara"], help="Location for dataset")
    parser.add_argument("--country", type=list, default=None, help="Location for dataset")
    parser.add_argument("--dim", type=int, default= config["GWNET"]["pixels"], help="")

    args = parser.parse_args()

    from p_drought_indices.analysis.DeepLearning.pipeline_gwnet import load_adj, MetricsRecorder, trainer,get_dataloader, data_preparation
    path = config["PRECIP"]["ERA5_land"]["path"]
    args.output_dir = os.path.join(path,  "graph_net")
    checkp_path = os.path.join(args.output_dir,  f"checkpoints/forecast_{args.forecast}")
    model_path = [os.path.join(checkp_path, f) for f in os.listdir(checkp_path) if "best" in f][0]
    
    sub_precp, ds =  data_preparation(args, CONFIG_PATH, precp_dataset=args.precp_product)

    print("Checking precipitation dataset...")
    check_xarray_dataset(args, sub_precp, save=True)
    print("Checking vegetation dataset...")
    check_xarray_dataset(args, ds, save=True)

    dataloader, num_nodes, x_df = get_dataloader(args, CONFIG_PATH, sub_precp, ds, check_matrix=True)
    dates = x_df.index
    cols = x_df.columns
    sub_cols = [ (i[1], i[2]) for i in cols]

    epochs = config["GWNET"]["epochs"]
    dim = args.dim
    device = torch.device(args.device)
    adj_path = os.path.join(os.path.join(args.output_dir,  "adjacency_matrix"), f"{args.precp_product}_{args.dim}_adj_dist.pkl")
    adj_mx = load_adj(adj_path,  args.adjtype)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    metrics_recorder = MetricsRecorder()

    if args.spi==True:
        checkp_path = os.path.join(args.output_dir,  f"checkpoints/forecast_{args.precp_product}_SPI_{args.latency}")
    else:
        checkp_path = os.path.join(args.output_dir,  f"checkpoints/forecast_{args.forecast}")

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = trainer(scaler, args.in_dim, args.seq_length, num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit)

    model = engine.model.load_state_dict(torch.load(model_path))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]
    print("realy dims is: {}", realy.shape)

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        print("single x shape: {}", x.shape)
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1,3)
            print("single testx shape: {}", testx.shape)
            print("single prediction shape: {}", preds.shape)
            print("squeezed pred dims is: {}", preds.squeeze().shape)
            outputs.append(preds[:,0,:,:].squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    for i in range(args.seq_length):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]

        print(pred.shape)
        print(real.shape)
        print(pred[0].shape)

        import pandas as pd
        import numpy as np
        base_path = os.path.join(args.output_dir, "predicted_data")
        ds = get_predicted_dataset(pred, dates, sub_cols)
        ds.to_netcdf(os.path.join(base_path, f"predicted_ndvi_{i}.nc"))

    





