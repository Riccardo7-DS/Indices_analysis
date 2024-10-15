if __name__=="__main__":
    from analysis.deep_learning.ConvLSTM.pipeline_convlstm import pipeline_convlstm
    from analysis import pipeline_gnn
    import argparse
    import os
    import pyproj
    import torch
    from analysis.configs.config_models import config_gwnet as model_config

    parser = argparse.ArgumentParser(description='test', conflict_handler="resolve")
    parser.add_argument('-f')
    parser.add_argument('--model', default=os.environ.get('MODEL', "GWNET"))
    parser.add_argument('--mode', default=os.environ.get('MODE', "train"))

    parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
    parser.add_argument('--gcn_bool',default=True,help='whether to add graph convolution layer')
    parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
    parser.add_argument('--addaptadj', default=True, help='whether add adaptive adj')
    parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')

    parser.add_argument("--country", type=list, default=["Kenya","Somalia","Ethiopia", "Djibouti"], help="Location for dataset")
    parser.add_argument("--region", type=list, default=None, help="Location for dataset")

    parser.add_argument('--step_length',type=int,default=os.environ.get('step_length', 15),help='days in the future')
    parser.add_argument('--feature_days',type=int,default=os.environ.get('feature_days', 90))

    parser.add_argument('--fillna',type=bool,default=False)
    parser.add_argument("--interpolate", type=bool, default=False, help="Input data interpolation over time")
    parser.add_argument("--normalize", type=bool, default=True, help="normalization")
    parser.add_argument("--scatterplot", type=bool, default=True, help="scatterplot")
    parser.add_argument('--crop_area',type=bool,default=False)
    parser.add_argument('--plotheatmap', default=False, help="Save adjacency matrix heatmap")


    parser.add_argument('--checkpoint',type=int,default=os.environ.get('checkpoint', 0))

    args = parser.parse_args()
    os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()

    if (args.mode == "eval") and (args.checkpoint==0):
        raise ValueError("Please chose a checkpoint if in evaluate mode")  

    for feature_days in [90]: 
        for window in [10, 15, 30]:
            if args.checkpoint > 0 :
                checkpoint_path =  model_config.output_dir + f"/{(args.model).lower()}" \
                f"/days_{window}/features_{feature_days}" \
                f"/checkpoints/checkpoint_epoch_{args.checkpoint}.pth.tar"
            else:
                checkpoint_path = None

            parser.add_argument('--step_length', type=int, default=window)
            parser.add_argument('--feature_days', type=int, default=feature_days)
            args = parser.parse_args()
            try:
                if args.model == "CONVLSTM":
                    pipeline_convlstm(args, precipitation_only=False)
                elif (args.model == "WNET") or (args.model == "GWNET"):
                    pipeline_gnn(args,
                        use_water_mask=True,
                        load_local_precipitation=True,
                        precipitation_only=False,
                        checkpoint_path=checkpoint_path)
                    
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("CUDA out of memory error caught.")
                    # Optionally, free up memory or handle the error as needed
                    torch.cuda.empty_cache()
                else:
                    raise e  # Re-raise the exception if it's not a CUDA out of memory error


