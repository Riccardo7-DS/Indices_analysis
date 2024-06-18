if __name__=="__main__":
    from analysis.deep_learning.ConvLSTM.pipeline_convlstm import pipeline_convlstm
    from analysis import pipeline_wavenet, pipeline_weatherGCNet
    import argparse
    import os
    import pyproj
    import torch

    os.environ['MODEL'] = "WNET"

    parser = argparse.ArgumentParser(description='test', conflict_handler="resolve")
    parser.add_argument('-f')
    # parser.add_argument('--model',type=str,default="CONVLSTM",help='DL model training')
    parser.add_argument("--normalize", type=bool, default=True, help="Input data normalization")
    parser.add_argument("--scatterplot", type=bool, default=True, help="Whether to visualize scatterplot")

    parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
    parser.add_argument('--gcn_bool',default=True,help='whether to add graph convolution layer')
    parser.add_argument('--aptonly',action='store_true',help='whether   adaptive adj')
    parser.add_argument('--addaptadj',default=True,help='whether add adaptive adj')
    parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
    parser.add_argument('--aptinit',action=None)
    parser.add_argument('--model', default=os.environ.get('MODEL'))

    args = parser.parse_args()
    os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()

    for feature_days in [90, 60, 45, 30, 15]: 
        for window in [15, 30, 45, 60, 90, 120]:
            parser.add_argument('--step_length', type=int, default=window)
            parser.add_argument('--feature_days', type=int, default=feature_days)
            args = parser.parse_args()
            try:
                if args.model == "CONVLSTM":
                    pipeline_convlstm(args, precipitation_only=False)
                elif args.model == "WNET":
                    pipeline_wavenet(args, None)
                elif args.model == "GWNET":
                    pipeline_weatherGCNet(args, None)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("CUDA out of memory error caught.")
                    # Optionally, free up memory or handle the error as needed
                    torch.cuda.empty_cache()
                else:
                    raise e  # Re-raise the exception if it's not a CUDA out of memory error
