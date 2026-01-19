if __name__=="__main__":
    from analysis.deep_learning.ConvLSTM.pipeline_convlstm import pipeline_convlstm
    from analysis import pipeline_gnn
    import argparse
    import os
    import pyproj
    import torch
    import matplotlib
    matplotlib.use("Agg")
    from analysis.configs.config_models import config_gwnet as model_config
    from analysis import create_runtime_paths


    parser = argparse.ArgumentParser(description='test', conflict_handler="resolve")
    parser.add_argument('-f')
    parser.add_argument('--model', default=os.environ.get('model', "GWNET"))
    parser.add_argument('--mode', default=os.environ.get('mode', "train"))
    parser.add_argument("--cross_val", default=os.environ.get("cross_val"), action="store_true")

    parser.add_argument('--loglevel', default=os.environ.get("loglevel"), action="store_true")
    parser.add_argument('--save_output', default=os.environ.get('save_output',True))

    parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
    parser.add_argument('--gcn_bool',default=True,help='whether to add graph convolution layer')
    parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
    parser.add_argument('--addaptadj', default=True, help='whether add adaptive adj')
    parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')

    parser.add_argument("--country", type=list, default=["Kenya","Somalia","Ethiopia", "Djibouti"], help="Location for dataset")
    parser.add_argument("--region", type=list, default=None, help="Location for dataset")

    parser.add_argument('--step_length',type=int,default=os.environ.get('step_length', 15),help='days in the future')
    parser.add_argument('--feature_days',type=int,default=os.environ.get('feature_days', 90))
    parser.add_argument('--step',type=int,default=os.environ.get('step', 1))
    parser.add_argument('--only_lag', default=bool(os.environ.get("only_lag")), action="store_true")

    parser.add_argument('--fillna',type=bool,default=False)
    parser.add_argument("--interpolate", type=bool, default=False, help="Input data interpolation over time")
    parser.add_argument("--normalize", type=bool, default=True, help="normalization")
    parser.add_argument("--scatterplot", type=bool, default=os.environ.get("scatterplot", False), help="scatterplot")
    parser.add_argument('--crop_area',type=bool,default=False)
    parser.add_argument('--plotheatmap', default=False, help="Save adjacency matrix heatmap")

    parser.add_argument('--checkpoint',type=int,default=os.environ.get('checkpoint', 0))


    ### Arguments for DDP training
    parser.add_argument('--local_rank', type=int, default=os.environ.get('local_rank', 0))
    parser.add_argument('--ddp', default=bool(os.environ.get('ddp', False)), action="store_true",
                        help="Use Distributed Data Parallel training")
    parser.add_argument("--num_nodes", type=int, default=os.environ.get("num_nodes", 1), help="Number of available nodes for DDP")
    parser.add_argument("--num-gpus", type=int, default=os.environ.get("num_gpus", 1), help="Number of available GPUs per node for DDP")
    parser.add_argument("--node-id", type=int, default=os.environ.get("node_id", 0), help="ID of the current node for DDP")

    args = parser.parse_args()
    os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()


    if (args.mode == "test") and (args.checkpoint==0):
        raise ValueError("Please chose a checkpoint if in evaluate mode")  

    def run_pipeline(local_args):
        from analysis.deep_learning.ConvLSTM.pipeline_convlstm import pipeline_convlstm
        from analysis import pipeline_gnn
        from analysis.configs.config_models import config_gwnet as model_config
        from analysis import create_runtime_paths
        import torch

        for feature_days in [local_args.feature_days]:
            for window in [local_args.step_length]:
                if local_args.checkpoint > 0:
                    _, _, _, checkpoint_dir = create_runtime_paths(local_args)
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{local_args.checkpoint}.pth.tar")
                else:
                    checkpoint_path = None

                # set these on the args object for downstream code
                local_args.step_length = window
                local_args.feature_days = feature_days

                try:
                    if local_args.model == "CONVLSTM":
                        pipeline_convlstm(local_args, precipitation_only=False, checkpoint_path=checkpoint_path)
                    elif (local_args.model == "WNET") or (local_args.model == "GWNET"):
                        pipeline_gnn(local_args,
                            use_water_mask=True,
                            load_local_precipitation=True,
                            precipitation_only=False,
                            checkpoint_path=checkpoint_path,
                            add_extra_data=True
                        )

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print("CUDA out of memory error caught.")
                        torch.cuda.empty_cache()
                    else:
                        raise e


    def spawn_worker(local_rank, args):
        import os
        # set sensible defaults for master address/port if not provided
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29500')
        # set the local rank and initialize process group + device
        args.local_rank = local_rank
        from analysis.deep_learning.utils_models import worker
        device, world_size = worker(args)
        # call the pipeline entrypoint; it will use args and the initialized process group
        run_pipeline(args)

    # Launch either spawned processes (DDP) or single-process run
    if args.ddp:
        import torch.multiprocessing as mp
        mp.spawn(spawn_worker, nprocs=args.num_gpus, args=(args,), join=True)
    else:
        run_pipeline(args)

