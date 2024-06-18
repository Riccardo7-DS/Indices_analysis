import logging
import numpy as np
import os
import torch
import time
import matplotlib.pyplot as plt
import pickle
from analysis import EarlyStopping, save_figures, create_runtime_paths, print_lr_change, MetricsRecorder, gwnet_train_loop, gwnet_val_loop, trainer, generate_adj_matrix, load_adj, get_train_valid_loader

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
            mask = norm_data == null_value
            transf[mask] = null_value
            return transf
        else:
            return transf

def pipeline_weatherGCNet(args, checkpoint_path=None):
    import torch.nn.functional as F
    import numpy as np
    from definitions import ROOT_DIR
    from analysis import WGCNModel, train_loop, valid_loop, EarlyStopping, update_tensorboard_scalars
    from utils.function_clns import init_logging, config
    from analysis import check_shape_dataloaders, init_tb
    import torch
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    import os
    from analysis.configs.config_models import config_gwnet 

    data_dir = config_gwnet.data_dir+"/data_convlstm"
    data = np.load(os.path.join(data_dir, "data.npy"))
    target = np.load(os.path.join(data_dir, "target.npy"))
    with open(os.path.join(ROOT_DIR,"../data/ndvi_scaler.pickle"), "rb") as handle:
            ndvi_scaler = pickle.loads(handle.read())

    ################################# Module level logging #############################
    _, log_path, img_path, checkpoint_dir = create_runtime_paths(args)
    init_logging("training_gwnet", verbose=False, log_file=os.path.join(log_path, 
                                                      f"gwnet_days_{args.step_length}"
                                                      f"features_{args.feature_days}.log"))
    writer = init_tb(log_path)
    logger = logging.getLogger("training_gwnet")

    train_dl, valid_dl, dataset = get_train_valid_loader(config_gwnet, 
                                                         args, 
                                                         data, 
                                                         target,
                                                         config.MODELS.split)
    check_shape_dataloaders(train_dl, valid_dl)
    model = WGCNModel(args, config_gwnet, dataset.data).to(device)

    metrics_recorder = MetricsRecorder()
    train_records, valid_records, test_records = [], [], []
    rmse_train, rmse_valid, rmse_test = [], [], []
    mape_train, mape_valid, mape_test = [], [], []

    loss_func = F.l1_loss
    loss_func_2 = F.mse_loss
    learning_rate = config_gwnet.learning_rate

    early_stopping = EarlyStopping(config_gwnet, logger, verbose=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=config_gwnet.scheduler_factor, 
        patience=config_gwnet.scheduler_patience)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_sched'])
        checkp_epoch = checkpoint['epoch']
        logger.info(f"Resuming training from epoch {checkp_epoch}")

    start_epoch = 0 if checkpoint_path is None else checkp_epoch

    for epoch in range(start_epoch, config_gwnet.epochs):
        epoch_records = train_loop(config_gwnet, args, model, train_dl, loss_func, 
                            optimizer, scaler=ndvi_scaler, mask=None, draw_scatter=False)
        
        train_records.append(np.mean(epoch_records['loss']))
        rmse_train.append(np.mean(epoch_records['rmse']))
        mape_train.append(np.mean(epoch_records['mape']))

        metrics_recorder.add_train_metrics(epoch_records, epoch)
        
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
        logger.info(log.format(epoch, np.mean(epoch_records['loss']), 
                               np.mean(epoch_records['mape']), 
                               np.mean(epoch_records['rmse'])))
        
        
        epoch_records = valid_loop(config_gwnet, args,  model, valid_dl, loss_func, 
                                   scaler=ndvi_scaler, mask=None, draw_scatter=args.scatterplot)
        
        valid_records.append(np.mean(epoch_records['loss']))
        rmse_valid.append(np.mean(epoch_records['rmse']))
        mape_valid.append(np.mean(epoch_records['mape']))

        log = 'Epoch: {:03d}, Val Loss: {:.4f}, Val MAPE: {:.4f}, Val RMSE: {:.4f}'
        logger.info(log.format(epoch, np.mean(epoch_records['loss']), 
                               np.mean(epoch_records['mape']), 
                               np.mean(epoch_records['rmse'])))
        
        metrics_recorder.add_val_metrics(epoch_records)
        
        model_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "lr_sched": scheduler.state_dict()
        }
    
        mean_loss = sum(epoch_records['loss']) / len(epoch_records['loss'])
        scheduler.step(mean_loss)
        new_lr = print_lr_change(learning_rate, scheduler)
        learning_rate = new_lr
        early_stopping( np.mean(epoch_records['loss']), 
                       model_dict, epoch, checkpoint_dir)
        
        update_tensorboard_scalars(writer, metrics_recorder)
        learning_rate = print_lr_change(learning_rate, scheduler)
        
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

        plt.plot(range(epoch - start_epoch + 1), train_records, label='train')
        plt.plot(range(epoch - start_epoch + 1), valid_records, label='valid')
        plt.legend()
        plt.savefig(os.path.join(img_path, f'learning_curve_feat_'
                                 f'{args.feature_days}.png'))
        plt.close()

# # # # # # # # # #

def pipeline_wavenet(args, checkpoint_path):
    import pickle
    from analysis.configs.config_models import config_gwnet
    from analysis import check_shape_dataloaders
    from definitions import ROOT_DIR
    from analysis import init_tb, update_tensorboard_scalars
    import sys
    from utils.function_clns import init_logging, config

    _, log_path, img_path, checkpoint_dir = create_runtime_paths(args)
    init_logging("training_wavenet", verbose=True, log_file=os.path.join(log_path, 
                                                      f"wavenet_days_{args.step_length}"
                                                      f"features_{args.feature_days}.log"))
    writer = init_tb(log_path)
    logger = logging.getLogger("training_wavenet")

    logger.info(f"Starting training WaveNet model for {args.step_length}"
                f" days in the future with {args.feature_days} days of features")

    device = config_gwnet.device
    data_dir = config_gwnet.data_dir+"/data_convlstm"
    data = np.load(os.path.join(data_dir, "data.npy"))
    target = np.load(os.path.join(data_dir, "target.npy"))
    adj_mx_path = os.path.join(config_gwnet.adj_path, "adj_dist.pkl")
    with open(os.path.join(config_gwnet.data_dir, "ndvi_scaler.pickle"), "rb") as handle:
            scaler = pickle.loads(handle.read())

    ################################  Adjacency matrix  ################################ 

    if not os.path.exists(adj_mx_path):
        generate_adj_matrix(args)

    adj_mx = load_adj(adj_mx_path, args.adjtype)

    train_dl, valid_dl, dataset = get_train_valid_loader(config_gwnet, 
                                                         args, 
                                                         data, 
                                                         target,
                                                         config["MODELS"]["split"])
    check_shape_dataloaders(train_dl, valid_dl)

    num_nodes = dataset.data.shape[-1]                         
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    metrics_recorder = MetricsRecorder()

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]
    if args.aptonly:
        supports = None

    engine = trainer(config_gwnet, scaler, args, num_nodes, supports, checkpoint_path)
    start_epoch = 0 if checkpoint_path is None else engine.checkp_epoch
    early_stopping = EarlyStopping(config_gwnet, logger, verbose=True)

    ################################  Training  ################################ 

    logger.info("Starting training...")
    
    his_loss, val_time, train_time = [],[],[]


    for epoch in range(start_epoch+1, config_gwnet.epochs+1):

        train_loss, train_mape, train_rmse = [],[],[]
        valid_loss, valid_mape, valid_rmse = [],[],[]

        epoch_records = gwnet_train_loop(config_gwnet, engine, train_dl)
        train_loss.append(epoch_records["loss"])
        train_mape.append(epoch_records["mape"])
        train_rmse.append(epoch_records["rmse"])
        metrics_recorder.add_train_metrics(epoch_records, epoch)
        
    ################################  Validation ###############################
        
        epoch_records = gwnet_val_loop(config_gwnet, engine, valid_dl)
        valid_loss.append(epoch_records["loss"])
        valid_mape.append(epoch_records["mape"])
        valid_rmse.append(epoch_records["rmse"])
        metrics_recorder.add_val_metrics(epoch_records)
        
        mtrain_loss = np.mean(train_loss)
        mtrain_rmse = np.mean(train_rmse)  
        mvalid_loss = np.mean(valid_loss)
        mvalid_rmse = np.mean(valid_rmse)
        # his_loss.append(mvalid_loss)

        save_figures(args=args, epoch=epoch-start_epoch, 
                     path=img_path, metrics_recorder=metrics_recorder)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}'
        logger.info(log.format(epoch, mtrain_loss, mtrain_rmse, mvalid_loss, mvalid_rmse))
        
        model_dict = {
            'epoch': epoch,
            'state_dict': engine.model.state_dict(),
            'optimizer': engine.optimizer.state_dict(),
            "lr_sched": engine.scheduler.state_dict()
        }

        early_stopping(mvalid_loss, model_dict, epoch, checkpoint_dir)
        update_tensorboard_scalars(writer, metrics_recorder)
        
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    sys.exit(0)

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(checkp_path +"/checkpoints_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))

    print(engine.model)


    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    test_loss, predictions, targets = engine.test(dataloader["test_loader"])

    yhat = torch.cat(predictions,dim=0).squeeze()
    yhat = yhat[:realy.size(0),...]

    output = open(dictionary['predFile'], 'wb')
    pickle.dump(yhat.cpu().detach().numpy(), output)
    output.close()

    target = open(dictionary['targetFile'], 'wb')
    pickle.dump(realy.cpu().detach().numpy(), target)
    target.close()

    logger.success("Training finished")
    logger.info("The valid loss on best model is {}".format(str(round(his_loss[bestid],4))))

    amae = []
    amape = []
    armse = []
    for i in range(args.seq_length):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = metric(pred,real) 
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        logger.info(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    logger.info(log.format(args.forecast, np.mean(amae),np.mean(amape),np.mean(armse)))
    torch.save(engine.model.state_dict(), checkp_path +"/checkpoints_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")
  


