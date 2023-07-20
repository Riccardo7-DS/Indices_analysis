import os.path as osp

import argparse
import numpy as np
import torch
from torch import nn
from torchvision import transforms
import random
import matplotlib.pyplot as plt
import pickle as pkl
from p_drought_indices.analysis.DeepLearning.dataset import MyDataset
import torch_geometric.transforms as T
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import train_test_split_edges
import sys
from sklearn.cluster import KMeans
import networkx as netx
import scipy
#import hydrostats as Hydrostats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()

        self.encoder=nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),    
            nn.Linear(16, latent_dim),
            nn.ReLU(True),   
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(True),    
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, input_dim),
            nn.Sigmoid(),     
        )

    def forward(self, features):
        x = self.encoder(features)
        x = self.decoder(x)

        return x

def add_noise(x):
    noise = torch.randn(x.size()) * 1e-6
    noisy_img = x + noise
    return noisy_img

def genAdjacency(featureMat, perc=98,smeasure='euclidean'):
    """Generate pairwise distances
    1/20: note pytorchgeometric can import distance directly
    Parameters
    ----------
    featureMat: matrix of features in columsn
    perc: threshold for cutoff
    smeasure: similarity measure to use for defining the adjacency 
    """
    from scipy.spatial.distance import cdist
    import sklearn.preprocessing as skp

    assert(smeasure in ["euclidean","mahalanobis","jensenshannon", "correlation"])
    D = cdist(featureMat, featureMat,smeasure)

    np.fill_diagonal(D,1)
    D = np.reciprocal(D) #get inverse distance
    #should we do this?
    row_normalize=False
    if row_normalize:
        #row normalization
        np.fill_diagonal(D,0) #exclude diagonal in the sum 
        sum_of_rows = D.sum(axis=1)
        D = D / sum_of_rows[:, np.newaxis]
        np.fill_diagonal(D,1.0)    
    else:
        #do max/min     
        np.fill_diagonal(D,0) #exclude diagonal
        amin,amax = (np.min(D),np.max(D)   )  
        D = (D-amin)/(amax-amin)        
        np.fill_diagonal(D,1.0)    
        
    #calculate histogram
    Dup=np.triu(D)
    pnt = np.percentile(Dup,perc)
    print (f'{perc} percentile is {pnt}')
    #these are distance so smaller ones are closer
    D[D<pnt] = 0.0
    A = np.copy(D)
    A[A>0.0] = 1.0
    id = np.where(A==1.0)[0]
    print ('percent connected {:.2f}'.format(len(id)/(A.shape[0]*A.shape[0])*100.0))

    return A,D

def train(df, reTrain=False, latent_dim  = 4):
    """Train autoencoder
    Parameters
    ---------
    latent_dim, dimension of the latent variable
    """
    dataset = MyDataset(df.to_numpy(),transform=None)
    if reTrain:
        dataloader = DataLoader(dataset, batch_size=32, 
                        shuffle=True,drop_last=False,num_workers=4)

    testdataloader = DataLoader(dataset, batch_size=1, 
                        shuffle=False,drop_last=False,num_workers=4)

    model = AE(input_dim=df.shape[1], latent_dim=latent_dim)
    model.to(device)
    model_path = f'models/camels_ae_{latent_dim}.pth'

    if reTrain:
        #train
        lr = 0.005
        nEpoch=100
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.MSELoss()


        model.train()    
        for i in range(nEpoch):
            loss=0.0
            for x in dataloader:
                x = x.view(x.size(0), -1)    
        
                optimizer.zero_grad()
                x= x.to(device)
                outputs = model(x)
                train_loss = criterion(x,outputs)
                train_loss.backward()

                loss+=train_loss.item()
                optimizer.step()
            loss = loss/len(dataloader)
            print (f'Epoch {i}, loss={loss:.4f}')
        torch.save(model.state_dict(), model_path)
    else:
        model.load_state_dict(torch.load(model_path))
        
    model.eval()
    latentMat = np.zeros((df.shape[0],latent_dim))
    for idx,x in enumerate(testdataloader):
        x = x.view(x.size(0), -1)  
        x=x.to(device)
        out = model.encoder(x)

        latentMat[idx,:] = out.data.cpu().numpy().squeeze()
    return latentMat

