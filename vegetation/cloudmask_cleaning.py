#!/usr/bin/env python
# coding: utf-8

import geopandas as gpd
from datetime import datetime, timedelta
import shutil
from shapely.geometry import Polygon, mapping
from satpy import DataQuery
from pyresample.geometry import SwathDefinition
from satpy import Scene
import xarray as xr
import geopandas as gpd
from satpy.dataset import combine_metadata
import matplotlib.pyplot as plt
from glob import glob
import os
#import datetime as datetime
import time
import yaml
import numpy as np
import re
import xskillscore as xs

def downsample(ds):
    monthly = ds.resample(time='5D', skipna=True).mean() #### Change here to change the timeframe over which to make the data imputation
    return monthly

def clean_ndvi(ds):
    ds = ds.where('ndvi'!=0.00)
    return ds

def load_config(CONFIG_PATH):
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)
        return config

def extract_apply_cloudmask(ds, ds_cl):
    ### normalize time in order for the two datasets to match
    ds_cl['time'] = ds_cl.indexes['time'].normalize()
    ds['time'] = ds.indexes['time'].normalize()

    ### apply time mask
    ds = ds.where(ds.time == ds_cl.time)

    ### apply mask and downsample to 5 days
    res_xr = ds.where(ds_cl.cloud_mask<2)
    res_xr_p = downsample(res_xr)
    
    ### mask all the values equal to 0 (clouds)
    mask_clouds = clean_ndvi(ds)
    mask_clouds_p = downsample(mask_clouds)

    return mask_clouds_p, res_xr_p,  mask_clouds, res_xr ### return 1) cleaned dataset with clouds 
                                                         ### 2) imputation with max over n days
                                                         ### 3) cloudmask dataset original sample
                                                         ### 4) cloudmask dataset downsampled


def plot_cloud_correction(mask_clouds_p, res_xr_p, time):
    mask_clouds_p['ndvi'].sel(time=time,  method = 'nearest').plot()
    plt.title('NDVI image with cloud mask')
    plt.show()
    res_xr_p['ndvi'].sel(time=time,  method = 'nearest').plot()
    plt.title('NDVI image with max pixel value cloud correction')
    plt.show()


def compute_difference(mask_clouds, mask_clouds_p, res_xr_p, res_xr, time):
    #### Difference in absolute value
    res = mask_clouds_p['ndvi'].sel(time=time,  method = 'nearest') - res_xr_p['ndvi'].sel(time=time,  method = 'nearest')
    diff = mask_clouds['ndvi'] - res_xr['ndvi']
    diff_p = mask_clouds_p['ndvi'] - res_xr_p['ndvi']
    abs(diff).mean('time', skipna=True).plot()
    plt.show()
    abs(diff_p).mean('time').plot()
    plt.show()

def compute_correlation(mask_clouds_p, res_xr_p):
    ndvi_= res_xr_p['ndvi'].chunk(dict(time=-1))
    mask_= mask_clouds_p['ndvi'].chunk(dict(time=-1))
    xs.pearson_r(ndvi_, mask_, dim='time', skipna=True).plot()
    plt.show()


if __name__=="__main__":

    CONFIG_PATH = r"./config.yaml"
    config = load_config(CONFIG_PATH)

    ### MSG dataset
    ndvi_dir = config['NDVI']['ndvi_path'] 
    ds = xr.open_mfdataset(os.path.join(ndvi_dir,'*.nc'))

    #### Cloudmask dataset
    cloud_path = config['NDVI']['cloud_path']
    ds_cl = xr.open_mfdataset(os.path.join(cloud_path,'*.nc'))
    
    mask_clouds_p, res_xr_p,  mask_clouds, res_xr = extract_apply_cloudmask(ds, ds_cl)

    mask_clouds_p['ndvi'].plot()
    plt.show()

    time = '2009-12-05'
    ### Plot and compare methods
    #plot_cloud_correction(mask_clouds_p, res_xr_p, time)

    #### Check correlation
    #compute_correlation(mask_clouds_p, res_xr_p)
    
    ### Export dataset
    output_file = os.path.join(ndvi_dir, 'processed_msg.nc')
    mask_clouds_p.to_netcdf(output_file)


