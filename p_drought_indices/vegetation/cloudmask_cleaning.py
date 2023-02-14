#!/usr/bin/env python
# coding: utf-8

import geopandas as gpd
from datetime import datetime, timedelta
import shutil
from shapely.geometry import Polygon, mapping
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from glob import glob
import os
#import datetime as datetime
import time
import yaml
import numpy as np
import re
import xskillscore as xs
import yaml
from p_drought_indices.functions.function_clns import load_config, cut_file
from p_drought_indices.functions.ndvi_functions import compute_ndvi, clean_ndvi, downsample
from xarray import DataArray


def extract_apply_cloudmask(ds, ds_cl, downsample=False):
    ### normalize time in order for the two datasets to match
    ds_cl['time'] = ds_cl.indexes['time'].normalize()
    ds['time'] = ds.indexes['time'].normalize()
    
    #### reproject cloud mask to base dataset
    reproj_cloud = ds_cl['cloud_mask'].rio.reproject_match(ds['channel_1'])
    ds_cl_rp = reproj_cloud.rename({'y':'lat', 'x':'lon'})

    ### apply time mask where values are equal to 1, hence no clouds
    ds_subset = ds.where(ds_cl_rp==1) #ds = ds.where(ds.time == ds_cl.time)
    ### recompute corrected ndvi
    res_xr = compute_ndvi(ds_subset)

    ### mask all the values equal to 0 (clouds)
    mask_clouds = clean_ndvi(ds)
    ### recompute corrected ndvi
    mask_clouds = compute_ndvi(mask_clouds)

    #### downsample to 5 days
    if downsample==True:
        "Starting downsampling the Dataset"
        res_xr_p = downsample(res_xr)
        #### downsampled df
        mask_clouds_p = downsample(mask_clouds)
        return mask_clouds_p, res_xr_p,  mask_clouds, res_xr ### return 1) cleaned dataset with clouds 
                                                         ### 2) imputation with max over n days
                                                         ### 3) cloudmask dataset original sample
                                                         ### 4) cloudmask dataset downsampled
    else:
        return mask_clouds, res_xr

def apply_whittaker(datarray:DataArray, prediction="P1D", time_dim="time"):
    from fusets import WhittakerTransformer
    from fusets._xarray_utils import _extract_dates, _output_dates, _topydate
    result = WhittakerTransformer().fit_transform(datarray.load(),smoothing_lambda=1,time_dimension=time_dim, prediction_period=prediction)
    dates = _extract_dates(datarray)
    expected_dates = _output_dates(prediction,dates[0],dates[-1])
    datarray['time'] = datarray.indexes['time'].normalize()
    datarray = datarray.assign_coords(time = datarray.indexes['time'].normalize())
    result['time'] = [np.datetime64(i) for i in expected_dates]
    return result

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

def subsetting_pipeline(CONFIG_PATH, xr_df, countries = ['Ethiopia','Kenya', 'Somalia']):
    CONFIG_PATH = r"./config.yaml"
    config = load_config(CONFIG_PATH)
    shapefile_path = config['SHAPE']['africa']
    gdf = gpd.read_file(shapefile_path)
    subset = gdf[gdf.ADM0_NAME.isin(countries)]
    return cut_file(xr_df, subset)

def compute_correlation(dataarray1, dataarray2):
    ndvi_= dataarray1.chunk(dict(time=-1))
    mask_= dataarray2.chunk(dict(time=-1))
    xs.pearson_r(ndvi_, mask_, dim='time', skipna=True).plot(cmap='YlGn')
    plt.show()



if __name__=="__main__":

    base_dir = r'D:\shareVM\MSG\cloudmask\processed_clouds\batch_2\nc_files\new\ndvi_mask.nc'
    #print([f for f in os.listdir(base_dir) if f.endswith('.nc')])
    chunks = {"lat": -1, "lon": -1, "time": 12}
    cl_df = xr.open_mfdataset(base_dir, chunks=chunks)

    ndvi_dir = r'D:\shareVM\MSG\msg_data\batch_2\processed\*.nc'
    xr_df = xr.open_mfdataset(ndvi_dir, chunks=chunks)

    countries = ['Ethiopia','Kenya']
    
    ds = subsetting_pipeline(xr_df, countries)
    ds_cl = subsetting_pipeline(cl_df, countries)
    mask_clouds_5D, res_xr_5D,  mask_clouds_1D, res_xr_1D = extract_apply_cloudmask(ds, ds_cl)

    #### Check correlation
    #compute_correlation(mask_clouds_p, res_xr_p)
    
    ### Export dataset
    #output_dir = r'D:\shareVM\MSG\msg_data\processed'
    #mask_clouds_5D.to_netcdf(os.path.join(output_dir, 'mask_clouds_5D_processed_msg.nc'))
    #res_xr_5D.to_netcdf(os.path.join(output_dir, 'res_xr_5D_processed_msg.nc'))
    #mask_clouds_1D.to_netcdf(os.path.join(output_dir, 'mask_clouds_1D_processed_msg.nc'))
    #res_xr_1D.to_netcdf(os.path.join(output_dir, 'res_xr_1D_processed_msg.nc'))

    vci = xr.open_dataset(r'D:\shareVM\MSG\msg_data\processed\vci_1D.nc')
    svi = xr.open_dataset(r'D:\shareVM\MSG\msg_data\processed\svi_1D.nc')
    







