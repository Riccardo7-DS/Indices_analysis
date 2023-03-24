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
import yaml
from p_drought_indices.functions.function_clns import load_config, cut_file, open_xarray_dataset
from p_drought_indices.functions.ndvi_functions import clean_outliers, compute_ndvi, clean_ndvi, downsample
from p_drought_indices.vegetation.NDVI_indices import compute_vci
from xarray import DataArray
from tqdm.auto import tqdm



def extract_apply_cloudmask(ds, ds_cl, resample=False, include_water =True,downsample=False):
    
    def checkVars(ds, var):
        assert var  in ds.data_vars, f"Variable {var} not in dataset"

    [checkVars(ds, var)  for var in ["channel_1","channel_2","ndvi"]]

    ### normalize time in order for the two datasets to match
    ds_cl['time'] = ds_cl.indexes['time'].normalize()
    ds['time'] = ds.indexes['time'].normalize()
    
    if resample==True:
    #### reproject cloud mask to base dataset
        reproj_cloud = ds_cl['cloud_mask'].rio.reproject_match(ds['ndvi'])
        ds_cl_rp = reproj_cloud.rename({'y':'lat', 'x':'lon'})

    else:
        ds_cl_rp = ds_cl

    ### apply time mask where values are equal to 1, hence no clouds over land, 0= no cloud over water
    if include_water==True:
        ds_subset = ds.where((ds_cl_rp==1)|(ds_cl_rp==0)) #ds = ds.where(ds.time == ds_cl.time)
    else:
        ds_subset = ds.where(ds_cl_rp==1)
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
    

def clean_water(ds, ds_cl):
    ds_cl['time'] = ds_cl.indexes['time'].normalize()
    ds['time'] = ds.indexes['time'].normalize()
    return ds.where(ds_cl=1)


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

def prepare(ds):
        ds.rio.write_crs("epsg:4326", inplace=True)
        ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
        return ds

def convert_to_raster(list_files:list, target_darray:xr.DataArray):
    for path in tqdm(list_files, desc="Files"):
        ds = xr.open_dataset(path)
        ds = prepare(ds)
        ds_ndvi = ds["ndvi"].rio.reproject_match(target_darray).rename({"x":"lon","y":"lat"})
        ds_ndvi = ds_ndvi.to_dataset()
        channel_1 = ds["channel_1"].rio.reproject_match(target_darray).rename({"x":"lon","y":"lat"})
        channel_2 = ds["channel_2"].rio.reproject_match(target_darray).rename({"x":"lon","y":"lat"})
        ds_ndvi = ds_ndvi.assign(channel_1 = channel_1, channel_2= channel_2)
        name = path.split("/")[-1]
        ds_ndvi.to_netcdf(os.path.join(ndvi_dir, "reprojected", name))

def computation_pipeline(CONFIG_PATH, countries = ['Ethiopia','Kenya','Somalia']):
    config = load_config(CONFIG_PATH)
    ndvi_dir = config['NDVI']['ndvi_prep']

    def concat_datasets(list_files_1, list_files_2):
        ds_1 = open_xarray_dataset(list_files_1)
        ds_2 = open_xarray_dataset(list_files_2)
        return xr.concat([ds_2, ds_1], dim="time")
    
    new_dir = os.path.join(ndvi_dir,"new_process")
    list_files_2 = [os.path.join(new_dir, f) for f in os.listdir(new_dir) if f.endswith(".nc")]
    ds_2 = xr.open_dataset(list_files_2[0])

    new_dir = os.path.join(ndvi_dir,"old_process")
    list_files_1 = [os.path.join(new_dir, f) for f in os.listdir(new_dir) if f.endswith(".nc")]
    ds_1 = xr.open_dataset(list_files_1[0])

    #convert_to_raster(list_files_1, target_darray=prepare(ds_2)["ndvi"])

    new_dir = os.path.join(ndvi_dir,"reprojected")
    list_files_3 = [os.path.join(new_dir, f) for f in os.listdir(new_dir) if f.endswith(".nc")]
    ds = concat_datasets(list_files_3, list_files_2)

    ### load and prepare cloudmask
    base_dir = 'nc_files/new/ndvi_mask.nc'
    cl_df = xr.open_dataset(os.path.join(config['NDVI']['cloud_path'], base_dir), chunks={"time":"250MB"})
    cl_df = cl_df.sel(time=slice(cl_df['time'].min(), '2020-12-31'))

    ds_cl = prepare(cl_df)
    ds_cl = ds_cl["cloud_mask"].rio.reproject_match(prepare(ds_2)["ndvi"]).rename({'y':'lat', 'x':'lon'})
    print("Succesfully loaded cloud mask")

    ### clean ndvi and apply cloudmask
    xr_df = clean_outliers(ds)
    xr_df = xr_df.sortby("time")
    xr_df.to_netcdf(os.path.join(config['NDVI']['ndvi_path'], 'ndvi_no_out.nc'))

    shapefile_path = config['SHAPE']['africa']
    gdf = gpd.read_file(shapefile_path)
    subset = gdf[gdf.ADM0_NAME.isin(countries)]

    ds_n = cut_file(xr_df, subset)
    ds_cl = cut_file(ds_cl, subset)
    print("Starting applying cloudmask on dataset...")
    mask_clouds, res_xr = extract_apply_cloudmask(ds_n, ds_cl, include_water=False)
    mask_clouds.to_netcdf(os.path.join(ndvi_dir, "final_ndvi.nc"))

    ### apply whittaker filter
    from p_drought_indices.vegetation.cloudmask_cleaning import apply_whittaker

    print("Applying Whittaker filter...")
    result = apply_whittaker(mask_clouds['ndvi'])
    result = clean_outliers(result)
    result = clean_water(ds, ds_cl)
    result.to_netcdf(os.path.join(config['NDVI']['ndvi_path'], 'smoothed_ndvi.nc'))

    print("Computing VCI index...")
    vci = compute_vci(result)
    vci.to_netcdf(os.path.join(config['NDVI']['ndvi_path'], 'vci_1D.nc'))
    print("Success") 

if __name__=="__main__":
    CONFIG_PATH = "config.yaml"
    computation_pipeline(CONFIG_PATH)
    







