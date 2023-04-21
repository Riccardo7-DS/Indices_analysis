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
from p_drought_indices.functions.function_clns import prepare, load_config, cut_file, open_xarray_dataset, crop_get_thresh
from p_drought_indices.functions.ndvi_functions import apply_whittaker, extract_apply_cloudmask, clean_outliers, compute_ndvi, clean_ndvi, downsample, clean_water
from p_drought_indices.vegetation.NDVI_indices import compute_vci
from xarray import DataArray
from p_drought_indices.ancillary_vars.FAO_HWSD import get_water_cover
from tqdm.auto import tqdm
import xskillscore as xs

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

def compute_correlation(dataarray1, dataarray2):
    ndvi_= dataarray1.chunk(dict(time=-1))
    mask_= dataarray2.chunk(dict(time=-1))
    xs.pearson_r(ndvi_, mask_, dim='time', skipna=True).plot(cmap='YlGn')
    plt.show()


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

def concat_datasets(list_files_1, list_files_2):
    ds_1 = open_xarray_dataset(list_files_1)
    ds_2 = open_xarray_dataset(list_files_2)
    return xr.concat([ds_2, ds_1], dim="time")

def computation_pipeline(CONFIG_PATH, countries = ['Ethiopia','Kenya','Somalia'], concatenate=False, subset=False):
    config = load_config(CONFIG_PATH)
    ndvi_dir = config['NDVI']['ndvi_prep']
    
    new_dir = os.path.join(ndvi_dir,"new_process")
    list_files_2 = [os.path.join(new_dir, f) for f in os.listdir(new_dir) if f.endswith(".nc")]
    ds_2 = xr.open_dataset(list_files_2[0])

    if concatenate ==True:
        new_dir = os.path.join(ndvi_dir,"reprojected")
        list_files_3 = [os.path.join(new_dir, f) for f in os.listdir(new_dir) if f.endswith(".nc")]

        new_dir = os.path.join(ndvi_dir,"old_process")
        list_files_1 = [os.path.join(new_dir, f) for f in os.listdir(new_dir) if f.endswith(".nc")]
        ds_1 = xr.open_dataset(list_files_1[0])
        print("Starting converting raster to destination dataset")
        convert_to_raster(list_files_1, target_darray=prepare(ds_2)["ndvi"])
        ds = concat_datasets(list_files_3, list_files_2)
    else:
        ds = xr.open_mfdataset(list_files_2)

    ### load and prepare cloudmask
    base_dir = 'nc_files/new/ndvi_mask.nc'
    cl_df = xr.open_dataset(os.path.join(config['NDVI']['cloud_path'], base_dir))
    cl_df = cl_df.sel(time=slice(cl_df['time'].min(), '2020-12-31'))

    if subset==True:
        days = 365
        ds = ds.isel(time=slice(0,days))
        cl_df = cl_df.isel(time=slice(0,days))

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

    print("Applying Whittaker filter...")
    result = apply_whittaker(mask_clouds['ndvi'])

    ### Cleaning water bodies
    ds_cover = get_water_cover(CONFIG_PATH, xr_df =result.to_dataset(), countries=countries, invert=False)
    res_ds = result.where(ds_cover==0)
    res_ds = clean_outliers(res_ds.to_dataset())
    res_ds.to_netcdf(os.path.join(config['NDVI']['ndvi_path'], 'smoothed_ndvi_1.nc'))

    print("Computing VCI index...")
    vci = compute_vci(res_ds["ndvi"])
    vci.to_netcdf(os.path.join(config['NDVI']['ndvi_path'], 'vci_1D.nc'))

    import numpy as np
    res_ds = crop_get_thresh(vci["ndvi"]).to_dataset()
    res_ds.to_netcdf(os.path.join(config['NDVI']['ndvi_path'], 'percentage_ndvi.nc'))
    print("Success") 

if __name__=="__main__":
    CONFIG_PATH = "config.yaml"
    computation_pipeline(CONFIG_PATH, subset=False)







