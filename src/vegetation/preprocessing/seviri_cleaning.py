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
from utils.function_clns import prepare, load_config, cut_file, open_xarray_dataset, crop_get_thresh
from utils.xarray_functions import apply_whittaker, drop_water_bodies_esa, extract_apply_cloudmask, clean_outliers, compute_ndvi, clean_ndvi, downsample, clean_water
from vegetation.analysis.indices import compute_vci
from xarray import DataArray
from ancillary.FAO_HWSD import get_water_cover
from tqdm.auto import tqdm
import xskillscore as xs
from typing import Literal, Union
import logging


def extract_clean_cloudmask_dataset(config_file, path, other_path):
    from utils.function_clns import subsetting_loop
    new_path = config_file["NDVI"]["cloud_path"]
    dest_path = os.path.join(new_path, "processed")
    subsetting_loop(new_path, delete_grid_mapping=True)

    from utils.xarray_functions import add_time

    list_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".nc")]
    new_path = os.path.join(path, "time")

    for file in list_files:
        name = str(file).split("/")[-1]
        ds = xr.open_dataset(file)
        ds_new = add_time(ds)
        ds_new.to_netcdf(os.path.join(new_path, name))

    list_files_1 = [os.path.join(other_path, f) for f in os.listdir(other_path) if f.endswith(".nc")]
    ds = prepare(xr.open_mfdataset(list_files_1))
    ds_2 = prepare(xr.open_dataset(list_files[0]))
    new_raster = ds["cloud_mask"].rio.reproject_match(ds_2["cloud_mask"]).rename({"x":"lon","y":"lat"})

    list_2 = [os.path.join(new_path, f) for f in os.listdir(new_path) if f.endswith(".nc")]
    ds_3 = prepare(xr.open_mfdataset(list_2))
    ds_3 = ds_3.drop("crs")
    new_raster = new_raster.drop("crs")
    ds_cl = xr.combine_by_coords([ds_3, new_raster.to_dataset()], combine_attrs="drop_conflicts")
    base_dir = 'nc_files/new/ndvi_mask.nc'
    ds_cl.to_netcdf(os.path.join(config_file['NDVI']['cloud_path'], base_dir))


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


def computation_pipeline(CONFIG_PATH, countries = ['Ethiopia','Kenya','Somalia'],
                         drop_water: Union[Literal["FAO"],Literal["ESA"],None]=None, 
                         concatenate=False, 
                         subset=False):
    
    from utils.function_clns import config

    logging.basicConfig(filename="./log.txt", level=logging.DEBUG)
    logging.info(f"Starting computing SEVIRI NDVI...")
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


    print(f"1) Dataset has dimensions {ds.sizes}")
    logging.info(f"1) Dataset has dimensions {ds.sizes}")

    ### load and prepare cloudmask
    base_dir = 'nc_files/new/ndvi_mask.nc'
    cl_df = xr.open_dataarray(os.path.join(config['NDVI']['cloud_path'], base_dir))
    cl_df = cl_df.sel(time=slice(cl_df['time'].min(), '2020-12-31'))

    if subset==True:
        days = 365*2
        ds = ds.isel(time=slice(0,days))
        cl_df = cl_df.isel(time=slice(0,days))

    #ds_cl = prepare(cl_df)
    #ds_cl = ds_cl["cloud_mask"].rio.reproject_match(prepare(ds_2)["ndvi"]).rename({'y':'lat', 'x':'lon'})
    print(f"Succesfully loaded cloud mask with dimensions {cl_df.sizes}")
    logging.info(f"Succesfully loaded cloud mask with dimensions {cl_df.sizes}")

    ### clean ndvi and apply cloudmask
    xr_df = clean_outliers(ds)
    xr_df = xr_df.sortby("time")
    print(f"2) Dataset has dimensions {xr_df.sizes}")
    logging.info(f"2) Dataset has dimensions {xr_df.sizes}")
    xr_df.to_netcdf(os.path.join(config['NDVI']['ndvi_path'], 'ndvi_no_out.nc'))

    shapefile_path = config['SHAPE']['africa']
    gdf = gpd.read_file(shapefile_path)
    subset = gdf[gdf.ADM0_NAME.isin(countries)]

    ds_n = cut_file(xr_df, subset)
    ds_cl = cut_file(cl_df, subset)
    print(f"3) Dataset has dimensions {ds_n.sizes}")
    logging.info(f"3) Dataset has dimensions {ds_n.sizes}")

    ### Cleaning water bodies
    if drop_water== "ESA":
        print(f"Dropping water bodies with {drop_water} Land map")
        res_ds = drop_water_bodies_esa(CONFIG_PATH, config, ds_n)

    elif drop_water == "FAO":
        print(f"Dropping water bodies with {drop_water} Land map")
        ds_cover = get_water_cover(CONFIG_PATH, xr_df =ds_n, countries=countries, invert=False)
        res_ds = ds_n["ndvi"].where(ds_cover==0)
        res_ds = clean_outliers(res_ds.to_dataset())

    print(f"4) Dataset has dimensions {res_ds.sizes}")
    logging.info(f"4) Dataset has dimensions {res_ds.sizes}")

    print("Starting applying cloudmask on dataset...")
    mask_clouds, res_xr = extract_apply_cloudmask(res_ds, ds_cl, include_water=True)
    res_xr.to_netcdf(os.path.join(ndvi_dir, "final_ndvi.nc"))

    print("Applying Whittaker filter...")
    result = apply_whittaker(res_xr['ndvi']).to_dataset()
    res = clean_outliers(result)
    res.to_netcdf(os.path.join(config['NDVI']['ndvi_path'], 'smoothed_ndvi.nc'))

    print("Computing VCI index...")
    vci = compute_vci(res["ndvi"]).to_dataset()
    vci.to_netcdf(os.path.join(config['NDVI']['ndvi_path'], 'vci_1D.nc'))

    import numpy as np
    res_ds = crop_get_thresh(vci["ndvi"]).to_dataset()
    res_ds.to_netcdf(os.path.join(config['NDVI']['ndvi_path'], 'percentage_ndvi.nc'))
    print("Success") 

def apply_smoother_only(config_file:dict, lambda_par:int):
    compression_level = 5
    encoding = {'ndvi': {'zlib': True, 'complevel': compression_level}}
    res_xr = xr.open_dataset(os.path.join(config_file["NDVI"]["ndvi_prep"], "final_ndvi.nc"))
    print(f"Applying Whittaker filter with a lambda parameter of {lambda_par}...")
    result = apply_whittaker(res_xr['ndvi'],lambda_par=lambda_par).to_dataset()
    result.to_netcdf(os.path.join(config_file['NDVI']['ndvi_path'], f'smoothed_ndvi_lambda_{lambda_par}.nc'),\
                     encoding=encoding)
    print("Success")

    







