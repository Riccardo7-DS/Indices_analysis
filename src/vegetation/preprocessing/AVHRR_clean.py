#!/usr/bin/env python
# coding: utf-8


import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import numpy as np
import os

def cut_file(xr_df, gdf):
    xr_df.rio.write_crs("epsg:4326", inplace=True)
    clipped = xr_df.rio.clip(gdf.geometry.apply(mapping), gdf.crs, drop=True)
    if 'crs' in clipped.data_vars:
        clipped = clipped.drop('crs')
    return clipped

def downsample(ds):
    monthly = ds.resample(time='5D', skipna=True).mean() #### Change here to change the timeframe over which to make the data imputation
    return monthly

def clean_ndvi(ds):
    ds = ds.where('ndvi'!=0)
    return ds

def process_eumetsat_ndvi(path:str, filename:str, complevel:int = 9):
    chunks = {"time":200, "lat":50, "lon":50}
    ds = xr.open_dataset(os.path.join(path, "ndvi_eumetsat.nc"), chunks=chunks)
    ds = ds.rename({"Band1":"ndvi"})

    compression = {"ndvi" :{'zlib': True, "complevel":complevel}}
    from utils.function_clns import config
    ds.to_netcdf(os.path.join(config["NDVI"]["ndvi_path"],filename),
                 encoding=compression)


