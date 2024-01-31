#!/usr/bin/env python
# coding: utf-8


import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import numpy as np
import os
import xarray as xr
from utils.ndvi_functions import convert_ndvi_tofloat

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


def load_landsaf_ndvi(path:str, crop_area:bool = True)->xr.Dataset:
    ds = xr.open_zarr(path)
    import pandas as pd
    # Convert nanoseconds to datetime objects
    ds['time'] = pd.to_datetime(ds['time'], unit='ns')
    # Extract the date part
    ds['time'] = ds['time'].dt.floor('D')

    if crop_area is True:
        from utils.function_clns import subsetting_pipeline
        ds = subsetting_pipeline(ds)

    ds["ndvi_10"] = convert_ndvi_tofloat(ds.ndvi_10)
    # ds["ndvi_10"] = ds.ndvi_10/255
    return ds 


