from p_drought_indices.functions.function_clns import load_config, cut_file, subsetting_pipeline
import xarray as xr 
import pandas as pd
import yaml
from datetime import datetime, timedelta
import shutil
from shapely.geometry import Polygon, mapping
import geopandas as gpd
import matplotlib.pyplot as plt
from glob import glob
import os
#import datetime as datetime
import time
import numpy as np
import re

import os

CONFIG_PATH = r"config.yaml"

def reproject_raster(veg_array, target_array):
    #veg_array.rio.set_spatial_dims(x_dim='lat', y_dim='lon', inplace=True)
    veg_array.rio.write_crs("epsg:4326", inplace=True)
    return veg_array.rio.reproject_match(target_array)

def prep_dataset(ds, transpose=False):
        if transpose==True:
            ds = ds.transpose('time', 'lon', 'lat')
        ds.rio.write_crs("epsg:4326", inplace=True)
        ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
        return ds

def swap_change_attrs(xr_df):
    xr_df = xr_df.rename({'lat':'lon','lon':'lat'})
    attrs_1 = xr_df['lat'].attrs
    attrs_2 = xr_df['lon'].attrs
    xr_df['lon'].attrs = attrs_1
    xr_df['lat'].attrs = attrs_2
    return xr_df

def print_dims(xr_df):
    vars = list(xr_df.data_vars)
    print('The dataset dimensions are:', xr_df.dims)
    print('The dataset coords are:', xr_df.coords)
    print('The datarray dimensions are:', xr_df[vars[0]].dims)
    print('The datarray coords are:', xr_df[vars[0]].coords)

def plot_first(xr_df):
    vars = list(xr_df.data_vars)
    xr_df[vars[0]].isel(time=0).plot(x='lon')
    plt.show()


def gen_empty_dataset(df, var):
    #pull variables from netcdf file
    latitude=df.latitude
    longitude=df.longitude
    time=df.time
    var_ = df[var]

    #new xarray dataset
    dataset = xr.Dataset(
        coords={
            "time": (time),"lat": (latitude),"lon": (longitude)
            },
        data_vars={
            var:(( "time", "lat","lon"),var_),
            }
    )
    return dataset

def generate_dataset():
    time_slice = slice('2005-01','2010-01-03')

    #### load NASA data on evapotransp and wind
    nasa_X = xr.open_dataset(r'D:\shareVM\MERRA2\nasapower_vars.nc')
    ds_vars = subsetting_pipeline(CONFIG_PATH, nasa_X).sel(time=time_slice)
    ds_vars = ds_vars.drop('elevation')
    ds_vars_sw = swap_change_attrs(ds_vars)

    #### load vegetation data
    veg_df = xr.open_dataset(r'D:\shareVM\MSG\msg_data\processed\smoothed_ndvi.nc')
    veg_df = veg_df[['lon','lat','time','ndvi']]

    repr_ds = reproject_raster(prep_dataset(veg_df['ndvi'],transpose=True), prep_dataset(ds_vars_sw, transpose=True)).rename({'x':'lat','y':'lon'})
    repr_ds = repr_ds.transpose('time','lat','lon')
    
    #### load precipitation data
    file_spi = r'D:\shareVM\CHIRPS\daily\SPI\CHIRPS_spi_gamma_180.nc'
    ds_precp = xr.open_dataset(file_spi).sel(time=time_slice)
    repr_prep = reproject_raster(prep_dataset(ds_precp['spi_gamma_180'] , transpose=True), prep_dataset(ds_vars_sw, transpose=True)).rename({'x':'lat','y':'lon'}).to_dataset()

    #### load different latencies of precipitation data
    for spi in ['30','60','90']:
        spi_name = 'spi_gamma_{}'.format(spi)
        file_spi = r'D:\shareVM\CHIRPS\daily\SPI\CHIRPS_spi_gamma_{}.nc'.format(spi)
        ds_precp = xr.open_dataset(file_spi).sel(time=time_slice)
        temp_ds = reproject_raster(prep_dataset(ds_precp[spi_name] , transpose=True), prep_dataset(ds_vars_sw, transpose=True)).rename({'x':'lat','y':'lon'})
        repr_prep[spi_name] = temp_ds

    #### merge data
    xr_df = xr.merge([repr_prep, ds_vars_sw, repr_ds])
    print(xr_df.data_vars)
    xr_df = xr_df.sel(time=time_slice)

    res_w = xr_df.resample(time='1W').mean()
    res_w = subsetting_pipeline(res_w)

    df = res_w.drop(['crs','spatial_ref']).to_dataframe()
    df = df.dropna(how='all')

    from p_drought_indices.ancillary_vars.FAO_HWSD import get_soil_vars

    ### load ancillary data on soil, land
    countries = ['Kenya','Ethiopia','Somalia']
    xr_X = get_soil_vars(countries, df = df, invert=False)

    ### generate final dataframe and save it
    final_df = xr_X.to_dataframe().dropna(how='all').merge(df, left_index=True, right_index=True)
    final_df.to_csv(r'./data/final_dataset_ndvi_spi.csv')

if __name__ == "__main__":
    generate_dataset()