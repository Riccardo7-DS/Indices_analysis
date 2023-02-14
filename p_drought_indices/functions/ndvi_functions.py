import os
import pandas as pd
import time
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import yaml
from p_drought_indices.functions.function_clns import load_config
import xskillscore as xs
from datetime import datetime
import xarray as xr
import numpy as np
from xarray import DataArray, Dataset

def downsample(ds):
    monthly = ds.resample(time='5D', skipna=True).mean() #### Change here to change the timeframe over which to make the data imputation
    return monthly

def clean_ndvi(ds):
    ds = ds.where('ndvi'!=0.00)
    return ds

def find_ndvi_outliers(datarray:DataArray):
    list_1 = datarray.where(datarray>1, drop=True).to_dataframe().dropna().reset_index()['time'].unique()
    list_2 = datarray.where(datarray<-1, drop=True).to_dataframe().dropna().reset_index()['time'].unique()
    return np.unique(np.concatenate((list_1, list_2), axis=0))

def clean_outliers(dataset:Dataset):
    list_out = find_ndvi_outliers(dataset['ndvi'])
    return dataset.drop_sel(time= list_out)


def compute_ndvi(xr_df):
    return xr_df.assign(ndvi=(xr_df['channel_2'] - xr_df['channel_1']) / (xr_df['channel_2'] + xr_df['channel_1']))

def compute_radiance(xr_df):
    satellite = xr_df.attrs['EPCT_product_name'][:4]
    if satellite == 'MSG2':
        xr_df['channel_1'] = xr_df['channel_1']/65.2065
        xr_df['channel_2'] = xr_df['channel_2']/73.0127
        
    elif satellite == 'MSG1':
        xr_df['channel_1'] = xr_df['channel_1']/65.2296 
        xr_df['channel_2'] = xr_df['channel_2']/73.1869
    
    elif satellite == 'MSG3':
        xr_df['channel_1'] = xr_df['channel_1']/65.5148 
        xr_df['channel_2'] = xr_df['channel_2']/73.1807
        
    elif satellite == 'MSG4':
        xr_df['channel_1'] = xr_df['channel_1']/65.2656
        xr_df['channel_2'] = xr_df['channel_2']/73.1692
    
    else:
        print('This product doesn\'t contain MSG1, MSG2, MSG3, MSG4 Seviri')
    
    return xr_df

def add_time(xr_df):
    my_date_string = xr_df.attrs['EPCT_start_sensing_time']#xr_df.attrs['date_time']
    date_xr = datetime.strptime(my_date_string,'%Y%m%dT%H%M%SZ') #datetime.strptime(my_date_string, '%Y%m%d/%H:%M')
    date_xr = pd.to_datetime(date_xr)
    xr_df = xr_df.assign_coords(time=date_xr)
    xr_df = xr_df.expand_dims(dim="time")
    return xr_df

def process_ndvi(base_dir, file):
    with xr.open_dataset(os.path.join(base_dir, file)) as ds:
        data = ds.load()
        xr_df = data.drop('channel_3')
        xr_df = add_time(data)
        xr_df = compute_radiance(xr_df)
        xr_df = xr_df.assign(ndvi=(xr_df['channel_2'] - xr_df['channel_1']) / (xr_df['channel_2'] + xr_df['channel_1']))
        xr_df.to_netcdf(os.path.join(base_dir,'processed', file)) 
        xr_df.close()
