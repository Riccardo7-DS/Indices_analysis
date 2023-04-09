from scipy.sparse import dok_matrix
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import seaborn as sns

from p_drought_indices.functions.function_clns import subsetting_pipeline, load_config
import xarray as xr 
import pandas as pd
import yaml
import os
import numpy as np
import re
from typing import Union
from xarray import Dataset, DataArray

##### Credits to https://github.com/MiniXC

def make_sparse(path, dtype='int'):
    with open(path, "r") as file:
        cols, rows = int(file.readline().split()[1]), int(file.readline().split()[1])
        x_corner, y_corner = int(file.readline().split()[1]), int(file.readline().split()[1])
        cellsize = float(file.readline().split()[1])
        nodata_val = file.readline().split()[1]
        line = file.readline()
        if dtype == 'int':
            M = dok_matrix((rows, cols), dtype=int)
        if dtype == 'float':
            M = dok_matrix((rows, cols), dtype=float)
        empty_line = ' ' + ' '.join([nodata_val] * cols) + '\n'
        for i in tqdm(range(rows)):
            if empty_line != line:
                for j, val in enumerate(line.split()):
                    if val != nodata_val:
                        if dtype == 'int':
                            M[i,j] = int(val)
                        if dtype == 'float':
                            M[i,j] = float(val)
            line = file.readline()
        return M, x_corner, y_corner, cellsize

def sparse2dict(M, x_corner, y_corner, cellsize, name):
    sparse_dict = {
        'lon': [],
        'lat': [],
        name: [],
    }

    for i in tqdm(M.items()):
        lat, lon  = i[0]
        lon, lat = (x_corner + lon*cellsize), (abs(y_corner) - lat*cellsize)
        sparse_dict['lon'].append(lon)
        sparse_dict['lat'].append(lat)
        sparse_dict[name].append(i[1])
    return sparse_dict


def get_soil_vars(CONFIG_PATH, countries:list, xr_df:Union[DataArray, Dataset]=None, df:Union[pd.DataFrame, None]=None, invert =True):
    """
    df: pandas dataframe to get initial raster and to sample vars
    """
    config = load_config(CONFIG_PATH)
    path = config['DEFAULT']['ancillary']
    chunks={'time':'500MB'}

    if ((df is None) & (xr_df is None)):
        print('Using the default MERRA-2 grid for querying FAO HWSD')
        ds = xr.open_dataset(os.path.join(config['MERRA2']['path'], 'nasapower_vars.nc'), chunks=chunks)
        elev_df = ds[['lat','lon']].to_dataframe().reset_index()
        
    elif ((df is not None) & (xr_df is not None)):
        raise ValueError('Need to specify only one input between the pandas and xarray datasets')

    elif ((df is not None) & (xr_df is None)):
        print('Using provided pandas dataframe for querying FAO HWSD')
        elev_df = df.reset_index()[['lat','lon','time']]
    else:
        ds = xr_df.copy()
        print('Using provided xarray dataframe for querying FAO HWSD')
        elev_df = ds[['lat','lon']].to_dataframe().reset_index()
        
    if invert ==True:
        elev_df.rename(columns={'lat':'lon','lon':'lat'},inplace=True)

    M, x_corner, y_corner, cellsize = make_sparse(os.path.join(path, "GloElev_5min.asc"))


    def lonlat2xy(lon, lat, x_corner, y_corner, cellsize):
        return abs(int(round((lat + y_corner)/cellsize))), int(round((lon - x_corner)/cellsize))

    def xy2lonlat(x, y, x_corner, y_corner, cellsize):
        return (x_corner + y*cellsize), (abs(y_corner) - x*cellsize)

    xy_list = [lonlat2xy(row['lon'], row['lat'], x_corner, y_corner, cellsize) for _, row in tqdm(elev_df.iterrows(), total=len(elev_df))]

    elev_df['arc_xy'] = xy_list


    def add_value(drought_df, path, name, dtype='int'):
        M, x_corner, y_corner, cellsize = make_sparse(path, dtype)
        elev_df[name] = elev_df['arc_xy'].apply(lambda x: M[x])
        return drought_df


    elev_df = add_value(elev_df, os.path.join(path, "GloElev_5min.asc"), "elevation")
    elev_df = add_value(elev_df, os.path.join(path, "GloLand_5min.asc"), "land")

    slopes = [
        ('GloSlopesCl1_5min.asc','slope1'),
        ('GloSlopesCl2_5min.asc','slope2'),
        ('GloSlopesCl3_5min.asc','slope3'),
        ('GloSlopesCl4_5min.asc','slope4'),
        ('GloSlopesCl5_5min.asc','slope5'),
        ('GloSlopesCl6_5min.asc','slope6'),
        ('GloSlopesCl7_5min.asc','slope7'),
        ('GloSlopesCl8_5min.asc','slope8'),
    ]

    print('Starting adding slopes...')
    for path_file, slope in slopes:
        elev_df = add_value(elev_df,  os.path.join(path, path_file), slope)
        elev_df[slope] = elev_df[slope]/10000

    land_covers = [
        'WAT', 'NVG', 'URB', 'GRS', 'FOR', 'CULTRF', 'CULTIR', 'CULT'
    ]

    print('Starting adding land covers...')

    for land in land_covers:
        elev_df = add_value(elev_df, os.path.join(path, '{}_2000.asc'.format(land)), '{}_LAND'.format(land), dtype='float')
        elev_df[f'{land}_LAND'] = elev_df[f'{land}_LAND']

    print('Starting adding soil covers...')
    sqs = list(range(1,8))

    for sq in sqs:
        elev_df = add_value(elev_df, os.path.join(path, 'sq{}.asc'.format(sq)), f'SQ{sq}', dtype='int')
        elev_df[f'SQ{sq}'] = elev_df[f'SQ{sq}']

    if 'time' in elev_df.columns:
        ds_el = elev_df.set_index(['lon','lat','time']).to_xarray()
    else:
        ds_el = elev_df.set_index(['lon','lat']).to_xarray()
    ds_el = ds_el.assign(elevation= ds_el['elevation'].where(ds_el['elevation']>=0, np.NaN))
    ds_el = ds_el.assign(land= ds_el['land'].where(ds_el['land']>=0, np.NaN))
    return ds_el

def get_water_cover(CONFIG_PATH, countries:list, xr_df:Union[DataArray, Dataset]=None, df:Union[pd.DataFrame, None]=None, invert =True):
    """
    df: pandas dataframe to get initial raster and to sample vars
    """
    config = load_config(CONFIG_PATH)
    path = config['DEFAULT']['ancillary']
    chunks={'time':'500MB'}

    if ((df is None) & (xr_df is None)):
        print('Using the default MERRA-2 grid for querying FAO HWSD')
        ds = xr.open_dataset(os.path.join(config['MERRA2']['path'], 'nasapower_vars.nc'), chunks=chunks)
        elev_df = ds[['lat','lon']].to_dataframe().reset_index()
        
    elif ((df is not None) & (xr_df is not None)):
        raise ValueError('Need to specify only one input between the pandas and xarray datasets')

    elif ((df is not None) & (xr_df is None)):
        print('Using provided pandas dataframe for querying FAO HWSD')
        elev_df = df.reset_index()[['lat','lon','time']]
    else:
        ds = xr_df.copy()
        print('Using provided xarray dataframe for querying FAO HWSD')
        elev_df = ds[['lat','lon']].to_dataframe().reset_index()
        
    if invert ==True:
        elev_df.rename(columns={'lat':'lon','lon':'lat'},inplace=True)

    M, x_corner, y_corner, cellsize = make_sparse(os.path.join(path, "GloElev_5min.asc"))


    def lonlat2xy(lon, lat, x_corner, y_corner, cellsize):
        return abs(int(round((lat + y_corner)/cellsize))), int(round((lon - x_corner)/cellsize))

    def xy2lonlat(x, y, x_corner, y_corner, cellsize):
        return (x_corner + y*cellsize), (abs(y_corner) - x*cellsize)

    xy_list = [lonlat2xy(row['lon'], row['lat'], x_corner, y_corner, cellsize) for _, row in tqdm(elev_df.iterrows(), total=len(elev_df))]

    elev_df['arc_xy'] = xy_list


    def add_value(drought_df, path, name, dtype='int'):
        M, x_corner, y_corner, cellsize = make_sparse(path, dtype)
        elev_df[name] = elev_df['arc_xy'].apply(lambda x: M[x])
        return drought_df


    land_covers = [
        'WAT' ]

    print('Starting adding water covers...')

    for land in land_covers:
        elev_df = add_value(elev_df, os.path.join(path, '{}_2000.asc'.format(land)), '{}_LAND'.format(land), dtype='float')
        elev_df[f'{land}_LAND'] = elev_df[f'{land}_LAND']

    if 'time' in elev_df.columns:
        ds_el = elev_df.set_index(['lon','lat','time']).to_xarray()
    else:
        ds_el = elev_df.set_index(['lon','lat']).to_xarray()
    return subsetting_pipeline(CONFIG_PATH, ds_el, countries=countries).transpose("lat","lon")["WAT_LAND"]

