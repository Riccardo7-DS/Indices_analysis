#!/usr/bin/env python
# coding: utf-8


import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import numpy as np

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

path = '/mnt/hgfs/share/AVHRR/*.nc'
shapefile_path = '/mnt/hgfs/share/afr_g2014_2013_0.zip'


#### Chose subset of countries
countries = ['Ethiopia','Somalia','Kenya']
gdf = gpd.read_file(shapefile_path)
subset = gdf[gdf.ADM0_NAME.isin(countries)]


data = xr.open_mfdataset(path)
#### Create a new dataset with nans
nan_data = data.copy()
nan_data['NDVI'] = nan_data['NDVI'].where(nan_data['NDVI']!= 0.00, np.NaN)
nan_data = downsample(nan_data)


#nan_data['NDVI'].isel(time=0).plot()
nan_data = cut_file(nan_data, subset)
#nan_data['NDVI'].isel(time=1).plot()
ds = cut_file(data, subset)
#ds['NDVI'].isel(time=5).plot()


# ### Logistic regression for p of drought

path_spi = '/mnt/hgfs/share/CHIRPS/daily/SPI/CHIRPS_spi_gamma_07.nc'
spi_7 = cut_file(xr.open_dataset(path_spi, chunks={"lat": -1, "lon": -1, "time": 12}), subset)

wet_cond = spi_7['spi_gamma_07'].where(spi_7.spi_gamma_07>2, 1, 0)
dry_cond = spi_7['spi_gamma_07'].where(spi_7.spi_gamma_07<2, 1, 0)

spi_7 = spi_7.assign(wet=wet_cond, dry=dry_cond)

down_spi7 = downsample(spi_7)

#### copy dataset and set it to nan where there are clouds (equal to 0)
nan_spi_7 = spi_7.copy() 
nan_spi_7['spi_gamma_07'] = nan_spi_7['spi_gamma_07'].where(nan_spi_7['spi_gamma_07']!= 0.00)
nan_spi_7 = downsample(nan_spi_7)

#### check when wet/dry condition is met
dry_cond = nan_spi_7['spi_gamma_07'].where(nan_spi_7.spi_gamma_07>=2)
wet_cond = nan_spi_7['spi_gamma_07'].where(nan_spi_7.spi_gamma_07<=-2)

#### assign new variables
nan_spi_7 = nan_spi_7.assign(wet=wet_cond, dry=dry_cond)

#### set the values which are not extremes to 0
nan_spi_7['dry'] = nan_spi_7['spi_gamma_07'].where(nan_spi_7['spi_gamma_07']==dry_cond,0)
nan_spi_7['wet'] = nan_spi_7['spi_gamma_07'].where(nan_spi_7['spi_gamma_07']==wet_cond,0)

#### set the values which are not extremes to 0
nan_spi_7['dry'] = nan_spi_7['dry'].where(nan_spi_7['dry']==0,1)
nan_spi_7['wet'] = nan_spi_7['wet'].where(nan_spi_7['wet']==0,1)
