from dataclasses import dataclass
import matplotlib.pyplot as plt
from SPI_wet_dry import spiObject
import numpy as np
import xarray as xr
from merge_daily_data import load_config, cut_file
import os
import pandas as pd
import time
import geopandas as gpd
from shapely.geometry import Polygon, mapping
from SPI_wet_dry import spiObject
import datetime as dt

CONFIG_PATH = r"../config.yaml"
config = load_config(CONFIG_PATH)

product_dir = config['SPI']['CHIRPS']['path']
files  = [f for f in os.listdir(product_dir) if f.endswith('.nc')]
path_spi = os.path.join(product_dir, files[0])
spi = spiObject(path_spi)
shapefile_path = os.path.join('..', config['SHAPE']['africa'])
#### Chose subset of countries
gdf = gpd.read_file(shapefile_path)
subset = gdf[gdf.ADM0_NAME.isin(['Ethiopia'])]
###open file and cut it
with xr.open_dataset(path_spi, chunks={"lat": -1, "lon": -1, "time": 12}) as data:
    res_xr = cut_file(data, subset)

####drop all the values after 180 days
time_thresh = res_xr['time'].isel(time=180).values
threshold = -1
ds = res_xr.where(res_xr.time > time_thresh, drop=True)
#### slice by one year only
ds = ds.sel(
    time=slice('2001-01-01', '2003-01-01'))

ds = ds.load()

#### create masks for null and for drought-non drought 
null_mask = ds.where(ds['spi_gamma_180'].notnull(), 0)
masked = xr.where(ds['spi_gamma_180']>= threshold, 1, 0)
#masked = ds.where(ds['spi_gamma_180']<= threshold, )
masked.isel(time=200).plot()
plt.title('Mask for area in Ethiopia not affected by drought')
plt.show()

ds = ds.assign(masked = masked)
new_xr = ds.assign(index = ds['masked'].cumsum(['time']))
new_xr['spi_gamma_180'].where(new_xr.masked ==0).isel(time=200).plot()
plt.title('Mask for area in Ethiopia affected by drought')
plt.show()

new_xr.where(new_xr.masked==0).sel(lat= 3.625, lon=38.375, method='nearest')['spi_gamma_180'].plot() #.where(null_mask['spi_gamma_180']!=0)
plt.title('Show behavior of SPI index in one location when it goes below the threshold')
plt.show()
#new_xr = new_xr.assign(year = new_xr['time'].dt.year, season= new_xr['time'].dt.month%12 // 3 + 1)

df= new_xr.where(new_xr.masked==0).to_dataframe().dropna(subset=['index','spi_gamma_180']).drop(columns={'spatial_ref'}) #.where(null_mask['spi_gamma_180']!=0)
duration_df = df.groupby(["lat", "lon",'index']).count().drop(columns={'spi_gamma_180'}).rename(columns={'masked':'duration'})
severity_df = df.groupby(["lat", "lon",'index']).sum().drop(columns={'masked'}).rename(columns={'spi_gamma_180':'severity'})
severity_df['severity'] = abs(severity_df['severity'])
res_df = severity_df.merge(duration_df, left_index=True, right_index=True)
res_df['intensity']= res_df['severity']/res_df['duration']

run_xr = new_xr.to_dataframe().reset_index().merge(res_df.reset_index(), on=['lat','lon','index'], how='left')\
    .drop(columns={'spatial_ref'}).set_index(['lat','lon','time']).to_xarray()

events_df = new_xr.to_dataframe().reset_index().merge(res_df.reset_index(), on=['lat','lon','index'], how='left')\
    .drop(columns={'spatial_ref','masked','spi_gamma_180'}).dropna(subset=['severity','duration','intensity'])\
        .drop_duplicates(subset=['lat','lon','index','severity','duration','intensity'], keep='first').drop(columns={'index'})

events_df['year'] = events_df['time'].dt.year
events_df['month'] = events_df['time'].dt.month


