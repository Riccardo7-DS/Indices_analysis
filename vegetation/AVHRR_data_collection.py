import numpy as np
import os
import urllib
from urllib.parse import urlparse
from urllib.parse import urljoin
from datetime import datetime, timedelta, date
from datetime import time as time_dt
from shapely.geometry import Polygon, mapping
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import requests
from bs4 import BeautifulSoup, SoupStrainer
import re

def cut_file(xr_df, gdf):
    xr_df = xr_df.drop_vars(['lon_bnds','lat_bnds'])
    xr_df['TIMEOFDAY'] = xr_df['TIMEOFDAY'].astype(int)
    xr_df.rio.set_spatial_dims(x_dim='latitude', y_dim='longitude', inplace=True)
    xr_df.rio.write_crs("epsg:4326", inplace=True)
    vars_list = list(xr_df.data_vars)
    for var in vars_list:
        del xr_df[var].attrs['grid_mapping']
    clipped = xr_df.rio.clip(gdf.geometry.apply(mapping), gdf.crs, drop=True)
    clipped = clipped.drop('crs')
    return clipped

def cut_only(download_dir,target_dir, gdf):
    files = [f for f in os.listdir(download_dir) if f.endswith('.nc')]
    for name in files:
        print('The file name is' , str(name))
        xr_df = xr.open_dataset(os.path.join(download_dir, name), engine='netcdf4',decode_coords=False)
        df = cut_file(xr_df, gdf)
        df.to_netcdf(os.path.join(target_dir, name))
        xr_df.close()
            
    [os.remove(os.path.join(download_dir, name)) for name in files]

gdf = gpd.read_file(r'C:\Users\Riccardo\Desktop\PhD_docs\Drought_prediction\afr_g2014_2013_0\afr_g2014_2013_0.shp')
download_dir = r'D:\MSG\AVHRR'
target_dir = r'D:\MSG\AVHRR\processed'


def cropping_loop():
    download_dir = r'D:\MSG\AVHRR\batch_1'
    target_dir = r'D:\MSG\AVHRR\processed'
    regex = re.compile("AVHRR-Land_v005_AVH13C1_NOAA-18_2007(.*)")
    files = [f for f in os.listdir(download_dir) if f.endswith('.nc')]
    selected = filter(regex.match, files)
    for file in selected:
        print('Cropping file ', str(file))
        xr_df = xr.open_dataset(os.path.join(download_dir, file), engine='netcdf4',decode_coords=False)
        df = cut_file(xr_df, gdf)
        df.to_netcdf(os.path.join(target_dir, file))
        xr_df.close()

def data_collection():

    years = [2005, 2008, 2009, 2010]

    for idx in years:
        URL = 'https://www.ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access/{}/'.format(idx)

        urls = []
        names = []
        
        try:
            response = requests.get(URL, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            
            a_tag = soup.find_all('td')
                
            for x in a_tag:
                y = x.find_all('a')
                for i in y:
                    if i.has_attr('href'):
                        urls.append(i['href'])
                        
            urls = list(filter(lambda x: x.endswith(".nc"), urls))
        
            for url in urls:
                
                URL_merged = urljoin(URL, url)
                name = str(url)
                names.append(name)
                urllib.request.urlretrieve(URL_merged, os.path.join(download_dir, name))
                print('Finished downloading product {}'.format(name))
            
            cut_only(download_dir, target_dir, gdf)

        except Exception as e:
            print('Couldn\'t download product {name} from year {year} because of error'.format(name=name, year=idx), e)


                
        

if __name__== '__main__':
    cropping_loop()