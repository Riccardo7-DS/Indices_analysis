import eumdac
import sys 
sys.path.append(r'C:\Users\Riccardo\Desktop\PhD_docs\Drought_prediction\Project\Indices_analysis\functions')
import geopandas as gpd
from datetime import datetime, timedelta
import os
import shutil
from shapely.geometry import Polygon, mapping
import xarray as xr
import matplotlib.pyplot as plt
from glob import glob
import os
import datetime as datetime
from p_drought_indices.eumetsat_data_collection.data_tailor.eptc_crop_clean import load_config
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from glob import glob
import os
import yaml
import zipfile

#### Chose the product of interest
collectionID ='EO:EUM:DAT:MSG:HRSEVIRI'     ####soil moisture: EO:EUM:DAT:METOP:H25   ###cloud mask: EO:EUM:DAT:MSG:CLM

##### Use environmnet gis2_py39 to run this script ######################

def unpack_all_in_dir(_dir, extension = ".zip"):
    for item in os.listdir(_dir):  # loop through items in dir
        abs_path = os.path.join(_dir, item)  # absolute path of dir or file
        if item.endswith(extension):  # check for ".zip" extension
            file_name = os.path.abspath(abs_path)  # get full path of file
            zip_ref = zipfile.ZipFile(file_name)  # create zipfile object
            zip_ref.extractall(_dir)  # extract file to dir
            zip_ref.close()  # close file
            os.remove(file_name)  # delete zipped file
        elif os.path.isdir(abs_path):
            unpack_all_in_dir(abs_path)  # recurse this function with inner folder

def storing_cleaning_seviri(collectionID, start_dt):

    CONFIG_PATH = r"./config.yaml"

    # Function to load yaml configuration file
    config = load_config(CONFIG_PATH)
    #time_config = load_config(TIME_PATH)

    time_window = timedelta(minutes=30)

    limit_dt = start_dt + time_window

    # Insert your personal key and secret into the single quotes
    consumer_key = config['DEFAULT']['key']
    consumer_secret = config['DEFAULT']['secret']

    credentials = (consumer_key, consumer_secret)
    token = eumdac.AccessToken(credentials)

    print(f"This token '{token}' expires {token.expiration}")

    datastore = eumdac.DataStore(token)
    datastore.collections

    selected_collection = datastore.get_collection(collectionID)
    bbox =[32.9, 3.2, 48, 15]

    # Add vertices for polygon, wrapping back to the start point.
    geometry = [[15,32.8],[2.9,32.8],[2.9,48],[48,15], [15,32.8]]  ###ethiopia coordinates

    download_dir = config['NDVI']['cloud_path']  
    
    # Retrieve datasets that match our filter
    product = selected_collection.search(
        geo='POLYGON(({}))'.format(','.join(["{} {}".format(*coord) for coord in geometry])),
        #bbox=bbox,
        dtstart=start_dt,
        dtend=limit_dt).first()
    selected_product = datastore.get_product(product_id=str(product), collection_id=collectionID)
    try:
        with selected_product.open() as fsrc, open(os.path.join(download_dir, fsrc.name), mode='wb') as fdst:
            #print(f'Downloading {fsrc.name}')
            shutil.copyfileobj(fsrc, fdst)
            print(f'Download of product {fsrc.name} finished.')
    except Exception as e:
        print('http error {} on day'.format(e), datetime.strftime(start_dt, format='%Y-%m-%d %H:%M:%S'))
    

if __name__ == "__main__":

    product = 'EO:EUM:DAT:MSG:CLM'

    list_dates = pd.read_csv(r'C:\Users\Riccardo\Desktop\PhD_docs\Drought_prediction\Project\Indices_analysis\data\cloudmask_collect.csv', index_col=0).iloc[:,0]
    print(list_dates)

    date = '2010-09-19'
    list_dates = list_dates[list_dates>date]

    for date in list_dates:
        print('Starting downloading product SEVIRI for day {}'.format(date))
        start_dt = datetime.strptime(date, '%Y-%m-%d %H:%M:%S') - timedelta(minutes=15)
        storing_cleaning_seviri(product, start_dt)

    #else:    ###manually specify dates
    #    start_date = '2006-09-24 11:45:00' 
    #    end_date='2009-01-01 11:45:00'
##
    #    start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    #    end_dt = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
##
    #    delta = timedelta(days=1)
##
    #    while start_dt < end_dt:
    #        storing_cleaning_seviri(start_dt)
    #        start_dt = start_dt + delta

    CONFIG_PATH = r"./config.yaml"
    config = load_config(CONFIG_PATH)
    download_dir = config['NDVI']['cloud_path']
    unpack_all_in_dir(download_dir)