#import eumdac
import geopandas as gpd
from datetime import datetime, timedelta, date
from datetime import time as time_dt
import os
import shutil
from shapely.geometry import Polygon, mapping
import xarray as xr
import geopandas as gpds
from glob import glob
import os
import time
import fnmatch
import shutil
import pandas as pd
import requests
from utils.function_clns import load_config, subsetting_pipeline, cut_file
from utils.xarray_functions import compute_radiance, add_time
import time
import eumdac
from tqdm import tqdm

def seviri_product_polygon(collectionID, start_dt, download_dir):

    from utils.function_clns import config

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
    



def check_status(customisation):
    status = "QUEUED"
    sleep_time = 10 # seconds
    
    # Customisation Loop
    while status == "QUEUED" or status == "RUNNING":
        # Get the status of the ongoing customisation
        status = customisation.status
    
        if "DONE" in status:
        #    print(f"SUCCESS")
            break
        elif "ERROR" in status or 'KILLED' in status:
            print(f"UNSUCCESS, exiting")
            break
        elif "QUEUED" in status:
        #    print(f"QUEUED")
            continue
        elif "RUNNING" in status:
            # print(f"RUNNING")
            continue
        elif "INACTIVE" in status:
            sleep_time = max(60*10, sleep_time*2)
            print(f"INACTIVE, doubling status polling time to {sleep_time} (max 10 mins)")
        time.sleep(sleep_time)


def check_reset_token(start_time, minutes = 60):
    from utils.function_clns import config
    now = time.time()
    res = (now - start_time)
    
    if res/(minutes*60)>=1:
        # Insert your personal key and secret into the single quotes
        consumer_key = config['DEFAULT']['key']
        consumer_secret = config['DEFAULT']['secret']

        credentials = (consumer_key, consumer_secret)

        token = eumdac.AccessToken(credentials)
        # print(f"This token '{token}' expires {token.expiration}")
        return now
    
    else:
        # print('The token is still vaild for {} minutes'.format(round(minutes - (res/60),2)))
        return start_time


def delete_old_job():

    from utils.function_clns import config
    # Insert your personal key and secret into the single quotes
    consumer_key = config['DEFAULT']['key']
    consumer_secret = config['DEFAULT']['secret']

    credentials = (consumer_key, consumer_secret)

    token = eumdac.AccessToken(credentials)
    print(f"This token '{token}' expires {token.expiration}")
    datatailor = eumdac.DataTailor(token)
     # Delete all jobs older than one month

    delete_datetime = datetime.combine(date.today() - timedelta(days=1), time_dt(0, 0))

    for customisation in datatailor.customisations:
        if customisation.creation_time <= delete_datetime:
            print(f'Delete customisation {customisation} from {customisation.creation_time}.')
            customisation.delete()

def datatailor_loop(product_code, start_date, end_date, delete_chain=False):

    # Function to load yaml configuration file
    from utils.function_clns import config
    # Insert your personal key and secret into the single quotes
    consumer_key = config['DEFAULT']['key']
    consumer_secret = config['DEFAULT']['secret']

    credentials = (consumer_key, consumer_secret)

    token = eumdac.AccessToken(credentials)
    start_time = time.time()
    print(f"This token '{token}' expires {token.expiration}")

    datastore = eumdac.DataStore(token)
    datatailor = eumdac.DataTailor(token)
    
    selected_collection = datastore.get_collection(product_code)

    if  product_code == "EO:EUM:DAT:MSG:CLM":
        default_chain = 'cloud_mask_chain'
        download_dir = config['NDVI']['cloud_download']
        product = 'MSGCLMK'
    elif  product_code == "EO:EUM:DAT:MSG:HRSEVIRI":
        default_chain = 'ndvi_chain'
        download_dir = config['NDVI']['seviri_download']
        product = 'HRSEVIRI'
    else:
        raise ValueError('There exist no predefined chain for the chosen product')
    
    if delete_chain == True:
        datatailor.chains.delete(default_chain)

    def chain_exists(product_id, chain_id):
        result = None

        for chain in datatailor.chains.search(product= product_id):
            if chain.id == chain_id:
               result = chain

        return result

    chain = chain_exists(product, default_chain)

    # Check if the 'hrseviri_nc_west-africa' chains exists
    if chain:
        print(f"The chain {default_chain} already exists: {chain}")
        # Read the chain
    else:
        print("The chain 'hrseviri_nc_west-africa' does not exist and will be created now:")
        chain = eumdac.tailor_models.Chain(
                id='hoa_chain',
                name='Native to netcdf of HOA',
                description='Chain to subset HOA region in netcdf',
                product='HRSEVIRI',
                format='netcdf4',
                filter='hrseviri_natural_color',
                projection='geographic',
                roi=None,
        )
        datatailor.chains.create(chain)
        chain = chain_exists('HRSEVIRI', default_chain)
        print(f"{chain}")

    hrseviri_westafrica = datatailor.chains.read(default_chain)

    start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    delta = timedelta(days=1)
    time_window = timedelta(minutes=30)

    # Calculate total number of days
    total_days = (end_dt - start_dt).days + 1
    
    # Wrap the loop with tqdm to monitor progress
    with tqdm(total=total_days, desc="Processing Dates", unit="day") as pbar:

        while start_dt <= end_dt:  
            start_time = check_reset_token(start_time)
            limit_dt = start_dt + time_window
            # Retrieve datasets that match our filter
            product = selected_collection.search(
                #geo='POLYGON(({}))'.format(','.join(["{} {}".format(*coord) for coord in geometry])),
                #bbox=bbox,
                dtstart=start_dt, 
                dtend=limit_dt).first()

            try:
                customisation = datatailor.new_customisation(product = product, chain=hrseviri_westafrica)
                check_status(customisation)
                net, = fnmatch.filter(customisation.outputs, '*.nc')
                with customisation.stream_output(net,) as stream:
                    with open(os.path.join(download_dir, stream.name), mode='wb') as fdst:
                            #print(f'Downloading {fsrc.name}')
                        shutil.copyfileobj(stream, fdst)
                        # print(f'Download of product {stream.name} finished.')

                    #ds = xr.open_dataset(os.path.join(download_dir, stream.name))
                    #ds = pipeline_ndvi(ds, download_dir, gdf)
                    #ds.to_netcdf(os.path.join(download_dir,'processed', stream.name))
                    customisation.delete()

            except Exception as e:
                error_time = datetime.strftime(start_dt, format='%Y-%m-%d %H:%M:%S')
                print('error {} on day'.format(e), error_time )
                temp_df = pd.DataFrame([error_time, e]).T
                temp_df.to_csv(r'./seviri_error_days.csv', mode='a', index=False, header=False)

            start_dt += delta
            pbar.update(1)

if __name__ == "__main__":
    from utils.function_clns import config
    product_code = config['SEVIRI']['cloud']
    start_date = '2005-01-01 10:35:00'
    end_date= '2023-12-31 10:35:00'

    datatailor_loop(product_code, start_date, end_date)
