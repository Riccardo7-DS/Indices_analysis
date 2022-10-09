import eumdac
import geopandas as gpd
from datetime import datetime, timedelta, date
from datetime import time as time_dt
import os
import shutil
from shapely.geometry import Polygon, mapping
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from glob import glob
import os
import time
import fnmatch
import shutil
import functions.function_clns as function_clns
import pandas as pd
import requests

def add_time(xr_df):
    my_date_string = xr_df.attrs['date_time'] 
    date_xr = datetime.strptime(my_date_string, '%Y%m%d/%H:%M')
    date_xr = pd.to_datetime(date_xr)
    xr_df = xr_df.assign_coords(time=date_xr)
    xr_df = xr_df.expand_dims(dim="time")
    return xr_df

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

def cut_file(xr_df, gdf):
    xr_df.rio.write_crs("epsg:4326", inplace=True)
    clipped = xr_df.rio.clip(gdf.geometry.apply(mapping), gdf.crs, drop=True)
    clipped = clipped.drop('crs')
    return clipped

def pipeline_ndvi(xr_df, base_dir, gdf):
    xr_df = cut_file(xr_df, gdf)
    xr_df = add_time(xr_df)
    xr_df = compute_radiance(xr_df)
    xr_df = xr_df.drop('channel_3')
    xr_df = xr_df.assign(ndvi=(xr_df['channel_2'] - xr_df['channel_1']) / (xr_df['channel_2'] + xr_df['channel_1']))
    return xr_df

def check_status(customisation):
    status = "QUEUED"
    sleep_time = 10 # seconds
    
    # Customisation Loop
    while status == "QUEUED" or status == "RUNNING":
        # Get the status of the ongoing customisation
        status = customisation.status
    
        if "DONE" in status:
            print(f"SUCCESS")
            break
        elif "ERROR" in status or 'KILLED' in status:
            print(f"UNSUCCESS, exiting")
            break
        elif "QUEUED" in status:
            print(f"QUEUED")
        elif "RUNNING" in status:
            print(f"RUNNING")
        elif "INACTIVE" in status:
            sleep_time = max(60*10, sleep_time*2)
            print(f"INACTIVE, doubling status polling time to {sleep_time} (max 10 mins)")
        time.sleep(sleep_time)


def delete_old_job():

    CONFIG_PATH = r"./config.yaml"

    # Function to load yaml configuration file
    config = function_clns.load_config(CONFIG_PATH)
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

def datatailor_loop():

    CONFIG_PATH = r"./config.yaml"

    # Function to load yaml configuration file
    config = function_clns.load_config(CONFIG_PATH)
    # Insert your personal key and secret into the single quotes
    consumer_key = config['DEFAULT']['key']
    consumer_secret = config['DEFAULT']['secret']

    credentials = (consumer_key, consumer_secret)

    token = eumdac.AccessToken(credentials)
    print(f"This token '{token}' expires {token.expiration}")

    gdf = gpd.read_file(r'C:\Users\Riccardo\Desktop\PhD_docs\Drought_prediction\ETH_adm\ETH_adm0.shp')

    datastore = eumdac.DataStore(token)
    datatailor = eumdac.DataTailor(token)
    
    
    selected_collection = datastore.get_collection('EO:EUM:DAT:MSG:HRSEVIRI') 

    def chain_exists(product_id, chain_id):
        result = None

        for chain in datatailor.chains.search(product= product_id):
            if chain.id == chain_id:
               result = chain

        return result

    chain = chain_exists('HRSEVIRI', 'hrseviri_netcdf')

    # Check if the 'hrseviri_nc_west-africa' chains exists
    if chain:
        print(f"The chain 'hrseviri_netcdf' already exists: {chain}")
        # Read the chain
    else:   
        print("The chain 'hrseviri_nc_west-africa' does not exist and will be created now:")
        chain = eumdac.tailor_models.Chain(
                id='hrseviri_netcdf',
                name='Native to netcdf',
                description='Convert a SEVIRI Native product to netcdf',
                product='HRSEVIRI',
                format='',
                filter='hrseviri_natural_color',
                projection='geographic',
        )
        datatailor.chains.create(chain)
        chain = chain_exists('HRSEVIRI', 'hrseviri_netcdf')
        print(f"{chain}")


    download_dir = r'D:\MSG\MSG_nat\batch_2'

    start_date = '2007-01-01 08:45:00'
    end_date= '2009-01-01 11:45:00'

    start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    delta = timedelta(days=1)
    time_window = timedelta(hours=6)

    while start_dt <= end_dt:  
        limit_dt = start_dt + time_window
        # Retrieve datasets that match our filter
        product = selected_collection.search(
            #geo='POLYGON(({}))'.format(','.join(["{} {}".format(*coord) for coord in geometry])),
            #bbox=bbox,
            dtstart=start_dt, 
            dtend=limit_dt).first()


        customisation = datatailor.new_customisation(product = product, chain=chain)
        check_status(customisation)
        net, = fnmatch.filter(customisation.outputs, '*.nc')

        try:
            with customisation.stream_output(net,) as stream:
                with open(os.path.join(download_dir, stream.name), mode='wb') as fdst:
                    #print(f'Downloading {fsrc.name}')
                    shutil.copyfileobj(stream, fdst)
                    print(f'Download of product {stream.name} finished.')

            ds = xr.open_dataset(os.path.join(download_dir, stream.name))
            ds = pipeline_ndvi(ds, download_dir, gdf)
            ds.to_netcdf(os.path.join(download_dir,'processed', stream.name))

        except Exception as e:
            print('error {} on day'.format(e), datetime.strftime(start_dt, format='%Y-%m-%d %H:%M:%S'))

        start_dt += delta


if __name__ == "__main__":
    #delete_old_job()
    datatailor_loop()
