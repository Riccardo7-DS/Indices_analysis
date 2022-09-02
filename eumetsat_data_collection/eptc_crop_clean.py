#### Cut Ethiopia and convert to xarray
import xarray as xr
from datetime import datetime, timedelta
from epct import api
import os
import pandas as pd
import zipfile
import time

def cropping_pipeline():

    
    extension = ".zip"

      # change directory from working dir to dir with files


    def unpack_all_in_dir(_dir):
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




    base_dir = r'D:\MSG\MSG_nat\batch_3'
    os.chdir(base_dir)
    unpack_all_in_dir(base_dir)

    time.sleep(10)

#fine the configuration of the functional chain to apply:
    chain_config = {"filter": "hrseviri_natural_color",
            "name": "Natural color disc",
            "id": "natural_color_disc",
            'product': 'HRSEVIRI',
            'format': 'netcdf4',
            'projection': 'geographic'
        }
    

    files = [f for f in os.listdir(base_dir) if f.endswith('.nat')]

    with open(r'C:\Users\Riccardo\Desktop\PhD_docs\Drought_prediction\ETH_adm.zip', 'rb') as f:
         shapefile_stream = f.read()

    for file in files:
        target_dir = r'D:\MSG\msg_data\batch_3'
        #n the chain and return the result as an `xarray` object
        output_xarray_dataset = api.run_chain_to_xarray(
           product_paths=[os.path.join(base_dir, file)],
           chain_config=chain_config,
           target_dir=target_dir,
           shapefile_stream=shapefile_stream
        )

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
        else:
            print('This product doesn\'t contain MSG1 or MSG2 Seviri')
        return xr_df


    target_dir = r'D:\MSG\msg_data\batch_3'
    files = [f for f in os.listdir(target_dir) if f.endswith('.nc')]
    for file in files:
        with xr.open_dataset(os.path.join(target_dir, file)) as ds:
            data = ds.load()
            xr_df = add_time(data)
            xr_df = compute_radiance(xr_df)
            xr_df = xr_df.drop('channel_3')
            xr_df = xr_df.assign(ndvi=(xr_df['channel_2'] - xr_df['channel_1']) / (xr_df['channel_2'] + xr_df['channel_1']))
            #xr_df['channel_2'].plot()
            xr_df.to_netcdf(os.path.join(base_dir,'processed', file)) 
            xr_df.close()

    

    #for item in test:
    #    if item.endswith(".zip"):
    #        os.remove(os.path.join(base_dir, item))


if __name__ == "__main__":
    cropping_pipeline()
