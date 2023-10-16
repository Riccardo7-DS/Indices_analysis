import os
from p_drought_indices.functions.function_clns import load_config
import xarray as xr
import numpy as np
from loguru import logger


def data_collection(config_path:str, dest_path:str, years:list):
    import cdsapi

    config = load_config(config_path)
    cdo_key = config["CDO"]["key"]
    c = cdsapi.Client(key = cdo_key ) #Replace UID:ApiKey with you UID and Api Key
    years = list(range(1979, 2021))
    dest_path = os.path.join(config["SPI"]["ERA5"]["path"], "ERA5_daily")

    for year in years:
        c.retrieve(
        'reanalysis-era5-land',
        {
            'variable': [
                'total_precipitation',
            ],
            'year': str(year),
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00',
            ],
            'area': [
                -5, 15, 32.7, 51.5, # Bounding box for Horn of Africa
            ],
            'format': 'netcdf',
        },
        os.path.join(dest_path, ('era5land_' + str(year) + '.nc')))

        print('era5land_' + str(year) + '.nc' + ' downloaded.')

def pipeline_era5_collection_cds(config_path):
    config = load_config(config_path)
    years = list(range(1979, 2021))
    dest_path = os.path.join(config["SPI"]["ERA5"]["path"], "ERA5_daily")
    data_collection(config_path, dest_path, years)
    list_files = [os.path.join(dest_path, f) for f in os.listdir(dest_path) 
                  if f.endswith(".nc")]
    ds = xr.open_mfdataset(list_files)
    ds = process_era5(ds)
    ds.to_netcdf(os.path.join(config["SPI"]["ERA5"]["path"],"era5_land_merged.nc"))

def process_era5(ds: xr.Dataset, var:str = "tp"):
    if "latitude" in ds.dims:
        ds = ds.rename({"longitude":'lon', "latitude": "lat"})
    attrs = ds[var].attrs
    attrs['units']='mm'
    ds[var] = ds[var]*1000
    ds[var].attrs = attrs
    return ds

"""
Functions to collect ERA5 data with google cloud
"""

def query_era5_gs(CONFIG_PATH):
    import fsspec
    fs = fsspec.filesystem('gs')
    import xarray as xr
    import os
    from loguru import logger
    from p_drought_indices.functions.function_clns import load_config, subsetting_pipeline

    config = load_config(CONFIG_PATH)
    ar_full_37_1h = xr.open_zarr(
        'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2/',
         chunks={'time': 48},
         consolidated=True
    ).rename({"latitude":"lat", "longitude":"lon"})
        
    test_ds = subsetting_pipeline(CONFIG_PATH, ar_full_37_1h)
    test_ds = process_era5(test_ds, var="total_precipitation")
    logger.info("Starting resampling from hourly to daily...")
    test_ds = test_ds["total_precipitation"].resample(time="1D").sum().to_dataset()

    logger.info("Saving dataset locally...")
    compress_kwargs = {"total_precipitation": {'zlib': True, 'complevel': 4}} # You can adjust 'complevel' based on your needs
    test_ds.to_netcdf(os.path.join(config["SPI"]["ERA5"]["path"], "era5_total_precipitation_gc.nc"),
                      encoding=compress_kwargs)

"""
Functions to collect ERA5 data with Earth Engine
"""

def ee_collection(start_date:str, end_date:str):
    import ee

# Initialize Earth Engine
    ee.Initialize()

    # Define the combined region of interest for the Horn of Africa
    horn_of_africa_roi = ee.Geometry.Rectangle([32, -5, 51, 15])

    # Load ERA5 Land daily data collection
    era5_land = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_RAW')\
        .filterBounds(horn_of_africa_roi)\
        .select('total_precipitation_sum')\
        .filterDate(ee.Date(start_date), ee.Date(end_date))

    # Define a function to export images for each date
    # Define a function to export images for each date
    def export_image(image):
        image_date = ee.Date(image.get('system:time_start'))
        description = f'ERA5_Land_HornOfAfrica_{image_date.format("YYYY-MM-dd").getInfo()}'
    
        download_options = {
            'scale': 1000,
            'crs': 'EPSG:4326',
            'region': horn_of_africa_roi,
            'fileFormat': 'GeoTIFF'
        }

        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder="era5_precipitation",
            **download_options
        )

        task.start()
    
    # Loop through the processed collection and export images
    processed_era5_list = era5_land.toList(era5_land.size())
    for i in range(era5_land.size().getInfo()):
        export_image(ee.Image(processed_era5_list.get(i)))
    

def collect_missing_days(CONFIG_PATH:str):
    import xarray as xr
    import os
    import numpy as np
    from p_drought_indices.functions.function_clns import load_config, subsetting_pipeline

    config_file = load_config(CONFIG_PATH)
    path = os.path.join(config_file["SPI"]["ERA5"]["path"], "era5_land_merged.nc")
    ds = subsetting_pipeline(CONFIG_PATH, xr.open_dataset(path))

    mask = ds['tp'] < 0
    arr = mask.sum(dim=["lat","lon"])
    print("Days with errors:", len(np.where(arr>0)[0]))

    list_arr = np.where(arr>0)[0]
    days = [np.datetime_as_string(ds.isel(time=i)["time"].values, unit="D") for i in list_arr]
    
    import pandas as pd
    from datetime import timedelta
    from tqdm.auto import tqdm
    from p_drought_indices.precipitation.data_collection.era5_daily_data import ee_collection
    import xarray as xr
    import os
    from p_drought_indices.functions.function_clns import load_config, subsetting_pipeline

    for day in tqdm(days):
        print(f"Collecting day {day}...")
        new_day = pd.to_datetime(day) + timedelta(days=1)
        end_day = new_day.strftime("%Y-%m-%d")
        ee_collection(day, end_day)

def check_wrong_precp_days(base_path:str):
    from p_drought_indices.functions.function_clns import subsetting_pipeline
    import pandas as pd
    path = os.path.join(base_path, "era5_land_merged.nc")
    precp_ds = subsetting_pipeline(CONFIG_PATH, xr.open_dataset(path))
    ### generate mask and count days  smaller than 0
    mask = precp_ds['tp'] < 0
    arr = mask.sum(dim=["lat","lon"])
    print("Days with errors:", len(np.where(arr>0)[0]))

    list_arr = np.where(arr>0)[0]
    days = [np.datetime_as_string(precp_ds.isel(time=i)["time"].values, unit="D") for i in list_arr]

    ds = subsetting_pipeline(xr.open_dataset(os.path.join(base_path, "era5_precp_batch.nc"), chunks={"time":"200MB" }))
    len(np.unique(ds["time"].values))

    list_array = [pd.to_datetime(i).strftime("%Y-%m-%d") for i in ds["time"].values]
    print(len(set(list_array)))
    missing_days = [d for d in days if d not in list_array]
    return missing_days

def convert_tif_netcdf(CONFIG_PATH:str, base_path:str, path:str, dest_path, dest_filename:str = "era5_precp_batch.nc", use_gdal:bool=False,
                       compress:bool=False):
    import xarray as xr
    import glob
    import os
    import pandas as pd
    from tqdm.auto import tqdm

    # List of downloaded GeoTIFF file paths
    geotiff_files = glob.glob(os.path.join(path,'*.tif'))

    def time_index_from_filenames(filenames):
        """helper function to create a pandas DatetimeIndex
        Filename example: 20150520_0164.tif"""
        return pd.DatetimeIndex([pd.Timestamp(f[-14:-4]) for f in filenames])
    
    ## get time variables
    time = xr.Variable('time', time_index_from_filenames(geotiff_files))

    if use_gdal is True:
        print("Using GDAL for transforming tif to netcdf...")
        from osgeo import gdal

        kwargs = {
        'format': 'NetCDF',
        'outputType': gdal.GDT_Float32
    }

        for file in tqdm(geotiff_files):
            ### convert files
            filename = file.split("\\")[-1][-14:-4] + ".nc"
            dst_file = os.path.join(dest_path, filename)
            gdal.Translate(dst_file, file, **kwargs)

        list_files = [os.path.join(dest_path, f) for f in os.listdir(dest_path) if f.endswith(".nc")]
        combined_dataset = xr.concat([xr.open_dataset(f, engine="netcdf4") for f in list_files], dim=time)
        ds = combined_dataset.rename({"Band1":"tp"}).drop_indexes(["lat","lon","time"])
        #from p_drought_indices.functions.function_clns import subsetting_pipeline
        #ds = subsetting_pipeline(CONFIG_PATH, ds)
        ds["tp"] = ds["tp"]/1000
        ds["tp"].attrs['units'] = "mm"
    
    else:
        combined_dataset = xr.Dataset()
        combined_dataset = xr.concat([xr.open_dataarray(f, engine="rasterio") for f in geotiff_files], dim=time)
        
        ds = combined_dataset.sel(band=1).drop({"band"})["band_data"].rename({"x":"lon", "y":"lat"}).to_dataset().rename({"band_data":"tp"})
        #ds = subsetting_pipeline(CONFIG_PATH, ds)
        ds["tp"] = ds["tp"]*1000
        ds = ds.drop_indexes(["lat","lon","time"])
        ds["tp"].attrs['units'] = "mm"
        ds["tp"] = ds["tp"].astype(np.float32)

    # Save the combined dataset as a NetCDF file
    if compress is True:
        print("Starting compressing and exporting file...")
        # Set compression options
        compress_kwargs = {"tp": {'zlib': True, 'complevel': 4}} # You can adjust 'complevel' based on your needs
        # Save the dataset to a compressed NetCDF4 file
        ds.to_netcdf(os.path.join(base_path, dest_filename), format='NETCDF4', engine='netcdf4', encoding=compress_kwargs)
    else:
        print("Starting exporting file...")
        ds.to_netcdf(os.path.join(base_path, dest_filename), format='NETCDF4', engine='netcdf4')

if __name__=="__main__":
    from loguru import logger
    import sys
    logger.add(sys.stdout, format="{time} - {level} - {message}", level="INFO")
    CONFIG_PATH= "config.yaml"
    query_era5_gs(CONFIG_PATH)
    