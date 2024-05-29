import os
import xarray as xr
import numpy as np
from typing import Union, Literal
import logging
logger = logging.getLogger(__name__)

def cdo_api(variables:Union[list, None]=None,
            dest_path:str=None, 
            years:list=None, 
            area:Union[None, list]=None,
            area_name:Literal["HOA", "Africa", None]=None):
    
    import cdsapi
    from tqdm.auto import tqdm
    from utils.function_clns import config

    cdo_key = config["CDO"]["key"]
    c = cdsapi.Client(key = cdo_key ) #Replace UID:ApiKey with you UID and Api Key
    if dest_path is None:
        dest_path = os.path.join(config["SPI"]["ERA5"]["path"], "ERA5_daily")

    if variables is None:
        ['total_precipitation',"2m_temperature", "total_evaporation", "potential_evaporation"]
    
    if area is None:
        #W, E, S, N
        if area_name == "Africa":
            area = [-17.48122, 50.360668,-34.463232, 25.422785]
        elif area_name == "HOA":
            area = [32.01630435, 51.48369565, -5.48369565, 15.48369565]

    year_min = 2005
    year_max = 2023

    if years is None:
        years = list(range(year_min, year_max+1))

    logger.info(f"Collecting variables {variables} for years {year_min}-{year_max}")

    for year in tqdm(years):
        c.retrieve(
        'reanalysis-era5-land',
        {
            'variable': variables,
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
            'area': area,
                #W, E, S, N
                #-5, 15, 32.7, 51.5, # Bounding box for Horn of Africa
            'format': 'netcdf',
        },
        os.path.join(dest_path, ('era5land_' + str(year) + '.nc')))

        print('era5land_' + str(year) + '.nc' + ' downloaded.')

def pipeline_era5_collection_cds(config_path):
    from utils.function_clns import config
    years = list(range(1979, 2021))
    dest_path = os.path.join(config["SPI"]["ERA5"]["path"], "ERA5_daily")
    cdo_api(dest_path=dest_path, years=years)
    list_files = [os.path.join(dest_path, f) for f in os.listdir(dest_path) 
                  if f.endswith(".nc")]
    ds = xr.open_mfdataset(list_files)
    ds = process_era5_precp(ds)
    ds.to_netcdf(os.path.join(config["SPI"]["ERA5"]["path"],"era5_land_merged.nc"))

def process_era5_precp(ds: xr.Dataset, var:str = "tp"):
    if "latitude" in ds.dims:
        ds = ds.rename({"longitude":'lon', "latitude": "lat"})
    attrs = ds[var].attrs
    attrs['units']='mm'
    ds[var] = ds[var]*1000
    ds[var].attrs = attrs
    return ds

"""
Functions to collect ERA5 ARCO data with google cloud storage
"""

def query_arco_era5(vars:list,
                    date_min:str=None,
                    date_max:str=None,
                    bounding_box:list=None,   
                    library_open:Literal["zarr", "xarray"] = "zarr",
                    chunks:dict={'time': -1, "latitude": "auto", "longitude":"auto"}):
    import xarray as xr
    import gcsfs
    from utils.zarr import handle_gcs_zarr, load_zarr_arrays

    url = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/'
    
    if library_open== "zarr":
        if date_max is None or date_min is None or bounding_box is None:
            error = ValueError("Please provide bounding box, maximum and minimum dates parameters")
            logger.error(error)
            raise error
        xr_ds = xr.open_zarr(url, chunks=chunks, consolidated=True)
        store = handle_gcs_zarr(url)
        test_ds = load_zarr_arrays(store, vars, date_min, date_max, bounding_box, xr_ds)

    elif library_open == "xarray":
        ds = xr.open_zarr(
            url,
            chunks=chunks,
            consolidated=True
        )
        test_ds = ds.rename({"latitude":"lat", "longitude":"lon"})
    
    if vars is not None:
        return test_ds[vars]
    else:
        test_ds

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
    

def collect_missing_days():
    import xarray as xr
    import os
    import numpy as np
    from utils.function_clns import config as config_file, subsetting_pipeline

    path = os.path.join(config_file["SPI"]["ERA5"]["path"], "era5_land_merged.nc")
    ds = subsetting_pipeline(xr.open_dataset(path))

    mask = ds['tp'] < 0
    arr = mask.sum(dim=["lat","lon"])
    print("Days with errors:", len(np.where(arr>0)[0]))

    list_arr = np.where(arr>0)[0]
    days = [np.datetime_as_string(ds.isel(time=i)["time"].values, unit="D") for i in list_arr]
    
    import pandas as pd
    from datetime import timedelta
    from tqdm.auto import tqdm
    from precipitation.data_collection.era5_daily_data import ee_collection
    import xarray as xr
    import os
    from utils.function_clns import load_config, subsetting_pipeline

    for day in tqdm(days):
        print(f"Collecting day {day}...")
        new_day = pd.to_datetime(day) + timedelta(days=1)
        end_day = new_day.strftime("%Y-%m-%d")
        ee_collection(day, end_day)

def check_wrong_precp_days(base_path:str):
    from utils.function_clns import subsetting_pipeline
    import pandas as pd
    path = os.path.join(base_path, "era5_land_merged.nc")
    precp_ds = subsetting_pipeline(xr.open_dataset(path))
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
    