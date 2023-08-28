import os
import cdsapi
from p_drought_indices.functions.function_clns import load_config
import xarray as xr
import numpy as np
import ee



def data_collection(config_path:str, dest_path:str, years:list):
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
    ds = ds.rename({"longitude":'lon', "latitude": "lat"})
    attrs = ds[var].attrs
    attrs['units']='mm'
    ds[var] = ds[var]*1000
    ds[var].attrs = attrs
    return ds

def ee_collection(start_date:str, end_date:str):

# Initialize Earth Engine
    ee.Initialize()

    # Define the combined region of interest for the Horn of Africa
    horn_of_africa_roi = ee.Geometry.Rectangle([32, -5, 51, 15])

    # Load ERA5 Land daily data collection
    era5_land = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_RAW')

    # Filter the collection by date and region
    filtered_era5 = era5_land.filterDate(ee.Date(start_date), ee.Date(end_date)).filterBounds(horn_of_africa_roi)

    # Select precipitation band and scale it
    def process_image(image):
        precipitation = image.select('total_precipitation_sum')
        return precipitation.multiply(0.1)

    processed_era5 = filtered_era5.map(process_image)

    # Create a single image by reducing the collection
    horn_of_africa_precipitation = processed_era5.sum()

    # Download the aggregated data
    download_options = {
        'scale': 1000,
        'crs': 'EPSG:4326',
        'fileFormat': 'GeoTIFF',
        'region': horn_of_africa_roi
    }

    # Start the download task to Google Drive
    task = ee.batch.Export.image.toDrive(
        image=horn_of_africa_precipitation,
        description='ERA5_Land_HornOfAfrica',
        folder="era5_download",
        **download_options
    )

    return task


if __name__=="__main__":
    from p_drought_indices.precipitation.data_collection.era5_daily_data import ee_collection
    import xarray as xr
    import os
    from p_drought_indices.functions.function_clns import load_config, subsetting_pipeline
    import ee
    ee.Authenticate()

    CONFIG_PATH= "../config.yaml"
    config = load_config(CONFIG_PATH)
    # Define the date range
    start_date = '1970-01-01'
    end_date = '2020-12-31'
    output_path = os.path.join(config["SPI"]["ERA5"]["path"],'daily/ee_era5_ecmwf.tif')
    task = ee_collection(start_date, end_date)

    