import os
import cdsapi
from p_drought_indices.functions.function_clns import load_config
import xarray as xr

cofig_path = "config.yaml"
config = load_config(cofig_path)
cdo_key = config["CDO"]["key"]

c = cdsapi.Client(key = cdo_key ) #Replace UID:ApiKey with you UID and Api Key

years = list(range(1979, 2021))

dest_path = os.path.join(config["SPI"]["ERA5"]["path"], "ERA5_daily")

def data_collection(dest_path:str, years:list):

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
        dest_path + 'era5land_' + str(year) + '.nc')

        print('era5land_' + str(year) + '.nc' + ' downloaded.')

def process_era5(ds: xr.Dataset, var:str = "tp"):
    ds = ds.rename({"longitude":'lon', "latitude": "lat"})
    attrs = ds[var].attrs
    attrs['units']='mm'
    ds[var] = ds[var]*1000
    ds[var].attrs = attrs
    return ds


if __name__=="__main__":
    import numpy as np
    list_files = [os.path.join(dest_path, f) for f in os.listdir(dest_path) 
                  if f.endswith(".nc")]
    ds = xr.open_mfdataset(list_files)
    ds = process_era5(ds)
    ds.to_netcdf(os.path.join(config["SPI"]["ERA5"]["path"],"era5_land_merged.nc"))
    