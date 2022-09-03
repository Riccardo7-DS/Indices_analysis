import geopandas as gpd
import os
from shapely.geometry import Polygon, mapping
from glob import glob
import xarray as xr
import yaml


def load_config(CONFIG_PATH):
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)
        return config

def cut_file(xr_df, gdf):
    xr_df.rio.set_spatial_dims(x_dim='lat', y_dim='lon', inplace=True)
    xr_df.rio.write_crs("epsg:4326", inplace=True)
    clipped = xr_df.rio.clip(gdf.geometry.apply(mapping), gdf.crs, drop=True)
    return clipped

def process_data():
    CONFIG_PATH = r"./config.yaml"

    config = load_config(CONFIG_PATH)
    list_files = glob(os.path.join(config['PRECIP']['imerg_path'],'*.nc'))
    ds = xr.open_mfdataset(list_files, chunks={"lat": -1, "lon": -1, "time": 12}, concat_dim='time',combine='nested')
    shapefile_path = config['SHAPE']['africa']

    #### Chose subset of countries
    countries = ['Ethiopia','Somalia','Kenya']
    gdf = gpd.read_file(shapefile_path)
    subset = gdf[gdf.ADM0_NAME.isin(countries)]
    ### Clip file
    clipped = cut_file(ds, subset)
    clipped.pr.attrs['units']='mm'
    dims = {'lat','lon','time'}
    output_file = os.path.join(config['PRECIP']['imerg_path'],'imerg_final_clipped_sek.nc')
    clipped.to_netcdf(output_file)
    
