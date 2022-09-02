import geopandas as gpd
import os
from shapely.geometry import Polygon, mapping
from glob import glob
import xarray as xr
from functions.function_clns import load_config, cut_file

def process_data():
    CONFIG_PATH = r"./config.yaml"

    config = load_config(CONFIG_PATH)
    list_files = glob(os.path.join(config['PRECIP']['imerg_path'],'*.nc'))
    ds = xr.open_mfdataset(list_files, chunks={"lat": -1, "lon": -1, "time": 12})
    shapefile_path = config['shapefiles']['africa']

    #### Chose subset of countries
    countries = ['Ethiopia','Somalia','Kenya']
    subset = gdf[gdf.ADM0_NAME.isin(countries)]
    gdf = gpd.read_file(shapefile_path)
    
    ### Clip file
    clipped = cut_file(ds, subset)
    output_file = os.path.join(config['PRECIP']['imerg_path'],'imerg_final_clipped_sek.nc')
    clipped.to_netcdf(output_file)


if __name__ == "__main__":
    process_data()
    
