from p_drought_indices.functions.function_clns import load_config,subsetting_pipeline, cut_file
import os
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt


def subsetting(ds, CONFIG_PATH, countries: list = ['Ethiopia','Kenya','Somalia']):
    config = load_config(CONFIG_PATH)
    shapefile_path = config['SHAPE']['africa']
    gdf = gpd.read_file(shapefile_path)
    subset = gdf[gdf.ADM0_NAME.isin(countries)]
    xr_df =  cut_file(ds, subset)
    return xr_df

from p_drought_indices.functions.ndvi_functions import add_time


def subsetting_loop(CONFIG_PATH, path, save=True, plot=False,delete_grid_mapping=False, dest_folder="processed"):
    config = load_config(CONFIG_PATH)
    list = [f for f  in os.listdir(path) if f.endswith('.nc')]
    for file in list:
        ds = xr.open_dataset(os.path.join(path, file))
        ds = add_time(ds)
        if delete_grid_mapping ==True:
            del ds["cloud_mask"].attrs['grid_mapping']
        if "longitude" in ds.dims:
            ds = ds.rename({"latitude":"lat","longitude":"lon"})
        ds_sub = subsetting(CONFIG_PATH=CONFIG_PATH, ds= ds)
        if plot==True:
            
            print("Start plotting")
            for var in ds_sub.data_vars:
                ds_sub[var].plot()
                plt.show()
            ds_sub
        if save==True:
            ds_sub.to_netcdf(os.path.join(path, dest_folder, f"{file}"))


if __name__=="__main__":
    CONFIG_PATH = "config.yaml"
    path = r"/media/BIFROST/N2/Riccardo/CHIRPS/daily/CHIRPS/0.05/processed"
    #new_path = os.path.join(path,"processed")
    #subsetting_loop(CONFIG_PATH, path, save=True, plot=False)
    ds = xr.open_mfdataset(os.path.join(path,"*.nc"))
    ds["precip"].attrs["units"] = "mm"
    ds.to_netcdf(os.path.join(path,"chirps_05_merged.nc"))