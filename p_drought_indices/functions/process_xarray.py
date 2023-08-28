from p_drought_indices.functions.function_clns import load_config, print_raster, subsetting_pipeline, cut_file
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
    return xr_df.transpose('time','lon','lat')

def subsetting_loop(CONFIG_PATH, path, save=True, plot=False, dest_folder="processed"):
    config = load_config(CONFIG_PATH)
    list = [f for f  in os.listdir(path) if f.endswith('.nc')]
    for file in list:
        ds = xr.open_dataset(os.path.join(path, file))
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
    path = r"D:\shareVM\CHIRPS\daily\CHIRPS\0.05"
    files = [f for f  in os.listdir(path) if f.endswith('.nc')]
    #new_path = os.path.join(path,"processed")
    ds = xr.open_dataset(os.path.join(path ,files[0]))
    ds = ds.rename({"latitude":"lat","longitude":"lon"})
    ds = subsetting(CONFIG_PATH = CONFIG_PATH, ds =ds)
    ds["precip"].isel(time=0).plot()
    plt.show()
    #subsetting_loop(CONFIG_PATH, path, save=True, plot=True)


