import xarray as xr
from typing import Union
from xarray import DataArray, Dataset
from pathlib import Path
from datetime import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.function_clns import load_config
import re

def plot_random_masked_over_time(data_array1: Union[xr.DataArray, xr.Dataset],
                                 mask: Union[xr.DataArray, xr.Dataset, None] = None,
                                 n_points: int = 1):

    # Step 1: Apply boolean mask
    valid_indices = np.where(mask == 1)

    # Step 2: Generate random indices
    random_indices = np.random.choice(len(valid_indices[0]), 
                                      size=n_points, replace=False)

    # Step 3: Retrieve latitudes and longitudes corresponding to selected indices
    selected_lats = data_array1.lat.values[valid_indices[0][random_indices]]
    selected_lons = data_array1.lon.values[valid_indices[1][random_indices]]

    # Step 4: Generate time axis
    time_axis = data_array1.time.values

    # Step 5: Plot latitude-longitude combinations over time
    plt.figure(figsize=(10, 6))
    for i in range(n_points):
        plt.plot(time_axis, data_array1.sel(lat=selected_lats[i], 
                lon=selected_lons[i]), 
                label=f'Lat: {selected_lats[i]}, Lon: {selected_lons[i]}')

    plt.xlabel('Time')
    plt.ylabel('Data Variable')
    plt.title('Latitude-Longitude Combinations Over Time')
    plt.grid(True)
    plt.show()

class VizNC():
    def __init__(self, root: Path, data:str,  period_start:str, period_end:str, plot:str='first', dims_mean=["time"]):
        '''
        root: the root directory of the data
        data: the name of the xarray dataset/specify "*.nc" to merge all the .nc files in the directory
        period_start: select either "min" for the smallest date or a date in format "%Y-%m-%d"
        period_end: select either "max" for the smallest date or a date in format "%Y-%m-%d"
        plot: chose if "first"/"all" to print only the first or all the variables
        dims_mean: dimensions over which to make a mean, either "time" or ["lat","lon"]
        '''
        assert plot in ['first', 'all']
        if data =="*.nc" in data:
            ds = xr.open_mfdataset(os.path.join(root,data))
        else:
            self.name = [f for f in os.listdir(root) if f.endswith('.nc') and data in f][0]
            data_dir = os.path.join(root, self.name)
            ds = xr.open_dataset(data_dir)
        
        vars_ = list(ds.data_vars)
        print(f"There are {len(vars_)} variables in the dataset: {vars_}")

        title, ylabel, cmap = self.get_plot_attrs(data)

        if period_start =="min":
            t = ds["time"].min().values
            period_start = str(np.datetime_as_string(t, unit='D'))
        if period_end =="max":
            t = ds["time"].max().values
            period_end = str(np.datetime_as_string(t, unit='D'))
        ds_sub = ds.sel(time=(slice(period_start, period_end)))
        if plot == "first":
            ds_sub[vars_[0]].mean(dims_mean).plot(cmap=cmap)
            if title:
                plt.title(title)
            plt.show()
        elif plot =="all":
            for var in vars_:
                ds_sub[var].mean(dims_mean).plot(cmap=cmap)
                if title:
                    plt.title(title)
                plt.show()

    def get_plot_attrs(self, data):
        if data =="vci_1D.nc":
            title = "mean VCI"
            ylabel = "VCI"
            cmap = plt.cm.YlGn
        elif "spi" in data:
            cmap = plt.cm.RdBu
            title = "SPI (Standard Precipitation Index)"
            ylabel = "SPI"
        else:
            title = None
            ylabel = None
            cmap=None
        return title, ylabel, cmap



from utils.xarray_functions import ndvi_colormap
from datetime import timedelta
import pandas as pd
import math
import cartopy.crs as ccrs

def plot_ndvi_days(dataset:xr.DataArray,
                   start_day:Union[str,None],
                   num_timesteps:int,
                   vmin: float = -0.2,
                   vmax: float = 1.,
                   cmap:Union[str, None]=None,
                   cities:Union[dict, None]=None)-> None:

    """ Function to plot NDVI daily series"""

    if "time" != dataset.dims[0]:
        dataset = dataset.transpose("time","lat","lon")

    if cmap is None:
        cmap = ndvi_colormap()

    if start_day ==None:
        start_day = "2007-08-01"
        print(f"Given that no day has been specified chosing sample day {start_day}")

    # Select the desired number of timesteps to plot
    date_end = pd.to_datetime(start_day) 
    new_end = date_end + timedelta(days = num_timesteps - 1)
    end = new_end.strftime('%Y-%m-%d')

    # Select the first 'num_timesteps' timesteps
    ndvi_subset =  dataset.sel(time=slice(start_day, end))

    # Determine the number of rows and columns for subplots
    square_root = math.sqrt(num_timesteps)
    num_rows = math.ceil(square_root)
    num_cols = math.ceil(num_timesteps/ num_rows)

    # Create a figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 3*num_rows), subplot_kw={'projection':ccrs.PlateCarree()})

    # Flatten the axes array to simplify indexing
    axes = axes.flatten()

    # Iterate over the timesteps and plot each one
    for i, ds in enumerate(ndvi_subset):
        ts = pd.to_datetime(str(ds["time"].values)) 
        d = ts.strftime('%Y-%m-%d')
        # Assuming the time coordinate is named 'time'
        ax = axes[i]
        # ndvi.plot(ax=ax, cmap=cmap_custom)
        p = ax.pcolormesh(ds, vmin=vmin, vmax=vmax, cmap=cmap,
                            transform=ccrs.PlateCarree())
        ax.set_title(f'Day {d}')
        
        # Adding latitude and longitude labels
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        if len(ndvi_subset.lon) > 10:
            lon_step = int(len(ndvi_subset.lon)//10)
            lat_step = int(len(ndvi_subset.lat)//10)
        else:
            lon_step = len(ndvi_subset.lon)
            lat_step = len(ndvi_subset.lat)

        ax.set_xticks(np.arange(0, len(ndvi_subset.lon), step=lon_step))  # Adjust the step size as needed
        ax.set_yticks(np.arange(0, len(ndvi_subset.lat), step=lat_step))  # Adjust the step size as needed
        ax.set_xticklabels(ndvi_subset.lon[::lon_step].values.round(2), rotation=45)
        ax.set_yticklabels(ndvi_subset.lat[::lat_step].values.round(2))

        if cities is not None:
            for city, (lat, lon) in cities.items():
                ax.scatter(lon, lat, color='grey', marker='o', transform=ccrs.PlateCarree())
                ax.text(lon, lat, city, color='black', fontsize=8, transform=ccrs.PlateCarree())


    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([1, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    fig.colorbar(p, cax=cbar_ax, orientation='vertical', label='NDVI')
    # Remove any extra empty subplots
    if len(ndvi_subset) < len(axes):
        for j in range(len(ndvi_subset), len(axes)):
            fig.delaxes(axes[j])

    # Adjust the layout and spacing
    fig.tight_layout()

    # Display the plot
    plt.show()
