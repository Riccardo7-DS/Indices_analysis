import xarray as xr
from typing import Union
from xarray import DataArray, Dataset
from pathlib import Path
from datetime import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
from p_drought_indices.functions.function_clns import load_config
import re

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



from p_drought_indices.functions.ndvi_functions import ndvi_colormap
from datetime import timedelta
import pandas as pd

def plot_ndvi_days(ds_smoot:xr.Dataset,start_day:Union[str,None],num_timesteps:int,var:str="ndvi")-> None:

    """ Function to plot NDVI saries"""

    cmap_custom = ndvi_colormap()

    if start_day ==None:
        start_day = "2007-08-01"
        print(f"Given that no day has been specified chosing sample day {start_day}")

    # Select the desired number of timesteps to plot
    date_end = pd.to_datetime(start_day) 
    new_end = date_end + timedelta(days = num_timesteps - 1)
    end = new_end.strftime('%d-%m-%Y')

    # Assuming 'ndvi' is a DataArray within the 'data' dataset
    ndvi_data = ds_smoot[var]

    # Select the first 'num_timesteps' timesteps
    ndvi_subset =  ndvi_data.sel(time=slice(start_day, end))

    # Determine the number of rows and columns for subplots
    num_rows = (num_timesteps + 4) // 5
    num_cols = min(num_timesteps, 5)

    # Create a figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 3*num_rows))

    # Flatten the axes array to simplify indexing
    axes = axes.flatten()

    # Iterate over the timesteps and plot each one
    for i, ndvi in enumerate(ndvi_subset):
        ts = pd.to_datetime(str(ndvi["time"].values)) 
        d = ts.strftime('%d-%m-%Y')
        # Assuming the time coordinate is named 'time'
        ax = axes[i]
        ndvi.plot(ax=ax, cmap=cmap_custom)
        ax.set_title(f'Day {d}')

    # Remove any extra empty subplots
    if len(ndvi_subset) < len(axes):
        for j in range(len(ndvi_subset), len(axes)):
            fig.delaxes(axes[j])

    # Adjust the layout and spacing
    fig.tight_layout()

    # Display the plot
    plt.show()


if __name__== "__main__":
    period_start = "2010-01-01"
    period_end = "2012-01-01"

    CONFIG = "./config.yaml"
    config = load_config(CONFIG)

        ### Visualize NDVI series
     #   path_ndvi = Path(config['NDVI']['ndvi_prep'])
     #   files = "*.nc"
 #   VizNC(path_ndvi, data=files, period_start=period_start, period_end=period_end)

    ### Visualize VCI
    #path_vci = Path(config['NDVI']['ndvi_path'])
    #file = "vci_1D.nc"
    #VizNC(path_vci, data=file, period_start=period_start, period_end=period_end)

    ### Visualize SPIs
    latency = 60
    
    for base_path in [config['SPI']['CHIRPS']['path'], config['SPI']['IMERG']['path'], config['SPI']['ERA5']['path'], config['SPI']['GPCC']['path']]:
        file_n = f"_spi_gamma_{latency}"
        file = [f for f in os.listdir(base_path) if re.search('(.*){}(.nc)'.format(file_n), f)][0]
        VizNC(base_path, data=file, period_start=period_start, period_end=period_end)
