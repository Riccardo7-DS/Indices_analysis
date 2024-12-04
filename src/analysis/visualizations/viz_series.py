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

from typing import Union
import numpy as np

def select_random_points(mask: Union[xr.DataArray, xr.Dataset, None] = None,
                         n_points: int = 1):
    # Step 1: Apply boolean mask
    valid_indices = np.where(mask == 1)

    # Step 2: Generate random indices
    random_indices = np.random.choice(len(valid_indices[0]), 
                                      size=n_points, replace=False)
    
    return valid_indices, random_indices

def plot_random_masked_over_time(data_array1: Union[xr.DataArray, xr.Dataset],
                                 valid_indices:tuple, 
                                 random_indices:np.ndarray,
                                 date_min:str = None,
                                 date_max:str = None):
    
    if date_max and date_max is not None:
        data_array1 = data_array1.sel(time=slice(date_min, date_max))

    n_points = len(random_indices)

    # Step 3: Retrieve latitudes and longitudes corresponding to selected indices
    selected_lats = data_array1.lat.values[valid_indices[0][random_indices]]
    selected_lons = data_array1.lon.values[valid_indices[1][random_indices]]

    # Step 4: Generate time axis
    time_axis = data_array1.time.values

    # Step 5: Plot latitude-longitude combinations over time
    plt.figure(figsize=(10, 6))
    for i in range(n_points):
        plt.plot(time_axis, data_array1.sel(lat=selected_lats[i], lon=selected_lons[i]),
                 label=f'Lat: {selected_lats[i]}, Lon: {selected_lons[i]}', 
                 color=plt.cm.viridis(i / n_points), linestyle='-', marker='o', markersize=4, alpha=0.7)

    plt.xlabel('Time')
    plt.ylabel('Data Variable')
    plt.title('Latitude-Longitude Combinations Over Time')
    plt.grid(True)
    plt.tight_layout()
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


"""
Functions for visualizing predicted models
"""

def process_error_by_time_aggregation(y, y_pred, start, aggregate_by="day", new_start=None, new_end=None):
    """
    Process average error aggregated by calendar day or month across multiple years.

    Parameters:
        y (np.ndarray): Ground truth data with shape [s, h, w].
        y_pred (np.ndarray): Predicted data with shape [s, h, w].
        start (str): Starting date in the format 'YYYY-MM-DD'.
        aggregate_by (str): Aggregation level ("day" or "month").
        new_start (str): Optional. Start of the subset range in 'YYYY-MM-DD'.
        new_end (str): Optional. End of the subset range in 'YYYY-MM-DD'.

    Returns:
        pd.Series: Aggregated average error indexed by time keys.
    """
    # Create a date range corresponding to the days in the dataset
    days = y.shape[0]
    start_pd = pd.to_datetime(start)
    end_pd = start_pd + timedelta(days=days - 1)
    range_dates = pd.date_range(start_pd, end_pd)

    # Subset the data if a range is specified
    if new_start is not None and new_end is not None:
        new_st_loc = range_dates.get_loc(pd.to_datetime(new_start))
        new_end_loc = range_dates.get_loc(pd.to_datetime(new_end))
        y = y[new_st_loc:new_end_loc + 1]
        y_pred = y_pred[new_st_loc:new_end_loc + 1]
        range_dates = range_dates[new_st_loc:new_end_loc + 1]

    # Calculate daily error for the entire image
    daily_error = np.nanmean(np.abs(y - y_pred), axis=(1, 2))  # Shape: [s]

    # Group errors by calendar day or month
    df = pd.DataFrame({'date': range_dates, 'error': daily_error})
    if aggregate_by == "day":
        df['time_key'] = df['date'].dt.strftime('%m-%d')  # Day of the year
        average_error_by_time = df.groupby('time_key')['error'].mean()
    elif aggregate_by == "month":
        df['time_key'] = df['date'].dt.month_name()  # Month name
        df['month_order'] = df['date'].dt.month  # Month number for sorting
        average_error_by_time = df.groupby(['time_key', 'month_order'])['error'].mean().reset_index()
        average_error_by_time = average_error_by_time.sort_values('month_order')
        average_error_by_time = average_error_by_time.set_index('time_key')['error']
    else:
        raise ValueError("Invalid value for 'aggregate_by'. Choose 'day' or 'month'.")

    return average_error_by_time



def plot_multiple_aggregations(data_dict, model_name, aggregate_by="day"):
    """
    Plot multiple aggregated error series on a single plot.

    Parameters:
        data_dict (dict): Dictionary where keys are labels (e.g., years) and values are pd.Series
                          with aggregated errors (indexed by time keys).
        aggregate_by (str): Aggregation level ("day" or "month").
    """
    plt.figure(figsize=(14, 7))

    # Loop through the data dictionary to plot each series
    for label, data in data_dict.items():
        plt.plot(data.index, data.values, label=label, linestyle='-', marker='.')

    # Adjust the x-axis ticks and labels
    if aggregate_by == "day":
        # Show every 15th day for clarity
        tick_indices = np.arange(0, len(list(data_dict.values())[0]), 15)
        tick_labels = pd.to_datetime(
            list(data_dict.values())[0].index[tick_indices], format='%m-%d'
        ).strftime('%d-%b')
        plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)
        plt.xlabel('Day of the Year', fontsize=14)
    elif aggregate_by == "month":
        # Use all months for clarity
        tick_indices = np.arange(len(list(data_dict.values())[0]))
        tick_labels = list(data_dict.values())[0].index
        plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)
        plt.xlabel('Month', fontsize=14)
    else:
        raise ValueError("Invalid value for 'aggregate_by'. Choose 'day' or 'month'.")

    # Final plot settings
    plt.title(f"{model_name} Average Error Aggregation by {aggregate_by.capitalize()}", fontsize=16)
    plt.ylabel('Average Error', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_error_time(y, y_pred, start, new_start=None, new_end=None):
    """
    Plot yearly mean error from daily data.

    Parameters:
        y (np.ndarray): Ground truth data with shape [s, h, w].
        y_pred (np.ndarray): Predicted data with shape [s, h, w].
        start (str): Starting date in the format 'YYYY-MM-DD'.
        new_start (str): Optional. Start of the subset range in 'YYYY-MM-DD'.
        new_end (str): Optional. End of the subset range in 'YYYY-MM-DD'.
    """
    # Create a date range corresponding to the days in the dataset
    days = y.shape[0]
    start_pd = pd.to_datetime(start)
    end_pd = start_pd + timedelta(days=days - 1)
    range_dates = pd.date_range(start_pd, end_pd)

    # Subset the data if a range is specified
    if new_start is not None and new_end is not None:
        new_st_loc = range_dates.get_loc(pd.to_datetime(new_start))
        new_end_loc = range_dates.get_loc(pd.to_datetime(new_end))
        y = y[new_st_loc:new_end_loc + 1]
        y_pred = y_pred[new_st_loc:new_end_loc + 1]
        range_dates = range_dates[new_st_loc:new_end_loc + 1]

    # Calculate daily error for the entire image
    daily_error = np.nanmean(np.abs(y - y_pred), axis=(1, 2))  # Shape: [s]

    # Group errors by year
    df = pd.DataFrame({'date': range_dates, 'error': daily_error})
    df['year'] = df['date'].dt.year
    yearly_mean_error = df.groupby('year')['error'].mean()

    # Plot yearly mean error
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_mean_error.index, yearly_mean_error.values, marker='o', linestyle='-', color='b')
    plt.title('Yearly Mean Error', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Mean Error', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(yearly_mean_error.index, rotation=45)
    plt.tight_layout()
    plt.show()

def plot_models_errors_overtime(results, model_name):
    """
    Plot yearly mean errors for multiple models stored in a dictionary with custom date ranges.

    Parameters:
        results (dict): Dictionary containing ground truth, predictions, masks, and custom start/end dates.
                        Keys for each model day are:
                        - 'y_<day>', 'y_pred_<day>', 'mask_<day>', 'start_<day>', 'end_<day>'.
        model_name (str): The name of the model being plotted, for use in the plot title.
    """
    plt.figure(figsize=(10, 6))

    for model_day in [10, 15, 30]:
        # Extract data and custom date range for the current model day
        y = results.get(f'y_{model_day}')
        y_pred = results.get(f'y_pred_{model_day}')
        start = results.get(f'start_{model_day}')
        end = results.get(f'end_{model_day}')

        # Ensure all required data exists
        if y is None or y_pred is None or start is None or end is None:
            print(f"Missing data for model day {model_day}. Skipping.")
            continue

        # Create a date range corresponding to the days in the dataset
        start_pd = pd.to_datetime(start)
        end_pd = pd.to_datetime(end)
        range_dates = pd.date_range(start_pd, end_pd, periods=y.shape[0])

        # Calculate daily error for the entire image
        daily_error = np.nanmean(np.abs(y - y_pred), axis=(1, 2))  # Shape: [s]

        # Group errors by year
        df = pd.DataFrame({'date': range_dates, 'error': daily_error})
        df['year'] = df['date'].dt.year
        yearly_mean_error = df.groupby('year')['error'].mean()

        # Plot yearly mean error for the current model day
        plt.plot(
            yearly_mean_error.index,
            yearly_mean_error.values,
            marker='o',
            linestyle='-',
            label=f'Model Day {model_day}'
        )

    # Finalize the plot
    plt.title(f"Yearly Mean Errors for Model {model_name} at Different Steps", fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Mean Error', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
