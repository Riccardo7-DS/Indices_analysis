import xarray as xr
from typing import Union
from xarray import DataArray, Dataset
from pathlib import Path
from datetime import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.function_clns import load_config
import seaborn as sns
from typing import Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_ndvi_output(config, model, days, features=90, vars:str=None, basemask:Union[int, None]=None):
    """
    Function to load the output of the DL models
    """
    assert model in ["dime", "gwnet", "wnet", "convlstm", "dime_attn"]
    if vars is None:
        basepath = config.output_dir + "/{m}/days_{d}/features_{f}/images/output_data"
        path = basepath.format(m=model, d=days, f=features)
    else:
        assert vars in ["autoenc", "climate"] 
        basepath = config.output_dir + "/{m}/{v}/days_{d}/features_{f}/images/output_data"
        path = basepath.format(v=vars, m=model, d=days, f=features)

    if basemask:
        if vars is None:
            maskpath = basepath.format(m=basemask["model"], d=basemask["days"], f=features)
        else:
            maskpath = basepath.format(v=vars, m=basemask["model"], d=basemask["days"], f=features)
    
        glob_mask = np.load(os.path.join(maskpath, "mask_dr.npy"))[:64, :64].astype(bool)
        lat, lon = glob_mask.shape[0], glob_mask.shape[1]

    y = np.load(os.path.join(path, "true_data_dr.npy"))#results_dime["y_15"]
    y_pred = np.load(os.path.join(path, "pred_data_dr.npy"))#[:-17]
    # print(y_pred.shape)
    mask = np.load(os.path.join(path, "mask_dr.npy"))

    y = np.where(y== -1, np.NaN, y)
    y = np.where(np.isnan(y), -1, y)
    y_pred = np.where(y_pred== -1, np.NaN, y_pred)
    y_pred = np.where(np.isnan(y_pred), -1, y_pred)
    # print(y_pred.shape)


    if basemask:
        y = y[:, :lat, :lon]
        y_pred = y_pred[:, :lat, :lon]
        # print(y.shape, y_pred.shape)
        y[:, ~glob_mask] = -1

        y_pred[:, ~glob_mask] = -1
        # print(y.shape, y_pred.shape)

    if "dime" in model:
        y = y[:-11]
        y_pred = y_pred[:-11]
    return y, y_pred, mask, glob_mask


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
    plt.figure(figsize=(6, 4))

    # Loop through the data dictionary to plot each series
    for label, data in data_dict.items():
        if data is None:
            print(data_dict)
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
    plt.title(f"{model_name} RMSE by {aggregate_by.capitalize()}", fontsize=16)
    plt.ylabel('RMSE', fontsize=14)
    plt.legend(fontsize=12, loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
from typing import Dict, List

def plot_error_time_multimodel(model_results: Dict[str, Dict], 
                               days: List[int] = [10, 15, 30],
                               new_start=None, new_end=None):
    """
    Plot yearly mean errors for multiple models in a single figure, considering different start dates.

    Parameters:
        model_results (dict): Dictionary with model names as keys and sub-dictionaries containing:
                              - 'y_<day>': Ground truth [s, h, w] for the given day
                              - 'y_pred_<day>': Predictions [s, h, w] for the given day
                              - 'start_<day>': Start date for that forecast horizon ('YYYY-MM-DD')
        days (list): List of forecast days to include (e.g., [10, 15, 30]).
        new_start (str): Optional. Start of the subset range ('YYYY-MM-DD').
        new_end (str): Optional. End of the subset range ('YYYY-MM-DD').
    """

    fig, axes = plt.subplots(1, len(days), figsize=(15, 5), sharey=True)  # Standardized y-axis
    colors = ['b', 'g', 'r', 'c', 'm', 'y']  # Colors for different models
    min_y, max_y = float('inf'), float('-inf')  # For standardizing y-axis
    yearly_errors = {day: {} for day in days}  # Store errors per forecast day

    # Compute yearly mean errors for each model and forecast step
    for model_name, model_data in model_results.items():
        for day in days:
            y_key, y_pred_key, start_key = f'y_{day}', f'y_pred_{day}', f'start_{day}'
            if y_key not in model_data or y_pred_key not in model_data or start_key not in model_data:
                print(f"Skipping {model_name} at {day} days: Missing data.")
                continue

            y, y_pred = model_data[y_key], model_data[y_pred_key]
            start = model_data[start_key]  # Get specific start date for this forecast horizon
            start_pd = pd.to_datetime(start)
            days_length = y.shape[0]
            end_pd = start_pd + timedelta(days=days_length - 1)
            range_dates = pd.date_range(start_pd, end_pd)

            # Subset if new date range is specified
            if new_start and new_end:
                new_start_dt, new_end_dt = pd.to_datetime(new_start), pd.to_datetime(new_end)
                valid_dates = (range_dates >= new_start_dt) & (range_dates <= new_end_dt)

                if valid_dates.any():
                    y = y[valid_dates]
                    y_pred = y_pred[valid_dates]
                    range_dates = range_dates[valid_dates]
                else:
                    print(f"Skipping {model_name} at {day} days: No data in range {new_start} to {new_end}.")
                    continue

            # Compute daily and yearly errors
            daily_error = np.nanmean(np.abs(y - y_pred), axis=(1, 2))
            df = pd.DataFrame({'date': range_dates, 'error': daily_error})
            df['year'] = df['date'].dt.year
            yearly_mean_error = df.groupby('year')['error'].mean()

            # Store errors
            yearly_errors[day][model_name] = yearly_mean_error
            min_y = min(min_y, yearly_mean_error.min())
            max_y = max(max_y, yearly_mean_error.max())

    model_labels = {"dime": "Diffusion", "gwnet": "WaveNet", "convlstm": "ConvLSTM"}

    # Plot each forecast day in a separate subplot
    for idx, day in enumerate(days):
        ax = axes[idx]

        for model_idx, (model_name, yearly_mean_error) in enumerate(yearly_errors[day].items()):
            label = model_labels.get(model_name, model_name)
            ax.plot(
                yearly_mean_error.index, yearly_mean_error.values,
                marker='o', linestyle='-', color=colors[model_idx % len(colors)], label=label
            )

        ax.set_title(f'Forecast {day} Days', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xticks(yearly_mean_error.index)
        ax.set_xticklabels(yearly_mean_error.index, rotation=45)

    # Standardize y-axis
    for ax in axes:
        ax.set_ylim(min_y, max_y)

    # Set common y-label
    fig.text(0.04, 0.5, 'Mean Absolute Error', va='center', rotation='vertical', fontsize=12)

    # Move legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout
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
    plt.figure(figsize=(6, 4))
    plt.plot(yearly_mean_error.index, yearly_mean_error.values, marker='o', linestyle='-', color='b')
    plt.title('Yearly RMSE', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(yearly_mean_error.index, rotation=45)
    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union

def plot_models_errors_overtime(
    results: dict, model_name: str, days: Union[list, None] = [10, 15, 30]
):
    """
    Plot yearly mean errors for multiple models stored in a dictionary with custom date ranges.

    Parameters:
        results (dict): Dictionary containing ground truth, predictions, masks, and custom start/end dates.
                        Keys for each model day are:
                        - 'y_<day>', 'y_pred_<day>', 'mask_<day>', 'start_<day>', 'end_<day>'.
        model_name (str): The name of the model being plotted, for use in the plot title.
        days (list): List of forecast days to plot.
    """

    plt.figure(figsize=(8, 5))  # Increase figure size for clarity
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Scientific color scheme
    markers = ['o', 's', '^', 'D', 'x']  # Different markers for distinct lines

    for idx, model_day in enumerate(days):
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
            marker=markers[idx % len(markers)],  # Cycle through markers
            linestyle='-',
            color=colors[idx % len(colors)],  # Cycle through colors
            label=f'{model_day}-Day Forecast'
        )

    # Improve plot aesthetics
    plt.xticks(np.arange(yearly_mean_error.index.min(), yearly_mean_error.index.max() + 1, step=1))
    plt.title(f"Yearly Mean Errors for {model_name}", fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)

    # Move legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12, frameon=True)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leaves space for the legend

    # Show the plot
    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from typing import Union

def plot_yearly_error_by_month(
        y: np.ndarray,
        y_pred: np.ndarray,
        start: str,
        aggregate_by: str = "day",
        new_start: Union[None, str] = None,
        new_end: Union[None, str] = None,
        mask: Union[None, np.ndarray] = None,
        plot: bool = False):

    """
    Plot average error aggregated by calendar day or month across multiple years.

    Parameters:
        y (np.ndarray): Ground truth data with shape [s, h, w].
        y_pred (np.ndarray): Predicted data with shape [s, h, w].
        start (str): Starting date in the format 'YYYY-MM-DD'.
        aggregate_by (str): Aggregation level ("day" or "month").
        new_start (str): Optional. Start of the subset range in 'YYYY-MM-DD'.
        new_end (str): Optional. End of the subset range in 'YYYY-MM-DD'.
        mask (np.ndarray): Optional. A 2D mask (h, w) to filter pixels used in loss calculation.
    """
    # Create a date range corresponding to the days in the dataset
    days = y.shape[0]
    start_pd = pd.to_datetime(start)
    end_pd = start_pd + timedelta(days=days - 1)
    range_dates = pd.date_range(start_pd, end_pd)

    # Extract the year from the first date
    year_label = start_pd.year

    # Determine slicing indices based on provided start and end
    new_st_loc = range_dates.get_loc(pd.to_datetime(new_start)) if new_start else 0
    new_end_loc = range_dates.get_loc(pd.to_datetime(new_end)) if new_end else len(range_dates) - 1

    # Subset the data
    y = y[new_st_loc:new_end_loc + 1]
    y_pred = y_pred[new_st_loc:new_end_loc + 1]
    range_dates = range_dates[new_st_loc:new_end_loc + 1]

    # Apply mask if provided
    if mask is not None:
        if mask.shape != y.shape[1:]:  # Ensure mask matches spatial dimensions
            raise ValueError("Mask shape must match the spatial dimensions (h, w) of y and y_pred.")
        y_pred = np.where(mask == 1, y_pred, np.NaN)
        y = np.where(mask == 1, y, np.NaN)

    # Calculate daily error for the selected pixels
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

        # Ensure all months are represented, even if missing in data
        all_months = ["January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]
        average_error_by_time = average_error_by_time.reindex(all_months)

    else:
        raise ValueError("Invalid value for 'aggregate_by'. Choose 'day' or 'month'.")

    # Generate x-axis tick labels based on aggregation
    if aggregate_by == "day":
        tick_indices = np.arange(0, len(average_error_by_time), max(1, len(average_error_by_time) // 12))
        tick_labels = pd.to_datetime(average_error_by_time.index[tick_indices], format='%m-%d').strftime('%d-%b')

    elif aggregate_by == "month":
        tick_indices = np.arange(len(average_error_by_time))  # Months are fewer; show all
        tick_labels = average_error_by_time.index

    if plot:
        # Plot average error for the chosen aggregation
        plt.figure(figsize=(14, 7))
        plt.plot(average_error_by_time.index, average_error_by_time.values, color='b', linestyle='-', marker='.')

        # Update title to include the year when aggregated by month
        if aggregate_by == "month":
            plt.title(f"Average Error Aggregated by Month - Year {year_label}", fontsize=16)
        else:
            plt.title(f"Average Error Aggregated by {aggregate_by.capitalize()}", fontsize=16)

        plt.xlabel(f'{aggregate_by.capitalize()}', fontsize=14)
        plt.ylabel('Average Error', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)
        plt.tight_layout()

    return average_error_by_time


def process_error_by_time_aggregations(results, model_day, aggregate_by="day", new_start=None, new_end=None):
    """
    Process average error aggregated by calendar day or month across multiple years.

    Parameters:
        results (dict): Dictionary containing ground truth, predictions, and start date for a specific model day.
                        Keys:
                        - 'y_<day>', 'y_pred_<day>', 'start_<day>'.
        model_day (int): The model day (e.g., 10, 15, 30).
        aggregate_by (str): Aggregation level ("day" or "month").
        new_start (str): Optional. Start of the subset range in 'YYYY-MM-DD'.
        new_end (str): Optional. End of the subset range in 'YYYY-MM-DD'.

    Returns:
        pd.Series: Aggregated average error indexed by time keys.
    """
    # Extract data and start date from the results dictionary
    y = results.get(f'y_{model_day}')
    y_pred = results.get(f'y_pred_{model_day}')
    start = results.get(f'start_{model_day}')

    if y is None or y_pred is None or start is None:
        raise ValueError(f"Missing data for model day {model_day} in results dictionary.")

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


def plot_multiple_aggregations_per_day(results, model_days, model_name, aggregate_by="day", new_start=None, new_end=None):
    """
    Plot multiple aggregated error series on a single plot.

    Parameters:
        results (dict): Dictionary containing ground truth, predictions, and start dates for all model days.
                        Keys for each day:
                        - 'y_<day>', 'y_pred_<day>', 'start_<day>'.
        model_days (list): List of model days to include (e.g., [10, 15, 30]).
        aggregate_by (str): Aggregation level ("day" or "month").
        new_start (str): Optional. Start of the subset range in 'YYYY-MM-DD'.
        new_end (str): Optional. End of the subset range in 'YYYY-MM-DD'.
    """
    plt.figure(figsize=(14, 7))

    data_dict = {}
    for model_day in model_days:
        try:
            aggregated_data = process_error_by_time_aggregations(
                results, model_day, aggregate_by=aggregate_by, new_start=new_start, new_end=new_end
            )
            data_dict[f'Model Day {model_day}'] = aggregated_data
        except ValueError as e:
            print(e)  # Log the error but continue with other model days

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
    plt.title(f'{model_name} Average Error Aggregation by {aggregate_by.capitalize()}', fontsize=16)
    plt.ylabel('Average Error', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_masked_pixel_subset_prediction_vs_real(y, y_pred, start, model_name, mask, n=5, new_start=None, new_end=None):
    """
    Plot the prediction vs. real value for a subset of masked pixels over time in individual subplots.

    Parameters:
        y (numpy.ndarray): Ground truth values of shape [s, h, w].
        y_pred (numpy.ndarray): Predicted values of shape [s, h, w].
        start (str): Start date for the dataset (e.g., '2000-01-01').
        mask (numpy.ndarray): Boolean mask of shape [h, w], where True indicates valid pixels for selection.
        n (int): Number of random pixels to plot.
        new_start (str, optional): Start date for the subset (e.g., '2021-01-01').
        new_end (str, optional): End date for the subset (e.g., '2021-12-31').
    """
    # Use Seaborn for styling
    sns.set_theme(style="whitegrid")

    # Time setup
    days = y.shape[0]
    start_pd = pd.to_datetime(start)
    end_pd = start_pd + timedelta(days=days - 1)
    range_dates = pd.date_range(start_pd, end_pd)

    # Subset the data if new_start and new_end are provided
    if new_start is not None and new_end is not None:
        new_start_pd = pd.to_datetime(new_start)
        new_end_pd = pd.to_datetime(new_end)
        mask_dates = (range_dates >= new_start_pd) & (range_dates <= new_end_pd)
        y = y[mask_dates]
        y_pred = y_pred[mask_dates]
        range_dates = range_dates[mask_dates]

    # Get valid pixel indices based on the mask
    h, w = y.shape[1:]
    valid_pixels = [(i, j) for i in range(h) for j in range(w) if mask[i, j]]

    # Check if there are enough valid pixels
    if len(valid_pixels) < n:
        raise ValueError(f"Mask contains fewer valid pixels ({len(valid_pixels)}) than the requested number ({n}).")

    # Select n random pixels from the valid ones
    selected_pixels = np.random.choice(len(valid_pixels), size=n, replace=False)
    pixel_indices = [valid_pixels[idx] for idx in selected_pixels]

    # Determine subplot layout
    columns = 3
    rows = (n + columns - 1) // columns  # Ensure all pixels are plotted
    fig, axes = plt.subplots(rows, columns, figsize=(8, 2 * rows), sharex=True, squeeze=False)

    # Flatten axes for easier iteration
    axes = axes.flatten()

    for idx, (ax, (row, col)) in enumerate(zip(axes, pixel_indices)):
        # Plot for each selected pixel
        ax.plot(range_dates, y[:, row, col], label='Real', linestyle='-', linewidth=1, color="green")
        ax.plot(range_dates, y_pred[:, row, col], label='Prediction', linestyle='--', linewidth=1, color="grey")

        # Customize y-axis and add legend
        ax.set_ylabel('Value', fontsize=10)
        ax.legend(fontsize=8, loc='upper left', frameon=True)
        ax.set_title(f'Pixel ({row}, {col})', fontsize=10)

    # Remove unused subplots
    for ax in axes[len(pixel_indices):]:
        ax.set_visible(False)

    # Customize x-axis for all subplots
    tick_indices = pd.date_range(range_dates[0], range_dates[-1], freq='180D')
    for ax in axes:
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_indices.strftime('%d-%b'), rotation=45, fontsize=8)

    # Add a common x-axis label
    fig.text(0.5, 0.04, 'Date', ha='center', fontsize=12)

    # Add title
    fig.suptitle(f"{model_name} prediction vs Real Value for {n} Masked Pixels Over Time", fontsize=14, weight='bold')

    # Adjust layout to avoid overlap
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Show the plot
    plt.show()


def plot_predicted_vs_real_maps(
    y, y_pred, start, new_start=None, new_end=None, cmap="viridis", step=1
):
    """
    Plot predicted maps vs real maps over time.

    Parameters:
        y (numpy.ndarray): Ground truth values of shape [s, h, w].
        y_pred (numpy.ndarray): Predicted values of shape [s, h, w].
        start (str): Start date for the dataset (e.g., '2000-01-01').
        new_start (str, optional): Start date for the subset (e.g., '2021-01-01').
        new_end (str, optional): End date for the subset (e.g., '2021-12-31').
        cmap (str): Colormap for the images (default: 'viridis').
        step (int): Step interval for selecting images to plot (default: 10).
    """
    # Time setup
    days = y.shape[0]
    start_pd = pd.to_datetime(start)
    end_pd = start_pd + timedelta(days=days - 1)
    range_dates = pd.date_range(start_pd, end_pd)

    # Subset the data if new_start and new_end are provided
    if new_start is not None:
        new_start_pd = pd.to_datetime(new_start)
        new_end_pd = pd.to_datetime(new_end)
        mask = (range_dates >= new_start_pd) & (range_dates <= new_end_pd)
        y = y[mask]
        y_pred = y_pred[mask]
        range_dates = range_dates[mask]

    # Select frames to plot
    indices = np.arange(0, y.shape[0], step)
    selected_dates = range_dates[indices]
    y_selected = y[indices]
    y_pred_selected = y_pred[indices]

    # Number of images to plot
    n_images = len(indices)

    # Plot configuration
    fig, axes = plt.subplots(
        2, n_images, figsize=(5 * n_images, 10), constrained_layout=True
    )

    for i, idx in enumerate(indices):
        # Real map
        ax_real = axes[0, i]
        im_real = ax_real.imshow(y_selected[i], cmap=cmap)
        ax_real.set_title(f"Real - {selected_dates[i].strftime('%d-%b-%Y')}", fontsize=10)
        ax_real.axis("off")
        fig.colorbar(im_real, ax=ax_real, orientation="vertical", fraction=0.05)

        # Predicted map
        ax_pred = axes[1, i]
        im_pred = ax_pred.imshow(y_pred_selected[i], cmap=cmap)
        ax_pred.set_title(f"Predicted - {selected_dates[i].strftime('%d-%b-%Y')}", fontsize=10)
        ax_pred.axis("off")
        fig.colorbar(im_pred, ax=ax_pred, orientation="vertical", fraction=0.05)

    # Set row titles
    axes[0, 0].set_ylabel("Real", fontsize=12, weight="bold")
    axes[1, 0].set_ylabel("Predicted", fontsize=12, weight="bold")

    # Show the plot
    plt.suptitle("Real vs Predicted Maps", fontsize=16, weight="bold")
    plt.show()


def plot_model_maps(date, results, target_model, models, days, cmap="viridis"):
    """
    Plot maps for real values and predictions for all models on a specific date for each forecast day.

    Parameters:
        date (str): Reference date for plotting (e.g., '2017-01-01').
        results (dict): Dictionary containing prediction data for all models.
        target_model (str): The model whose real values will be plotted (e.g., 'convlstm').
        models (list): List of models to include in the plot for predictions.
        days (list): List of forecast days to plot (e.g., [10, 15, 30]).
        cmap (str): Colormap to use for plots (e.g., 'viridis').
    """
    sns.set_theme(style="whitegrid")

    # Define number of rows (models) and columns (forecast days) + 1 for the first column
    n_models = len(models) + 1  # First row is for real values
    n_days = len(days)  # Columns correspond to different forecast days

    # Add 1 column for model labels
    fig, axes = plt.subplots(n_models, n_days + 1,
                             figsize=(2 * (n_days + 1), 2 * n_models),
                             squeeze=False,
                             gridspec_kw={'width_ratios': [1] + [1] * n_days})

    # Model names for labeling
    model_labels = {
        "dime": "Diffusion",
        "gwnet": "WaveNet",
        "convlstm": "ConvLSTM"
    }

    # Iterate over models (rows)
    for model_idx, model in enumerate([target_model] + models):
        for day_idx, day in enumerate(days):
            col_idx = day_idx + 1  # Offset by 1 due to model label column

            # Retrieve the correct keys
            key_y = f"y_{day}" if model_idx == 0 else f"y_pred_{day}"
            key_start = f"start_{day}"
            key_end = f"end_{day}"

            # Extract the real or predicted values
            y_data = results[model][key_y]
            start = pd.to_datetime(results[model][key_start])
            end = pd.to_datetime(results[model][key_end])

            # Convert input date to index
            ref_date = pd.to_datetime(date)
            if not (start <= ref_date <= end):
                raise ValueError(f"The date {date} is out of range for {model} at {day}-day prediction.")

            date_idx = (ref_date - start).days
            map_data = y_data[date_idx]  # Extract the specific day’s map

            # Plot
            ax = axes[model_idx, col_idx]
            im = ax.imshow(map_data, cmap=cmap, aspect="auto", vmin=-0.2, vmax=1)

            # Set the first row's column titles to "Real - n Days"
            if model_idx == 0:
                ax.set_title(f"{day} Days", fontsize=10, pad=10)  # Increased spacing with `pad=10`

            # Remove axes ticks for cleaner visualization
            ax.set_xticks([])
            ax.set_yticks([])

            # Add colorbar only for last column
            if day_idx == len(days) - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        row_label = "Real values" if model_idx == 0 else model_labels.get(model, model)
        # Add model label in the first column (horizontal text)
        axes[model_idx, 0].axis("off")  # Hide axes for label column
        axes[model_idx, 0].text(0.5, 0.5, row_label, fontsize=12, fontweight="bold",
                                ha="center", va="center")

    # Adjust layout and add external labels
    fig.supxlabel('Days', fontsize=14, fontweight='bold', y=0.02)  # External x-axis label
    fig.supylabel('Models', fontsize=14, fontweight='bold', x=0.02)  # External y-axis label

    plt.tight_layout(rect=[0.05, 0.08, 1, 0.95])  # Adjust layout so global labels & subtitle have space
    plt.suptitle(f"Maps for {date}", fontsize=18, weight="bold", y=1.05)  # Increased spacing above

    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_predictions_all_dates_hexbin(results, models, days, cmap="viridis", min_freq=10, gridsize=50):
    """
    Plot hexbin plots for real vs. predicted values for all models and forecast days over all available dates.

    Parameters:
        results (dict): Dictionary containing prediction data for all models.
        models (list): List of models to include in the plot (e.g., ['gwnet', 'convlstm']).
        days (list): List of forecast days to plot (e.g., [10, 15, 30]).
        cmap (str): Colormap to use for plots (e.g., 'viridis').
        min_freq (int): Minimum frequency to display in the hexbin plot.
        gridsize (int): The size of the hexagons in the hexbin plot.
    """
    sns.set_theme(style="whitegrid")

    # Initialize subplots
    n_models = len(models)
    n_days = len(days)
    fig, axes = plt.subplots(n_days, n_models, figsize=(3 * n_models, 2.5 * n_days), squeeze=False)

    # Create a placeholder for the hexbin plot to use for the colorbar
    hb = None

    # Iterate through models and days
    for model_idx, model in enumerate(models):
        for day_idx, day in enumerate(days):
            # Get relevant keys
            key_y = f"y_{day}"
            key_y_pred = f"y_pred_{day}"
            key_start = f"start_{day}"
            key_end = f"end_{day}"

            # Retrieve data
            y = results[model][key_y]
            y_pred = results[model][key_y_pred]

            # Flatten arrays
            y_flat = y.reshape(-1)
            y_pred_flat = y_pred.reshape(-1)

            # Keep only values in range [0,1]
            mask = (y_pred_flat >= 0) & (y_pred_flat <= 1) & (y_flat >= 0) & (y_flat <= 1)

            # Apply mask to all arrays
            y_flat_filtered = y_flat[mask]
            y_pred_flat_filtered = y_pred_flat[mask]

            # Plot hexbin
            ax = axes[day_idx, model_idx]
            hb = ax.hexbin(
                y_flat_filtered, y_pred_flat_filtered,
                gridsize=gridsize, cmap=cmap, mincnt=min_freq, bins="log"
            )

            # Set the plot title and labels
            ax.set_title(f"{model.upper()} - {day} Days", fontsize=10)
            ax.set_xlabel("Real Values", fontsize=8)
            ax.set_ylabel("Predicted Values", fontsize=8)

            # Set limits from 0 to 1
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            # Add tick marks every 0.2
            ax.set_xticks(np.arange(0, 1.1, 0.2))
            ax.set_yticks(np.arange(0, 1.1, 0.2))

            # Add quadrant lines
            # ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
            # ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8)

            # Add red diagonal reference line
            ax.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1)

    # Add a single colorbar for all hexbin plots
    cbar = fig.colorbar(hb, ax=axes, orientation="vertical", fraction=0.02, pad=0.04, anchor=(0.0, 0.5))
    cbar.set_label("Hexbin Frequency", fontsize=10)
    # cbar.set_ticks([])  # Remove colorbar ticks

    # Adjust layout to make room for the colorbar
    plt.tight_layout()
    plt.subplots_adjust(right=0.85, top=0.90)  # Adjust the right margin for the colorbar
    plt.suptitle("Model Predictions Over All Dates - Hexbin Plots", fontsize=14, weight="bold")
    plt.show()


def plot_drought_aggregations(results:dict,
    model:str,
    step:int,
    mask:np.ndarray,
    start_date:str,
    n:int,
    include_percentiles=True):

    """
    Plots median drought severity over n future days, with optional 25th and 75th percentiles.

    Parameters:
    start_date (str): The starting date in 'YYYY-MM-DD' format.
    n (int): Number of future days to aggregate.
    include_percentiles (bool): Whether to include 25th and 75th percentiles in the plot.
    """

    real = results[model][f"y_{step}"]
    pred = results[model][f"y_pred_{step}"]

    real_start_date = results[model][f"start_{step}"]
    end_date =  results[model][f"end_{step}"]

    # Convert string dates to datetime
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Compute index range based on start_date and n
    date_range = np.array([start_dt + timedelta(days=i) for i in range(n)])
    start_idx = (start_dt - datetime.strptime(real_start_date, "%Y-%m-%d")).days
    end_idx = min(start_idx + n, real.shape[0])

    # Masking invalid values
    real_masked = np.where(mask[None, :, :], real[start_idx:end_idx], np.nan)
    pred_masked = np.where(mask[None, :, :], pred[start_idx:end_idx], np.nan)

    # Compute statistics
    real_median = np.nanmedian(real_masked, axis=(1, 2))
    pred_median = np.nanmedian(pred_masked, axis=(1, 2))

    if include_percentiles:
        real_p25 = np.nanpercentile(real_masked, 25, axis=(1, 2))
        real_p75 = np.nanpercentile(real_masked, 75, axis=(1, 2))
        pred_p25 = np.nanpercentile(pred_masked, 25, axis=(1, 2))
        pred_p75 = np.nanpercentile(pred_masked, 75, axis=(1, 2))

    # Plot results
    plt.figure(figsize=(7, 3))
    plt.plot(date_range[:end_idx - start_idx], real_median, label='Real Median', color='blue')
    plt.plot(date_range[:end_idx - start_idx], pred_median, label='Predicted Median', color='red', linestyle='dashed')

    if include_percentiles:
        plt.fill_between(date_range[:end_idx - start_idx], real_p25, real_p75, color='blue', alpha=0.2, label='Real IQR')
        plt.fill_between(date_range[:end_idx - start_idx], pred_p25, pred_p75, color='red', alpha=0.2, label='Predicted IQR')

    plt.xlabel("Date")
    plt.ylabel("NDVI")
    plt.title(f"NDVI Predictions {step} days in advance vs Real")
    plt.legend(loc='upper left')
    plt.grid()
    plt.xticks(rotation=45)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def plot_drought_at_steps(results: dict, model: str, mask: np.ndarray, start_date: str):
    """
    Plots median drought severity with real values as a green line and future predictions as dots,
    including interquartile range (IQR) shading.

    Parameters:
    results (dict): Dictionary containing real and predicted drought severity data.
    model (str): Model name to extract data from results.
    mask (np.ndarray): 2D mask array indicating valid regions.
    start_date (str): The starting date in 'YYYY-MM-DD' format.
    """
    forecast_steps = [10, 15, 30, 45, 60]
    plt.figure(figsize=(6, 3))

    real_median = None  # Store real median to plot only once
    real_p25 = None
    real_p75 = None
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=forecast_steps[0])

    for step in forecast_steps:
        real = results[model][f"y_{step}"]
        pred = results[model][f"y_pred_{step}"]
        real_start_date = results[model][f"start_{step}"]

        # Compute index range based on start_date
        start_idx = (start_dt - datetime.strptime(real_start_date, "%Y-%m-%d")).days
        end_idx = min(start_idx + step, real.shape[0])
        date_range = np.array([start_dt + timedelta(days=i) for i in range(step)])

        # Masking invalid values
        real_masked = np.where(mask[None, :, :], real[start_idx:end_idx], np.nan)
        pred_masked = np.where(mask[None, :, :], pred[start_idx:end_idx], np.nan)

        # Compute statistics
        if step == forecast_steps[-1]:
            real_median = np.nanmedian(real_masked, axis=(1, 2))
            real_p25 = np.nanpercentile(real_masked, 25, axis=(1, 2))
            real_p75 = np.nanpercentile(real_masked, 75, axis=(1, 2))
        pred_median = np.nanmedian(pred_masked[-1], axis=(0, 1))  # Last prediction point

        # Plot prediction as a dot
        plt.scatter(date_range[-1], pred_median, label=f'Predicted {step} days', marker='o')

    # Plot real median as a green line
    plt.plot(date_range[:len(real_median)], real_median, label='Real Median', color='green')

    # Plot IQR as shaded area
    plt.fill_between(date_range[:len(real_median)], real_p25, real_p75, color='green', alpha=0.2, label='Real IQR')

    plt.xlabel("Date")
    plt.ylabel("NDVI")
    plt.title(f"Median NDVI Predictions at Multiple Forecast Steps on the {start_date} ")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.grid()
    plt.xticks(rotation=45)
    plt.show()


def plot_model_predictions_all_dates(results, models, days, cmap="viridis"):
    """
    Plot scatterplots for real vs. predicted values for all models and forecast days over all available dates.

    Parameters:
        results (dict): Dictionary containing prediction data for all models.
        models (list): List of models to include in the plot (e.g., ['gwnet', 'convlstm']).
        days (list): List of forecast days to plot (e.g., [10, 15, 30]).
        colormap (str): Colormap to use for plots (e.g., 'viridis').
    """
    sns.set_theme(style="whitegrid")

    # Initialize subplots
    n_models = len(models)
    n_days = len(days)
    fig, axes = plt.subplots(n_models, n_days, figsize=(4 * n_days, 3 * n_models), squeeze=False)

    # Iterate through models and days
    for model_idx, model in enumerate(models):
        for day_idx, day in enumerate(days):
            # Get relevant keys
            key_y = f"y_{day}"
            key_y_pred = f"y_pred_{day}"
            key_start = f"start_{day}"
            key_end = f"end_{day}"

            # Retrieve data
            y = results[model][key_y]
            y_pred = results[model][key_y_pred]
            start = pd.to_datetime(results[model][key_start])
            end = pd.to_datetime(results[model][key_end])
            date_range = pd.date_range(start, end)

            # Flatten arrays for all dates
            y_flat = y.reshape(-1)
            y_pred_flat = y_pred.reshape(-1)

            # Generate color array
            color_array = np.linspace(0, 1, len(y_flat))

            # Plot scatter
            ax = axes[model_idx, day_idx]
            scatter = ax.scatter(
                y_flat,
                y_pred_flat,
                c=color_array,
                cmap=cmap,
                s=10,
                alpha=0.7
            )
            ax.plot([y_flat.min(), y_flat.max()], [y_flat.min(), y_flat.max()], color="red", linestyle="--", linewidth=1)
            ax.set_title(f"{model.upper()} - {day} Days", fontsize=10)
            ax.set_xlabel("Real Values", fontsize=8)
            ax.set_ylabel("Predicted Values", fontsize=8)

    # Add a colorbar
    fig.colorbar(scatter, ax=axes, orientation="horizontal", fraction=0.02, pad=0.1)
    plt.tight_layout()
    plt.suptitle("Model Predictions Over All Dates", fontsize=14, weight="bold")
    plt.subplots_adjust(top=0.9)
    plt.show()

def plot_model_predictions(date, results, models, days, cmap="viridis"):
    """
    Plot predictions for specified models and forecast days on a given date.

    Parameters:
        date (str): The reference date for plotting (e.g., '2017-01-01').
        results (dict): Dictionary containing prediction data for all models.
                        Each model key contains keys like 'y_10', 'y_pred_10', etc.
        models (list): List of models to include in the plot (e.g., ['gwnet', 'other_model']).
        days (list): List of forecast days to plot (e.g., [10, 15, 30]).
        colormap (str): Colormap to use for plots (e.g., 'viridis').
    """
    sns.set_theme(style="whitegrid")

    # Convert the input date to a pandas datetime object
    ref_date = pd.to_datetime(date)

    # Initialize subplots
    n_models = len(models)
    n_days = len(days)
    fig, axes = plt.subplots(n_models, n_days, figsize=(4 * n_days, 3 * n_models), squeeze=False)

    # Iterate through models and days to create subplots
    for model_idx, model in enumerate(models):
        for day_idx, day in enumerate(days):
            # Construct the relevant keys for the model and forecast day
            key_y = f"y_{day}"
            key_y_pred = f"y_pred_{day}"
            key_start = f"start_{day}"
            key_end = f"end_{day}"

            # Retrieve data from the results dictionary
            y = results[model][key_y]
            y_pred = results[model][key_y_pred]
            start = pd.to_datetime(results[model][key_start])
            end = pd.to_datetime(results[model][key_end])

            # Ensure the reference date is within the valid range
            if not (start <= ref_date <= end):
                axes[model_idx, day_idx].text(0.5, 0.5, "Date out of range", fontsize=12, ha='center')
                axes[model_idx, day_idx].set_axis_off()
                continue

            # Compute the index corresponding to the reference date
            date_idx = (ref_date - start).days
            if date_idx < 0 or date_idx >= y.shape[0]:
                axes[model_idx, day_idx].text(0.5, 0.5, "Invalid date index", fontsize=12, ha='center')
                axes[model_idx, day_idx].set_axis_off()
                continue


            # Flatten y and y_pred for plotting
            y_flat = y[date_idx].flatten()
            y_pred_flat = y_pred[date_idx].flatten()

            mask = y_flat >= 0  # Keep only values >= 0

            # Apply mask to all arrays
            y_flat_filtered = y_flat[mask]
            y_pred_flat_filtered = y_pred_flat[mask]

            # Use filtered arrays in scatter plot
            ax = axes[model_idx, day_idx]
            scatter = ax.scatter(
                y_flat_filtered,
                y_pred_flat_filtered,
                c=y_flat_filtered,
                cmap=cmap,
                s=10,
                alpha=0.7
            )

            ax.plot([y_flat_filtered.min(), y_flat_filtered.max()],
                    [y_flat_filtered.min(), y_flat_filtered.max()],
                    color="red", linestyle="--", linewidth=1)
            ax.set_title(f"{model.upper()} - {day} Days", fontsize=10)
            ax.set_xlabel("Real Values", fontsize=8)
            ax.set_ylabel("Predicted Values", fontsize=8)

    # Adjust layout and add a colorbar
    # fig.colorbar(scatter, ax=axes, orientation="horizontal", fraction=0.02, pad=0.1)
    # Create a separate axis for the colorbar
    cbar_ax = fig.add_axes([1.05, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Color Scale", fontsize=10)

    plt.tight_layout()
    plt.suptitle(f"Model Predictions on {date}", fontsize=14, weight="bold")
    plt.subplots_adjust(top=0.9)
    plt.show()

def plot_models_metrics_over_steps(metrics_results, days):

    # Initialize target dictionaries
    deterministic_models = {}
    probabilistic_models = {}

    # Define mappings for model names and types
    model_mapping = {
        'convlstm': ('ConvLSTM', deterministic_models),
        'gwnet': ('WaveNet', deterministic_models),
        'dime': ('Diffusion', probabilistic_models)
    }

    # Populate the new structure
    for model, horizons in metrics_results.items():
        new_name, target_dict = model_mapping[model]
        rmse_values = [round(horizons[horizon]['rmse'], 3) for horizon in days]
        ssim_values = [round(horizons[horizon]['ssim'], 3) for horizon in days]
        target_dict[new_name] = {"RMSE": rmse_values, "SSIM": ssim_values}


    # Data extracted from the tables
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # RMSE Plot
    for model, metrics in deterministic_models.items():
        axes[0].plot(days, metrics["RMSE"], marker="o", label=model)
    for model, metrics in probabilistic_models.items():
        axes[0].plot(days, metrics["RMSE"], marker="o", linestyle="--", label=model)

    axes[0].set_title("RMSE over Time")
    axes[0].set_xlabel("Days")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()
    axes[0].grid()

    # SSIM Plot
    for model, metrics in deterministic_models.items():
        axes[1].plot(days, metrics["SSIM"], marker="o", label=model)
    for model, metrics in probabilistic_models.items():
        axes[1].plot(days, metrics["SSIM"], marker="o", linestyle="--", label=model)

    axes[1].set_title("SSIM over Time")
    axes[1].set_xlabel("Days")
    axes[1].set_ylabel("SSIM")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.show()


if __name__== "__main__":
    from analysis import custom_subset_data, diffusion_arguments
    from analysis.visualizations.viz_series import load_ndvi_output
    from analysis.configs.config_models import config_gwnet as model_config
    import numpy as np
    from datetime import datetime, timedelta
    from typing import Union
    import pandas as pd
    import os
    from utils.xarray_functions import ndvi_colormap
    import seaborn as sns
    import matplotlib.pyplot as plt
    cmap = ndvi_colormap("diverging")


    start, end = "2019-01-01", "2022-12-31"
    # data_name = "data_gnn_drought"

    results = {}
    basemask = {"days": 10, "model": "dime"}


    for model in ["dime", "convlstm",  "gwnet"]:
        # Initialize the model key in the results dictionary
        results[model] = {}

        for day in [10]:
            # Load the data for the current model and day
            y, y_pred, mask = load_ndvi_output(model_config, model, day, basemask=basemask)

            # Store the results in the nested dictionary
            results[model][f"y_{day}"] = y
            results[model][f"y_pred_{day}"] = y_pred
            results[model][f"mask_{day}"] = mask

            # Calculate the date range based on the model and day
            if model == "dime":
                range_dates = pd.date_range(
                    pd.to_datetime(start),
                    pd.to_datetime(end)
                )
            else:
                range_dates = pd.date_range(
                    pd.to_datetime(start) - timedelta(days=1) + timedelta(days=day + 90),
                    pd.to_datetime(end)
                )

            # Store the start and end dates in the nested dictionary
            results[model][f"start_{day}"] = range_dates[0].strftime("%Y-%m-%d")
            results[model][f"end_{day}"] = range_dates[-1].strftime("%Y-%m-%d")


    range_dates = pd.date_range(pd.to_datetime(start) - timedelta(days=1) + timedelta(days=15+90), pd.to_datetime(end))
    print("For the prediction over 15 days with 90 days of features there are {} samples".format(len(range_dates)))

    print(results["convlstm"]["y_pred_10"].shape)
    print(results["dime"]["y_pred_10"].shape)
    print(results["gwnet"]["y_pred_10"].shape)

    print(results["dime"]["start_10"])
    print(results["dime"]["end_10"])


    res = results["convlstm"]
    model_name = "convlstm"

    mask = results["dime"]["mask_10"]
    y = res["y_10"]
    y_pred = res["y_pred_10"]
    start = res["start_10"]

    # Process data for different time ranges
    error_2019 = plot_yearly_error_by_month(y, y_pred, start, aggregate_by="month", new_start=start, new_end="2019-12-31", mask=mask)
    error_2020 = plot_yearly_error_by_month(y, y_pred, start, aggregate_by="month", new_start="2020-01-01", new_end="2020-12-31")
    error_2021 = plot_yearly_error_by_month(y, y_pred, start, aggregate_by="month", new_start="2021-01-01", new_end="2021-12-31")
    error_2022 = plot_yearly_error_by_month(y, y_pred, start, aggregate_by="month", new_start="2022-01-01", new_end="2022-12-30")
    error_tot = plot_yearly_error_by_month(y, y_pred, start, aggregate_by="month")

    plot_multiple_aggregations(data_dict = { "2019":error_2019, '2020': error_2020, '2021': error_2021, "2022":error_2022, "total": error_tot}, model_name=model_name, aggregate_by="month")