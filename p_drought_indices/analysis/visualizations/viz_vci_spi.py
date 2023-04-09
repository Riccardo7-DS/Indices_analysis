from p_drought_indices.functions.function_clns import load_config, cut_file, subsetting_pipeline
from p_drought_indices.functions.ndvi_functions import downsample, clean_ndvi, compute_ndvi, clean_outliers
from p_drought_indices.vegetation.cloudmask_cleaning import extract_apply_cloudmask, plot_cloud_correction, compute_difference, compute_correlation
import xarray as xr 
import pandas as pd
import yaml
from datetime import datetime, timedelta
import shutil
#from shapely.geometry import Polygon, mapping
#import geopandas as gpd
import matplotlib.pyplot as plt
from glob import glob
import os
#import datetime as datetime
import time
import numpy as np
import re
from p_drought_indices.vegetation.NDVI_indices import compute_svi, compute_vci
from p_drought_indices.analysis.metrics_table import MetricTable
from p_drought_indices.functions.function_clns import open_xarray_dataset, crop_get_spi, crop_get_thresh

CONFIG_PATH = r"../../../config.yaml"


import xarray as xr
import os

from p_drought_indices.functions.function_clns import load_config
import pandas as pd



def get_dates(gap_year=False):
    if gap_year==False:
        return pd.date_range("01-Jan-2021", "31-Dec-2021", freq="D").to_series().dt.strftime('%d-%b').values
    else:
        return pd.date_range("01-Jan-2020", "31-Dec-2020", freq="D").to_series().dt.strftime('%d-%b').values
    

import calendar
import matplotlib.pyplot as plt
import numpy as np
from typing import Union

def box_plot_year(ds, var:str="ndvi", year:Union[None, int,list]=None, title:str=None, figsize =(15, 7), show_means:bool=False):
    if year==None:
        days = 366
    elif type(year)==list:
        if True in [True for y in year if calendar.isleap(y)]:
            days=366
        else: days = 365
    else:
        days=366 if calendar.isleap(year) else 365

    day_obj = dfDay(ds = ds, var=var)
    df_list = []
    for day in range(1,days+1):
        day_obj.get_day(day)
        locals()[day_obj.df_name] = day_obj.df
        df_list.append(locals()[day_obj.df_name][var])

    bool_days = False if days==365 else True
    list_dates = get_dates(gap_year=bool_days)
 
    fig, ax = plt.subplots(figsize =figsize)
    
    # Creating plot
    if show_means==True:
        boxplot = ax.boxplot(df_list, showfliers=False, patch_artist=True,labels=list_dates, showmeans=True,medianprops=dict(color="green",ls="--",lw=1), meanline=True, meanprops=dict(color="red", ls="-", lw=2))
    else:
        boxplot = ax.boxplot(df_list, showfliers=False, patch_artist=True, labels=list_dates)
    n=7
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='grey', linestyle='dashed')
    if title !=None:
        ax.set_title(title)

    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
    n= 10
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines()) if i % n != 0]

    ax.tick_params(labelrotation=45)
    for median in boxplot['medians']:
        median.set_color('red')

    for box in boxplot["boxes"]:
        box.set_color("lightblue")

    for whisk in boxplot["whiskers"]:
        whisk.set_color("lightgrey")
    
    # show plot
    plt.show()


class dfDay:
    def __init__(self, ds:xr.Dataset, var:str) -> None:
        ds = ds.assign(dayofyear = ds["time"].dt.dayofyear)
        self.dataset = ds
        self.var = var

  
    def get_day(self, day:int):
        df = self.dataset.where(self.dataset["dayofyear"]==day, drop=True)[self.var].to_dataframe()
        self.df = self._clen_df(df)
        self.df_name = f"data_{day}"
    
    def _clen_df(self, df):
        df.reset_index(drop=False, inplace=True)
        df = df.dropna(subset=[self.var])
        if "spatial_ref" in df.columns:
            df = df.drop(columns={"spatial_ref"})
        return df.reset_index(drop=True)
    
def get_subplot_year(ds, var:str="ndvi", year:Union[None, int,list]=None):
    if year==None:
        days = 366
    elif type(year)==list:
        if True in [True for y in year if calendar.isleap(y)]:
            days=366
        else: days = 365
    else:
        days=366 if calendar.isleap(year) else 365

    day_obj = dfDay(ds = ds, var=var)
    df_list = []
    for day in range(1,days+1):
        day_obj.get_day(day)
        locals()[day_obj.df_name] = day_obj.df
        df_list.append(locals()[day_obj.df_name][var])
    
    bool_days = False if days==365 else True
    list_dates = get_dates(gap_year=bool_days)

    return df_list, list_dates

def get_year_compare(ds:xr.DataArray, var:str, year:list):

    days=366 if calendar.isleap(year) else 365
    ds_subset = ds.sel(time=ds.time.dt.year.isin([year]))

    #final_df = pd.DataFrame()

    day_obj_full = dfDay(ds = ds, var=var)
    day_obj = dfDay(ds = ds_subset, var=var)
    df_list = []
    for day in range(1,days+1):
        day_obj_full.get_day(day)

        if (days==365) & (day==60):
            series = [np.NaN for _ in range(len(day_obj_full.df[var]))]
            new_df = pd.DataFrame([series,  day_obj_full.df[var]]).T
        else:
            day_obj.get_day(day)
            new_df = pd.DataFrame([day_obj.df[var],  day_obj_full.df[var]]).T

        new_df.columns = ["year", "whole"]
        df_list.append(new_df)

    if days==365:
        del df_list[59]
    bool_days = False if days==365 else True
    list_dates = get_dates(gap_year=bool_days)
    return df_list, list_dates

def multiple_spi_boxplots(list_late, list_data, list_dates, title, figsize=None, show_means=False):
    fig, axes = plt.subplots(nrows=2,ncols=2, figsize =(15, 7))
    for i, (spi, ax) in enumerate(zip(list_late, axes.ravel())):
        plt.suptitle(title, fontsize=18, y=1)
    
        if show_means==True:
            boxplot = ax.boxplot(list_data[i], showfliers=False, patch_artist=True,labels=list_dates, showmeans=True,medianprops=dict(color="green",ls="--",lw=1), meanline=True, meanprops=dict(color="red", ls="-", lw=2))
        else:
            boxplot = ax.boxplot(list_data[i], showfliers=False, patch_artist=True, labels=list_dates)
        # Creating plot
        #boxplot = ax.boxplot(list_data[i], showfliers=False, patch_artist=True, labels=list_dates)
        n=15
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='grey', linestyle='dashed')
        ax.set_title(f"SPI_GAMMA_{spi}")

        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        n= 10
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines()) if i % n != 0]
        ax.tick_params(labelrotation=45)
        for median in boxplot['medians']:
            median.set_color('red')
        for box in boxplot["boxes"]:
            box.set_color("lightblue")
        for whisk in boxplot["whiskers"]:
            whisk.set_color("lightgrey")

        # show plot
    fig.subplots_adjust(hspace=0.5)
    plt.show()

def get_year(year):
    if type(year) == int:
        return year
    else:
        return str(year[0]) + "-" + str(year[-1])
    


def plot_products_drought_pixels(year:Union[list, int], prod_directories:list, list_late:list = [30, 60, 90, 180]):
    for product_dir in prod_directories:
        print("Plotting new product...")
        list_data = []
        for late in list_late:
            var_target = f"spi_gamma_{late}"
            files = [f for f in os.listdir(product_dir) if var_target in f ]
            spi_temp = xr.open_dataset(os.path.join(product_dir, files[0]))
            subset_ds=spi_temp.sel(time=spi_temp.time.dt.year.isin(year))
            locals()[f"spi_{late}"] = crop_get_spi(subset_ds)
            locals()[f"df_list_{late}"],list_dates = get_subplot_year(locals()[f"spi_{late}"] , var=var_target, year=year)
            list_data.append(locals()[f"df_list_{late}"],)
            spi_temp.close()

        product = files[0].split("_")[0]
        title = f"Percentage of drought pixels for product {product} in year {get_year(year)}"
        multiple_spi_boxplots(list_late, list_data, title=title, list_dates = list_dates, show_means=True)