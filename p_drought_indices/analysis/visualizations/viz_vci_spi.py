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
from matplotlib import gridspec
import matplotlib.patches as mpatches
from matplotlib import gridspec
import re
from p_drought_indices.vegetation.NDVI_indices import compute_svi, compute_vci
from p_drought_indices.analysis.metrics_table import MetricTable
from p_drought_indices.functions.function_clns import open_xarray_dataset, crop_get_spi,load_config
import xarray as xr
import os
import pandas as pd
import calendar
import matplotlib.pyplot as plt
import numpy as np
from typing import Union

CONFIG_PATH = r"../../../config.yaml"


def get_dates(gap_year=False):
    if gap_year==False:
        return pd.date_range("01-Jan-2021", "31-Dec-2021", freq="D").to_series().dt.strftime('%d-%b').values
    else:
        return pd.date_range("01-Jan-2020", "31-Dec-2020", freq="D").to_series().dt.strftime('%d-%b').values


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
    
def str_month(month):
    if month>9:
        return str(month)
    else:
        return "0" + str(month)
    
def subsetting_whole(df_list_all, months, year=2020):
    last_day = calendar.monthrange(year, months[-1])[1]
    date_start = f"01-{str_month(months[0])}-{year}"
    date_end = f"{last_day}-{str_month(months[-1])}-{year}"
    int_st = time.strptime(date_start, "%d-%m-%Y").tm_yday
    int_end = time.strptime(date_end, "%d-%m-%Y").tm_yday
    return  df_list_all[int_st-1:int_end]

def get_xarray_time_subset(ds:xr.DataArray,  year:Union[list, int],month:Union[None, list, int]=None,variable:str="ndvi"):
    if month ==None:
        ds_subset = ds.sel(time=ds.time.dt.year.isin([year]))
        df_list, list_dates= get_subplot_year(ds_subset, year =year, var=variable)
    else:
        ds_subset = ds.where(((ds['time.year'].isin([year])) & (ds['time.month'].isin([month]))), drop=True)
        df_list, list_dates= get_subplot_year(ds_subset, year =year, var=variable, months=month)
    return df_list, list_dates
    
def get_subplot_year(ds, var:str="ndvi", year:Union[None, int,list]=None, months:Union[None, list,int]=None):

    if year==None:
        days = 366
    elif type(year)==list:
        if True in [True for y in year if calendar.isleap(y)]:
            days=366
        else: days = 365
    else:
        days=366 if calendar.isleap(year) else 365


    bool_days = False if days==365 else True
    list_dates = get_dates(gap_year=bool_days)
    print(f"days are {days}")

    if (months!=None) & (type(year)==int):
        print(f"For year {year} obtaining only months {months[0]} to {months[-1]} for boxplot")
        last_day = calendar.monthrange(year, months[-1])[1]
        date_start = f"01-{str_month(months[0])}-{year}"
        date_end = f"{last_day}-{str_month(months[-1])}-{year}"
        int_st = time.strptime(date_start, "%d-%m-%Y").tm_yday
        int_end = time.strptime(date_end, "%d-%m-%Y").tm_yday 
        
        day_obj = dfDay(ds = ds, var=var)
        list_new = pd.date_range(datetime.strptime(date_start,"%d-%m-%Y"), datetime.strptime(date_end,"%d-%m-%Y"), freq="D").to_series().dt.strftime('%d-%b').values
        df_list = []
        for day in range(int_st,int_end+1):
            day_obj.get_day(day)
            locals()[day_obj.df_name] = day_obj.df
            df_list.append(locals()[day_obj.df_name][var])
            
        return df_list, list_new

    else:
        print("Calculating the full year for boxplot")
        day_obj = dfDay(ds = ds, var=var)
        df_list = []
        for day in range(1,days+1):
            day_obj.get_day(day)
            locals()[day_obj.df_name] = day_obj.df
            df_list.append(locals()[day_obj.df_name][var])

        print(f"The days are {len(df_list)}")
    
        return df_list, list_dates

def adjust_full_list(year, df_list_all, months=None):
    def str_month(month):
        if month>9:
            return str(month)
        else:
            return "0" + str(month)
    
    df_list_new = df_list_all.copy()

    days=366 if calendar.isleap(year) else 365

    if months!=None:
        last_day = calendar.monthrange(year, months[-1])[1]
        date_start = f"01-{str_month(months[0])}-{year}"
        date_end = f"{last_day}-{str_month(months[-1])}-{year}"
        int_st = time.strptime(date_start, "%d-%m-%Y").tm_yday
        int_end = time.strptime(date_end, "%d-%m-%Y").tm_yday

        if (2 in months) & (1 not in months) & (days ==365) :
            raise NotImplementedError("Subsetting without January in not leap year not implemented") 
        elif (days==365) & (2 in months):
            del df_list_new[59]
            return df_list_new
        else:
            return df_list_new
    else:
        if (days==365):
            del df_list_new[59]
            return df_list_new
        else:
            return df_list_new

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

def plot_vci_2009_event(ds, path=None):
    df_list_all, list_dates_all = get_subplot_year(ds)

    months = [i for i in np.arange(9,13)]
    year = 2009
    df_list_1, list_dates_1 = get_xarray_time_subset(ds=ds, year=year, month=months)
    df_list_all_1 = subsetting_whole(df_list_all = df_list_all, year = year,months=months)

    months = [i for i in np.arange(1,6)]
    year=2010
    df_list_2, list_dates_2 = get_xarray_time_subset(ds=ds, year=year, month=months)
    df_list_all_2 = subsetting_whole(df_list_all =df_list_all, year = year, months=months)

    list_med = pd.Series([p.mean() for p in df_list_1])
    list_med_2 = pd.Series([p.mean() for p in df_list_2])

    all_1 = pd.Series([p.mean() for p in df_list_all_1])
    all_2= pd.Series([p.mean() for p in df_list_all_2])

    list_med.index=list_dates_1
    list_med_2.index=list_dates_2


    fig = plt.figure(figsize=(22,6))
    # set height ratios for subplots
    gs = gridspec.GridSpec(1, 2) 

    # the first subplot
    ax0 = fig.add_subplot(gs[0])

    ##Legend

    pop_a = mpatches.Patch(color='red', label='Mean 2009')
    pop_b = mpatches.Patch(color='darkgreen', label='Mean climatology 2005-2020')
    pop_d = mpatches.Patch(color='red', label='Mean 2010')


    ax0.legend(handles=[pop_a, pop_b],loc="upper right", fontsize=16)
    ax0.set_xticklabels(list_dates_1)
    ax0.set_ylabel("VCI value", fontsize=14)
    ax0.set_xlabel("2009", fontsize=16)


    # log scale for axis Y of the first subplot
    line0 = ax0.plot(list_med, c="red", linestyle="--")
    line2 = ax0.plot(all_1,c="darkgreen")

    # the second subplot
    # shared axis X
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    ax1.set_xticklabels(list_dates_2)
    ax1.legend(handles=[pop_d,pop_b],loc="upper right", fontsize=16)
    ax1.set_xlabel("2010", fontsize=16)


    line3 = ax1.plot(list_med_2, c="red",linestyle="--")
    line4 = ax1.plot(all_2,c="darkgreen")

    n=15
    plt.setp(ax1.get_yticklabels(), visible=False)
    for ax in [ax0, ax1]:
        ax.margins(x=0)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='grey', linestyle='dashed')
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines()) if i % n != 0]
        ax.tick_params(labelrotation=45, tick1On=False)

    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    plt.suptitle("Vegetation Condition Index (VCI)", fontsize=16)
    if path!=None:
        plt.savefig(path)
    plt.show()

def plot_veg_2009_event(ds, path=None):
    df_list_all, list_dates_all = get_subplot_year(ds)

    months = [i for i in np.arange(9,13)]
    year = 2009
    df_list_1, list_dates_1 = get_xarray_time_subset(ds=ds, year=year,month=months, variable="ndvi")
    df_list_all_1 = subsetting_whole(df_list_all =df_list_all, year = year, months=months)

    months = [i for i in np.arange(1,6)]
    year=2010
    df_list_2, list_dates_2 = get_xarray_time_subset(ds=ds, year=year, month=months, variable="ndvi")
    df_list_all_2 = subsetting_whole(df_list_all =df_list_all, months=months, year = year)


    fig = plt.figure(figsize=(22,6))
    # set height ratios for subplots
    gs = gridspec.GridSpec(1, 2) 

    ##Legend

    pop_a = mpatches.Patch(color='red', label='Median 2009')
    pop_b = mpatches.Patch(color='darkgreen', label='Median climatology 2005-2020')
    pop_c = mpatches.Patch(color='lightblue', label='IQR climatology 2005-2020')
    pop_d = mpatches.Patch(color='lightgrey', label='IQR 2009')
    pop_e = mpatches.Patch(color='red', label='Median 2010')
    pop_f = mpatches.Patch(color='lightgrey', label='IQR 2010')


    # the first subplot
    ax0 = fig.add_subplot(gs[0])
    #ax0.set_title("NDVI for 2009")
    ax0.legend(handles=[pop_a, pop_b, pop_d, pop_c], fontsize=16, loc="upper left")
    ax0.set_ylabel("NDVI value", fontsize=14)
    ax0.set_xlabel("2009", fontsize=16)

    # log scale for axis Y of the first subplot
    line0 = ax0.boxplot(df_list_1, showfliers=False, labels=list_dates_1, patch_artist=True,showcaps=False)
    line2 = ax0.boxplot(df_list_all_1, showfliers=False, labels=list_dates_1, patch_artist=True,showcaps=False, manage_ticks=False)

    # the second subplot
    # shared axis X
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    #ax1.set_title("NDVI for 2010")
    ax1.set_xlabel("2010", fontsize=16)
    ax1.legend(handles=[pop_e, pop_b, pop_f, pop_c], fontsize=16, loc="upper left")

    line3 = ax1.boxplot(df_list_2, showfliers=+False, labels=list_dates_2, patch_artist=True,showcaps=False)
    line4 = ax1.boxplot(df_list_all_2, showfliers=False, labels=list_dates_2, patch_artist=True,showcaps=False, manage_ticks=False)

    plt.setp(ax1.get_yticklabels(), visible=False)
    n=30
    for ax in [ax0, ax1]:
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='grey', linestyle='dashed')
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines()) if i % n != 0]
        ax.tick_params(labelrotation=45, tick1On=False)

    for med in line0['medians'], line3['medians']:
        for median in med:
            median.set_color('red')
    for boxes in line0["boxes"] ,line3["boxes"]:
        for box in boxes:
            box.set_color("lightgrey")
            box.set_alpha(0.8)
    for whisker in line0["whiskers"], line3["whiskers"]:
        for whisk in whisker:
            whisk.set_color("white")

    for med in [line2['medians'], line4['medians']]:
        for median in med:
            median.set_color('darkgreen')
    for boxes in [line2["boxes"] ,line4["boxes"]]:
        for box in boxes:
            box.set_color("lightblue")
            box.set_alpha(0.4)
    for whisker in [line2["whiskers"], line4["whiskers"]]:
        for whisk in whisker:
            whisk.set_color("white")

    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    plt.suptitle("Daily NDVI boxplot", fontsize=18)
    plt.subplots_adjust(top=0.95)
    if path!=None:
        plt.savefig(path)
    plt.show()

def plot_veg_3_years(ds, years:list, path=None):
    df_list_all, list_dates_all = get_subplot_year(ds)

    months = [i for i in np.arange(9,13)]
    year_1 = years[0]
    df_list_1, list_dates_1 = get_xarray_time_subset(ds=ds, year=year_1, variable="ndvi")
    df_list_all_1 = adjust_full_list(df_list_all =df_list_all, year = year_1)

    months = [i for i in np.arange(1,6)]
    year_2= years[1]
    df_list_2, list_dates_2 = get_xarray_time_subset(ds=ds, year=year_2, variable="ndvi")
    df_list_all_2 = adjust_full_list(df_list_all =df_list_all, year = year_2)

    year_3 = years[2]
    df_list_3, list_dates_3 = get_xarray_time_subset(ds=ds, year=year_3, variable="ndvi")
    df_list_all_3 = adjust_full_list(df_list_all=df_list_all, year = year_3)


    fig = plt.figure(figsize=(22,6))
    # set height ratios for subplots
    gs = gridspec.GridSpec(1, 3) 

    ##Legend

    pop_a = mpatches.Patch(color='red', label=f'Median {year_1}')
    pop_b = mpatches.Patch(color='darkgreen', label='Median climatology 2005-2020')
    pop_c = mpatches.Patch(color='lightblue', label='IQR climatology 2005-2020')
    pop_d = mpatches.Patch(color='lightgrey', label=f'IQR {year_1}')
    pop_e = mpatches.Patch(color='red', label=f'Median {year_2}')
    pop_f = mpatches.Patch(color='lightgrey', label=f'IQR {year_2}')
    pop_g = mpatches.Patch(color='red', label=f'Median {year_3}')
    pop_h = mpatches.Patch(color='lightgrey', label=f'IQR {year_3}')


    # the first subplot
    ax0 = fig.add_subplot(gs[0])
    #ax0.set_title("NDVI for 2009")
    ax0.legend(handles=[pop_a, pop_b, pop_d, pop_c], fontsize=16, loc="upper left")
    ax0.set_ylabel("NDVI value", fontsize=14)
    ax0.set_xlabel(f"{year_1}", fontsize=16)

    # log scale for axis Y of the first subplot
    line0 = ax0.boxplot(df_list_1, showfliers=False, labels=list_dates_1, patch_artist=True,showcaps=False)
    line2 = ax0.boxplot(df_list_all_1, showfliers=False, labels=list_dates_1, patch_artist=True,showcaps=False, manage_ticks=False)

    # the second subplot
    # shared axis X
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    #ax1.set_title("NDVI for 2010")
    ax1.set_xlabel(f"{year_2}", fontsize=16)
    ax1.legend(handles=[pop_e, pop_b, pop_f, pop_c], fontsize=16, loc="upper left")

    line3 = ax1.boxplot(df_list_2, showfliers=+False, labels=list_dates_2, patch_artist=True,showcaps=False)
    line4 = ax1.boxplot(df_list_all_2, showfliers=False, labels=list_dates_2, patch_artist=True,showcaps=False, manage_ticks=False)

    # the third subplot
    # shared axis X
    ax2 = fig.add_subplot(gs[2], sharey=ax0)
    #ax1.set_title("NDVI for 2010")
    ax2.set_xlabel(f"{year_3}", fontsize=16)
    ax2.legend(handles=[pop_g, pop_b, pop_h, pop_c], fontsize=16, loc="upper left")

    line5 = ax2.boxplot(df_list_3, showfliers=False, labels=list_dates_3, patch_artist=True,showcaps=False)
    line6 = ax2.boxplot(df_list_all_3, showfliers=False, labels=list_dates_3, patch_artist=True,showcaps=False, manage_ticks=False)

    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.ylim(0, 0.35)
    n=30
    for ax in [ax0, ax1, ax2]:
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='grey', linestyle='dashed')
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines()) if i % n != 0]
        ax.tick_params(labelrotation=45, tick1On=False)

    for med in line0['medians'], line3['medians'], line5['medians']:
        for median in med:
            median.set_color('red')
    for boxes in line0["boxes"] ,line3["boxes"],line5["boxes"]:
        for box in boxes:
            box.set_color("lightgrey")
            box.set_alpha(0.8)
    for whisker in line0["whiskers"], line3["whiskers"], line5["whiskers"]:
        for whisk in whisker:
            whisk.set_color("white")

    for med in [line2['medians'], line4['medians'], line6['medians']]:
        for median in med:
            median.set_color('darkgreen')
    for boxes in [line2["boxes"] ,line4["boxes"],line6["boxes"]]:
        for box in boxes:
            box.set_color("lightblue")
            box.set_alpha(0.4)
    for whisker in [line2["whiskers"], line4["whiskers"], line6["whiskers"]]:
        for whisk in whisker:
            whisk.set_color("white")

    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    plt.suptitle("Daily NDVI boxplot", fontsize=18)
    plt.subplots_adjust(top=0.95)
    if path!=None:
        plt.savefig(path)
    plt.show()

def plot_vci_3_years(ds:xr.Dataset, years:list, path=None):
    df_list_all, list_dates_all = get_subplot_year(ds)

    months = [i for i in np.arange(9,13)]
    year_1 = years[0]
    df_list_1, list_dates_1 = get_xarray_time_subset(ds=ds, year=year_1)
    df_list_all_1 = adjust_full_list(df_list_all = df_list_all, year = year_1)

    months = [i for i in np.arange(1,6)]
    year_2=years[1]
    df_list_2, list_dates_2 = get_xarray_time_subset(ds=ds, year=year_2)
    df_list_all_2 = adjust_full_list(df_list_all =df_list_all, year = year_2)

    year_3=years[2]
    df_list_3, list_dates_3 = get_xarray_time_subset(ds=ds, year=year_3)
    df_list_all_3 = adjust_full_list(df_list_all =df_list_all, year = year_3)

    list_med = pd.Series([p.mean() for p in df_list_1])
    list_med_2 = pd.Series([p.mean() for p in df_list_2])
    list_med_3 = pd.Series([p.mean() for p in df_list_3])

    all_1 = pd.Series([p.mean() for p in df_list_all_1])
    all_2= pd.Series([p.mean() for p in df_list_all_2])
    all_3= pd.Series([p.mean() for p in df_list_all_3])

    list_med.index=list_dates_1
    list_med_2.index=list_dates_2
    list_med_3.index=list_dates_3
    fig = plt.figure(figsize=(22,6))
    # set height ratios for subplots
    gs = gridspec.GridSpec(1, 3) 

    # the first subplot
    ax0 = fig.add_subplot(gs[0])

    ##Legend

    pop_a = mpatches.Patch(color='red', label=f'Mean {year_1}')
    pop_b = mpatches.Patch(color='darkgreen', label='Mean climatology 2005-2020')
    pop_d = mpatches.Patch(color='red', label=f'Mean {year_2}')
    pop_e = mpatches.Patch(color='red', label=f'Mean {year_3}')


    ax0.legend(handles=[pop_a, pop_b],loc="upper right", fontsize=16)
    ax0.set_xticklabels(list_dates_1)
    ax0.set_ylabel("VCI value", fontsize=14)
    ax0.set_xlabel(f"{year_1}", fontsize=16)


    # log scale for axis Y of the first subplot
    line0 = ax0.plot(list_med, c="red", linestyle="--")
    line2 = ax0.plot(all_1,c="darkgreen")

    # the second subplot
    # shared axis X
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    ax1.set_xticklabels(list_dates_2)
    ax1.legend(handles=[pop_d,pop_b],loc="upper right", fontsize=16)
    ax1.set_xlabel(f"{year_2}", fontsize=16)


    line3 = ax1.plot(list_med_2, c="red",linestyle="--")
    line4 = ax1.plot(all_2,c="darkgreen")

    # the third subplot
    # shared axis X
    ax2 = fig.add_subplot(gs[2], sharey=ax0)
    ax2.set_xticklabels(list_dates_3)
    ax2.legend(handles=[pop_e,pop_b],loc="upper right", fontsize=16)
    ax2.set_xlabel(f"{year_3}", fontsize=16)


    line5 = ax2.plot(list_med_3, c="red",linestyle="--")
    line6 = ax2.plot(all_3,c="darkgreen")

    n=30
    plt.setp(ax1.get_yticklabels(), visible=False)
    for ax in [ax0, ax1, ax2]:
        ax.margins(x=0)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='grey', linestyle='dashed')
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines()) if i % n != 0]
        ax.tick_params(labelrotation=45, tick1On=False)

    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    plt.suptitle("Vegetation Condition Index (VCI)", fontsize=16)
    if path!=None:
        plt.savefig(path)
    plt.show()

def get_precp_hist(ds:xr.Dataset, variable):
    df_list_all, list_dates_all = get_subplot_year(ds, var=variable)
    print("Gathered the whole climatology to build precipitation boxplot")
    return df_list_all

def plot_precp_3_years(ds:xr.Dataset, years:list, variable, df_list_all:Union[list, None]=None):
    if df_list_all==None:
        print("The climatology data was not provided, now proceeding with its computation...")
        df_list_all, list_dates_all = get_subplot_year(ds, var=variable)

    months = [i for i in np.arange(9,13)]
    year_1 = years[0]
    df_list_1, list_dates_1 = get_xarray_time_subset(ds=ds, year=year_1, variable=variable)
    df_list_all_1 = adjust_full_list(df_list_all =df_list_all, year = year_1)

    months = [i for i in np.arange(1,6)]
    year_2= years[1]
    df_list_2, list_dates_2 = get_xarray_time_subset(ds=ds, year=year_2, variable=variable)
    df_list_all_2 = adjust_full_list(df_list_all =df_list_all, year = year_2)

    year_3= years[2]
    df_list_3, list_dates_3 = get_xarray_time_subset(ds=ds, year=year_3, variable=variable)
    df_list_all_3 = adjust_full_list(df_list_all =df_list_all, year = year_3)

    fig = plt.figure(figsize=(22,6))

    pop_a = mpatches.Patch(color='red', label=f'Median {year_1}')
    pop_b = mpatches.Patch(color='navy', label=f'IQR year {year_1}')
    pop_e = mpatches.Patch(color='red', label=f'Median {year_2}')
    pop_f = mpatches.Patch(color='navy', label=f'IQR year {year_2}')
    pop_c = mpatches.Patch(color='limegreen', label='Median climatology 1979-2020')
    pop_d = mpatches.Patch(color='lightblue', label='IQR climatology 1979-2020')
    pop_g = mpatches.Patch(color='red', label=f'Median {year_3}')
    pop_h = mpatches.Patch(color='navy', label=f'IQR year {year_3}')


    # set height ratios for subplots
    gs = gridspec.GridSpec(1, 3) 

    # the first subplot
    ax0 = fig.add_subplot(gs[0])
    #ax0.set_title(f"{prod} precipitation for 2009", fontsize=20)
    ax0.set_ylabel("Precipitation ERA5 (mm)", fontsize=14)
    ax0.set_xlabel(f"{year_1}", fontsize=16)


    # log scale for axis Y of the first subplot
    line0 = ax0.boxplot(df_list_1, showfliers=False, whis=0,labels = list_dates_1, patch_artist=True,showcaps=False)
    line2 = ax0.boxplot(df_list_all_1, showfliers=False,whis=0, labels = list_dates_1, patch_artist=True,showcaps=False, manage_ticks=False)
    ax0.legend(handles=[pop_a,pop_b, pop_c,pop_d], fontsize=16)

    # the second subplot
    # shared axis X
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    #ax1.set_title(f"{prod} precipitation for 2010",fontsize=20)
    ax1.set_xlabel(f"{year_2}", fontsize=16)
    ax1.legend(handles=[pop_e,pop_f, pop_c,pop_d], fontsize=16)
    line3 = ax1.boxplot(df_list_2, showfliers=False, whis=0,labels=list_dates_2, patch_artist=True,showcaps=False)
    line4 = ax1.boxplot(df_list_all_2, showfliers=False, whis=0, labels=list_dates_2, patch_artist=True,showcaps=False, manage_ticks=False)

    ax2 = fig.add_subplot(gs[2], sharey=ax0)
    ax2.set_xlabel(f"{year_3}", fontsize=16)
    ax2.legend(handles=[pop_g,pop_h, pop_c,pop_d], fontsize=16)

    line5 = ax2.boxplot(df_list_3, showfliers=False, whis=0,labels=list_dates_3, patch_artist=True,showcaps=False)
    line6 = ax2.boxplot(df_list_all_3, showfliers=False, whis=0, labels=list_dates_3, patch_artist=True,showcaps=False, manage_ticks=False)

    plt.setp(ax1.get_yticklabels(), visible=False)
    #plt.setp(ax2.get_yticklabels(), visible=False)

    n=30
    for ax in [ax0, ax1, ax2]:
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='grey', linestyle='dashed')
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines())]
        ax.tick_params(labelrotation=45,tick1On=False)



    for med in line0['medians'], line3['medians'],line5['medians']:
        for median in med:
            median.set_color('red')
    for boxes in line0["boxes"] ,line3["boxes"], line5['boxes']:
        for box in boxes:
            box.set_color("navy")
            box.set_alpha(0.8)
    for whisker in line0["whiskers"], line3["whiskers"], line5['whiskers']:
        for whisk in whisker:
            whisk.set_color("white")
    for med in line2['medians'], line4['medians'], line6['medians']:
        for median in med:
            median.set_color('limegreen')
    for boxes in line2["boxes"] ,line4["boxes"], line6['boxes']:
        for box in boxes:
            box.set_color("lightblue")
            box.set_alpha(0.6)
    for whisker in line2["whiskers"], line4["whiskers"], line6['whiskers']:
        for whisk in whisker:
            whisk.set_color("white")

    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    plt.suptitle("Daily precipitation boxplot", fontsize=18)
    plt.subplots_adjust(top=0.95)
    plt.show()

def plot_precp_2009_event(ds, variable, path=None):
    df_list_all, list_dates_all = get_subplot_year(ds, var=variable)

    months = [i for i in np.arange(9,13)]
    year = 2009
    df_list_1, list_dates_1 = get_xarray_time_subset(ds=ds, year=year, variable=variable, month=months)
    df_list_all_1 = subsetting_whole(df_list_all =df_list_all, year = year, months=months)

    months = [i for i in np.arange(1,6)]
    year=2010
    df_list_2, list_dates_2 = get_xarray_time_subset(ds=ds, year=year, variable=variable, month=months)
    df_list_all_2 = subsetting_whole(df_list_all =df_list_all, year = year, months=months)

    fig = plt.figure(figsize=(22,6))

    pop_a = mpatches.Patch(color='red', label='Median 2009')
    pop_b = mpatches.Patch(color='navy', label='IQR year 2009')
    pop_e = mpatches.Patch(color='red', label='Median 2010')
    pop_f = mpatches.Patch(color='navy', label='IQR year 2010')
    pop_c = mpatches.Patch(color='limegreen', label='Median climatology 1979-2020')
    pop_d = mpatches.Patch(color='lightblue', label='IQR climatology 1979-2020')


    # set height ratios for subplots
    gs = gridspec.GridSpec(1, 2) 

    # the first subplot
    ax0 = fig.add_subplot(gs[0])
    #ax0.set_title(f"{prod} precipitation for 2009", fontsize=20)
    ax0.set_ylabel("Precipitation ERA5 (mm)", fontsize=14)
    ax0.set_xlabel("2009", fontsize=16)


    # log scale for axis Y of the first subplot
    line0 = ax0.boxplot(df_list_1, showfliers=False, whis=0,labels = list_dates_1, patch_artist=True,showcaps=False)
    line2 = ax0.boxplot(df_list_all_1, showfliers=False,whis=0, labels = list_dates_1, patch_artist=True,showcaps=False, manage_ticks=False)
    ax0.legend(handles=[pop_a,pop_b, pop_c,pop_d], fontsize=16)

    # the second subplot
    # shared axis X
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    #ax1.set_title(f"{prod} precipitation for 2010",fontsize=20)
    ax1.set_xlabel("2010", fontsize=16)
    ax1.legend(handles=[pop_e,pop_f, pop_c,pop_d], fontsize=16)
    line3 = ax1.boxplot(df_list_2, showfliers=False, whis=0,labels=list_dates_2, patch_artist=True,showcaps=False)
    line4 = ax1.boxplot(df_list_all_2, showfliers=False, whis=0, labels=list_dates_2, patch_artist=True,showcaps=False, manage_ticks=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    #plt.setp(ax2.get_yticklabels(), visible=False)

    n=15
    for ax in [ax0, ax1]:
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='grey', linestyle='dashed')
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines())]
        ax.tick_params(labelrotation=45,tick1On=False)

    for med in line0['medians'], line3['medians']:
        for median in med:
            median.set_color('red')
    for boxes in line0["boxes"] ,line3["boxes"]:
        for box in boxes:
            box.set_color("navy")
            box.set_alpha(0.8)
    for whisker in line0["whiskers"], line3["whiskers"]:
        for whisk in whisker:
            whisk.set_color("white")
    for med in line2['medians'], line4['medians']:
        for median in med:
            median.set_color('limegreen')
    for boxes in line2["boxes"] ,line4["boxes"]:
        for box in boxes:
            box.set_color("lightblue")
            box.set_alpha(0.6)
    for whisker in line2["whiskers"], line4["whiskers"]:
        for whisk in whisker:
            whisk.set_color("white")

    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    plt.suptitle("Daily precipitation boxplot", fontsize=18)
    plt.subplots_adjust(top=0.95)
    if path!=None:
        plt.savefig(path)
    plt.show()


def plot_spi_3_years(ds, years:list, variable):

    #df_list_all, list_dates_all = get_subplot_year(ds, var=var_target)

    months = [i for i in np.arange(9,13)]
    year_1 = years[0]
    df_list_1, list_dates_1 = get_xarray_time_subset(ds=ds, year=year_1, variable=variable)
    #df_list_all_1 = subsetting_whole(df_list_all, months, year = year)

    months = [i for i in np.arange(1,6)]
    year_2=years[1]
    df_list_2, list_dates_2 = get_xarray_time_subset(ds=ds, year=year_2, variable=variable)
    #df_list_all_2 = subsetting_whole(df_list_all, months, year = year)

    year_3= years[2]
    df_list_3, list_dates_3 = get_xarray_time_subset(ds=ds, year=year_3, variable=variable)


    pop_a = mpatches.Patch(color='red', label=f'SPI median {year_1}')
    pop_b = mpatches.Patch(color='lightblue', label=f'SPI IQR {year_1}')

    pop_c = mpatches.Patch(color='red', label=f'SPI median {year_2}')
    pop_d = mpatches.Patch(color='lightblue', label=f'SPI IQR {year_2}')

    pop_e = mpatches.Patch(color='red', label=f'SPI median {year_3}')
    pop_f = mpatches.Patch(color='lightblue', label=f'SPI IQR {year_3}')

    fig = plt.figure(figsize=(22,6))
    # set height ratios for subplots
    gs = gridspec.GridSpec(1, 3) 

    # the first subplot
    ax0 = fig.add_subplot(gs[0])
    #ax0.set_title(f"{prod} SPI {late} for 2009")

    line0 = ax0.boxplot(df_list_1, showfliers=False, labels=list_dates_1, whis=0,patch_artist=True,showcaps=False,showmeans=False,medianprops=dict(color="red",ls="--",lw=1), meanline=True, meanprops=dict(color="red", ls="-", lw=2))
    ax0.set_xlabel(f"{year_1}", fontsize=16)
    ax0.legend(handles=[pop_a,pop_b], fontsize=16)
    ax0.set_ylabel("SPI value", fontsize=14)

    # the second subplot
    # shared axis X
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    #x1.set_title(f"{prod} SPI {late} for 2010")
    line3 = ax1.boxplot(df_list_2, showfliers=False, labels=list_dates_2,whis=0, patch_artist=True,showcaps=False, showmeans=False,medianprops=dict(color="red",ls="--",lw=1), meanline=True, meanprops=dict(color="red", ls="-", lw=2))
    ax1.set_xlabel(f"{year_2}", fontsize=16)
    ax1.legend(handles=[pop_c,pop_d], fontsize=16)

    ax2 = fig.add_subplot(gs[2], sharey=ax0)
    ax2.set_xlabel(f"{year_3}", fontsize=16)
    ax2.legend(handles=[pop_e,pop_f], fontsize=16)
    line5 = ax2.boxplot(df_list_3, showfliers=False, labels=list_dates_3, whis=0, patch_artist=True,showcaps=False, showmeans=False,medianprops=dict(color="red",ls="--",lw=1), meanline=True, meanprops=dict(color="red", ls="-", lw=2))


    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)

    n=30
    for ax in [ax0, ax1,ax2]:
        ax.set_axisbelow(True)
        ax.axhline(y=0, color='grey', linestyle='--')
        ax.yaxis.grid(color='grey', linestyle='dashed')
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines()) if i % n != 0]
        ax.tick_params(labelrotation=45,tick1On=False)

    for med in line0['medians'], line3['medians'],  line5['medians']:
        for median in med:
            median.set_color('red')
    for boxes in line0["boxes"] ,line3['boxes'],  line5['boxes']:
        for box in boxes:
            box.set_color("lightblue")
    for whisker in line0["whiskers"], line3["whiskers"],  line5['whiskers']:
        for whisk in whisker:
            whisk.set_color("lightgrey")

    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    plt.suptitle("Daily SPI boxplot", fontsize=18)
    plt.subplots_adjust(top=0.95)
    plt.show()

def plot_spi_2009_event(ds, variable, path=None):
    months = [i for i in np.arange(9,13)]
    year = 2009
    df_list_1, list_dates_1 = get_xarray_time_subset(ds=ds, year=year,month=months, variable=variable)
    #df_list_all_1 = subsetting_whole(df_list_all, months, year = year)

    months = [i for i in np.arange(1,6)]
    year=2010
    df_list_2, list_dates_2 = get_xarray_time_subset(ds=ds, year=year, month=months,variable=variable)
    #df_list_all_2 = subsetting_whole(df_list_all, months, year = year)


    pop_a = mpatches.Patch(color='red', label='SPI median 2009')
    pop_b = mpatches.Patch(color='lightblue', label='SPI IQR 2009')

    pop_c = mpatches.Patch(color='red', label='SPI median 2010')
    pop_d = mpatches.Patch(color='lightblue', label='SPI IQR 2010')

    pop_e = mpatches.Patch(color='red', label='SPI median 2011')
    pop_f = mpatches.Patch(color='lightblue', label='SPI IQR 2011')

    fig = plt.figure(figsize=(22,6))
    # set height ratios for subplots
    gs = gridspec.GridSpec(1, 2) 

    # the first subplot
    ax0 = fig.add_subplot(gs[0])
    #ax0.set_title(f"{prod} SPI {late} for 2009")

    line0 = ax0.boxplot(df_list_1, showfliers=False, labels=list_dates_1, whis=0,patch_artist=True,showcaps=False,showmeans=False,medianprops=dict(color="red",ls="--",lw=1), meanline=True, meanprops=dict(color="red", ls="-", lw=2))
    ax0.set_xlabel("2009", fontsize=16)
    ax0.legend(handles=[pop_a,pop_b], fontsize=16)
    ax0.set_ylabel("SPI value", fontsize=14)

    # the second subplot
    # shared axis X
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    #x1.set_title(f"{prod} SPI {late} for 2010")
    line3 = ax1.boxplot(df_list_2, showfliers=False, labels=list_dates_2,whis=0, patch_artist=True,showcaps=False, showmeans=False,medianprops=dict(color="red",ls="--",lw=1), meanline=True, meanprops=dict(color="red", ls="-", lw=2))
    ax1.set_xlabel("2010", fontsize=16)
    ax1.legend(handles=[pop_c,pop_d], fontsize=16)

    plt.setp(ax1.get_yticklabels(), visible=False)

    n=15
    for ax in [ax0, ax1]:
        ax.set_axisbelow(True)
        ax.axhline(y=0, color='grey', linestyle='--')
        ax.yaxis.grid(color='grey', linestyle='dashed')
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines()) if i % n != 0]
        ax.tick_params(labelrotation=45,tick1On=False)

    for med in line0['medians'], line3['medians']:
        for median in med:
            median.set_color('red')
    for boxes in line0["boxes"] ,line3['boxes']:
        for box in boxes:
            box.set_color("lightblue")
    for whisker in line0["whiskers"], line3["whiskers"]:
        for whisk in whisker:
            whisk.set_color("lightgrey")

    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    plt.suptitle("Daily SPI boxplot", fontsize=18)
    plt.subplots_adjust(top=0.95)
    if path!=None:
        plt.savefig(path)
    plt.show()
    
def loop_soil(CONFIG_PATH, level1=True): 
    import pandas as pd
    from datetime import datetime
    import time
    from p_drought_indices.analysis.visualizations.viz_vci_spi import get_subplot_year, adjust_full_list, str_month

    import xarray as xr
    from p_drought_indices.functions.function_clns import load_config, prepare, subsetting_pipeline
    import os
    from p_drought_indices.functions.function_clns import crop_get_spi, crop_get_thresh
    from p_drought_indices.analysis.visualizations.viz_vci_spi import box_plot_year, get_xarray_time_subset, multiple_spi_boxplots, get_subplot_year, subsetting_whole
    import numpy as np
    import warnings
    from matplotlib import gridspec
    import matplotlib.patches as mpatches
    from matplotlib import gridspec
    import calendar
    import matplotlib.pyplot as plt
    from p_drought_indices.analysis.visualizations.viz_vci_spi import plot_precp_2009_event,plot_veg_2009_event, plot_spi_2009_event
    from p_drought_indices.ancillary_vars.esa_landuse import get_level_colors, get_cover_dataset


    warnings.filterwarnings('ignore')


    config = load_config(CONFIG_PATH)
    ds_ndvi = xr.open_dataset(os.path.join(config['NDVI']['ndvi_path'], 'smoothed_ndvi_1.nc'))
    vci = xr.open_dataset(os.path.join(config['NDVI']['ndvi_path'], 'vci_1D.nc'))
    res_ds = xr.open_dataset(os.path.join(config['NDVI']['ndvi_path'], 'percentage_ndvi.nc'))

    config_directories = [config['SPI']['IMERG']['path'], config['SPI']['GPCC']['path'], config['SPI']['CHIRPS']['path'], config['SPI']['ERA5']['path'], config['SPI']['MSWEP']['path'] ]
    config_dir_precp = [config['PRECIP']['IMERG']['path'],config['PRECIP']['CHIRPS_05']['path'], config['PRECIP']['GPCC']['path'], config['PRECIP']['CHIRPS']['path'], config['PRECIP']['ERA5']['path'],  config['PRECIP']['TAMSTAT']['path'],config['PRECIP']['MSWEP']['path']]


    prod = "ERA5"
    late = 60
    product_dir = [f for f in config_dir_precp if prod in f][0]
    list_files = [f for f in os.listdir(product_dir) if (f.endswith(".nc")) and ("merged" in f)]
    precp_ds = xr.open_dataset(os.path.join(product_dir, list_files[0]))
    variable = [var for var in precp_ds.data_vars if var!= "spatial_ref"][0]

    spi_dir = [f for f in config_directories if prod in f][0]
    var_target = f"spi_gamma_{late}"
    files = [f for f in os.listdir(spi_dir) if var_target in f ]
    spi_ds = xr.open_dataset(os.path.join(spi_dir, files[0]))


    from p_drought_indices.ancillary_vars.esa_landuse import get_level_colors, get_cover_dataset

    ndvi_res =prepare(ds_ndvi)
    path = config["DEFAULT"]["images"]
    img_path = os.path.join(path, 'chirps_esa')
    ds_cover = get_cover_dataset(CONFIG_PATH, ndvi_res["ndvi"], img_path, level1=True)
    

    cmap, levels = get_level_colors(ds_cover["Band1"].isel(time=0), level1=True)
    ds_cover.isel(time=0)["Band1"].plot(colors=cmap, levels=levels)

    def clean_multi_nulls(ds):
        # Create a MultiIndex
        ds = ds.stack(pixel=("lat", "lon"))
        # Drop the pixels that only have NA values.
        ds = ds.dropna("pixel", how="all")
        ds = ds.unstack(["pixel"]).sortby(["lat","lon"])
        return ds

    base_path = os.path.join(path, 'soil_type')
    soil_types = np.unique(ds_cover["Band1"].values)[:-1]

    if level1 == True:
    
        values_land_cover = {0	:'Unknown', 20:	'Shrubland',30:'Herbaceous vegetation',40:	'Cropland',
                        50:	'Built-up',60:	'Bare sparse vegetation',70:'Snow and ice', 80:	'Permanent water bodies',
                        90:'Herbaceous wetland',100: 'Moss and lichen', 11:"Closed forest", 
                        12: "Open forest,", 200: "Oceans, seas"}
    
    else:
        values_land_cover = {0	:'Unknown', 20:	'Shrubs',30:	'Herbaceous vegetation',40:	'Cultivated and managed vegetation/agriculture',
                            50:	'Urban',60:	'Bare',70:	'Snow and ice',80:	'Permanent water bodies',90:	'Herbaceous wetland',100: 'Moss and lichen',111: 'Closed forest, evergreen needle leaf',
                            112: 'Closed forest, evergreen broad leaf',115: 'Closed forest, mixed',125: 'Open forest, mixed',113: 'Closed forest, deciduous needle leaf',
                            114: 'Closed forest, deciduous broad leaf',116: 'Closed forest, not matching any of the others',121: 'Open forest, evergreen needle leaf',122: 'Open forest, evergreen broad leaf',
                            123: 'Open forest, deciduous needle leaf',124: 'Open forest, deciduous broad leaf',126: 'Open forest, not matching any of the others',200: 'Oceans, seas'}


    precp_ds =prepare(precp_ds)
    ds_cover_precp = get_cover_dataset(CONFIG_PATH, precp_ds[variable], img_path, level1=True)

    spi_ds =prepare(spi_ds).transpose("time","lat","lon")
    ds_cover_spi = get_cover_dataset(CONFIG_PATH, spi_ds[var_target], img_path, level1=True)

    ds_ndvi =prepare(ds_ndvi)
    ds_cover_ndvi = get_cover_dataset(CONFIG_PATH, ndvi_res["ndvi"], img_path, level1=True)

    for soil_type in soil_types[1:]:
        soil_name = values_land_cover[soil_type].replace(" ","_").replace("/","_")
        print(f"Starting analysis for {soil_name}")

        ### Raw precipitation
        ds_soil = ds_cover_precp[variable].where(ds_cover_precp["Band1"]==soil_type).to_dataset()
        ds_soil = clean_multi_nulls(ds_soil)
        path = os.path.join(base_path,"precp" + "_" + soil_name)
        plot_precp_2009_event(ds_soil,variable=variable, path=path)

        ### SPI
        ds_soil = ds_cover_spi[var_target].where(ds_cover_spi["Band1"]==soil_type).to_dataset()
        ds_soil = clean_multi_nulls(ds_soil)    
        path = os.path.join(base_path,"spi" + "_" + soil_name)
        plot_spi_2009_event(ds_soil,variable=var_target, path=path)

        ### NDVI   
        ds_soil = ds_cover_ndvi["ndvi"].where(ds_cover_ndvi["Band1"]==soil_type).to_dataset()
        ds_soil = clean_multi_nulls(ds_soil)
        path = os.path.join(base_path,"ndvi" + "_" + soil_name)
        plot_veg_2009_event(ds_soil, path=path)
