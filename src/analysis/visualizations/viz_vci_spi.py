import xarray as xr 
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
#import datetime as datetime
import time
import numpy as np
from matplotlib import gridspec
import matplotlib.patches as mpatches
from matplotlib import gridspec
import os
import calendar
from typing import Union
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

def get_dates(gap_year=False):
    if gap_year==False:
        return pd.date_range("01-Jan-2021", "31-Dec-2021", freq="D").to_series().dt.strftime('%d-%b').values
    else:
        return pd.date_range("01-Jan-2020", "31-Dec-2020", freq="D").to_series().dt.strftime('%d-%b').values


def box_plot_year(ds, 
                  var:str="ndvi", 
                  year:Union[None, int,list]=None, 
                  title:str=None, 
                  figsize =(15, 7), 
                  show_means:bool=False):
    
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
    for day in tqdm(range(1,days+1)):
        day_obj.get_day(day)
        locals()[day_obj.df_name] = day_obj.df
        df_list.append(locals()[day_obj.df_name][var])

    bool_days = False if days==365 else True
    list_dates = get_dates(gap_year=bool_days)
 
    fig, ax = plt.subplots(figsize =figsize)
    
    # Creating plot
    if show_means==True:
        boxplot = ax.boxplot(df_list, showfliers=False, 
                             patch_artist=True, labels=list_dates, 
                             showmeans=True,
                             medianprops=dict(color="green",ls="--",lw=1), 
                             meanline=True, 
                             meanprops=dict(color="red", ls="-", lw=2))
    else:
        boxplot = ax.boxplot(df_list, showfliers=False, 
                             patch_artist=True, labels=list_dates)
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
    date_start = f"{year}-{str_month(months[0])}-01"
    date_end = f"{year}-{str_month(months[-1])}-{last_day}"
    int_st = time.strptime(date_start, "%Y-%m-%d").tm_yday
    int_end = time.strptime(date_end, "%Y-%m-%d").tm_yday
    return  df_list_all[int_st-1:int_end]


def get_xarray_time_subset(ds: xr.DataArray, year: Union[list, int], months: Union[None, list, int] = None, variable: str = "ndvi"):
    # Check if year was passed as int
    single_year = isinstance(year, int)
    year = [year] if single_year else year

    if months is not None:
        months = [months] if isinstance(months, int) else months

    if months is None:
        ds_subset = ds.sel(time=ds.time.dt.year.isin(year))
        call_year_arg = year[0] if single_year else year  # depends on what get_subplot_year expects
        df_list, list_dates = get_subplot_year(ds_subset, year=call_year_arg, var=variable)
    else:
        ds_subset = ds.where(((ds.time.dt.year.isin(year)) & (ds.time.dt.month.isin(months))), drop=True)
        call_year_arg = year[0] if single_year else year
        df_list, list_dates = get_subplot_year(ds_subset, year=call_year_arg, var=variable, months=months)

    return df_list, list_dates


def get_subplot_year(ds, 
                     var:str="ndvi", 
                     year:Union[None, int,list]=None, 
                     months:Union[None, list,int]=None, 
                     dask_compute:bool=True):

    import dask
    from dask.diagnostics import ProgressBar

    def process_day(day_of_year, ds, var):
        subset = ds.sel(time=(ds['time.dayofyear'] == day_of_year))
        df = subset.to_dataframe().reset_index(drop=True)
        df = df.dropna(subset=[var])
        return df[var]
    
    def get_idx_dates_months(year, months):
        print(f"For year {year} obtaining only months {months[0]} to {months[-1]} for boxplot")
        last_day = calendar.monthrange(year, months[-1])[1]
        date_start = f"{year}-{str_month(months[0])}-01"
        date_end = f"{year}-{str_month(months[-1])}-{last_day}"
        int_st = time.strptime(date_start, "%Y-%m-%d").tm_yday
        int_end = time.strptime(date_end, "%Y-%m-%d").tm_yday 
        list_new = pd.date_range(datetime.strptime(date_start,"%Y-%m-%d"), \
                                 datetime.strptime(date_end,"%Y-%m-%d"), freq="D")\
                                .to_series().dt.strftime('%d-%b').values
        return int_st, int_end, list_new

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

    if dask_compute==True:
        ds = ds.chunk({'time': -1})
        if months is None:
            delayed_results = [dask.delayed(process_day)(day_of_year, ds, var) for day_of_year in range(1, days+1)]
            
        else:
            int_st, int_end, list_dates = get_idx_dates_months(year, months)
            delayed_results = [dask.delayed(process_day)(day_of_year, ds, var) for day_of_year in range(int_st, int_end+1)]

        with ProgressBar():
            # Compute the delayed computations in parallel
            computed_results = dask.compute(*delayed_results)

            # The computed_results will be a list of DataFrames
            dataframes_list = list(computed_results)   

            return dataframes_list, list_dates 
    
    else:

        if (months!=None) & (type(year)==int):
            int_st, int_end, list_new = get_idx_dates_months(year, months)
            day_obj = dfDay(ds = ds, var=var)
            df_list = []
            for day in tqdm(range(int_st,int_end+1)):
                day_obj.get_day(day)
                locals()[day_obj.df_name] = day_obj.df
                df_list.append(locals()[day_obj.df_name][var])

            return df_list, list_new

        else:
            print("Calculating the full year for boxplot")
            day_obj = dfDay(ds = ds, var=var)
            df_list = []
            for day in tqdm(range(1,days+1)):
                day_obj.get_day(day)
                locals()[day_obj.df_name] = day_obj.df.compute()
                df_list.append(locals()[day_obj.df_name][var])

            print(f"The days are {len(df_list)}")

            return df_list, list_dates

def get_subplot_year(ds, 
                     var: str = "ndvi", 
                     year: Union[None, int, list] = None, 
                     months: Union[None, list, int, list[list]] = None, 
                     dask_compute: bool = True):

    import dask
    from dask.diagnostics import ProgressBar
    import calendar
    import time
    import pandas as pd
    from datetime import datetime
    from tqdm import tqdm

    def str_month(month):
        return f"{month:02d}"

    def process_day(day_of_year, ds, var):
        subset = ds.sel(time=(ds['time.dayofyear'] == day_of_year))
        df = subset.to_dataframe().reset_index(drop=True)
        df = df.dropna(subset=[var])
        return df[var]

    def get_idx_dates_months(year, months):
        print(f"For year {year} obtaining only months {months[0]} to {months[-1]} for boxplot")
        last_day = calendar.monthrange(year, months[-1])[1]
        date_start = f"{year}-{str_month(months[0])}-01"
        date_end = f"{year}-{str_month(months[-1])}-{last_day}"
        int_st = time.strptime(date_start, "%Y-%m-%d").tm_yday
        int_end = time.strptime(date_end, "%Y-%m-%d").tm_yday
        list_new = pd.date_range(
            datetime.strptime(date_start, "%Y-%m-%d"), 
            datetime.strptime(date_end, "%Y-%m-%d"), 
            freq="D"
        ).to_series().dt.strftime('%d-%b').values
        return int_st, int_end, list_new


    # Determine number of days
    if year is None:
        days = 366
        is_leap = True
    elif isinstance(year, list):
        is_leap = any(calendar.isleap(y) for y in year)
        days = 366 if is_leap else 365
    else:
        is_leap = calendar.isleap(year)
        days = 366 if is_leap else 365

    list_dates = get_dates(gap_year=is_leap)
    logger.info(f"days are {days}")

    if dask_compute:
        ds = ds.chunk({'time': -1})
        delayed_results = []
        combined_dates = []

        if months is None:
            delayed_results = [dask.delayed(process_day)(day, ds, var) for day in range(1, days + 1)]
            combined_dates = list_dates

        elif isinstance(year, list) and isinstance(months[0], list):
            for y, m_list in zip(year, months):
                int_st, int_end, date_strs = get_idx_dates_months(y, m_list)
                combined_dates.extend(date_strs)
                delayed_results.extend([dask.delayed(process_day)(day, ds, var) for day in range(int_st, int_end + 1)])
        else:
            int_st, int_end, combined_dates = get_idx_dates_months(year, months)
            delayed_results = [dask.delayed(process_day)(day, ds, var) for day in range(int_st, int_end + 1)]

        with ProgressBar():
            computed_results = dask.compute(*delayed_results)
            df_list = list(computed_results)

        return df_list, combined_dates

    else:
        df_list = []
        combined_dates = []

        if months is None:
            logger.info("Calculating the full year for boxplot")
            for day in tqdm(range(1, days + 1)):
                df = ds.sel(time=(ds['time.dayofyear'] == day)).to_dataframe().dropna(subset=[var])
                df_list.append(df[var])
            combined_dates = list_dates

        elif isinstance(year, list) and isinstance(months[0], list):
            for y, m_list in zip(year, months):
                int_st, int_end, date_strs = get_idx_dates_months(y, m_list)
                combined_dates.extend(date_strs)
                for day in tqdm(range(int_st, int_end + 1)):
                    df = ds.sel(time=(ds['time.dayofyear'] == day)).to_dataframe().dropna(subset=[var])
                    df_list.append(df[var])
        else:
            int_st, int_end, combined_dates = get_idx_dates_months(year, months)
            for day in tqdm(range(int_st, int_end + 1)):
                df = ds.sel(time=(ds['time.dayofyear'] == day)).to_dataframe().dropna(subset=[var])
                df_list.append(df[var])

        return df_list, combined_dates


# def adjust_full_list(year, df_list_all, months=None):
#     def str_month(month):
#         if month>9:
#             return str(month)
#         else:
#             return "0" + str(month)

#     df_list_new = df_list_all.copy()

#     days=366 if calendar.isleap(year) else 365

#     if months!=None:
#         last_day = calendar.monthrange(year, months[-1])[1]
#         date_start = f"01-{str_month(months[0])}-{year}"
#         date_end = f"{last_day}-{str_month(months[-1])}-{year}"
#         int_st = time.strptime(date_start, "%d-%m-%Y").tm_yday
#         int_end = time.strptime(date_end, "%d-%m-%Y").tm_yday

#         if (2 in months) & (1 not in months) & (days ==365) :
#             raise NotImplementedError("Subsetting without January in not leap year not implemented") 
#         elif (days==365) & (2 in months):
#             del df_list_new[59]
#             return df_list_new
#         else:
#             return df_list_new
#     else:
#         if (days==365):
#             del df_list_new[59]
#             return df_list_new
#         else:
#             return df_list_new


def adjust_full_list(year: int, df_list_all: list, months: list = None) -> list:
    """
    Adjusts the full-year daily list (df_list_all) to remove Feb 29 for non-leap years,
    and filters by specific months if provided.

    Args:
        year (int): Target year.
        df_list_all (list): List of 365 or 366 daily values.
        months (list, optional): List of integer months to retain (1–12). Defaults to None.

    Returns:
        list: Filtered and adjusted list of daily values.
    """
    import calendar
    from datetime import datetime

    is_leap = calendar.isleap(year)
    df_list_new = df_list_all.copy()

    feb29_removed = False
    if not is_leap and len(df_list_new) == 366:
        del df_list_new[59]  # Remove Feb 29
        feb29_removed = True

    if months is not None:
        doy_indices = []
        for month in months:
            num_days = calendar.monthrange(year, month)[1]
            for day in range(1, num_days + 1):
                dt = datetime(year, month, day)
                doy = dt.timetuple().tm_yday

                # If Feb 29 was removed, shift DOYs after Feb 28 back by one
                if feb29_removed and doy > 59:
                    doy -= 1

                doy_indices.append(doy - 1)  # 0-based indexing

        df_list_new = [df_list_new[i] for i in doy_indices]

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

# def plot_veg_2009_event(ds, df_list_all=None, path=None):
#     if df_list_all is None:
#         df_list_all, list_dates_all = get_subplot_year(ds)

#     months = [i for i in np.arange(9,13)]
#     year = 2009
#     df_list_1, list_dates_1 = get_xarray_time_subset(ds=ds, year=year,month=months, variable="ndvi")
#     df_list_all_1 = subsetting_whole(df_list_all =df_list_all, year = year, months=months)

#     i = max([np.percentile(l, 75) for l in df_list_1])
#     j = max([np.percentile(l, 75) for l in df_list_all_1])
#     max_1 = max(i, j)

#     i = min([np.percentile(l, 25) for l in df_list_1])
#     j = min([np.percentile(l, 25) for l in df_list_all_1])
#     min_1 = min(i, j)

#     months = [i for i in np.arange(1,6)]
#     year=2010
#     df_list_2, list_dates_2 = get_xarray_time_subset(ds=ds, year=year, month=months, variable="ndvi")
#     df_list_all_2 = subsetting_whole(df_list_all =df_list_all, months=months, year = year)

#     i = max([np.percentile(l, 75) for l in df_list_2])
#     j = max([np.percentile(l, 75) for l in df_list_all_2])
#     max_2 = max(i, j)

#     i = min([np.percentile(l, 25) for l in df_list_2])
#     j = min([np.percentile(l, 25) for l in df_list_all_2])
#     min_2 = min(i, j)

#     ndvi_min = min(min_1, min_2)

#     ndvi_max = max(max_1, max_2)

#     fig = plt.figure(figsize=(22,6))
#     # set height ratios for subplots
#     gs = gridspec.GridSpec(1, 2) 

#     ##Legend

#     pop_a = mpatches.Patch(color='red', label='Median 2009')
#     pop_b = mpatches.Patch(color='darkgreen', label='Median climatology 2005-2020')
#     pop_c = mpatches.Patch(color='lightblue', label='IQR climatology 2005-2020')
#     pop_d = mpatches.Patch(color='lightgrey', label='IQR 2009')
#     pop_e = mpatches.Patch(color='red', label='Median 2010')
#     pop_f = mpatches.Patch(color='lightgrey', label='IQR 2010')


#     # the first subplot
#     ax0 = fig.add_subplot(gs[0])
#     #ax0.set_title("NDVI for 2009")
#     ax0.legend(handles=[pop_a, pop_b, pop_d, pop_c], fontsize=16, loc="upper left")
#     ax0.set_ylabel("NDVI value", fontsize=14)
#     ax0.set_xlabel("2009", fontsize=16)

#     # log scale for axis Y of the first subplot
#     line0 = ax0.boxplot(df_list_1, showfliers=False, labels=list_dates_1, patch_artist=True,showcaps=False)
#     line2 = ax0.boxplot(df_list_all_1, showfliers=False, labels=list_dates_1, patch_artist=True,showcaps=False, manage_ticks=False)

#     # the second subplot
#     # shared axis X
#     ax1 = fig.add_subplot(gs[1], sharey=ax0)
#     #ax1.set_title("NDVI for 2010")
#     ax1.set_xlabel("2010", fontsize=16)
#     ax1.legend(handles=[pop_e, pop_b, pop_f, pop_c], fontsize=16, loc="upper left")

#     line3 = ax1.boxplot(df_list_2, showfliers=+False, labels=list_dates_2, patch_artist=True,showcaps=False)
#     line4 = ax1.boxplot(df_list_all_2, showfliers=False, labels=list_dates_2, patch_artist=True,showcaps=False, manage_ticks=False)

#     plt.setp(ax1.get_yticklabels(), visible=False)
#     n=30
#     for ax in [ax0, ax1]:
#         ax.set_axisbelow(True)
#         ax.yaxis.grid(color='grey', linestyle='dashed')
#         [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
#         [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines()) if i % n != 0]
#         ax.tick_params(labelrotation=45, tick1On=False)

#     for med in line0['medians'], line3['medians']:
#         for median in med:
#             median.set_color('red')
#     for boxes in line0["boxes"] ,line3["boxes"]:
#         for box in boxes:
#             box.set_color("lightgrey")
#             box.set_alpha(0.8)
#     for whisker in line0["whiskers"], line3["whiskers"]:
#         for whisk in whisker:
#             whisk.set_color("white")

#     for med in [line2['medians'], line4['medians']]:
#         for median in med:
#             median.set_color('darkgreen')
#     for boxes in [line2["boxes"] ,line4["boxes"]]:
#         for box in boxes:
#             box.set_color("lightblue")
#             box.set_alpha(0.4)
#     for whisker in [line2["whiskers"], line4["whiskers"]]:
#         for whisk in whisker:
#             whisk.set_color("white")

#     plt.ylim(ndvi_min-0.05, ndvi_max+0.05)

#     gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
#     plt.suptitle("Daily NDVI boxplot", fontsize=18)
#     plt.subplots_adjust(top=0.95)
#     if path!=None:
#         plt.savefig(path)
#     plt.show()

# # def plot_veg_3_years(ds, years:list, path=None):
#     df_list_all, list_dates_all = get_subplot_year(ds)

#     months = [i for i in np.arange(9,13)]
#     year_1 = years[0]
#     df_list_1, list_dates_1 = get_xarray_time_subset(ds=ds, year=year_1, variable="ndvi")
#     df_list_all_1 = adjust_full_list(df_list_all =df_list_all, year = year_1)

#     i = max([np.percentile(l, 75) for l in df_list_1])
#     j = max([np.percentile(l, 75) for l in df_list_all_1])
#     max_1 = max(i, j)

#     i = min([np.percentile(l, 25) for l in df_list_1])
#     j = min([np.percentile(l, 25) for l in df_list_all_1])
#     min_1 = min(i, j)

#     months = [i for i in np.arange(1,6)]
#     year_2= years[1]
#     df_list_2, list_dates_2 = get_xarray_time_subset(ds=ds, year=year_2, variable="ndvi")
#     df_list_all_2 = adjust_full_list(df_list_all =df_list_all, year = year_2)

#     i = max([np.percentile(l, 75) for l in df_list_2])
#     j = i = max([np.percentile(l, 75) for l in df_list_all_2])
#     max_2 = max(i, j)

#     i = min([np.percentile(l, 25) for l in df_list_2])
#     j = i = min([np.percentile(l, 25) for l in df_list_all_2])
#     min_2 = min(i, j)

#     year_3 = years[2]
#     df_list_3, list_dates_3 = get_xarray_time_subset(ds=ds, year=year_3, variable="ndvi")
#     df_list_all_3 = adjust_full_list(df_list_all=df_list_all, year = year_3)

#     i = max([np.percentile(l, 75) for l in df_list_3])
#     j = i = max([np.percentile(l, 75) for l in df_list_all_3])
#     max_3 = max(i, j)

#     i = min([np.percentile(l, 25) for l in df_list_3])
#     j = i = min([np.percentile(l, 25) for l in df_list_all_3])
#     min_3 = min(i, j)

#     max_ndvi =max(max_1, max_2, max_3)
#     min_ndvi = min(min_1, min_2, min_3)

#     fig = plt.figure(figsize=(22,6))
#     # set height ratios for subplots
#     gs = gridspec.GridSpec(1, 3) 

#     ##Legend

#     pop_a = mpatches.Patch(color='red', label=f'Median {year_1}')
#     pop_b = mpatches.Patch(color='darkgreen', label='Median climatology 2005-2020')
#     pop_c = mpatches.Patch(color='lightblue', label='IQR climatology 2005-2020')
#     pop_d = mpatches.Patch(color='lightgrey', label=f'IQR {year_1}')
#     pop_e = mpatches.Patch(color='red', label=f'Median {year_2}')
#     pop_f = mpatches.Patch(color='lightgrey', label=f'IQR {year_2}')
#     pop_g = mpatches.Patch(color='red', label=f'Median {year_3}')
#     pop_h = mpatches.Patch(color='lightgrey', label=f'IQR {year_3}')


#     # the first subplot
#     ax0 = fig.add_subplot(gs[0])
#     #ax0.set_title("NDVI for 2009")
#     ax0.legend(handles=[pop_a, pop_b, pop_d, pop_c], fontsize=16, loc="upper left")
#     ax0.set_ylabel("NDVI value", fontsize=14)
#     ax0.set_xlabel(f"{year_1}", fontsize=16)

#     # log scale for axis Y of the first subplot
#     line0 = ax0.boxplot(df_list_1, showfliers=False, labels=list_dates_1, patch_artist=True,showcaps=False)
#     line2 = ax0.boxplot(df_list_all_1, showfliers=False, labels=list_dates_1, patch_artist=True,showcaps=False, manage_ticks=False)

#     # the second subplot
#     # shared axis X
#     ax1 = fig.add_subplot(gs[1], sharey=ax0)
#     #ax1.set_title("NDVI for 2010")
#     ax1.set_xlabel(f"{year_2}", fontsize=16)
#     ax1.legend(handles=[pop_e, pop_b, pop_f, pop_c], fontsize=16, loc="upper left")

#     line3 = ax1.boxplot(df_list_2, showfliers=+False, labels=list_dates_2, patch_artist=True,showcaps=False)
#     line4 = ax1.boxplot(df_list_all_2, showfliers=False, labels=list_dates_2, patch_artist=True,showcaps=False, manage_ticks=False)

#     # the third subplot
#     # shared axis X
#     ax2 = fig.add_subplot(gs[2], sharey=ax0)
#     #ax1.set_title("NDVI for 2010")
#     ax2.set_xlabel(f"{year_3}", fontsize=16)
#     ax2.legend(handles=[pop_g, pop_b, pop_h, pop_c], fontsize=16, loc="upper left")

#     line5 = ax2.boxplot(df_list_3, showfliers=False, labels=list_dates_3, patch_artist=True,showcaps=False)
#     line6 = ax2.boxplot(df_list_all_3, showfliers=False, labels=list_dates_3, patch_artist=True,showcaps=False, manage_ticks=False)

#     plt.setp(ax1.get_yticklabels(), visible=False)

#     plt.ylim(min_ndvi-0.05, max_ndvi+0.05)
#     n=30
#     for ax in [ax0, ax1, ax2]:
#         ax.set_axisbelow(True)
#         ax.yaxis.grid(color='grey', linestyle='dashed')
#         [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
#         [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines()) if i % n != 0]
#         ax.tick_params(labelrotation=45, tick1On=False)

#     for med in line0['medians'], line3['medians'], line5['medians']:
#         for median in med:
#             median.set_color('red')
#     for boxes in line0["boxes"] ,line3["boxes"],line5["boxes"]:
#         for box in boxes:
#             box.set_color("lightgrey")
#             box.set_alpha(0.8)
#     for whisker in line0["whiskers"], line3["whiskers"], line5["whiskers"]:
#         for whisk in whisker:
#             whisk.set_color("white")

#     for med in [line2['medians'], line4['medians'], line6['medians']]:
#         for median in med:
#             median.set_color('darkgreen')
#     for boxes in [line2["boxes"] ,line4["boxes"],line6["boxes"]]:
#         for box in boxes:
#             box.set_color("lightblue")
#             box.set_alpha(0.4)
#     for whisker in [line2["whiskers"], line4["whiskers"], line6["whiskers"]]:
#         for whisk in whisker:
#             whisk.set_color("white")

#     gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
#     # plt.suptitle("Daily NDVI boxplot", fontsize=18)
#     plt.subplots_adjust(top=0.95)
#     if path!=None:
#         plt.savefig(path)
#     plt.show()


def plot_veg_multiple_years(ds, years: list, path=None):
    df_list_all, list_dates_all = get_subplot_year(ds)
    
    fig_cols = len(years)
    fig = plt.figure(figsize=(7 * fig_cols, 4))  # width proportional to the number of years
    gs = gridspec.GridSpec(1, fig_cols)
    
    all_percentiles_75 = []
    all_percentiles_25 = []
    axes = []
    
    for idx, year in enumerate(years):
        df_list, list_dates = get_xarray_time_subset(ds=ds, year=year, variable="ndvi")
        df_list_all_year = adjust_full_list(df_list_all=df_list_all, year=year)

        # Collect all 75th and 25th percentiles across years
        all_percentiles_75.extend(np.percentile(l, 75) for l in df_list if len(l) > 0)
        all_percentiles_75.extend(np.percentile(l, 75) for l in df_list_all_year if len(l) > 0)
        all_percentiles_25.extend(np.percentile(l, 25) for l in df_list if len(l) > 0)
        all_percentiles_25.extend(np.percentile(l, 25) for l in df_list_all_year if len(l) > 0)
        
        ax = fig.add_subplot(gs[idx])
        axes.append(ax)
        
        line_year = ax.boxplot(df_list, showfliers=False, labels=list_dates, patch_artist=True, showcaps=False)
        line_clim = ax.boxplot(df_list_all_year, showfliers=False, labels=list_dates, patch_artist=True, showcaps=False, manage_ticks=False)
        
        ax.set_xlabel(f"{year}", fontsize=16)
        if idx > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        
        # Rotate and adjust x-axis labels
        for label in ax.get_xticklabels():
            label.set_fontsize(14)
            label.set_rotation(45)
        
        # Show only every n-th label
        n = 30
        ticks = ax.xaxis.get_major_ticks()
        for i, (tick, label) in enumerate(zip(ticks, ax.get_xticklabels())):
            if i % n != 0:
                label.set_visible(False)
                tick.set_visible(False)
        
        # Median and box colors
        for median in line_year['medians']:
            median.set_color('red')
            median.set_linewidth(2.5)
        for box in line_year['boxes']:
            box.set_color("lightgrey")
            box.set_alpha(0.8)
        for whisk in line_year['whiskers']:
            whisk.set_color("white")
        
        for median in line_clim['medians']:
            median.set_color('darkgreen')
            median.set_linewidth(2.5)
        for box in line_clim['boxes']:
            box.set_color("lightblue")
            box.set_alpha(0.4)
        for whisk in line_clim['whiskers']:
            whisk.set_color("white")
        
        # Grid and ticks
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='grey', linestyle='dashed')
    
    # Set same y-limits for all axes based on all years
    global_min = min(all_percentiles_25) - 0.05
    global_max = max(all_percentiles_75) + 0.05
    for ax in axes:
        ax.set_ylim(global_min, global_max)
    
    axes[0].set_ylabel("NDVI value", fontsize=14)
    
    # Create a shared legend outside
    pop_year = mpatches.Patch(color='red', label='Median year')
    pop_clim_median = mpatches.Patch(color='darkgreen', label='Median climatology 2005-2020')
    pop_clim_iqr = mpatches.Patch(color='lightblue', label='IQR climatology 2005-2020')
    pop_year_iqr = mpatches.Patch(color='lightgrey', label='IQR year')

    fig.legend(handles=[pop_year, pop_clim_median, pop_year_iqr, pop_clim_iqr],
               loc='center left', fontsize=14, bbox_to_anchor=(1.05, 1.0))

    gs.tight_layout(fig, rect=[0, 0, 0.95, 1])
    plt.subplots_adjust(top=0.9)

    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    plt.show(block=False)
    time.sleep(5)
    plt.close()


# def plot_vci_3_years(ds:xr.Dataset, years:list, path=None):
#     df_list_all, list_dates_all = get_subplot_year(ds)

#     months = [i for i in np.arange(9,13)]
#     year_1 = years[0]
#     df_list_1, list_dates_1 = get_xarray_time_subset(ds=ds, year=year_1)
#     df_list_all_1 = adjust_full_list(df_list_all = df_list_all, year = year_1)

#     months = [i for i in np.arange(1,6)]
#     year_2=years[1]
#     df_list_2, list_dates_2 = get_xarray_time_subset(ds=ds, year=year_2)
#     df_list_all_2 = adjust_full_list(df_list_all =df_list_all, year = year_2)

#     year_3=years[2]
#     df_list_3, list_dates_3 = get_xarray_time_subset(ds=ds, year=year_3)
#     df_list_all_3 = adjust_full_list(df_list_all =df_list_all, year = year_3)

#     list_med = pd.Series([p.mean() for p in df_list_1])
#     list_med_2 = pd.Series([p.mean() for p in df_list_2])
#     list_med_3 = pd.Series([p.mean() for p in df_list_3])

#     all_1 = pd.Series([p.mean() for p in df_list_all_1])
#     all_2= pd.Series([p.mean() for p in df_list_all_2])
#     all_3= pd.Series([p.mean() for p in df_list_all_3])

#     list_med.index=list_dates_1
#     list_med_2.index=list_dates_2
#     list_med_3.index=list_dates_3
#     fig = plt.figure(figsize=(12,4))
#     # set height ratios for subplots
#     gs = gridspec.GridSpec(1, 3) 

#     # the first subplot
#     ax0 = fig.add_subplot(gs[0])

#     ##Legend

#     pop_a = mpatches.Patch(color='red', label=f'Mean {year_1}')
#     pop_b = mpatches.Patch(color='darkgreen', label='Mean climatology 2005-2020')
#     pop_d = mpatches.Patch(color='red', label=f'Mean {year_2}')
#     pop_e = mpatches.Patch(color='red', label=f'Mean {year_3}')


#     ax0.legend(handles=[pop_a, pop_b],loc="upper right", fontsize=16)
#     ax0.set_xticklabels(list_dates_1)
#     ax0.set_ylabel("VCI value", fontsize=14)
#     ax0.set_xlabel(f"{year_1}", fontsize=16)


#     # log scale for axis Y of the first subplot
#     line0 = ax0.plot(list_med, c="red", linestyle="--")
#     line2 = ax0.plot(all_1,c="darkgreen")

#     # the second subplot
#     # shared axis X
#     ax1 = fig.add_subplot(gs[1], sharey=ax0)
#     ax1.set_xticklabels(list_dates_2)
#     ax1.legend(handles=[pop_d,pop_b],loc="upper right", fontsize=16)
#     ax1.set_xlabel(f"{year_2}", fontsize=16)


#     line3 = ax1.plot(list_med_2, c="red",linestyle="--")
#     line4 = ax1.plot(all_2,c="darkgreen")

#     # the third subplot
#     # shared axis X
#     ax2 = fig.add_subplot(gs[2], sharey=ax0)
#     ax2.set_xticklabels(list_dates_3)
#     ax2.legend(handles=[pop_e,pop_b],loc="upper right", fontsize=16)
#     ax2.set_xlabel(f"{year_3}", fontsize=16)


#     line5 = ax2.plot(list_med_3, c="red",linestyle="--")
#     line6 = ax2.plot(all_3,c="darkgreen")

#     n=30
#     plt.setp(ax1.get_yticklabels(), visible=False)
#     for ax in [ax0, ax1, ax2]:
#         ax.margins(x=0)
#         ax.set_axisbelow(True)
#         ax.yaxis.grid(color='grey', linestyle='dashed')
#         [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
#         [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines()) if i % n != 0]
#         ax.tick_params(labelrotation=45, tick1On=False)

#     gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
#     plt.suptitle("Vegetation Condition Index (VCI)", fontsize=16)
#     if path!=None:
#         plt.savefig(path)
#     plt.show()


def plot_vci_multiple_years(ds: xr.Dataset, years: list, var:str="ndvi", path=None, label_every=30):
    df_list_all, list_dates_all = get_subplot_year(ds, var)
    
    fig = plt.figure(figsize=(7 * len(years), 4))
    gs = gridspec.GridSpec(1, len(years), wspace=0.3)

    handles = [
        mpatches.Patch(color='red', label='Mean year'),
        mpatches.Patch(color='darkgreen', label='Mean climatology')
    ]

    for idx, year in enumerate(years):
        df_list, list_dates = get_xarray_time_subset(ds=ds, year=year, variable=var)
        df_list_all_year = adjust_full_list(df_list_all=df_list_all, year=year, )

        list_med = pd.Series([p.mean() for p in df_list])
        all_clim = pd.Series([p.mean() for p in df_list_all_year])

        list_med.index = list_dates
        all_clim.index = list_dates

        # Drop the last value
        list_med = list_med.iloc[:-1]
        all_clim = all_clim.iloc[:-1]

        ax = fig.add_subplot(gs[idx], sharey=fig.axes[0] if idx > 0 else None)

        ax.plot(list_med, c="red", linestyle="--")
        ax.plot(all_clim, c="darkgreen")

        ax.set_xlabel(str(year), fontsize=14)

        # Set x-ticks every `label_every` records
        xticks = list_med.index[::label_every]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=45, ha='right', fontsize=14)


        if idx == 0:
            ax.set_ylabel("VCI value", fontsize=14)
        else:
            ax.tick_params(labelleft=False)  # Hide y-axis labels but keep ticks and grid

        ax.grid(axis='y', linestyle='dashed', color='grey')

    # Add a legend outside the whole figure
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1.0), fontsize=12)

    fig.tight_layout(rect=[0, 0, 0.95, 0.95])

    if path:
        plt.savefig(path, bbox_inches='tight')
    plt.show(block=False)
    time.sleep(5)
    plt.close()

def get_precp_hist(ds:xr.Dataset, variable):
    df_list_all, list_dates_all = get_subplot_year(ds, var=variable)
    print("Gathered the whole climatology to build precipitation boxplot")
    return df_list_all

# def plot_precp_3_years(ds:xr.Dataset, years:list, variable, df_list_all:Union[list, None]=None):
#     if df_list_all==None:
#         print("The climatology data was not provided, now proceeding with its computation...")
#         df_list_all, list_dates_all = get_subplot_year(ds, var=variable)

#     months = [i for i in np.arange(9,13)]
#     year_1 = years[0]
#     df_list_1, list_dates_1 = get_xarray_time_subset(ds=ds, year=year_1, variable=variable)
#     df_list_all_1 = adjust_full_list(df_list_all =df_list_all, year = year_1)

#     months = [i for i in np.arange(1,6)]
#     year_2= years[1]
#     df_list_2, list_dates_2 = get_xarray_time_subset(ds=ds, year=year_2, variable=variable)
#     df_list_all_2 = adjust_full_list(df_list_all =df_list_all, year = year_2)

#     year_3= years[2]
#     df_list_3, list_dates_3 = get_xarray_time_subset(ds=ds, year=year_3, variable=variable)
#     df_list_all_3 = adjust_full_list(df_list_all =df_list_all, year = year_3)

#     fig = plt.figure(figsize=(22,6))

#     pop_a = mpatches.Patch(color='red', label=f'Median {year_1}')
#     pop_b = mpatches.Patch(color='navy', label=f'IQR year {year_1}')
#     pop_e = mpatches.Patch(color='red', label=f'Median {year_2}')
#     pop_f = mpatches.Patch(color='navy', label=f'IQR year {year_2}')
#     pop_c = mpatches.Patch(color='limegreen', label='Median climatology 1979-2020')
#     pop_d = mpatches.Patch(color='lightblue', label='IQR climatology 1979-2020')
#     pop_g = mpatches.Patch(color='red', label=f'Median {year_3}')
#     pop_h = mpatches.Patch(color='navy', label=f'IQR year {year_3}')


#     # set height ratios for subplots
#     gs = gridspec.GridSpec(1, 3) 

#     # the first subplot
#     ax0 = fig.add_subplot(gs[0])
#     #ax0.set_title(f"{prod} precipitation for 2009", fontsize=20)
#     ax0.set_ylabel("Precipitation ERA5 (mm)", fontsize=14)
#     ax0.set_xlabel(f"{year_1}", fontsize=16)


#     # log scale for axis Y of the first subplot
#     line0 = ax0.boxplot(df_list_1, showfliers=False, whis=0,labels = list_dates_1, patch_artist=True,showcaps=False)
#     line2 = ax0.boxplot(df_list_all_1, showfliers=False,whis=0, labels = list_dates_1, patch_artist=True,showcaps=False, manage_ticks=False)
#     ax0.legend(handles=[pop_a,pop_b, pop_c,pop_d], fontsize=16)

#     # the second subplot
#     # shared axis X
#     ax1 = fig.add_subplot(gs[1], sharey=ax0)
#     #ax1.set_title(f"{prod} precipitation for 2010",fontsize=20)
#     ax1.set_xlabel(f"{year_2}", fontsize=16)
#     ax1.legend(handles=[pop_e,pop_f, pop_c,pop_d], fontsize=16)
#     line3 = ax1.boxplot(df_list_2, showfliers=False, whis=0,labels=list_dates_2, patch_artist=True,showcaps=False)
#     line4 = ax1.boxplot(df_list_all_2, showfliers=False, whis=0, labels=list_dates_2, patch_artist=True,showcaps=False, manage_ticks=False)

#     ax2 = fig.add_subplot(gs[2], sharey=ax0)
#     ax2.set_xlabel(f"{year_3}", fontsize=16)
#     ax2.legend(handles=[pop_g,pop_h, pop_c,pop_d], fontsize=16)

#     line5 = ax2.boxplot(df_list_3, showfliers=False, whis=0,labels=list_dates_3, patch_artist=True,showcaps=False)
#     line6 = ax2.boxplot(df_list_all_3, showfliers=False, whis=0, labels=list_dates_3, patch_artist=True,showcaps=False, manage_ticks=False)

#     plt.setp(ax1.get_yticklabels(), visible=False)
#     #plt.setp(ax2.get_yticklabels(), visible=False)

#     n=30
#     for ax in [ax0, ax1, ax2]:
#         ax.set_axisbelow(True)
#         ax.yaxis.grid(color='grey', linestyle='dashed')
#         [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
#         [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines())]
#         ax.tick_params(labelrotation=45,tick1On=False)



#     for med in line0['medians'], line3['medians'],line5['medians']:
#         for median in med:
#             median.set_color('red')
#     for boxes in line0["boxes"] ,line3["boxes"], line5['boxes']:
#         for box in boxes:
#             box.set_color("navy")
#             box.set_alpha(0.8)
#     for whisker in line0["whiskers"], line3["whiskers"], line5['whiskers']:
#         for whisk in whisker:
#             whisk.set_color("white")
#     for med in line2['medians'], line4['medians'], line6['medians']:
#         for median in med:
#             median.set_color('limegreen')
#     for boxes in line2["boxes"] ,line4["boxes"], line6['boxes']:
#         for box in boxes:
#             box.set_color("lightblue")
#             box.set_alpha(0.6)
#     for whisker in line2["whiskers"], line4["whiskers"], line6['whiskers']:
#         for whisk in whisker:
#             whisk.set_color("white")

#     gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
#     plt.suptitle("Daily precipitation boxplot", fontsize=18)
#     plt.subplots_adjust(top=0.95)
#     plt.show()

# def plot_precp_multiple_years(ds:xr.Dataset, years:list, variable, months:Union[None, list]=None, path=None, df_list_all:Union[list, None]=None):
#     import matplotlib.patches as mpatches
#     import matplotlib.gridspec as gridspec
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import pandas as pd

#     if df_list_all is None:
#         print("The climatology data was not provided, now proceeding with its computation...")
#         df_list_all, list_dates_all = get_subplot_year(ds, var=variable)

#     fig = plt.figure(figsize=(6 * len(years), 4))
#     gs = gridspec.GridSpec(1, len(years)) 

#     colors = ['red', 'navy', 'limegreen', 'lightblue']
#     # colors = ["mediumblue", "orange", "forestgreen", "indigo" ] 
#     # colors = ["indianred", "darkgreen", "gold", "royalblue"]
#     median_color = colors[0]
#     iqr_color = colors[1]
#     climatology_median_color = colors[2]
#     climatology_iqr_color = colors[3]

#     handles = [
#         mpatches.Patch(color=median_color, label='Median year'),
#         mpatches.Patch(color=iqr_color, label='IQR year'),
#         mpatches.Patch(color=climatology_median_color, label='Median climatology'),
#         mpatches.Patch(color=climatology_iqr_color, label='IQR climatology')
#     ]

#     axes = []
#     for idx, year in enumerate(years):
#         # months = np.arange(9,13) if idx == 0 else np.arange(1,6)
#         df_list, list_dates = get_xarray_time_subset(ds=ds, year=year, variable=variable, months=months)
#         df_list_all_year = adjust_full_list(df_list_all=df_list_all, year=year, months=months)

#         ax = fig.add_subplot(gs[idx], sharey=axes[0] if axes else None)
#         axes.append(ax)

#         box1 = ax.boxplot(df_list, showfliers=False, whis=0, labels=list_dates, patch_artist=True, showcaps=False)
#         box2 = ax.boxplot(df_list_all_year, showfliers=False, whis=0, labels=list_dates, patch_artist=True, showcaps=False, manage_ticks=False)

#         ax.set_xlabel(f"{year}", fontsize=14)
#         ax.set_ylim(0, 10)
#         if idx == 0:
#             ax.set_ylabel("Precipitation (mm)", fontsize=14)
#         else:
#             plt.setp(ax.get_yticklabels(), visible=False)

#         n = 30
#         ax.set_axisbelow(True)
#         ax.yaxis.grid(color='grey', linestyle='dashed')
#         [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
#         [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines())]
#         ax.tick_params(labelrotation=45, tick1On=False, labelsize=12)

#         # Custom boxplot styling
#         for med in box1['medians']:
#             med.set_color(median_color)
#             med.set_linewidth(4)
#         for box in box1['boxes']:
#             box.set_color(iqr_color)
#             box.set_alpha(0.8)
#         for whisk in box1['whiskers']:
#             whisk.set_color("white")
#         for med in box2['medians']:
#             med.set_color(climatology_median_color)
#             med.set_linewidth(4)
#         for box in box2['boxes']:
#             box.set_color(climatology_iqr_color)
#             box.set_alpha(0.6)
#         for whisk in box2['whiskers']:
#             whisk.set_color("white")

#     gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
#     # plt.suptitle("Daily precipitation boxplot", fontsize=18)

#     # Place legend outside
#     fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05 , 1), fontsize=12)
#     plt.subplots_adjust(top=0.85)

#     if path:
#         plt.savefig(path, bbox_inches='tight')
#     plt.show()

def plot_precp_multiple_years(ds:xr.Dataset, years:list, variable, months:Union[None, list]=None, path=None, df_list_all:Union[list, None]=None):
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if df_list_all is None:
        print("The climatology data was not provided, now proceeding with its computation...")
        df_list_all, list_dates_all = get_subplot_year(ds, var=variable)

    fig = plt.figure(figsize=(6 * len(years), 4))
    gs = gridspec.GridSpec(1, len(years)) 

    colors = ['red', 'navy', 'limegreen', 'lightblue']
    median_color = colors[0]
    iqr_color = colors[1]
    climatology_median_color = colors[2]
    climatology_iqr_color = colors[3]

    handles = [
        mpatches.Patch(color=median_color, label='Median year'),
        mpatches.Patch(color=iqr_color, label='IQR year'),
        mpatches.Patch(color=climatology_median_color, label='Median climatology'),
        mpatches.Patch(color=climatology_iqr_color, label='IQR climatology')
    ]

    # Check if months is a list of lists (one list of months per year)
    if isinstance(months, list) and all(isinstance(m, list) for m in months) and len(months) == len(years):
        year_month_pairs = zip(years, months)
    else:
        year_month_pairs = ((y, months) for y in years)  # same months (or None) for all years

    axes = []
    for idx, (year, month_set) in enumerate(year_month_pairs):
        df_list, list_dates = get_xarray_time_subset(ds=ds, year=year, variable=variable, months=month_set)
        df_list_all_year = adjust_full_list(df_list_all=df_list_all, year=year, months=month_set)

        ax = fig.add_subplot(gs[idx], sharey=axes[0] if axes else None)
        axes.append(ax)

        box1 = ax.boxplot(df_list, showfliers=False, whis=0, labels=list_dates, patch_artist=True, showcaps=False)
        box2 = ax.boxplot(df_list_all_year, showfliers=False, whis=0, labels=list_dates, patch_artist=True, showcaps=False, manage_ticks=False)

        ax.set_xlabel(f"{year}", fontsize=14)
        ax.set_ylim(0, 15)
        if idx == 0:
            ax.set_ylabel("Precipitation (mm)", fontsize=14)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

        n = 30
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='grey', linestyle='dashed')
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines())]
        ax.tick_params(labelrotation=45, tick1On=False, labelsize=12)

        # Custom boxplot styling
        for med in box1['medians']:
            med.set_color(median_color)
            med.set_linewidth(4)
        for box in box1['boxes']:
            box.set_color(iqr_color)
            box.set_alpha(0.8)
        for whisk in box1['whiskers']:
            whisk.set_color("white")
        for med in box2['medians']:
            med.set_color(climatology_median_color)
            med.set_linewidth(4)
        for box in box2['boxes']:
            box.set_color(climatology_iqr_color)
            box.set_alpha(0.6)
        for whisk in box2['whiskers']:
            whisk.set_color("white")

    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02 , 0.5), fontsize=12)
    plt.subplots_adjust(top=0.85)

    if path:
        plt.savefig(path, bbox_inches='tight')
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


# def plot_spi_3_years(ds, years:list, variable,  df_list_all:Union[list, None]=None):
#     #if df_list_all==None:
#         #print("The climatology data was not provided, now proceeding with its computation...")
#         #df_list_all, list_dates_all = get_subplot_year(ds, var=variable)
#     #df_list_all, list_dates_all = get_subplot_year(ds, var=var_target)

#     months = [i for i in np.arange(9,13)]
#     year_1 = years[0]
#     df_list_1, list_dates_1 = get_xarray_time_subset(ds=ds, year=year_1, variable=variable)
#     #df_list_all_1 = subsetting_whole(df_list_all, months, year = year)

#     months = [i for i in np.arange(1,6)]
#     year_2=years[1]
#     df_list_2, list_dates_2 = get_xarray_time_subset(ds=ds, year=year_2, variable=variable)
#     #df_list_all_2 = subsetting_whole(df_list_all, months, year = year)

#     year_3= years[2]
#     df_list_3, list_dates_3 = get_xarray_time_subset(ds=ds, year=year_3, variable=variable)


#     pop_a = mpatches.Patch(color='red', label=f'SPI median {year_1}')
#     pop_b = mpatches.Patch(color='lightblue', label=f'SPI IQR {year_1}')

#     pop_c = mpatches.Patch(color='red', label=f'SPI median {year_2}')
#     pop_d = mpatches.Patch(color='lightblue', label=f'SPI IQR {year_2}')

#     pop_e = mpatches.Patch(color='red', label=f'SPI median {year_3}')
#     pop_f = mpatches.Patch(color='lightblue', label=f'SPI IQR {year_3}')

#     fig = plt.figure(figsize=(22,6))
#     # set height ratios for subplots
#     gs = gridspec.GridSpec(1, 3) 

#     # the first subplot
#     ax0 = fig.add_subplot(gs[0])
#     #ax0.set_title(f"{prod} SPI {late} for 2009")

#     line0 = ax0.boxplot(df_list_1, showfliers=False, labels=list_dates_1, whis=0,patch_artist=True,showcaps=False,showmeans=False,medianprops=dict(color="red",ls="--",lw=1), meanline=True, meanprops=dict(color="red", ls="-", lw=2))
#     ax0.set_xlabel(f"{year_1}", fontsize=16)
#     ax0.legend(handles=[pop_a,pop_b], fontsize=16)
#     ax0.set_ylabel("SPI value", fontsize=14)

#     # the second subplot
#     # shared axis X
#     ax1 = fig.add_subplot(gs[1], sharey=ax0)
#     #x1.set_title(f"{prod} SPI {late} for 2010")
#     line3 = ax1.boxplot(df_list_2, showfliers=False, labels=list_dates_2,whis=0, patch_artist=True,showcaps=False, showmeans=False,medianprops=dict(color="red",ls="--",lw=1), meanline=True, meanprops=dict(color="red", ls="-", lw=2))
#     ax1.set_xlabel(f"{year_2}", fontsize=16)
#     ax1.legend(handles=[pop_c,pop_d], fontsize=16)

#     ax2 = fig.add_subplot(gs[2], sharey=ax0)
#     ax2.set_xlabel(f"{year_3}", fontsize=16)
#     ax2.legend(handles=[pop_e,pop_f], fontsize=16)
#     line5 = ax2.boxplot(df_list_3, showfliers=False, labels=list_dates_3, whis=0, patch_artist=True,showcaps=False, showmeans=False,medianprops=dict(color="red",ls="--",lw=1), meanline=True, meanprops=dict(color="red", ls="-", lw=2))


#     plt.setp(ax1.get_yticklabels(), visible=False)
#     plt.setp(ax2.get_yticklabels(), visible=False)

#     n=30
#     for ax in [ax0, ax1,ax2]:
#         ax.set_axisbelow(True)
#         ax.axhline(y=0, color='grey', linestyle='--')
#         ax.yaxis.grid(color='grey', linestyle='dashed')
#         [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
#         [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines()) if i % n != 0]
#         ax.tick_params(labelrotation=45,tick1On=False)

#     for med in line0['medians'], line3['medians'],  line5['medians']:
#         for median in med:
#             median.set_color('red')
#     for boxes in line0["boxes"] ,line3['boxes'],  line5['boxes']:
#         for box in boxes:
#             box.set_color("lightblue")
#     for whisker in line0["whiskers"], line3["whiskers"],  line5['whiskers']:
#         for whisk in whisker:
#             whisk.set_color("lightgrey")

#     gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
#     plt.suptitle("Daily SPI boxplot", fontsize=18)
#     plt.subplots_adjust(top=0.95)
#     plt.show()


def plot_spi_multiple_years(ds, years: list, variable, path=None, df_list_all: Union[list, None] = None):
    if len(years) < 1 or len(years) > 3:
        raise ValueError("Please provide between 1 and 3 years.")

    fig = plt.figure(figsize=(8 * len(years), 4))  # Adjust figure width dynamically
    gs = gridspec.GridSpec(1, len(years))

    axes = []
    lines = []

    # Create patches for the global legend
    pop_median = mpatches.Patch(color='red', label='SPI median')
    pop_iqr = mpatches.Patch(color='lightblue', label='SPI IQR')

    for idx, year in enumerate(years):
        df_list, list_dates = get_xarray_time_subset(ds=ds[variable] , year=year, variable=variable)
        ax = fig.add_subplot(gs[idx], sharey=axes[0] if axes else None)
        line = ax.boxplot(
            df_list,
            showfliers=False,
            labels=list_dates,
            whis=0,
            patch_artist=True,
            showcaps=False,
            showmeans=False,
            medianprops=dict(color="red", ls="--", lw=2),
            meanline=True,
            meanprops=dict(color="red", ls="-", lw=2)
        )

        ax.set_xlabel(f"{year}", fontsize=16)
        if idx > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            ax.set_ylabel("SPI value", fontsize=14)

        axes.append(ax)
        lines.append(line)

    # Styling loop
    n = 30
    for ax in axes:
        ax.set_axisbelow(True)
        ax.axhline(y=0, color='grey', linestyle='--')
        ax.yaxis.grid(color='grey', linestyle='dashed')
        [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_gridlines()) if i % n != 0]
        ax.tick_params(labelrotation=45, tick1On=False, labelsize=12)

    for line in lines:
        for median in line['medians']:
            median.set_color('red')
            median.set_linewidth(3)
        for box in line['boxes']:
            box.set_color('lightblue')
        for whisk in line['whiskers']:
            whisk.set_color('lightgrey')

    # Layout and legend
    gs.tight_layout(fig, rect=[0, 0, 0.95, 0.95])
    fig.legend(handles=[pop_median, pop_iqr], loc='center left', bbox_to_anchor=(1.05, 1), fontsize=14)
    plt.suptitle("Daily SPI boxplot", fontsize=18)
    plt.subplots_adjust(top=0.9)
    if path:
        plt.savefig(path, bbox_inches='tight')
    plt.show()

def plot_spi_event(ds:xr.Dataset, variable:str, year:int, months:list, path=None, df_list_all=None):
    if df_list_all is None:
        df_list_all, list_dates_all = get_subplot_year(ds, var=variable)

    df_list_1, list_dates_1 = get_xarray_time_subset(ds=ds, year=year,month=months, variable=variable)
    df_list_all_1 = subsetting_whole(df_list_all =df_list_all, year = year, months=months)

    pop_a = mpatches.Patch(color='red', label=f'SPI median {year}')
    pop_b = mpatches.Patch(color='lightblue', label=f'SPI IQR {year}')


    fig = plt.figure(figsize=(8,5))
    # set height ratios for subplots
    gs = gridspec.GridSpec(1, 1) 

    # the first subplot
    ax0 = fig.add_subplot(gs[0])
    #ax0.set_title(f"{prod} SPI {late} for 2009")

    line0 = ax0.boxplot(df_list_1, showfliers=False, labels=list_dates_1, whis=0,patch_artist=True,showcaps=False,showmeans=False,medianprops=dict(color="red",ls="--",lw=1), meanline=True, meanprops=dict(color="red", ls="-", lw=2))
    ax0.set_xlabel(f"{year}", fontsize=16)
    ax0.legend(handles=[pop_a,pop_b], fontsize=16)
    ax0.set_ylabel("SPI value", fontsize=14)

    n=30
    for ax in [ax0]:
        ax.set_axisbelow(True)
        ax.axhline(y=0, color='grey', linestyle='--')
        ax.yaxis.grid(color='grey', linestyle='dashed')
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines()) if i % n != 0]
        ax.tick_params(labelrotation=45,tick1On=False)

    for median in line0['medians']:
        median.set_color('red')
    for box in line0["boxes"]:
        box.set_color("lightblue")
    for whisk in line0["whiskers"]:
        whisk.set_color("lightgrey")

    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    # plt.suptitle(f"Year {year} daily SPI boxplot", fontsize=18)
    plt.subplots_adjust(top=0.95)
    if path!=None:
        plt.savefig(path)
    plt.show()

from typing import List

def plot_precp_event(ds:xr.Dataset, var:str, year:int, months:list, df_list_all:list=None, path:str=None):
    if df_list_all is None:
        df_list_all, list_dates_all = get_subplot_year(ds, var=var)

    df_list_1, list_dates_1 = get_xarray_time_subset(ds=ds, year=year, month=months, 
                                                     variable=var)
    df_list_all_1 = subsetting_whole(df_list_all=df_list_all, year=year, months=months)

    i = max([np.percentile(l, 75) for l in df_list_1])
    j = max([np.percentile(l, 75) for l in df_list_all_1])
    max_precp = max(i, j)

    i = min([np.percentile(l, 25) for l in df_list_1])
    j = min([np.percentile(l, 25) for l in df_list_all_1])
    min_precp = min(i, j)

    fig = plt.figure(figsize=(8,4))
    gs = gridspec.GridSpec(1, 1)

    ##Legend
    pop_a = mpatches.Patch(color='red', label=f'Median {year}')
    pop_c = mpatches.Patch(color='limegreen', label='Median climatology')
    pop_d = mpatches.Patch(color='lightblue', label='IQR climatology')
    pop_b = mpatches.Patch(color='navy', label=f'IQR {year}')

    ax0 = fig.add_subplot(gs[0])
    ax0.legend(handles=[pop_a, pop_b, pop_c, pop_d], loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
    ax0.set_ylabel("Precipitation (mm)", fontsize=14)
    ax0.set_xlabel(f"{year}", fontsize=16)

    line0 = ax0.boxplot(df_list_1, showfliers=False, labels=list_dates_1, patch_artist=True, showcaps=False)
    line2 = ax0.boxplot(df_list_all_1, showfliers=False, labels=list_dates_1, patch_artist=True, showcaps=False, manage_ticks=False)

    n=30
    for ax in [ax0]:
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='grey', linestyle='dashed')
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines()) if i % n != 0]
        ax.tick_params(labelrotation=45, tick1On=False)

    for median in line0['medians']:
        median.set_color('red')
    for box in line0["boxes"]:
        box.set_color("navy")
        box.set_alpha(0.8)
    for whisk in line0["whiskers"]:
        whisk.set_color("white")
    for median in line2['medians']:
        median.set_color('limegreen')
    for box in line2["boxes"]:
        box.set_color("lightblue")
        box.set_alpha(0.6)
    for whisk in line2["whiskers"]:
        whisk.set_color("white")

    plt.ylim(0, 10)
    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    # plt.suptitle("Daily Precipitation Boxplot", fontsize=18)

    plt.subplots_adjust(top=0.95)
    if path is not None:
        plt.savefig(path)
        
    plt.show(block=False)
    time.sleep(5)
    plt.close()

def plot_veg_event(ds:xr.Dataset, year:int, months:list, df_list_all:list=None, path:str=None):
    if df_list_all is None:
        df_list_all, list_dates_all = get_subplot_year(ds)

    df_list_1, list_dates_1 = get_xarray_time_subset(ds=ds, year=year,month=months, variable="ndvi")
    df_list_all_1 = subsetting_whole(df_list_all =df_list_all, year = year, months=months)

    i = max([np.percentile(l, 75) for l in df_list_1])
    j = max([np.percentile(l, 75) for l in df_list_all_1])
    max_ndvi = max(i, j)

    i = min([np.percentile(l, 25) for l in df_list_1])
    j = min([np.percentile(l, 25) for l in df_list_all_1])
    min_ndvi = min(i, j)

    fig = plt.figure(figsize=(8,4))
    # set height ratios for subplots
    gs = gridspec.GridSpec(1, 1) 

    ##Legend

    pop_a = mpatches.Patch(color='red', label=f'Median {year}')
    pop_b = mpatches.Patch(color='darkgreen', label='Median climatology')
    pop_c = mpatches.Patch(color='lightblue', label='IQR climatology')
    pop_d = mpatches.Patch(color='lightgrey', label=f'IQR {year}')


    # the first subplot
    ax0 = fig.add_subplot(gs[0])
    #ax0.set_title("NDVI for 2009")
    ax0.legend(handles=[pop_a, pop_b, pop_d, pop_c], loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
    ax0.set_ylabel("NDVI value", fontsize=14)
    ax0.set_xlabel(f"{year}", fontsize=16)

    # log scale for axis Y of the first subplot
    line0 = ax0.boxplot(df_list_1, showfliers=False, labels=list_dates_1, patch_artist=True,showcaps=False)
    line2 = ax0.boxplot(df_list_all_1, showfliers=False, labels=list_dates_1, patch_artist=True,showcaps=False, manage_ticks=False)

    n=30
    for ax in [ax0]:
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='grey', linestyle='dashed')
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_gridlines()) if i % n != 0]
        ax.tick_params(labelrotation=45, tick1On=False)

    for median in line0['medians']:
        median.set_color('red')
    for box in line0["boxes"]:
        box.set_color("lightgrey")
        box.set_alpha(0.8)
    for whisk in line0["whiskers"]:
        whisk.set_color("white")
    for median in line2['medians']:
        median.set_color('darkgreen')
    for box in line2["boxes"]:
        box.set_color("lightblue")
        box.set_alpha(0.4)
    for whisk in line2["whiskers"]:
        whisk.set_color("white")

    plt.ylim(min_ndvi-0.05, max_ndvi+0.05)
    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    # plt.suptitle("Daily NDVI boxplot", fontsize=18)
    fig.legend()

    plt.subplots_adjust(top=0.95)
    if path!=None:
        plt.savefig(path)
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



def plot_whole_period_spi_veg(spi_ds:Union[xr.DataArray, xr.Dataset], 
                              vci:Union[xr.DataArray, xr.Dataset], 
                              late:int, var_target:str, 
                              start_date:str="2006-01-01",
                              end_date:str="2019-12-31"):
    import matplotlib.dates as mdates


    spi_med = spi_ds.sel(time=slice(start_date, end_date))[var_target].median(["lat","lon"])
    veg_med = vci.sel(time=slice(start_date, end_date))["ndvi"].mean(["lat","lon"])
    
    time_vals = spi_med.time.values
    time_vals = [pd.to_datetime(x).strftime("%Y-%m-%d") for x in time_vals]
    dates = [pd.to_datetime(i) for i in time_vals]
    
    # Convert DataArrays to pandas DataFrames
    spi_df = spi_med.to_dataframe(name='spi')
    ndvi_df = veg_med.to_dataframe(name='ndvi')
    
    # Calculate spi_color based on conditions
    spi_color = np.where(spi_med > 0, 'blue', 'red')
    
    # Convert DataArrays to pandas DataFrames
    spi_df = pd.DataFrame({'spi': spi_df["spi"], 'spi_color': spi_color}, index=dates)
    ndvi_df = pd.DataFrame({'ndvi': ndvi_df["ndvi"]}, index=dates)
    
    # Set up the figure and axis
    fig, ax1 = plt.subplots(figsize=(40, 6))
    ax2 = ax1.twinx()
    
    # Plot SPI as bar chart with colored bars
    bar_width = 1
    dates_shifted = spi_df.index.to_series().apply(lambda x: x - pd.DateOffset(days=bar_width/2))
    ax1.bar(dates_shifted, spi_df['spi'], width=bar_width, color=spi_df['spi_color'], alpha=0.4)
    
    # Plot NDVI as a line chart
    ndvi_df['ndvi'].plot.line(ax=ax2, color='green')
    
    # Set labels and titles
    ax1.set_ylabel(f'SPI {late}', fontsize=22)
    ax2.set_ylabel('NDVI', fontsize=22)
    ax1.set_xlabel('Dates', fontsize=22, labelpad=20)
    ax1.yaxis.set_tick_params(labelsize=22)
    ax2.yaxis.set_tick_params(labelsize=22)

        # Set y-axis ticks with steps of 0.5

    # Set x-axis label with more padding
    #ax1.set_title(f'SPI {late} and VCI Comparison')
    
    # Set y-axis limits for SPI and NDVI
    ndvi_min =  ndvi_df["ndvi"].min() - 0.05
    ndvi_max =  ndvi_df["ndvi"].max()
    spi_min = np.floor(spi_df["spi"].min() * 2) / 2
    spi_max = np.ceil(spi_df["spi"].max() * 2) / 2
    ax1.set_ylim([spi_min, spi_max])
    ax1.set_yticks(np.arange(spi_min, spi_max + 0.5, 0.5))
    ax2.set_ylim([ndvi_min, ndvi_max])
    
    # Set y-axis tick labels for SPI (blue = >0, red = <=0)
    #ax1.set_yticklabels(np.where(ax1.get_yticks() > 0, ax1.get_yticks(), '0'))
    
    # Adjust x-axis ticks to display one value per year (every 12 months)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    #ax1.xaxis.set_major_locator(mdates.MonthLocator())
    
    # Format x-axis date labels
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.tick_params(labelsize=22, axis="x",which="both")
    
    # Rotate x-axis tick labels for better readability
    plt.tick_params(rotation=45)

    # Add gray shaded areas for March-May and October-December for every year
    year_range = pd.date_range(start=start_date, end=end_date, freq='Y')
    for year in year_range:
        march_may = pd.date_range(start=year.replace(month=3, day=1), end=year.replace(month=5, day=31), freq='D')
        jun_sep = pd.date_range(start=year.replace(month=6, day=1), end=year.replace(month=9, day=30), freq='D')
        oct_dec = pd.date_range(start=year.replace(month=10, day=1), end=year.replace(month=12, day=31), freq='D')
        ax1.fill_between(march_may,  spi_min,  spi_max, facecolor='lightgray', alpha=0.3)
        ax1.fill_between(jun_sep,  spi_min,  spi_max, facecolor='silver', alpha=0.3)
        ax1.fill_between(oct_dec,  spi_min,  spi_max, facecolor='darkgrey', alpha=0.3)

    
    plt.savefig(f"/media/BIFROST/N2/Riccardo/Indices_analysis/data/images/spi_{late}_vci.png")
    # Show the plot
    plt.show(block=False)
    time.sleep(5)
    plt.close()




    
    
def loop_soil(ndvi_ds:xr.DataArray, 
              spi_ds:xr.DataArray=None,
              precp_ds:xr.DataArray=None, 
              ndvi_var:str=None, 
              level1=True,
              one_forest:bool=False,
              path=None): 
    
    from utils.function_clns import config, prepare
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from analysis.visualizations.viz_vci_spi import plot_precp_event, plot_spi_event, plot_veg_multiple_years
    from ancillary.esa_landuse import get_level_colors, create_copernicus_covermap

    # ndvi_res = prepare(ndvi_ds)
    if path is None:
        path = config["DEFAULT"]["images"]
        img_path = os.path.join(path, 'chirps_esa')
        if not os.path.exists(img_path):
            os.makedirs(img_path)
    else:
        img_path = path
    ds_cover = create_copernicus_covermap(ndvi_ds, export=False, level1=level1, one_forest=one_forest)
    ndvi_ds = ndvi_ds.to_dataset(name="ndvi").assign(Band1 = ds_cover)
    cmap, levels, values_land_cover = get_level_colors(ds_cover, level1=level1, one_forest=one_forest)
    # ds_cover.plot(colors=cmap, levels=levels)

    def clean_multi_nulls(ds):
        # Create a MultiIndex
        ds = ds.stack(pixel=("lat", "lon"))
        # Drop the pixels that only have NA values.
        ds = ds.dropna("pixel", how="all")
        ds = ds.unstack(["pixel"]).sortby(["lat","lon"])
        return ds

    to_discard = ['Snow and ice', 'Permanent water bodies','Moss and lichen', "Oceans, seas", 'Unknown'] 
    soil_types =[f for f in levels if f not in to_discard] #np.unique(ds_cover.values)[:-1]
    logger.info(soil_types)
    months = [i for i in np.arange(9, 13)]

    if precp_ds is not None:
        df_list_all_precp, list_dates_all_precp = get_subplot_year(
            precp_ds.to_dataset(name="precp_var"), 
            var="precp_var")
        
    if spi_ds is not None:
        df_list_all_spi, list_dates_all_spi = get_subplot_year(
            spi_ds.to_dataset(name="spi_var"), 
            var="spi_var")
        
    if ndvi_var is not None:
        df_list_all_ndvi, list_dates_all_ndvi = get_subplot_year(
            ndvi_ds, 
            var="ndvi")    

    for soil_type in soil_types:
        soil_name = values_land_cover[soil_type].replace(" ","_").replace("/","_")
        print(f"Starting analysis for {soil_name}")

        ### Raw precipitation
        if precp_ds is not None:
            precp_ds = prepare(precp_ds)
            ds_cover_precp = create_copernicus_covermap(precp_ds, export=False, level1=level1, one_forest=one_forest).to_dataset(name="Band1") 
            ds_cover_precp = ds_cover_precp.assign(precp_var= precp_ds)
            ds_soil = ds_cover_precp["precp_var"].where(ds_cover_precp["Band1"]==soil_type).to_dataset()
            ds_soil = clean_multi_nulls(ds_soil)
            path = os.path.join(img_path,"precp" + "_" + soil_name)
            if ds_soil.isnull().all() == False:
                plot_precp_event(ds_soil, year=2009, var="precp_var", months=months, df_list_all=df_list_all_precp, path=path)
            # plot_precp_2009_event(ds_soil,variable="precp_var", path=path)
        
        if spi_ds is not None:
        ### SPI
            spi_ds = prepare(spi_ds)#.transpose("time","lat","lon")
            ds_cover_spi = create_copernicus_covermap(spi_ds, export=False, level1=level1, one_forest=one_forest).to_dataset(name="Band1")
            ds_cover_spi = ds_cover_spi.assign(spi_var= spi_ds)
            ds_soil = ds_cover_spi["spi_var"].where(ds_cover_spi["Band1"]==soil_type).to_dataset()
            ds_soil = clean_multi_nulls(ds_soil)    
            path = os.path.join(img_path,"spi" + "_" + soil_name)
            if ds_soil.isnull().all() == False:
                plot_spi_event(ds_soil, variable="spi_var", year=2009, months=months, df_list_all=df_list_all_spi, path=path)
    
        if ndvi_var is not None:
            ### NDVI
            ds_soil = ndvi_ds[ndvi_var].where(ndvi_ds["Band1"]==soil_type).to_dataset()
            ds_soil = clean_multi_nulls(ds_soil)
            path = os.path.join(img_path,"ndvi" + "_" + soil_name)
            if ds_soil.isnull().all() == False:
                plot_veg_multiple_years(ds_soil, years=[2009, 2010], path=path)
                # plot_veg_event(ds_soil, year=2009, months=months, df_list_all=df_list_all_ndvi, path=path)
                # logger.info(f"Saved file in path {path}")
