import xarray as xr
from merge_daily_data import load_config
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import re
CONFIG_PATH = r"./config.yaml"


def inspect_time(ds):
    ##### COmpute the max time
    time_max = ds["time"].where(ds==ds.max("time")).max("time")
    print('The most recent time date is:', time_max['spatial_ref'].values)

    ##### COmpute the min time
    time_min = ds["time"].where(ds==ds.min("time")).min("time")
    print('The oldest time date is:', time_min['spatial_ref'].values)

def get_avg_nan(path):
    #check_time(path, var)
    string_path = path.split('\\')[-1]
    var = re.search('(.*)(spi_gamma_\d+)(.nc)', string_path).group(2)
    ds = xr.open_dataset(path)

    ### compute notnull values
    vars_nnull = ds[var].count(['lat','lon'])
    num_points = int(vars_nnull.max())
    mean_points = int(vars_nnull.mean())
    return num_points, round(mean_points/num_points,2), mean_points*num_points

def get_nans(path):
    #check_time(path, var)
    string_path = path.split('\\')[-1]
    var = re.search('(.*)(spi_gamma_\d+)(.nc)', string_path).group(2)
    product_name = re.search('(.*)(spi_gamma_\d+)(.nc)', string_path).group(1)
    ds = xr.open_dataset(path)

    ### compute notnull values
    vars_nnull = ds[var].count(['lat','lon']).values
    min_time = ds['time'].min().values
    max_time = ds['time'].max().values
    num_points = int(vars_nnull.max())
    #min_time = pd.to_datetime(ds.attrs['time_coverage_start'])
    #max_time = pd.to_datetime(ds.attrs['time_coverage_end'])
    vars_null = 1 - vars_nnull/num_points
    temp_time = pd.date_range(min_time, max_time, freq='D')
    temp_df = pd.DataFrame(vars_null, index=temp_time)
    temp_df.columns= ['NaNs']
    temp_df.plot()
    plt.title('Variable {car} for product {p}'.format(p=product_name[:-1], car=var))
    plt.show()

def check_time(path, var):
    ds = xr.open_dataset(os.path.join(path,'GPCC_{}.nc'.format(var)))
    # create basic example dataset
    min_time = ds.attrs['time_coverage_start']
    max_time = ds.attrs['time_coverage_end']
    diff = pd.to_datetime(max_time) - pd.to_datetime(min_time)

    reduced_ds = ds.groupby('time').mean(['lat','lon'])
    print(reduced_ds)
    #lat_s = abs(ds['lat'][1] - ds['lat'][2]) 
    #lons = np.linspace(-120, -60, 4)
    #lats = np.linspace(25, 55, lat_s)

    da = xr.DataArray(
        np.linspace(0, diff.days, num=diff.days+1),
        coords=[pd.date_range(min_time, max_time, freq="D")],
        dims="time",
    )
    #miss_xr = xr.merge([ds, da], join="outer")
    miss_xr =  da.combine_first(ds) 
    list_missing = list(miss_xr['time'].values)
    pd.Series(list_missing).to_csv(r'./dates.csv')


    # the meat of the answer: making selection of data per year easier
    #groups = da.groupby("time.year")
    # example usage of that
    #for year, group in groups:    # iterate over yearly blocks
    #    group.plot(label=year)
    #    plt.show() 

config = load_config(CONFIG_PATH)

if __name__=="__main__":

    distribution = 'gamma'
    product_directories = [config['SPI']['CHIRPS']['path'], config['SPI']['IMERG']['path'], config['SPI']['ERA5']['path'],
                            config['SPI']['GPCC']['path']]

    df = pd.DataFrame()

    for product_dir in product_directories:
        files  = [f for f in os.listdir(product_dir) if f.endswith('.nc')]
        new_list = [x for x in files if distribution in x]

        for file_path in new_list:

            
            path_spi = os.path.join(product_dir, file_path)

            #### compute variables
            string_path = path_spi.split('\\')[-1]
            var = re.search('(.*)(spi_gamma_\d+)(.nc)', string_path).group(2)
            product_name = re.search('(.*)(spi_gamma_\d+)(.nc)', string_path).group(1)
            tot_points, mean_point, valid_tot = get_avg_nan(path_spi)
            temp_df = pd.DataFrame([product_name[:-1], var,tot_points, mean_point, valid_tot]).T
            temp_df.columns= ['product','var', 'points','valid', 'valid_points']
            df = df.append(temp_df, ignore_index=True)

    df.to_csv(r'./product_nans.csv')
    print(df)

    #diff = pd.to_datetime(max_time) - pd.to_datetime(min_time)
    #print(diff.days)
    #ds[var].isel(time=30).plot()
    #plt.show()


