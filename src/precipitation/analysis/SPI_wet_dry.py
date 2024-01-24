from distutils.command.config import config
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import numpy as np
# libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import re
import yaml
from xarray import Dataset
from typing import Mapping,Any

def load_config(CONFIG_PATH):
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)
        return config

def cut_file(xr_df, gdf):
    xr_df.rio.write_crs("epsg:4326", inplace=True)
    clipped = xr_df.rio.clip(gdf.geometry.apply(mapping), gdf.crs, drop=True)
    if 'crs' in clipped.data_vars:
        clipped = clipped.drop('crs')
    return clipped

def downsample(ds):
    monthly = ds.resample(time='5D', skipna=True).mean() #### Change here to change the timeframe over which to make the data imputation
    return monthly

def clean_ndvi(ds):
    ds = ds.where('ndvi'!=0)
    return ds

#### Function to plot by year the p of dry and wet

def plot_p_year(spi_corr,year):
    print('Mean probabilty of being classified as wet in year {}'.format(year))
    sub_corr = spi_corr['wet'].sel(time=slice('{}-01-01'.format(year), '{}-12-31'.format(year)))
    sub_corr.mean(dim='time').plot()
    plt.show()
    print('Mean probabilty of being classified as dry in year {}'.format(year))
    sub_corr = spi_corr['dry'].sel(time=slice('{}-01-01'.format(year), '{}-12-31'.format(year)))
    sub_corr.mean(dim='time').plot(cmap=plt.cm.YlOrRd)
    plt.show()
    
### Function to plot SPI
def plot_index(df: pd.DataFrame, date_col: str, precip_col: str, save_file: str=None,
               index_type: str='SPI', bin_width: int=22):

    pos_index = df.loc[df[precip_col] >= 0]
    neg_index = df.loc[df[precip_col] < 0]

    fig, ax = plt.subplots()
    ax.bar(pos_index[date_col], pos_index[precip_col], width=bin_width, align='center', color='b')
    ax.bar(neg_index[date_col], neg_index[precip_col], width=bin_width, align='center', color='r')
    ax.grid(True)
    ax.set_xlabel("Date")
    ax.set_ylabel(index_type)

    if save_file:
        plt.savefig(save_file, dpi=400)

    return fig

def calculate_points_xr(spi_xr, spi_obj):
   #### check when wet/dry condition is met
   spi_xr= spi_xr.assign(wet = spi_xr[spi_obj.abbrev].where(spi_xr[spi_obj.abbrev]>=spi_obj.fl_lim, 0),\
           dry = spi_xr[spi_obj.abbrev].where(spi_xr[spi_obj.abbrev]<=spi_obj.dr_lim,0))
   spi_xr = spi_xr.assign(wet = spi_xr[spi_obj.abbrev].where(spi_xr.wet==0,1),\
           dry = spi_xr[spi_obj.abbrev].where(spi_xr.dry==0,1))
   spi_xr['dry_cond'] = spi_xr['dry'].rolling(time=spi_obj.freq).sum()
   spi_xr['wet_cond'] = spi_xr['wet'].rolling(time=spi_obj.freq).sum()
   d = spi_xr.dims
   dims = [k for k in sorted(d.keys())]
   sizes = [d[k]for k in sorted(d.keys())]
   spi_xr = spi_xr.assign(dict(DW=(dims, np.ones(sizes))))
   spi_xr = spi_xr.assign(dict(WD=(dims, np.ones(sizes))))
   spi_xr['DW'] = spi_xr['DW'].where((spi_xr.wet_cond>=1) & (spi_xr.dry==1),0)
   spi_xr['WD'] = spi_xr['WD'].where((spi_xr.dry_cond>=1) & (spi_xr.wet==1),0)
   print('Computed new variables for xarray')
   return spi_xr

def plotting_mean_wd(country, gdf, ds):
    subset = gdf[gdf.ADM0_NAME.isin([country])]
    cut_file(ds, subset)['DW'].mean(['lat','lon']).plot()
    plt.title('Drought flood 7 days lag for {}'.format(country))
    plt.show()


def plot_country_heatmap(df, variable, country):
    # Normalize it by row:
    df_norm_row = df.loc[variable][country]#.apply(lambda x: (x-x.mean())/x.std(), axis = 1)
    # And see the result
    sns.heatmap(df_norm_row, cmap='viridis')
    plt.show()

class spiObject(object):
    def __init__(self, path_spi, dr_lim:float=-2, fl_lim:int=2):
        self.path = path_spi
        self.name = os.path.basename(os.path.normpath(self.path))
        self.freq = int(re.search(r'\d+',self.name).group())
        self.abbrev = re.search('(.*)(spi_gamma_\d+)(.nc)', self.name).group(2)
        self.dr_lim = dr_lim
        self.fl_lim = fl_lim
        self.product = re.search('(.*)(spi_gamma_\d+)(.nc)', self.name).group(1)
        self.xr = xr.Dataset()

    def calculate_max_min(self):
        self.xr = self.xr.assign(wet_max = self.xr[self.abbrev].resample(time='QS-Mar').max().resample(time='1D').ffill(),\
                dry_min = self.xr[self.abbrev].resample(time='QS-Mar').min().resample(time='1D').ffill())
        self.xr = self.xr.assign(diff_max = self.xr.wet_max - self.xr[self.abbrev],\
                diff_min = self.xr[self.abbrev] - self.xr.dry_min)
        return self.xr

    def calculate_points_xr(self):   #### check when wet/dry condition is met
        if self.xr.time.size == 0:
            raise AttributeError("Empty Dataset. You need to initialize an xarray Dataset before computing the points")
        self.xr= self.xr.assign(wet = self.xr[self.abbrev].where(self.xr[self.abbrev]>=self.fl_lim, 0),\
                dry = self.xr[self.abbrev].where(self.xr[self.abbrev]<=self.dr_lim,0))
        self.xr = self.xr.assign(wet = self.xr['wet'].where(self.xr.wet==0,1),\
                dry = self.xr['dry'].where(self.xr.dry==0,1))
        self.xr['dry_cond'] = self.xr['dry'].rolling(time=self.freq).sum()
        self.xr['wet_cond'] = self.xr['wet'].rolling(time=self.freq).sum()
        d = self.xr.dims
        dims = [k for k in sorted(d.keys())]
        sizes = [d[k]for k in sorted(d.keys())]
        self.xr = self.xr.assign(dict(DW=(dims, np.ones(sizes))))
        self.xr = self.xr.assign(dict(WD=(dims, np.ones(sizes))))
        self.xr['DW'] = self.xr['DW'].where((self.xr.wet_cond>=1) & (self.xr.dry==1),0)
        self.xr['WD'] = self.xr['WD'].where((self.xr.dry_cond>=1) & (self.xr.wet==1),0)
        print('Computed new variables for xarray')
        return self.xr

    def run_theory(self):
        time_thresh = self.xr['time'].isel(time=self.freq).values
        threshold = -1
        ds = self.xr.where(self.xr.time > time_thresh, drop=True)
        ds = ds.load()

        #### create masks for null and for drought(0)-non drought (1)
        null_mask = ds.where(ds[self.abbrev].notnull(), 0)
        masked = xr.where(ds[self.abbrev]>= threshold, 1, 0)
        ds = ds.assign(masked = masked)
        new_xr = ds.assign(index = ds['masked'].cumsum(['time']))

        df= new_xr.where(new_xr.masked==0).to_dataframe().dropna(subset=['index',self.abbrev]).drop(columns={'spatial_ref'}) #.where(null_mask[self.abbrev]!=0)
        duration_df = df.groupby(["lat", "lon",'index']).count().drop(columns={self.abbrev}).rename(columns={'masked':'duration'})
        severity_df = df.groupby(["lat", "lon",'index']).sum().drop(columns={'masked'}).rename(columns={self.abbrev:'severity'})
        severity_df['severity'] = abs(severity_df['severity'])
        res_df = severity_df.merge(duration_df, left_index=True, right_index=True)
        res_df['intensity']= res_df['severity']/res_df['duration']

        events_df = new_xr.to_dataframe().reset_index().merge(res_df.reset_index(), on=['lat','lon','index'], how='left')\
            .drop(columns={'spatial_ref','masked',self.abbrev}).dropna(subset=['severity','duration','intensity'])\
                .drop_duplicates(subset=['lat','lon','index','severity','duration','intensity'], keep='first').drop(columns={'index'})

        events_df['year'] = events_df['time'].dt.year
        events_df['month'] = events_df['time'].dt.month
        events_df['product'] = self.product[:-1]
        events_df['latency'] = self.abbrev

        return events_df

def compute_dry_wet(path_spi, countries,  year_start, year_end, aggr='season'):
    if aggr not in ['season','month','day','year']:
        raise ValueError("The specified aggregation time is not correct")
    spi = spiObject(path_spi)
    shapefile_path = config['SHAPE']['africa']
    aggregation = 'time.{}'.format(aggr)

    #### Chose subset of countries
    gdf = gpd.read_file(shapefile_path)
    subset = gdf[gdf.ADM0_NAME.isin(countries)]

    ###open file and cut it
    spi.xr = cut_file(xr.open_dataset(path_spi, chunks={"lat": -1, "lon": -1, "time": 12}), subset)
    
    res_xr = cut_file(spi.calculate_points_xr(), subset)
    print('The final product has the following variables:', res_xr.data_vars)
    ### create new xarray with calculated points and cut it
    #res_xr = cut_file(calculate_points_xr(spi_xr, spi), subset)

    
    ############################# compute monthly weights ###########################
    #month_length = res_xr.time.dt.days_in_month
    #weights = (
    #    month_length.groupby(aggregation) / month_length.groupby(aggregation).sum()
    #)

    ## Test that the sum of the weights for each season/month is 1.0
    #np.testing.assert_allclose(weights.groupby(aggregation).sum().values, np.ones(12))

    ## Calculate the weighted average
    #ds_weighted = (res_xr * weights).groupby(aggregation).sum(dim="time")

    ## only used for comparisons
    #ds_unweighted = res_xr.groupby(aggregation).mean("time")
    #ds_diff = ds_weighted - ds_unweighted

    ############################# compute results dataframe ###########################
    print('Starting computing new dataframe')
    new_df = pd.DataFrame()

    for year in range(year_start,year_end):
        season_xr = res_xr.sel(time=slice('{}-01-01'.format(year), '{}-12-31'.format(year)))\
        .groupby(aggregation).mean(dim="time")

        for country in countries:
            # get the start time
            st = time.time()
            for i, season in enumerate(("DJF", "MAM", "JJA", "SON")):
                subset = gdf[gdf.ADM0_NAME.isin([country])]
                tempo_xr = cut_file(season_xr, subset)
                mean_dw = tempo_xr["DW"].sel(season=season).mean(['lon','lat']).values
                mean_wd = tempo_xr["WD"].sel(season=season).mean(['lon','lat']).values
                temp_df = pd.DataFrame([mean_dw, mean_wd, country, season, year]).transpose()
                new_df = new_df.append(temp_df,ignore_index=True)

            # get the end time
            et = time.time()
            # get the execution time
            elapsed_time = et - st
            print('Execution time by season for country {c} in year {y} :'.format(y=year, c= country), elapsed_time, 'seconds')

    new_df.columns = ['DW', 'WD', 'Country','Season','Year']
    #final_df['DW'] = round(final_df['DW'].astype(float),4)
    #final_df['WD'] = round(final_df['WD'].astype(float),4)
    new_df.to_csv(r'./data/season_wd_gha_{ys}-{ye}_{s}_{pr}.csv'.format(s=spi.abbrev, ys=year_start, ye=year_end, pr=spi.product))

    #stacked_df = pd.pivot_table(new_df, columns=['Country', 'Season','Year']).stack()

if __name__== "__main__":

    CONFIG_PATH = r"./config.yaml"
    config = load_config(CONFIG_PATH)
    
    product_directories = [config['SPI']['IMERG']['path']]
    
    #for product_dir in product_directories:
    #    files  = [f for f in os.listdir(product_dir) if f.endswith('.nc')]
    #    xr_example = xr.open_dataset(os.path.join(product_dir,files[0]))
    #    year_start = int(xr_example['time'].dt.year.min())
    #    year_end= int(xr_example['time'].dt.year.max()) +1
    #    print('The prdouct {p} ranges from {ys} to {ye}'.format(ye=year_end, ys=year_start, p=os.path.basename(os.path.normpath(files[0]))))
    #    
    #    #### Computing new dataframe for the selected SPI product
    #    for file_path in files:
    #        countries = ['Ethiopia','Somalia','Kenya']
    #        path_spi = os.path.join(product_dir, file_path)
    #        compute_dry_wet(path_spi, countries,  year_start, year_end, aggr='season')

    for product_dir in product_directories:
        files  = [f for f in os.listdir(product_dir) if f.endswith('.nc')]
        year_start = 2004
        year_end= 2015

        shapefile_path = config['SHAPE']['africa']
        countries = ['Ethiopia','Somalia','Kenya']
        #### Chose subset of countries
        gdf = gpd.read_file(shapefile_path)
        for file_path in files:
            spi = spiObject(file_path)
            subset = gdf[gdf.ADM0_NAME.isin(countries)]
            ds = xr.open_dataset(os.path.join(product_dir,file_path), chunks={"lat": -1, "lon": -1, "time": 12}).sel(
                time=slice('{}-01-01'.format(year_start) , '{}-12-31'.format(year_end)))
            ###open file and cut it
            spi.xr = cut_file(ds, subset)
            events_df = spi.run_theory().reset_index(drop=True)
            print('Finished computing run theory for product {p} at {f} scale'.format(p=spi.product, f=spi.abbrev))
            events_df.to_csv(r'data\events\run_theory_hoa_{ys}_{ye}_{p}_{f}.csv'.format(ye=year_end, ys=year_start, p=spi.product, f=spi.abbrev))
        
