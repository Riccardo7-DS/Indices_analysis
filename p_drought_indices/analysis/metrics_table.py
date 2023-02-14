import xarray as xr
from p_drought_indices.precipitation.regridding_products import transpose_dataset, date_compat_check
from p_drought_indices.functions.function_clns import subsetting_pipeline, load_config, cut_file
import pandas as pd
import re
import geopandas as gpd
import matplotlib.pyplot as plt
import xskillscore as xs
from p_drought_indices.functions.function_clns import load_config, cut_file
from p_drought_indices.functions.ndvi_functions import downsample, clean_ndvi, compute_ndvi
from p_drought_indices.vegetation.cloudmask_cleaning import extract_apply_cloudmask, plot_cloud_correction, compute_difference, compute_correlation
import xarray as xr 
import pandas as pd
import yaml
from datetime import datetime, timedelta
import shutil
from shapely.geometry import Polygon, mapping
import geopandas as gpd
import matplotlib.pyplot as plt
from glob import glob
import os
#import datetime as datetime
import time
import numpy as np
import re
from p_drought_indices.vegetation.NDVI_indices import compute_svi, compute_vci
from p_drought_indices.ancillary_vars.esa_landuse import get_level_colors, get_description
from rasterio.enums import Resampling



CONFIG_PATH = r"../config.yaml"

config = load_config(CONFIG_PATH)

class MetricTable(object):
    def __init__(self, obs_directory, fcst_directory, var_obs:str, var_fcst:str, CONFIG_PATH, vci_tresh = 10, spi_tresh=-2, countries = None):
        """
        Script to calculate metrics for drought prediction for observed and forecasted variable
        obs_directory: path of observed variable
        fcst_directory: path of forecasted variable
        var_obs: name of observed variable
        var_fcst: name of forecasted variable
        CONFIG_PATH: path for config file

        """
        config = load_config(CONFIG_PATH)
        if countries == None:
            self.countries = config['AREA']['3countr']
        else:
            self.countries = countries

        self.chunks = {'time':'600MB'}
        self.spi_tresh = spi_tresh
        self.vci_tresh = vci_tresh
        file_obs  = [f for f in os.listdir(obs_directory) if f.endswith('.nc') and var_obs in f]
        self.path_obs = os.path.join(obs_directory,file_obs[0])
        self.name_obs = var_obs

        file_fcst  = [f for f in os.listdir(fcst_directory) if f.endswith('.nc') and var_fcst in f]
        self.path_fcst = os.path.join(fcst_directory,file_fcst[0])
        self.name_fcst =  os.path.basename(os.path.normpath(self.path_fcst))
        self.spi_freq = int(re.search(r'\d+',self.name_fcst).group())
        self.abbrev_fcst = re.search('(.*)(spi_gamma_\d+)(.nc)', self.name_fcst).group(2)
        self.product_fcst = re.search('(.*)(_spi_gamma_\d+)(.nc)', self.name_fcst).group(1)
        ### resample datasets to same spatial and temporal resolution
        self._load_process_datasets()    
        ### add landcover and df for cover data
        self._get_land_cover()

    def _compute_metrics(self, freq, dataset=None):
        if dataset is None:
            ####compute metrics over lat-lon for default dataset
            accuracy, far, pod, fb, rmse, mse = self._get_eval_metrics()
        else:
            accuracy, far, pod, fb, rmse, mse = self._get_eval_metrics(dataset)
        
        if freq=="daily":
            metrics_df = pd.DataFrame([accuracy.values, far.values, pod.values, rmse.values, \
            mse.values, dataset['time'].values]).T

            metrics_df['product'] = self.product_fcst 
            metrics_df['veg_idx'] = self.name_obs
            metrics_df['precp_idx'] = self.abbrev_fcst
            metrics_df.columns=['accuracy', 'far','pod', 'rmse', 'mse','time', 'product', 'veg_idx','precp_idx' ]
            return metrics_df
        elif freq=="all":
            metrics_df = pd.DataFrame([float(accuracy.mean(['time']).values), float(far.mean(['time']).values), float(pod.mean(['time']).values), float(rmse.mean(['time'])), \
                float(mse.mean(['time']).values), self.product_fcst, self.name_obs,  self.abbrev_fcst]).T
            metrics_df.columns=['accuracy_m', 'far_m','pod_m', 'rmse_m', 'mse_m', 'product', 'veg_idx','precp_idx']
            return metrics_df
        else:
            raise NotImplementedError

    def compute_metrics_soil(self, freq="daily"):
        assert freq in ['daily','all']
        self.df_cover = self._metrics_soil(freq=freq)

    def _metrics_soil(self, freq):
        metric_all = pd.DataFrame()
        for cat in self.land_categories:
            sub_data = self.dataset.where(self.dataset['land']==cat)
            metrics_df = self._compute_metrics(dataset = sub_data, freq=freq)
            metrics_df['land_cat'] = cat
            metrics_df['country'] = ' '.join(self.countries)
            metrics_df = get_description(metrics_df, 'land_cat')
            metric_all = pd.concat([metric_all, metrics_df], ignore_index=True)
        return metric_all

    def add_cover(self):
        self._get_land_cover()

    def _load_process_datasets(self):
        ##### load xr for forecasted variable and change dimensions name due to error in preprocessing
        xds_fcst = xr.open_dataset(self.path_fcst)#.rename({'lat':'x','lon':'y'}).rename({'x':'lon','y':'lat'})
        xr_fcst_proc = subsetting_pipeline(CONFIG_PATH, xds_fcst[self.abbrev_fcst], self.countries)
        xr_fcst_final = self._prep_dataset(xr_fcst_proc,  transpose=True)

        ##### generate a datarray to be subsetted and used for reprojection
        ds_obs =  xr.open_dataarray(self.path_obs)
        ds_obs = subsetting_pipeline(CONFIG_PATH, ds_obs, self.countries)
        ds_final = self._prep_dataset(ds_obs, transpose=True)

        #### subset the dataset and the datarray
        target_new = date_compat_check(target_xr =xr_fcst_final , xr_df= ds_final)
        res_xr = date_compat_check(target_xr =xds_fcst, xr_df = ds_final)

        #### reproject it and get variables
        ds = self._reproject_get_vars(res_xr, ds_final, target_new)
        self.dataset = transpose_dataset(ds,'time', 'lat','lon')

    def _prep_dataset(self, ds, transpose=False):
        if transpose==True:
            ds = transpose_dataset(ds,'time', 'lon', 'lat')
        ds.rio.write_crs("epsg:4326", inplace=True)
        ds.rio.set_spatial_dims(x_dim='lat', y_dim='lon', inplace=True)
        return ds
        
    def _reproject_get_vars(self, res_xr, ds_final, target_new):
        ### Reproject datarray
        target_other = self._prep_dataset(target_new, transpose=True)
        xds_repr_match = ds_final.rio.reproject_match(target_other,  resampling=Resampling.bilinear)
        xr_regrid = xds_repr_match.rename({'x':'lat','y':'lon'})
        null_var = xr.where(target_new.notnull(), 1,np.NaN)
        condition_var = xr.where(target_new<=-self.spi_tresh,1,0)
        res_vars = condition_var.where(null_var==1)
        ### assign calculated variables to xarray dataset and subset vars
        res_xr = res_xr.assign(drt_precp = res_vars,\
                drt_veg = xr.where(xr_regrid <=self.vci_tresh, 1, 0), target_new = target_new)
        res_xr['drt_veg'] = res_xr['drt_veg'].where(res_xr['target_new'].notnull())
        #res_xr['drt_precp'] = subsetting_pipeline(CONFIG_PATH, res_xr['drt_precp'], countries=self.countries)
        #res_xr['drt_veg'] = subsetting_pipeline(CONFIG_PATH, res_xr['drt_veg'], countries=self.countries)
        return res_xr

    def _get_eval_metrics(self, dataset= None, drt_obs:str='drt_veg', drt_forec:str = 'drt_precp', dim:list=["lat","lon"]):
        """
        This is a function to calculate the target metrics for an xarray over a desired dimension, by default over lat/lon.
        The standard name for the observation is 'drt_veg' whereas for the forecast is 'drt_precp'
        Returns accuracy, far, pod, bias ration, rmse, mse.
        """
        if dataset is None:
            dataset = self.dataset
        dichotomous_category_edges = np.array([0, 0.5, 1])  # "dichotomous" mean two-category
        dichotomous_contingency = xs.Contingency(
            dataset[drt_obs], dataset[drt_forec], dichotomous_category_edges, dichotomous_category_edges, dim=dim
            )
        accuracy = dichotomous_contingency.accuracy()
        far = dichotomous_contingency.false_alarm_rate()#.mean().values
        pod = dichotomous_contingency.hit_rate()#.mean().values
        fb  = dichotomous_contingency.bias_score()
        rmse = xs.rmse(dataset[drt_obs], dataset[drt_forec],skipna=True, dim=dim)
        mse = xs.mse(dataset[drt_obs], dataset[drt_forec],skipna=True, dim=dim)
        return accuracy, far, pod, fb, rmse, mse

    def plot_indices(self, far, pod, accuracy, rmse, save=True):
        """
        Function to plot and save metrics for far, pod, accuracy and rmse 
        """
        t = self.product_fcst
        b = self.name_obs
        g = self.abbrev_fcst

        pod.plot()
        plt.title('Compute mean POD between {t} and {b} for {g}'.format(t= t, b= b, g=g))
        if save == True:
            file_name = 'pod_{t}_{b}_{g}.png'.format(t= t, b= b, g=g)
            plt.savefig(os.path.join(config['DEFAULT']['image_path'], file_name))
        plt.show()
        far.plot()
        plt.title('Compute mean FAR between {t} and {b} for {g}'.format(t= t, b= b, g=g))
        if save == True:
            file_name = 'far_{t}_{b}_{g}.png'.format(t= t, b= b, g=g)
            plt.savefig(os.path.join(config['DEFAULT']['image_path'], file_name))
        plt.show()
        accuracy.plot()
        plt.title('Compute mean accuracy between {t} and {b} for {g}'.format(t= t, b= b, g=g))
        if save == True:
            file_name = 'acrcy_{t}_{b}_{g}.png'.format(t= t, b= b, g=g)
            plt.savefig(os.path.join(config['DEFAULT']['image_path'], file_name))
        rmse.plot()
        plt.title('Compute RMSE between {t} and {b} for {g}'.format(t= t, b= b, g=g))
        if save == True:
            file_name = 'rmse_{t}_{b}_{g}.png'.format(t= t, b= b, g=g)
            plt.savefig(os.path.join(config['DEFAULT']['image_path'], file_name))
        plt.show()

    def _get_land_cover(self, ds_path= r'../data/images/esa_cover.nc'):
        ds_cover = xr.open_dataset(ds_path)
        ds_cover = subsetting_pipeline(CONFIG_PATH, ds_cover, invert=False)
        #### get categories and levels with colors to plot the land cover dataset
        cmap, levels = get_level_colors(ds_cover)
        self.land_categories = ds_cover.to_dataframe()['Band1'].dropna().astype(int).unique().tolist()
        #### reproject land cover ds in the format of the precipitation dataset
        res = ds_cover['Band1'].rio.reproject_match(self.dataset[self.abbrev_fcst])
        res = res.rename({'x':'lon','y':'lat'})
        #### generate empty time dimension and expand it
        time_dim = self.dataset['time'].values
        time_da = xr.DataArray(time_dim, [('time', time_dim)])
        res = res.expand_dims(time=time_da)
        self.dataset = self.dataset.assign(land = res)