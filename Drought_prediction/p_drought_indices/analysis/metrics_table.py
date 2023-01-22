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

        self.dataset = self._load_process_datasets()
        ####compute metrics over lat-lon
        accuracy_t, far_t, pod_t, fb_t, rmse_t, mse_t = self._get_eval_metrics()
        ##### compute metrics over time
        accuracy_s, far_s, pod_s, fb_s, rmse_s, mse_s = self._get_eval_metrics(dim=['time'])
        #self.plot_indices(far_s, pod_s, accuracy_s, rmse_s, save=True)
#
        ##### gather the statistics
        metrics_df = pd.DataFrame([float(accuracy_t.mean(['time']).values), float(far_t.mean(['time']).values), float(pod_t.mean(['time']).values), float(rmse_t.mean(['time'])), \
            float(mse_t.mean(['time']).values), self.product_fcst, self.name_obs,  self.abbrev_fcst]).T
        metrics_df.columns=['accuracy_m', 'far_m','pod_m', 'rmse_m', 'mse_m', 'product', 'veg_idx','precp_idx']
        self.table_res = metrics_df

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
        target_new = date_compat_check(ds_final, xr_fcst_final)
        res_xr = date_compat_check(ds_final, xds_fcst)

        #### reproject it and get variables
        ds = self._reproject_get_vars(res_xr, ds_final, target_new)
        return transpose_dataset(ds,'time', 'lat','lon')

    def _prep_dataset(self, ds, transpose=False):
        if transpose==True:
            ds = transpose_dataset(ds,'time', 'lon', 'lat')
        ds.rio.write_crs("epsg:4326", inplace=True)
        ds.rio.set_spatial_dims(x_dim='lat', y_dim='lon', inplace=True)
        return ds
        
    def _reproject_get_vars(self, res_xr, ds_final, target_new):
        ### Reproject datarray
        target_new = self._prep_dataset(target_new, transpose=True)
        xds_repr_match = ds_final.rio.reproject_match(target_new)
        xr_regrid = xds_repr_match.rename({'x':'lat','y':'lon'})
        masked_data = xr_regrid.where(target_new.notnull())
        null_var = xr.where(target_new.notnull(), 1,np.NaN)
        condition_var = xr.where(target_new<=-2,1,0)
        res_vars = condition_var.where(null_var==1)
        ### assign calculated variables to xarray dataset and subset vars
        res_xr = res_xr.assign(drt_precp = res_vars,\
                drt_veg = xr.where(masked_data <=self.vci_tresh, 1, 0), target_new = target_new)
        #res_xr['drt_precp'] = subsetting_pipeline(CONFIG_PATH, res_xr['drt_precp'], countries=self.countries)
        res_xr['drt_veg'] = subsetting_pipeline(CONFIG_PATH, res_xr['drt_veg'], countries=self.countries)
        return res_xr

    def _get_eval_metrics(self, drt_obs:str='drt_veg', drt_forec:str = 'drt_precp', dim:list=["lat","lon"]):
        """
        This is a function to calculate the target metrics for an xarray over a desired dimension, by default over lat/lon.
        The standard name for the observation is 'drt_veg' whereas for the forecast is 'drt_precp'
        Returns accuracy, far, pod, bias ration, rmse, mse.
        """
        dichotomous_category_edges = np.array([0, 0.5, 1])  # "dichotomous" mean two-category
        dichotomous_contingency = xs.Contingency(
            self.dataset[drt_obs], self.dataset[drt_forec], dichotomous_category_edges, dichotomous_category_edges, dim=dim
            )
        accuracy = dichotomous_contingency.accuracy()
        far = dichotomous_contingency.false_alarm_rate()#.mean().values
        pod = dichotomous_contingency.hit_rate()#.mean().values
        fb  = dichotomous_contingency.bias_score()
        rmse = xs.rmse(self.dataset[drt_obs], self.dataset[drt_forec],skipna=True, dim=dim)
        mse = xs.mse(self.dataset[drt_obs], self.dataset[drt_forec],skipna=True, dim=dim)
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
