#!/usr/bin/env python
# coding: utf-8


import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import numpy as np
import os
import xarray as xr
from utils.xarray_functions import convert_ndvi_tofloat
from collections.abc import Hashable, Mapping, Sequence
from typing import Any
import pandas as pd
import xarray
from typing import Union
import dask
import numpy as np
from modape.whittaker import ws2doptv, ws2d, ws2doptvp
from array import array
from dask.diagnostics import ProgressBar
from typing import Literal

def process_eumetsat_ndvi(path:str, filename:str, complevel:int = 9):
    chunks = {"time":200, "lat":50, "lon":50}
    ds = xr.open_dataset(os.path.join(path, "ndvi_eumetsat.nc"), chunks=chunks)
    ds = ds.rename({"Band1":"ndvi"})

    compression = {"ndvi" :{'zlib': True, "complevel":complevel}}
    from utils.function_clns import config
    ds.to_netcdf(os.path.join(config["NDVI"]["ndvi_path"],filename),
                 encoding=compression)


def load_landsaf_ndvi(path:str, crop_area:bool = True)->xr.Dataset:
    ds = xr.open_zarr(path)
    import pandas as pd
    # Convert nanoseconds to datetime objects
    ds['time'] = pd.to_datetime(ds['time'], unit='ns')
    # Extract the date part
    ds['time'] = ds['time'].dt.floor('D')

    if crop_area is True:
        from utils.function_clns import subsetting_pipeline
        ds = subsetting_pipeline(ds)

    ds["ndvi_10"] = convert_ndvi_tofloat(ds.ndvi_10)
    # ds["ndvi_10"] = ds.ndvi_10/255
    return ds 

def load_modis_cloudmask(file, 
                         plot=Literal["None", "Simple","Basemap"]):
    
    from utils.function_clns import read_hdf
    from utils.xarray_functions import swath_to_grid
    import dask.array as da
    from xarray import DataArray
    from pyresample.bilinear import XArrayBilinearResampler

    DATAFIELD_NAME='Cloud_Mask'
    data, lat,lon = read_hdf(file, DATAFIELD_NAME)
    cloud_mask = np.uint8(data)

    cloud = cloud_mask & 6 # get bits 1 and 2
    cloud[cloud == 0] = 1 # 00 = confident cloudy
    cloud[cloud != 1] = 0

    water = cloud_mask & 192 # get bits 6 and 7
    water[water == 0] = 1 # 00 = water
    water[water != 1] = 0

    coast = cloud_mask & 192 # get bits 6 and 7
    coast[coast == 64] = 1 # 01 = coastal
    coast[coast != 1] = 0
    area_def, swath_def, area_dict, area_extent = swath_to_grid(lat, lon)
    data_xr = DataArray(da.from_array(cloud), dims=('y', 'x'))
    lons_xr = da.from_array(lon)
    lats_xr = da.from_array(lat)

    # clouds = (cloud_mask >> 1) & 0b11
    # cloud_mask = np.where(clouds>1, 1, 0)
    # day = (cloud_mask >> 2) & 1
    # day_mask =  np.where(day==0, 1, 0)

    if plot=="Simple":
        import matplotlib.pyplot as plt
        plt.imshow(result)
        plt.xlabel('x [pixels]')
        plt.ylabel('y [pixels]')
        plt.colorbar()
        plt.title('byte cloud mask')
        plt.show()
    elif plot is "Basemap":
        from utils.xarray_functions import plot_swath_basemap
        plot_swath_basemap(result, area_dict, area_extent)

    resampler = XArrayBilinearResampler(swath_def, area_def, 30e3)
    result = resampler.resample(data_xr)
    return result, lons_xr, lats_xr

def remove_ndvi_outliers(ds:xr.DataArray):
    return ds.where((ds<=1)&(ds>=-1))

"""
Functions to smooth NDVI
"""


def smooth_coordinate(datarray:xr.DataArray, lat:float, lon:float, p:float=0.99,
                      lambda_min:float =-2, lambda_max:float = 3):
    test_year = 2017 ## choose one year for viz purpose

    time_series = datarray.sel(lat=lat, lon=lon, method="nearest")\
        .sel(time=datarray["time"].dt.year==test_year)
    
    ### trasnform weights and series to double
    w = xr.where(time_series.isnull(), 0, 1).values.astype(np.double)
    y = time_series.where(time_series.notnull(),0).astype(np.double).values

    from modape.whittaker import ws2doptv, ws2d, ws2doptvp
    from array import array

    z, sopt = ws2doptvp( y, w, array("d", np.arange(lambda_min, lambda_max, 0.2).round(2)), p=p)
    return z


def apply_datacube(ds: xr.DataArray, window:int, poly_order:int) -> xr.DataArray:
    """ 
    Code to apply Savigol filter along time dimension in datarray
    """
    from scipy.signal import savgol_filter
    #filled = ds.rio.interpolate_na('time')
    smoothed_array = savgol_filter(ds.values, window, poly_order, axis=0)
    lat = ds["lat"]
    lon = ds["lon"]
    time= ds["time"]
    ndvi_fill = xr.DataArray(smoothed_array, dims=("time","lat","lon"),
                      coords={'time': time, 'lat': lat, 'lon': lon})

    ds_fill = ds.to_dataset().assign(filled_ndvi = ndvi_fill)
    return ds_fill


def epct_cropping_pipeline(target_dir:str,
                           shapefile_path:str):

    from utils.function_clns import config
    from epct import api
    from utils.xarray_functions import add_time, compute_radiance

    #fine the configuration of the functional chain to apply:
    chain_config = {"filter": "hrseviri_natural_color",
            "name": "Natural color disc",
            "id": "natural_color_disc",
            'product': 'HRSEVIRI',
            'format': 'netcdf4',
            'projection': 'geographic'
        }

    base_dir = target_dir.copy()

    files = [f for f in os.listdir(base_dir) if f.endswith('.nat')]

    with open(shapefile_path, 'rb') as f:
         shapefile_stream = f.read()

    for file in files:
        #n the chain and return the result as an `xarray` object
        output_xarray_dataset = api.run_chain_to_xarray(
           product_paths=[os.path.join(base_dir, file)],
           chain_config=chain_config,
           target_dir=target_dir,
           shapefile_stream=shapefile_stream
        )
    
    
    
    ##### Compute NDVI and adjust for radiance
    files = [f for f in os.listdir(target_dir) if f.endswith('.nc')]
    for file in files:
        with xr.open_dataset(os.path.join(target_dir, file)) as ds:
            data = ds.load()
            xr_df = add_time(data)
            xr_df = compute_radiance(xr_df)
            xr_df = xr_df.drop('channel_3')
            xr_df = xr_df.assign(ndvi=(
                xr_df['channel_2'] - xr_df['channel_1']) / (xr_df['channel_2'] + xr_df['channel_1']
                        )
                )
            #xr_df['channel_2'].plot()
            xr_df.to_netcdf(os.path.join(base_dir,'processed', file)) 
            xr_df.close()

def pipeline_ndvi(xr_df:Union[xr.DataArray, xr.Dataset], gdf):
    from utils.function_clns import cut_file
    from utils.xarray_functions import add_time, compute_radiance
    xr_df = cut_file(xr_df, gdf)
    xr_df = add_time(xr_df)
    xr_df = compute_radiance(xr_df)
    xr_df = xr_df.drop('channel_3')
    xr_df = xr_df.assign(ndvi=(
        xr_df['channel_2'] - xr_df['channel_1']) / (xr_df['channel_2'] + xr_df['channel_1']))
    return xr_df


"""
Smoothing Pipelines
"""

class XarrayWS(xarray.Dataset):
    def __init__(self, data_vars: Union[Mapping[Any, Any], None] = None, coords: Union[Mapping[Any, Any], None] = None, 
                 attrs: Union[Mapping[Any, Any], None] = None) -> None:
        super().__init__(data_vars, coords, attrs)

    def apply_ws2doptvp(self, variable, p, lambda_min=-2, lambda_max = 3):
        series, weights = self._generate_weights(self[variable])
        data = self._apply_vectorized_funct(
                            self._apply_ws2doptvp, 
                            series,
                            weights,
                            p,
                            lambda_min,
                            lambda_max)
        #self = self.assign(smoothed_series = data)
        return data.to_dataset()
        
        
    def _generate_weights(self, datarray:xarray.DataArray)->xarray.DataArray:
        w = xr.where(datarray.astype(np.float32).isnull(), 0, 1)
        y = datarray.astype(np.float32).where(datarray.notnull(),0)
        return y, w
    
    def _apply_ws2doptvp(self, y, w, p, lambda_min, lambda_max):
        w_corr = w.astype(np.double)
        y_corr = y.astype(np.double)
        z, sopt = ws2doptvp(y_corr, w_corr, 
                            array("d", np.arange(lambda_min, lambda_max, 0.2).round(2)), 
                            p=p)
        z_arr =  np.array(z, dtype=np.float32)
        return z_arr
    
    def _smooth_along_axis(self, f, datarray, w, lambda_min, lambda_max):
        dim = datarray.get_axis_num('time')
        return dask.array.apply_along_axis(self, f, dim, datarray, w, lambda_min, lambda_max)

    def _apply_vectorized_funct(self, f, datarray, w, p, lambda_min, lambda_max):
        print("Calculating...")
        results= xarray.apply_ufunc(f,
                            datarray,
                            w,
                            p,
                            lambda_min,
                            lambda_max,
                            input_core_dims=[["time"],["time"],[],[],[]],
                            output_core_dims=[["time"]],        # Specify the output dimension
                            vectorize=True,               # Vectorize the operation
                            dask="parallelized",
                            output_dtypes=[np.float32]).compute()
        
        return results
    
def apply_seviri_cloudmask(dataset:xr.Dataset, cloud_mask:xr.Dataset):
    dataset = dataset.drop_duplicates(dim=["time"])
    dataset['time'] = dataset.indexes['time'].normalize()
    cloud_mask['time'] = cloud_mask.indexes['time'].normalize()
    ds1, ds2 = xr.align(dataset, cloud_mask)
    return ds1.where(ds2.cloud_mask!=2)
    

def extract_apply_cloudmask(ds, ds_cl, resample=False, 
                            include_water =True,
                            downsample=False):
    """
    Pipeline to process SEVIRI data with clouds, water bodies and WS
    """
    from utils.xarray_functions import compute_ndvi, clean_ndvi

    def checkVars(ds, var):
        assert var  in ds.data_vars, f"Variable {var} not in dataset"

    [checkVars(ds, var)  for var in ["channel_1","channel_2","ndvi"]]

    ### normalize time in order for the two datasets to match
    ds_cl['time'] = ds_cl.indexes['time'].normalize()
    ds['time'] = ds.indexes['time'].normalize()
    
    if resample==True:
    #### reproject cloud mask to base dataset
        reproj_cloud = ds_cl['cloud_mask'].rio.reproject_match(ds['ndvi'])
        ds_cl_rp = reproj_cloud.rename({'y':'lat', 'x':'lon'})

    else:
        ds_cl_rp = ds_cl

    ### apply time mask where values are equal to 1, hence no clouds over land, 0 = no cloud over water
    if include_water==True:
        ds_subset = ds.where((ds_cl_rp==1)|(ds_cl_rp==0)) #ds = ds.where(ds.time == ds_cl.time)
    else:
        ds_subset = ds.where(ds_cl_rp==1)
    ### recompute corrected ndvi
    res_xr = compute_ndvi(ds_subset)

    ### mask all the values equal to 0 (clouds)

    mask_clouds = clean_ndvi(ds)
    ### recompute corrected ndvi
    mask_clouds = compute_ndvi(mask_clouds)

    #### downsample to 5 days
    if downsample==True:
        "Starting downsampling the Dataset"
        res_xr_p = downsample(res_xr)
        #### downsampled df
        mask_clouds_p = downsample(mask_clouds)
        return mask_clouds_p, res_xr_p,  mask_clouds, res_xr ### return 1) cleaned dataset with clouds 
                                                         ### 2) imputation with max over n days
                                                         ### 3) cloudmask dataset original sample
                                                         ### 4) cloudmask dataset downsampled
    else:
        return mask_clouds, res_xr  ###1) dataset without zeros (clouds) 2) dataset with cloud mask applied


def apply_whittaker(datarray:xr.DataArray, 
                    lambda_par:int=1, 
                    prediction:str="P1D", 
                    time_dim:str="time"):
    
    from fusets import WhittakerTransformer
    from fusets._xarray_utils import _extract_dates, _output_dates, _topydate
    
    result = WhittakerTransformer().fit_transform(datarray.load(),
                                                  smoothing_lambda=lambda_par, 
                                                  time_dimension=time_dim, 
                                                  prediction_period=prediction)
    dates = _extract_dates(datarray)
    expected_dates = _output_dates(prediction,dates[0],dates[-1])
    datarray['time'] = datarray.indexes['time'].normalize()
    datarray = datarray.assign_coords(time = datarray.indexes['time'].normalize())
    result['time'] = [np.datetime64(i) for i in expected_dates]
    return result