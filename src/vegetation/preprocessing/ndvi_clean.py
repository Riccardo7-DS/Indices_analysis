#!/usr/bin/env python
# coding: utf-8


import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import numpy as np
import os
import xarray as xr
from utils.ndvi_functions import convert_ndvi_tofloat

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

class XarrayWS(xarray.Dataset):
    def __init__(self, data_vars: Union[Mapping[Any, Any], None] = None, coords: Union[Mapping[Any, Any], None] = None, 
                 attrs: Union[Mapping[Any, Any], None] = None) -> None:
        super().__init__(data_vars, coords, attrs)

    def apply_ws2doptvp(self, variable, p, lambda_min=-2, lambda_max = 3):
        series, weights = self._generate_weights(self[variable])
        data = self._apply_smooth_method(
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

    def _apply_smooth_method(self, f, datarray, w, p, lambda_min, lambda_max):
        print("Calculating...")
        with ProgressBar():
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