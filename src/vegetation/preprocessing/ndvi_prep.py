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
from array import array
from dask.diagnostics import ProgressBar
from typing import Literal
import logging 
logger = logging.getLogger(__name__)

def SeviriMVCpipeline():
    import re
    import os
    import xarray as xr
    from utils.xarray_functions import add_time, compute_radiance
    from vegetation.analysis.indices import compute_ndvi
    from utils.function_clns import config, safe_open_mfdataset, read_netcdfs, subsetting_pipeline
    import logging
    import pyproj
    from vegetation.preprocessing.ndvi_prep import apply_seviri_cloudmask, remove_ndvi_outliers, NDVIPreprocess

    chunks = {"time":-1, "lat":"auto", "lon":"auto"}

    def preprocess_file(ds):
        ds = add_time(ds)
        ds = compute_radiance(ds)
        return ds

    def preprocess_cloud(ds):
        if "time" not in ds.dims:
            ds = add_time(ds)
        return ds

    base_dir = config["SEVIRI"]["download"]
    cloud_dir = config["NDVI"]["cloud_path"]
    hour_pattern = re.compile(r'\d{2}_\d{2}')
    hour_folders = [folder for folder in os.listdir(base_dir) if hour_pattern.match(folder)]
    hour_folders = [hour_folders[2], hour_folders[0],hour_folders[1]]

    datasets = []

    # Loop through hour folders
    for hour_folder in ["09_15", "10_30", "12_15"]:
        logging.info(f"Processing SEVIRI data for hour {hour_folder}")
        folder_path = os.path.join(base_dir, hour_folder)
        cloud_path = os.path.join(cloud_dir, hour_folder)

        # Filter NetCDF files
        ds_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nc')]
        cl_files = [os.path.join(cloud_path, f) for f in os.listdir(cloud_path) if f.endswith('.nc')]

        # Open datasets
        ds_temp = safe_open_mfdataset(ds_files, preprocess_file, chunks)
        ndvi = compute_ndvi(band1=ds_temp["channel_1"], 
                            band2=ds_temp["channel_2"])
        ds_cl = safe_open_mfdataset(cl_files, preprocess_cloud, chunks)

        # Apply cloud
        logging.info(f"Applying cloudmask for hour {hour_folder}")
        temp_ds = apply_seviri_cloudmask(ndvi, ds_cl, align=False)
        datasets.append(temp_ds)
    
    return datasets

def seviri_pipeline():
    from utils.function_clns import config
    rawfolder = "batch_1"
    output_ndvi = "ndvi_full_image.zarr"
    cloudpath = "09_15"
    lambda_max = 4
    SeviriProcess(rawfolder=rawfolder,
                  ndvifile=output_ndvi, 
                  cloudfolder=cloudpath,
                  lambda_max=lambda_max)

class SeviriProcess():
    def __init__(self, rawfolder:str,
                 cloudfolder:str,
                 ndvifile:str,
                 parallel:bool = False,
                 p:float=0.99,
                 lambda_max:float=3):
        
        """
        rawfolder: name of the source MSG seviri images folder
        cloudfolder: name of the cloud mask folder
        ndvifile: name of the already created ndvi file, otherwise name of the output ndvi file to be created
        parallel: whether to load files in parallel using dask
        p: parameter for Whittaker filter (WS)
        lambda_max: paramter for optimal curve to choose lambda for WS
        """

        self.p = p
        self.lambda_max = lambda_max
        
        ### SEVIRI loading and cloud masking
        from utils.function_clns import config 
        from vegetation.preprocessing.ndvi_prep import remove_ndvi_outliers

        chunks = {"time":-1, "lat":"auto", "lon":"auto"}
        ndvi_path = os.path.join(config["NDVI"]["ndvi_path"], ndvifile)
    
        if os.path.isfile(ndvi_path) is False:
            raw_path = (config["SEVIRI"]["download"], rawfolder)
            self._generate_ndvi(raw_path)
        
        ds_ndvi = xr.open_zarr(ndvi_path, chunks=chunks)
        ds_cloud = self._load_cloud_ds(cloudfolder, parallel=parallel)

        ds = apply_seviri_cloudmask(ds_ndvi, ds_cloud)
        qf_ds = self._generate_qflag(ds["ndvi"])
        ds_chunked = ds.chunk(chunks={"time":-1, "lat":90, "lon":90})
        ds_whit = self._apply_whittaker(ds_chunked)
        # ds_whit["ndvi"] = remove_ndvi_outliers(ds_whit["ndvi"])

    def _preprocess(self, ds):
        from utils.xarray_functions import add_time
        ds = add_time(ds)
        return ds
    
    def _load_cloud_ds(self, cloudfolder, parallel):
        from utils.function_clns import config
        cloud_path = os.path.join(config["NDVI"]["cloud_path"], cloudfolder)
        files = [os.path.join(cloud_path, file) for file in os.listdir(cloud_path) if file.endswith(".nc")]
        return xr.open_mfdataset(files, 
                                preprocess=self._preprocess, 
                                engine='netcdf4', 
                                parallel=parallel)
    
    def _generate_ndvi(self, path:str, outputname:str):
        from utils.xarray_functions import add_time, compute_radiance, compute_ndvi
        import zarr
        from utils.function_clns import config
        from dask.diagnostics import ProgressBar
        
        logger.info("Loading single MSG SEVIRI files...")
        chunks = {"time":200, "lat":50, "lon":50}

        def preprocess(ds):
            ds = add_time(ds)
            ds = compute_radiance(ds)
            return ds 
    
        files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".nc")]
        ds = xr.open_mfdataset(files, chunks=chunks, 
                               preprocess=preprocess, 
                               parallel=True)
        
        ndvi = compute_ndvi(ds)
        
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
        # encodings
        enc = {x: {"compressor": compressor} for x in ndvi}

        with ProgressBar():
            ndvi.to_zarr(os.path.join(config["NDVI"]["ndvi_path"], 
                                      outputname),
                     encoding=enc)
    
    def _apply_whittaker(self, ds):
        from utils.function_clns import config
        import zarr
        from dask.diagnostics import ProgressBar
        import logging 
        logger = logging.getLogger(__name__)

        logger.info("Starting applying Whittaker smoother...")

        ws_ds = XarrayWS(ds)

        with ProgressBar():
           ds_clean = ws_ds.apply_ws2doptvp(variable="ndvi", 
                                            p=self.p, 
                                            lambda_max=self.lambda_max)

        filename = "seviri_full_image_smoothed.zarr"
        smoothed_ndvi_path = os.path.join(config["NDVI"]["ndvi_path"], filename)
        
        if not os.path.isfile(smoothed_ndvi_path):
            compressor = zarr.Blosc(cname="zstd", clevel=4, shuffle=2)
            # encodings
            enc = {x: {"compressor": compressor} for x in ds_clean}
            with ProgressBar():
                ds_clean.to_zarr(smoothed_ndvi_path,
                             encoding=enc, mode="w")

        return ds_clean

    def _find_streaks(self, arr):
        streaks = np.zeros_like(arr)  # Create an array of zeros with the same shape as input
        quality_flag =  np.zeros_like(arr) 
        current_streak = 0
        for i, value in enumerate(arr):
            if np.isnan(value):
                current_streak += 1
                streaks[i] = np.NaN
                quality_flag[i] = np.NaN
            else:
                if current_streak > 0:
                    streaks[i] = current_streak
                    quality_flag[i - current_streak:i] = current_streak
                    current_streak = 0
        if current_streak > 0:
            streaks[i] = current_streak
            quality_flag[-current_streak:] = current_streak
        return streaks, quality_flag

    def _generate_qflag(self, datarray):
        from utils.function_clns import config
        logger.info("Starting calculating quality flag...")
        # Apply find_streaks along the time dimension with dask.delayed
        streaks_data, quality_flag = xr.apply_ufunc(
            self._find_streaks,
            datarray.chunk({'time': -1, "lat":80, "lon":80}),
            input_core_dims=[['time']],
            output_core_dims=[['time'], ["time"]],
            # exclude_dims={"time"},
            vectorize=True,
            dask='parallelized',
            # dask_gufunc_kwargs={"output_sizes": {"time": len(ds["time"])}},
            output_dtypes=[np.float32,np.float32], 
        )

        filename = "mask_num_streaks.zarr"
        dest_path = os.path.join(config["NDVI"]["ndvi_path"], filename)
        
        ds = datarray.to_dataset()
        # Assign the results back to the dataset
        ds['streaks'] = streaks_data
        ds['quality_flag'] = quality_flag

        if not os.path.isfile(dest_path):
            with ProgressBar():
                ds.to_zarr(dest_path, mode="w")

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


def generate_max_eumetsat_ndvi(temp_path:str, chunks="auto"):
    files = [os.path.join(temp_path,f) for f in os.listdir(temp_path) if f.endswith(".nc")]

    import re
    from datetime import datetime
    from utils.function_clns import subsetting_pipeline

    def extract_date(filename):
        filename = filename.split("/")[-1]

        # Use regex to find the date part in the string
        date_match = re.search(r'(\d{14})', filename)

        if date_match:
            # Extract the matched date string
            date_str = date_match.group(1)

            # Parse the date string to a datetime object
            date_obj = datetime.strptime(date_str, "%Y%m%d%H%M%S")

            return date_obj
        else:
            return None

    def preprocess(ds):
        path = ds.encoding["source"]
        date = extract_date(path)
        return ds.expand_dims(time=[date])

    ds_max_temp = xr.open_mfdataset(files, chunks=chunks, preprocess=preprocess)
    ds_ndvi_max = subsetting_pipeline(ds_max_temp).rename({"Band1":"ndvi"})
    ds_ndvi_max["ndvi"] = xr.where(ds_ndvi_max["ndvi"]==255, np.NaN, ds_ndvi_max["ndvi"])
    ds_ndvi_max = ds_ndvi_max["ndvi"]/100
    ds_ndvi_max = NDVIPreprocess(ds_ndvi_max).get_processed_data()
    ds_ndvi_max["time"] = ds_ndvi_max["time"] - pd.Timedelta(hours=12)
    return ds_ndvi_max


def load_eumetsat_ndvi_max(filepath:str, 
                  chunks:dict={'time': 50, "lat": 250, "lon":250},
                  save_file:bool=False):
    from utils.function_clns import subsetting_pipeline
    import logging
    logger = logging.getLogger(__name__)

    if filepath is not None:
        if os.path.isfile(filepath):
            max_ndvi = xr.open_dataset(filepath, chunks=chunks)["ndvi"]
        else:
            logger.info("The file does not exist")

    else:
        path = "/media/BIFROST/N2/Riccardo/MSG/msg_data/NDVI/archive.eumetsat.int/umarf-gwt/onlinedownload/riccardo7/4859700/temp/time/ndvi_eumetsat.nc"
        ds_ndvi = xr.open_mfdataset(path, engine="netcdf4", chunks=chunks)
        ds_ndvi = subsetting_pipeline(ds_ndvi).rename({"Band1":"ndvi"})
        ds_ndvi["ndvi"] = xr.where(ds_ndvi["ndvi"]==255, np.NaN, ds_ndvi["ndvi"])
        max_ndvi = ds_ndvi["ndvi"]/100

        if save_file is True:
            from dask.diagnostics import ProgressBar
            filename = "seviri_daily_ndvimax.nc"
            compression = {"ndvi" :{'zlib': True, "complevel":4}}
            from utils.function_clns import config

            with ProgressBar():
                max_ndvi.to_dataset(name="ndvi").to_netcdf(os.path.join(config["NDVI"]["ndvi_path"],
                                                            filename), encoding=compression)
            
    return max_ndvi


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

def remove_ndvi_outliers(ds:xr.DataArray, impute:bool=False):
    if impute is True:
        ndvi = xr.where(ds<-1,-1, ds)
        return xr.where(ds>1, 1, ndvi)
    else:
        return ds.where((ds<=1)&(ds>=-1))

"""
Functions to clean, smooth NDVI
"""

def correct_ndvi_bias(dataset):
    return 0.015705526 + dataset * 1.256689


class NDVIPreprocess():
    def __init__(self, data)->None:
        from utils.function_clns import prepare, subsetting_pipeline
        import numpy as np
        data = prepare(data)
        data = subsetting_pipeline(data)
        data = data.astype(np.float32)
        data = self._set_nulls(data)
        data = data.transpose("time","lat","lon")
        self.processed_data = prepare(data)
    
    def _set_nulls(self, data):
        import numpy as np
        data = data.rio.write_nodata(np.nan, inplace=True)
        data = data.rio.write_nodata("nan", inplace=True)
        return data

    def get_processed_data(self):
        return self.processed_data


def smooth_coordinate(datarray:xr.DataArray, 
                      lat:float, lon:float, p:float=0.99,
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


def savigol_apply_datacube(ds: xr.DataArray, 
                   window:int, 
                   poly_order:int) -> xr.DataArray:
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
        from modape.whittaker import ws2doptv, ws2d, ws2doptvp

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
    
def apply_seviri_cloudmask(dataset:xr.Dataset, 
                           cloud_mask:xr.Dataset,
                           align:bool=True):
    dataset = dataset.drop_duplicates(dim=["time"])
    dataset['time'] = dataset.indexes['time'].normalize()
    cloud_mask['time'] = cloud_mask.indexes['time'].normalize()
    if align is True:
        dataset, cloud_mask = xr.align(dataset, cloud_mask)
    return dataset.where(cloud_mask.cloud_mask!=2)
    

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
    from utils.xarray_functions import _extract_dates, _output_dates, _topydate
    
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