from utils.function_clns import config, prepare, handle_unhandled_exception, prepare_datarray, get_lat_lon_window, crop_image_left, crop_image_right
from typing import Literal, Union
import os 
import xarray as xr
import numpy as np
import rasterio
import sys
from dask.diagnostics import ProgressBar
from rasterio.enums import Resampling
import logging
import pickle
from datetime import timedelta
from definitions import ROOT_DIR
logger = logging.getLogger(__name__)
sys.excepthook = handle_unhandled_exception

class PrecipDataPreparation():
    def __init__(self, 
                 args:dict,
                 variables:list,
                 precipitation_data:str="ERA5", 
                 ndvi_data:str="seviri_full_image_smoothed.zarr",
                 load_zarr_features:bool = False,
                 load_local_precp:bool = False,
                 precp_filename:str=None,
                 precp_format:Literal["zarr", "nc"]="zarr",
                 interpolate:bool = False) -> None:
        
        cb = ProgressBar().register()
        self.basepath = config["DEFAULT"]["basepath"]
        self.time_end = config['DEFAULT']['date_end']
        self.time_start = config['DEFAULT']['date_start']
        self.model = args.model
        self.precp_product = precipitation_data
        self.ndvi_filename = ndvi_data

        assert precp_format in ["zarr", "nc"], "Precipitation dataset must be either \"zarr\" or \"nc\"" 

        if load_zarr_features is False:
            if self.model == "ERA5" and precp_filename is None:
                raise ValueError("ERA5 not implemented in file, specify folder name")
            self.precp_ds = self.process_input_vars(variables, 
                                load_local_precp, 
                                interpolate=interpolate, 
                                precp_folder=precp_filename,
                                precp_format= precp_format,
                                save=False)
        else:
            self._load_local_precipitation(precipitation_data)
            path = os.path.join(self.precp_path, 
                                precp_filename + ".zarr")
            # path = os.path.join(config["PRECIP"]["ERA5_land"]["path"],
            #                                 "final_vars_filled.zarr")
            self.precp_ds = xr.open_zarr(path)\
                .sel(time=slice(self.time_start, self.time_end))
        
        ndvi_ds = self._load_processed_ndvi(self.basepath, 
                                            self.ndvi_filename)
        ndvi_ds = self._preprocess_array(ndvi_ds)
        ndvi_ds = self._reproject_odc(ndvi_ds, self.precp_ds)
        # ndvi_ds = self._reproject_raster(ndvi_ds, self.precp_ds, 
        #                                  method=Resampling.bilinear)

        logger.debug("Proceeding with interpolation of instance over time")
        from utils.xarray_functions import extend_dataset_over_time 
        self.ndvi_ds = extend_dataset_over_time(ndvi_ds)
        
        logger.debug("Normalizing datasets...")
        if args.normalize is True:
            datasets, scalers = \
                self._normalize_datasets(self.ndvi_ds, self.precp_ds)
            self.ndvi_ds, self.precp_ds, self.ndvi_scaler = datasets[0], datasets[1], scalers[0]

        if args.fillna is True:
            logger.debug("Filling null values...")
            self.precp_ds = self.precp_ds.fillna(-1)
            self.ndvi_ds = self.ndvi_ds.fillna(-1)
            
        if args.crop_area is True:
            logger.debug("Cropping datasets...")
            self.precp_ds = self._crop_area_for_dl(self.precp_ds)
            self.ndvi_ds = self._crop_area_for_dl(self.ndvi_ds)

        ### Check nulls
        # self._count_nulls(self.precp_ds)
        # self._count_nulls(self.ndvi_ds)

    def _estimate_gigs(self, dataset:xr.Dataset, description:str):
        logger.info("{d} dataset has {g} GiB".format(d=description, 
                                                     g=dataset.nbytes * 1e-9))

    def process_input_vars(self, 
                           variables:list, 
                           load_local_precp:bool, 
                           interpolate:bool, 
                           save:bool=False,
                           precp_folder:str="batch_final",
                           precp_format="nc"):
        
        def save_dataset_tozarr(dataset, dest_path):
            logger.debug("Starting exporting processed data to zarr...")
            out = dataset.to_zarr(dest_path, mode="w", compute=False)
            res = out.compute()
            logger.debug("Successfully saved preprocessed variables")

        if load_local_precp is True:
            self._load_local_precipitation(self.precp_product, 
                                           folder=precp_folder,
                                           format=precp_format)
            precp_ds = self._preprocess_array(path = self.precp_path,
                                               filename=self.precp_filename, 
                                               interpolate=interpolate)
            precp_ds = self._local_precipitation_cleaning(precp_ds, self.precp_product)
            if len(variables) > 1:
                logger.debug(f"Adding soil moisture to variables {variables}")
                precp_ds = self._transform_ancillary(precp_ds, temporal_resolution="daily")
                self._estimate_gigs(precp_ds, "Hydro variables")
                logger.info(precp_ds)
        else:
            # File for ERA5 res 0.25
            era5_dest_path = os.path.join(self.basepath, "hydro_vars.zarr")

            if os.path.exists(era5_dest_path):
                logger.debug("Found zarr file in destination path. Proceeding with loading...")
                precp_ds = xr.open_zarr(era5_dest_path)
            else:
                precp_ds, era5_data = self._load_arco_precipitation(variables)

            temp_resolution_hours = self._get_temporal_resolution(precp_ds)
            precp_ds = self._preprocess_array(precp_ds, 
                                              interpolate=interpolate)

            if (len(variables) > 1) and (temp_resolution_hours==1):
                logger.debug(f"Processing ancillary variables {variables} from hourly to"
                             f"daily temporal resolution")
                precp_ds = self._transform_ancillary(precp_ds, era5_data, 
                                                     temporal_resolution="hourly")
            elif temp_resolution_hours==24:
                precp_ds = self._transform_ancillary(precp_ds, temporal_resolution="daily")
            
            self._estimate_gigs(precp_ds, "Hydro variables")
            logger.info(precp_ds)

            if save is True:
                logger.info("Saving final processed features with soil moisture and ERA5 to zarr")
                save_dataset_tozarr(precp_ds, os.path.join(self.basepath),"all_hydro.zarr")
            

        return precp_ds
    
    
    def _impute_values(self, dataset:Union[xr.Dataset, xr.DataArray], value:float):
        return dataset.fillna(value)
    
    def _get_temporal_resolution(self, dataset:Union[xr.Dataset, xr.DataArray]):
        return int((dataset["time"].diff(dim='time')[0].dt.total_seconds()/60/60).values)

    def _local_precipitation_cleaning(self, dataset, precp_dataset):
        from utils.xarray_functions import shift_xarray_overtime
        if precp_dataset == "ERA5_land":
            ds = shift_xarray_overtime(dataset, "1D") 
            return ds.rename({"pev":"potential_evaporation", "e":"evaporation",
                         "t2m":"2m_temperature","tp":"total_precipitation"})
        else:
            return dataset


    def _load_local_precipitation(self, 
                                  precp_dataset:str,
                                  folder=None, 
                                  format:Literal["nc", "zarr"]="zarr"):
        import os
        import re
        from utils.function_clns import config
        logger.debug(f"Loading local {precp_dataset} precipitation file in format {format}")

        def get_path(config_dict, dataset, product):
            """
            Function to retrieve the path for a given metric and product.

            Parameters:
            config_dict (dict): The configuration dictionary containing paths.
            dataset (str): The metric type (e.g., 'SPI' or 'PRECIP').
            product (str): The product name (e.g., 'IMERG', 'GPCC').

            Returns:
            str: The path associated with the specified metric and product.
            """
            try:
                return config_dict[dataset][product]
            except KeyError:
                return f"Path not found for metric '{dataset}' and product '{product}'"

        # Create the dictionary
        config_dict = {
            "SPI": {
                "IMERG": config['SPI']['IMERG']['path'],
                "GPCC": config['SPI']['GPCC']['path'],
                "CHIRPS": config['SPI']['CHIRPS']['path'],
                "ERA5": config['SPI']['ERA5']['path'],
                "MSWEP": config['SPI']['MSWEP']['path']
            },
            "PRECIP": {
                "IMERG": config['PRECIP']['IMERG']['path'],
                "GPCC": config['PRECIP']['GPCC']['path'],
                "CHIRPS":config['PRECIP']['CHIRPS']['path'],
                "ERA5": config['PRECIP']['ERA5']['path'],
                "TAMSTAT": config['PRECIP']['TAMSTAT']['path'],
                "MSWEP": config['PRECIP']['MSWEP']['path'],
                "ERA5_land": config["PRECIP"]["ERA5_land"]["path"]
            }
        }
        if folder is not None:
            path = get_path(config_dict,"PRECIP", precp_dataset)
            path = os.path.join(path, folder)
            filename = None

        elif "SPI" in precp_dataset:
            precp_dataset = precp_dataset.replace("SPI_","")
            path = get_path(config_dict, "SPI", precp_dataset)
            late =  re.search(r'\d+', path).group()
            filename = "spi_gamma_{}".format(late)
            filename = [f for f in os.listdir(path) if filename in f][0]
            self.spi_latency = late
        else:
            path = get_path(config_dict,"PRECIP", precp_dataset)
            filename = f"{precp_dataset}_merged.{format}"

        self.precp_path = path
        self.precp_filename = filename

    def _load_arco_precipitation(self, variables:Union[list, str]):
        from ancillary.hydro_data import query_arco_era5
        from datetime import datetime, timedelta
        from utils.function_clns import hoa_bbox
        
        logger.debug(f"Querying ARCO data from Google Cloud Storage"\
                     f" for dates {self.time_start} to {self.time_end}")
        
        bbox = hoa_bbox()
        
        input_data = query_arco_era5(variables, 
                                     bounding_box=bbox,
                                     date_min=self.time_start,
                                     date_max=(datetime.strptime(self.time_end, "%Y-%m-%d") + \
                                        timedelta(days=1)).strftime("%Y-%m-%d"))
        input_data = input_data.shift(time=1)\
            .sel(time=slice(self.time_start, self.time_end))
        logger.debug("Processing input data...")
        precp_data = self._process_precp_arco(input_data)
        return precp_data, input_data
    
    def _extract_shape_bbox(self, bbox):
        from shapely.geometry import box
        import geopandas as gpd
        geom = box(bbox[1], bbox[0],bbox[3], bbox[2])
        return gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[geom]) 
    
    def _adjust_sm_coords(self, ds):
        # ds["lat"] = (ds.coords['lat'] + 90) % 180 - 90
        ds = ds.sortby(ds["lon"])
        ds = ds.sortby(ds["lat"]).isel(lat=slice(None, None, -1))
        return ds
    
    def _adjust_era5_coords(self, ds):
        ds["lon"] = (ds.coords['lon'] + 180) % 360
        ds["lat"] = (ds.coords['lat'] + 90) % 180 - 90
        ds = ds.sortby(ds["lon"])
        ds = ds.sortby(ds["lat"]).isel(lat=slice(None, None, -1))
        return ds

    
    def _transform_ancillary(self, precp_data:xr.DataArray, 
                        input_data:xr.Dataset=None, 
                        temporal_resolution:Literal["daily","hourly"]="hourly")-> xr.Dataset:
        
        if temporal_resolution == "hourly":
            ds_temp_min, ds_temp_max = self._process_temperature_arco(input_data)
            evap, pot_evap = self._process_evapo_arco(input_data)
            precp_data = precp_data.to_dataset()
            precp_data = precp_data.assing(evap = evap, potential_evap = pot_evap, 
                                    temp_max= ds_temp_max, temp_min = ds_temp_min)

        hydro_data = self._process_soil_moisture(precp_data)\
            .chunk(chunks={"time":-1, "lat":"auto", "lon":"auto"})

        return hydro_data
    
    def _load_processed_ndvi(self, 
                             path:str, 
                             filename:str):
        from  vegetation.preprocessing.ndvi_prep import remove_ndvi_outliers
        chunks={"time":-1, "lat":"auto", "lon":"auto"}
        ds_clean = xr.open_zarr(os.path.join(path, filename), 
                        chunks=chunks)
        streaks_data = xr.open_zarr(os.path.join(
            path, 
            "mask_num_streaks.zarr"),
            chunks=chunks)
        
        ### set to null all the values with more than 45 days of gaps
        ds_clean["ndvi"] = ds_clean["ndvi"].where(
            streaks_data.quality_flag<=45)
        
        ds_clean["ndvi"] = remove_ndvi_outliers(
            ds_clean["ndvi"], impute=True)
        
        return ds_clean["ndvi"]


    def _set_nulls(self, dataset, value=np.nan):

        if isinstance(dataset, xr.Dataset):
            for var in dataset.data_vars:
                dataset[var].rio.write_nodata(value, inplace=True)
        elif isinstance(dataset, xr.DataArray):
            dataset.rio.write_nodata(value, inplace=True)
        else:
            error = ValueError("The provided dataset must be in xarray format")
            logging.error(error)
            raise error
        # data = data.rio.write_nodata("nan", inplace=True)
        return dataset
    
    def _preprocess_array(self,
                          dataset: Union[xr.DataArray, xr.Dataset] = None,
                          path: str = None,
                          filename: str = None,
                          variable: str = None, 
                          interpolate:bool = False,
                          invert:bool=False) -> xr.DataArray:
        from utils.function_clns import clip_file
        from dask.diagnostics import ProgressBar

        ############## 1) Open dataset ############## 
        if dataset is None:
            # if path is None or filename is None:
            #     error = ValueError(f"Either 'dataset' must be provided or both 'path'" 
            #                      f" and 'filename' must be provided.")
            #     logging.error(error)
            #     raise error
            # else:
        # Open the precipitation file with xarray
            tot_files = len([f for f in os.listdir(path) if f.endswith(".nc")])
            if (filename is None) and \
                (tot_files>1):
                logger.info(f"Found {tot_files} files in directory, proceeding with lazy loading...")
                dataset = xr.open_mfdataset(os.path.join(path, "*.nc"))
                logger.info("Success")
            elif filename.endswith(".nc"):
                dataset = xr.open_dataset(os.path.join(path, filename))
            elif filename.endswith(".zarr"):
                dataset = xr.open_zarr(os.path.join(path, filename))
            else:
                err = logger.error(f"Please select a valid file format between "
                                   f"\"zarr\" or \"netcdf\"")
                raise err
                
        ############## 2) Clip dataset ##############

        dataset = prepare(clip_file(dataset, gdf=None, invert=invert))\
            .sel(time=slice(self.time_start, self.time_end))
        
        ############## 3) Set nulls ##############

        dataset = self._set_nulls(dataset)

        ############## 4) Interpolating ############

        if interpolate is True:
            logger.info("Interpolating dataset")
            with ProgressBar():
                dataset = dataset.chunk(dict(time=-1))
                dataset = dataset.interpolate_na(dim="time", 
                                                 method="nearest")
        ############## 5) Selecting vars ############

        if variable is not None:
            dataset = dataset[["time", "lat", "lon", variable]]
        return dataset

    def _count_nulls(self, dataset):
        def log(dataset, var=None):
            if var is None:
                var = dataset.name
            logger.info("On average at pixel-level, "
                        "there are {p} nulls for variable {v}"\
                        .format(p=dataset.isnull().sum(["time"]).mean().values,
                        v=var))

        if isinstance(dataset, xr.Dataset):
            for var in dataset.data_vars:
                if "time" in dataset[var].dims:
                    log(dataset[var], var)
        elif isinstance(dataset, xr.DataArray):
            log(dataset)

    def _normalize_datasets(self, *datasets):
        from analysis.deep_learning.utils_models import StandardNormalizer

        normalized_datasets = []
        scalers = []

        for dataset in datasets:
            if isinstance(dataset, xr.Dataset):
                dataset_copy = dataset.copy()  # Make a copy to avoid modifying the original dataset
                for var in dataset_copy.data_vars:
                    scaler = StandardNormalizer(min=np.nanmin(dataset_copy[var]), 
                                            max=np.nanmax(dataset_copy[var]))
                    dataset_copy[var] = scaler.transform(dataset_copy[var])
                normalized_datasets.append(dataset_copy)
                scalers.append(scaler)

            elif isinstance(dataset, xr.DataArray):
                dataset_copy = dataset.copy()  # Make a copy to avoid modifying the original dataset
                scaler = StandardNormalizer(min=np.nanmin(dataset_copy), 
                                        max=np.nanmax(dataset_copy))
                normalized_datasets.append(scaler.transform(dataset_copy))
                scalers.append(scaler)

        return normalized_datasets, scalers
    
    def _crop_area_for_dl(self, 
                          dataset:Union[xr.DataArray, xr.Dataset]):
        
        if self.model == "GWNET" or self.model=="WNET":
            logger.info("Selecting data for GCNN WaveNet")
            try:
                self.dim = config['GWNET']['dim']
                idx_lat, lat_max, idx_lon, lon_min = get_lat_lon_window(dataset, self.dim)
                sub_dataset = dataset.sel(lat=slice(lat_max, idx_lat), 
                                         lon=slice(lon_min, idx_lon))
            except IndexError:
                logger.error("The dataset {} is out of bounds when using a subset, using original product"\
                             .format(self.precp_product))
                self.dim = max(len(sub_dataset["lat"]),
                               len(sub_dataset["lon"]))

        elif self.model =="CONVLSTM":
            logger.info("Selecting data for ConvLSTM")
            self.dim = config["CONVLSTM"]["dim"]
            idx_lat, lat_max, idx_lon, lon_min = crop_image_left(dataset, self.dim)
            sub_dataset = dataset.sel(lat=slice(lat_max, idx_lat), 
                                      lon=slice(lon_min, idx_lon))
            
        else:
            from utils.function_clns import subsetting_pipeline
            sub_dataset = subsetting_pipeline(dataset)

        return sub_dataset
    
    def _reproject_odc(self,
                       resample_ds:Union[xr.Dataset, xr.Dataset],
                       target_ds:Union[xr.Dataset, xr.DataArray],
                       resampling:str = "bilinear"):
        
        from utils.xarray_functions import odc_reproject
        logger.info(f"Proceeding with reprojection of source dataset with {resampling} resampling...")
        ds_repr = odc_reproject(resample_ds, target_ds, resampling=resampling)\
                .rename({"longitude": "lon", "latitude": "lat"})
        ds_repr["lat"] = target_ds["lat"]
        ds_repr["lon"] = target_ds["lon"]
        return ds_repr
    
    def _reproject_raster(self, 
                          resample_ds:xr.DataArray,
                          target_ds:Union[xr.Dataset, xr.DataArray],
                          method:rasterio.enums.Resampling):
        if type(target_ds) == xr.Dataset:
            var_target = [var for var in target_ds.data_vars][0]
            target_ds = prepare(target_ds[var_target])

        # elif type(target_ds)== xr.DataArray:
            
        ds_repr = resample_ds.transpose("time","lat","lon").rio.reproject_match(
            target_ds, resampling=method).rename({'x':'lon','y':'lat'})
        return prepare(ds_repr)
    
    def _process_precp_arco(self, 
                            test_ds:xr.DataArray, 
                            save:bool=False):
        from ancillary.hydro_data import process_era5_precp
        from utils.function_clns import config
        ### change one timeframe in order to accomodate for how ERA5 computes the accumulations
        #1st January 2017 time = 01 - 23  will give you total precipitation data to cover 00 - 23 UTC for 1st January 2017
        #2nd January 2017 time = 00 will give you total precipitation data to cover 23 - 24 UTC for 1st January 2017
        ### add dataset attrs and convert to mm
        test_ds = process_era5_precp(test_ds, var="total_precipitation")

        ### resample to daily
        logger.debug("Starting resampling from hourly to daily...")
        precp_ds = test_ds["total_precipitation"].resample(time="1D")\
            .sum()
        
        logger.debug("Setting precipitation values inferior than 0 to 0")
        precp_ds = precp_ds.where(precp_ds>0, 0)

        if save is True:
            logger.info("Saving dataset locally...")
            compress_kwargs = {"total_precipitation": 
                               {'zlib': True, 'complevel': 4}} # You can adjust 'complevel' based on your needs
            precp_ds.to_netcdf(os.path.join(config["SPI"]["ERA5"]["path"], 
                                           "era5_total_precipitation_gc.nc"),
                              encoding=compress_kwargs)
    
        return precp_ds
    
    def _process_temperature_arco(self, ds):
        logging.debug("Processing temperature data")
        temp = ds["2m_temperature"].resample(time='D')
        ds_temp_max = temp.max(dim='time')
        ds_temp_min = temp.min(dim='time')

        logging.debug("Setting nan values for temperature")
        ds_temp_max = self._set_nulls(ds_temp_max, np.nan)
        ds_temp_min = self._set_nulls(ds_temp_min, np.nan)
        return ds_temp_min, ds_temp_max
    
    def _process_evapo_arco(self, ds):
        logging.debug("Processing evaporation data")
        evap = ds["evaporation"].resample(time="D").sum()
        pot_evap = ds["potential_evaporation"].resample(time="D").sum()

        logging.debug("Setting nan values for evaporation")
        evap = self._set_nulls(evap, np.nan)
        pot_evap = self._set_nulls(pot_evap, np.nan)
        return evap, pot_evap
    
    def _load_soil_moisture(self):
        from utils.function_clns import config, prepare
        import os
        import xarray as xr
        import pandas as pd
        import glob
        from datetime import datetime
        logging.info("Loading soil moisture data")

        base_path = config["SOIL_MO"]["path"]
        years = range(datetime.strptime(config["DEFAULT"]["date_start"],"%Y-%m-%d").year, 
                      datetime.strptime(config["DEFAULT"]["date_end"],"%Y-%m-%d").year +1)

        chunks={'time': -1, "latitude": "auto", "longitude":"auto"}

        # Generate paths
        paths = []
        for year in years:
            year_path = os.path.join(base_path, f"{year}", "*.nc")
            paths.extend(glob.glob(year_path))  # Collect all matching files for each year

        ds = xr.open_mfdataset(paths, chunks=chunks)
        return prepare(ds).drop_dims(["depth", "bnds"])
    
    def _process_soil_moisture(self, target_ds):
        from utils.function_clns import hoa_bbox
        from shapely.geometry import mapping
        ds = self._load_soil_moisture()
        source_resolution = ds.rio.resolution()
        target_resolution = target_ds.rio.resolution()
        # Reproject the entire dataset
        logging.info(f"Reprojecting soil moisture data from spatial resolution of "
                     f"{source_resolution} to target grid of {target_resolution}")
        
        ds_adjusted = prepare(self._adjust_sm_coords(ds))
        ds_cropped = self._preprocess_array(ds_adjusted)
        ds_reprojected = self._reproject_odc(ds_cropped, target_ds)

        logging.debug("Setting nan values for soil moisture")
        ds_reprojected = self._set_nulls(ds_reprojected, np.nan)
        
        logger.info(f"The cropped soil moisture dataset has {ds_reprojected.sizes}")
        if isinstance(target_ds, xr.DataArray):
            target_ds = target_ds.to_dataset()
        target_ds = target_ds.assign(sm_1 = ds_reprojected["var40"], 
                                        sm_2 = ds_reprojected["var41"],
                                        sm_3 = ds_reprojected["var42"], 
                                        sm_4 = ds_reprojected["var43"])
        return target_ds


