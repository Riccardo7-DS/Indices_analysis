from utils.function_clns import config, prepare, handle_unhandled_exception, prepare_datarray, get_lat_lon_window, crop_image_left, crop_image_right
from typing import Literal, Union
import os 
import xarray as xr
import numpy as np
import sys
from dask.diagnostics import ProgressBar
import logging
import pickle
from definitions import ROOT_DIR
logger = logging.getLogger(__name__)
sys.excepthook = handle_unhandled_exception

class PrecipDataPreparation():
    def __init__(self, 
                 args:dict,
                 variables:list,
                 precipitation_data:str="ERA5", 
                 ndvi_data:str="seviri_full_image_smoothed.zarr",
                 load_local_precp:bool = False,
                 interpolate:bool = False) -> None:
        
        cb = ProgressBar().register()
        self.basepath = config["DEFAULT"]["basepath"]
        self.time_end = config['DEFAULT']['date_end']
        self.time_start = config['DEFAULT']['date_start']
        self.model = args.model
        self.precp_product = precipitation_data
        self.ndvi_filename = ndvi_data

        precp_ds = self.process_input_vars(variables, 
                                load_local_precp, 
                                interpolate=False, 
                                save=True)
        
        ndvi_ds = self._load_processed_ndvi(self.basepath, 
                                            self.ndvi_filename)
        ndvi_ds = self._preprocess_array(ndvi_ds)
        ndvi_ds = self._reproject_odc(ndvi_ds, precp_ds) 
        
        if args.normalize is True:
            datasets, scalers = \
                self._normalize_datasets(ndvi_ds, precp_ds)
            ndvi_ds, precp_ds, self.ndvi_scaler = datasets[0], datasets[1], scalers[0]
            dump_obj = pickle.dumps(self.ndvi_scaler, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(ROOT_DIR,"../data/ndvi_scaler.pickle"), "wb") as handle:
                handle.write(dump_obj)
            
        self.precp_ds = self._crop_area_for_dl(precp_ds)
        self.ndvi_ds = self._crop_area_for_dl(ndvi_ds)

        if args.model == "GWNET":
            precp_ds = self.precp_ds.where(self.ndvi_ds.notnull())
            self.hydro_data = precp_ds[[var for var in precp_ds.data_vars][0]]
        else:
            self.hydro_data = self.precp_ds

        self.hydro_data = self._impute_values(self.hydro_data, -99)
        self.ndvi_ds = self._impute_values(self.ndvi_ds, -99)

    def _estimate_gigs(self, dataset:xr.Dataset, description:str):
        logger.info("{d} dataset has {g} GiB".format(d=description, 
                                                     g=dataset.nbytes * 1e-9))

    def process_input_vars(self, variables:list, load_local_precp:bool, 
                           interpolate:bool, save:bool=False):
        if load_local_precp is True:
            self._load_local_precipitation(self.precp_product)
            precp_ds = self._preprocess_array(path = self.precp_path,
                                               filename=self.precp_filename, 
                                               interpolate=interpolate)
        else:
            dest_path = os.path.join(self.basepath, "hydro_vars.zarr")

            if os.path.isfile(dest_path):
                logger.debug("Found zarr file in destination path. Proceeding with loading...")
                precp_ds = xr.open_zarr(dest_path)
            else:
                precp_ds, era5_data = self._load_arco_precipitation(variables)
                precp_ds = self._preprocess_array(precp_ds, 
                                                  interpolate=interpolate)
                self._estimate_gigs(precp_ds, "Precipitation only")
                logging.info(precp_ds)

                if len(variables) > 1:
                    logger.debug(f"Processing ancillary variables {variables}")
                    precp_ds = self._transform_ancillary(precp_ds, era5_data, interpolate=False)
                    self._estimate_gigs(precp_ds, "Hydro variables")
                    logging.info(precp_ds)

                if save is True:
                    logger.debug("Starting exporting processed data to zarr...")
                    out = precp_ds.to_zarr(dest_path, mode="w", compute=False)
                    res = out.compute()
                    logger.debug("Successfully saved preprocessed variables")

        return precp_ds
    

    def _impute_values(self, dataset:Union[xr.Dataset, xr.DataArray], value:float):
        return dataset.fillna(value)

    def _load_local_precipitation(self, precp_dataset:str):
        import os
        import re
        from utils.function_clns import config
        logger.debug(f"Loading local {precp_dataset} precipitation file")

        config_directories = [config['SPI']['IMERG']['path'], config['SPI']['GPCC']['path'], 
                          config['SPI']['CHIRPS']['path'], config['SPI']['ERA5']['path'], 
                          config['SPI']['MSWEP']['path'] ]
        config_dir_precp = [config['PRECIP']['IMERG']['path'],config['PRECIP']['GPCC']['path'], 
                            config['PRECIP']['CHIRPS']['path'], config['PRECIP']['ERA5']['path'],
                            config['PRECIP']['TAMSTAT']['path'],config['PRECIP']['MSWEP']['path'],
                            config["PRECIP"]["ERA5_land"]["path"]]
    
        list_precp_prods = ["ERA5", "GPCC","CHIRPS","SPI_ERA5", "SPI_GPCC","SPI_CHIRPS",
                            "ERA5_land"]

        if precp_dataset not in list_precp_prods:
            error = ValueError(f"Precipitation product must be one of {list_precp_prods}")
            logger.error(error)
            raise error

        if "SPI" in precp_dataset:
            precp_dataset = precp_dataset.replace("SPI_","")
            path = [f for f in config_directories if precp_dataset in f][0]
            late =  re.search(r'\d+', path).group()
            filename = "spi_gamma_{}".format(late)
            file = [f for f in os.listdir(path) if filename in f][0]
            self.spi_latency = late
        else:
            path = [f for f in config_dir_precp if precp_dataset in f][0]
            filename = f"{precp_dataset}_merged.nc"

        self.precp_path = path
        self.precp_filename = file
        self.time_end = config['PRECIP'][precp_dataset]['date_end']
        self.time_start = config['PRECIP'][precp_dataset]['date_start']

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
                        input_data:xr.Dataset, 
                        interpolate:bool)-> xr.Dataset:
        
        ds_temp_min, ds_temp_max = self._process_temperature_arco(input_data)
        evap, pot_evap = self._process_evapo_arco(input_data)

        precp_dataset = precp_data.to_dataset()
        sm_data = self._process_soil_moisture(precp_dataset)
        
        precp_dataset = precp_dataset.assign(
                                    evap = evap, potential_evap= pot_evap, 
                                    temp_max= ds_temp_max, temp_min = ds_temp_min,
                                    sm_1 = sm_data["var40"], sm_2 = sm_data["var41"],
                                    sm_3 = sm_data["var42"], sm_4 = sm_data["var43"])
        
        precp_dataset = precp_dataset.chunk(chunks={"time":-1, "lat":"auto", "lon":"auto"})

        return precp_dataset
    
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


    def _set_nulls(self, data):
        import numpy as np
        data = data.rio.write_nodata(np.nan, inplace=True)
        data = data.rio.write_nodata("nan", inplace=True)
        return data
    
    def _preprocess_array(self,
                          dataset: Union[xr.DataArray, xr.Dataset] = None,
                          path: str = None,
                          filename: str = None,
                          variable: str = None, 
                          interpolate:bool = False,
                          invert:bool=False) -> xr.DataArray:
        from utils.function_clns import clip_file
        from dask.diagnostics import ProgressBar

        if dataset is None:
            if path is None or filename is None:
                error = ValueError(f"Either 'dataset' must be provided or both 'path'" 
                                 f" and 'filename' must be provided.")
                logging.error(error)
                raise error
            # Open the precipitation file with xarray
            dataset = xr.open_dataset(os.path.join(path, filename))
                
        dataset = prepare(clip_file(dataset, gdf=None, invert=invert))\
            .sel(time=slice(self.time_start, self.time_end))
        
        if isinstance(dataset, xr.Dataset):
            for var in dataset.data_vars:
                dataset[var] = self._set_nulls(dataset[var])
        elif isinstance(dataset, xr.DataArray):
            dataset = self._set_nulls(dataset)
        else:
            error = ValueError("The provided dataset must be in xarray format")
            logging.error(error)
            raise error

        if interpolate is True:
            logger.info("Interpolating dataset")
            with ProgressBar():
                dataset = dataset.chunk(dict(time=-1))
                dataset = dataset.interpolate_na(dim="time", 
                                                 method="nearest")

        if variable is not None:
            dataset = dataset[["time", "lat", "lon", variable]]
        return dataset


    def _normalize_datasets(self, *datasets):
        from analysis.deep_learning.GWNET.pipeline_gwnet import StandardScaler

        normalized_datasets = []
        scalers = []

        for dataset in datasets:
            if isinstance(dataset, xr.Dataset):
                dataset_copy = dataset.copy()  # Make a copy to avoid modifying the original dataset
                for var in dataset_copy.data_vars:
                    scaler = StandardScaler(mean=np.nanmean(dataset_copy[var]), 
                                            std=np.nanstd(dataset_copy[var]))
                    dataset_copy[var] = scaler.transform(dataset_copy[var])
                normalized_datasets.append(dataset_copy)
                scalers.append(scaler)

            elif isinstance(dataset, xr.DataArray):
                dataset_copy = dataset.copy()  # Make a copy to avoid modifying the original dataset
                scaler = StandardScaler(mean=np.nanmean(dataset_copy), 
                                        std=np.nanstd(dataset_copy))
                normalized_datasets.append(scaler.transform(dataset_copy))
                scalers.append(scaler)

        return normalized_datasets, scalers
    
    def _crop_area_for_dl(self, 
                          dataset:Union[xr.DataArray, xr.Dataset]):
        
        if self.model == "GWNET":
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

        return sub_dataset
    
    def _reproject_odc(self,
                       resample_ds:Union[xr.Dataset, xr.Dataset],
                       target_ds:Union[xr.Dataset, xr.DataArray],
                       resampling:str = "bilinear"):
        
        from utils.xarray_functions import odc_reproject
        ds_repr = odc_reproject(resample_ds, target_ds, resampling=resampling)\
                .rename({"longitude": "lon", "latitude": "lat"})
        ds_repr["lat"] = target_ds["lat"]
        ds_repr["lon"] = target_ds["lon"]
        return ds_repr
    
    def _reproject_raster(self, 
                          resample_ds:xr.DataArray,
                          target_ds:Union[xr.Dataset, xr.DataArray]):
        if type(target_ds) == xr.Dataset:
            var_target = [var for var in target_ds.data_vars][0]
            target_ds = prepare(target_ds[var_target])

        # elif type(target_ds)== xr.DataArray:
            
        ds_repr = resample_ds.rio.reproject_match(
            target_ds).rename({'x':'lon','y':'lat'})
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
        return ds_temp_min, ds_temp_max
    
    def _process_evapo_arco(self, ds):
        logging.debug("Processing evaporation data")
        evap = ds["evaporation"].resample(time="D").sum()
        pot_evap = ds["potential_evaporation"].resample(time="D").sum()
        return evap, pot_evap
    
    def _load_soil_moisture(self):
        from utils.function_clns import config, subsetting_pipeline, prepare
        import os
        import xarray as xr
        logging.debug("Loading soil moisture data")

        path = os.path.join(config["SOIL_MO"]["path"], "2019/*.nc") #*
        chunks={'time': -1, "latitude": "auto", "longitude":"auto"}
        ds = xr.open_mfdataset(path, chunks=chunks)
        # ds_ = subsetting_pipeline(prepare(ds).
        #                           drop_dims(["depth", "bnds"]))
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
        
        logger.info(f"The cropped soil moisture dataset has {ds_reprojected.sizes}")
        return ds_reprojected
