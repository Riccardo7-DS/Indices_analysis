from utils.function_clns import config, prepare, prepare_datarray, get_lat_lon_window, crop_image_left, crop_image_right
from typing import Literal, Union
import os 
import xarray as xr
import numpy as np
import logging

class PrecipDataPreparation():
    def __init__(self, args:dict,
                 variables:list,
                 precipitation_data:str="ERA5", 
                 ndvi_data:str="seviri_full_image_smoothed.zarr",
                 model:str = Literal["GWNET", "CONVLSTM"],
                 load_local_precp:bool = False,
                 interpolate:bool = False) -> None:
        
        self.model = model
        self.precp_product = precipitation_data

        if load_local_precp is True:
            self._load_local_precipitation(precipitation_data)
            precp_ds = self._preprocess_array(path = self.precp_path,
                                               filename=self.precp_filename, 
                                               interpolate=interpolate)
        else:
            precp_ds, era5_data = self._load_arco_precipitation(variables)
            precp_ds = self._preprocess_array(precp_ds, interpolate=interpolate)

            if len(variables) > 1:
                logging.info(f"Processing ancillary variables {variables}")
                precp_ds = self._transform_ancillary(precp_ds, era5_data)
            
        self.ndvi_filename = ndvi_data
        self.ndvi_path = config["DEFAULT"]["local"]
        ndvi_ds = self._load_processed_ndvi(self.ndvi_path, 
                                            self.ndvi_filename)
        ndvi_ds = self._preprocess_array(ndvi_ds)
        ndvi_ds = self._reproject_odc(ndvi_ds, precp_ds)
        
        if args.normalize is True:
            ndvi_ds, self.ndvi_scaler, precp_ds, _ = \
                self._normalize_datasets(precp_ds, ndvi_ds)
            
        self.precp_ds = self._crop_area_for_dl(precp_ds)
        self.ndvi_ds = self._crop_area_for_dl(ndvi_ds)

        if self.model == "GWNET":
            precp_ds = self.precp_ds.where(self.ndvi_ds.notnull())
            self.hydro_data = precp_ds[[var for var in precp_ds.data_vars][0]]
        else:
            self.hydro_data = self.precp_ds

    
    def _load_local_precipitation(self, precp_dataset:str):
        import os
        import re
        from utils.function_clns import config
        logging.info(f"Loading local {precp_dataset} precipitation file")

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
            raise ValueError(f"Precipitation product must be one of {list_precp_prods}")

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
        self.time_end = config['DEFAULT']['date_end']
        self.time_start = config['DEFAULT']['date_start']
        
        logging.info("Querying ARCO data from Google Cloud Storage...")
        input_data = query_arco_era5(variables)
        input_data = input_data.shift(time=1)\
            .sel(time=slice(self.time_start, self.time_end))
        logging.info("Processing input data...")
        precp_data = self._process_precp_arco(input_data)
        return precp_data, input_data
    
    def _transform_ancillary(self, precp_data:xr.DataArray, 
                        input_data:xr.Dataset)-> xr.Dataset:
        
        ds_temp_min, ds_temp_max = self._process_temperature_arco(input_data)
        evap, pot_evap = self._process_evapo_arco(input_data)

        precp_dataset = precp_data.to_dataset()
        sm_data = self._process_soil_moisture(precp_dataset)

        precp_dataset = precp_dataset.assign(evap = evap, potential_evap= pot_evap, 
                                       temp_max= ds_temp_max, temp_min = ds_temp_min,
                                       sm_1 = sm_data["var40"], sm_2 = sm_data["var41"],
                                       sm_3 = sm_data["var42"], sm_4 = sm_data["var43"])
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
                          interpolate:bool = False) -> xr.DataArray:
        from utils.function_clns import clip_file
        from dask.diagnostics import ProgressBar

        if dataset is None:
            if path is None or filename is None:
                raise ValueError(f"Either 'dataset' must be provided or both 'path'" 
                                 f"and 'filename' must be provided.")
            # Open the precipitation file with xarray
            dataset = xr.open_dataset(os.path.join(path, filename))
                
        dataset = prepare(clip_file(dataset, gdf=None))\
            .sel(time=slice(self.time_start, self.time_end))
        
        if isinstance(dataset, xr.Dataset):
            for var in dataset.data_vars:
                dataset[var] = self._set_nulls(dataset[var])
        elif isinstance(dataset, xr.DataArray):
            dataset = self._set_nulls(dataset)
        else:
            raise ValueError("The provided dataset must be in xarray format")

        if interpolate is True:
            logging.info()
            with ProgressBar():
                dataset = dataset.interpolate_na(dim="time", 
                                                 method="nearest")

        if variable is not None:
            dataset = dataset[["time", "lat", "lon", variable]]
        return dataset


    def _normalize_datasets(self, *args):
        from analysis.deep_learning.GWNET.pipeline_gwnet import StandardScaler

        if isinstance(args, xr.Dataset):
            for var in dataset.data_vars:
                scaler = StandardScaler(mean=np.nanmean(args[var]), 
                                           std=np.nanstd(args[var]))
                args[var] = scaler.transform(args)

        elif isinstance(args, xr.DataArray):
            scaler = StandardScaler(mean=np.nanmean(args), 
                                               std=np.nanstd(args))
            dataset = scaler.transform(args)
        
        return dataset, scaler
    
    def _crop_area_for_dl(self, 
                          dataset:Union[xr.DataArray, xr.Dataset]):
        
        if self.model == "GWNET":
            print("Selecting data for GCNN WaveNet")
            try:
                self.dim = config['GWNET']['dim']
                idx_lat, lat_max, idx_lon, lon_min = get_lat_lon_window(dataset, self.dim)
                sub_dataset = dataset.sel(lat=slice(lat_max, idx_lat), 
                                         lon=slice(lon_min, idx_lon))
            except IndexError:
                logging.error("The dataset {} is out of bounds when using a subset, using original product"\
                             .format(self.precp_product))
                self.dim = max(len(sub_dataset["lat"]),
                               len(sub_dataset["lon"]))

        elif self.model =="CONVLSTM":
            print("Selecting data for ConvLSTM")
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
        logging.info("Starting resampling from hourly to daily...")
        precp_ds = test_ds["total_precipitation"].resample(time="1D")\
            .sum()

        if save is True:
            logging.info("Saving dataset locally...")
            compress_kwargs = {"total_precipitation": 
                               {'zlib': True, 'complevel': 4}} # You can adjust 'complevel' based on your needs
            precp_ds.to_netcdf(os.path.join(config["SPI"]["ERA5"]["path"], 
                                           "era5_total_precipitation_gc.nc"),
                              encoding=compress_kwargs)
    
        return precp_ds
    
    def _process_temperature_arco(self, ds):
        temp = ds["2m_temperature"].resample(time='D')
        ds_temp_max = temp.max(dim='time')
        ds_temp_min = temp.min(dim='time')
        return ds_temp_min, ds_temp_max
    
    def _process_evapo_arco(self, ds):
        evap = ds["evaporation"].resample(time="D").sum()
        pot_evap = ds["potential_evaporation"].resample(time="D").sum()
        return evap, pot_evap
    
    def _load_soil_moisture(self):
        from utils.function_clns import config, subsetting_pipeline, prepare
        import os
        import xarray as xr

        path = os.path.join(config["SOIL_MO"]["path"], "*/*.nc")
        chunks={'time': 50, "latitude": 100, "longitude":100}
        ds = xr.open_mfdataset(path, chunks=chunks)
        # ds_ = subsetting_pipeline(prepare(ds).
        #                           drop_dims(["depth", "bnds"]))
        return prepare(ds).drop_dims(["depth", "bnds"])
    
    def _process_soil_moisture(self, target_ds):
        ds = self._load_soil_moisture()
        # Reproject the entire dataset
        ds_reprojected = self._reproject_odc(ds, target_ds)
        return ds_reprojected
