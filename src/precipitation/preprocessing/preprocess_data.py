from utils.function_clns import config, subsetting_pipeline, prepare, prepare_datarray, get_lat_lon_window, crop_image_left, crop_image_right
from typing import Literal, Union
import os 
import xarray as xr
from loguru import logger
import numpy as np

class PrecipDataPreparation():
    def __init__(self, args:dict,
                 variables:list,
                 precipitation_data:str="ERA5", 
                 ndvi_data:str="ndvi_smoothed_w2s.nc",
                 model:str = Literal["GWNET", "CONVLSTM"],
                 load_local_precp:bool = False) -> None:
        
        self.model = model
        self.precp_product = precipitation_data

        if load_local_precp is True:
            self._load_local_precipitation(precipitation_data)
            precp_ds = self._preprocess_array(args, path = self.precp_path,
                                               filename=self.precp_filename)
        else:
            precp_ds, era5_data = self._load_arco_precipitation(variables)
            precp_ds = self._preprocess_array(args, dataset = precp_ds)

            if len(variables) > 1:
                logger.info(f"Processing ancillary variables {variables}")
                precp_ds = self._transform_ancillary(precp_ds, era5_data)
            
        self.ndvi_filename = ndvi_data
        self.ndvi_path = config['NDVI']['ndvi_path']
        ndvi_ds = self._preprocess_array(args, path = self.ndvi_path,
                                              filename=self.ndvi_filename, 
                                              variable="ndvi")
        
        if args.normalize is True:
            ndvi_ds, self.ndvi_scaler, precp_ds, _ = \
                self._normalize_datasets(precp_ds, ndvi_ds)
            
        precp_ds = self._crop_area_for_dl(precp_ds)
        self.ndvi_ds = self._reproject_raster(precp_ds, ndvi_ds["ndvi"])

        if self.model == "GWNET":
            precp_ds.where(self.ndvi_ds.notnull())
            self.hydro_data = precp_ds[[var for var in precp_ds.data_vars][0]]
        else:
            self.hydro_data = self.precp_ds

    
    def _load_local_precipitation(self, precp_dataset:str):
        import os
        import re
        from utils.function_clns import config
        logger.info(f"Loading local {precp_dataset} precipitation file")

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
        
        logger.info("Querying ARCO data from Google Cloud Storage...")
        input_data = query_arco_era5(variables, subset=False)
        input_data = input_data.shift(time=1)\
            .sel(time=slice(self.time_start, self.time_end))
        logger.info("Processing input data...")
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
    
    def _preprocess_array(self, args: dict,
                          dataset: Union[xr.DataArray, xr.Dataset] = None,
                          path: str = None,
                          filename: str = None,
                          variable: str = None) -> xr.DataArray:

        if dataset is None and (path is None or filename is None):
            raise ValueError("Either 'dataset' must be provided or both 'path' and 'filename' must be provided.")

        if dataset is None:
            # Open the precipitation file with xarray
            dataset = prepare(subsetting_pipeline(
                xr.open_dataset(os.path.join(path, filename)),
                countries=args.country, regions=args.region))\
                .sel(time=slice(self.time_start, self.time_end))
            
        else:
            # Open the precipitation file with xarray
            dataset = prepare(subsetting_pipeline(
                dataset,
                countries=args.country, regions=args.region))\
                .sel(time=slice(self.time_start, self.time_end))           

        if type(dataset) == xr.Dataset:
            for var in dataset.data_vars:
                # dataset[var] = prepare_datarray(dataset[var], 
                #                              interpolate=False)
                dataset[var] = dataset[var].rio.write_nodata(np.NaN)

        elif type(dataset)== xr.DataArray:
            # dataset = prepare_datarray(dataset, interpolate=False)
            dataset = dataset.rio.write_nodata(np.NaN)
            
        else:
            raise ValueError("The provided dataset must be in xarray format")
        
        if variable is  not None:
            dataset = dataset[["time", "lat", "lon", variable]]
            # variable = [var for var in dataset.data_vars][0]

        return dataset
    
    def _normalize_datasets(self, *args):
        from analysis.deep_learning.GWNET.pipeline_gwnet import StandardScaler

        if type(args) == xr.Dataset:
            for var in dataset.data_vars:
                scaler = StandardScaler(mean=np.nanmean(args[var]), 
                                           std=np.nanstd(args[var]))
                args[var] = scaler.transform(args)

        elif type(dataset)== xr.DataArray:
            scaler = StandardScaler(mean=np.nanmean(args), 
                                               std=np.nanstd(args))
            dataset = scaler.transform(args)
        
        return dataset, scaler
    
    def _crop_area_for_dl(self, dataset:Union[xr.DataArray, xr.Dataset]):
        
        if self.model == "GWNET":
            print("Selecting data for GCNN WaveNet")
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
            print("Selecting data for ConvLSTM")
            self.dim = config["CONVLSTM"]["dim"]
            idx_lat, lat_max, idx_lon, lon_min = crop_image_left(dataset, self.dim)
            sub_dataset = dataset.sel(lat=slice(lat_max, idx_lat), lon=slice(lon_min, idx_lon))

        return sub_dataset
    
    def _reproject_raster(self, target_ds:Union[xr.Dataset, xr.DataArray], 
                          resample_ds:xr.DataArray):
        if type(target_ds) == xr.Dataset:
            var_target = [var for var in target_ds.data_vars][0]
            target_ds = prepare(target_ds[var_target])

        # elif type(target_ds)== xr.DataArray:
            
        ds = resample_ds.rio.reproject_match(
            target_ds).rename({'x':'lon','y':'lat'})
        return prepare(ds)
    
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
        logger.info("Starting resampling from hourly to daily...")
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
        ds_ = subsetting_pipeline(prepare(ds).
                                  drop_dims(["depth", "bnds"]))
        return ds_
    
    def _process_soil_moisture(self, target_ds):
        import xesmf as xe
        ds = self._load_soil_moisture()
        # Create a regridder using xesmf
        regridder = xe.Regridder(ds, target_ds, 'bilinear')
        # Reproject the entire dataset
        ds_reprojected = regridder(ds)
        return ds_reprojected
