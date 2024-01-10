import xarray as xr
from loguru import logger
from precipitation.data_collection.era5_daily_data import query_arco_era5, process_era5_precp
import os 


"""
Pipeline to collect Hydrological variables
"""

class InputHydroVariables:
    def __init__(self, variables:list, start_date:str, end_date:str) -> None:
        from loguru import logger
        assert variables is not None, "Variables cannot be null, provide a list of variables"
        self.start_date = start_date
        self.end_date = end_date
        logger.info("Querying ARCO data from Google Cloud Storage...")
        input_data = query_arco_era5(variables)
        input_data = input_data.shift(time=1)\
            .sel(time=slice(start_date, end_date))
        logger.info("Processing input data...")
        precp_data = self._process_precp_arco(input_data)
        ds_temp_min, ds_temp_max = self._process_temperature_arco(input_data)
        evap, pot_evap = self._process_evapo_arco(input_data)

        precp_dataset = precp_data.to_dataset()
        sm_data = self._process_soil_moisture(precp_dataset)

        precp_dataset = precp_dataset.assign(evap = evap, potential_evap= pot_evap, 
                                       temp_max= ds_temp_max, temp_min = ds_temp_min,
                                       sm_1 = sm_data["var40"], sm_2 = sm_data["var41"],
                                       sm_3 = sm_data["var42"], sm_4 = sm_data["var43"])
        self.data = precp_dataset


    def _process_precp_arco(self, test_ds:xr.DataArray, save:bool=False):
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
        evap = ds["evaporation"].resample(time="1D")\
            .sum()
        pot_evap = ds["potential_evaporation"].resample(time="1D")\
            .sum()
        return evap, pot_evap
    
    def _load_soil_moisture(self):
        from utils.function_clns import config, subsetting_pipeline, prepare
        import os
        import xarray as xr

        path = os.path.join(config["SOIL_MO"]["path"], "*/*.nc")
        chunks={'time': -1, "latitude": "100MB", "longitude":"100MB"}
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
    

def hydro_zarr_pipeline(dir:str=None, start_date:str = "2005-01-01", 
                        end_date:str="2020-12-31", 
                        chunks:dict= {'time': -1, "lat": 100, "lon":100}):
    
    import xarray_beam as xbeam
    import apache_beam as beam
    from utils.function_clns import config
    import xarray as xr
    import warnings
    warnings.filterwarnings('ignore')

    Era5variables = ["potential_evaporation", "evaporation",
                 "2m_temperature","total_precipitation"]

    logger.info("Starting querying and preprocessing hydrological variables...")
    X_data = InputHydroVariables(Era5variables, 
                             start_date, end_date)
    
    logger.info("Converting variables to xarray beam...")
    dataset = xbeam.Dataset.from_xarray(X_data.data, 
                                        chunks=chunks)
    
    if dir is None:
        temp_dir = config["DEFAULT"]["train_data"]
    else:
        temp_dir = dir

    logger.info("Storing dataset to zarr...")
    to_zarr = dataset.to_zarr(temp_dir)
    with beam.Pipeline() as p:
      p |= to_zarr
