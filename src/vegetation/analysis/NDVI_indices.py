# ### Calculate NDVI indices
import xarray as xr
import os

###nomral anomaly
def lta_anomaly(ds):
    climatology = ds.groupby("time.month").mean("time")  ###12 months only, average per month
    anomalies =ds.groupby("time.month") - climatology  ### group data by month and subtract the average value
    anomalies = anomalies.drop('month')
    ds = ds.assign(ndvi_lta = anomalies['ndvi'])
    return ds

### reference anomaly with max month
def ref_anomaly(ds):
    climatology = ds.groupby("time.month").max('time').drop('crs') ###12 months only, average per month
    anomalies = ds.groupby("time.month") - climatology ### group data by month and subtract the average value
    #anomalies = anomalies.drop('month')
    ds = ds.assign(ndvi_ref = anomalies['ndvi'])
    return ds

### anomaly using daily data

def daily_anomaly(ds):
    monthly = ds.resample(time='1MS').mean()  ###resample by average value per month
    upsampled_monthly = monthly.resample(time='1D').ffill()  ### upsample filling constant values per month
    anomalies = monthly - upsampled_monthly  ### calculating the anomaly
    return anomalies


#### Implementation of the formula

def lta_anomaly_(ds):
    monthly = ds['ndvi'].resample(time='1MS').mean('time')
    month_12 = ds.groupby("time.month").mean("time")
    daily = monthly.resample(time='1D').ffill()
    monthly_res = daily.resample(time='1MS').first(skipna=False)
    return (monthly - monthly_res)/ monthly_res

def ref_anomaly_(ds):
    monthly = ds['ndvi'].resample(time='1MS').mean('time')
    daily = monthly.resample(time='1D').ffill()
    monthly_res = daily.resample(time='1MS').first(skipna=False)
    return (monthly - monthly_res)/ monthly_res

### Calculation of SVI (standardized vegetation index)

def compute_svi(darray):
    gb = darray.groupby('time.dayofyear')
    clim = gb.mean(dim='time')
    std_clim = gb.std(dim='time')
    res = xr.apply_ufunc(lambda x, m, s: (x - m) / s, gb, clim, std_clim, dask='allowed')
    return res

### Calculation of VCI (vegetation condition index)
def compute_vci(darray:xr.DataArray)->xr.DataArray: 
    grouped= darray.groupby('time.dayofyear')
    min_ndvi = grouped.min('time')
    max_ndvi = grouped.max('time')
    vci = xr.apply_ufunc(lambda x, m, s: ((x - m) / (s-m))*100, grouped, min_ndvi, max_ndvi, dask='allowed')
    #vci = ((grouped- min_ndvi)/(max_ndvi-min_ndvi))*100
    return vci

def get_time(y, ds_ref):
    import pandas as pd
    time = ds_ref["time"]
    df = pd.DataFrame(time)
    df.columns= ["time"]
    df["y"] = y
    return df.set_index(["time"])

def compute_ndvi(band1:xr.DataArray, band2:xr.DataArray)->xr.DataArray:
    return (band2-band1)/(band2+band1)


if __name__=="__main__":
    from p_drought_indices.functions.function_clns import load_config
    import matplotlib.pyplot as plt
    import time 
    from p_drought_indices.functions.ndvi_functions import clean_outliers, get_missing_datarray
    import sys

    start = time.time()

    CONFIG_PATH = "config.yaml"
    config_file = load_config(CONFIG_PATH)
    chunks = {"time":-1,"lat":200, "lon":200}
    orig_ds = xr.open_dataset(os.path.join(config_file["NDVI"]["ndvi_path"],"origin_ndvi.nc"), 
                          chunks=chunks).drop_duplicates(dim=["time"])
    ### normalize dates
    orig_ds["time"] = orig_ds.indexes['time'].normalize()   
    #### clean outliers
    orig_ds = clean_outliers(orig_ds)
    #### add missing dates with zeros
    temp_ds = get_missing_datarray(orig_ds["ndvi"]).to_dataset()
    new_dataset = xr.concat([orig_ds, temp_ds], dim='time')
    new_dataset = new_dataset.sortby('time').chunk(chunks)
    
    test_ds = new_dataset.sel(lat =slice(6,6.1), lon=slice(40,40.1))
    w2s = XarrayWS(new_dataset)
    new_ds = w2s.apply_ws2doptvp("ndvi", p=0.99)
    encoding = {'ndvi': {'zlib': True, 'complevel': 5}}
    new_ds.to_netcdf(os.path.join(config_file["NDVI"]["ndvi_path"],"ndvi_smoothed_w2s_1.nc"),encoding=encoding)
    plot = False
    if plot==True:
        year = 2017
        test_ds["ndvi"].sel(time=test_ds.time.dt.year==year).isel(lat=0, lon=0).plot()
        new_ds["ndvi"].sel(time=new_ds.time.dt.year==year).isel(lat=0, lon=0).plot()
        plt.show()
        
    end = time.time()
    total_time = end - start
    print("\n The script took "+ time.strftime("%H%M:%S", \
                                                    time.gmtime(total_time)) + "to run")



