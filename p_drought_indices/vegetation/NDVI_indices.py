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


#if __name__=="__main___":
#    import os
#    import xarray as xr
#
#    chunks = {"lat": -1, "lon": -1, "time": 12}
#    base_dir = r'D:\shareVM\MSG\msg_data\processed'
#    path = os.path.join(base_dir, 'mask_clouds_1D_processed_msg.nc')
#    ds = xr.open_dataset(path, chunks = chunks)
