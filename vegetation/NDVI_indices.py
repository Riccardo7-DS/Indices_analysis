# ### Calculate NDVI indices



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





