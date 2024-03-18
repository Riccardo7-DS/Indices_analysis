# ### Calculate NDVI indices
import xarray as xr
import os

"""
NDVI indices calculations
"""

def lta_anomaly(ds):
    """
    nomral anomaly
    """
    climatology = ds.groupby("time.month").mean("time")  ###12 months only, average per month
    anomalies =ds.groupby("time.month") - climatology  ### group data by month and subtract the average value
    anomalies = anomalies.drop('month')
    ds = ds.assign(ndvi_lta = anomalies['ndvi'])
    return ds

def ref_anomaly(ds:xr.DataArray):
    """
    reference anomaly with max month
    """
    climatology = ds.groupby("time.month").max('time').drop('crs') ###12 months only, average per month
    anomalies = ds.groupby("time.month") - climatology ### group data by month and subtract the average value
    #anomalies = anomalies.drop('month')
    ds = ds.assign(ndvi_ref = anomalies['ndvi'])
    return ds

def daily_anomaly(ds:xr.DataArray):
    """
    anomaly using daily data
    """
    monthly = ds.resample(time='1MS').mean()  ###resample by average value per month
    upsampled_monthly = monthly.resample(time='1D').ffill()  ### upsample filling constant values per month
    anomalies = monthly - upsampled_monthly  ### calculating the anomaly
    return anomalies


#### Implementation of the formula

def lta_anomaly_(ds:xr.DataArray):
    monthly = ds['ndvi'].resample(time='1MS').mean('time')
    month_12 = ds.groupby("time.month").mean("time")
    daily = monthly.resample(time='1D').ffill()
    monthly_res = daily.resample(time='1MS').first(skipna=False)
    return (monthly - monthly_res)/ monthly_res

def ref_anomaly_(ds:xr.DataArray):
    monthly = ds['ndvi'].resample(time='1MS').mean('time')
    daily = monthly.resample(time='1D').ffill()
    monthly_res = daily.resample(time='1MS').first(skipna=False)
    return (monthly - monthly_res)/ monthly_res


def compute_svi(darray:xr.DataArray):
    """
    Calculation of SVI (standardized vegetation index)
    """
    gb = darray.groupby('time.dayofyear')
    clim = gb.mean(dim='time')
    std_clim = gb.std(dim='time')
    res = xr.apply_ufunc(lambda x, m, s: (x - m) / s, gb, clim, std_clim, dask='allowed')
    return res

def compute_vci(darray:xr.DataArray)->xr.DataArray: 
    """
    Calculation of VCI (vegetation condition index)
    """
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

def compute_ndvi(band1:xr.DataArray, 
                 band2:xr.DataArray)->xr.DataArray:
    return (band2-band1)/(band2+band1)

"""
Outcome metrics
"""

def abs_diff(ds:xr.DataArray, dim="time", lead=1)->xr.DataArray:
    """
    Metric for the absolute difference between consecutive records
    """
    diff = ds.diff(dim=dim, n=lead)
    return abs(diff)


def autocorr(ds, dim='time', nlags=None, skipna:bool=False):
    from tqdm.auto import tqdm
    """
    Adapted from esmtools library https://github.com/esm-tools/esm_tools
    Compute the autocorrelation function of a time series to a specific lag.

    .. note::

        The correlation coefficients presented here are from the lagged
        cross correlation of ``ds`` with itself. This means that the
        correlation coefficients are normalized by the variance contained
        in the sub-series of ``x``. This is opposed to a true ACF, which
        uses the entire series' to compute the variance. See
        https://stackoverflow.com/questions/36038927/
        whats-the-difference-between-pandas-acf-and-statsmodel-acf

    Args:
      ds (xarray object): Dataset or DataArray containing the time series.
      dim (str, optional): Dimension to apply ``autocorr`` over. Defaults to 'time'.
      nlags (int, optional): Number of lags to compute ACF over. If None,
                            compute for length of `dim` on `ds`.

    Returns:
      Dataset or DataArray with ACF results.

    """
    if nlags is None:
        nlags = ds[dim].size - 2

    acf = []
    # The factor of 2 accounts for fact that time series reduces in size for
    # each lag.
    for i in tqdm(range(1, nlags+1), desc="Processing lags..."):
        res = corr(ds, ds, lead=i, dim=dim, skipna=skipna)
        acf.append(res)
    acf = xr.concat(acf, dim='lead')
    return acf


def corr(x, y, dim='time', lead=0, return_p=False, skipna:bool=False):
    import numpy as np
    """Computes the Pearson product-moment coefficient of linear correlation.

    Args:
        x, y (xarray object): Time series being correlated.
        dim (str, optional): Dimension to calculate correlation over. Defaults to
            'time'.
        lead (int, optional): If lead > 0, ``x`` leads ``y`` by that many time steps.
            If lead < 0, ``x`` lags ``y`` by that many time steps. Defaults to 0.
        return_p (bool, optional). If True, return both the correlation coefficient
            and p value. Otherwise, just returns the correlation coefficient.

    Returns:
        corrcoef (xarray object): Pearson correlation coefficient.
        pval (xarray object): p value, if ``return_p`` is True.

    """

    def _lag_correlate(x, y, dim, lead, return_p, skipna):
        """Helper function to shift the two time series and correlate."""
        from xskillscore import pearson_r, pearson_r_p_value
        N = x[dim].size
        normal = x.isel({dim: slice(0, N - lead)})
        shifted = y.isel({dim: slice(0 + lead, N)})
        # Align dimensions for xarray operation.
        shifted[dim] = normal[dim]
        corrcoef = pearson_r(normal, shifted, dim, skipna=skipna)
        if return_p:
            pval = pearson_r_p_value(normal, shifted, dim, skipna=skipna)
            return corrcoef, pval
        else:
            return corrcoef

    # Broadcasts a time series to the same coordinates/size as the grid. If they
    # are both grids, this function does nothing and isn't expensive.
    x, y = xr.broadcast(x, y)

    # I don't want to guess coordinates for the user.
    if (dim not in list(x.coords)) or (dim not in list(y.coords)):
        raise ValueError(
            f'Make sure that the dimension {dim} has coordinates. '
            "`xarray` apply_ufunc alignments break when they can't reference "
            " coordinates. If your coordinates don't matter just do "
            ' `x[dim] = np.arange(x[dim].size).'
        )

    N = x[dim].size
    assert (
        np.abs(lead) <= N
    ), f'Requested lead [{lead}] is larger than dim [{dim}] size.'

    if lead < 0:
        return _lag_correlate(y, x, dim, np.abs(lead), return_p, skipna)
    else:
        return _lag_correlate(x, y, dim, lead, return_p, skipna)

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



