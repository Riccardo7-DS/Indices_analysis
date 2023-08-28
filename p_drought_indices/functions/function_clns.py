import yaml
from shapely.geometry import Polygon, mapping
import geopandas as gpd
import shapely
import pandas as pd
import xarray as xr

def load_config(CONFIG_PATH):
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)
        return config

def cut_file(xr_df, gdf):
    xr_df.rio.set_spatial_dims(x_dim='lat', y_dim='lon', inplace=True)
    xr_df.rio.write_crs("epsg:4326", inplace=True)
    clipped = xr_df.rio.clip(gdf.geometry.apply(mapping), gdf.crs, drop=True)
    return clipped

def subsetting_pipeline(CONFIG_PATH, xr_df, countries = ['Ethiopia','Kenya', 'Somalia'], invert=True):
    config = load_config(CONFIG_PATH)
    shapefile_path = config['SHAPE']['africa']
    gdf = gpd.read_file(shapefile_path)
    subset = gdf[gdf.ADM0_NAME.isin(countries)]
    if invert==True:
        subset = subset['geometry'].map(lambda polygon: shapely.ops.transform(lambda x, y: (y, x), polygon))
    return cut_file(xr_df, subset)

def print_raster(raster):
    print(
        f"shape: {raster.rio.shape}\n"
        f"resolution: {raster.rio.resolution()}\n"
        f"bounds: {raster.rio.bounds()}\n"
        #f"sum: {raster.sum().item()}\n"
        f"CRS: {raster.rio.crs}\n"
    )

def check_missing_series(datarray, prediction="P1D") :
    from fusets._xarray_utils import _extract_dates, _output_dates, _topydate
    dates = _extract_dates(datarray)
    expected_dates = _output_dates(prediction,dates[0],dates[-1])
    array_time = datarray.indexes['time'].normalize()
    [print(i) for i in expected_dates if i not in dates]

def check_previous_and_impute(ds, start_year, end_year):
    times = pd.date_range(f"{str(start_year)}-01-01", f"{str(end_year)}-12-31")
    ds = ds.reindex({"time":times})
    ds = ds.bfill("time")
    ds = ds.ffill("time")
    #from datetime import timedelta, datetime
    #from fusets._xarray_utils import _extract_dates
    #ds_dates = _extract_dates(datarray)
    #list_dates = [t for t in times if t not in ds_dates]
    #print("There are {} missing dates".format(len(list_dates)))
#
    #new_xr = ds.copy()
    #for date in list_dates:
    #    date_new = date + timedelta(days=1)
    #    idx = times.index(date_new)
    #    string_time = datetime.strftime(times[idx], "%Y-%m-%d")
    #    data_2 = ds.sel(time=string_time)
    #    data_2["time"] = date
    #    new_xr = xr.concat([new_xr, data_2], dim="time")
    
    return ds