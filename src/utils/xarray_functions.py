import os
import pandas as pd
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
from xarray import Dataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Literal
import logging
logger = logging.getLogger(__name__)
"""
In this script can be found all the reading, standard cleaning and plotting helper functions
"""

def ndvi_colormap(colormap: Literal["diverging","sequential"]):
    from  matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap


    if colormap == "diverging":

        # List of corresponding colors in hexadecimal format (reversed order)
        cols = [
            "#c0c0c0",
            "#954535",
            "#FF0000",
            "#E97451",
            "#FFA500",
            "#FFD700",
            "#DFFF00",
            "#CCFF00",
            "#00FF00",
            "#00BB00",
            "#008800",
            "#006600",
            "#7F00FF"
        ]

    elif colormap == "sequential":
        cols = ["#ffffe5","#f7fcb9","#d9f0a3","#addd8e","#78c679","#41ab5d",
                "#238443","#006837","#004529"]

    cmap_custom = ListedColormap(cols)
    return cmap_custom

def dataset_set_nulls(dataset:Union[xr.Dataset, xr.DataArray], value=np.nan):
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

def downsample(ds, time='5D'):
    monthly = ds.resample(time=time, skipna=True).mean() #### Change here to change the timeframe over which to make the data imputation
    return monthly

def clean_ndvi(ds):
    ds = ds.where('ndvi'!=0.00)
    return ds

def clean_outliers(dataset:Dataset):
    ds = dataset.where((dataset["ndvi"]<=1) & (dataset["ndvi"]>=-1))
    return ds.dropna(dim="lon", how="all")

def clean_water(ds, ds_cl):
    ds_cl['time'] = ds_cl.indexes['time'].normalize()
    ds['time'] = ds.indexes['time'].normalize()
    return ds.where(ds_cl==1)

def compute_ndvi(xr_df):
    return xr_df.assign(ndvi=(
        xr_df['channel_2'] - xr_df['channel_1']) / (xr_df['channel_2'] + xr_df['channel_1']))

def shift_xarray_overtime(ds:xr.Dataset, diff:str):
    import re
    import numpy as np
    import pandas  as pd

    if "D" in diff:
        unit = "D"
    else:
        err = NotImplementedError(f"Only 'D' day unit is implemented")
    
    integer = int(re.search(r'\d+', diff).group())
    time = ds["time"].values
    time_min = time.min()
    time_max = time.max()
    delta = np.timedelta64(integer, unit)
    new_range = pd.date_range(time_min + delta, time_max+ delta, freq="1D")

    return ds.assign_coords(time=new_range)    

def shift_time(ds:xr.Dataset, 
               time_shift:Union[None, timedelta]=None, 
               add_last_day:bool=False):
    import pandas as pd
    from datetime import timedelta

    if time_shift is None:
        time_shift = timedelta(days=1)

    start_date = pd.to_datetime(ds["time"].min().values) - time_shift
    end_date = pd.to_datetime(ds["time"].max().values) - time_shift
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    ds["time"] = date_range

    if add_last_day is True:
        # Step 1: Select the last time step
        last_time_step = ds.isel(time=-1)
        # Step 2: Create a new time coordinate
        # Assuming the time coordinate is a datetime64 type, and adding one day to the last time step
        new_time = pd.to_datetime(ds.time.values[-1]) + pd.Timedelta(days=1)
        # Step 3: Create a new DataArray with the new time step
        new_data = last_time_step.expand_dims(time=[new_time])
        # Step 4: Concatenate the new DataArray to the original Dataset along the time dimension
        return xr.concat([ds, new_data], dim='time')

    else:
        return ds


def set_negative_zero(data_array:xr.DataArray):
    return xr.where(data_array < 0, 0, data_array)

def get_irradiances(satellite):
    if satellite == 'MSG2':
        irradiance_vis6 = 65.2065
        irradiance_vis8 = 73.0127
        
    elif satellite == 'MSG1':
        irradiance_vis6 =65.2296 
        irradiance_vis8 =73.1869
    
    elif satellite == 'MSG3':
        irradiance_vis6 =65.5148 
        irradiance_vis8 =73.1807
        
    elif satellite == 'MSG4':
        irradiance_vis6 =65.2656
        irradiance_vis8 =73.1692

    return irradiance_vis6, irradiance_vis8

def compute_ndvi_corr(xr_df, irradiance_vis6, irradiance_vis8):
    return xr_df.assign(ndvi=(xr_df['channel_2']*irradiance_vis6 - xr_df['channel_1']*irradiance_vis8) / (xr_df['channel_2']*irradiance_vis6 + xr_df['channel_1']*irradiance_vis8))


def compute_radiance(xr_df):
    satellite = xr_df.attrs['EPCT_product_name'][:4]
    if satellite == 'MSG2':
        xr_df['channel_1'] = xr_df['channel_1']/65.2065
        xr_df['channel_2'] = xr_df['channel_2']/73.0127
        
    elif satellite == 'MSG1':
        xr_df['channel_1'] = xr_df['channel_1']/65.2296 
        xr_df['channel_2'] = xr_df['channel_2']/73.1869
    
    elif satellite == 'MSG3':
        xr_df['channel_1'] = xr_df['channel_1']/65.5148 
        xr_df['channel_2'] = xr_df['channel_2']/73.1807
        
    elif satellite == 'MSG4':
        xr_df['channel_1'] = xr_df['channel_1']/65.2656
        xr_df['channel_2'] = xr_df['channel_2']/73.1692
    
    else:
        print('This product doesn\'t contain MSG1, MSG2, MSG3, MSG4 Seviri')
    
    return xr_df

def convert_ndvi_tofloat(datarray:xr.DataArray):
    datarray = xr.where(datarray==255, np.NaN, datarray)
    ndvi = -0.08+(datarray*0.004)
    return ndvi

def add_time(xr_df):
    if 'EPCT_start_sensing_time' in xr_df.attrs:
        my_date_string = xr_df.attrs['EPCT_start_sensing_time']#xr_df.attrs['date_time']
        date_xr = datetime.strptime(my_date_string,'%Y%m%dT%H%M%SZ') #datetime.strptime(my_date_string, '%Y%m%d/%H:%M')
        date_xr = pd.to_datetime(date_xr)
    else:
        import re
        string_time = xr_df.encoding["source"].split("/")[-1]
        pattern = r'(\d{4}\d{2}\d{2})(\d{2}\d{2}\d{2})'

        # Use re.search() to find the pattern in the string
        match = re.search(pattern, string_time)
        if match:
            start_date = match.group(1)
            start_time = match.group(2)
            date_xr = pd.to_datetime(start_date + start_time, format='%Y%m%d%H%M%S')
        else:
            print("Could not parse time in xarray dataset {}".format(string_time))
            return xr_df
    
    xr_df = xr_df.assign_coords(time=date_xr)
    xr_df = xr_df.expand_dims(dim="time")
    return xr_df

def add_time_tiff(ds):
    import pandas as pd
    from utils.xarray_functions import add_time
    from datetime import datetime
    time_str = ds.encoding["source"].split("/")[-1][:-4]
    time_ = time_str.replace("_","-")
    date_xr = datetime.strptime(time_,'%Y-%m-%d') #datetime.strptime(my_date_string, '%Y%m%d/%H:%M')
    date_xr = pd.to_datetime(date_xr)
    xr_ds = ds.assign_coords(time=date_xr)
    xr_ds = xr_ds.expand_dims(dim="time")
    return xr_ds

def dataset_to_dekad(dataset):
    import pandas as pd
    from datetime import datetime

    def get_next_threshold_day(current_day, threshold_days= [1, 11, 21]):
        for i, day in enumerate(threshold_days):
            if current_day < day:
                return threshold_days[i-1], threshold_days[i]
        return threshold_days[-1], threshold_days[0]

    def group_into_dekads(date_list:list):

        result = []
        correct_dates = []

        # Sort the list of dates to ensure correct grouping
        sorted_dates = [pd.to_datetime(t) for t in sorted(date_list)]
        last_date = sorted_dates[-1]

        # Group dates into dekads
        current_dekad = []
        current_threshold, next_threshold = get_next_threshold_day(sorted_dates[0].day)

        for idx, date in enumerate(sorted_dates):  # Iterate up to the second-to-last element
            current_dekad.append(date)

            if date is not last_date:
                next_day = sorted_dates[idx+1].day

            if next_day <= 21:
                if next_day >= next_threshold or date==last_date:
                    # print("Next day:", next_day, "next threshold:", next_threshold, current_dekad)
                    result.append(current_dekad)
                    str_time = f"{date.year}-{date.month}-{current_threshold}"
                    correct_dates.append(datetime.strptime(str_time, "%Y-%m-%d"))
                    current_threshold, next_threshold = get_next_threshold_day(next_day)
                    current_dekad = []

        return result, correct_dates

    time_values = dataset["time"].values
    list_dates, correct_dates = group_into_dekads(time_values)

    final_dataset = None

    for idx, dates in enumerate(list_dates):
        # Perform the operation on the subset of the dataset for the current date
        temp_ds = dataset.sel(time=dates).max(["time"])
        temp_ds["time"] = correct_dates[idx]

        # Append the resulting dataset to the final dataset
        if final_dataset is None:
            final_dataset = temp_ds
        else:
            final_dataset = xr.concat([final_dataset, temp_ds], dim="time")

    # Optionally, you can sort the final dataset by time
    final_dataset = final_dataset.sortby("time")

    return final_dataset


def swath_to_grid(lat, lon):
    from pyresample import geometry
    import pyproj
    proj_id = 'laea'
    datum = 'WGS84'
    lat_0_txt = lat.min()
    lon_0_txt= lon.min()

    # arguments needed by pyproj
    #
    area_dict = dict(datum=datum,
                    lat_0=lat_0_txt,
                    lon_0=lon_0_txt,
                    proj=proj_id,units='m')
    #
    # create the projection
    #
    prj=pyproj.Proj(area_dict)
    x, y = prj(lon.reshape(-1,1), lat.reshape(-1,1))
    #
    # find the corners in map space
    #
    minx, maxx=np.min(x),np.max(x)
    miny, maxy=np.min(y),np.max(y)
    #
    # back transform these to lon/lat
    #
    area_extent=[minx,miny,maxx,maxy]
    x_pixel=1000
    y_pixel=1000
    xsize=int((area_extent[2] - area_extent[0])/x_pixel)
    ysize=int((area_extent[3] - area_extent[1])/y_pixel)

    fill_value=-9999.
    area_id = 'granule'
    area_name = 'swath granule'
    #
    # here are all the arguments pyresample needs to regrid the swath
    #
    area_def_args= dict(area_id=area_id,
                        area_name=area_name,
                        proj_id=proj_id,
                        area_dict=area_dict,
                        xsize=xsize,
                        ysize=ysize,
                        area_extent=area_extent)

    area_def = geometry.AreaDefinition(area_id, 
                                       area_name, 
                                       proj_id, 
                                       area_dict, 
                                       xsize, 
                                       ysize, 
                                       area_extent)

    swath_def = geometry.SwathDefinition(lons=lon, lats=lat)

    print('\ndump area definition:\n{}\n'.format(area_def))
    print('\nx and y pixel dimensions in meters:\n{}\n{}\n'.format(area_def.pixel_size_x,area_def.pixel_size_y))
    return area_def, swath_def, area_dict, area_extent

from odc.geo.geobox import GeoBox
from typing import Union
from odc.geo import resyx_, wh_
from odc.geo.crs import CRS

def get_reproject_info(geobox):
    return {
        "crs": f"EPSG:{str(geobox.crs.epsg)}",
        "crsTransform": list(geobox.transform)[:6]
    }

def geobox_from_rio(xds: Union[xr.Dataset, xr.DataArray]) -> GeoBox:
    """This function retrieves the geobox using rioxarray extension.

    Parameters
    ----------
    xds: :obj:`xarray.DataArray` or :obj:`xarray.Dataset`
        The xarray dataset to get the geobox from.

    Returns
    -------
    :obj:`odc.geo.geobox.GeoBox`

    """
    height, width = xds.rio.shape
    try:
        transform = xds.rio.transform()
    except AttributeError:
        transform = xds[xds.rio.vars[0]].rio.transform()
    return GeoBox(
        shape=wh_(width, height),
        affine=transform,
        crs=CRS(xds.rio.crs.to_wkt()),
    )

def odc_reproject(ds, target_ds, resampling:str):
    from odc.geo.xr import xr_reproject
    if target_ds.odc.geobox.crs.geographic:
        ds = ds.rename({"lon": "longitude", "lat": "latitude"})
    return xr_reproject(ds, target_ds.odc.geobox, 
                        resampling=resampling, 
                        resolution="same")

def plot_swath_basemap(result, area_dict, 
                       area_extent):
    
    from matplotlib import cm
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.basemap import Basemap
    import pyresample
    import pyproj

    prj=pyproj.Proj(area_dict)

    llcrnrlon,llcrnrlat=prj(area_extent[0],area_extent[1],inverse=True)
    urcrnrlon,urcrnrlat=prj(area_extent[2],area_extent[3],inverse=True)

    x_pixel=1.3e3
    y_pixel=1.3e3
    xsize=int((area_extent[2] - area_extent[0])/x_pixel)
    ysize=int((area_extent[3] - area_extent[1])/y_pixel)
    #
    #  here's the dictionary we need for basemap
    #
    a, b = pyresample.plot.ellps2axis('wgs84')
    rsphere = (a, b)
    basemap_args=dict()
    basemap_args['rsphere'] = rsphere
    basemap_args['llcrnrlon'] = llcrnrlon
    basemap_args['llcrnrlat'] = llcrnrlat
    basemap_args['urcrnrlon'] = urcrnrlon
    basemap_args['urcrnrlat'] = urcrnrlat
    basemap_args['projection'] = area_dict['proj']
    basemap_args['lat_0']=  area_dict["lat_0"]
    basemap_args['lon_0']= area_dict["lon_0"]
    print('image will be {} columns x {} rows'.format(xsize,ysize))

    cmap=cm.autumn  #see http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
    cmap.set_over('w')
    cmap.set_under('b',alpha=0.2)
    cmap.set_bad('0.75') #75% grey

    plt.close('all')
    fig,ax = plt.subplots(1,1, figsize=(12,12))
    #
    # add the resolutiona and axis in separately, so we can
    # change in other plots
    #
    basemap_kws=dict(resolution='c',ax=ax)
    basemap_kws.update(basemap_args)
    bmap=Basemap(**basemap_kws)
    print('here are the basemap keywords: ',basemap_kws)
    num_meridians=180
    num_parallels = 90
    vmin=None; vmax=None
    col = bmap.imshow(result, origin='upper',cmap=cmap)
    lon_sep, lat_sep = 5,5
    parallels = np.arange(-90, 90, lat_sep)
    meridians = np.arange(0, 360, lon_sep)
    bmap.drawparallels(parallels, labels=[1, 0, 0, 0],
                           fontsize=10, latmax=90)
    bmap.drawmeridians(meridians, labels=[0, 0, 0, 1],
                           fontsize=10, latmax=90)
    bmap.drawcoastlines()
    colorbar=fig.colorbar(col, shrink=0.5, pad=0.05,extend='both')
    colorbar.set_label('Cloud Mask',rotation=-90,verticalalignment='bottom')
    _=ax.set(title='HOA')
    plt.show()

def _extract_dates(array):
    time_coords = [c for c in array.coords.values() if c.dtype.type == np.datetime64]
    if len(time_coords) == 0:
        raise ValueError(
            "Whittaker expects an input with exactly one coordinate of type numpy.datetime64, which represents the time dimension, but found none."
        )
    if len(time_coords) > 1:
        raise ValueError(
            f"Whittaker expects an input with exactly one coordinate of type numpy.datetime64, which represents the time dimension, but found multiple: {time_coords}"
        )
    dates = time_coords[0]
    assert dates.dtype.type == np.datetime64

    dates = list(dates.values)
    dates = [_topydate(d) for d in dates]
    return dates

def _topydate(t):
    return datetime.utcfromtimestamp((t - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s"))

def _output_dates(prediction_period, start_date, end_date):
    period = pd.Timedelta(prediction_period)
    range = pd.date_range(start_date, end_date, freq=period)
    return [_topydate(d) for d in range.values]

"""
Xarray processing functions
"""

def process_ndvi(base_dir, file):
    with xr.open_dataset(os.path.join(base_dir, file)) as ds:
        data = ds.load()
        xr_df = data.drop('channel_3')
        xr_df = add_time(data)
        xr_df = compute_radiance(xr_df)
        xr_df = xr_df.assign(ndvi=(xr_df['channel_2'] - xr_df['channel_1']) / (xr_df['channel_2'] + xr_df['channel_1']))
        xr_df.to_netcdf(os.path.join(base_dir,'processed', file)) 
        xr_df.close()

def drop_water_bodies_esa(
                          dataset:xr.Dataset, 
                          var:str="ndvi") ->xr.Dataset:
    from ancillary.esa_landuse import get_level_colors, get_cover_dataset
    from utils.function_clns import config

    img_path = os.path.join(config["DEFAULT"]["images"], "chirps_esa")
    ds_cover = get_cover_dataset(dataset[var], img_path)
    
    water_mask = xr.where((ds_cover["Band1"]==80) | (ds_cover["Band1"]==200), 1,0)
    ds_process = ds_cover.where(water_mask==0).drop_vars("Band1")
    dataset = dataset.assign(ndvi=ds_process[var])
    return dataset

def get_missing_datarray(datarray, prediction="P1D", output_dataset:bool=False):
    datarray['time'] = datarray.indexes['time'].normalize()
    datarray = datarray.assign_coords(time = datarray.indexes['time'].normalize())
    dates = _extract_dates(datarray)
    expected_dates = _output_dates(prediction, dates[0],dates[-1])
    
    dates = [np.datetime64(i) for i in expected_dates]
    missing_dates = [i for i in dates if i not in datarray['time'].values]
    logger.info(f"Missing dates are: {missing_dates}")

    if output_dataset is True:
        lat = datarray["lat"]
        lon = datarray["lon"]
        array_empty = np.full((len(missing_dates), len(lat), len(lon)),  np.nan, np.float32)
        new_ds = xr.DataArray(array_empty,
                                coords={"time":missing_dates, "lat": lat, "lon":lon},
                                dims= ["time","lat","lon"],
                                name="ndvi"
                                )
        return new_ds
    
def extend_dataset_over_time(datarray:xr.Dataset, 
                             interpolation_dim:str = "time",
                             interpolation_method:str="linear"):
    new_da = get_missing_datarray(datarray, output_dataset = True)
    new_da = xr.concat([new_da, datarray], dim="time")
    new_da = new_da.sortby("time").chunk(dict(time=-1))
    return new_da.interpolate_na(dim=interpolation_dim, method=interpolation_method)