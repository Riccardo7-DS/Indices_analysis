import os
import pandas as pd
from datetime import datetime
import xarray as xr
import numpy as np
from xarray import Dataset
from  matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

"""
In this script can be found all the reading, standard cleaning and plotting helper functions
"""

def ndvi_colormap():
    # List of upper boundaries for NDVI values (reversed order)
    vals = [-0.2, -0.1, 0.00, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

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
    cmap= ListedColormap(cols,  name='custom_colormap')
    bounds = np.array(vals)
    # Normalize the colormap
    norm = BoundaryNorm(bounds, cmap.N)

    #fig, ax = plt.subplots(figsize=(12, 1))
    #fig.subplots_adjust(bottom=0.5)
    #fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_custom),
    #             cax=ax, orientation='horizontal', label='Colorbar')
    #plt.show()
    return cmap, norm

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
    my_date_string = xr_df.attrs['EPCT_start_sensing_time']#xr_df.attrs['date_time']
    date_xr = datetime.strptime(my_date_string,'%Y%m%dT%H%M%SZ') #datetime.strptime(my_date_string, '%Y%m%d/%H:%M')
    date_xr = pd.to_datetime(date_xr)
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

def drop_water_bodies_esa(CONFIG_PATH:str,
                          dataset:xr.Dataset, 
                          var:str="ndvi") ->xr.Dataset:
    from ancillary.esa_landuse import get_level_colors, get_cover_dataset
    from utils.function_clns import config

    img_path = os.path.join(config["DEFAULT"]["images"], "chirps_esa")
    ds_cover = get_cover_dataset(CONFIG_PATH, dataset[var], img_path)
    
    water_mask = xr.where((ds_cover["Band1"]==80) | (ds_cover["Band1"]==200), 1,0)
    ds_process = ds_cover.where(water_mask==0).drop_vars("Band1")
    dataset = dataset.assign(ndvi=ds_process[var])
    return dataset

def get_missing_datarray(datarray, prediction="P1D"):
    from fusets._xarray_utils import _extract_dates, _output_dates
    datarray['time'] = datarray.indexes['time'].normalize()
    datarray = datarray.assign_coords(time = datarray.indexes['time'].normalize())
    dates = _extract_dates(datarray)
    expected_dates = _output_dates(prediction, dates[0],dates[-1])
    
    dates = [np.datetime64(i) for i in expected_dates]
    missing_dates = [i for i in dates if i not in datarray['time'].values]
    print("Missing dates are:" , missing_dates)
    lat = datarray["lat"]
    lon = datarray["lon"]
    array_zero = np.zeros((len(lat), len(lon), len(missing_dates)))
    print(array_zero)
    print(array_zero.shape)
    new_ds = xr.DataArray(array_zero,
                            coords={"lat": lat, "lon":lon, "time":missing_dates},
                            dims= ["lat","lon","time"],
                            name="ndvi"
                            )
    return new_ds