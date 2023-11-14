import os
import pandas as pd
import time
import geopandas as gpd
from shapely.geometry import Polygon, mapping
from datetime import datetime
import xarray as xr
import numpy as np
from xarray import DataArray, Dataset
from  matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np






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

def downsample(ds):
    monthly = ds.resample(time='5D', skipna=True).mean() #### Change here to change the timeframe over which to make the data imputation
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
    return xr_df.assign(ndvi=(xr_df['channel_2'] - xr_df['channel_1']) / (xr_df['channel_2'] + xr_df['channel_1']))


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

def add_time(xr_df):
    my_date_string = xr_df.attrs['EPCT_start_sensing_time']#xr_df.attrs['date_time']
    date_xr = datetime.strptime(my_date_string,'%Y%m%dT%H%M%SZ') #datetime.strptime(my_date_string, '%Y%m%d/%H:%M')
    date_xr = pd.to_datetime(date_xr)
    xr_df = xr_df.assign_coords(time=date_xr)
    xr_df = xr_df.expand_dims(dim="time")
    return xr_df

def process_ndvi(base_dir, file):
    with xr.open_dataset(os.path.join(base_dir, file)) as ds:
        data = ds.load()
        xr_df = data.drop('channel_3')
        xr_df = add_time(data)
        xr_df = compute_radiance(xr_df)
        xr_df = xr_df.assign(ndvi=(xr_df['channel_2'] - xr_df['channel_1']) / (xr_df['channel_2'] + xr_df['channel_1']))
        xr_df.to_netcdf(os.path.join(base_dir,'processed', file)) 
        xr_df.close()


def drop_water_bodies_esa(CONFIG_PATH:str, config:dict, dataset:xr.Dataset, var:str="ndvi") ->xr.Dataset:
    from ancillary_vars.esa_landuse import get_level_colors, get_cover_dataset

    img_path = os.path.join(config["DEFAULT"]["images"], "chirps_esa")
    ds_cover = get_cover_dataset(CONFIG_PATH, dataset[var], img_path)
    
    water_mask = xr.where((ds_cover["Band1"]==80) | (ds_cover["Band1"]==200), 1,0)
    ds_process = ds_cover.where(water_mask==0).drop_vars("Band1")
    dataset = dataset.assign(ndvi=ds_process[var])
    return dataset

def extract_apply_cloudmask(ds, ds_cl, resample=False, include_water =True,downsample=False):
    
    def checkVars(ds, var):
        assert var  in ds.data_vars, f"Variable {var} not in dataset"

    [checkVars(ds, var)  for var in ["channel_1","channel_2","ndvi"]]

    ### normalize time in order for the two datasets to match
    ds_cl['time'] = ds_cl.indexes['time'].normalize()
    ds['time'] = ds.indexes['time'].normalize()
    
    if resample==True:
    #### reproject cloud mask to base dataset
        reproj_cloud = ds_cl['cloud_mask'].rio.reproject_match(ds['ndvi'])
        ds_cl_rp = reproj_cloud.rename({'y':'lat', 'x':'lon'})

    else:
        ds_cl_rp = ds_cl

    ### apply time mask where values are equal to 1, hence no clouds over land, 0= no cloud over water
    if include_water==True:
        ds_subset = ds.where((ds_cl_rp==1)|(ds_cl_rp==0)) #ds = ds.where(ds.time == ds_cl.time)
    else:
        ds_subset = ds.where(ds_cl_rp==1)
    ### recompute corrected ndvi
    res_xr = compute_ndvi(ds_subset)

    ### mask all the values equal to 0 (clouds)

    mask_clouds = clean_ndvi(ds)
    ### recompute corrected ndvi
    mask_clouds = compute_ndvi(mask_clouds)

    #### downsample to 5 days
    if downsample==True:
        "Starting downsampling the Dataset"
        res_xr_p = downsample(res_xr)
        #### downsampled df
        mask_clouds_p = downsample(mask_clouds)
        return mask_clouds_p, res_xr_p,  mask_clouds, res_xr ### return 1) cleaned dataset with clouds 
                                                         ### 2) imputation with max over n days
                                                         ### 3) cloudmask dataset original sample
                                                         ### 4) cloudmask dataset downsampled
    else:
        return mask_clouds, res_xr  ###1) dataset without zeros (clouds) 2) dataset with cloud mask applied


def apply_whittaker(datarray:DataArray, lambda_par:int=1, prediction="P1D", time_dim="time"):
    from fusets import WhittakerTransformer
    from fusets._xarray_utils import _extract_dates, _output_dates, _topydate
    result = WhittakerTransformer().fit_transform(datarray.load(),smoothing_lambda=lambda_par, time_dimension=time_dim, prediction_period=prediction)
    dates = _extract_dates(datarray)
    expected_dates = _output_dates(prediction,dates[0],dates[-1])
    datarray['time'] = datarray.indexes['time'].normalize()
    datarray = datarray.assign_coords(time = datarray.indexes['time'].normalize())
    result['time'] = [np.datetime64(i) for i in expected_dates]
    return result


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