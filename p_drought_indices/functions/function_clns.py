import yaml
from shapely.geometry import Polygon, mapping
import geopandas as gpd
import shapely
import xarray as xr
import glob
import numpy as np
from typing import Union, Literal

def load_config(CONFIG_PATH):
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)
        return config

def prepare(ds):
        ds.rio.write_crs("epsg:4326", inplace=True)
        ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
        return ds

def cut_file(xr_df, gdf):
    xr_df.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
    xr_df.rio.write_crs("epsg:4326", inplace=True)
    clipped = xr_df.rio.clip(gdf.geometry.apply(mapping), gdf.crs, drop=True)
    return clipped

def subsetting_pipeline(CONFIG_PATH, xr_df, countries = ['Ethiopia','Kenya', 'Somalia'], invert=False):
    config = load_config(CONFIG_PATH)
    shapefile_path = config['SHAPE']['africa']
    gdf = gpd.read_file(shapefile_path)
    subset = gdf[gdf.ADM0_NAME.isin(countries)]
    if invert==True:
        subset = subset['geometry'].map(lambda polygon: shapely.ops.transform(lambda x, y: (y, x), polygon))
    return cut_file(xr_df, subset)

from typing import Union, Optional, Sequence
from math import ceil, sqrt

def get_lon_dim_name(ds: Union[xr.Dataset, xr.DataArray]) -> Optional[str]:
    """
    Get the name of the longitude dimension.
    :param ds: An xarray Dataset
    :return: the name or None
    """
    return _get_dim_name(ds, ['lon', 'longitude', 'long'])


def get_lat_dim_name(ds: Union[xr.Dataset, xr.DataArray]) -> Optional[str]:
    """
    Get the name of the latitude dimension.
    :param ds: An xarray Dataset
    :return: the name or None
    """
    return _get_dim_name(ds, ['lat', 'latitude'])


def _get_dim_name(ds: Union[xr.Dataset, xr.DataArray], possible_names: Sequence[str]) -> Optional[str]:
    for name in possible_names:
        if name in ds.dims:
            return name
    return None
# noinspection PyUnresolvedReferences,PyProtectedMember
def open_xarray_dataset(paths, **kwargs) -> xr.Dataset:
    """
    Open multiple files as a single dataset. This uses dask. If each individual file
    of the dataset is small, one dask chunk will coincide with one temporal slice,
    e.g. the whole array in the file. Otherwise smaller dask chunks will be used
    to split the dataset.
    :param paths: Either a string glob in the form "path/to/my/files/\*.nc" or an explicit
        list of files to open.
    :param concat_dim: Dimension to concatenate files along. You only
        need to provide this argument if the dimension along which you want to
        concatenate is not a dimension in the original datasets, e.g., if you
        want to stack a collection of 2D arrays along a third dimension.
    :param kwargs: Keyword arguments directly passed to ``xarray.open_mfdataset()``
    """
    # By default the dask chunk size of xr.open_mfdataset is (lat,lon,1). E.g.,
    # the whole array is one dask slice irrespective of chunking on disk.
    #
    # netCDF files can also feature a significant level of compression rendering
    # the known file size on disk useless to determine if the default dask chunk
    # will be small enough that a few of them ccould comfortably fit in memory for
    # parallel processing.
    #
    # Hence we open the first file of the dataset, find out its uncompressed size
    # and use that, together with an empirically determined threshold, to find out
    # the smallest amount of chunks such that each chunk is smaller than the
    # threshold and the number of chunks is a squared number so that both axes,
    # lat and lon could be divided evenly. We use a squared number to avoid
    # in addition to all of this finding the best way how to split the number of
    # chunks into two coefficients that would produce sane chunk shapes.
    #
    # When the number of chunks has been found, we use its root as the divisor
    # to construct the dask chunks dictionary to use when actually opening
    # the dataset.
    #
    # If the number of chunks is one, we use the default chunking.
    #
    # Check if the uncompressed file (the default dask Chunk) is too large to
    # comfortably fit in memory
    threshold = 250 * (2 ** 20)  # 250 MB

    # Find number of chunks as the closest larger squared number (1,4,9,..)
    try:
        print(paths[0])
        temp_ds = xr.open_dataset(paths[0])
        print('paramapampam')
    except (OSError, RuntimeError):
        # netcdf4 >=1.2.2 raises RuntimeError
        # We have a glob not a list
        temp_ds = xr.open_dataset(glob.glob(paths)[0])

    n_chunks = ceil(sqrt(temp_ds.nbytes / threshold)) ** 2

    if n_chunks == 1:
        # The file size is fine
        return xr.open_mfdataset(paths,combine='by_coords', **kwargs)

    # lat/lon names are not yet known
    lat = get_lat_dim_name(temp_ds)
    lon = get_lon_dim_name(temp_ds)
    n_lat = len(temp_ds[lat])
    n_lon = len(temp_ds[lon])

    # temp_ds is no longer used
    temp_ds.close()

    if n_chunks == 1:
        # The file size is fine
        return xr.open_mfdataset(paths,combine='by_coords', **kwargs)

    divisor = sqrt(n_chunks)

    # Chunking will pretty much 'always' be 2x2, very rarely 3x3 or 4x4. 5x5
    # would imply an uncompressed single file of ~6GB! All expected grids
    # should be divisible by 2,3 and 4.
    if not (n_lat % divisor == 0) or not (n_lon % divisor == 0):
        raise ValueError("Can't find a good chunking strategy for the given"
                         "data source. Are lat/lon coordinates divisible by "
                         "{}?".format(divisor))

    chunks = {lat: n_lat // divisor, lon: n_lon // divisor}

    return xr.open_mfdataset(paths, chunks=chunks,combine='by_coords', **kwargs)


def read_netcdfs(files, dim, transform_func=None):
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    paths = sorted(files)
    datasets = [process_one_path(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined


def reproject_odc(ds_1, ds_2):
    from odc.algo import xr_reproject
    if ds_2.geobox.crs.geographic:
        ds_1 = ds_1.rename({"lon": "longitude", "lat": "latitude"})
    ds_repr = xr_reproject(ds_1, ds_2.geobox, resampling="bilinear")
    return ds_repr.rename({"longitude": "lon", "latitude": "lat"})

def crop_get_spi(ds:xr.DataArray, thresh:int=-2):
    null_var = xr.where(ds.notnull(), 1,np.NaN)
    condition_var = xr.where(ds<=thresh,1,0)
    return condition_var.where(null_var==1)

def crop_get_thresh(ds:xr.DataArray, thresh:int=10):
    null_var = xr.where(ds.notnull(), 1,np.NaN)
    condition_var = xr.where(ds<thresh,1,0)
    return condition_var.where(null_var==1)


"""
Functions for the Deep Learning models
"""

def add_channel(data, n_samples):

    # define the desired size of the time steps and number of channels 
    # ##output: (num_samples, num_frames, num_channels, height, width)
    n_timesteps = n_samples
    n_channels = 1

    # determine the number of samples based on the desired number of time steps
    n_samples = data.shape[-1] // n_timesteps

    # reshape the input data into a 4D tensor
    input_data = np.reshape(data, (data.shape[0], data.shape[1], n_timesteps, n_samples))

    # add an extra dimension for the channels
    input_data = np.reshape(input_data, (n_samples, n_timesteps,n_channels, input_data.shape[0], input_data.shape[1]))

    # check the shape of the input data
    print("The input data has shape:", input_data.shape) # should print (n_samples, n_timesteps, lat, lon, n_channels)
    return input_data


def CNN_imputation(ds:Union[xr.DataArray, xr.Dataset], ds_target:Union[xr.DataArray, xr.Dataset], var_origin:str, var_target:str, preprocess_type:Literal["constant", "nearest","median", "None"]="constant", impute_value:Union[None, float, int]=None):
    """
    Function to preprocess data for Convolutional Neural Networks, can be processed with either a constant, using the rioxarray function "interpolate_na()", nearest neighbors or with the median
    """
    if preprocess_type not in ["constant","nearest", "median","None"]:
        raise ValueError("Preprocessing type must be either \"constant\", \"nearest\", \"median\"")
    
    if ((preprocess_type == "constant") and (isinstance(impute_value, (int, float)))==False):
        raise ValueError("A value muste be specified for impute_value in order to impute data with a constant")
    
    ### preprocess data
    #ds = prepare(ds)
    #sub_precp = prepare(ds_target)

    sub_precp = ds_target
    veg_repr = ds

    #veg_repr = ds[var_origin].rio.reproject_match(sub_precp[var_target]).rename({'x':'lon','y':'lat'})
    
    if preprocess_type == "nearest":
        print("Preprocessing data with nearest neighbor from scipy.interpolate.griddata")
        ds = ds.transpose("time","lat","lon")
        sub_precp = sub_precp.transpose("time","lat","lon")
        sub_precp[var_target] = sub_precp[var_target].rio.interpolate_na()
        sub_veg = veg_repr.rio.interpolate_na()
        sub_precp = sub_precp.assign(null_precp =  sub_precp[var_target])  
        var = "null_precp"

    elif preprocess_type == "constant":
        print(f"Preprocessing data with constant {impute_value}")
        sub_veg = veg_repr.where(veg_repr.notnull(), impute_value)
        sub_precp = sub_precp.assign(null_precp = sub_precp[var_target].where(sub_precp[var_target].notnull(), impute_value))
        var = "null_precp"

    elif preprocess_type == "median":
        raise NotImplementedError
    
    elif preprocess_type == "None":
        sub_veg = veg_repr
        print("Not applying any imputation")
        var = var_target



    # Read the data as a numpy array
    target = sub_veg.transpose("lat","lon","time").values #
    data = sub_precp[var].transpose("lat","lon","time").values #.rio.interpolate_na()

    target = np.array(target)
    data = np.array(data)

    return target, data


def CNN_split(data:np.array, target:np.array, split_percentage:float=0.8):

    ###splitting test and train
    n_samples = data.shape[-1]
    train_samples = int(round(split_percentage*n_samples, 0))

    input_data = add_channel(data, n_samples)
    target_data = add_channel(target, n_samples)
    train_data = input_data[:,:train_samples,:,:]
    test_data =  input_data[:,train_samples:,:,:]
    train_label = target_data[:,:train_samples,:,:]
    test_label =  target_data[:,train_samples:,:,:]

    return train_data, test_data, train_label, test_label


def CNN_preprocessing(ds:Union[xr.DataArray, xr.Dataset], ds_target:Union[xr.DataArray, xr.Dataset], var_origin:str, var_target:str,  preprocess_type:Literal["constant", "nearest","median", "None"]="constant", impute_value:Union[None, float, int]=None, split:float=0.8):

    if (ds.isnull().any()==True) or (ds_target.isnull().any()==True):
        target, data = CNN_imputation(ds, ds_target, var_origin, var_target, preprocess_type=preprocess_type, impute_value=impute_value)

    else:
        target = np.array(ds[var_origin])
        data = np.array(ds_target[var_target])

    train_data, test_data, train_label, test_label = CNN_split(data, target, split_percentage=split)

    return train_data, test_data, train_label, test_label

def interpolate_prepare(sub_precp:xr.Dataset, ds:xr.DataArray):
    var_target = [var for var in sub_precp.data_vars][0]
    ds = ds.transpose("time","lat","lon")
    sub_precp = sub_precp.transpose("time","lat","lon")
    sub_precp[var_target] = sub_precp[var_target].rio.write_nodata("nan")
    sub_precp[var_target] = sub_precp[var_target].rio.interpolate_na()
    sub_veg = ds.rio.interpolate_na()
    sub_precp = sub_precp.assign(null_precp =  sub_precp[var_target])  
    var = "null_precp"

    # Read the data as a numpy array
    target = sub_veg.transpose("lat","lon","time").values #
    data = sub_precp[var].transpose("lat","lon","time").values #.rio.interpolate_na()
    target = np.array(target)
    data = np.array(data)
    return data, target


def get_lat_lon_window(temp_ds, target_pixels):
    dict_lat = temp_ds["lat"].values
    lat_max = temp_ds["lat"].max().values
    idx_lat= np.where(dict_lat==lat_max)[0][0]
    idx_tagt_lat = dict_lat[idx_lat+target_pixels-1]
    
    dict_lon = temp_ds["lon"].values
    lon_min = temp_ds["lon"].min().values
    idx_lon= np.where(dict_lon==lon_min)[0][0]
    idx_tagt_lon = dict_lon[idx_lon+target_pixels-1]
    return idx_tagt_lat, lat_max, idx_tagt_lon, lon_min


def check_xarray_dataset(args, data: xr.DataArray, save:bool=False):
    import matplotlib.pyplot as plt
    import os
    import time
    # Detect and inspect coordinates
    for dim in data.dims:
        if dim != "time":
            coord_values = data.coords[dim].values
            print(f"{dim}-axis values:", coord_values)
    # Inspect dimensions and size
    print("Dimensions:", data.dims)
    print("Size:", data.size)
    print("Number of Dimensions:", data.ndim)

    # Inspect coordinates
    print("Coordinates:", data.coords)

    # Check for missing values
    print("Is null:", data.isnull().sum())
    print("Not null:", data.notnull().sum())

    fig = plt.figure() 

    print("Plotting the dataset...")
    data.isel(time=0).plot()
    if save is True:
        name = data.name
        plt.savefig(os.path.join(args.output_dir, f"images_results/forecast_{args.forecast}/{name}_dataset.png"))
        plt.close(fig)
    else:
        plt.show()
        # Wait for 5 seconds (adjust the time as needed)
        time.sleep(3)
        # Close the image window
        plt.close()