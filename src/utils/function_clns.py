import yaml
import xarray as xr
import glob
import numpy as np
from typing import Union, Literal, Sequence, Optional

"""
General utility functions
"""

def load_config(CONFIG_PATH:str):
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)
        return config
    
from definitions import CONFIG_PATH
config = load_config(CONFIG_PATH)

def prepare(ds:Union[xr.DataArray, xr.Dataset]):
    if "longitude" in ds.dims:
        ds = ds.rename({"latitude":"lat", "longitude":"lon"})
    if "x" in ds.dims:
        ds = ds.rename({"y":"lat", "x":"lon"})
    if "X" in ds.dims:
        ds = ds.rename({"Y":"lat", "X":"lon"})
    ds.rio.write_crs("epsg:4326", inplace=True)
    ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
    return ds

def cut_file(xr_df, gdf):
    from shapely.geometry import Polygon, mapping
    xr_df.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
    xr_df.rio.write_crs("epsg:4326", inplace=True)
    clipped = xr_df.rio.clip(gdf.geometry.apply(mapping), gdf.crs, drop=True)
    return clipped

def subsetting_pipeline(xr_df:Union[xr.DataArray, xr.Dataset], 
                        countries:Union[list, None] = ['Ethiopia','Kenya', 'Somalia'], 
                        regions: Union[list, None] = None,
                        invert=False):
    import geopandas as gpd
    from utils.function_clns import config
    import shapely
    if regions is None and countries is None:
        raise ValueError("You must specify either a list of countries or regions")
    if regions is not None and countries is not None:
        raise ValueError("You must specify either country or regions list, not both")
    if countries is not None:
        shapefile_path = config['SHAPE']['africa']
        column = "ADM0_NAME"
        location =countries
    else:
        shapefile_path = config['SHAPE']['ethiopia']
        column = "REGIONNAME"
        location=regions

    gdf = gpd.read_file(shapefile_path)
    subset = gdf[gdf[column].isin(location)]
    if invert==True:
        subset = subset['geometry'].map(
            lambda polygon: shapely.ops.transform(lambda x, y: (y, x), polygon))
    return cut_file(xr_df, subset)

def xesmf_regrid_align(dataset1:Union[xr.DataArray, xr.Dataset],
                dataset2:Union[xr.DataArray, xr.Dataset],
                repr_method:str="bilinear",
                chunks:dict={"time":"auto", "lat":"auto", "lon":"auto"},
                align:bool = True):
    
    """
    Function to reproject and align two xarray datasets/datarray
    """
    import xesmf as xe
    import pandas as pd
    
    dataset1 = prepare(dataset1)
    dataset2 = prepare(dataset2)
    
    if float(dataset1.rio.resolution()[0])< float(dataset2.rio.resolution()[0]):
        reproj_dataset, target_dataset = dataset1, dataset2
    else:
        reproj_dataset, target_dataset = dataset2, dataset1

    regridder = xe.Regridder(reproj_dataset, target_dataset, repr_method)

    # Reproject the entire dataset
    ds_reprojected = regridder(reproj_dataset)
    ds_reprojected = prepare(ds_reprojected.transpose("time","lon","lat"))

    print("Destination dataset resolution:", target_dataset.rio.resolution())
    print("Reprojected dataset resolution:", ds_reprojected.rio.resolution())

    if align is True:
        ds1, ds2 = align_datasets(ds_reprojected, target_dataset)
        return ds1, ds2
    else:
        return ds_reprojected, target_dataset

def align_datasets(dataset1:Union[xr.DataArray, xr.Dataset],
                dataset2:Union[xr.DataArray, xr.Dataset],
                dataset3:Union[xr.DataArray, xr.Dataset, None]=None,
                dataset4:Union[xr.DataArray, xr.Dataset, None]=None,
                chunks:dict={"time":"auto", "lat":"auto", "lon":"auto"}):
    import pandas as pd

    def preprocess_dataset(dataset):
        if dataset is not None:
            dataset['time'] = dataset['time'].drop_duplicates(dim=["time"])
            dataset['time'] = pd.to_datetime(dataset['time'].values, format='%Y-%m-%d')
            dataset['time'] = dataset.indexes["time"].normalize()
            dataset = dataset
        return dataset

    dataset1 = preprocess_dataset(dataset1)
    dataset2 = preprocess_dataset(dataset2)
    dataset3 = preprocess_dataset(dataset3)
    dataset4 = preprocess_dataset(dataset4)

    # Find the common time values between ds1 and ds2

    ds1, ds2 = xr.align(dataset1, dataset2)

    if dataset3 is not None and dataset4 is not None:
        ds1, ds2, dataset3, dataset4 = xr.align(ds1,
                                                ds2, 
                                                dataset3,
                                                dataset4, 
                                                join='inner')
    
    if dataset3 is not None:
        ds1, dataset3 = xr.align(ds1, dataset3, join='inner')
        ds2, dataset3 = xr.align(ds2, dataset3, join='inner')

    if dataset4 is not None:
        ds1, dataset4 = xr.align(ds1, dataset4, join='inner')
        ds2, dataset4 = xr.align(ds2, dataset4, join='inner')

    return ds1, ds2, dataset3, dataset4


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


def read_netcdfs(files, dim, transform_func=None):
    
    def process_one_path(path):
        try:
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
        except Exception as e:
            print(f"Skipping corrupted file: {path}. Error: {e}")
            return None

    paths = sorted(files)
    datasets = [process_one_path(p) for p in paths]
    datasets = [ds for ds in datasets if ds is not None]
    if len(datasets) == 0:
        raise ValueError("All files are corrupted.")
    combined = xr.concat(datasets, dim)
    return combined

def read_hdf(file, DATAFIELD_NAME:str, 
             filter_nan:bool=True):
    from pyhdf.SD import SD, SDC
    import pandas as pd
    import numpy as np
    '''
    -read a specified datafield ('SDS') from a MODIS L2 hdf file
    -screen missing values and correct real values according to in-file offset and scale-factor attributes
    -read corresponding latlons
    -return the analysis-ready datafield (as a masked array) and latlons to parent process
    
    '''
    hdf=SD(file,SDC.READ)
    data2d=hdf.select(DATAFIELD_NAME)
    data=data2d[:,:].astype(np.double)

    
    #get latlons and variable attributes
    lat=hdf.select('Latitude')
    latitude=lat[:,:]
    lon=hdf.select('Longitude')
    longitude=lon[:,:]

    attrs=data2d.attributes(full=1)
    add_offset=attrs['add_offset'][0]
    _FillValue=attrs['_FillValue'][0]
    scale_factor=attrs['scale_factor'][0]
    units=attrs['units'][0]

    if filter_nan is True:
        #prepare datafield for processing
        data[data==_FillValue]=np.nan
        data=(data-add_offset)*scale_factor 
        datam=np.ma.masked_array(data,np.isnan(data))

    else:
        datam = data
    
    #return datafield prepared for analysis, and latlons
    return datam,latitude,longitude


def unpack_all_in_dir(_dir, extension = ".zip"):
    import os
    import zipfile
    for item in os.listdir(_dir):  # loop through items in dir
        abs_path = os.path.join(_dir, item)  # absolute path of dir or file
        if item.endswith(extension):  # check for ".zip" extension
            file_name = os.path.abspath(abs_path)  # get full path of file
            zip_ref = zipfile.ZipFile(file_name)  # create zipfile object
            zip_ref.extractall(_dir)  # extract file to dir
            zip_ref.close()  # close file
            os.remove(file_name)  # delete zipped file
        elif os.path.isdir(abs_path):
            unpack_all_in_dir(abs_path)  # recurse this function with inner folder

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


def CNN_imputation(ds:Union[xr.DataArray, xr.Dataset], 
                   ds_target:Union[xr.DataArray, xr.Dataset], 
                   var_origin:str, 
                   var_target:str, 
                   preprocess_type:Literal["constant", "nearest","median", "None"]="constant", 
                   impute_value:Union[None, float, int]=None):
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


def CNN_split(data:np.array, 
              target:np.array, 
              split_percentage:float=0.7, 
              test_split:float=0.1):

    print(f"{test_split:.0%} of the training data will be used as independent test")
    ###splitting test and train
    n_samples = data.shape[-1]
    train_samples = int(round(split_percentage*n_samples, 0))
    test_samples = int(round(test_split*n_samples, 0))
    val_samples = n_samples - (train_samples + test_samples)

    data = np.expand_dims(data.transpose(2,0,1), 0)
    target = np.expand_dims(target.transpose(2,0,1), 0)

    train_data = data[:,:train_samples,:,:]
    test_data =  data[:,train_samples:,:,:]
    train_label = target[:,:train_samples,:,:]
    test_label =  target[:,train_samples:,:,:]

    train_data = data[:,:train_samples,:,:]
    val_data =  data[:,train_samples:train_samples+val_samples,:,:]
    test_data= data[:,train_samples+ val_samples:,:,:]

    train_label = target[:,:train_samples,:,:]
    val_label =  target[:,train_samples:train_samples+val_samples,:,:]
    test_label = target[:,train_samples+ val_samples:,:,:]

    return train_data, val_data, train_label, val_label, test_data, test_label


def CNN_preprocessing(ds:Union[xr.DataArray, xr.Dataset], 
                      ds_target:Union[xr.DataArray, xr.Dataset], 
                      var_origin:str, var_target:str,  
                      preprocess_type:Literal["constant", "nearest","median", "None"]="constant", 
                      impute_value:Union[None, float, int]=None, 
                      split:float=0.8):

    if (ds.isnull().any()==True) or (ds_target.isnull().any()==True):
        target, data = CNN_imputation(ds, ds_target, var_origin, var_target, preprocess_type=preprocess_type, impute_value=impute_value)

    else:
        target = np.array(ds[var_origin])
        data = np.array(ds_target[var_target])

    train_data, test_data, train_label, test_label = CNN_split(data, target, split_percentage=split)

    return train_data, test_data, train_label, test_label


def prepare_datarray(datarray:xr.DataArray, 
               interpolate:bool = False, 
               check_data:bool = False,
               convert_to_float:bool = False):
    
    import numpy as np

    def convert_strings_to_float(arr):
        # Vectorized conversion using np.where
        return np.where(np.char.isnumeric(arr), 
                        arr.astype(np.float32), arr)

    if convert_to_float is True:
        datarray.values = convert_strings_to_float(datarray.values)
    
    if datarray.values.dtype == np.str_:
        datarray= datarray.rio.write_nodata("nan")
    else:
        datarray = datarray.rio.write_nodata(np.nan)

    # datarray = datarray.astype(np.float32)

    if interpolate is True:
        datarray = datarray.rio.interpolate_na()
    if check_data is True:
        check_xarray_dataset(None, datarray, save=False, plot = False)
    
    return datarray.transpose("time","lat","lon")

def interpolate_prepare(
                        input_data:Union[xr.Dataset, xr.DataArray], 
                        target_data:xr.DataArray, 
                        interpolate:bool=True,
                        convert_to_float:bool=False):
    """
    Function to prepare rand interpolate provided datasets w.r.t a given target dataset
    """

    if type(input_data) == xr.Dataset:

        for var in input_data.data_vars:
            input_data[var] = prepare_datarray(input_data[var], 
                                               interpolate=interpolate,
                                               convert_to_float=convert_to_float)

        shape = (len(input_data['time']), len(input_data.data_vars), 
                 len(input_data['lat']), len(input_data['lon']))
        
        result_array = np.empty(shape)

        # Populate the numpy array
        for i, variable in enumerate(input_data.data_vars):
            result_array[:, i, :, :] = input_data[variable]

    else:
        input_data = prepare_datarray(input_data, interpolate=interpolate,
                                      convert_to_float=convert_to_float)
        result_array = np.array(result_array)

    target_data = prepare_datarray(target_data)
    target = np.array(target_data)

    # if interpolate is True:
    #     null_target = target_data.rio.interpolate_na()

    #     input_data = input_data.assign(null_precp =  data_null)  
    #     var = "null_input" 
    #     target = null_target.transpose("lat","lon","time").values #
    #     check_xarray_dataset(args, [null_target,  input_data[var]], save=False, plot=False)
    # else:
    #     target = target_data.transpose("lat","lon","time")
    #     var = var_target

    # Read the data as a numpy array

    return result_array, target

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


def crop_image_right(temp_ds, target_pixels):
    dict_lat = temp_ds["lat"].values
    lat_max = temp_ds["lat"].max().values
    idx_lat= np.where(dict_lat==lat_max)[0][0]
    idx_tagt_lat = dict_lat[idx_lat+target_pixels-1]
    
    dict_lon = temp_ds["lon"].values
    lon_max = temp_ds["lon"].max().values
    idx_lon= np.where(dict_lon==lon_max)[0][0]
    idx_tagt_lon = dict_lon[idx_lon-target_pixels+1]
    return idx_tagt_lat, lat_max, idx_tagt_lon, lon_max

def crop_image_left(temp_ds, target_pixels):
    dict_lat = temp_ds["lat"].values
    lat_max = temp_ds["lat"].max().values
    idx_lat= np.where(dict_lat==lat_max)[0][0] ###get idx of max lat point
    idx_tagt_lat = dict_lat[idx_lat+target_pixels-1] 
    
    dict_lon = temp_ds["lon"].values
    lon_min = temp_ds["lon"].min().values
    idx_lon= np.where(dict_lon==lon_min)[0][0]
    idx_tagt_lon = dict_lon[idx_lon+target_pixels-1]
    return idx_tagt_lat, lat_max, idx_tagt_lon, lon_min


def check_timeformat_arrays(array_1:xr.DataArray, array_2:xr.DataArray):
    if array_1.indexes["time"][0] == array_2.indexes["time"][0]:
        return array_1, array_2
    else:
        array_1["time"] = array_1.indexes["time"].normalize()
        array_2["time"] = array_2.indexes["time"].normalize()
        return array_1, array_2
    
def check_nulls_overtime(datarray):
    # Compute a boolean mask for missing values
    missing_mask = xr.where(np.isnan(datarray), True, False)

    # Check if the mask values are constant over time
    are_missing_values_constant = missing_mask.all(dim='time')

    # Create a mask to identify points with non-constant missing values
    non_constant_missing_mask = ~are_missing_values_constant

    # Apply the mask to the DataArray to keep only points with non-constant missing values
    filtered_data = datarray.where(non_constant_missing_mask, drop=True)

    # Print the filtered DataArray
    print(filtered_data)
    return filtered_data

def check_xarray_dataset(args, data: Union[xr.DataArray, list], save:bool=False, plot:bool=True):
    import matplotlib.pyplot as plt
    import os
    import time
    
    def datarray_check(data):
        from utils.function_clns import config
        # Detect and inspect coordinates
        for dim in data.dims:
            if dim != "time":
                coord_values = data.coords[dim].values
                print(f"{dim}-axis values:", coord_values)
        # Inspect dimensions and size
        print("Dimensions:", data.dims)
        print("Size:", data.size)
        print("Number of Dimensions:", data.ndim)
        print("Shape:", data.shape)

        # Inspect coordinates
        print("Coordinates:", data.coords)

        # Check for missing values
        print("Is null:", data.isnull().sum())
        print("Not null:", data.notnull().sum())

        if plot is True:
            fig = plt.figure() 
            print("Plotting the dataset...")
            data.isel(time=0).plot()
            plt.close(fig)
        if save is True:
            name = data.name
            if args.spi==False:
                plt.savefig(os.path.join(args.output_dir, f"images_results/forecast_\
                                         {config[args.pipeline]['forecast']}/{name}_dataset.png"))
            else:
                plt.savefig(os.path.join(args.output_dir, f"images_results/forecast_\
                                         {args.precp_product}_SPI_{config[args.pipeline]['latency']}\
                                         {name}_dataset.png"))
            
    
    if type(data)==list:
        for ds in data:
            datarray_check(ds)
    else:
        datarray_check(data)