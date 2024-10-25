import yaml
import xarray as xr
import sys
import numpy as np
from typing import Union, Literal, Sequence, Optional
import logging
import psutil
logger = logging.getLogger(__name__)

"""
General utility functions
"""

def get_tensor_memory(tensor):
    memory_size = tensor.element_size() * tensor.nelement()
    return memory_size

def get_vram():
    import torch
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    total = torch.cuda.mem_get_info()[1] / 1024 ** 3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    return f'VRAM: {total - free:.2f}/{total:.2f}GB\t VRAM:[' + (
            total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'

def get_ram():
    import psutil
    mem = psutil.virtual_memory()
    free = mem.available / 1024 ** 3
    total = mem.total / 1024 ** 3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    return f'RAM: {total - free:.2f}/{total:.2f}GB\t RAM:[' + (total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'



def create_xarray_datarray(var_name:str, data, time, lat, lon):
    return xr.DataArray(
                data, 
                dims   = ['time',"lat","lon"],
                coords = {'time': time, "lat": lat,"lon": lon},
                name=var_name)

def display_usage(cpu_usage, mem_usage, bars=50):
    cpu_percent = (cpu_usage/100)
    cpu_bar = "" * int(cpu_percent * bars) + "-" * (bars-int(cpu_percent * bars)) 

    mem_percent = (mem_usage/ 100)
    mem_bar = "" * int(mem_percent * bars) + "-" * (bars-int(mem_percent * bars)) 

    print(f"\rCPU Usage: |{cpu_bar}| {cpu_usage:.2f}% ", end="")
    print(f"\MEM Usage: |{mem_bar}| {mem_usage:.2f}% ", end="\r")

    

def init_logging(log_file=None, verbose=False):
    import os
    # Determine the logging level
    if verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Define the logging format
    formatter = "%(asctime)s : %(levelname)s : [%(filename)s:%(lineno)s - %(funcName)s()] : %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    
    # Setup basic configuration for logging
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            level=level,
            format=formatter,
            datefmt=datefmt,
            handlers=[
                logging.FileHandler(log_file, "w"),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=level,
            format=formatter,
            datefmt=datefmt,
            handlers=[
                logging.StreamHandler()
            ]
        )

    logger = logging.getLogger()
    return logger

#Creating a handler
def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
                #Will call default excepthook
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
        #Create a critical level log message with info from the except hook.
    logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))


def extract_grid(grid_shape, return_pandas:bool=False):

    min_lat, min_lon, max_lat, max_lon = hoa_bbox()

    lats = np.linspace(min_lat, max_lat, grid_shape[0])
    lons = np.linspace(min_lon, max_lon, grid_shape[1])

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    lat_lon_grid = np.stack([lat_grid, lon_grid], axis=-1)
    reshaped_gird = lat_lon_grid.reshape(-1, 2)

    if return_pandas is True:
        import pandas as pd
        return pd.DataFrame(reshaped_gird, 
                            columns=["lon","lat"])
    else:
        return reshaped_gird

def hoa_bbox(invert:bool = False):
    minx = -5.48369565
    miny = 32.01630435
    maxx = 15.48369565
    maxy = 51.48369565

    if invert is False:
        return [miny, minx, maxy, maxx]
    else:
        return [minx, miny, maxx, maxy]

def load_config(CONFIG_PATH:str):
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)
        return config
    
from definitions import CONFIG_PATH
config = load_config(CONFIG_PATH)

def prepare(dataset:Union[xr.DataArray, xr.Dataset]):
    if "longitude" in dataset.dims:
        dataset = dataset.rename({"latitude":"lat", "longitude":"lon"})
    if "x" in dataset.dims:
        dataset = dataset.rename({"y":"lat", "x":"lon"})
    if "X" in dataset.dims:
        dataset = dataset.rename({"Y":"lat", "X":"lon"})
    dataset.rio.write_crs("epsg:4326", inplace=True)
    dataset.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
    return dataset

def roll_longitude(ds):
    return ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')

def clip_file(dataset:Union[xr.DataArray, xr.Dataset], 
             gdf=None, invert=False):
    
    from shapely.geometry import Polygon, mapping, box
    import geopandas as gpd
    epsg_coords = "EPSG:4326"
    dataset = prepare(dataset)
    
    if gdf is not None:
        clipped = dataset.rio.clip(gdf.geometry.apply(mapping), 
                             gdf.crs, 
                             drop=True,
                             invert=invert)
    elif gdf is None:
        logging.info(f"Using the defeault bbox for the Horn of Africa"
                     f" to clip the images")
        bbox = hoa_bbox(invert=invert)
        geodf = gpd.GeoDataFrame(
                geometry=[box(bbox[0], bbox[1], bbox[2], bbox[3])],
                crs=epsg_coords)
        
        clipped = dataset.rio.clip(geodf.geometry.values, 
                                       geodf.crs,
                                       drop=True,
                                       invert=invert)
    return clipped

def subsetting_pipeline(dataset:Union[xr.DataArray, xr.Dataset], 
                        countries:Union[list, None] = ['Ethiopia','Kenya', 'Somalia',"Djibouti"], 
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
    return clip_file(dataset, subset)

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
            dataset = dataset.drop_duplicates(dim=["time"])
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

def extract_chunksize(ds:xr.Dataset)->dict:
    new_dict = {}
    for dim in ds.dims:
        len_dim = len(ds[dim])
        chunk_len = len(dict(ds.ndvi.chunksizes)[dim])
        pair = {dim : round(len_dim/chunk_len)}
        new_dict.update(pair)
    return new_dict

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

def safe_open_mfdataset(files, preprocess_func, chunks):
    try:
        return xr.open_mfdataset(files, parallel=False, chunks=chunks, preprocess=preprocess_func)
    except Exception as e:
        logging.error(f"Exception {e} on files {files}")
        return read_netcdfs(files, dim="time", transform_func=preprocess_func)

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


def find_checkpoint_path(model_config, args, return_latest:bool=False):
    import os
    import glob

    folder_path =  model_config.output_dir + f"/{(args.model).lower()}" \
                f"/days_{args.step_length}/features_{args.feature_days}" \
                f"/checkpoints"
    
    if return_latest:
        # Find all files matching the pattern 'checkpoint_*'
        checkpoint_files = glob.glob(os.path.join(folder_path, 'checkpoint_epoch_*'))

        # Get the most recently created file
        most_recent_file = max(checkpoint_files, key=os.path.getctime)

        # Output the most recently created file
        print(f"The most recently created checkpoint is: {os.path.basename(most_recent_file)}")
        return most_recent_file
    else:
        return folder_path


def CNN_split(data:np.array, 
              target:np.array, 
              split_percentage:float=0.7,
              val_split:float=0.5,
              print_actual_days:bool=True):
    tot_perc_val = (1 - split_percentage) * val_split
    tot_perc_test = 1 - tot_perc_val - split_percentage 
    logger.info(f"Data is divided as {split_percentage :.0%} train," 
                f" {tot_perc_val :.0%} validation and {tot_perc_test :.0%} independent test")
    ###splitting test and train
    n_samples = data.shape[0]
    train_samples = int(round(split_percentage*n_samples, 0))
    test_samples = int(round(tot_perc_test*n_samples, 0))
    val_samples = n_samples - (train_samples + test_samples)

    train_data = data[:train_samples]
    val_data =  data[train_samples:train_samples+val_samples]
    test_data= data[train_samples+ val_samples:]

    train_label = target[:train_samples]
    val_label =  target[train_samples:train_samples+val_samples]
    test_label = target[train_samples+ val_samples:]

    if print_actual_days:
        from utils.function_clns import config
        import pandas as pd
        from datetime import timedelta, datetime
        start_data = config["DEFAULT"]["date_start"]
        end_data = config["DEFAULT"]["date_end"]
        start_pd = pd.to_datetime(start_data, format='%Y-%m-%d')
        end_pd = pd.to_datetime(end_data, format='%Y-%m-%d')
        date_range = pd.date_range(start_pd, end_pd)

        end_1 = start_pd + timedelta(days = train_samples-1)
        new_start = end_1 + timedelta(days = 1)

        end_2 = new_start + timedelta(days = val_samples-1)
        new_start_2 = end_2 + timedelta(days = 1)

        final_end = new_start_2 + timedelta(days=test_samples-1)

        logger.info(f"Training is from {start_data} to {datetime.strftime(end_1, '%Y-%m-%d')}, "
            f"validation from {datetime.strftime(new_start, '%Y-%m-%d')} to {datetime.strftime(end_2, '%Y-%m-%d')}, "
            f"testing from {datetime.strftime(new_start_2, '%Y-%m-%d')} to {datetime.strftime(final_end, '%Y-%m-%d')}")

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

    train_data, test_data, train_label, test_label = CNN_split(data, 
                                                               target, 
                                                               split_percentage=split)

    return train_data, test_data, train_label, test_label


def prepare_datarray(datarray:xr.DataArray, 
               interpolate:bool = False, 
               check_data:bool = False,
               convert_to_float:bool = False):
    
    import numpy as np
    from utils.function_clns import prepare
    from dask.diagnostics import ProgressBar

    def convert_strings_to_float(arr):
        # Vectorized conversion using np.where
        return np.where(np.char.isnumeric(arr), 
                        arr.astype(np.float32), arr)

    if convert_to_float is True:
        datarray.values = convert_strings_to_float(datarray.values)

    datarray= datarray.rio.write_nodata("nan")
    datarray = datarray.rio.write_nodata(np.nan)

    if interpolate is True:
        with ProgressBar():
            datarray = datarray.rio.interpolate_na()
    if check_data is True:
        check_xarray_dataset(None, datarray, save=False, plot = False)
    
    return datarray.transpose("time","lat","lon")

def drop_extra_vars(dataset):
    vars = ["crs", "spatial_ref"]
    if isinstance(dataset, xr.Dataset):
        for var in vars:
            if var in dataset.data_vars:
                dataset = dataset.drop_vars(var)
    return dataset


def bias_correction(y_true, y_pred):
    from sklearn.linear_model import LinearRegression

    s, w, h = y_true.shape
    # Create linear regression model
    model = LinearRegression()
    model.fit(y_pred.reshape( s*w*h).reshape(-1, 1),y_true.reshape(s*w*h).reshape(-1, 1))

    def shift_data(data, intercept, coefficient):
        return data*coefficient[0] + intercept[0]

    # Compute bias
    coefficient = model.coef_  # Coefficients of the model
    intercept = model.intercept_  # Intercept of the model

    # print(f"Coefficients: {coefficient}")
    # print(f"Intercept: {intercept}")

    y_corr = shift_data(y_pred, intercept, coefficient)

    return y_corr, coefficient, intercept

def interpolate_prepare(input_data:Union[xr.Dataset, xr.DataArray], 
                        target_data:xr.DataArray, 
                        interpolate:bool=True,
                        convert_to_float:bool=False):
    """
    Function to prepare rand interpolate provided datasets w.r.t a given target dataset
    """

    if interpolate is True:
        logging.info("Interpolating values over the time dimension using a nearest neighbor")
        input_data = input_data.interpolate_na(dim="time", method="nearest")
        target_data = target_data.interpolate_na(dim="time", method="nearest")

    input_data = drop_extra_vars(input_data)

    if isinstance(input_data, xr.Dataset):
            
        shape = (len(input_data['time']), len(input_data.data_vars), 
                 len(input_data['lat']), len(input_data['lon']))
        
        result_array = np.empty(shape)
        logger.debug("Loading variables to unique array...")
        # Populate the numpy array
        for i, variable in enumerate(input_data.data_vars):
            logger.info(f"Channel {i}: {input_data[variable].name}")
            result_array[:, i, :, :] = input_data[variable]

    elif isinstance(input_data, xr.DataArray):
        result_array = input_data.to_numpy()
    else:
        raise TypeError("The input data type is not xarray-compatible")
    
    target = target_data.to_numpy()

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