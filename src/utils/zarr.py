"""
This script contains utils for zarr files
"""

from typing import List, Union
import itertools
import zarr
import xarray as xr
import logging
logger = logging.getLogger(__name__)


def handle_gcs_zarr(url:str):
    """
    Function to handle zarr files from google cloud storage
    """
    import gcsfs
    import zarr
    # Connect to Google Cloud Storage
    fs = gcsfs.GCSFileSystem(token='anon', 
                             access='read_only')
    # create a MutableMapping from a store URL
    mapper = fs.get_mapper( url)
    store = zarr.open(mapper)
    return store

def gather_coordinate_dimensions(group: zarr.Group) -> List[str]:
    """
    Gather and return a list of unique coordinate dimensions found in a Zarr-store.

    Args:
        group (zarr.Group): The Zarr store to inspect.

    Returns:
        List[str]: A list of unique coordinate dimensions found in the store.
    """
    return set(
        itertools.chain(*(group[var].attrs.get("_ARRAY_DIMENSIONS", []) for var in group)))

def gather_vars(store:zarr.hierarchy.Group):
    coords = gather_coordinate_dimensions(store)
    data_vars = list(set(store.keys()) - coords)
    return coords, data_vars

def get_variable_dims(store:zarr.hierarchy.Group, var:str):
    return store[var].attrs.get("_ARRAY_DIMENSIONS", [])

def load_zarr_arrays(store:zarr.hierarchy.Group,
                     arrays:list,
                     min_time:Union[int, str],
                     max_time:Union[int, str],
                     bounding_box:list,
                     ds: xr.Dataset=None):
    
    import pandas as pd
    import numpy as np
    from utils.function_clns import config, create_xarray_datarray
    
    logger.debug("Loading coordinates from zarr file...")
    lat = store["latitude"][:]
    lon = store["longitude"][:]
    time = store["time"][:]

    lat_condition = np.where(((lat>=bounding_box[1]) & (lat<=bounding_box[3])),True, False)
    lon_condition = np.where(((lon>=bounding_box[0]) & (lon<=bounding_box[2])),True, False)

    if ds is not None:
        logger.debug("Using xarray dataset to get maximum and minimum time strings")
        file_time_min = ds["time"].min()
        file_time_max = ds["time"].max()
        time_vector = pd.date_range(file_time_min.values, 
                                    
                                    file_time_max.values, 
                                    freq="1h")
        time_condition = np.where((time_vector>=pd.to_datetime(min_time))
              & (time_vector<=pd.to_datetime(max_time)),True, False)
        valid_times = time_vector[time_condition]
    else:
        time_condition = np.where((time>=min_time)&(time<=max_time))
        valid_times = time[time_condition]

    valid_lat = lat[lat_condition]
    valid_lon = lon[lon_condition]
    
    dataarray_list = []
    for var in arrays:
        logger.debug(f"Processing variable {var}")
        data = store[var].get_orthogonal_selection((time_condition, lat_condition, lon_condition))
        temp_da = create_xarray_datarray(var, data, valid_times, valid_lat, valid_lon)
        dataarray_list.append(temp_da)

    if len(dataarray_list)>1:
        return xr.merge(dataarray_list)
    else:
        return dataarray_list[0]
    