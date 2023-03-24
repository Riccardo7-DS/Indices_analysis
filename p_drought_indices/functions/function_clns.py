import yaml
from shapely.geometry import Polygon, mapping
import geopandas as gpd
import shapely
import xarray as xr
import glob

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