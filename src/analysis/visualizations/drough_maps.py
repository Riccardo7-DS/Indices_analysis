import zarr
import pandas as pd
import numpy as np
import os
import xarray as xr
from utils.function_clns import create_xarray_datarray, config

from utils.function_clns import config 
import geopandas as gpd
import regionmask
from tqdm.auto import tqdm

def reduce_region(shapefile_path, location):
    gdf = gpd.read_file(shapefile_path)
    subset = gdf.loc[gdf["ADM0_NAME"].isin(location)]
    return subset

def extract_aggreg_map(vci, region, date_start, date_end, weights=False):
    from utils.function_clns import subsetting_pipeline
    vci = subsetting_pipeline(vci,location)
    cali_mask = regionmask.mask_3D_geopandas(region,
                                             vci.lon,
                                             vci.lat)
    two_month_vci = vci.sel(time=slice(date_start,date_end))

    vci_masked = two_month_vci.where(cali_mask)
    
    # Calculate the mean by region, but keep the lat/lon dimensions
    # region_means = (two_month_vci * cali_mask).groupby("region", squeeze=False).mean(dim="region", skipna=True)
    if weights is True:
        weights = np.cos(np.deg2rad(vci.lat))
        regional_mean = (vci_masked * weights).sum(dim=["lat", "lon"], skipna=True) / weights.sum(dim=["lat", "lon"], skipna=True)
    else:
        regional_mean = (vci_masked * cali_mask).sum(dim=["lat", "lon"], skipna=True) / cali_mask.sum(dim=["lat", "lon"], skipna=True)

    result = xr.full_like(two_month_vci, fill_value=np.nan)

    # Loop over each region and apply the regional mean back to the corresponding masked area
    for region in tqdm(range(cali_mask.shape[0])):
        mask = cali_mask[region]  # Get the mask for the current region
        result = result.where(~mask, other=regional_mean.isel(region=region))

    return result

def aggregate_vci(zarr_vci_path, region, min_time, max_time):
    
    ds_vci = xr.open_zarr(zarr_vci_path)   
    store = zarr.open(zarr_vci_path)
    var = "ndvi_clean"

    lat = store["lat"][:]
    lon = store["lon"][:]

    lat_condition = np.where(lat, True, False)
    lon_condition = np.where(lon, True, False)

    time_vector = ds_vci["time"]
    time_condition = np.where((time_vector>=pd.to_datetime(min_time))
          & (time_vector<=pd.to_datetime(max_time)),True, False)
    valid_times = time_vector[time_condition]

    data = store[var].get_orthogonal_selection((time_condition, lat_condition, lon_condition))

    temp_da = create_xarray_datarray(var, data, valid_times, lat, lon)

    result = extract_aggreg_map(temp_da, region,  min_time, max_time)
    return result

# def add_admin_data(dataset, shapefile):


# Function to mask and color the image
def mask_and_color(data_array, regions):
    mask = regions.mask(data_array)
    masked_array = data_array.where(mask.isnull())  # Set the masked area to null
    return masked_array  


def plot_vci_series(result, location, day=None, list_dates=None, crop_only = True):
    import matplotlib.pyplot as plt
    import pandas as pd
    import geopandas as gpd
    import regionmask
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from datetime import datetime, timedelta
    from utils.function_clns import config
    import cartopy.feature as cfeature
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
    def calculate_rows_cols(n):
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        return rows, cols
    # Load the shapefile
    shapefile_path = "/home/woden/Downloads/asap_warnings_gaul2_crop_2019_03_11.zip"
    shapefile_regions_mask = gpd.read_file(shapefile_path)
    if crop_only is True:
        shapefile_regions = shapefile_regions_mask[shapefile_regions_mask["w_crop"] == 99]
        subregions = regionmask.Regions(shapefile_regions.geometry, 
                                    overlap=False)
    
    if day is not None:
        n = 30
        end_time = pd.to_datetime(day) + timedelta(days=n)
        end_time = datetime.strftime(end_time, "%Y-%m-%d")
        temp_dataset = result.sel(time=slice(day, end_time))
    elif list_dates is not None:
        n = len(list_dates)
        dates_to_subset = np.array(list_dates, dtype='datetime64')
        temp_dataset = result.sel(time=dates_to_subset)
        day = list_dates[0]
        end_time = list_dates[-1]
    else:
        raise ValueError("either day or list_dates parameters must be not None")
    
# Calculate rows and columns based on n
    rows, cols = calculate_rows_cols(n)
    
    # Create a figure with 30 subplots (6 rows and 5 columns)
    figs, axess = plt.subplots(1, 5, figsize=(22, 18), 
                               subplot_kw=dict(projection=ccrs.Mercator()))
    
    axess = np.ravel(axess)

    levels = [0, 10, 20, 30, 40]
    orig_cmap = plt.get_cmap('Reds').reversed()
    # Create a modified colormap by slicing to exclude the first color
    newcolors = orig_cmap(np.linspace(0, 0.9, orig_cmap.N))
    cmap = LinearSegmentedColormap.from_list('Reds_modified', newcolors)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    reader = shpreader.Reader(config["SHAPE"]["africa_level2"])

    for i in range(n):
        ax = axess[i]  # Determine subplot position (row, col)
        # Check if the time index is within the range of the dataset
        if i < temp_dataset.time.size:
            image = temp_dataset.isel(time=i)
            masked_image = image.where(image <= 40)

            if crop_only is True:
                masked_image = mask_and_color(masked_image, subregions)
            plot = ax.pcolormesh(masked_image.lon.values, masked_image.lat.values,
                                 masked_image.values,
                                 transform=ccrs.PlateCarree(),
                                 cmap=cmap, norm=norm, shading="auto")
            # divider = make_axes_locatable(ax)
            # ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
            # figs.add_axes(ax_cb)
            # plt.colorbar(plot, cax=ax_cb)
            ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
            ax.add_feature(cfeature.BORDERS, alpha=1, edgecolor='black')
            ax.add_feature(cfeature.COASTLINE, alpha=0.1, linestyle='-')
            shape_feature = cfeature.ShapelyFeature(reader.geometries(),
                                                    ccrs.PlateCarree(),
                                                    facecolor='none',
                                                    edgecolor='grey',

                                                    linewidth=0.5)
            ax.add_feature(shape_feature)
            # Convert the time value to a datetime and then format it
            date_str = pd.to_datetime(temp_dataset.isel(time=i).time.values).strftime('%Y-%m-%d')
            ax.set_title(f'{date_str}', fontsize=16)  # Add a title for each subplot
        else:
            ax.set_visible(False)  # Hide the subplot if no data is available for the day
    for j in range(n, len(axess)):
        figs.delaxes(axess[j])

    
    # Create a new axis for the colorbar on the right side
    colorbar_height = 1.0 - 0.1 - 0.775  # top - bottom - hspace
    colorbar_width = 0.005  # width of the colorbar

    # Position for the colorbar axis
    cbar_ax = figs.add_axes([0.93, 0.425, colorbar_width, colorbar_height])  # [left, bottom, width, height]
 # [left, bottom, width, height]
    cbar = figs.colorbar(plot, cax=cbar_ax)
    cbar.set_label('VCI', fontsize=15) 

    cbar.ax.tick_params(labelsize=15)

    type_img = "crop" if crop_only else "rangeland"
    plt.subplots_adjust(wspace=0.3, hspace=0)  # Adjust wspace (width) and hspace (height) as needed
    dest_path = os.path.join(config["DEFAULT"]["images"], location, type_img)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    plt.savefig(os.path.join(dest_path, f"vci_plots_{day}_{end_time}.png"))
    plt.close()


if __name__== "__main__":
    import pyproj
    from datetime import datetime, timedelta
    import logging
    logger = logging.getLogger(__name__)
    os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()

    zarr_vci_path = os.path.join(config["DEFAULT"]["basepath"], "data_vci.zarr")
    shapefile_path = "/media/BIFROST/N2/Riccardo/Projects/Indices_analysis/src/shapefiles/africa_level2/afr_g2014_2013_2.shp"
    location = ["Somalia"]
    subset = reduce_region(shapefile_path, location)
    min_time = "2019-01-01"
    max_time = "2019-12-31"
    result = aggregate_vci(zarr_vci_path, subset,  min_time, max_time)

    dates = ["2019-01-06","2019-02-24", "2019-03-07","2019-03-26","2019-06-10"]
    plot_vci_series(result, location[0],  list_dates=dates)

    # start_dt = pd.to_datetime(min_time)
    # end_dt = pd.to_datetime(max_time)

    # while (end_dt - start_dt).days >= 30:
    #     day = datetime.strftime(start_dt, "%Y-%m-%d")
    #     logger.info(f"Generating images for day {day}")
    #     plot_vci_series(result, location[0], day, crop_only=False)
    #     plot_vci_series(result, location[0], day, crop_only=True)
    #     start_dt = start_dt + timedelta(days=30)
