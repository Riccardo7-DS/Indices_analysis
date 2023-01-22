from p_drought_indices.functions.function_clns import load_config, cut_file
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from glob import glob
import os
import datetime as datetime
from p_drought_indices.precipitation.SPI_wet_dry import spiObject

def subsetting_pipeline(CONFIG_PATH, xr_df, countries = ['Ethiopia','Somalia','Kenya']):
    config = load_config(CONFIG_PATH)
    shapefile_path = config['SHAPE']['africa']
    gdf = gpd.read_file(shapefile_path)
    subset = gdf[gdf.ADM0_NAME.isin(countries)]
    return cut_file(xr_df, subset)

def print_raster(raster):
    print(f"Product: {raster}\n"
        f"shape: {raster.rio.shape}\n"
        f"resolution: {raster.rio.resolution()}\n"
        f"bounds: {raster.rio.bounds()}\n"
        #f"sum: {raster.sum().item()}\n"
        f"CRS: {raster.rio.crs}\n"
    )

def transpose_dataset(xr_df, var1='time', var2='lat', var3='lon'):
    return xr_df.transpose(var1, var2, var3)


def regridding_product(var, product_directory, product_dir_target):
    final_path = []
    file  = [f for f in os.listdir(product_directory) if f.endswith('.nc') and var in f]
    final_path = os.path.join(product_directory,file[0])

    ### Choose destination product
    file = [f for f in os.listdir(product_dir_target) if f.endswith('.nc') and var in f]
    xds_target = transpose_dataset(xr.open_dataset(os.path.join(product_dir_target,file[0])))
    xr_dest = subsetting_pipeline(xds_target)
    xr_dest.rio.write_crs("epsg:4326", inplace=True)
    
    ds =  xr.open_dataset(final_path)
    xr_df = transpose_dataset(subsetting_pipeline(ds))
    xr_df.rio.write_crs("epsg:4326", inplace=True)

    ### Reproject datarray
    xds_repr_match = xr_df[var].rio.reproject_match(xr_dest[var])
    return xds_repr_match

def date_compat_check(xr_df, target_xr):
    min_target = target_xr['time'].min().values
    max_target = target_xr['time'].max().values
    min_xr = xr_df['time'].min().values
    max_xr = xr_df['time'].max().values

    if (min_target < min_xr) & (max_target>max_xr):
        return target_xr.sel(time=slice(min_xr, max_xr))
    elif (min_target > min_xr) & (max_target>max_xr):
        return target_xr.sel(time=slice(min_target, max_xr))
    elif (min_target < min_xr) & (max_target<max_xr):
        return target_xr.sel(time=slice(min_xr, max_target))
    else:
        print('The target dataset has inferior dimensions than the origin')
        return target_xr

def regridding_pipeline(var, product_directories, product_dir_target):
    for product_dir in product_directories:
        xr_regrid = regridding_product(var, product_dir, product_dir_target)
        file  = [f for f in os.listdir(product_dir) if f.endswith('.nc') and var in f]
        xr_df = xr.open_dataset(os.path.join(product_dir,file[0]))
        xr_df['regrid_{}'.format(var)] = xr_regrid
        xr_df.to_netcdf(os.path.join(product_dir,'regridded',file[0]))

def matching_pipeline(var, product_directories, product_dir_target):
    file_base = [f for f in os.listdir(product_dir_target) if f.endswith('.nc') and var in f]
    base_path = os.path.join(product_dir_target,file_base[0])
    base_xr = xr.open_dataset(base_path)

    for product_dir in product_directories:
        target_dir = os.path.join(product_dir, 'regridded')
        file  = [f for f in os.listdir(target_dir) if f.endswith('.nc') and var in f]
        xr_df = xr.open_dataset(os.path.join(target_dir,file[0]))
        xr_df = xr_df.drop_dims(['lat','lon']).rename({'x':'lon', 'y':'lat'})
        xr_df = xr_df.rename({'regrid_{}'.format(var): var})
        target_new = date_compat_check(xr_df, base_xr)
        masked_data = xr_df.where(target_new.notnull())
        target_path = os.path.join(target_dir, 'matched', file[0])   
        masked_data.to_netcdf(target_path)
        far, pod, accuracy = precp_indices_pipeline(target_path, base_path, target_new)


def pod_index(ds, ds_t, var):
    A_var = ds_t[var] == ds[var]
    C_var = (ds[var] == 1) & (A_var == False)
    pod = A_var/(A_var+C_var)
    return pod

def far_index(ds, ds_t, var):
    A_var = ds_t[var] == ds[var]
    B_var = (ds_t[var] == 1) & (A_var == False)
    far = B_var/(B_var+A_var)
    return far

def accuracy_index(ds, ds_t, var):
    A_var = ds_t[var] == ds[var]
    B_var = (ds_t[var] == 1) & (A_var == False)
    C_var = (ds[var] == 1) & (A_var == False)
    accuracy = 2*A_var/(2*A_var+B_var+C_var)
    return accuracy

def compute_indixes(ds, ds_t, var):
    far = far_index(ds, ds_t, var)
    pod = pod_index(ds, ds_t, var)
    accuracy = accuracy_index(ds, ds_t, var)
    return far, pod, accuracy

def plot_indices(spi_target, spi_base, far, pod, accuracy, save):
    t = spi_base.product[:-1]
    b = spi_target.product[:-1]
    g = spi_base.abbrev
    pod.mean(['time']).plot()
    plt.title('Compute mean POD between {t} and {b} for {g}'.format(t= t, b= b, g=g))
    if save == True:
        plt.savefig('precipitation/images/pod_{t}_{b}_{g}.png'.format(t= t, b= b, g=g))
    plt.show()
    far.mean(['time']).plot()
    plt.title('Compute mean FAR between {t} and {b} for {g}'.format(t= t, b= b, g=g))
    if save == True:
        plt.savefig('precipitation/images/far_{t}_{b}_{g}.png'.format(t= t, b= b, g=g))
    plt.show()
    accuracy.mean(['time']).plot()
    plt.title('Compute mean accuracy between {t} and {b} for {g}'.format(t= t, b= b, g=g))
    if save == True:
        plt.savefig('precipitation/images/acrcy_{t}_{b}_{g}.png'.format(t= t, b= b, g=g))
    plt.show()

def precp_indices_pipeline(target_path, base_path, target_new, extreme='drought', plot=True, save=True):
    if extreme == "drought":
        var = 'dry'
    elif extreme == "flood":
        var == 'wet'
    else: raise ValueError('The chosen extreme needs to be either \'drought\' or \'flood\'')
    
    ### initialize an spi object for the regridded dataset
    spi_target =  spiObject(target_path)
    spi_target.xr = xr.open_dataset(target_path)
    spi_ = spi_target.calculate_points_xr()

    ### initialize a new object for the target dataset
    spi_base = spiObject(base_path)
    spi_base.xr =  target_new
    gpcc_ = spi_base.calculate_points_xr()

    ###subset the image
    ds = subsetting_pipeline(gpcc_)
    ds_t = subsetting_pipeline(spi_)

    ### compute the indices
    far, pod, accuracy = compute_indixes(ds, ds_t, var)
    print('The mean POD for {v} product {p} at latency {l} is'.format(v=extreme, p=spi_target.product, l= spi_target.freq ), pod.mean(['time','lat','lon']).values)
    print('The mean FAR for {v} product {p} at latency {l} is'.format(v=extreme, p=spi_target.product, l= spi_target.freq ), far.mean(['time','lat','lon']).values)
    print('The mean accuracy for {v} product {p} at latency {l} is'.format(v=extreme, p=spi_target.product, l= spi_target.freq ), accuracy.mean(['time','lat','lon']).values)
    if plot == True:
        plot_indices(spi_base, spi_target, far, pod, accuracy, save=save)
    return far, pod, accuracy

if __name__=="__main__":
    
    CONFIG_PATH = r"./config.yaml"
    config = load_config(CONFIG_PATH)
    
    var = 'spi_gamma_60'
    product_dir_target = config['SPI']['GPCC']['path']
    product_directories = [config['SPI']['IMERG']['path'], config['SPI']['CHIRPS']['path'],  config['SPI']['ERA5']['path']]
    #regridding_pipeline(var, product_directories, product_dir_target)
    matching_pipeline(var, product_directories, product_dir_target)


    
    



