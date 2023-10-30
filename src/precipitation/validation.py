import pandas as pd
import xarray as xr
from p_drought_indices.functions.function_clns import load_config, prepare, subsetting_pipeline, crop_get_spi, crop_get_thresh
import os
import numpy as np
import warnings
from typing import Union, Literal
import xskillscore as xs

warnings.filterwarnings('ignore')
            
def binary_rain(dataset:xr.Dataset, CONFIG_PATH:str)->xr.Dataset:
    variable = [var for var in dataset.data_vars if var!= "spatial_ref"][0]        
    bool_var = xr.where(dataset[variable]>1, 1,0)
    return subsetting_pipeline(CONFIG_PATH, dataset.assign(rain_bool=bool_var))


def get_eval_metrics(datarray_orig:xr.DataArray, datarray_frcst:xr.DataArray, dim:list=["lat","lon"]):
    """
    This is a function to calculate the target metrics for an xarray over a desired dimension, by default over lat/lon.
    The origin dataset is the reference, whereas forecasted is the dataset to evaluate
    Returns accuracy, far, pod, bias ration, rmse, mse.
    """
    dichotomous_category_edges = np.array([0, 0.5, 1])  # "dichotomous" mean two-category
    dichotomous_contingency = xs.Contingency(
        datarray_orig, datarray_frcst, dichotomous_category_edges, dichotomous_category_edges, dim=dim
        )
    accuracy = dichotomous_contingency.accuracy()
    far = dichotomous_contingency.false_alarm_rate()#.mean().values
    pod = dichotomous_contingency.hit_rate()#.mean().values
    fb  = dichotomous_contingency.bias_score()
    rmse = xs.rmse(datarray_orig, datarray_frcst,skipna=True, dim=dim)
    mse = xs.mse(datarray_orig,datarray_frcst,skipna=True, dim=dim)
    return accuracy, far, pod, fb, rmse, mse


def format_results(datarray_orig:xr.DataArray, datarray_frcst:xr.DataArray, dim:list=["lat","lon"])->pd.DataFrame:
    accuracy, far, pod, fb, rmse, mse = get_eval_metrics(datarray_orig, datarray_frcst, dim)
    metrics_df = pd.DataFrame([float(accuracy.mean(['time']).values), float(far.mean(['time']).values), float(pod.mean(['time']).values), float(rmse.mean(['time'])), \
        float(mse.mean(['time']).values),  float(fb.mean(['time']).values)]).T
    metrics_df.columns=['accuracy_m', 'far_m','pod_m', 'rmse_m', 'mse_m',"bias_m"]
    return metrics_df

def date_compat_check(xr_df, format_xr)->xr.Dataset:

    "the target dataset is the dataset to be reformatted"

    xr_df["time"] = pd.to_datetime(xr_df['time'].values).strftime('%Y-%m-%d')
    format_xr["time"] = pd.to_datetime(format_xr['time'].values).strftime('%Y-%m-%d')

    # Find the common time range between the two datasets
    common_time_range = xr.cftime_range(start=max(xr_df['time'].values.min(), format_xr['time'].values.min()),
                                   end=min(xr_df['time'].values.max(), format_xr['time'].values.max()))
    
    common_time_range = [i.strftime('%Y-%m-%d') for i in common_time_range]

    # Subset the datasets to the common time range
    dataset1_subset = xr_df.sel(time=common_time_range)
    dataset2_subset = format_xr.sel(time=common_time_range)

    return dataset1_subset, dataset2_subset

def loop_products(config):
    config_dir_precp = [config['PRECIP']['IMERG']['path'], config['PRECIP']['CHIRPS']['path'], config['PRECIP']['ERA5']['path'],  config['PRECIP']['TAMSTAT']['path'],
                    config['PRECIP']['MSWEP']['path']]

    dest_prod = config['PRECIP']['GPCC']['path']

    list_files = [f for f in os.listdir(dest_prod) if (f.endswith(".nc")) and ("merged" in f)]
    target_ds = xr.open_dataset(os.path.join(dest_prod, list_files[0]))
    var_target = [var for var in target_ds.data_vars if var!= "spatial_ref"][0]
    binary_target = binary_rain(target_ds, CONFIG_PATH)

    final_df = pd.DataFrame()
    for file in config_dir_precp:
        prod = file.split("/")[5]
        list_files = [f for f in os.listdir(file) if (f.endswith(".nc")) and ("merged" in f)]
        precp_ds = xr.open_dataset(os.path.join(file, list_files[0]))
        variable = [var for var in precp_ds.data_vars if var!= "spatial_ref"][0]
        precp_ds = prepare(precp_ds)
        repr_ds = precp_ds[variable].rio.reproject_match(prepare(target_ds)[var_target]).rename({"x":"lon","y":"lat"}).to_dataset()

        #####get binary dataset
        ds_rain = binary_rain(repr_ds, CONFIG_PATH)
        binary_target, ds_rain = date_compat_check( xr_df = binary_target , format_xr= ds_rain)
        metrics_df = format_results(binary_target["rain_bool"], ds_rain["rain_bool"])
        metrics_df["product"] = prod
        final_df = pd.concat([metrics_df, final_df], ignore_index=True)
        print(f"For the product {prod} the metrics are:")
        print(metrics_df)

    final_df.to_csv(os.path.join(config["DEFAULT"]["data"],"comparison_metrics.csv"))
    print(final_df)


from p_drought_indices.precipitation.validation import binary_rain, date_compat_check, format_results

def single_compare(ds_new:xr.Dataset, CONFIG_PATH)->pd.DataFrame:
    prod = ds_new.attrs["title"]
    config = load_config(CONFIG_PATH)
    dest_prod = config['PRECIP']['GPCC']['path']
    list_files = [f for f in os.listdir(dest_prod) if (f.endswith(".nc")) and ("merged" in f)]
    target_ds = xr.open_dataset(os.path.join(dest_prod, list_files[0]))
    var_target = [var for var in target_ds.data_vars if var!= "spatial_ref"][0]
    binary_target = binary_rain(target_ds, CONFIG_PATH)

    variable = [var for var in ds_new.data_vars if var!= "spatial_ref"][0]
    precp_ds = prepare(ds_new)
    repr_ds = precp_ds[variable].rio.reproject_match(prepare(target_ds)[var_target]).rename({"x":"lon","y":"lat"}).to_dataset()
    #####get binary dataset
    ds_rain = binary_rain(repr_ds, CONFIG_PATH)
    binary_target, ds_rain = date_compat_check( xr_df = binary_target , format_xr= ds_rain)
    metrics_df = format_results(binary_target["rain_bool"], ds_rain["rain_bool"])
    metrics_df["product"] = prod
    return metrics_df

if __name__=="__main__":
    CONFIG_PATH = "config.yaml"
    config = load_config(CONFIG_PATH)
    
    loop_products(config)



