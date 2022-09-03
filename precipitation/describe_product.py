import xarray as xr
from merge_daily_data import load_config
import os

CONFIG_PATH = r"./config.yaml"


config = load_config(CONFIG_PATH)
ds = xr.open_dataset(os.path.join(config['PRECIP']['imerg_path'],'imerg_final_clipped_sek.nc'))
#print(ds)

time_max = ds["time"].where(ds==ds.max("time")).max("time")
print('The most recent time date is:', time_max['spatial_ref'].values)


time_min = ds["time"].where(ds==ds.min("time")).min("time")
print('The oldest time date is:', time_min['spatial_ref'].values)