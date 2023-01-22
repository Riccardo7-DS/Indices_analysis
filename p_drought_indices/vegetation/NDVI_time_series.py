from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from glob import glob
import os
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from glob import glob
import os
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import time
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import yaml
from p_drought_indices.functions.function_clns import load_config, subsetting_pipeline
from p_drought_indices.functions.ndvi_functions import add_time



if __name__=="__main__":
    base_dir = r'D:\shareVM\MSG\cloudmask\processed_clouds\batch_2\nc_files'
    files = [f for f in os.listdir(base_dir) if f.endswith('.nc')]

    for file in files:
        xr_df = xr.open_dataset(os.path.join(base_dir, file))
        print(xr_df.attrs)
        ds = add_time(xr_df)
        ds.to_netcdf(os.path.join(base_dir,'new',file))
