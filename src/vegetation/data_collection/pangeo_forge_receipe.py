from pangeo_forge_recipes.patterns import ConcatDim, FilePattern
from pangeo_forge_recipes.transforms import PrepareZarrTarget, OpenURLWithFSSpec, OpenWithXarray, StoreToZarr
# import zarr
import apache_beam as beam
# import requests
import zipfile
from io import BytesIO
# import rioxarray as rio
from utils.function_clns import config 
import os
import pandas as pd
import fsspec
from typing import Union, Optional, Tuple
import aiohttp
# from zipfile import ZipFiles
import io
from dataclasses import dataclass, field
from pangeo_forge_recipes.transforms import _add_keys, MapWithConcurrencyLimit
from pangeo_forge_recipes.openers import OpenFileType
import pangeo_forge_recipes
from datetime import datetime

@dataclass
class UnzipFilter(beam.PTransform):
    
    num: Optional[int] = 1
    file_format: Optional[str] = None
    file_name: Optional[str] = None
    file_substring: Optional[str] = None

    def expand(self, pcoll):
        refs = pcoll  | "Unzip and filter" >> beam.Map(
            _unzip_and_filter,
            num=self.num,
            file_format=self.file_format,
            file_name=self.file_name,
            file_substring=self.file_name,
        )
        return refs

def _unzip_and_filter(
        response: Tuple[pangeo_forge_recipes.types.Index, OpenFileType], 
        num:int=1, 
        file_format:Union[None,str]=None,
        file_name:Union[None,str]=None, 
        file_substring:Union[None,str]=None):
    
    import io
    with response[1] as f:
        zip_contents = f.read()

    # Step 2: Create a BytesIO object to treat the contents as an in-memory file
    zip_buffer = io.BytesIO(zip_contents)

    # Step 3: Use zipfile to extract the files from the in-memory buffer
    with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
        # Filter files based on the specified pattern
        if file_name is not None:
            zip_file_list = [file for file in zip_ref.namelist() 
                             if file_name == file]
        elif file_substring is not None:
            zip_file_list = [file for file in zip_ref.namelist() 
                             if file_substring in file]
        elif file_format is not None:
            zip_file_list = [file for file in zip_ref.namelist() 
                             if file.endswith(file_format)]

        if num ==1:
            zip_ref.read(zip_file_list[0]) 
        else:
            raise NotImplementedError

def unzip_response(response):
    if response.status_code == 200:
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            # List all files in the zip archive
            file_list = z.namelist()
            # Select the file ending with "africa.tif"
            africa_tif_file = [file for file in file_list if file.endswith("AFR_NDV.tif")]

            if africa_tif_file:
                # # Extract the selected file
                # print(f"\nFound 'africa.tif' file: {africa_tif_file[0]}")
                return rio.open_rasterio(BytesIO(z.read(africa_tif_file[0])))
                #return  z.read(africa_tif_file[0])
                #return fsspec.filesystem("memory").open(f"{africa_tif_file[0]}", mode="wb", auto_mkdir=True).write(z.read(africa_tif_file[0]))
            else:
                # print("\nNo 'africa.tif' file found in the zip archive.")
                return None
    else:
        print(f"Failed to extract zip file. Status code: {response.status_code}")

def make_url(time):
    url = URL_FORMAT.format(time=time, day=time.day, 
                             month=time.month, 
                             year=time.year)
    return url

from pangeo_forge_recipes.patterns import ConcatDim, FilePattern
from pangeo_forge_recipes.transforms import PrepareZarrTarget, Indexed, T, OpenWithXarray, StoreToZarr
import apache_beam as beam
import zipfile
from io import BytesIO
import rioxarray as rio
from utils.function_clns import config, subsetting_pipeline, prepare
import os
import pandas as pd
import fsspec
import xarray as xr

import numpy as np 

days = [1, 11, 21]
months = np.arange(1, 13)
years = np.arange(2008, 2021)

# Generate all combinations of dates
all_dates = []

for year in years:
    for month in months:
        for day in days:
            date = f"{year}-{month:02d}-{day:02d}"
            all_dates.append(date)

dates = [datetime.strptime(date, "%Y-%m-%d") for date in all_dates]

username = config["LANDSAF"]["user"]
password = config["LANDSAF"]["password"]

path = "/media/BIFROST/N2/Riccardo/output"
target_store = "output_file.zarr"

URL_FORMAT = (
    "https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/EPS/ENDVI10/ENVI/"
    "{year:4d}/{month:02d}/{day:02d}/METOP_AVHRR_{time:%Y%m%d}_S10_AFR_V200.zip"
)

time_concat_dim = ConcatDim("time", dates, nitems_per_file=1)
pattern = FilePattern(make_url, time_concat_dim, file_type="netcdf4") #, fsspec_open_kwargs=secrets)


class Preprocess(beam.PTransform):
    """Preprocessor transform."""

    @staticmethod
    def _preproc(item: Indexed[T]) -> Indexed[xr.Dataset]:
        import numpy as np
        import requests

        index, url = item
        time_dim = index.find_concat_dim('time')
        time_index = index[time_dim].value
        time = dates[time_index]
        
        response = requests.get(url, auth=(username, password))
        file = unzip_response(response).isel(band=0)
        da = file.rename({'x': 'lon', 'y': 'lat'})
        ds = da.to_dataset(name='ndvi_10')
        ds = ds.expand_dims(time=np.array([time]))

        return index, ds

    def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
        return pcoll | beam.Map(self._preproc)


recipe = (
    beam.Create(pattern.items())
    | Preprocess()
    | StoreToZarr(
        target_root=path,
        store_name=target_store,
        combine_dims=pattern.combine_dim_keys,
        # target_chunks={'time': 48, 'lat': 32, 'lon': 32},
    )
)
from apache_beam.pipeline import PipelineOptions
options = PipelineOptions(runner="DirectRunner")
with beam.Pipeline() as p:
    p | recipe