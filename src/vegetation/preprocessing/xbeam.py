from utils.function_clns import extract_chunksize
from vegetation.preprocessing.ndvi_prep import SeviriMVCpipeline
import xarray as xr
import logging
import os

def beam_pipeline():
    from utils.function_clns import config
    from datetime import datetime

    logger = logging.getLogger()
    time_format = "%Y-%m-%d %H:%M:%S"

    now = datetime.now()
    current_time = now.strftime(time_format)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', 
                                  datefmt=time_format)
    handler = logging.FileHandler(f"./output/log/sevirimax_log_{current_time}.log")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
                
    base_path = config["DEFAULT"]["basepath"]

    datasets = SeviriMVCpipeline()
    chunks = {"time":-1, "lat": 80, "lon":80}
    ds_final = xr.concat([datasets[0], 
                          datasets[1]], dim="time").to_dataset(name="ndvi")
    
    origin_chunks = extract_chunksize(ds_final)
    output_zarr = os.path.join(base_path, "seviri_daily_maximum.zarr")

    import xarray_beam as xbeam
    import apache_beam as beam
    template = xbeam.make_template(ds_final).groupby("time").first(["time"])
    logging.info("Created template")

    try:
        with beam.Pipeline() as p:
            (
                p
                | xbeam.DatasetToChunks(ds_final, origin_chunks) 
                | xbeam.Rechunk(ds_final.sizes, origin_chunks, chunks, itemsize=4)
                # insert additional transforms here
                | beam.MapTuple(lambda k, v: (k.with_offsets(time=None), v.groupby("time").max('time')))
                | xbeam.ChunksToZarr(output_zarr, template, chunks)

            )
    except Exception as e:
        logger.error(e, stack_info=True, exc_info=True)


if __name__ == "__main__":
    beam_pipeline()