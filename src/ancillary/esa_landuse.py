from utils.function_clns import load_config, prepare, subsetting_pipeline
import geopandas as gpd
import os
import xarray as xr
#from osgeo import gdal
import matplotlib.pyplot as plt
import pandas as pd
from rasterio.enums import Resampling
from typing import Union, Literal
import logging 
logger = logging.getLogger(__name__)

values_land_descr = {0	:'Unknown', 20:	'Shrubland',30:'Herbaceous vegetation',40:	'Cropland',
                        50:	'Built-up',60:	'Bare sparse vegetation',70:'Snow and ice', 
                        80:	'Permanent water bodies',
                        90:'Herbaceous wetland',
                        100: 'Moss and lichen', 11:"Closed forest", 
                        12: "Open forest,", 200: "Oceans, seas"}


def get_description(df:pd.DataFrame, column:str):
    values_land_cover = {0	:'Unknown', 20:	'Shrubs',30:	'Herbaceous vegetation',40:	'Cultivated and managed vegetation/agriculture',
                        50:	'Urban',60:	'Bare',70:	'Snow and ice',80:	'Permanent water bodies',90:	'Herbaceous wetland',100: 'Moss and lichen',111: 'Closed forest, evergreen needle leaf',
                        112: 'Closed forest, evergreen broad leaf',115: 'Closed forest, mixed',125: 'Open forest, mixed',113: 'Closed forest, deciduous needle leaf',
                        114: 'Closed forest, deciduous broad leaf',116: 'Closed forest, not matching any of the others',121: 'Open forest, evergreen needle leaf',122: 'Open forest, evergreen broad leaf',
                        123: 'Open forest, deciduous needle leaf',124: 'Open forest, deciduous broad leaf',126: 'Open forest, not matching any of the others',200: 'Oceans, seas'}

    df['description'] = df[column].replace(values_land_cover.keys(),values_land_cover.values())
    return df

def prepare_covermask(dataset:xr.Dataset):
        cover_ds = create_copernicus_covermap(dataset)
        cover_ds = subsetting_pipeline(cover_ds)
        cover_ds["lat"] = dataset["lat"]
        cover_ds["lon"] = dataset["lon"]
        return get_level_1(cover_ds, name="band_data").isel(band=0)

def create_copernicus_covermap(dataset:xr.DataArray, 
                               crop_strategy:Literal["shapefile", "geometry","dataset", None]="shapefile",
                               export:bool=True):
    import geemap    
    import ee
    from utils.function_clns import config, prepare
    import geopandas as gpd
    from utils.xarray_functions import geobox_from_rio
    import logging

    def generate_new_data(dataset, crop_strategy:str):
        logger.info("Generating new landcover dataset")

        ee.Authenticate()
        ee.Initialize()
        geemap.ee_initialize()
        epsg_coords ='EPSG:4326'

        geobox = geobox_from_rio(prepare(dataset))
        dataset_geometry = ee.Geometry.Rectangle(dataset.rio.bounds())
        proj = ee.Projection(crs=str(dataset.rio.crs), 
                     transform=dataset.rio.transform()[:6])

        landcover = ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019")\
            .select("discrete_classification")\
            .reproject(crs = epsg_coords, crsTransform=list(geobox.transform)[:6])\
            .clip(dataset_geometry)

        if crop_strategy =="shapefile":
            gdf = gpd.read_file(shapefile_path)
            fc_poly = geemap.geopandas_to_ee(gdf)
            poly_geometry = fc_poly.geometry()

        elif crop_strategy == "dataset":
            poly_geometry = dataset_geometry

        elif crop_strategy == "geometry":
            poly_geometry = ee.Geometry.Rectangle(30.288233396779802,  -5.949173816626356 , 
                                             51.9972177717798, 15.808293611760663)
        elif crop_strategy is None:
            logging.info("No cropping strategy selected")

        if export is True:
            geemap.ee_export_image(landcover, 
                    filename = filename, 
                    region = poly_geometry)
            return prepare(xr.open_dataset(filename, engine="rasterio"))
        else:
            ds = xr.open_dataset(ee.ImageCollection(landcover), engine="ee", 
                                           projection=proj)
            repr_ds =  prepare(ds.discrete_classification.transpose("time","lat","lon"))\
                .rio.reproject_match(prepare(dataset)).isel(time=0)
            return prepare(repr_ds)
    
    shapefile_path = config['SHAPE']['HOA']
    path_img = config["DEFAULT"]["images"]
    filename = os.path.join(path_img, "temp_cover.tif")

    if os.path.isfile(filename):
        ds = prepare(xr.open_dataset(filename, 
                                     engine="rasterio"))

        if ds.rio.resolution()[0] == prepare(dataset).rio.resolution()[0] and \
            len(ds["lat"])==len(dataset["lat"]) and len(ds["lon"])==len(dataset["lon"]):
            logger.info("Loading extisting landcover dataset")
            return ds
        else:
            ds = generate_new_data(dataset, crop_strategy)
            return ds
    else:
        ds = generate_new_data(dataset, crop_strategy)
        return ds

def get_level_colors(ds_cover, level1=True):
    df = ds_cover.to_dataframe()
    df = df.reset_index()
    df = df.dropna(subset=['Band1'])
    df['Band1'] = df['Band1'].astype(int)

    if level1==False:

        values_land_cover = {0	:'Unknown', 20:	'Shrubs',30:	'Herbaceous vegetation',40:	'Cultivated and managed vegetation/agriculture',
                        50:	'Urban',60:	'Bare',70:	'Snow and ice',80:	'Permanent water bodies',90:	'Herbaceous wetland',100: 'Moss and lichen',111: 'Closed forest, evergreen needle leaf',
                        112: 'Closed forest, evergreen broad leaf',115: 'Closed forest, mixed',125: 'Open forest, mixed',113: 'Closed forest, deciduous needle leaf',
                        114: 'Closed forest, deciduous broad leaf',116: 'Closed forest, not matching any of the others',121: 'Open forest, evergreen needle leaf',122: 'Open forest, evergreen broad leaf',
                        123: 'Open forest, deciduous needle leaf',124: 'Open forest, deciduous broad leaf',126: 'Open forest, not matching any of the others',200: 'Oceans, seas'}

        colors = {0:'#282828',20:'#FFBB22',30:'#FFFF4C',40:'#F096FF',50:'#FA0000',60:'#B4B4B4',70:'#F0F0F0',80:'#0032C8',90:'#0096A0',100:'#FAE6A0',111:'#58481F',112:'#009900',113:'#70663E',
        114:'#00CC00',115:'#4E751F',116:'#007800',121:'#666000',122:'#8DB400',123:'#8D7400',124:'#A0DC00',125:'#929900',126:'#648C00',200:'#000080'}

    else:
        values_land_cover = {0	:'Unknown', 20:	'Shrubland',30:'Herbaceous vegetation',40:	'Cropland',
                        50:	'Built-up',60:	'Bare sparse vegetation',70:'Snow and ice', 80:	'Permanent water bodies',
                        90:'Herbaceous wetland',100: 'Moss and lichen', 11:"Closed forest", 
                        12: "Open forest,", 200: "Oceans, seas"}

        colors = {0:'#282828',20:'#FFBB22',30:'#FFFF4C',40:'#F096FF',50:'#FA0000',60:'#B4B4B4',70:'#F0F0F0',80:'#0032C8',90:'#0096A0',100:'#FAE6A0',11:'#58481F',
                  12:'#666000', 200:'#000080'}
        
    df['colors'] = df['Band1'].replace(colors.keys(),colors.values())
    df['description'] = df['Band1'].replace(values_land_cover.keys(),values_land_cover.values())

    cmap = df.sort_values('Band1')['colors'].unique().tolist()
    #cmap = [c.lower() for c in cmap]
    levels = df.sort_values('Band1')['Band1'].unique().tolist()
    return cmap, levels, values_land_cover

def visualize_map(land_proj):
    import ee 
    import geemap
    Map = geemap.Map(center=(5, 40), zoom=5)
    Map.addLayer(land_proj, {}, 'Land cover')
    Map

def export_land_cover(target_resolution:str, 
                      export_path =r'../data/images'):
    import ee 
    import geemap
    from osgeo import gdal
    assert target_resolution in ['IMERG','CHIRPS']
    ee.Authenticate()
    ee.Initialize()
    landcover = ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019").select('discrete_classification')
    
    if target_resolution=="IMERG":
        range = ee.Date('2019-09-03').getRange('month')
        
        imerg = ee.ImageCollection('NASA/GPM_L3/IMERG_V06').filter(ee.Filter.date(range)).select('precipitationCal').first()
        land_proj = landcover.reproject(imerg.projection(), None, imerg.projection().nominalScale())

    elif target_resolution =="CHIRPS":
        landcover = ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019").select('discrete_classification')
        chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filter(ee.Filter.date('2018-05-01', '2018-05-02'))\
           .select('precipitation').first()
        #reproj_land = landcover.reproject(crs='EPSG:4326', crsTransform=[1, 0, 0, 0, 1, 0])
        land_proj = landcover.reproject(chirps.projection(), None, chirps.projection().nominalScale())

    from utils.function_clns import config
    shapefile_path = config['SHAPE']['HOA']
    gdf = gpd.read_file(shapefile_path)

    fc_poly = geemap.geopandas_to_ee(gdf)
    poly_geometry = fc_poly.geometry()
    path_img = export_path
    geemap.ee_export_image(land_proj, 
                           filename = os.path.join(path_img, "esa_cover.tif"),
                           region = poly_geometry)
 
    #Change the following variables to the file you want to convert (inputfile) and
    #what you want to name your output file (outputfile).
    inputfile = os.path.join(path_img, "esa_cover.tif")
    outputfile = os.path.join(path_img, "esa_cover.nc")
    #Do not change this line, the following command will convert the geoTIFF to a netCDF
    ds = gdal.Translate(outputfile, inputfile, format='NetCDF')
    cover_ds = xr.open_dataset(os.path.join(path_img, "esa_cover.nc"))

    df = cover_ds.to_dataframe()
    df = df.reset_index()
    df['Band1'] = df['Band1'].astype(int)

    values_land_cover = {0	:'Unknown', 20:	'Shrubs',30:	'Herbaceous vegetation',40:	'Cultivated and managed vegetation/agriculture',
                        50:	'Urban',60:	'Bare',70:	'Snow and ice',80:	'Permanent water bodies',90:	'Herbaceous wetland',100: 'Moss and lichen',111: 'Closed forest, evergreen needle leaf',
                        112: 'Closed forest, evergreen broad leaf',115: 'Closed forest, mixed',125: 'Open forest, mixed',113: 'Closed forest, deciduous needle leaf',
                        114: 'Closed forest, deciduous broad leaf',116: 'Closed forest, not matching any of the others',121: 'Open forest, evergreen needle leaf',122: 'Open forest, evergreen broad leaf',
                        123: 'Open forest, deciduous needle leaf',124: 'Open forest, deciduous broad leaf',126: 'Open forest, not matching any of the others',200: 'Oceans, seas'}

    colors = {0:'#282828',20:'#FFBB22',30:'#FFFF4C',40:'#F096FF',50:'#FA0000',60:'#B4B4B4',70:'#F0F0F0',80:'#0032C8',90:'#0096A0',100:'#FAE6A0',111:'#58481F',112:'#009900',113:'#70663E',
    114:'#00CC00',115:'#4E751F',116:'#007800',121:'#666000',122:'#8DB400',123:'#8D7400',124:'#A0DC00',125:'#929900',126:'#648C00',200:'#000080'}

    df['colors'] = df['Band1'].replace(colors.keys(),colors.values())
    df['description'] = df['Band1'].replace(values_land_cover.keys(),values_land_cover.values())

    cmap = df.sort_values('Band1')['colors'].unique().tolist()
    cmap = [c.lower() for c in cmap]
    levels = df.sort_values('Band1')['Band1'].unique().tolist()
    cover_ds['Band1'].plot(levels = levels, colors=cmap)

    plt.show()

    return land_proj, cover_ds



def get_level_1(ds:xr.DataArray, name:str="Band1")->xr.Dataset:
    values_land_cover = {0	:0, 20:	20, 30:30, 40:40, 50:50, 60:60, 70:	70 ,80:	80,90:90, 100: 100, 
                        111:11, 112: 11,115: 11,125: 12,113: 11, 114: 11,116: 11,
                        121: 12,122: 12, 123: 12,124: 12,126: 12, 200: 200}
    
    values_land_descr = {0	:'Unknown', 20:	'Shrubland',30:'Herbaceous vegetation',40:	'Cropland',
                        50:	'Built-up',60:	'Bare sparse vegetation',70:'Snow and ice', 80:	'Permanent water bodies',
                        90:'Herbaceous wetland',100: 'Moss and lichen', 11:"Closed forest", 
                        12: "Open forest,", 200: "Oceans, seas"}
    
    if isinstance(ds, xr.DataArray):
        df = ds.to_dataframe(name=name)
    elif isinstance(ds, xr.Dataset):
        df = ds[name].to_dataframe(name=name)
    df["level1"] = df[name].replace(values_land_cover.keys(),values_land_cover.values())
    ds = df["level1"].to_xarray().to_dataset()
    return ds.rename({"level1":name})

def get_cover_dataset(datarray:xr.DataArray, img_path:str, 
                      img_name:str="esa_cover.nc",level1=True, 
                      resample=True)->xr.Dataset:
    
    ds_cover = prepare(xr.open_dataset(os.path.join(img_path, img_name)))
    if level1==True:
        ds_cover = get_level_1(ds_cover)
    ds_cover = subsetting_pipeline(ds_cover)
    if resample ==True:
        logger.info(f"Starting reprojection to destionation resolution of {ds_cover.rio.resolution()}")
        ds = datarray.rio.reproject_match(ds_cover["Band1"],
                    resampling = Resampling.mode).rename({"x":"lon","y":"lat"})
        ds = ds.to_dataset().assign(Band1=ds_cover["Band1"])
    
    else:
        ds = datarray.to_dataset().assign(Band1=ds_cover["Band1"])
    ds["Band1"] = ds["Band1"].expand_dims({"time":len(ds["time"])})
    return ds

def drop_water_bodies_copernicus(dataset: Union[xr.Dataset, xr.DataArray], 
                                 preprocess=None):
    cover_ds = create_copernicus_covermap(dataset, 
                                          crop_strategy="dataset", 
                                          export=False)
    if preprocess is not None:
        cover_ds = cover_ds.apply(preprocess)
    cover_ds = get_level_1(cover_ds, name="band_data")
    if "band" in cover_ds.dims:
        cover_ds = cover_ds.isel(band=0)
    water_bodies = xr.where(
        cover_ds["band_data"].isin([80, 200]),
        1, 0 ).transpose("lat","lon")
    return dataset.where(water_bodies==0)

def drop_water_bodies_esa_downsample(ds):
    from ancillary.esa_landuse import get_cover_dataset, get_level_1
    from rasterio.enums import Resampling
    from utils.function_clns import config

    img_path = os.path.join(config["DEFAULT"]["images"], "chirps_esa")
    ds_cover = prepare(xr.open_dataset(os.path.join(img_path, "esa_cover.nc")))
    ds_cover = subsetting_pipeline(get_level_1(ds_cover))
    ds_cover = ds_cover["Band1"].rio.reproject_match(ds,
                        resampling = Resampling.mode).rename({"x":"lon","y":"lat"})
    if "time" in ds.dims:
        ds_cover = ds_cover.expand_dims({"time":len(ds["time"])})
    water_mask = xr.where((ds_cover==80) | (ds_cover==200), 1,0)
    return ds.where(water_mask==0)

if __name__ == "__main__":
    from ancillary.esa_landuse import export_land_cover
    export_land_cover(target_resolution="CHIRPS",
                    export_path =r'../data/images/chirps_esa')

   
