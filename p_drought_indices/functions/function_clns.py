import yaml
from shapely.geometry import Polygon, mapping
import geopandas as gpd
import shapely

def load_config(CONFIG_PATH):
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)
        return config

def cut_file(xr_df, gdf):
    xr_df.rio.set_spatial_dims(x_dim='lat', y_dim='lon', inplace=True)
    xr_df.rio.write_crs("epsg:4326", inplace=True)
    clipped = xr_df.rio.clip(gdf.geometry.apply(mapping), gdf.crs, drop=True)
    return clipped

def subsetting_pipeline(CONFIG_PATH, xr_df, countries = ['Ethiopia','Kenya', 'Somalia'], invert=True):
    config = load_config(CONFIG_PATH)
    shapefile_path = config['SHAPE']['africa']
    gdf = gpd.read_file(shapefile_path)
    subset = gdf[gdf.ADM0_NAME.isin(countries)]
    if invert==True:
        subset = subset['geometry'].map(lambda polygon: shapely.ops.transform(lambda x, y: (y, x), polygon))
    return cut_file(xr_df, subset)