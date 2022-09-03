import yaml
from shapely.geometry import Polygon, mapping

def load_config(CONFIG_PATH):
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)
        return config

def cut_file(xr_df, gdf):
    xr_df.rio.set_spatial_dims(x_dim='lat', y_dim='lon', inplace=True)
    xr_df.rio.write_crs("epsg:4326", inplace=True)
    clipped = xr_df.rio.clip(gdf.geometry.apply(mapping), gdf.crs, drop=True)
    return clipped