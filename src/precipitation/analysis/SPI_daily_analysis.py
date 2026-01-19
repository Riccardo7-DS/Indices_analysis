import xarray as xr
import os
import pandas as pd
import time
import geopandas as gpd
from shapely.geometry import Polygon, mapping


def compute_dry_wet(path_spi, countries,  year_start, year_end, aggr='season'):
    if aggr not in [None, 'season','month','day','year']:
        raise ValueError("The specified aggregation time is not correct")
    spi = spiObject(path_spi)
    shapefile_path = config['SHAPE']['africa']

    #### Chose subset of countries
    gdf = gpd.read_file(shapefile_path)
    subset = gdf[gdf.ADM0_NAME.isin(countries)]

    ###open file and cut it
    spi.xr = cut_file(xr.open_dataset(path_spi, chunks={"lat": -1, "lon": -1, "time": 12}), subset)
    
    res_xr = cut_file(spi.calculate_points_xr(), subset)
    print('The final product has the following variables:', res_xr.data_vars)

    if aggr==None:
        product_df = pd.DataFrame()
        for country in countries:
            subset = gdf[gdf.ADM0_NAME.isin([country])]
            tempo_xr = cut_file(res_xr, subset)
            res_df = res_xr[['time','lat','lon','dry','wet']].to_dataframe()
            res_df['country'] = country
            res_df['product'] = spi.product
            res_df['latency'] = spi.abbrev
            product_df = product_df.append(res_df,ignore_index=True)
        product_df.to_csv(r'./data/dry_wet_p/unrpocess_wd_hoa_{ys}-{ye}_{s}_{pr}.csv'.format(s=spi.abbrev, ys=year_start, ye=year_end, pr=spi.product))

    else:
        aggregation = 'time.{}'.format(aggr)

        ############################# compute results dataframe ###########################
        print('Starting computing new dataframe')
        new_df = pd.DataFrame()

        for year in range(year_start,year_end):
            season_xr = res_xr.sel(time=slice('{}-01-01'.format(year), '{}-12-31'.format(year)))\
            .groupby(aggregation).mean(dim="time")

            for country in countries:
                # get the start time
                st = time.time()
                for i, season in enumerate(("DJF", "MAM", "JJA", "SON")):
                    subset = gdf[gdf.ADM0_NAME.isin([country])]
                    tempo_xr = cut_file(season_xr, subset)
                    mean_dry = tempo_xr["dry"].sel(season=season).mean(['lon','lat']).values
                    mean_wet = tempo_xr["wet"].sel(season=season).mean(['lon','lat']).values
                    temp_df = pd.DataFrame([mean_dry, mean_wet, country, season, year]).transpose()
                    new_df = new_df.append(temp_df,ignore_index=True)

                # get the end time
                et = time.time()
                # get the execution time
                elapsed_time = et - st
                print('Execution time by season for country {c} in year {y} :'.format(y=year, c= country), elapsed_time, 'seconds')

        new_df.columns = ['Dry_p', 'Wet_p', 'Country','Season','Year']
        #final_df['DW'] = round(final_df['DW'].astype(float),4)
        #final_df['WD'] = round(final_df['WD'].astype(float),4)
        new_df.to_csv(r'./data/dry_wet_p/season_wd_hoa_{ys}-{ye}_{s}_{pr}.csv'.format(s=spi.abbrev, ys=year_start, ye=year_end, pr=spi.product))

        #stacked_df = pd.pivot_table(new_df, columns=['Country', 'Season','Year']).stack()

if __name__== "__main__":

    #### Events:
    #### 1) SPI
    #### Calculate by product probability of having drought/flood in the years [2005, 2006, 2008, 2009, 2010, 2011, 2012, 2017]

    # CONFIG_PATH = r"./config.yaml"
    # config = load_config(CONFIG_PATH)

    # years = [2005, 2006, 2008, 2009, 2010, 2011, 2012]
    
    # distribution = 'gamma' ###pearson alternative

    # product_directories = [config['SPI']['CHIRPS']['path'], config['SPI']['IMERG']['path'], config['SPI']['ERA5']['path'],
    #                         config['SPI']['GPCC']['path']]

    # for product_dir in product_directories:
    #     files  = [f for f in os.listdir(product_dir) if f.endswith('.nc')]
    #     new_list = [x for x in files if distribution in x]

    #     #### Open one file to verify time series length
    #     xr_example = xr.open_dataset(os.path.join(product_dir,new_list[0]))
    #     year_start_prod = int(xr_example['time'].dt.year.min())
    #     year_end_prod= int(xr_example['time'].dt.year.max()) +1
    #     print('The prdouct {p} ranges from {ys} to {ye}'.format(ye=year_end_prod, ys=year_start_prod, p=os.path.basename(os.path.normpath(new_list[0]))))
        
    #     year_start = years[0]
    #     year_end = years[-1]

    #     if year_start_prod> year_start:
    #         "The product does not start in {ye} taking {ys} year as min".format(ys=year_start, ye=years[0])
    #         year_start = year_start_prod

    #     if year_end_prod< year_end:
    #         "The product does not end in {ye} taking {ys} year as max".format(ys=year_end, ye=years[-1])
    #         year_end = year_end_prod
        
    #     #### Computing new dataframe for the selected SPI product
    #     for file_path in new_list:
    #         countries = ['Ethiopia','Somalia','Kenya']
    #         path_spi = os.path.join(product_dir, file_path)
    #         compute_dry_wet(path_spi, countries,  year_start, year_end, aggr=None)


    import os
    from utils.function_clns import config
    import xarray as xr
    from utils.function_clns import subsetting_pipeline
    from dask.diagnostics import ProgressBar
    import os
    import sys
    import argparse
    import pyproj
    from precipitation import PrecipDataPreparation
    import warnings
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    warnings.filterwarnings("ignore")
    os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()

    class JupyterArgParser:
        def __init__(self):
            self.parser = argparse.ArgumentParser()
            self.parser.add_argument('-f')  # Required for Jupyter compatibility
            self.parser.add_argument('--normalize',default=False)
            self.parser.add_argument('--model',default="None")
            self.parser.add_argument('--fillna', default=False)
            self.parser.add_argument('--crop_area', default=True)

        def parse_args(self, args=None):
            # Remove Jupyter's default arguments if they exist
            if args is None:
                args = sys.argv[1:]

            # Parse and return arguments
            return self.parser.parse_args(args)

    arg_parser = JupyterArgParser()
    args = arg_parser.parse_args([]) 
    tb = ProgressBar().register()

    # precp_path = os.path.join(config['DEFAULT']["basepath"], "hydro_vars.zarr")
    # precp_ds = xr.open_zarr(precp_path)#.sel(time=slice("2005-01-01","2007-12-31"))


    variables = ["precipitation"] 

    precp_data = PrecipDataPreparation(
                    args, 
                    precipitation_data="MSWEP",
                    variables=variables,
                    load_local_precp=True,
                    precp_format="nc",
                    precp_filename="Daily_data",
                    interpolate = False,
                )

    ndvi_ds = precp_data.ndvi_ds



