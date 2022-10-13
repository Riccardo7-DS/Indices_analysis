import xarray as xr
from merge_daily_data import load_config, cut_file
import os
import pandas as pd
import time
import geopandas as gpd
from shapely.geometry import Polygon, mapping
from SPI_wet_dry import spiObject

CONFIG_PATH = r"./config.yaml"
config = load_config(CONFIG_PATH)


def compute_dry_wet(path_spi, countries,  year_start, year_end, aggr='season'):
    if aggr not in ['season','month','day','year']:
        raise ValueError("The specified aggregation time is not correct")
    spi = spiObject(path_spi)
    shapefile_path = config['SHAPE']['africa']
    aggregation = 'time.{}'.format(aggr)

    #### Chose subset of countries
    gdf = gpd.read_file(shapefile_path)
    subset = gdf[gdf.ADM0_NAME.isin(countries)]

    ###open file and cut it
    spi.xr = cut_file(xr.open_dataset(path_spi, chunks={"lat": -1, "lon": -1, "time": 12}), subset)
    
    res_xr = cut_file(spi.calculate_points_xr(), subset)
    print('The final product has the following variables:', res_xr.data_vars)

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

    CONFIG_PATH = r"./config.yaml"
    config = load_config(CONFIG_PATH)

    years = [2005, 2006, 2008, 2009, 2010, 2011, 2012]
    
    distribution = 'gamma' ###pearson alternative

    product_directories = [config['SPI']['CHIRPS']['path'], config['SPI']['IMERG']['path'], config['SPI']['ERA5']['path'],
                            config['SPI']['GPCC']['path']]

    for product_dir in product_directories:
        files  = [f for f in os.listdir(product_dir) if f.endswith('.nc')]
        new_list = [x for x in files if distribution in x]
        #### Open one file to verify time series length
        xr_example = xr.open_dataset(os.path.join(product_dir,new_list[0]))
        year_start_prod = int(xr_example['time'].dt.year.min())
        year_end_prod= int(xr_example['time'].dt.year.max()) +1
        print('The prdouct {p} ranges from {ys} to {ye}'.format(ye=year_end_prod, ys=year_start_prod, p=os.path.basename(os.path.normpath(new_list[0]))))
        
        year_start = years[0]
        year_end = years[-1]

        if year_start_prod> year_start:
            "The product does not start in {ye} taking {ys} year as min".format(ys=year_start, ye=years[0])
            year_start = year_start_prod

        if year_end_prod< year_end:
            "The product does not end in {ye} taking {ys} year as max".format(ys=year_end, ye=years[-1])
            year_end = year_end_prod
        
        #### Computing new dataframe for the selected SPI product
        for file_path in new_list:
            countries = ['Ethiopia','Somalia','Kenya']
            path_spi = os.path.join(product_dir, file_path)
            compute_dry_wet(path_spi, countries,  year_start, year_end, aggr='season')



