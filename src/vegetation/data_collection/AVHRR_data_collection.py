import numpy as np
import os
import urllib
from urllib.parse import urlparse
from urllib.parse import urljoin
from datetime import datetime, timedelta, date
from datetime import time as time_dt
from shapely.geometry import Polygon, mapping
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import requests
from bs4 import BeautifulSoup, SoupStrainer
import re

def avhrr_data_collection():

    years = [2005, 2008, 2009, 2010]

    for idx in years:
        URL = 'https://www.ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access/{}/'.format(idx)

        urls = []
        names = []
        
        try:
            response = requests.get(URL, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            
            a_tag = soup.find_all('td')
                
            for x in a_tag:
                y = x.find_all('a')
                for i in y:
                    if i.has_attr('href'):
                        urls.append(i['href'])
                        
            urls = list(filter(lambda x: x.endswith(".nc"), urls))
        
            for url in urls:
                
                URL_merged = urljoin(URL, url)
                name = str(url)
                names.append(name)
                urllib.request.urlretrieve(URL_merged, os.path.join(download_dir, name))
                print('Finished downloading product {}'.format(name))
            
            cut_only(download_dir, target_dir, gdf)

        except Exception as e:
            print('Couldn\'t download product {name} from year {year} because of error'.format(name=name, year=idx), e)


import os
import numpy as np

def lsaf_product_collection(product = "MDLAI", years = range(2005, 2021)):
    from getpass import getpass 
    baseurl = f"https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/{product}/NETCDF"
    response = requests.get(baseurl, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    save_folder = f"/media/BIFROST/N2/Riccardo/MSG/{product}"


    # Prompt user for username and password
    username = input("Enter your username: ")
    password = getpass("Enter your password: ") 

    # Create a folder to save downloaded files
    os.makedirs(save_folder, exist_ok=True)

    # Specify the month, and day range
    months = [str(month).zfill(2) for month in range(1, 13)]
    days = [str(day).zfill(2) for day in range(1, 32)]

    # Loop through years, months, and days
    for year in years:
        for month in months:
            for day in days:
                # Construct the URL for the specific year, month, and day
                url = f"{baseurl}/{year}/{month}/{day}/"

                # Fetch the page content
                response = requests.get(url, timeout=5)

                # Check if the page exists
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Find all links to files in subfolders
                    links = soup.find_all("a", href=True)
                    file_links = [link["href"] for link in links if link["href"].endswith(".nc")]


                    # Download files
                    for file_link in file_links:
                        absolute_url = urljoin(url, file_link)
                        file_name = os.path.join(save_folder, os.path.basename(file_link))

                        print(f"Downloading {absolute_url} to {file_name}")
                        #response = requests.get(absolute_url, timeout=30)

                        response = requests.get(absolute_url, auth=(username, password))

                        with open(file_name, "wb") as file:
                            file.write(response.content)
                else:
                    print(f"Page not found for {year}/{month}/{day}")

    print("Download complete.")
        

if __name__=="__main__":
    years = range(2006, 2021)
    lsaf_product_collection(product="MDLAI", years=years)