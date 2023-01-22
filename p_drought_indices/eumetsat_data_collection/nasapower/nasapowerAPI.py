'''
*Version: 1.0 Published: 2020/02/11* Source: [NASA POWER](https://power.larc.nasa.gov/)
POWER API Multipoint Download (CSV)
This is an overview of the process to request data from multiple data points from the POWER API.
'''

import os, sys, time, json, urllib3, requests, multiprocessing
from p_drought_indices.eumetsat_data_collection.nasapower.json_to_nc import jsonNASA

urllib3.disable_warnings()

import numpy as np

def download_function(collection):
    ''' '''
    base_dir = r'C:\Users\Riccardo\Desktop\PhD_docs\Drought_prediction\Project\Indices_analysis\data\nasapower'
    target_dir = os.path.join(base_dir, 'nc_files')
    request, filepath = collection
    response = requests.get(url=request, verify=False, timeout=30.00).json()

    with open(os.path.join(base_dir,filepath), 'w') as file_object:
        json.dump(response, file_object)

class Process():

    def __init__(self):

        self.processes = 5 # Please do not go more than five concurrent requests.

        self.request_template = r"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2MDEW,T2MWET,TS,T2M_MAX,T2M_MIN,WS2M,WS10M,WS2M_MAX,WS2M_MIN,EVLAND,EVPTRNS&community=RE&longitude={longitude}&latitude={latitude}&start=20050101&end=20200101&format=JSON"
        self.filename_template = "File_Lat_{latitude}_Lon_{longitude}.json"

        self.messages = []
        self.times = {}

    def execute(self):

        Start_Time = time.time()

        latitudes = np.arange(-5, 15, 0.5) # Update your download extent.
        longitudes = np.arange(32, 53, 0.5) # Update your download extent.

        requests = []
        for longitude in longitudes:
            for latitude in latitudes:
                request = self.request_template.format(latitude=latitude, longitude=longitude)
                filename = self.filename_template.format(latitude=latitude, longitude=longitude)
                requests.append((request, filename))

        requests_total = len(requests)

        pool = multiprocessing.Pool(self.processes)
        x = pool.imap_unordered(download_function, requests)

        for i, df in enumerate(x, 1):
            sys.stderr.write('\rExporting {0:%}'.format(i/requests_total))

        self.times["Total Script"] = round((time.time() - Start_Time), 2)

        print ("\n")
        print ("Total Script Time:", self.times["Total Script"])

if __name__ == '__main__':
    Process().execute()