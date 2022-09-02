import subprocess
from datetime import datetime, timedelta
import yaml
import sys
import time


if __name__ == "__main__":

    ### Chose start and end date to collect the product
    start_date = '2013-01-01 11:45:00'
    end_date= '2013-01-05 12:20:00'

    start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    delta = timedelta(days=1)

    while start_dt <= end_dt:


        dict_file = {'start_time' : datetime.strftime(start_dt, '%Y-%m-%d %H:%M:%S'),
            'end_time' : datetime.strftime(end_dt, '%Y-%m-%d %H:%M:%S')}


        with open(r'./time.yaml', 'w') as file:
            documents = yaml.dump(dict_file, file, default_flow_style=False)  
        
        python_bin = r'C:\Users\Riccardo\anaconda3\envs\gis_py39\python.exe'
        script_file = r'C:\Users\Riccardo\Desktop\PhD_docs\Drought_prediction\Project\Indices_analysis\eumdac_collection.py'

        proc1 = subprocess.run([python_bin, script_file])
        
        

        python_bin = r'C:\Users\Riccardo\anaconda3\envs\epct-2.5\python.exe' 
        script_file= r'C:\Users\Riccardo\Desktop\PhD_docs\Drought_prediction\Project\Indices_analysis\eptc_crop_clean.py'
        
        proc2 = subprocess.run([python_bin, script_file])

        #while True:
        # check if either sub-process has finished
            #proc2.poll()
            #time.sleep(1)
            #proc1.poll()

           # if proc1.returncode is not None or proc2.returncode is not None:
              #  break

        print('Finished downloading and cropping dataset')

        start_dt += delta 