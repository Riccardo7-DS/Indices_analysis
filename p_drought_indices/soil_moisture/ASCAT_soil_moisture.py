import os
import cartopy
from datetime import datetime
import matplotlib.pyplot as plt
import findlibs

from satpy import Scene
from glob import glob
from satpy import available_readers
print(available_readers())

eps_path = r'D:\shareVM\soil_moisture\eumetsat'
#filename = os.path.join(eps_path, 'ASCA_SMO_02_M02_20070602103300Z_20070602121157Z_R_O_20090321004726Z.nat')
filenames = glob(os.path.join(eps_path,'*.nat'))

#reader='ascat_l2_soilmoisture_bufr'

#scene = Scene(filenames,
#                    reader=reader)

from ascat.eumetsat.level2 import AscatL2File
from ascat.eumetsat.level2 import AscatL2BufrFile
from ascat.eumetsat.level2 import AscatL2BufrFileList
from ascat.eumetsat.level2 import AscatL2NcFile
from ascat.eumetsat.level2 import AscatL2NcFileList
from ascat.eumetsat.level2 import AscatL2EpsFile
from ascat.eumetsat.level2 import AscatL2EpsFileList
#
eps_path = r'D:\shareVM\soil_moisture\eumetsat\ASCA_25_L2'
filename = os.path.join(eps_path, 'ASCA_SMO_02_M02_20110101090600Z_20110101104758Z_N_O_20110101105402Z.nat')
#
eps_file = AscatL2EpsFile(filename)
data = eps_file.read()
print(data)

