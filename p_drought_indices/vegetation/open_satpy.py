

# ### Open MSG files with Satpy



from satpy import DataQuery
from satpy import Scene
import xarray as xr
from satpy.dataset import combine_metadata
import matplotlib.pyplot as plt
from glob import glob

my_channel_id = DataQuery(name=['VIS006'], calibration='reflectance')
my_channel_id_2 = DataQuery(name=['VIS008'], calibration='reflectance')

scn.load([my_channel_id, my_channel_id_2])




lon, lat = scn[0.8].attrs['area'].get_lonlats()




ndvi = (scn[0.8] - scn[0.6]) / (scn[0.8] + scn[0.6])
ndvi.attrs = combine_metadata(scn[0.8], scn[0.6])
scn['ndvi'] = ndvi
#scn.show('ndvi')



plt.imshow(lat.astype('float32'))
plt.colorbar()
plt.show()



local_scn = scn.resample("africa")




print(local_scn)
local_scn.show('ndvi')


