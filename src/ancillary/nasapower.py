import os, json
import pandas as pd


from pandas import json_normalize
class jsonNASA(object):
    def __init__(self, json_text) -> None:
        self.json_text = json_text
        self.parameters = list(json_text[ 'parameters'].keys())
        temp_df = json_normalize(json_text['properties']['parameter']).T
        clean_df = self._clean_df(temp_df)
        lat, lon, elev  = self._add_latlon()
        clean_df[['lat','lon']] = lat, lon
        self.df = clean_df
        #metrics_df = self._add_metrics()
        #self.df = clean_df.merge(metrics_df, on='parameter')
        xr_df = pd.pivot(self.df, index=['lat','lon','time'], values='value', columns='parameter').to_xarray()
        xr_df = self._add_attributes(xr_df)
        self.xr = xr_df.assign(elevation = elev)


    def _clean_df(self, temp_df):
        temp_df = temp_df.reset_index()
        other_df = temp_df['index'].str.rsplit('.', expand=True)
        temp_df[['parameter','date']] = other_df
        temp_df['time'] = pd.to_datetime(temp_df['date'], format="%Y%m%d")
        temp_df = temp_df.drop(columns={'index','date'}).rename(columns={0:'value'})
        return temp_df

    def _add_metrics(self):
        final_df = pd.DataFrame()
        for param in self.parameters:
            t_df = json_normalize(self.json_text['parameters'][param])
            t_df['parameter'] = param
            final_df = pd.concat([final_df, t_df])
        return final_df

    def _add_latlon(self):
        lat, lon, elev = self.json_text['geometry']['coordinates']
        return lat, lon, elev

    def _add_attributes(self, data):
        data['lat'].attrs= {'standard_name': 'latitude',
         'long_name': 'latitude',
         'units': 'degrees_north',
         'axis': 'Y'}
        data['lon'].attrs={'standard_name': 'longitude',
         'long_name': 'longitude',
         'units': 'degrees_east',
         'axis': 'X'}

        for param in self.parameters:
            data[param].attrs['units'] = self.json_text['parameters'][param]['units'] 
            data[param].attrs['long_name'] = self.json_text['parameters'][param]['longname']
        return data
        
if __name__=="__main__":
    
    path_to_json = r'C:\Users\Riccardo\Desktop\PhD_docs\Drought_prediction\Project\Indices_analysis\data\nasapower'
    target_dir= os.path.join(path_to_json,'nc_files')
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    
    for js in json_files:
        with open(os.path.join(path_to_json, js)) as json_file:
            json_text = json.load(json_file)
            nasatxt = jsonNASA(json_text)
            nasatxt.xr.to_netcdf(os.path.join(target_dir, '{}.nc'.format(js.rsplit('.',1)[0])))

    import xarray as xr

    ds = xr.open_mfdataset(os.path.join(target_dir,'*.nc'))
    ds_evo = xr.open_mfdataset(os.path.join(target_dir,'evapo','nc_files','*.nc'))
    final_df = xr.combine_by_coords([ds,ds_evo])
    final_df.to_netcdf(r'D:\shareVM\MERRA2\nasapower_vars.nc')