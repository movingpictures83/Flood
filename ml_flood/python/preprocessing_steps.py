
import os
import sys
import numpy as np
import pandas as np
import seaborn as sb
import xarray as xr
import os.path as path
from aux.utils import open_data
from aux.ml_flood_config import path_to_data
import dask
from dask.distributed import Client
client = Client(processes=False)  #memory_limit='16GB', 

# load dask client
client
# define some vars
data_path = f'{path_to_data}danube/monthly_files/'
print(data_path)

from aux.utils import rename_files
#rename_files(path=data_path, old='day.', new='dayavg.', str_constraint='temperature')

from aux.utils import cdo_daily_means
#cdo_daily_means(path=data_path, file_includes='single_level')
#cdo_daily_means(path=data_path, file_includes='850_700_500')

from aux.utils import cdo_precip_sums
#cdo_precip_sums(path=data_path, file_includes='large_scale_precipitation')

from aux.utils import cdo_clean_precip
#cdo_clean_precip(path=data_path)

from aux.utils import cdo_merge_time
# danube
#cdo_merge_time(path=data_path, file_includes='large_scale_precipitation', new_file='era5_lsp_1981-2017_daysum.nc')
#cdo_merge_time(path=data_path, file_includes='convective_precipitation', new_file='era5_tp_cp_1981-2017_daysum.nc')
#cdo_merge_time(path=data_path, file_includes='_temperature_', new_file='era5_z_t_q_1981-2017_dayavg.nc')
#cdo_merge_time(path=data_path, file_includes='soil_water', new_file='era5_swvl1_swvl2_ro_tcwv_1981-2017_dayavg.nc')

# usa
#cdo_merge_time(path=data_path, file_includes='geopotential,temperature', new_file='era5_z_t_q_1981-2017_dayavg.nc')
#cdo_merge_time(path=data_path, file_includes='convective_precipitation', new_file='era5_lsp_cp_1981-2017_daysum.nc')
#cdo_merge_time(path=data_path, file_includes='runoff', new_file='era5_ro_1981-2017_daysum.nc')
#cdo_merge_time(path=data_path, file_includes='soil_water', new_file='era5_swvl1_swvl2_tcwv_1981-2017_dayavg.nc')

# asia
#cdo_merge_time(path=data_path, file_includes='geopotential,temperature', new_file='era5_z_t_q_1981-2017_dayavg.nc')
#cdo_merge_time(path=data_path, file_includes='convective_precipitation', new_file='era5_lsp_cp_1981-2017_daysum.nc')
#cdo_merge_time(path=data_path, file_includes='runoff', new_file='era5_ro_1981-2017_daysum.nc')
#cdo_merge_time(path=data_path, file_includes='soil_water', new_file='era5_swvl1_swvl2_tcwv_1981-2017_dayavg.nc')

# extract data from tar files
import os

data_path = f'{path_to_data}glofas/'
for name in os.listdir(data_path):
    if 'glofas' in name:
        print(f'extracting data from {name} ...')
        file = data_path+name
        #os.system(f'tar -xvf {file}')

from aux.utils import cdo_spatial_cut
from aux.utils import cdo_merge_time
# define some vars
data_path = f'{path_to_data}glofas/'
print(data_path)
#!ls /home/srvx11/lehre/users/a1303583/ipython/ml_flood/data/glofas/

# DANUBE
# cut out spatial region
#cdo_spatial_cut(path=data_path, file_includes='dis_', new_file_includes='danube', lonmin=7, lonmax=20, latmin=47, latmax=50)
# merge into one file
#cdo_merge_time(path=data_path, file_includes='spatial_cut_danube', new_file='glofas_reanalysis_danube_2003-2018.nc')

# USA
#cdo_spatial_cut(path=data_path, file_includes='dis_', new_file_includes='usa', lonmin=-125, lonmax=-70, latmin=25, latmax=50)
#cdo_merge_time(path=data_path, file_includes='spatial_cut_usa', new_file='glofas_reanalysis_usa_2003-2018.nc')

# ASIA
#cdo_spatial_cut(path=data_path, file_includes='dis_', new_file_includes='asia', lonmin=35, lonmax=140, latmin=0, latmax=55)
#cdo_merge_time(path=data_path, file_includes='spatial_cut_asia', new_file='glofas_reanalysis_asia_2003-2018.nc')

def open_data(path, kw='era5'):
    """
    Opens all available ERA5/glofas datasets (depending on the keyword) in the specified path and resamples time to match
    the timestamp /per day (through the use of cdo YYYYMMDD 23z is the corresponding time
    stamp) in the case of era5, or renames lat lon in the case of glofas.
    """
    if kw is 'era5':    
        ds = xr.open_mfdataset(data_path+'*era5*')
        ds.coords['time'] = pd.to_datetime(ds.coords['time'].values) - datetime.timedelta(hours=23)
    elif kw is 'glofas_ra':
        ds = xr.open_mfdataset(data_path+'*glofas_reanalysis*')
        ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
    elif kw is 'glofas_fr':
        ds = xr.open_mfdataset(data_path+'*glofas_forecast*')
        ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
    return ds

era5 = open_data(data_path, kw='era5')
glofas = open_data(data_path, kw='glofas_ra')

print(era5)
print(glofas)

#era5 = open_data(data_path, kw='era5')
print(era5)

era5.chunk(chunks=dict(time=-1)).to_netcdf('era5_danube_pressure_and_single_levels.nc', engine='netcdf4')
