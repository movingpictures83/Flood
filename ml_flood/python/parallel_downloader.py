from ml_flood_config import path_to_data
from aux.data_download import CDS_Dataset

# define areas of interest
area_dict = {'danube': '50/7/47/20',
             'asia': '55/35/0/140',
             'usa': '50/-125/25/-70'}

ds = CDS_Dataset(dataset_name="reanalysis-era5-pressure-levels",
                 save_to_folder=path_to_data  # path to where datasets shall be stored
                ) 

ds.get(years = [str(y) for y in range(1980, 2018)], 
       months = [str(a).zfill(2) for a in range(1,13)], 
       request = dict(product_type='reanalysis', format='netcdf',
                      area=area_dict['asia'],
                      variable=['geopotential', 'temperature', 'specific_humidity'], 
                      pressure_level=['850', '700', '500']), 
       N_parallel_requests=10)

ds = CDS_Dataset(dataset_name="reanalysis-era5-single-levels",
                 save_to_folder=path_to_data  # path to where datasets shall be stored
                )

ds.get(years = [str(y) for y in range(1980, 2017)], 
       months = [str(a).zfill(2) for a in range(1,13)], 
       request = dict(product_type='reanalysis', format='netcdf',
                      area=area_dict['usa'],
                      variable=['lsp', 'convective_precipitation']
                     ),
       N_parallel_requests=10)

ds = CDS_Dataset(dataset_name="reanalysis-era5-single-levels",
                 save_to_folder=path_to_data  # path to where datasets shall be stored
                ) 

ds.get(years = [str(y) for y in range(1980, 2017)], 
       months = [str(a).zfill(2) for a in range(1,13)], 
       request = dict(product_type='reanalysis', format='netcdf',
                      area=area_dict['usa'],
                      variable=['volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 
                                'slope_of_sub_gridscale_orography', 'land_sea_mask', 
                                'soil_type', 'runoff', 'total_column_water_vapour']
                     ),
       N_parallel_requests=10)