UID = '143083'
API_key = '88445245-3ed7-466e-b483-7e9ed19a56a1'

# Write the keys into the file `~/.cdsapirc` in the home directory of your user
import os
with open(os.path.join(os.path.expanduser('~'), '.cdsapirc2'), 'w') as f:
    f.write('url: https://cds.climate.copernicus.eu/api/v2\n')
    f.write(f'key: {UID}:{API_key}')

# Import cdsapi and create a Client instance
import cdsapi
c = cdsapi.Client()

# More complex request
c.retrieve("reanalysis-era5-pressure-levels", {
            "product_type":   "reanalysis",
            "format":         "netcdf",
            "area":           "52.00/2.00/40.00/20.00", # N/W/S/E
            "variable":       "geopotential",
            "pressure_level": "500",
            "year":           "2017",
            "month":          "01",
            "day":            "12",
            "time":           "00"
            }, "example_era5_geopot_700.nc")