import numpy as np
import datetime as dt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import dask
dask.config.set(scheduler='threads')
import xarray as xr

era5 = xr.open_dataset('../../data/smallsampledata-era5.nc')
glofas = xr.open_dataset('../../data/smallsampledata-glofas.nc')
era5

era5['cp']

import sys
sys.path.append("../../")
from python.misc.utils_floodmodel import get_mask_of_basin

danube_catchment = get_mask_of_basin(glofas['dis'].isel(time=0))
dis = glofas['dis'].where(danube_catchment)

maximum = dis.where(dis==dis.max(), drop=True)
lat, lon = float(maximum.latitude), float(maximum.longitude)
lat, lon

poi = dict(latitude=48.35, longitude=13.95)
dis_mean = dis.mean('time')
dis_mean.plot()
plt.gca().plot(lon, lat, color='cyan', marker='o', 
               markersize=20, mew=4, markerfacecolor='none')

X = era5[['lsp', 'cp']]

from python.aux.utils_floodmodel import shift_and_aggregate

for var in ['lsp', 'cp']:
    for i in range(1,6):
        newvar = var+'-'+str(i)
        X[newvar] = X[var].shift(time=i)  # previous precip as current day variable

for var in ['lsp', 'cp']:
    for i in range(1,14):
        newvar = var+'+'+str(i)
        X[newvar] = X[var].shift(time=-i) # future precip as current day variable
        
X.data_vars

X['lsp-5-11'] = shift_and_aggregate(X['lsp'], shift=5, aggregate=7)
X['lsp-12-25'] = shift_and_aggregate(X['lsp'], shift=12, aggregate=14)
X['lsp-26-55'] = shift_and_aggregate(X['lsp'], shift=26, aggregate=30)
X['lsp-56-180'] = shift_and_aggregate(X['lsp'], shift=56, aggregate=125)

fig, ax = plt.subplots(figsize=(15,5))

X['lsp'][:60,0,0].plot(ax=ax, label='precipitation current day')
X['lsp-5-11'][:60,0,0].plot(ax=ax, label='precipitation 5-11 days before')
X['lsp-12-25'][:60,0,0].plot(ax=ax, label='precipitation 12-25 days before')
ax.legend()
plt.title('Example timeseries of predictors')

X

from python.aux.utils_floodmodel import reshape_multiday_predictand

y = glofas['dis'].interp(latitude=lat, longitude=lon)

var = y.name
y = y.to_dataset()
for i in range(1,14):
    newvar = var+'+'+str(i)
    y[newvar] = y[var].shift(time=-i) # future precip as current day variable
y = y.to_array(dim='forecast_day')
y.coords['forecast_day'] = range(1,len(y.forecast_day)+1)
y

# aggregate over the space dimension (more complex in the next notebook)
Xagg = X.mean(['latitude', 'longitude'])

Xda, yda = reshape_multiday_predictand(Xagg, y)

for q in [0.25, .5, .75]:
    print('percentile', q, ': ', round(float(dis_mean.quantile(q)),3), 'm^3/s')

dist = dis_mean.values.ravel()
sns.distplot(dist, bins=np.logspace(-1, 4), hist_kws=dict(cumulative=True))
plt.xlim(0,100)
plt.grid()
plt.ylabel('cumulative distribution')
plt.xlabel('discharge [m$^3$/s]')

from python.aux.utils_floodmodel import cluster_by_discharge
from python.aux.utils import calc_area, nandot

bin_edges = [0, 0.8, 2.5, 10.25, 10000]
cluster = cluster_by_discharge(dis_mean, bin_edges)

for c in cluster:
    plt.figure()
    cluster[c].plot()
    plt.title('#points: '+str(int(cluster[c].sum())))

image = dis_mean*0.
image.name = 'spatial feature cluster'
for i, c in enumerate(cluster):
    image = image.where(~cluster[c], i)
    
image.plot(cmap = mpl.colors.ListedColormap(['grey', 'orange', 'blue', 'darkblue']))

cluster = cluster.to_array('clusterId')
cluster.coords 

def aggregate_clustersum(ds, cluster, clusterdim):
    """Aggregate a 3-dimensional array over certain points (latitude, longitude).

    Parameters
    ----------
    ds : xr.Dataset
        the array to aggregate (collapse) spatially
    cluster : xr.DataArray
        3-dimensional array (clusterdim, latitude, longitude),
        `clusterdim` contains the True/False mask of points to aggregate over
        e.g. len(clusterdim)=4 means you have 4 clusters
    clusterdim : str
        dimension name to access the different True/False masks

    Returns
    -------
    xr.DataArray
        1-dimensional
    """
    out = xr.Dataset()

    # enforce same coordinates
    interp = True
    if (len(ds.latitude.values) == len(cluster.latitude.values) and
            len(ds.longitude.values) == len(cluster.longitude.values)):
        if (np.allclose(ds.latitude.values, cluster.latitude.values) and
                np.allclose(ds.longitude.values, cluster.longitude.values)):
            interp = False
    if interp:
        ds = ds.interp(latitude=cluster.latitude, longitude=cluster.longitude)
    area_per_gridpoint = calc_area(ds.isel(time=0))

    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()

    for var in ds:
        for cl in cluster.coords[clusterdim]:
            newname = var+'_cluster'+str(cl.values)
            this_cluster = cluster.sel({clusterdim: cl})

            da = ds[var].where(this_cluster, 0.)  # no contribution from outside cluster
            #print(da)
            out[newname] = xr.dot(da, area_per_gridpoint)
    return out.drop(clusterdim)

# later, we can import the function from here
# from python.aux.utils_flowmodel import aggregate_clustersum

Xagg = aggregate_clustersum(X, cluster, 'clusterId')

# drop these predictors
for v in Xagg:
    if 'cluster0' in v:
        for vn in ['lsp-5-11', 'lsp-12-25', 'lsp-26-55', 'lsp-56-180']:
            if vn in v:
                Xagg = Xagg.drop(v)
                break

# drop these predictors (predictand time)
for v in Xagg:
    for vn in ['lsp_cluster', 'cp_cluster']:
        if v.startswith(vn):
            Xagg = Xagg.drop(v)
            break

if False:  # alternative: aggregating over space by taking the mean
    Xagg = X.mean(['latitude', 'longitude'])

Xagg
