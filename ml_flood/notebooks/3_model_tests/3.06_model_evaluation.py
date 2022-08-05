import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns
import xarray as xr

import sys
sys.path.append('../../')
from python.misc.plot import plot_recurrent
from python.misc.utils import xr_to_datetime
from python.misc.utils_floodmodel import add_valid_time
from python.misc.verification import verify, ME, RMSE, RMSE_persistence, NSE, NSE_diff

def multif_metrics(pred, obs, method='total', case=True, name=None):
    """Calculates RMSE and NSE metrics for xr.DataArray in the multiforecast
    shape as defined in the previous notebooks.
    Use case=True if the input data is in the form of a case study: multiple forecasts vor one set period.

    Parameters
    ----------
        pred   : xr.DataArray
        obs    : xr.DataArray
        case   : Boolean
        method : str
    """
    if method == 'total':
        y_o = obs.loc[{'time': pred.time.values.ravel()}].values
        y_m = pred.values.ravel()
        rmse = np.sqrt(np.nanmean((y_m - y_o)**2))
        nse = 1 - np.sum((y_m - y_o)**2)/np.sum(((y_o - np.nanmean(y_o))**2))
        metr = xr.DataArray(data=np.array([rmse, nse]),
                            dims=['metric'],
                            coords=[['rmse', 'nse']],
                            name='total_metrics')

    elif method == 'per_day':
        forecast_range = pred.forecast_day
        fd_list = []
        for fd in forecast_range:
            pred_day = pred.sel(forecast_day=fd)
            y_o = obs.loc[{'time': pred_day.time.values.ravel()}].values
            y_m = pred_day.values.ravel()
            rmse = np.sqrt(np.nanmean((y_m - y_o)**2))
            nse = 1 - np.sum((y_m - y_o)**2)/np.sum(((y_o - np.nanmean(y_o))**2))
            fd_list.append([rmse, nse])
        metr = xr.DataArray(data=np.array(fd_list),
                            dims=['forecast_day', 'metric'],
                            coords=[range(np.array(fd_list).shape[0]), ['rmse', 'nse']],
                            name='per_day_metrics')
    if name:
        metr.name = f"{name}-{metr.name}"
    return metr

truth_test = xr.open_dataset('../../data/glofas_reanalysis_test_period.nc')['dis']
fc_lr_test = xr.open_dataset('../../data/models/LinearRegression/linear_regression_result_test_period.nc')['prediction']
fc_svr_test = xr.open_dataset('../../data/models/SVR/support_vector_regression_result_test_period.nc')['prediction']
fc_gb_test = xr.open_dataset('../../data/models/GradientBoost/gradient_boost_result_test_period.nc')['prediction']
fc_tdnn_test = xr.open_dataset('../../data/models/TimeDelayNeuralNet/tdnn_result_test_period.nc')['prediction']

# initial state used for persistence
truth_init = truth_test.values.reshape(fc_lr_test.shape)[:,0]
# repeat the initial state for the whole forecast period i.e. persistence forecast
persistence_data = np.tile(truth_init, (15, 1)).transpose()
# easy way to get the same shape, dims, coords etc
persistence_test = fc_lr_test.copy()
persistence_test.values = persistence_data

lr_test = multif_metrics(fc_lr_test, truth_test, method='per_day', name='LinReg')
svr_test = multif_metrics(fc_svr_test, truth_test, method='per_day', name='SVR')
gb_test = multif_metrics(fc_gb_test, truth_test, method='per_day', name='GradBoost')
tdnn_test = multif_metrics(fc_tdnn_test, truth_test, method='per_day', name='TimeDelay_neural_net')
per_test = multif_metrics(persistence_test, truth_test, method='per_day', name='Persistence')

list_of_models = [lr_test, svr_test, gb_test, tdnn_test, per_test]

def plot_metrics(list_of_models, obs=None, frerun=None, suptitle=None):
    """Convenience function for plotting the metrics of models contained in the input
    list_of_models, and comparison to persistence and glofas forecast_reruns if available.
    
    Parameters
    ----------
        list_of_models : list containing a xr.DataArray for each model
        obs            : xr.DataArray
        frerun         : xr.DataArray
    """
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))
    cdict = {'LinReg': 'tab:orange',
             'SVR': 'tab:purple',
             'GradBoost': 'tab:cyan',
             'TimeDelay_neural_net': 'tab:green',
             'Persistence': 'tab:pink',
             'forecast_rerun_median': 'dimgray'}
    
    for model in list_of_models:
        model_name = model.name.split("-")[0]
        model.sel(metric='rmse').plot(ax=ax1, label=model_name, color=cdict[model_name], linewidth=2)
        model.sel(metric='nse').plot(ax=ax2, label=model_name, color=cdict[model_name], linewidth=2)
    
    if suptitle:
        plt.suptitle(suptitle, fontsize=18)
    ax1.set_title('Root Mean Square Error')
    ax1.set_ylabel('RMSE discharge [m$^3$/s]')
    ax1.legend(fontsize=10, loc=2)
    ax2.set_title('Nash-Sutcliffe Efficiency')
    ax2.set_ylabel('NSE')
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, 1])
    ax1.grid()
    ax2.grid()
    return None

plot_metrics(list_of_models, suptitle='Metrics for ML 14-day forecasts for the test period: 2012 to 2016')
plt.savefig('test_period_metrics_14d_fcst.png', dpi=600, bbox_inches='tight')

import matplotlib
matplotlib.rcParams.update({'font.size': 14})

fig, ax = plt.subplots(figsize=(15, 5))
truth_test.to_pandas().plot(ax=ax, label='reanalysis')
cdict = {'LinReg': 'tab:orange',
         'SVR': 'tab:purple',
         'GradBoost': 'tab:cyan',
         'TimeDelay_neural_net': 'tab:green',
         'Persistence': 'tab:olive',
         'forecast_rerun_median': 'dimgray'}

for i in fc_lr_test.num_of_forecast:
    ind = i.values
    time_vec = truth_test.to_pandas().index[(ind-1)*15:ind*15]
    pd.Series(fc_lr_test.sel(num_of_forecast=i).values, index=time_vec).plot(ax=ax, label='LinReg', color=cdict['LinReg'])
    pd.Series(fc_svr_test.sel(num_of_forecast=i).values, index=time_vec).plot(ax=ax, label='SVR', color=cdict['SVR'])
    pd.Series(fc_gb_test.sel(num_of_forecast=i).values, index=time_vec).plot(ax=ax, label='GradBoost', color=cdict['GradBoost'])
    pd.Series(fc_tdnn_test.sel(num_of_forecast=i).values, index=time_vec).plot(ax=ax, label='TimeDelay_neural_net', color=cdict['TimeDelay_neural_net'])

ax.set_ylabel('river discharge [m$^3$/s]')
plt.legend(['GloFAS Reanalysis', 'LinReg', 'SVR', 'GradBoost', 'TimeDelay_neural_net'], fontsize=12)
plt.title('14-day ML forecasts for the test period: 2012 - 2016', fontsize=16);
plt.savefig('test_period_timeseries_14d_fcst.png', dpi=600, bbox_inches='tight')

truth_case = xr.open_dataset('../../data/glofas_reanalysis_case_study.nc')['dis']
fc_lr_case = xr.open_dataset('../../data/models/LinearRegression/linear_regression_result_case_study.nc')['prediction']
fc_svr_case = xr.open_dataset('../../data/models/SVR/support_vector_regression_result_case_study.nc')['prediction']
fc_gb_case = xr.open_dataset('../../data/models/GradientBoost/gradient_boost_result_case_study.nc')['prediction']
fc_tdnn_case = xr.open_dataset('../../data/models/TimeDelayNeuralNet/tdnn_result_case_study.nc')['prediction']
glo_fr_case = xr.open_dataset('../../data/glofas_freruns_case_study.nc')['forecast rerun']

glo_fr_case_median = glo_fr_case.median(dim='ensemble');

def preproc_multif_case(multif):
    multif_list = []
    fc_range = range(0,31)
    for fc in multif.num_of_forecast:
        data = multif.sel(num_of_forecast=fc).dropna(dim='time')
        fc_da = xr.DataArray(data.values,
                             coords={'forecast_day': fc_range,
                                     'time': (('forecast_day'), data.time.values)},
                             dims='forecast_day')
        multif_list.append(fc_da)
    new_da = xr.concat(multif_list, dim='num_of_forecast')
    new_da.name = 'prediction'
    return new_da

# directly select the few starting points for the small case study
truth_init = np.array([truth_case.sel(time='2013-05-18').values,
                       truth_case.sel(time='2013-05-22').values,
                       truth_case.sel(time='2013-05-25').values,
                       truth_case.sel(time='2013-05-29').values])
persistence_data = np.tile(truth_init, (31, 1)).transpose()
persistence_case = preproc_multif_case(fc_lr_case).copy()
persistence_case.values = persistence_data

lr_case = multif_metrics(preproc_multif_case(fc_lr_case), truth_case, method='per_day', name='LinReg')
svr_case = multif_metrics(preproc_multif_case(fc_svr_case), truth_case, method='per_day', name='SVR')
gb_case = multif_metrics(preproc_multif_case(fc_gb_case), truth_case, method='per_day', name='GradBoost')
tdnn_case = multif_metrics(preproc_multif_case(fc_tdnn_case), truth_case, method='per_day', name='TimeDelay_neural_net')
fr_case = multif_metrics(preproc_multif_case(glo_fr_case_median), truth_case, method='per_day', name='forecast_rerun_median')
per_case = multif_metrics(persistence_case, truth_case, method='per_day', name='Persistence')

list_of_models_case = [lr_case, svr_case, gb_case, tdnn_case, fr_case, per_case]

plot_metrics(list_of_models_case, suptitle='model comparison for the case study period: 18.05.2013 - 28.06.2013')
plt.ylim([0., 1]);
plt.savefig('case_study_metrics_30d_fcst.png', dpi=600, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(15, 5))
truth_case.to_pandas().plot(ax=ax, label='reanalysis', linewidth=4)

frerun_c = ['silver', 'darkgray', 'gray', 'dimgray']
for i in fc_lr_case.num_of_forecast:
    fc_lr_case.sel(num_of_forecast=i).to_pandas().T.plot(ax=ax, label='LinReg', color=cdict['LinReg'], linewidth=2)
    fc_svr_case.sel(num_of_forecast=i).to_pandas().T.plot(ax=ax, label='SVR', color=cdict['SVR'], linewidth=2)
    fc_gb_case.sel(num_of_forecast=i).to_pandas().T.plot(ax=ax, label='GradBoost', color=cdict['GradBoost'], linewidth=2)
    fc_tdnn_case.sel(num_of_forecast=i).to_pandas().T.plot(ax=ax, label='TimeDelay_neural_net', color=cdict['TimeDelay_neural_net'], linewidth=2)
    glo_fr_case.sel(num_of_forecast=i).to_pandas().T.plot(ax=ax, label='GloFAS Forecast_Rerun', linewidth=1,
                                                          linestyle='--', color=frerun_c[i.values])

ax.set_ylabel('river discharge [m$^3$/s]')
plt.legend(['GloFAS Reanalysis', 'LinReg', 'SVR', 'GradBoost', 'TimeDelay_neural_net', 'GloFAS Forecast_rerun'], loc=2)
plt.title('Case study May/June 2013');
plt.savefig('case_study_timeseries_30d_fcst.png', dpi=600, bbox_inches='tight');

truth_case.to_pandas().T.plot(linewidth=4)
for i in fc_tdnn_case.num_of_forecast:
    fc_tdnn_case.sel(num_of_forecast=i).to_pandas().T.plot(color=cdict['TimeDelay_neural_net'], linewidth=2)
plt.ylim(1300, 1700)
plt.legend(['GloFAS Reanalysis', 'TimeDelay_neural_net']);
plt.savefig('case_check_bad_NSE.png', dpi=600, bbox_inches='tight')

fig, axes = plt.subplots(4, 2, figsize=(15, 18), sharex='col')

frerun_c = ['silver', 'darkgray', 'gray', 'dimgray']
for i in fc_lr_case.num_of_forecast:
    ax = axes[i.values, 0]
    ax2 = axes[i.values, 1]
    reana = truth_case.loc[{'time': preproc_multif_case(fc_lr_case).sel(num_of_forecast=i).time.values}].to_pandas().T
    reana.index = preproc_multif_case(fc_lr_case).sel(num_of_forecast=i).to_pandas().index
    reana.plot(ax=ax, label='reanalysis', linewidth=4)
    preproc_multif_case(fc_lr_case).sel(num_of_forecast=i).to_pandas().T.plot(ax=ax, label='LinReg', color=cdict['LinReg'], linewidth=2)
    preproc_multif_case(fc_svr_case).sel(num_of_forecast=i).to_pandas().T.plot(ax=ax, label='SVR', color=cdict['SVR'], linewidth=2)
    preproc_multif_case(fc_gb_case).sel(num_of_forecast=i).to_pandas().T.plot(ax=ax, label='GradBoost', color=cdict['GradBoost'], linewidth=2)
    preproc_multif_case(fc_tdnn_case).sel(num_of_forecast=i).to_pandas().T.plot(ax=ax, label='TimeDelay_neural_net', color=cdict['TimeDelay_neural_net'], linewidth=2)
    preproc_multif_case(glo_fr_case_median).sel(num_of_forecast=i).to_pandas().T.plot(ax=ax, label='GloFAS Forecast_Rerun', linewidth=1,
                                                          linestyle='--', color='dimgray')
    
    day_cut = 5
    reana[:day_cut].plot(ax=ax2, label='reanalysis', linewidth=4)
    preproc_multif_case(fc_lr_case).sel(num_of_forecast=i).to_pandas().T[:day_cut].plot(ax=ax2, label='LinReg', color=cdict['LinReg'], linewidth=2)
    preproc_multif_case(fc_svr_case).sel(num_of_forecast=i).to_pandas().T[:day_cut].plot(ax=ax2, label='SVR', color=cdict['SVR'], linewidth=2)
    preproc_multif_case(fc_gb_case).sel(num_of_forecast=i).to_pandas().T[:day_cut].plot(ax=ax2, label='GradBoost', color=cdict['GradBoost'], linewidth=2)
    preproc_multif_case(fc_tdnn_case).sel(num_of_forecast=i).to_pandas().T[:day_cut].plot(ax=ax2, label='TimeDelay_neural_net', color=cdict['TimeDelay_neural_net'], linewidth=2)
    preproc_multif_case(glo_fr_case_median).sel(num_of_forecast=i).to_pandas().T[:day_cut].plot(ax=ax2, label='GloFAS Forecast_Rerun', linewidth=1,
                                                          linestyle='--', color='dimgray')
    plt_str = f"init time: {str(preproc_multif_case(fc_lr_case).sel(num_of_forecast=i).time[0].dt.day.values)}-{str(preproc_multif_case(fc_lr_case).sel(num_of_forecast=1).time[0].dt.month.values)}-{str(preproc_multif_case(fc_lr_case).sel(num_of_forecast=1).time[0].dt.year.values)}"
    ax.set_title(f"full time span {plt_str}")
    ax2.set_title(f"limited time span {plt_str}")
#    ax2.set_xlim([0, 7])
    ax.set_ylabel('river discharge [m$^3$/s]')
    ax.grid()
    ax2.grid()
    
#ax.set_ylabel('river discharge [m$^3$/s]')
plt.legend(['GloFAS Reanalysis', 'LinReg', 'SVR', 'GradBoost', 'TimeDelay_neural_net', 'GloFAS Forecast_rerun Median'], bbox_to_anchor=(0.7, 5.), ncol=3)
plt.suptitle('Case study May/June 2013');
plt.savefig('case_study_tile_error_analysis_30d_fcst.png', dpi=600, bbox_inches='tight');

truth_test = xr.open_dataset('../../data/glofas_reanalysis_test_period.nc')['dis']
fc_lr_test = xr.open_dataset('../../data/models/LinearRegression/linear_regression_result_test_period.nc')['prediction']
fc_svr_test = xr.open_dataset('../../data/models/SVR/support_vector_regression_result_test_period.nc')['prediction']
fc_gb_test = xr.open_dataset('../../data/models/GradientBoost/gradient_boost_result_test_period.nc')['prediction']
fc_tdnn_test = xr.open_dataset('../../data/models/TimeDelayNeuralNet/tdnn_result_test_period.nc')['prediction']

fig, ax = plt.subplots(figsize=(15, 5))
truth_test.to_pandas().plot()
ax.set_ylabel('river discharge [m$^3$/s]')
plt.title('Glofas Reanalysis');

truth_diff = truth_test.diff('time')
event_times = truth_diff.where(truth_diff >= np.sort(truth_diff.values)[-25], drop=True).time.values
print(event_times)

fig, ax = plt.subplots(figsize=(15, 5))
truth_test.to_pandas().plot()
for event in event_times:
    ax.axvline(event, color='r', linestyle='--', linewidth=0.6)
ax.set_ylabel('river discharge [m$^3$/s]')
plt.title('Glofas Reanalysis; strong change in discharge marked as red vertical dashed line');
plt.savefig('samples_test_period_timeseries.png', dpi=600, bbox_inches='tight')

fc_lr_test.sel(num_of_forecast=111).time

# keys, corresponding to the closest forecast init time from the event time
keys = [10, 42, 63, 111]

def plot_single_case_prediction(list_of_models, y_truth, title=None):
    """Convenience function for plotting multiple model forecasts and truth.
    
    Parameters
    ----------
        pred_multif     : list containing xr.DataArray objects
        y_truth         : xr.DataArray
        forecast_range  : int
        title           : str
    """
    cdict = {'LinReg': 'tab:orange',
             'SVR': 'tab:purple',
             'GradBoost': 'tab:cyan',
             'TimeDelay_neural_net': 'tab:green',
             'Persistence': 'tab:pink',
             'forecast_rerun_median': 'dimgray'}
    
    fig, ax = plt.subplots(figsize=(15,5))
    time_index = list_of_models[0].time.values
    y_truth.sel({'time': time_index}).to_pandas().plot(ax=ax, label='GloFAS Reanalysis', linewidth=4)
    
    color_scheme = ['orange', 'y', 'cyan', 'magenta', 'brown']
    pd.Series(data=list_of_models[0].values, index=time_index).plot(ax=ax, label='LinReg', color=cdict['LinReg'], linewidth=2)
    pd.Series(data=list_of_models[1].values, index=time_index).plot(ax=ax, label='SVR', color=cdict['SVR'], linewidth=2)
    pd.Series(data=list_of_models[2].values, index=time_index).plot(ax=ax, label='GradBoost', color=cdict['GradBoost'], linewidth=2)
    pd.Series(data=list_of_models[3].values, index=time_index).plot(ax=ax, label='TimeDelay_neural_net',
                                                                    color=cdict['TimeDelay_neural_net'], linewidth=2)
    
    
    plt.legend()
    plt.title(f'14-day ML forecast: sample {run}')
    ax.set_ylabel('river discharge [m$^3$/s]')
    return fig, ax

run = 1
for k in keys:
    list_of_models_test = [fc_lr_test[k,:], fc_svr_test[k,:], fc_gb_test[k,:], fc_tdnn_test[k,:]]
    plot_single_case_prediction(list_of_models_test, truth_test)
    plt.savefig(f'sample_{run}_test_period.png', dpi=600, bbox_inches='tight')
    run += 1

# initial state used for persistence
truth_init = truth_test.values.reshape(fc_lr_test.shape)[:,0]
# repeat the initial state for the whole forecast period i.e. persistence forecast
persistence_data = np.tile(truth_init, (15, 1)).transpose()
# easy way to get the same shape, dims, coords etc
persistence_test = fc_lr_test.copy()
persistence_test.values = persistence_data

lr_test = multif_metrics(fc_lr_test[keys,:], truth_test, method='per_day', name='LinReg')
svr_test = multif_metrics(fc_svr_test[keys,:], truth_test, method='per_day', name='SVR')
gb_test = multif_metrics(fc_gb_test[keys,:], truth_test, method='per_day', name='GradBoost')
tdnn_test = multif_metrics(fc_tdnn_test[keys,:], truth_test, method='per_day', name='TimeDelay_neural_net')
per_test = multif_metrics(persistence_test[keys,:], truth_test, method='per_day', name='Persistence')

list_of_models = [lr_test, svr_test, gb_test, tdnn_test, per_test]
plot_metrics(list_of_models, suptitle='model comparison for sample events in the test period: 2012 to 2016')
plt.savefig('samples_test_period_metrics.png', dpi=600, bbox_inches='tight')
