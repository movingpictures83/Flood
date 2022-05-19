#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : ConvTransformerTS
@ FileName: rnn_singlestep.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 1/11/22 15:18
"""

from tensorflow import keras
from tensorflow.keras import layers
from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.layers import SimpleRNN
from keras.layers.core import Dense, Dropout
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.optimizers import Adam
from pandas import concat
import numpy as np
from helper import series_to_supervised, stage_series_to_supervised
from pandas import read_csv


# dataset = read_csv('data/zeda/Merged_with_MeanGridRain.csv', index_col=0)
dataset = read_csv('data/zeda/Merged_hourly.csv', index_col=0)
# dataset = read_csv('data/zeda/Merged.csv', index_col=0)
dataset.fillna(0, inplace=True)
print(dataset.shape)
print(dataset.columns)


data = dataset[:578448]   # 578448
print(data.shape)
print(list(data.columns))


# specify the number of lag hours
n_hours = 72
n_features = 12   # 1 rainfall + 2FG_S25A + 2FG_S25B + 2FG_S26 + 8ws + pump_S26
K = 12


# Pre-processing
# Stage --> 8 stages
stages = data[['WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26']]
print("stages.shape:", stages.shape)
stages_supervised = stage_series_to_supervised(stages, n_hours, K, 1)
print("stages_supervised.shape:", stages_supervised.shape)


# Non-Stage --> 7 = 1 rainfall + 2FG_S25A + 2FG_S25B + 'PUMP_S26'
# non_stages = data[['FLOW_S25A', 'GATE_S25A', 'FLOW_S25B', 'GATE_S25B', 'FLOW_S26', 'GATE_S26', 'PUMP_S26', 'mean']]
# non_stages = data[['FLOW_S25A', 'GATE_S25A', 'FLOW_S25B', 'GATE_S25B', 'FLOW_S26', 'GATE_S26', 'mean']]
# non_stages = data[['GATE_S25A', 'GATE_S25B', 'GATE_S26', 'PUMP_S26', 'mean']]
non_stages = data[['GATE_S25A', 'GATE_S25B', 'GATE_S26', 'mean']]
print("non_stages.shape:", non_stages.shape)
non_stages_supervised = series_to_supervised(non_stages, n_hours, 1)
print("non_stages_supervised.shape:", non_stages_supervised.shape)


# Concatenation
non_stages.reset_index(drop=True, inplace=True)
non_stages_supervised.reset_index(drop=True, inplace=True)
stages_supervised.reset_index(drop=True, inplace=True)
all_data = concat([non_stages.iloc[0:len(stages_supervised), -1],    # add rainfall to measure heavy/medium/light
                   non_stages_supervised.iloc[0:len(stages_supervised), 0:-non_stages.shape[1]],
                   stages_supervised.iloc[:, :-3]],
                   axis=1)

# print("all_data", all_data)
print("all_data.shape:", all_data.shape)

# split into train and test sets
all_data = all_data.values
n_train_hours = int(len(all_data)*0.8)
print("n_train_hours:", n_train_hours)


train = all_data[:n_train_hours, 1:]    # 0 column is the rainfall to measure heavy/medium/light
test = all_data[n_train_hours:, 1:]

#
# rain_trainset = pd.DataFrame(all_data[:n_train_hours, 0])
# rain_testset = pd.DataFrame(all_data[n_train_hours:, 0])
#
# data_00_15 = rain_testset.loc[rain_testset.iloc[:, 0] < 0.15]
# data_15_25 = rain_testset.loc[(rain_testset.iloc[:, 0] >= 0.15) & (rain_testset.iloc[:, 0] <= 0.25)]
# data_25_50 = rain_testset.loc[rain_testset.iloc[:, 0] > 0.25]
# data_00_15_index = list(data_00_15.index)
# data_15_25_index = list(data_15_25.index)
# data_25_50_index = list(data_25_50.index)
# print("data_00_15_index:", len(data_00_15_index), data_00_15_index)
# print("data_15_25_index:", len(data_15_25_index), data_15_25_index)
# print("data_25_50_index:", len(data_25_50_index), data_25_50_index)
# print("len(data_25_50_index):", len(data_25_50_index))

# traindata_00_15 = rain_trainset.loc[rain_trainset.iloc[:, 0] < 0.15]
# traindata_15_25 = rain_trainset.loc[(rain_trainset.iloc[:, 0] >= 0.15) & (rain_trainset.iloc[:, 0] <= 0.25)]
# traindata_25_50 = rain_trainset.loc[rain_trainset.iloc[:, 0] > 0.25]
# traindata_00_15_index = list(traindata_00_15.index)
# traindata_15_25_index = list(traindata_15_25.index)
# traindata_25_50_index = list(traindata_25_50.index)
# print("data_00_15_index:", len(data_00_15_index), data_00_15_index)
# print("data_15_25_index:", len(data_15_25_index), data_15_25_index)
# print("data_25_50_index:", len(data_25_50_index), data_25_50_index)
# print("len(traindata_25_50_index):", len(traindata_25_50_index))


# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -5:]   # 5 means 5 WS among all 8 WS except 3 HWS
test_X, test_y = test[:, :n_obs], test[:, -5:]
print("train_X.shape, train_y.shape, test_X.shape, test_y.shape", train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
train_X = scaler.fit_transform(train_X)
train_y = scaler.fit_transform(train_y)
test_X = scaler.fit_transform(test_X)
test_y = scaler.fit_transform(test_y)


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print("train_X.shape, train_y.shape, test_X.shape, test_y.shape: \n", train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# Simple RNN Model
model = keras.Sequential()
model.add(layers.Flatten(input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
# model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(5))   # Regression -> No Need for Activation


lr = 0.00001
EPOCHS = 100
model.compile(
              optimizer=Adam(learning_rate=lr, decay=lr/EPOCHS),
              # optimizer='adam',
              loss='mse',
              metrics=['mae'])
history = model.fit(train_X, train_y,
                    batch_size=256,
                    epochs=EPOCHS,
                    validation_data=(test_X, test_y),
                    verbose=2,
                    shuffle=False)

plt.rcParams["figure.figsize"] = (8, 6)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.legend(fontsize=14)
plt.title("Training loss V.S. Testing loss", fontsize=18)
# plt.savefig('graph/rnn_loss.png', dpi=300)
plt.show()
plt.close()


yhat = model.predict(test_X)
inv_yhat = scaler.inverse_transform(yhat)
inv_y = scaler.inverse_transform(test_y)

inv_yhat = pd.DataFrame(inv_yhat)
inv_y = pd.DataFrame(inv_y)
print("inv_y.shape, inv_yhat.shape", inv_y.shape, inv_yhat.shape)


# # Heavy Storm: WS = ['WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']
# Heavy_RMSE, Heavy_MAE = [], []
# for i in range(inv_yhat.shape[1]):
#     RMSE = sqrt(mean_squared_error(inv_y.loc[data_25_50_index, i], inv_yhat.loc[data_25_50_index, i]))
#     MAE = mean_absolute_error(inv_y.loc[data_25_50_index, i], inv_yhat.loc[data_25_50_index, i])
#     Heavy_RMSE.append(float("{:.4f}".format(RMSE)))
#     Heavy_MAE.append(float("{:.4f}".format(MAE)))
# print("(Heavy Storm) Test RMSE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26:", Heavy_RMSE)
# print("(Heavy Storm) Test MAE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26:", Heavy_MAE)
#
# # Medium Storm: WS = ['WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']
# Medium_RMSE, Medium_MAE = [], []
# for i in range(inv_yhat.shape[1]):
#     RMSE = sqrt(mean_squared_error(inv_y.loc[data_15_25_index, i], inv_yhat.loc[data_15_25_index, i]))
#     MAE = mean_absolute_error(inv_y.loc[data_15_25_index, i], inv_yhat.loc[data_15_25_index, i])
#     Medium_RMSE.append(float("{:.4f}".format(RMSE)))
#     Medium_MAE.append(float("{:.4f}".format(MAE)))
# print("(Medium Storm) Test RMSE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26:", Medium_RMSE)
# print("(Medium Storm) Test MAE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26:", Medium_MAE)
#
# # Small Storm: WS = ['WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']
# Small_RMSE, Small_MAE = [], []
# for i in range(inv_yhat.shape[1]):
#     RMSE = sqrt(mean_squared_error(inv_y.loc[data_00_15_index, i], inv_yhat.loc[data_00_15_index, i]))
#     MAE = mean_absolute_error(inv_y.loc[data_00_15_index, i], inv_yhat.loc[data_00_15_index, i])
#     Small_RMSE.append(float("{:.4f}".format(RMSE)))
#     Small_MAE.append(float("{:.4f}".format(MAE)))
# print("(Small Storm) Test RMSE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26:", Small_RMSE)
# print("(Small Storm) Test MAE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26:", Small_MAE)


# Whole test set: WS = ['WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']
RMSES, MAES = [], []
for i in range(inv_yhat.shape[1]):
    RMSE = sqrt(mean_squared_error(inv_y.iloc[:, i], inv_yhat.iloc[:, i]))
    MAE = mean_absolute_error(inv_y.iloc[:, i], inv_yhat.iloc[:, i])
    RMSES.append(float("{:.4f}".format(RMSE)))
    MAES.append(float("{:.4f}".format(MAE)))
print("Test RMSE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26:", RMSES)
print("Test MAE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26:", MAES)


# # 'WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26'
plt.rcParams["figure.figsize"] = (8, 6)
plt.title('Comparison RMSE & MAE at Different Locations', fontsize=18)
plt.plot(RMSES, label='RMSE', marker='D')
plt.plot(MAES, label='MAE', marker='o')
plt.xlabel('Locations', fontsize=16)
plt.ylabel('Error', fontsize=16)
plt.xticks(np.arange(5), ['S1', 'S4', 'S25A', 'S25B', 'S26'], fontsize=14)
plt.yticks(fontsize=14)
# plt.axhline(y=0.15, color='red', linestyle='-', linewidth=2)
# plt.axhline(y=0.25, color='orange', linestyle='-', linewidth=2)
# plt.text(0, 0.26, 'R=0.25', fontsize=14)
# plt.text(0, 0.16, 'R=0.15', fontsize=14)
plt.legend(fontsize=14)
plt.show()

#
# date = ['09/10', '09/11', '09/12', '09/13', '09/14', '09/15', '09/16']
# # # 'WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26'  [99635:100499, 0]
# RMSE_WS_S1 = sqrt(mean_squared_error(inv_y.iloc[99635:100499, 0], inv_yhat.iloc[99635:100499, 0]))
# MAE_WS_S1 = mean_absolute_error(inv_y.iloc[99635:100499, 0], inv_yhat.iloc[99635:100499, 0])
# plt.rcParams["figure.figsize"] = (8, 6)
#
# plt.plot(inv_yhat.iloc[99635:100499, 0], label='prediction', linewidth=2)
# plt.plot(inv_y.iloc[99635:100499, 0], label='truth', linewidth=2)
# plt.title('Predicted & Actual Value of WS_S1', fontsize=18)
# plt.text(99858, 3.35, 'MAE: {:.4f}'.format(MAE_WS_S1), fontsize=13)
# plt.xlabel('Time', fontsize=16)
# plt.ylabel('Water Stage', fontsize=16)
# plt.xticks(np.arange(99635, 100500, 144), date, fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=13, loc='upper left')
# plt.show()
# plt.close()
#
# # # 'WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26'
# RMSE_WS_S4 = sqrt(mean_squared_error(inv_y.iloc[99635:100499, 1], inv_yhat.iloc[99635:100499, 1]))
# MAE_WS_S4 = mean_absolute_error(inv_y.iloc[99635:100499, 1], inv_yhat.iloc[99635:100499, 1])
# plt.rcParams["figure.figsize"] = (8, 6)
# plt.plot(inv_yhat.iloc[99635:100499, 1], label='prediction', linewidth=2)
# plt.plot(inv_y.iloc[99635:100499, 1], label='truth', linewidth=2)
# plt.title('Predicted & Actual Value of WS_S4', fontsize=18)
# plt.text(99858, 3.3, 'MAE: {:.4f}'.format(MAE_WS_S4), fontsize=13)
# plt.xlabel('Time', fontsize=16)
# plt.ylabel('Water Stage', fontsize=16)
# plt.xticks(np.arange(99635, 100500, 144), date, fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=13, loc='upper left')
# plt.show()
# plt.close()
#
# # # 'WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26'
# RMSE_TWS_S25A = sqrt(mean_squared_error(inv_y.iloc[99635:100499, 2], inv_yhat.iloc[99635:100499, 2]))
# MAE_TWS_S25A = mean_absolute_error(inv_y.iloc[99635:100499, 2], inv_yhat.iloc[99635:100499, 2])
# plt.rcParams["figure.figsize"] = (8, 6)
# plt.plot(inv_yhat.iloc[99635:100499, 2], label='prediction', linewidth=2)
# plt.plot(inv_y.iloc[99635:100499, 2], label='truth', linewidth=2)
# plt.title('Predicted & Actual Value of TWS_S25A', fontsize=18)
# plt.text(99858, 3.35, 'MAE: {:.4f}'.format(MAE_TWS_S25A), fontsize=13)
# plt.xlabel('Time', fontsize=16)
# plt.ylabel('Water Stage', fontsize=16)
# plt.xticks(np.arange(99635, 100500, 144), date, fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=13, loc='upper left')
# plt.show()
# plt.close()
#
# # # 'WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26'
# RMSE_TWS_S25B = sqrt(mean_squared_error(inv_y.iloc[99635:100499, 3], inv_yhat.iloc[99635:100499, 3]))
# MAE_TWS_S25B = mean_absolute_error(inv_y.iloc[99635:100499, 3], inv_yhat.iloc[99635:100499, 3])
# plt.rcParams["figure.figsize"] = (8, 6)
# plt.plot(inv_yhat.iloc[99635:100499, 3], label='prediction', linewidth=2)
# plt.plot(inv_y.iloc[99635:100499, 3], label='truth', linewidth=2)
# plt.title('Predicted & Actual Value of TWS_S25B', fontsize=18)
# plt.text(99858, 3.35, 'MAE: {:.4f}'.format(MAE_TWS_S25B), fontsize=13)
# plt.xlabel('Time', fontsize=16)
# plt.ylabel('Water Stage', fontsize=16)
# plt.xticks(np.arange(99635, 100500, 144), date, fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=13, loc='upper left')
# plt.show()
# plt.close()
#
# # # 'WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26'
# RMSE_TWS_S26 = sqrt(mean_squared_error(inv_y.iloc[99635:100499, 4], inv_yhat.iloc[99635:100499, 4]))
# MAE_TWS_S26 = mean_absolute_error(inv_y.iloc[99635:100499, 4], inv_yhat.iloc[99635:100499, 4])
# plt.rcParams["figure.figsize"] = (8, 6)
# plt.plot(inv_yhat.iloc[99635:100499, 4], label='prediction', linewidth=2)
# plt.plot(inv_y.iloc[99635:100499, 4], label='truth', linewidth=2)
# plt.title('Predicted & Actual Value of TWS_S26', fontsize=18)
# plt.text(99858, 3.35, 'MAE: {:.4f}'.format(MAE_TWS_S26), fontsize=13)
# plt.xlabel('Time', fontsize=16)
# plt.ylabel('Water Stage', fontsize=16)
# plt.xticks(np.arange(99635, 100500, 144), date, fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=13, loc='upper left')
# plt.show()
# plt.close()
#
#
# date = ['09/10', '09/11', '09/12', '09/13', '09/14', '09/15', '09/16']
# # # 'WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26'  --> 99635:100499
# RMSE_WS_S1 = sqrt(mean_squared_error(inv_y.iloc[:216, 0], inv_yhat.iloc[:216, 0]))
# MAE_WS_S1 = mean_absolute_error(inv_y.iloc[:216, 0], inv_yhat.iloc[:216, 0])
# plt.rcParams["figure.figsize"] = (8, 6)
#
# plt.plot(inv_yhat.iloc[:216, 0], label='prediction', linewidth=2)
# plt.plot(inv_y.iloc[:216, 0], label='truth', linewidth=2)
# plt.title('Predicted & Actual Value of WS_S1', fontsize=18)
# # plt.text(99858, 3.35, 'MAE: {:.4f}'.format(MAE_WS_S1), fontsize=13)
# plt.xlabel('Time', fontsize=16)
# plt.ylabel('Water Stage', fontsize=16)
# # plt.xticks(np.arange(99635, 100500, 144), date, fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=13, loc='upper left')
# plt.show()
# plt.close()
#
# # # 'WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26' --> 99635:100499
# RMSE_WS_S4 = sqrt(mean_squared_error(inv_y.iloc[:216, 1], inv_yhat.iloc[:216, 1]))
# MAE_WS_S4 = mean_absolute_error(inv_y.iloc[:216, 1], inv_yhat.iloc[:216, 1])
# plt.rcParams["figure.figsize"] = (8, 6)
# plt.plot(inv_yhat.iloc[:216, 1], label='prediction', linewidth=2)
# plt.plot(inv_y.iloc[:216, 1], label='truth', linewidth=2)
# plt.title('Predicted & Actual Value of WS_S4', fontsize=18)
# # plt.text(99858, 3.3, 'MAE: {:.4f}'.format(MAE_WS_S4), fontsize=13)
# plt.xlabel('Time', fontsize=16)
# plt.ylabel('Water Stage', fontsize=16)
# # plt.xticks(np.arange(99635, 100500, 144), date, fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=13, loc='upper left')
# plt.show()
# plt.close()
#
# # # 'WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26' --> 99635:100499
# RMSE_TWS_S25A = sqrt(mean_squared_error(inv_y.iloc[:216, 2], inv_yhat.iloc[:216, 2]))
# MAE_TWS_S25A = mean_absolute_error(inv_y.iloc[:216, 2], inv_yhat.iloc[:216, 2])
# plt.rcParams["figure.figsize"] = (8, 6)
# plt.plot(inv_yhat.iloc[:216, 2], label='prediction', linewidth=2)
# plt.plot(inv_y.iloc[:216, 2], label='truth', linewidth=2)
# plt.title('Predicted & Actual Value of TWS_S25A', fontsize=18)
# # plt.text(99858, 3.35, 'MAE: {:.4f}'.format(MAE_TWS_S25A), fontsize=13)
# plt.xlabel('Time', fontsize=16)
# plt.ylabel('Water Stage', fontsize=16)
# # plt.xticks(np.arange(99635, 100500, 144), date, fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=13, loc='upper left')
# plt.show()
# plt.close()
#
# # # 'WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26' --> 99635:100499
# RMSE_TWS_S25B = sqrt(mean_squared_error(inv_y.iloc[:216, 3], inv_yhat.iloc[:216, 3]))
# MAE_TWS_S25B = mean_absolute_error(inv_y.iloc[:216, 3], inv_yhat.iloc[:216, 3])
# plt.rcParams["figure.figsize"] = (8, 6)
# plt.plot(inv_yhat.iloc[:216, 3], label='prediction', linewidth=2)
# plt.plot(inv_y.iloc[:216, 3], label='truth', linewidth=2)
# plt.title('Predicted & Actual Value of TWS_S25B', fontsize=18)
# # plt.text(99858, 3.35, 'MAE: {:.4f}'.format(MAE_TWS_S25B), fontsize=13)
# plt.xlabel('Time', fontsize=16)
# plt.ylabel('Water Stage', fontsize=16)
# # plt.xticks(np.arange(99635, 100500, 144), date, fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=13, loc='upper left')
# plt.show()
# plt.close()
#
# # # 'WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26' --> 99635:100499
# RMSE_TWS_S26 = sqrt(mean_squared_error(inv_y.iloc[:216, 4], inv_yhat.iloc[:216, 4]))
# MAE_TWS_S26 = mean_absolute_error(inv_y.iloc[:216, 4], inv_yhat.iloc[:216, 4])
# plt.rcParams["figure.figsize"] = (8, 6)
# plt.plot(inv_yhat.iloc[:216, 4], label='prediction', linewidth=2)
# plt.plot(inv_y.iloc[:216, 4], label='truth', linewidth=2)
# plt.title('Predicted & Actual Value of TWS_S26', fontsize=18)
# # plt.text(99858, 3.35, 'MAE: {:.4f}'.format(MAE_TWS_S26), fontsize=13)
# plt.xlabel('Time', fontsize=16)
# plt.ylabel('Water Stage', fontsize=16)
# # plt.xticks(np.arange(99635, 100500, 144), date, fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=13, loc='upper left')
# plt.show()
# plt.close()

# (Heavy Storm) Test RMSE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26: [0.4645, 0.3176, 0.3956, 0.5927, 0.6795]
# (Heavy Storm) Test MAE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26: [0.334, 0.2699, 0.3217, 0.3823, 0.5489]
# (Medium Storm) Test RMSE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26: [0.4029, 0.4254, 0.3291, 0.4103, 0.4894]
# (Medium Storm) Test MAE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26: [0.331, 0.2968, 0.2226, 0.2703, 0.4083]
# (Small Storm) Test RMSE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26: [0.2846, 0.1992, 0.2153, 0.2193, 0.6075]
# (Small Storm) Test MAE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26: [0.2484, 0.1614, 0.1486, 0.1631, 0.4984]

# (Heavy Storm) Test RMSE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26: [0.4645, 0.3176, 0.3956, 0.5927, 0.6795]
# (Heavy Storm) Test MAE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26: [0.334, 0.2699, 0.3217, 0.3823, 0.5489]
# (Medium Storm) Test RMSE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26: [0.4029, 0.4254, 0.3291, 0.4103, 0.4894]
# (Medium Storm) Test MAE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26: [0.331, 0.2968, 0.2226, 0.2703, 0.4083]
# (Small Storm) Test RMSE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26: [0.2846, 0.1992, 0.2153, 0.2193, 0.6075]
# (Small Storm) Test MAE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26: [0.2484, 0.1614, 0.1486, 0.1631, 0.4984]