#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : ConvTransformerTS
@ FileName: rnn_singlestep.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 1/11/22 15:18
"""

import pandas as pd
from pandas import concat
from helper import series_to_supervised, stage_series_to_supervised
from math import sqrt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from pandas import read_csv
from sklearn.metrics import mean_absolute_error


dataset = read_csv('data/zeda/Merged.csv', index_col=0)
# dataset = read_csv('data/zeda/Merged.csv', index_col=0)
dataset.fillna(0, inplace=True)
print(dataset.shape)
print(dataset.columns)

data = dataset[:578448]  # 578448
# data = dataset.loc[:, ['WS_S1', 'WS_S4', 'FLOW_S25A', 'GATE_S25A', 'HWS_S25A', 'TWS_S25A', 'FLOW_S25B', 'GATE_S25B', 'HWS_S25B', 'TWS_S25B', 'FLOW_S26', 'GATE_S26', 'HWS_S26', 'TWS_S26', 'PUMP_S26', 'mean']]

print(data.shape)
print(list(data.columns))

# specify the number of lag hours
n_hours = 72
n_features = 13  # 1 rainfall + 2FG_S25A + 2FG_S25B + 2FG_S26 + 8ws + pump_S26
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
non_stages = data[['GATE_S25A', 'GATE_S25B', 'GATE_S26', 'PUMP_S26', 'mean']]
print("non_stages.shape:", non_stages.shape)
non_stages_supervised = series_to_supervised(non_stages, n_hours, 1)
print("non_stages_supervised.shape:", non_stages_supervised.shape)

# Concatenation
non_stages.reset_index(drop=True, inplace=True)
non_stages_supervised.reset_index(drop=True, inplace=True)
stages_supervised.reset_index(drop=True, inplace=True)
all_data = concat([non_stages.iloc[0:len(stages_supervised), -1],  # add rainfall to measure heavy/medium/light
                   non_stages_supervised.iloc[0:len(stages_supervised), 0:-non_stages.shape[1]],
                   stages_supervised.iloc[:, :-3]],
                  axis=1)

# print("all_data", all_data)
print("all_data.shape:", all_data.shape)

# split into train and test sets
all_data = all_data.values
n_train_hours = int(len(all_data) * 0.8)
print("n_train_hours:", n_train_hours)

train = all_data[:n_train_hours, 1:]  # 0 column is the rainfall to measure heavy/medium/light
test = all_data[n_train_hours:, 1:]


# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -5:]  # 5 means 5 WS among all 8 WS except 3 HWS
test_X, test_y = test[:, :n_obs], test[:, -5:]
print("train_X.shape, train_y.shape, test_X.shape, test_y.shape", train_X.shape, train_y.shape, test_X.shape,
      test_y.shape)

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
train_X = scaler.fit_transform(train_X)
train_y = scaler.fit_transform(train_y)
test_X = scaler.fit_transform(test_X)
test_y = scaler.fit_transform(test_y)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print("train_X.shape, train_y.shape, test_X.shape, test_y.shape: \n", train_X.shape, train_y.shape, test_X.shape,
      test_y.shape)

# LSTM Model
model = Sequential()
model.add(LSTM(75, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(train_y.shape[1]))
model.summary()


# training
lr = 0.00001
EPOCHS = 100
model.compile(
    optimizer=Adam(learning_rate=lr, decay=lr / EPOCHS),
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
plt.title("Training loss vs Testing loss", fontsize=18)
# plt.savefig('graph/rnn_loss.png', dpi=300)
plt.show()
plt.close()

yhat = model.predict(test_X)
inv_yhat = scaler.inverse_transform(yhat)
inv_y = scaler.inverse_transform(test_y)

inv_yhat = pd.DataFrame(inv_yhat)
inv_y = pd.DataFrame(inv_y)
print("inv_y.shape, inv_yhat.shape", inv_y.shape, inv_yhat.shape)


# Whole test set: WS = ['WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']
RMSES, MAES = [], []
for i in range(inv_yhat.shape[1]):
    RMSE = sqrt(mean_squared_error(inv_y.iloc[:, i], inv_yhat.iloc[:, i]))
    MAE = mean_absolute_error(inv_y.iloc[:, i], inv_yhat.iloc[:, i])
    RMSES.append(float("{:.4f}".format(RMSE)))
    MAES.append(float("{:.4f}".format(MAE)))
print("Test RMSE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26:", RMSES)
print("Test MAE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26:", MAES)

# Test error: 'WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26'
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

# # Some period among test set
# date = ['09/10', '09/11', '09/12', '09/13', '09/14', '09/15', '09/16']
# station = ['WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']
# for i in range(5):
#     rmse = sqrt(mean_squared_error(inv_y.iloc[99635:100499, i], inv_yhat.iloc[99635:100499, i]))
#     mae = mean_absolute_error(inv_y.iloc[99635:100499, 0], inv_yhat.iloc[99635:100499, 0])
#     plt.rcParams["figure.figsize"] = (8, 6)
#     plt.plot(inv_yhat.iloc[99635:100499, i], label='prediction', linewidth=2)
#     plt.plot(inv_y.iloc[99635:100499, i], label='truth', linewidth=2)
#     plt.title('Predicted & Actual Value of {}'.format(station[i]), fontsize=18)
#     # plt.text(99858, 3.35, 'MAE: {:.4f}'.format(MAE_WS_S1), fontsize=13)
#     plt.xlabel('Time', fontsize=16)
#     plt.ylabel('Water Stage', fontsize=16)
#     plt.xticks(np.arange(99635, 100500, 144), date, fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.legend(fontsize=13, loc='upper left')
#     plt.show()
#     plt.close()