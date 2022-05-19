#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : ConvTransformerTS
@ FileName: plot_testdata.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 1/21/22 14:49
"""
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


dataset = read_csv('data/zeda/Merged_with_GridRain.csv')  # index_col=0
dataset.fillna(0, inplace=True)
dataset['date'] = pd.to_datetime(dataset['Time']).dt.date
data_ = dataset[:578448]
print(data_.shape)
data = data_.drop(columns=['RAIN_S26'])

print(data.shape)
print(list(data.columns))

# data = data.values
n_train_hours = int(len(data)*0.8)         # 462,758 train, 115,690 test
print("n_train_hours:", n_train_hours)

train = data.iloc[:n_train_hours, :]
test = data.iloc[n_train_hours:, :]


# plot rainfall in test set
# test_rain_mean = test.loc[:, 'mean']

# plt.rcParams["figure.figsize"] = (8, 6)
# plt.plot(test_rain_mean)
# plt.title('Rainfall of Test Dataset (115,690 samples)', fontsize=18)
# plt.xlabel('Timestamps', fontsize=18)
# plt.ylabel('Rainfall', fontsize=18)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.axhline(y=0.15, color='red', linestyle='-', linewidth=2)
# plt.axhline(y=0.25, color='orange', linestyle='-', linewidth=2)
# plt.text(0, 0.26, 'R=0.25', fontsize=14)
# plt.text(0, 0.16, 'R=0.15', fontsize=14)
# plt.show()


plt.rcParams["figure.figsize"] = (8, 6)
fig, ax = plt.subplots(1, 1)   # figsize=(18, 16), constrained_layout=True

ax.plot('date', 'mean', data=test)
# Major ticks every half year, minor ticks every month,
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
# ax.set_title('mean', loc='left', y=0.65, x=0.01, fontsize=22)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=16)
ax.tick_params(axis='x', colors='black')  # setting up X-axis tick color to red
# ax.tick_params(axis='y', colors='black')  # setting up Y-axis tick color to black
ax.grid(axis="x")

plt.title('Rainfall of Test Dataset (115,719 samples)', fontsize=18)    # 115,690 samples
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.axhline(y=0.15, color='red', linestyle='-', linewidth=2)
plt.axhline(y=0.25, color='orange', linestyle='-', linewidth=2)
plt.xlabel('Time', fontsize=18)
plt.ylabel('Rainfall', fontsize=18)
# plt.text(2018-10, 0.26, 'R=0.25', fontsize=14)
# plt.text(2018-10, 0.16, 'R=0.15', fontsize=14)
# plt.savefig('graph/Figure7.png', dpi=300)   # if want to save, then need to comment plt.show()
plt.show()
