#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : ConvTransformerTS
@ FileName: plot_rain_distribution.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 1/25/22 16:06
"""
from pandas import read_csv
import matplotlib.pyplot as plt

dataset = read_csv('data/zeda/Merged_with_GridRain.csv', index_col=0)
dataset.fillna(0, inplace=True)

# copy the data
df_z_scaled = dataset.copy()

# apply normalization technique to Column 1
column = 'Column 1'
df_z_scaled[column] = (df_z_scaled[column] - df_z_scaled[column].mean()) / df_z_scaled[column].std()

# view normalized data
print(df_z_scaled)