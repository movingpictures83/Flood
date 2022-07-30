#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : ConvTransformerTS
@ FileName: draft.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 12/28/21 17:38
"""

from tensorflow import keras
from tensorflow.keras import layers
from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.layers import SimpleRNN
# from tensorflow.keras.layers.core import Dense, Dropout
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.optimizers import Adam
from numpy import savetxt
from pandas import DataFrame
from pandas import concat
import numpy as np
from helper import series_to_supervised, stage_series_to_supervised
from pandas import read_csv
from pandas import read_excel

dataset = read_csv('data/zeda/Merged_with_GridRain.csv', index_col=0)
dataset.fillna(0, inplace=True)

data_ = dataset[:578448]
print(data_.shape)
data = data_.drop(columns=['RAIN_S26'])

print(data.shape)
print(list(data.columns))
