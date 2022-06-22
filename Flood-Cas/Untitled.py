#!/usr/bin/env python
# coding: utf-8

# In[2]:


# from tensorflow import keras
# from tensorflow.keras import layers
# from math import sqrt
# from numpy import concatenate
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from keras.layers import SimpleRNN
# from keras.layers.core import Dense, Dropout
# import matplotlib.pyplot as plt
# import pandas as pd
# from tensorflow.keras.optimizers import Adam
# from pandas import concat
# import numpy as np
# from helper import series_to_supervised, stage_series_to_supervised
# from pandas import read_csv


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# ### Dataset

# In[30]:


data1 = pd.read_csv('data/zeda/Merged_with_MeanGridRain.csv', index_col=0)
data1


# In[49]:


data1.to_csv('./data/zeda/Merged_all')


# In[31]:


data1.describe()


# In[33]:


data1['mean'].value_counts()


# In[51]:


data1[data1['mean'] == 0]


# In[50]:


data1[data1['mean'] > 0]


# In[ ]:





# In[37]:


# dataset = read_csv('data/zeda/Merged_with_MeanGridRain.csv', index_col=0)
dataset = pd.read_csv('data/zeda/Merged.csv', index_col=0)
dataset = dataset[:578448]
dataset


# ### Dataset Information

# In[38]:


print(list(dataset.columns))


# In[41]:


dates = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021'] 
# 2020: 2010-01-01-00:00
plt.plot(dataset.loc[:, 'mean'])
plt.xlabel('Time', fontsize='14')
plt.ylabel('Rainfall', fontsize='14')
plt.xticks(np.arange(0, 578593, 52599), dates, rotation=30, fontsize=12)
plt.yticks(fontsize=12)
plt.title('Mean of Rainfall (2010-2020)')
plt.savefig('./data/rain_mean.png', dpi=300)
plt.show()


# In[42]:


count_0000_0001 = dataset.loc[dataset.loc[:, 'mean'] < 0.0001]
count_0001_0006 = dataset.loc[(dataset.loc[:, 'mean'] >= 0.0001) & (dataset.loc[:, 'mean'] < 0.0006)]
count_0006_0013 = dataset.loc[(dataset.loc[:, 'mean'] >= 0.0006) & (dataset.loc[:, 'mean'] < 0.0013)]
count_0013_0200 = dataset.loc[(dataset.loc[:, 'mean'] >= 0.0013) & (dataset.loc[:, 'mean'] < 0.002)]  # 0.0013 - 0.02


count_0200_0400 = dataset.loc[(dataset.loc[:, 'mean'] >= 0.02) & (dataset.loc[:, 'mean'] < 0.04)]  # 0.02 - 0.04
count_0400_0600 = dataset.loc[(dataset.loc[:, 'mean'] >= 0.04) & (dataset.loc[:, 'mean'] < 0.06)]  # 0.04 - 0.06 
count_0600_0800 = dataset.loc[(dataset.loc[:, 'mean'] >= 0.06) & (dataset.loc[:, 'mean'] < 0.08)]  # 0.06 - 0.08 
count_0800_0100 = dataset.loc[(dataset.loc[:, 'mean'] >= 0.08) & (dataset.loc[:, 'mean'] < 0.10)]  # 0.08 - 0.10 


count_1000_2000 = dataset.loc[(dataset.loc[:, 'mean'] >= 0.10) & (dataset.loc[:, 'mean'] < 0.20)]  # 0.10 - 0.20
count_2000_3000 = dataset.loc[(dataset.loc[:, 'mean'] >= 0.20) & (dataset.loc[:, 'mean'] < 0.30)]  # 0.20 - 0.30
count_3000_4000 = dataset.loc[(dataset.loc[:, 'mean'] >= 0.30) & (dataset.loc[:, 'mean'] < 0.40)]  # 0.30 - 0.40
count_4000_ = dataset.loc[dataset.loc[:, 'mean'] >= 0.40]                                          # >= 0.4


# In[43]:


print("<=0.0001: ", len(count_0000_0001))
print("0.0001 - 0.0006: ", len(count_0006_0013))
print("0.0006 - 0.0013: ", len(count_0006_0013))
print("0.0013 - 0.020: ", len(count_0013_0200))

print("0.02 - 0.04: ", len(count_0002_0004))
print("0.04 - 0.06: ", len(count_0004_0006))
print("0.06 - 0.08: ", len(count_0006_0008))
print("0.08 - 0.10: ", len(count_0008_0010))

print("0.10 - 0.20: ", len(count_0010_0020))
print("0.20 - 0.30: ", len(count_0020_0030))
print("0.20 - 0.30: ", len(count_0030_0040))
print(">= 0.4: ", len(count_0040_0050))


# In[45]:


dataset['mean'].value_counts()
# no-rain: 501,343
# rain: 77,105


# In[20]:


sns.histplot(dataset['mean'])


# In[21]:


list(count_0040_0050.index)


# In[26]:


dataset.describe()


# In[25]:


data = dataset.loc[:, 'mean']
data.describe()


# In[ ]:





# ### N_out, N_in, K

# In[22]:


# specify the number of lag hours
n_hours = 72
n_features = 15   # 1 rainfall + 2FG_S25A + 2FG_S25B + 2FG_S26 + 8ws + pump_S26
K = 12


# ### Pre-processing

# #### Stage

# In[23]:


# Pre-processing
# Stage --> 8 stages
stages = dataset[['WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26']]
print("stages.shape:", stages.shape)

stages_supervised = stage_series_to_supervised(stages, n_hours, K, 1)
print("stages_supervised.shape:", stages_supervised.shape)


# In[26]:


stages_supervised


# #### Non-stage

# In[25]:


# Non-Stage --> 7 = 1 rainfall + 2FG_S25A + 2FG_S25B + 'PUMP_S26'
non_stages = dataset[['FLOW_S25A', 'GATE_S25A', 'FLOW_S25B', 'GATE_S25B', 'FLOW_S26', 'GATE_S26', 'mean']]
print("non_stages.shape:", non_stages.shape)

non_stages_supervised = series_to_supervised(non_stages, n_hours, 1)
print("non_stages_supervised.shape:", non_stages_supervised.shape)


# In[27]:


non_stages_supervised


# ### Concatenation

# In[28]:


non_stages.reset_index(drop=True, inplace=True)
non_stages_supervised.reset_index(drop=True, inplace=True)
stages_supervised.reset_index(drop=True, inplace=True)

all_data = concat([non_stages.iloc[0:len(stages_supervised), -1],    # add rainfall to measure heavy/medium/light
                   non_stages_supervised.iloc[0:len(stages_supervised), 0:-non_stages.shape[1]],
                   stages_supervised.iloc[:, :-3]],
                   axis=1)


# In[30]:


# print("all_data", all_data)
print("all_data.shape:", all_data.shape)


# ### Train & Test set

# In[31]:


all_data = all_data.values
n_train_hours = int(len(all_data)*0.8)
print("n_train_hours:", n_train_hours)


train = all_data[:n_train_hours, 1:]    # 0 column is the rainfall to measure heavy/medium/light
test = all_data[n_train_hours:, 1:]


# In[ ]:


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


# ### Normalization

# In[32]:


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


# In[33]:


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print("train_X.shape, train_y.shape, test_X.shape, test_y.shape: \n", train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# ### Model

# In[34]:


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


# In[35]:


lr = 0.00001
EPOCHS = 100
model.compile(
              optimizer=Adam(learning_rate=lr, decay=lr/EPOCHS),
#               optimizer='adam',
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


# ### Prediction

# In[36]:


yhat = model.predict(test_X)
inv_yhat = scaler.inverse_transform(yhat)
inv_y = scaler.inverse_transform(test_y)

inv_yhat = pd.DataFrame(inv_yhat)
inv_y = pd.DataFrame(inv_y)
print("inv_y.shape, inv_yhat.shape", inv_y.shape, inv_yhat.shape)


# ### Performance

# In[38]:


# # Whole test set: WS = ['WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']
# RMSES, MAES = [], []
# for i in range(inv_yhat.shape[1]):
#     RMSE = sqrt(mean_squared_error(inv_y.iloc[:, i], inv_yhat.iloc[:, i]))
#     MAE = mean_absolute_error(inv_y.iloc[:, i], inv_yhat.iloc[:, i])
#     RMSES.append(float("{:.4f}".format(RMSE)))
#     MAES.append(float("{:.4f}".format(MAE)))
# print("Test RMSE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26:", RMSES)
# print("Test MAE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26:", MAES)


# In[ ]:


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


date = ['09/10', '09/11', '09/12', '09/13', '09/14', '09/15', '09/16']
# # 'WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26'  [99635:100499, 0]
RMSE_WS_S1 = sqrt(mean_squared_error(inv_y.iloc[99635:100499, 0], inv_yhat.iloc[99635:100499, 0]))
MAE_WS_S1 = mean_absolute_error(inv_y.iloc[99635:100499, 0], inv_yhat.iloc[99635:100499, 0])
plt.rcParams["figure.figsize"] = (8, 6)

plt.plot(inv_yhat.iloc[99635:100499, 0], label='prediction', linewidth=2)
plt.plot(inv_y.iloc[99635:100499, 0], label='truth', linewidth=2)
plt.title('Predicted & Actual Value of WS_S1', fontsize=18)
plt.text(99858, 3.35, 'MAE: {:.4f}'.format(MAE_WS_S1), fontsize=13)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Water Stage', fontsize=16)
plt.xticks(np.arange(99635, 100500, 144), date, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=13, loc='upper left')
plt.show()
plt.close()

# # 'WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26'
RMSE_WS_S4 = sqrt(mean_squared_error(inv_y.iloc[99635:100499, 1], inv_yhat.iloc[99635:100499, 1]))
MAE_WS_S4 = mean_absolute_error(inv_y.iloc[99635:100499, 1], inv_yhat.iloc[99635:100499, 1])
plt.rcParams["figure.figsize"] = (8, 6)
plt.plot(inv_yhat.iloc[99635:100499, 1], label='prediction', linewidth=2)
plt.plot(inv_y.iloc[99635:100499, 1], label='truth', linewidth=2)
plt.title('Predicted & Actual Value of WS_S4', fontsize=18)
plt.text(99858, 3.3, 'MAE: {:.4f}'.format(MAE_WS_S4), fontsize=13)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Water Stage', fontsize=16)
plt.xticks(np.arange(99635, 100500, 144), date, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=13, loc='upper left')
plt.show()
plt.close()

# # 'WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26'
RMSE_TWS_S25A = sqrt(mean_squared_error(inv_y.iloc[99635:100499, 2], inv_yhat.iloc[99635:100499, 2]))
MAE_TWS_S25A = mean_absolute_error(inv_y.iloc[99635:100499, 2], inv_yhat.iloc[99635:100499, 2])
plt.rcParams["figure.figsize"] = (8, 6)
plt.plot(inv_yhat.iloc[99635:100499, 2], label='prediction', linewidth=2)
plt.plot(inv_y.iloc[99635:100499, 2], label='truth', linewidth=2)
plt.title('Predicted & Actual Value of TWS_S25A', fontsize=18)
plt.text(99858, 3.35, 'MAE: {:.4f}'.format(MAE_TWS_S25A), fontsize=13)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Water Stage', fontsize=16)
plt.xticks(np.arange(99635, 100500, 144), date, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=13, loc='upper left')
plt.show()
plt.close()

# # 'WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26'
RMSE_TWS_S25B = sqrt(mean_squared_error(inv_y.iloc[99635:100499, 3], inv_yhat.iloc[99635:100499, 3]))
MAE_TWS_S25B = mean_absolute_error(inv_y.iloc[99635:100499, 3], inv_yhat.iloc[99635:100499, 3])
plt.rcParams["figure.figsize"] = (8, 6)
plt.plot(inv_yhat.iloc[99635:100499, 3], label='prediction', linewidth=2)
plt.plot(inv_y.iloc[99635:100499, 3], label='truth', linewidth=2)
plt.title('Predicted & Actual Value of TWS_S25B', fontsize=18)
plt.text(99858, 3.35, 'MAE: {:.4f}'.format(MAE_TWS_S25B), fontsize=13)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Water Stage', fontsize=16)
plt.xticks(np.arange(99635, 100500, 144), date, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=13, loc='upper left')
plt.show()
plt.close()

# # 'WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26'
RMSE_TWS_S26 = sqrt(mean_squared_error(inv_y.iloc[99635:100499, 4], inv_yhat.iloc[99635:100499, 4]))
MAE_TWS_S26 = mean_absolute_error(inv_y.iloc[99635:100499, 4], inv_yhat.iloc[99635:100499, 4])
plt.rcParams["figure.figsize"] = (8, 6)
plt.plot(inv_yhat.iloc[99635:100499, 4], label='prediction', linewidth=2)
plt.plot(inv_y.iloc[99635:100499, 4], label='truth', linewidth=2)
plt.title('Predicted & Actual Value of TWS_S26', fontsize=18)
plt.text(99858, 3.35, 'MAE: {:.4f}'.format(MAE_TWS_S26), fontsize=13)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Water Stage', fontsize=16)
plt.xticks(np.arange(99635, 100500, 144), date, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=13, loc='upper left')
plt.show()
plt.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[153]:


dataset['date'] = pd.to_datetime(dataset['Time']).dt.date
dataset


# In[133]:


dataset.iloc[:, -30:]


# In[154]:


print(dataset.iloc[:, 18:49].mean())


# In[ ]:





# In[134]:


# temp = dataset.iloc[:, -30]
# temp.max()
dataset


# In[39]:


data = dataset.iloc[:, -2:]
data


# In[50]:


plt.plot(data.loc[:, 'mean'])
plt.grid()


# In[105]:


import matplotlib.dates as mdates
fig, ax = plt.subplots(1, 1) # figsize=(18, 16), constrained_layout=True
ax.plot('date', 'mean_normalized', data=data)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 1)))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')


# In[135]:


print(dataset.columns)


# In[147]:


rain = dataset.iloc[:, 18:43]
rain


# In[150]:


# print(rain.mean())
print(rain.max())
# print(data['mean'].std())


# In[40]:


column = 'mean'
data['mean_normalized'] = (data[column] - data[column].mean()) / data[column].std()


# In[41]:


data


# In[117]:


data.describe()


# In[111]:


print(data['mean_normalized'].min())
print(data['mean_normalized'].max())


# In[122]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
  
# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-10, 10, 0.01)
  
# Calculating mean and standard deviation
mean = 0.0016015965350384123
std = 0.010961477599057507
  
plt.plot(x_axis, norm.pdf(x_axis, mean, std))
plt.show()


# In[155]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
  
# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-0.1, 0.1, 0.01)
  
# Calculating mean and standard deviation
mean = 0.0016015965350384123
std = 0.010961477599057507
  
plt.plot(x_axis, norm.pdf(x_axis, mean, std))
plt.show()


# In[64]:


import matplotlib.dates as mdates
fig, ax = plt.subplots(1, 1) # figsize=(18, 16), constrained_layout=True
ax.plot('date', 'mean_normalized', data=data)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 1)))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')


# In[24]:





# In[95]:


# import pandas as pd
# import numpy as np
 
# #Create a DataFrame
# d = {
#     'Name':['Alisa','Bobby','Cathrine','Madonna','Rocky','Sebastian','Jaqluine',
#    'Rahul','David','Andrew','Ajay','Teresa'],
# #    'Score1':[62,47,55,74,31,77,85,63,42,32,71,57],
# #    'Score2':[89,87,67,55,47,72,76,79,44,92,99,69],
#    'Score3':[56,86,77,45,73,62,74,89,71,67,97,68]}

# df = pd.DataFrame(d)
# df


# In[96]:


# std = df.std()
# print(std)

# mean = df.mean()
# print(mean)


# In[81]:


# # column = 'Score3'
# df['Score3_norm'] = (df[column] - df[column].mean()) / df[column].std()


# In[94]:


# df


# In[93]:


# plt.plot(df['Score3_norm'])


# In[88]:


np.random.seed([3,1415])
df = pd.DataFrame(dict(
        Name='matt joe adam farley'.split() * 100,
        Seconds=np.random.randint(4000, 5000, 400)
    ))

df


# In[97]:


plt.plot(df['Seconds'])


# In[90]:


df['Zscore'] = df.groupby('Name').Seconds.apply(lambda x: x.div(x.mean()))
df


# In[91]:


df.groupby('Name').Zscore.plot.kde()


# In[92]:


data


# In[ ]:


data['Zscore'] = data.groupby('Name').Seconds.apply(lambda x: x.div(x.mean()))


# In[ ]:




