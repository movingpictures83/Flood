#!/usr/bin/env python
# coding: utf-8

# In[2]:

# In[4]:

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

dataset = pd.read_csv('Dummy CSV - Sheet1.csv')
train = dataset[0:4]
test = dataset[5:]

y_hat_avg = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train['Humidity'])).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot( train['Humidity'], label='Train')
plt.plot(test['Humidity'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()


dataset.columns = ['num', 'Date', 'Time', 'Temperature', 'Wind', 'Pressure',
       'Humidity', 'Precipation', 'Heavyrainfall', 'LMH']

dataset.columns




# HOLT'S WINTER MODEL 7000 7001 3
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#df = pd.read_csv('rajahmundry_finall.csv',parse_dates=['Month'],index_col='Month')
#df = pd.read_csv('rajahmundry_finall.csv')
df = dataset
#df.index.freq = '3H'
train, test = df.iloc[:4,[0,1,3]], df.iloc[5:,[0,1,3]]
#train 2880
model = ExponentialSmoothing(train.Temperature, seasonal='add', seasonal_periods=4).fit()
#test.Date[8135]
pred = model.predict(start=test.num[5], end=test.num[6])
#test.loc[7001,'Date']
plt.figure(figsize=(64,8))
plt.scatter(train.num, train.Temperature, label='Train',s=1)
plt.scatter(test.num, test.Temperature, label='Test',s=1)
plt.scatter(test.num,pred,s=1)
plt.legend(loc='best')
plt.savefig('holt_winter2880scatter.png')
plt.show()


# In[104]:

rms = sqrt(mean_squared_error(test.Temperature, pred))
print(rms)


# SARIMA MODEL


