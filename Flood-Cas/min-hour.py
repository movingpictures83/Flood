#!/usr/bin/env python
# coding: utf-8

# In[143]:


import pandas as pd
import numpy as np


# In[144]:


# df = pd.read_csv('./data/zeda/Merged-tiny.csv', index_col=0)
df = pd.read_csv('./data/zeda/Merged-tiny.csv')
df.head()


# In[145]:


print(df.shape)


# In[146]:


datetime_series = pd.to_datetime(df['Time'])
df['Timestamp'] = pd.DatetimeIndex(datetime_series.values)
df = df.set_index('Timestamp')
df


# In[147]:


df.index.name


# In[148]:


# df.resample('H', on='Timestamp').mean()
# df.groupby(Timestamp.hour).mean()
# df
hourly = df.resample('H').mean()
hourly


# In[149]:


# df = pd.read_csv('./data/zeda/Merged-tiny.csv', index_col=0)
merged = pd.read_csv('./data/zeda/Merged.csv')
merged.head()


# In[150]:


merged.shape


# In[151]:


datetime_series = pd.to_datetime(merged['Time'])
merged['Timestamp'] = pd.DatetimeIndex(datetime_series.values)
merged = merged.set_index('Timestamp')
merged


# In[152]:


merged.index.name


# In[154]:


Merged_hourly = merged.resample('H').mean()
Merged_hourly


# In[155]:


Merged_hourly.to_csv('./data/zeda/Merged_hourly.csv')


# In[ ]:





# In[ ]:





# In[12]:




