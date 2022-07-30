import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def Holt_Model(filename):
	dataset = pd.read_csv(filename)
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


# HOLT'S WINTER MODEL Original Values: 7000 7001 3
def Holt_Model_2(filename):

	df = pd.read_csv(filname)
	train, test = df.iloc[:4,[0,1,3]], df.iloc[5:,[0,1,3]]
	model = ExponentialSmoothing(train.Temperature, seasonal='add', seasonal_periods=4).fit()
	pred = model.predict(start=test.num[5], end=test.num[6])
	#test.loc[7001,'Date']
	plt.figure(figsize=(64,8))
	plt.scatter(train.num, train.Temperature, label='Train',s=1)
	plt.scatter(test.num, test.Temperature, label='Test',s=1)
	plt.scatter(test.num,pred,s=1)
	plt.legend(loc='best')
	plt.savefig('holt_winter2880scatter.png')
	plt.show()
	rms = sqrt(mean_squared_error(test.Temperature, pred))
	print(rms)

def main():
	Holt_Model('Dummy CSV - Sheet1.csv')
	Holt_Model_2('Dummy CSV - Sheet1.csv')


if __name__ == "__main__":
	main()
