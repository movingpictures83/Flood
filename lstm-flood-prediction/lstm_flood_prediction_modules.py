# import print_function
from __future__ import print_function
# os import
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
# data anlysis imports
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
# Tensorflow . keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
# sklearn import
from sklearn.preprocessing import MinMaxScaler

# fix random seed for reproducibility
def fix_random_seed(num):
    np.random.seed(num)

# Read from csv and extract dataset
def extract_dataset(filepath):
    dataframe = pandas.read_csv(filepath, usecols=[1], engine='python', skipfooter=3)
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    return dataset

# Normalize dataset
def normalize_dataset(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    return dataset,scaler

# split dataset into train and test set
def split_into_train_test_sets(dataset):
    train_size = int(len(dataset) * 0.67)
    test_size   = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    return train, test

# this function creates a sliding window of the data set
def create_dataset_with_sliding_window(dataset, sliding_window=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-sliding_window-1):
        a = dataset[i:(i+sliding_window), 0]
        dataX.append(a)
        dataY.append(dataset[i + sliding_window, 0])
    return np.array(dataX), np.array(dataY)

# Reshape X sets
def reshape_X_sets(trainX, testX):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX  = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    return trainX,testX

# Set up the LSTM with Sequential(), Dense, loss = mean_squared_error', and adam as optimizer
def create_LSTM_model_lstmfloodprediction(trainX, trainY, testX, testY):
    model = Sequential()
    model.add(LSTM(4, input_dim=10))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
   # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    model.fit(trainX, trainY, nb_epoch=50, batch_size=1, verbose=2) 
    return model

# Evaluate trainScore and testScore
def evaluate_trainScore_testScore(model, trainX, trainY, testX, testY,scaler):
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    trainScore = math.sqrt(trainScore)
    trainScore = scaler.inverse_transform(np.array([[trainScore]]))
    testScore = model.evaluate(testX, testY, verbose=0)
    testScore = math.sqrt(testScore)
    testScore = scaler.inverse_transform(np.array([[testScore]]))
    
# Predict
def make_prediction(model, trainX, testX):
    trainPredict = model.predict(trainX)
    testPredict  = model.predict(testX)

# Test the network on an unseen data
def read_createTest_clean_createDataset_reshape_predict_unseen(filepath, trainX, trainY, testX, testY):
    scaler = MinMaxScaler(feature_range=(0, 1))
    unseen = pandas.read_csv('dataset/flood_test.csv',sep=',')
    unseen_test = unseen['waterlevel'].values
    unseen_clean = []
    for i in unseen_test:
        unseen_clean.append([i])
    unseen_clean = np.asarray(unseen_clean).astype('float32')
    unseen_clean = scaler.fit_transform(unseen_clean)
    features, labels = create_dataset_with_sliding_window(unseen_clean, 10)
    features = np.reshape(features, (109186,1, 10)) 
    model = create_LSTM_model_lstmfloodprediction(trainX, trainY, testX, testY)
    unseen_results = model.predict(features)
    testScore = model.evaluate(features, labels, verbose=0)
    testScore = math.sqrt(testScore)
    testScore = scaler.inverse_transform(np.array([[testScore]]))

def main():
    fix_random_seed(10)
    dataset = extract_dataset('dataset/flood_train.csv')
    dataset,scaler = normalize_dataset(dataset)
    train, test = split_into_train_test_sets(dataset)
    trainX, trainY = create_dataset_with_sliding_window(train, 10)
    testX, testY   = create_dataset_with_sliding_window(test, 10)
    trainX, testX =  reshape_X_sets(trainX, testX)
    model = create_LSTM_model_lstmfloodprediction(trainX, trainY, testX, testY)
    evaluate_trainScore_testScore(model, trainX, trainY, testX, testY,scaler)
    make_prediction(model, trainX, testX)
    read_createTest_clean_createDataset_reshape_predict_unseen('dataset/flood_test.csv', trainX, trainY, testX, testY)