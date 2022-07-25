# Pandas imports
import pandas as pd
from pandas import concat
from pandas import read_csv

# helper files imports
from helper import series_to_supervised, stage_series_to_supervised

# Math imports
from math import sqrt

# Numpy imports
import numpy as np

# Sklearn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Tensorflow . Keras imports
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# matplotlib imports
import matplotlib.pyplot as plt


# Read a csv into a pandas DataFrame and set index column
# fill in place with na, slice dataset, and return the data
def read_and_slice_dataset(filename, indx):
    dataset = read_csv(filename, index_col=indx)
    dataset.fillna(0, inplace=True)
    data = dataset[:578448]
    return data

# Specify parameters for feature engineering
def set_lag_hours(num_hrs, num_feats, K):
    n_hours = num_hrs
    n_features = num_feats
    K = K
    return n_hours, n_features, K

# Preprocessing
# Can be use to create stages and non-stages
def create_set_for_staging(data, *col_names):
    list = []
    for name in col_names:
        list += name
    print(list)
    stage = data[list]
    print("stage.shape: ", stage.shape)
    return stage

# 
def stage_series_to_supervised(data, n_in, K, n_out, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in+K, K, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    print("stage_series_to_supervised agg.shape: ", agg.shape)
    return agg


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    print("series_to_supervised agg.shape: ", agg.shape)
    return agg


# Concatenation
def concat_preprocessed(stages_df):
    reset_stages_df = [reset_indx_dropT_inplaceT(stage_df) for stage_df in stages_df]
    print("length of stages_supervised is: ", len(reset_stages_df[0]), " and the shape is :", reset_stages_df[0].shape)
    print("length of non_stages is: ", len(reset_stages_df[1]),  " and     the shape is :", reset_stages_df[1].shape)
    print("length of non_stages_supervised: ", len(reset_stages_df[2]),  " and     the shape is :", reset_stages_df[2].shape) 
    #all_data = pd.concat((reset_stages_df[1].iloc[0:len(reset_stages_df[0]), -1], reset_stages_df[2].iloc[0:len(reset_stages_df[0]), 0:reset_stages_df[1].shape[1]], reset_stages_df[0].iloc[:, :-3]), axis=1)
    all_data = pd.concat(reset_stages_df, axis=1)
    print("all_data.shape:", all_data.shape)
    return all_data

# helper function that takes a DataFrame and resets its index, sets drop=True and inplace=True
def reset_indx_dropT_inplaceT(df):
    df.reset_index(drop=True, inplace=True)
    return df

# Split data into train and test sets
def split_into_train_and_test(all_data):
    all_data = all_data.values
    n_train_hours = int(len(all_data) * 0.8)
    print("n_train_hours:", n_train_hours)
    train = all_data[:n_train_hours, 1:]
    test = all_data[n_train_hours:, 1:]
    return train, test

# Split train and test sets into input and outputs
def split_into_input_and_output(n_hours, n_features, train_set, test_set):
    n_obs =n_hours * n_features
    train_X, train_y = train_set[:, :n_obs], train_set[:, -5:]
    test_X, test_y = test_set[:, :n_obs], test_set[:, -5:]
    print("train_X.shape, train_y.shape, test_X.shape, test_y.shape: \n", train_X.shape, train_y.shape, test_X.shape,
      test_y.shape)
    return train_X, train_y, test_X, test_y

# Normalize fetures using fit_transform of MinMaxScaler.
# Takes 4 sets
def normalize_features(*sets):
    scaler = MinMaxScaler(feature_range=(0,1))
    normalized_sets = np.array(scaler.fit(set).transform(set) for set in sets)
    #normalized_sets = np.array(scaler.fit_transform(set) for set in sets)
    return normalized_sets, scaler

# Reshape X sets input to be 3D [samples, timesteps, features]
def reshape_X_sets(train_X, test_X, n_hours, n_features):
    print(train_X.shape[0])
    print(test_X.shape[0])
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print("train_X.shape, test_X.shape: \n", train_X.shape, test_X.shape)
    return train_X, test_X

# Create LSTM model as Sequential with units, Dropout, and Dense
def create_LSTM_model(type, units, train_X, train_y):
    if(type == 'Sequential'):
        model = Sequential()
        model.add(LSTM(units, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(train_y.shape[1]))
        return model
    else:
        pass

# Create GRU model as Sequential with units, Dropout, and Dense
def create_GRU_model(type, units, train_X, train_y):
    if(type == 'Sequential'):
        model = Sequential()
    else:
        pass
    model.add(GRU(units, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(train_y.shape[1]))
    model.summary()
    return model

# Create RNN model as Sequential with Flatten, Dense, activation_units, regression_units
def create_RNN_model(type, activation_units, regression_units, train_X):
    if(type == 'Sequential'):
        model = Sequential()
    else:
        pass
    model.add(layers.Flatten(input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(layers.Dense(activation_units, activation='relu'))
    model.add(layers.Dense(activation_units, activation='relu'))
    model.add(layers.Dense(regression_units))
    return model

# Training
# Train function with variable Epochs, variable learning rate, compile method, Adam optimizer, loss as Mean Squared Error
def train_LSTM(model, lr, epochs, train_X, train_y, test_X, test_y):
    model.compile(optimizer=Adam(learning_rate=lr, decay=lr/epochs), loss='mse', metrics=['mae'])
    history =  model.fit(train_X, train_y,
    batch_size=256,
    epochs=epochs,
    validation_data=(test_X, test_y),
    verbose=2,
    shuffle=False)

# Training GRU
# 
def train_GRU(model, lr, epochs, train_X, train_y, test_X, test_y):
    model.compile(optimizer=Adam(learning_rate=lr, decay=lr / epochs), loss='mse', metrics=['mae'])
    history = model.fit(train_X, train_y, batch_size=256, epochs=epochs, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# Training RNN
#
def train_RNN(model, lr, epochs, train_X, train_y, test_X, test_y):
     model.compile(optimizer=Adam(learning_rate=lr, decay=lr / epochs), loss='mse', metrics=['mae'])
     history = model.fit(train_X, train_y, batch_size=256, epochs=epochs, validation_data=(test_X, test_y), verbose=2, shuffle=False)


# Predict
# Use model to make prediction based on given test sets
def predict(test_X, test_y, model, scaler):
    yhat = model.predict(test_X)
    #nsamples, nx, ny = test_X.shape
    #test_X = test_X.reshape((nsamples, nx*ny))
    obj = scaler.fit(yhat)
    inv_yhat = scaler.inverse_transform(yhat)
    obj = scaler.fit(test_y)
    inv_y = scaler.inverse_transform(test_y)
    inv_yhat = pd.DataFrame(inv_yhat)
    inv_y = pd.DataFrame(inv_y)
    return inv_y, inv_yhat

def main():
    scaler = MinMaxScaler(feature_range=(0,1))
    # Get data and create dataset
    data = read_and_slice_dataset('data/zeda/Merged.csv', 0)
    # Set params for the training
    n_hours, n_features, K = set_lag_hours(72, 16, 12)
    # Create stages and non-stages, and supervised
    stages = create_set_for_staging(data, ['WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26'])
    non_stages = create_set_for_staging(data, ['FLOW_S25A', 'GATE_S25A', 'FLOW_S25B', 'GATE_S25B', 'FLOW_S26', 'GATE_S26', 'PUMP_S26', 'mean'])
    stages_supervised = stage_series_to_supervised(stages, n_hours, K, 1)
    non_stages_supervised = series_to_supervised(non_stages, n_hours, 1)
    stages_df = [stages_supervised, non_stages, non_stages_supervised]
    # Concatenate stages dataframes
    all_data = concat_preprocessed(stages_df)
    # split dataset into train and test sets
    train, test = split_into_train_and_test(all_data)
    # Split train and test sets into X and y
    train_X, train_y, test_X, test_y = split_into_input_and_output(n_hours, n_features, train, test)
    sets = [train_X, train_y, test_X, test_y]
    # Normalize sets with MinMaxScaler
    normalized_sets, scaler = normalize_features(sets)
    # Reshape train_X and test_X
    train_X, test_X = reshape_X_sets(train_X, test_X, n_hours, n_features)
    
    # LSTM PIPELINE
    #model = create_LSTM_model('Sequential', 75, train_X, train_y)
    #train_LSTM(model, 0.00001, 2, train_X, train_y, test_X, test_y)
    
    
    # GRU PIPELINE
    #model = create_GRU_model('Sequential', 65, train_X, train_y)
    #train_GRU(model, 0.00001, 2, train_X, train_y, test_X, test_y) 
    
    # RNN PIPELINE
    model = create_RNN_model('Sequential', 8, 5, train_X)
    train_RNN(model, 0.00001, 2, train_X, train_y, test_X, test_y)


    # Predict stage
    inv_y, inv_yhat = predict(test_X, test_y, model, scaler)

if __name__ == "__main__":
    main()


