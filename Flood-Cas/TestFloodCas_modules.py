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
# TO DO: - File checking, Parameter Parsing 
def read_and_slice_dataset(filename, indx):
    print("Starting execution of read_and_slice_dataset function... \n")
    print("INFO on the Pipeline: This function is usually executed after Data Collection and Data Validation but before Data Preprocessing. \n")
    dataset = read_csv(filename, index_col=indx)
    print("This is the file the program is reading from: ", filename +".\n")
    dataset.fillna(0, inplace=True)
    data = dataset[:578448]
    print("Finished execution of read_and_slice_dataset function. \n")
    return data

# Specify parameters for feature engineering
# TO DO: - Parameter Parsing
def set_lag_hours(num_hrs, num_feats, K):
    print("Starting execution of the set_lag_hours function... \n")
    print("INFO on the Pipeline: This function is usually executed after Data Collection and Data Validation but before Data Preprocessing. \n")
    n_hours = num_hrs
    n_features = num_feats
    K = K
    print("To do feature engineering, the number of hours is " + str(n_hours) + ", the number of features is " + str(n_features) + ", and the K variable is " + str(K) + ".\n") 
    print("Finished execution of set_lag_hours function. \n")
    return n_hours, n_features, K

# Preprocessing
# Can be use to create stages and non-stages
# TO DO: - Provide more information about the stage that is being returned
def create_set_for_staging(data, *col_names):
    print("Starting execution of the create_set_for_staging function... \n")
    print("INFO on the Pipeline: This function is usually executed during Data Preprocessing. \n")
    list = []
    for name in col_names:
        list += name
    print("Here is the list of column names that the program is using to create a set for staging, " + str(list) + ".\n")
    stage = data[list]
    print("The shape of the current stage is " + str(stage.shape) + ".\n")
    print("Finished execution of create_set_for_staging. \n")
    return stage

# TO DO: - Provide more informartion about the data argument and the returned dataset
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
    print("Starting execution of the stage_series_to_supervised algorithm... \n")
    print("INFO on the Pipeline: This function is usually executed during Model Training and Model Tuning. \n")
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
    print("After performing the stage_series_to_supervised algorithm, the program returns an aggregate set of shape: " + str(agg.shape) + ".\n")
    print("Finished execution of the stage_series_to_supervised algorithm. \n")
    return agg

# TO DO: - Provide more informartion about the data argument and the returned dataset
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
    print("Starting execution of the series_to_supervised algorithm... \n")
    print("INFO on the Pipeline: This function is usually executed during Model Training and Model Tuning. \n")
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
    print("After performing the series_to_supervised algorithm, the program frames a time series as a supervised learning dataset of shape: " + str(agg.shape))
    print("Finised execution of series_to_supervised algorithm. \n")
    return agg

# TO DO: - Allow for more customized concatenation, at the iloc level.
#        - Provide a helper function that assists with the math. This info will be used later for reshaping.
# Concatenation
def concat_preprocessed(stages_df):
    print("Starting execution of the concat_preprocessed function... \n")
    print("INFO on the Pipeline: This function is usually executed during Model Training and Model Tuning. \n")
    reset_stages_df = [reset_indx_dropT_inplaceT(stage_df) for stage_df in stages_df]
    print("length of stages_supervised dataset is ", len(reset_stages_df[0]), " and the shape is ", reset_stages_df[0].shape, ".\n")
    print("length of non_stages dataset is ", len(reset_stages_df[1]),  " and the shape is ", reset_stages_df[1].shape, ".\n")
    print("length of non_stages_supervised dataset is ", len(reset_stages_df[2]),  " and the shape is ", reset_stages_df[2].shape, ".\n") 
    #all_data = pd.concat((reset_stages_df[1].iloc[0:len(reset_stages_df[0]), -1], reset_stages_df[2].iloc[0:len(reset_stages_df[0]), 0:reset_stages_df[1].shape[1]], reset_stages_df[0].iloc[:, :-3]), axis=1)
    all_data = pd.concat(reset_stages_df, axis=1)
    print("The dataset which is the result of the concatenation of the sets above has shape ", all_data.shape, ".\n")
    print("Finished execution of concat_preprocessed function. \n")
    return all_data

# helper function that takes a DataFrame and resets its index, sets drop=True and inplace=True
def reset_indx_dropT_inplaceT(df):
    print("Starting execution of the reset_indx_dropT_inplace function... \n")
    print("INFO on the Pipeline: This function formats data. It is usually executed during Data Validation, Data Preprocessing, Model Training and Model Tuning. In this case it is executed during Model Training. \n")
    df.reset_index(drop=True, inplace=True)
    print("Finished execution of reset_indx_dropT_inplace function. \n")
    return df

# TO DO: - Provide customization on the number of train hours.
# Split data into train and test sets
def split_into_train_and_test(all_data):
    print("Starting execution of the split_into_train_and_test function... \n")
    print("INFO on the Pipeline: This function is usually executed during Model Training and Model Tuning. \n")
    all_data = all_data.values
    n_train_hours = int(len(all_data) * 0.8)
    print("The number of train hours is ", n_train_hours, ".\n")
    train = all_data[:n_train_hours, 1:]
    print("Created train set. \n")
    test = all_data[n_train_hours:, 1:]
    print("Created test set. \n")
    print("Finished execution of split_into_train_and_test function. \n")
    return train, test

# Split train and test sets into input and outputs
def split_into_input_and_output(n_hours, n_features, train_set, test_set):
    print("Starting execution of the split_into_input_and_output function... \n")
    print("INFO on the Pipeline: This function is usually executed during Model Training and Model Tuning. \n")
    n_obs =n_hours * n_features
    train_X, train_y = train_set[:, :n_obs], train_set[:, -5:]
    print("Created X and y sets of the train set. \n")
    test_X, test_y = test_set[:, :n_obs], test_set[:, -5:]
    print("Created X and y sets of the test set. \n")
    print("The shape of the train X set is ", train_X.shape, ".\n", "The shape of the train y set is ", train_y.shape, ".\n", "The shape of the test X set is ", test_X.shape, ".\n", "The shape of the test y set is ", test_y.shape, ".\n")
    print("Finished execution of split_into_input_and_output function. \n")
    return train_X, train_y, test_X, test_y


# Normalize fetures using fit_transform of MinMaxScaler.
# Takes 4 sets
# TO DO: - Provide opportunity to customize scaler
def normalize_features(*sets):
    print("Starting execution of normalize_features function... \n")
    print("INFO on the Pipeline: This function is usually executed during Model Training and Model Tuning. \n")
    scaler = MinMaxScaler(feature_range=(0,1))
    print("Created MinMaxScaler with feature range (0,1). \n")
    normalized_sets = np.array(scaler.fit(set).transform(set) for set in sets)
    print("Normalized sets. \n")
    #normalized_sets = np.array(scaler.fit_transform(set) for set in sets)
    print("Finished execution of normalize_features function. \n")
    return normalized_sets, scaler

# Reshape X sets input to be 3D [samples, timesteps, features]
def reshape_X_sets(train_X, test_X, n_hours, n_features):
    print("Starting execution of the reshape_X_sets function... \n")
    print("INFO on the Pipeline: This function is usually executed during Model Training and Model Tuning. \n")
    print("Current shape of the train X set is ", train_X.shape[0], ".\n")
    print("Current shape of the test X set is ", test_X.shape[0], ".\n")
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print("Reshaped train X set. New shape is ", train_X.shape, "Reshaped test X set. New shape is ", test_X.shape, ".\n")
    print("Finished execution of reshape_X_sets function. \n")
    return train_X, test_X


# Create LSTM model as Sequential with units, Dropout, and Dense
# TO DO: - Allow for customization on the number of Dropout and Dense
def create_LSTM_model(type, units, train_X, train_y):
    print("Starting execution of create_LSTM_model function... \n")
    print("INFO on the Pipeline: This function is usually executed during Model Training and Model Tuning. \n")
    if(type == 'Sequential'):
        model = Sequential()
        print("Created model of type Sequential. \n")
        model.add(LSTM(units, input_shape=(train_X.shape[1], train_X.shape[2])))
        print("Added LSTM to model with ", units, " units. \n")
        model.add(Dropout(0.2))
        print("Added Dropout and passed 0.2 to it. \n")
        model.add(Dense(train_y.shape[1]))
        print("Added Dense and passed the shape at index 1 of train y set. \n")
        print("Finished execution of create_LSTM_model function \n")
        return model
    else:
        pass

# Create GRU model as Sequential with units, Dropout, and Dense
# TO DO: - Allow for customization on the number of Dropout and Dense
def create_GRU_model(type, units, train_X, train_y):
    print("Starting execution of the create_GRU_model function... \n")
    print("INFO on the Pipeline: This function is usually executed during Model Training and Model Tuning. \n")
    if(type == 'Sequential'):
        model = Sequential()
        print("Created model of type Sequential. \n")
    else:
        pass
    model.add(GRU(units, input_shape=(train_X.shape[1], train_X.shape[2])))
    print("Added GRU to the model with ", units, " units. \n")
    model.add(Dropout(0.2))
    print("Added Dropout and passed 0.2 to it. \n")
    model.add(Dense(train_y.shape[1]))
    print("Added Dense and passed the shape at index 1 of train y set. \    n")
    model.summary()
    print("Finished execution of create_GRU_model function. \n")
    return model

# Create RNN model as Sequential with Flatten, Dense, activation_units, regression_units
# TO DO: - Allow for customization on the number of Flatten, Dropout and Dense
def create_RNN_model(type, activation_units, regression_units, train_X):
    print("Starting execution of the create_RNN_model function... \n")
    print("INFO on the Pipeline: This function is usually executed during Model Training and Model Tuning. \n")
    if(type == 'Sequential'):
        model = Sequential()
        print("Created model of type Sequential. \n")
    else:
        pass
    model.add(layers.Flatten(input_shape=(train_X.shape[1], train_X.shape[2])))
    print("Added Flatten to the model.")
    model.add(layers.Dense(activation_units, activation='relu'))
    print("Added Dense and passed ", activation_units, " activation units to it. \n")
    model.add(layers.Dense(activation_units, activation='relu'))
    print("Added another Dense and passed ", activation_units, " activation units to it. \n")
    model.add(layers.Dense(regression_units))
    print("Added another Dense and passed ", regression_units, " regression units to it. \n")
    return model

# Training
# Train function with variable Epochs, variable learning rate, compile method, Adam optimizer, loss as Mean Squared Error
# TO DO: - Allow for customization of the optimizer
def train_LSTM(model, lr, epochs, train_X, train_y, test_X, test_y):
    print("Starting execution of the train_LSTM function... \n")
    print("INFO on the Pipeline: This function is usually executed during Model Training and Model Tuning. \n")
    model.compile(optimizer=Adam(learning_rate=lr, decay=lr/epochs), loss='mse', metrics=['mae'])
    history =  model.fit(train_X, train_y,
    batch_size=256,
    epochs=epochs,
    validation_data=(test_X, test_y),
    verbose=2,
    shuffle=False)
    print("Calling model to compile using an Adam optimizer, a learning rate of ", lr, " and ", epochs, " epochs. \n")
    print("Finished execution of train_LSTM function. \n")

# Training GRU
# TO DO: - Allow for customization of the optimizer
def train_GRU(model, lr, epochs, train_X, train_y, test_X, test_y):
    print("Starting execution of the train_GRU function... \n")
    print("INFO on the Pipeline: This function is usually executed during Model Training and Model Tuning. \n")
    model.compile(optimizer=Adam(learning_rate=lr, decay=lr / epochs), loss='mse', metrics=['mae'])
    history = model.fit(train_X, train_y, batch_size=256, epochs=epochs, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    print("Calling model to compile using an Adam optimizer, a learning rate of ", lr, " and ", epochs, " epochs. \n")
    print("Finished execution of train_GRU. \n")

# Training RNN
# TO DO: - Allow for customization of the optimizer
def train_RNN(model, lr, epochs, train_X, train_y, test_X, test_y):
    print("Starting execution of the train_RNN function... \n")
    print("INFO on the Pipeline: This function is usually executed during Model Training and Model Tuning. \n")
    model.compile(optimizer=Adam(learning_rate=lr, decay=lr / epochs), loss='mse', metrics=['mae'])
    history = model.fit(train_X, train_y, batch_size=256, epochs=epochs, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    print("Calling model to compile using an Adam optimizer, a learning rate of ", lr, " and ", epochs, " epochs. \n")
    print("Finished execution of train_RNN function. \n")

# Predict
# Use model to make prediction based on given test sets
def predict(test_X, test_y, model, scaler):
    print("Starting execution of the predict function... \n")
    print("INFO on the Pipeline: This function is usually executed during Model Training and Model Tuning. \n")
    yhat = model.predict(test_X)
    #nsamples, nx, ny = test_X.shape
    #test_X = test_X.reshape((nsamples, nx*ny))
    obj = scaler.fit(yhat)
    inv_yhat = scaler.inverse_transform(yhat)
    obj = scaler.fit(test_y)
    inv_y = scaler.inverse_transform(test_y)
    inv_yhat = pd.DataFrame(inv_yhat)
    inv_y = pd.DataFrame(inv_y)
    print("Finished execution of the predict function. \n")
    return inv_y, inv_yhat


def main():
    print("Starting main function... \n")
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
    #model = create_RNN_model('Sequential', 8, 5, train_X)
    #train_RNN(model, 0.00001, 2, train_X, train_y, test_X, test_y)


    # Predict stage
    inv_y, inv_yhat = predict(test_X, test_y, model, scaler)
    print(inv_y)
    print("Variables inv_y and inv_yhat contain the results of the prediction. \n")
    print("Program ended successfuly. \n")
if __name__ == "__main__":
    main()


