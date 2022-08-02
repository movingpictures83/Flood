from TestFloodCas_modules import *

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
    model = create_LSTM_model('Sequential', 75, train_X, train_y)
    train_LSTM(model, 0.00001, 2, train_X, train_y, test_X, test_y)
    
    #Predict stage
    inv_y, inv_yhat = predict(test_X, test_y, model, scaler)

    print("Variables inv_y and inv_yhat contain the results of the prediction. \n")
    print("Program ended successfuly. \n")
if __name__ == "__main__":
    main()


