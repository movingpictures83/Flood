from lstm_flood_prediction_modules import *

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

if __name__ == "__main__":
    main()