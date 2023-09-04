import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

WINDOW_SIZE = 10


def load_data(file_path):
    df = pd.read_json(file_path)
    df = df[['closing_price']].values
    return df


def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    x_train, y_train = [], []
    for i in range(WINDOW_SIZE, len(scaled_data)):
        x_train.append(scaled_data[i - WINDOW_SIZE:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    return x_train, y_train


def create_model(model_name, x_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(x_train.shape[1], 1)))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=2, batch_size=1, verbose=2)
    
    model.save(f'{model_name}.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json-path', required=True)
    parser.add_argument('-m', '--model-name', required=True)
    args = parser.parse_args()
    
    # print(args)

    df = load_data(args.json_path)
    x_train, y_train = preprocess_data(df)
    create_model(args.model_name, x_train, y_train)
