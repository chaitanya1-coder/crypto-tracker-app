import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

WINDOW_SIZE = 10


def prep_data(lst, scalar):
    df = pd.DataFrame(lst, columns=['closing_price'])
    scaled_data = scalar.fit_transform(df)
    return np.array([scaled_data])


def predict_next(model_path, x_test, scalar):
    model = load_model(model_path)
    closing_price = model.predict(x_test)
    closing_price = scalar.inverse_transform(closing_price)
    return closing_price


def save_val(val):
    with open('./result', 'w') as _f:
        _f.write(str(val[0][0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-m', '--model-path', required=True)
    args = parser.parse_args()

    # print(args)
    input_lst = [float(num) for num in args.input.split(',')]
    scalar = MinMaxScaler(feature_range=(0, 1))
    input_data = prep_data(input_lst, scalar)
    output = predict_next(args.model_path, input_data, scalar)
    save_val(output)
