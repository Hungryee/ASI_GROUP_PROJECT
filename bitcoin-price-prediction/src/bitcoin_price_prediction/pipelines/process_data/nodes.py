"""
This is a boilerplate pipeline 'process_data'
generated using Kedro 0.18.3
"""
import datetime

import numpy as np
import optuna
import pandas as pd

import tensorflow as tf
from keras import layers
from keras.losses import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def drop_unused(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price'])
    return df


def timestamp_to_datetime_indexed(df: pd.DataFrame) -> pd.DataFrame:
    df['Timestamp'] = df['Timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    df = df.set_index(keys='Timestamp')
    return df


def convert_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby(pd.Grouper(freq='15min')).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    df = df.reset_index()
    return df


def prepare_for_learn(df: pd.DataFrame, SEQ_LEN: int):
    scaler = MinMaxScaler()
    close_price = df['Close'].values.reshape(-1, 1)
    scaled_close = scaler.fit_transform(close_price)
    scaled_close = scaled_close[~np.isnan(scaled_close)]
    scaled_close = scaled_close.reshape(-1, 1)

    def to_sequences(data, seq_len):
        d = []
        for index in range(len(data) - seq_len):
            d.append(data[index: index + seq_len])
        return np.array(d)

    def preprocess(data_raw, seq_len, train_split):
        data = to_sequences(data_raw, seq_len)
        num_train = int(train_split * data.shape[0])
        X_train = data[:num_train, :-1, :]
        y_train = data[:num_train, -1, :]
        X_test = data[num_train:, :-1, :]
        y_test = data[num_train:, -1, :]
        return X_train, y_train, X_test, y_test

    X_train, y_train, X_test, y_test = \
        preprocess(scaled_close, SEQ_LEN, train_split=0.80)

    return X_train, y_train, X_test, y_test, scaler


def optuna_optimization(df: pd.DataFrame):
    def objective(trial: optuna.Trial):
        input_shape = trial.suggest_int('input_shape', low=10, high=100)
        units = trial.suggest_int('units', low=10, high=100)
        dropout = trial.suggest_float('dropout', low=0.05, high=0.5)

        model = tf.keras.Sequential()
        model.add(layers.LSTM(units=units, return_sequences=False,
                              input_shape=(input_shape, 1), dropout=dropout))
        # model.add(layers.LSTM(units=units, return_sequences=True,
        #                       dropout=0.2))
        # model.add(layers.LSTM(units=units//2, dropout=dropout))
        model.add(layers.Dense(units=1))
        model.summary()

        model.compile(
            loss='mean_squared_error',
            optimizer='adam'
        )

        x_train, y_train, x_test, y_test, scaler = prepare_for_learn(df, input_shape+1)
        BATCH_SIZE = 1024
        model.fit(
            x_train,
            y_train,
            epochs=10,
            batch_size=BATCH_SIZE,
            shuffle=False,
            validation_split=0.1
        )
        y_hat = model.predict(x_test)

        y_test_inverse = scaler.inverse_transform(y_test)
        y_hat_inverse = scaler.inverse_transform(y_hat)

        plt.plot(y_test_inverse, label="Actual Price", color='green')
        plt.plot(y_hat_inverse, label="Predicted Price", color='red')

        plt.title('Bitcoin price prediction')
        plt.xlabel('Time [days]')
        plt.ylabel('Price')
        plt.legend(loc='best')

        mse = model.evaluate(x_test, y_test)
        return mse

    study = optuna.create_study(direction='minimize',
                                storage='sqlite:///trials.db')
    study.optimize(objective, n_trials=1)

    optuna_results = study.best_trial
    return optuna_results.params
