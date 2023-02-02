"""
This is a boilerplate pipeline 'process_data'
generated using Kedro 0.18.3
"""
import datetime
import pickle

import numpy as np
import optuna
import pandas as pd
import wandb
from wandb.keras import WandbCallback

import tensorflow as tf
from keras import layers
from keras.losses import mean_squared_error
from matplotlib import pyplot as plt
from pycaret.time_series import *
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


def pycaret_research(df: pd.DataFrame):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index(keys='Timestamp').fillna(axis=0, method='ffill')
    s = setup(df, fh=3, fold=5, target='Close', numeric_imputation_target='mean', numeric_imputation_exogenous='mean',
              session_id=123)
    best = compare_models()
    plot_model(best, plot='forecast', data_kwargs={'fh': 24})

    return best


def get_model_with_params(dropout, input_shape, units):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(units=units, return_sequences=False,
                          input_shape=(input_shape, 1), dropout=dropout))
    model.add(layers.Dense(units=1))
    model.summary()
    model.compile(
        loss='mean_squared_error',
        optimizer='adam'
    )
    return model


def optuna_optimization(df: pd.DataFrame, user_n_epochs: int = 1):
    N_TRIALS = 3

    def objective(trial: optuna.Trial):
        EPOCHS = user_n_epochs
        BATCH_SIZE = 1024
        wandb.init(project="bitcoin-price-prediction", entity="asi_group2")
        input_shape = trial.suggest_int('input_shape', low=10, high=100)
        units = trial.suggest_int('units', low=10, high=100)
        dropout = trial.suggest_float('dropout', low=0.05, high=0.5)

        model = get_model_with_params(dropout, input_shape, units)
        wandb.config = {
            "learning_rate": model.optimizer.learning_rate,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE
        }
        x_train, y_train, x_test, y_test, scaler = prepare_for_learn(df, input_shape + 1)

        model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=False,
            validation_split=0.1, callbacks=[WandbCallback()]
        )
        mse = model.evaluate(x_test, y_test)
        # save each trial to a pickle file
        with open("pkls/trials/{}.pickle".format(trial.number), "w+b") as fout:
            pickle.dump(model, fout)
        return mse

    study = optuna.create_study(direction='minimize',
                                storage='sqlite:///trials.db')
    study.optimize(objective, n_trials=N_TRIALS)

    # load best model by trial number
    with open("pkls/trials/{}.pickle".format(study.best_trial.number), "rb") as fin:
        best_model = pickle.load(fin)

    # save best_model to separate dir
    with open("pkls/best_trial/{}.pickle".format(study.best_trial.number), "w+b") as fout:
        pickle.dump(best_model, fout)
    best_model.save('saved_models/btc_model')

    optuna_results = study.best_trial.params
    return optuna_results
