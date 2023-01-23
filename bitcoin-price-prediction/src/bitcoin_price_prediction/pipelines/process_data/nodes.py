"""
This is a boilerplate pipeline 'process_data'
generated using Kedro 0.18.3
"""
import datetime

import pandas as pd


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