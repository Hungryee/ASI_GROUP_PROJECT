"""
This is a boilerplate pipeline 'process_data'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            drop_unused,
            inputs='bitcoin_historical_data',
            outputs='with_dropped_columns',
            name='drop_unused_node'
        ),
        node(
            timestamp_to_datetime_indexed,
            inputs='with_dropped_columns',
            outputs='with_datetime',
            name='timestamp_to_datetime_node'
        ),
        node(
            convert_to_15min,
            inputs='with_datetime',
            outputs='with_15min_timeframe',
            name='convert_to_15min_node'
        ),
        node(optuna_optimization,
            inputs=['with_15min_timeframe'],
            outputs='optuna_best_model_results'
        )
    ])

#todo
#--dvc
#--kedro
#--optuna
#--ansible
#fastapi
#presentation

# add: several models
