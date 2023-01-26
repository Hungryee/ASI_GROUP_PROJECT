from typing import Iterable

from kedro.pipeline import Pipeline

from bitcoin_price_prediction.pipelines.process_data import create_pipeline

def get_pipeline() -> Iterable[Pipeline]:
    yield create_pipeline()