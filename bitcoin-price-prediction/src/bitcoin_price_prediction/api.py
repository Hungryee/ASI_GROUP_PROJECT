from typing import Iterable

from fastapi import FastAPI, Depends
from kedro.pipeline import Pipeline
from bitcoin_price_prediction.schemas import Employee
from bitcoin_price_prediction.dependencies import create_variance_pipeline
from kedro.runner import SequentialRunner

app = FastAPI()



@app.get('/result')
def get_result(employees: list[Employee], pipeline: Pipeline = Depends(create_variance_pipeline)) -> dict[str,str]:
    print(pipeline)
    return {'result':'testtest'}