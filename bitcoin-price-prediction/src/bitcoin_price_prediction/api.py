import asyncio
from typing import Iterable

import optuna
import starlette.websockets
from fastapi import FastAPI, Depends, Request
from kedro.config import ConfigLoader
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from starlette.responses import HTMLResponse
from websocket import WebSocket
from fastapi.templating import Jinja2Templates
from bitcoin_price_prediction.dependencies import create_pipeline
from kedro.runner import SequentialRunner

app = FastAPI()
import logging

optuna.logging.enable_default_handler()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
fh = logging.FileHandler(filename='src/bitcoin_price_prediction/fastapi_logs/logfile.log')
formatter = logging.Formatter(
    "%(asctime)s - %(module)s - %(funcName)s - line:%(lineno)d - %(levelname)s - %(message)s"
)

ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)  # Exporting logs to the screen
logger.addHandler(fh)

templates = Jinja2Templates(directory="src/bitcoin_price_prediction/templates")


async def log_reader(n=5):
    log_lines = []
    with open(f"src/bitcoin_price_prediction/fastapi_logs/logfile.log", "r") as file:
        for line in file.readlines()[-n:]:
            if line.__contains__("ERROR"):
                log_lines.append(f'<span class="text-red-400">{line}</span><br/>')
            elif line.__contains__("WARNING"):
                log_lines.append(f'<span class="text-orange-300">{line}</span><br/>')
            else:
                log_lines.append(f"{line}<br/>")
        return log_lines


@app.websocket("/ws/log")
async def websocket_endpoint_log(websocket: starlette.websockets.WebSocket):
    await websocket.accept()

    try:
        while True:
            await asyncio.sleep(1)
            logs = await log_reader(30)
            await websocket.send_text(logs)
    except Exception as e:
        print(e)
    finally:
        await websocket.close()

"""
# todo2 
--serialize model into pkl
--w&b weights and biases? register account...
--restapi (== input test prediction data)
ansible (maybe) - missing ssh server creation and key generation

"""
@app.post('/run_pipeline')
def get_result(n_epochs: int):
    conf_loader = ConfigLoader("conf")
    conf_catalog = conf_loader.get("catalog.yml")

    pipeline = create_pipeline()
    logging.info('Setting up catalog')
    catalog = DataCatalog.from_config(conf_catalog)
    catalog.add_feed_dict({
        'user_n_epochs': n_epochs
    }, replace=True)
    runner = SequentialRunner()
    runner.run(pipeline, catalog)

    return catalog.load('optuna_best_model_results')


@app.get('/logs', response_class=HTMLResponse)
def logs(request: Request):
    return templates.TemplateResponse("logs.html", {"request": request, "context": {}})