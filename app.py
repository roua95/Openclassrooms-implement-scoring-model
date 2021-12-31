import uvicorn
import lightgbm as lgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

class Client(BaseModel):
    AMT_ANNUITY: float
    EXT_SOURCES_MAX: float

def my_score(y_pred, y_true):
    tp = np.sum(y_pred * y_true, axis=0)
    fp = np.sum(y_pred * (1 - y_true), axis=0)
    fn = np.sum((1 - y_pred) * y_true, axis=0)
    cost = 15 * fn + 5 * fp  # / (2*tp + fn + fp + 1e-16)
    return cost


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# with open("./model.dat", "rb") as f:

#    model = pickle.load(f)
model = lgb.Booster(model_file='model.txt')


@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}


@app.get('/client/{id}')
def get_client_id(id: int):
    return {'message': f'Hello! @{id}'}


@app.get('/prediction')
def get_model_decision(data: Client):
    received = data.dict()
    AMT_ANNUITY = received['AMT_ANNUITY']
    EXT_SOURCES_MAX = received['EXT_SOURCES_MAX']
    pred_name = model.predict([[AMT_ANNUITY, EXT_SOURCES_MAX]]).tolist()[0]
    return {'prediction': pred_name}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)