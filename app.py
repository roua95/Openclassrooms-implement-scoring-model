import uvicorn
import lightgbm as lgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd



#load dataframe
url='https://drive.google.com/file/d/1fK0EPuQys4fxwe50FBnZ175JvMQf4lrs/view?usp=sharing'
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
df = pd.read_csv(url,on_bad_lines='skip')


class Client(BaseModel):
    AMT_ANNUITY: float
    EXT_SOURCES_MAX: float


def my_score(y_pred, y_true):
    tp = np.sum(y_pred * y_true, axis=0)
    fp = np.sum(y_pred * (1 - y_true), axis=0)
    fn = np.sum((1 - y_pred) * y_true, axis=0)
    cost = 15 * fn + 5 * fp
    return cost


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

#with open("./model (4).pkl", "rb") as f:

#   model = joblib.load(f)
model = lgb.Booster(model_file='model1.txt')

print("model.feature_name()="+str(model.feature_name()))

@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}


@app.get('/predict/{idd}')
def get_model_decision_from_id(idd: int):
    requested_client = df[df.SK_ID_CURR == idd]
    pred_name = model.predict(requested_client,predict_disable_shape_check=True).tolist()[0]
    return {'prediction': pred_name}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)
