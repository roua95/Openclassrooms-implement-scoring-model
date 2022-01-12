import uvicorn
import lightgbm as lgb
from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from IPython.display import HTML

#load dataframe
url='https://drive.google.com/file/d/1fK0EPuQys4fxwe50FBnZ175JvMQf4lrs/view?usp=sharing'
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
df = pd.read_csv(url,on_bad_lines='skip')
url_X_train='https://drive.google.com/file/d/1zBxY6cvnEvvqn7kb5OcB2t0NYhOiHsHw/view?usp=sharing'
url_X_train ='https://drive.google.com/uc?id=' + url_X_train.split('/')[-2]
X_train =pd.read_csv(url_X_train,on_bad_lines='skip')
#print(list(set(X_train.columns)-set(df.columns)))

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

#with open("./model (4).pkl", "rb") as f:

#   model = joblib.load(f)
model = lgb.Booster(model_file='model1.txt')

print("model.feature_name()="+str(model.feature_name()))

@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}


@app.get('/client/{idd}')
def get_client_id(idd: int):
    return {'message': f'Hello! @{idd}'}


@app.get('/prediction')
def get_model_decision(data: Client):
    received = data.dict()
    AMT_ANNUITY = received['AMT_ANNUITY']
    EXT_SOURCES_MAX = received['EXT_SOURCES_MAX']
    pred_name = model.predict([[AMT_ANNUITY, EXT_SOURCES_MAX]]).tolist()[0]
    return {'prediction': pred_name}


@app.get('/predict/{idd}')
def get_model_decision_from_id(idd: int):
    requested_client = df[df.SK_ID_CURR == idd]
    pred_name = model.predict(requested_client,predict_disable_shape_check=True).tolist()[0]
    return {'prediction': pred_name}

@app.get('/shap/global')
def shap_summary_plot_dot():

    try:
        #diff=list(set(X_train.drop(columns=['Unnamed: 0']).columns) - set(model.feature_name()))
        #print("len diff = "+str(len(diff)))
        #print("diff = "+str(diff))
        shap.initjs()
        shap_values = shap.TreeExplainer(model).shap_values(X_train.drop(columns=['Unnamed: 0']))
        output=shap.summary_plot(shap_values[0], X_train.drop(columns=['Unnamed: 0']),plot_type='dot',show=False,matplotlib=True)
        with open('./file.html', 'w') as f:
            f.write(output)
        plt.savefig("summary_plot.pdf")

        plt.close()
        output='ok'
    except Exception as e:
        print(e)
        output='Error occured!'
    return output


@app.get('/client/{idd}/explain')
def shap_plot(idd:int):
    shap.initjs()
    df = pd.read_csv(url, on_bad_lines='skip')
    print(df.shape)
    print(X_train.shape)
    print(df.columns)
    #df= df.set_index('SK_ID_CURR')
    model.params['objective'] = 'binary'
    explainerModel = shap.TreeExplainer(model)

    shap_values_Model = explainerModel.shap_values(df.drop(columns=['SK_ID_CURR', 'TARGET','Unnamed: 0']))

    id=df.index[df.SK_ID_CURR == idd]
    p = shap.force_plot(explainerModel.expected_value[0], shap_values_Model[0][id],
                        df.iloc[id].drop(columns=['SK_ID_CURR', 'TARGET','Unnamed: 0']), matplotlib=False)
    return (p)



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)
