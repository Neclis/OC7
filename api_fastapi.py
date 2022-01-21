# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import uvicorn ##ASGI
from fastapi import FastAPI
from fastapi.responses import FileResponse 
from api_fastapi_datamodel import datamodel

app = FastAPI(title='NM FastAPI')
favicon_path = 'static/favicon.ico'

# load ML model & data
with open('model_opti.pickle', 'rb') as f:
    model = pickle.load(f)
data2 = pd.read_pickle("data2_sample.pickle")
main_features_pd = pd.read_csv("main_features_pd.csv", index_col="index")
X_train2_sc_pd_mean = pd.read_csv("X_train2_sc_pd_mean.csv", index_col="index")
sample = pd.read_csv("X_test2_sc_pd_sample.csv", index_col="index")

#### APP : Welcome page
@app.get('/')
def home():
    hello = "Hello World! /n Available commands are : /n /read/id with ID = [90265, 75598, 40776, 68707, 28645, 54948, 65586,  3629,  3963] /n /enterid (get / post) /n /enterdata (post) "
    return {"data": hello}

@app.get('/favicon.ico')
def favicon():
    return FileResponse(favicon_path)

#### API : READ DATA from data2
@app.get('/read/{ide}')
def get(ide : int):
    data_found=data2.loc[data2.index == ide].to_dict()
    return {"data_found": data_found}


#### API : predict data from test set id
def enterid(ide:int):
    pred_0 = model.predict_proba(sample.loc[sample.index ==ide])[0][0]
    pred_1 = model.predict_proba(sample.loc[sample.index ==ide])[0][1]
    dict_pred = {"proba_0" : pred_0 , "proba_1" : pred_1}
    return dict_pred

@app.get('/enterid/{ide}')
def enterid_get(ide:int):
    return enterid(ide)


#### API : predict data from test set id
@app.post('/enterid/{ide}')
def enterid_get(ide:int):
    return enterid(ide)


#### API : predict data from data input
@app.post('/enterdata')
def enterdata(data:datamodel):
    data = data.dict()
    # data = np.zeros(len(main_features_pd.index))
    # for i, var in enumerate(main_features_pd.index):
    #     data[i] = request.form[var]
    #     print(var, data[i])

    for var in main_features_pd.index:
        X_train2_sc_pd_mean[var] = data[var]

    pred_0 = model.predict_proba([X_train2_sc_pd_mean])[0][0]
    pred_1 = model.predict_proba([X_train2_sc_pd_mean])[0][1]
    dict_pred = {"proba_0" : pred_0 , "proba_1" : pred_1}
    return dict_pred


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn api_fastapi:app --reload