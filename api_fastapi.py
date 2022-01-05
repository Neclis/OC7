# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import uvicorn ##ASGI
from fastapi import FastAPI
# from api_fastapi_datamodel import datamodel

app = FastAPI()

# load ML model & data
model=pickle.load(open('model_opti.pickle', 'rb'))
data = pd.read_csv('P7_raw/application_train.csv')
data2 = pd.read_pickle("data2.pickle")
X_test2_sc = pd.read_pickle("X_test2_sc.pickle")
X_train2_sc_pd_mean = pd.read_pickle("X_train2_sc_pd_mean.pickle")
main_features_pd = pd.read_pickle("main_features_pd.pickle")

#### APP : Welcome page
@app.get('/')
def home():
    return {"data": "Hello, World!"}

# @app.get('/favicon.ico')
# def favicon():
#     return send_from_directory(os.path.join(app.root_path, 'static'),
#                           'favicon.ico',mimetype='image/vnd.microsoft.icon')

#### API : READ DATA from data2
# GET request on the url will hit this function
@app.get('/read')
def get(ID : int):
    # find data from csv based on user input
    print(data.loc[data.index == ID].iloc[:,:3])
    # data_found=data.loc[data.index == ID].to_dict()
    
    # return data found in csv
    return {"data_found":"yes"}

#### API : predict data from test set id
@app.get('/enterid_get')
def enterid_get(ide:int):
    # print("enterid get", ide)
    pred_0 = model.predict_proba(X_test2_sc[ide].reshape(1,-1))[0][0]
    pred_1 = model.predict_proba(X_test2_sc[ide].reshape(1,-1))[0][1]
    dict_pred = {"proba_0" : pred_0 , "proba_1" : pred_1}
    # print("proba to dict done")
    return dict_pred

#### API : predict data from test set id
@app.post('/enterid_post')
def enterid_post(ide:int):
    # print("enterid post", ide)
    pred_0 = model.predict_proba(X_test2_sc[ide].reshape(1,-1))[0][0]
    pred_1 = model.predict_proba(X_test2_sc[ide].reshape(1,-1))[0][1]
    dict_pred = {"proba_0" : pred_0 , "proba_1" : pred_1}
    # print("proba to dict done")
    return dict_pred


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
        print(var, X_train2_sc_pd_mean[var])

    pred_0 = model.predict_proba([X_train2_sc_pd_mean])[0][0]
    pred_1 = model.predict_proba([X_train2_sc_pd_mean])[0][1]

    dict_pred = {"proba_0" : pred_0 , "proba_1" : pred_1}

    return dict_pred


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn api_fastapi:app --reload