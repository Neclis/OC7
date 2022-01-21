# -*- coding: utf-8 -*-
from flask import Flask, jsonify, redirect, url_for, request, send_from_directory
from flask_restful import Resource, Api, reqparse 
import pickle
import pandas as pd
import os

app = Flask(__name__)
api = Api(app)

# load ML model & data
with open('model_opti.pickle', 'rb') as f:
    model = pickle.load(f)
data2 = pd.read_pickle("data2_sample.pickle")
main_features_pd = pd.read_csv("main_features_pd.csv", index_col="index")
X_train2_sc_pd_mean = pd.read_csv("X_train2_sc_pd_mean.csv", index_col="index")
sample = pd.read_csv("X_test2_sc_pd_sample.csv", index_col="index")


#### APP : Welcome page
@app.route("/")
def hello():
    hello = "Hello World! \n Available commands are : <br> /read/id with ID = [90265, 75598, 40776, 68707, 28645, 54948, 65586,  3629,  3963] </br> /enterid (get / post) </br> /enterdata (post)"
    return hello

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                          'favicon.ico',mimetype='image/vnd.microsoft.icon')

#### API : READ DATA from data2
@app.route('/read/<int:ide>', methods = ['GET'])
def get(ide : int):
    data_found=data2.loc[data2.index == ide].to_dict()
    return jsonify({"data_found": data_found})

#### API : predict data from test set id
@app.route('/enterid', methods = ['POST', 'GET'])
def enterid():
   if request.method == 'POST':
      ide = request.form['ide']
      print("enterid post", ide)
      return redirect(url_for('proba',ide = ide))
   else:
      ide = request.args.get('ide')
      print("enterid get", ide)
      return redirect(url_for('proba',ide = ide))

@app.route('/proba/<ide>')
def proba(ide):
    ide = int(ide)
    pred_0 = model.predict_proba(sample.loc[sample.index ==ide])[0][0]
    pred_1 = model.predict_proba(sample.loc[sample.index ==ide])[0][1]
    dict_pred = {"proba_0" : pred_0 , "proba_1" : pred_1}
    return jsonify(dict_pred)

#### API : predict data from data input
@app.route('/enterdata', methods = ['POST'])
def enterdata():

    for var in main_features_pd.index:
        X_train2_sc_pd_mean[var] = request.form[var]

    pred_0 = model.predict_proba([X_train2_sc_pd_mean])[0][0]
    pred_1 = model.predict_proba([X_train2_sc_pd_mean])[0][1]
    dict_pred = {"proba_0" : pred_0 , "proba_1" : pred_1}

    return jsonify(dict_pred)

if __name__ == "__main__":
    app.run()

