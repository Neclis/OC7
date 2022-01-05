# -*- coding: utf-8 -*-
from flask import Flask, jsonify, redirect, url_for, request, send_from_directory
from flask_restful import Resource, Api, reqparse 
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
api = Api(app)

# load ML model & data
model=pickle.load(open('model_opti.pickle', 'rb'))
data = pd.read_csv('P7_raw/application_train.csv')
data2 = pd.read_pickle("data2.pickle")
with open('X_test2_sc.pickle', 'rb') as f:
    X_test2_sc = pickle.load(f)

main_features_pd = pd.read_pickle("main_features_pd.pickle")
X_train2_sc_pd_mean = pd.read_pickle("X_train2_sc_pd_mean.pickle")


#### APP : Welcome page
@app.route("/")
def hello():
    return "Hello World!"

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                          'favicon.ico',mimetype='image/vnd.microsoft.icon')

#### API : READ DATA from data2
class read(Resource):
    def __init__(self):
        # read csv file
        self.data = data
    
    # GET request on the url will hit this function
    def get(self,ID):
        # find data from csv based on user input
        data_found=self.data.loc[self.data.index == ID].to_dict()
        # return data found in csv
        return jsonify(data_found)
api.add_resource(read, '/read/<int:ID>')

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
    pred_0 = model.predict_proba(X_test2_sc[ide].reshape(1,-1))[0][0]
    pred_1 = model.predict_proba(X_test2_sc[ide].reshape(1,-1))[0][1]
    dict_pred = {"proba_0" : pred_0 , "proba_1" : pred_1}
    print("proba to dict done")
    return jsonify(dict_pred)

#### API : predict data from data input
@app.route('/enterdata', methods = ['POST'])
def enterdata():
    
    # data = np.zeros(len(main_features_pd.index))
    # for i, var in enumerate(main_features_pd.index):
    #     data[i] = request.form[var]
    #     print(var, data[i])

    for var in main_features_pd.index:
        X_train2_sc_pd_mean[var] = request.form[var]
        print(var, X_train2_sc_pd_mean[var])

    pred_0 = model.predict_proba([X_train2_sc_pd_mean])[0][0]
    pred_1 = model.predict_proba([X_train2_sc_pd_mean])[0][1]
    dict_pred = {"proba_0" : pred_0 , "proba_1" : pred_1}

    return jsonify(dict_pred)

if __name__ == "__main__":
    app.run()

    # host = os.getenv('IP','0.0.0.0')
    # port = int(os.getenv('PORT',5000))
    # app.secret_key = os.urandom(24)
    # app.run(host=host,port=port)