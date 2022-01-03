import pandas as pd
import streamlit as st
import numpy as np
import requests

main_features = ['EXT_SOURCE_1',
                'EXT_SOURCE_2',
                'EXT_SOURCE_3',
                'CODE_GENDER',
                'NAME_FAMILY_STATUS',
                'AMT_REQ_CREDIT_BUREAU_TOTAL',
                'BIRTH_EMPLOYED_INTERVEL',
                'AMT_INCOME_TOTAL',
                'AMT_GOODS_PRICE',
                'AMT_CREDIT_SUM_DEBT'
                ]

sample = pd.read_pickle("X_train2_sc_pd_sample.pickle")
main_features_pd = pd.read_pickle("main_features_pd.pickle")
X_train2_sc_pd_mean = pd.read_pickle("X_train2_sc_pd_mean.pickle")

class App_customer_data:
    def app():
        # PAGE CUSTOMER DATA
        st.header('Customer Data')
        
        client_ID = st.selectbox("Client ID", sample.index)
        
        st.write('Voici les données pour le client {}'.format(client_ID))
        for col in sample[main_features].columns:
            st.write('Le {} est de : {:.2f}'.format(col, sample.loc[client_ID, col]))
        
        results = pd.DataFrame(sample.loc[client_ID, main_features].values, index=main_features, columns=["data"])
        results

        st.bar_chart(results)

def request_prediction(model_uri, data):
    data_json = {'ide': data}

    request = requests.post(model_uri, data=data_json)

    if request.status_code != 200:
        raise Exception("Request failed with status {}, {}".format(request.status_code, request.text))

    return request.json()

class App_prediction_from_id:
    def app():
        st.header('Solvability Prediction by id')
        
        FLASK_URL = "http://127.0.0.1:5000/enterid"

        # revenu_med = st.number_input('Revenu médian dans le secteur (en 10K de dollars)',
        #                             min_value=0., value=3.87, step=1.)

        client_ID = st.selectbox("Client ID", sample.index)
        predict_btn = st.button('Prédire')

        if predict_btn:
            # data = [[revenu_med, age_med, nb_piece_med, nb_chambre_moy,
                    # taille_pop, occupation_moy, latitude, longitude]]
            data = client_ID
            pred = None
            pred = request_prediction(FLASK_URL, data)
            
            for (key, value) in zip(pred.keys(), pred.values()):
                st.write(key,  value)

def request_prediction_data(model_uri, data):
    
    data_json = { main_features_pd.index[i] : data[i] for i in range(len(main_features_pd.index)) }
    
    # for (key, value) in zip(data_json.keys(), data_json.values()):
    #             st.write(key,  value)

    request = requests.post(model_uri, data=data_json)

    if request.status_code != 200:
        raise Exception("Request failed with status {}, {}".format(request.status_code, request.text))

    return request.json()

class App_prediction_from_data:
    def app():
        st.header('Solvability Prediction from data input')
        
        FLASK_URL = "http://127.0.0.1:5000/enterdata"

        st.dataframe(main_features_pd[["Min","Mean","Med","Max"]])

        data = np.zeros(len(main_features_pd.index))

        for i,var in enumerate(main_features_pd.index):
            data[i] = st.number_input(var, min_value=main_features_pd.loc[var,"Min"], 
                value=main_features_pd.loc[var,"Med"], 
                max_value=main_features_pd.loc[var,"Max"],
                step=0.01)

        predict_btn = st.button('Prédire')

        if predict_btn:
            pred = None
            pred = request_prediction_data(FLASK_URL, data)
                        
            for (key, value) in zip(pred.keys(), pred.values()):
                st.write(key,  value)
            

#### Main function of the App
def main():
    
    st.title('Loan Dashboard')

    PAGES = {
    "Customer Data": App_customer_data,
    "Prediction from cust": App_prediction_from_id,
    "Prediction from input" : App_prediction_from_data
            }
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page.app()


if __name__ == '__main__':
    main()