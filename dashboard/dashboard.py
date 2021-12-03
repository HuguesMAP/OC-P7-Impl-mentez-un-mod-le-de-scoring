# To load locally : streamlit run dashboard/dashboard.py

import streamlit as st
import pandas as pd
import pickle
import requests
from imblearn.pipeline import Pipeline as imbpipeline
import shap


df_data = pd.read_csv('X_test.csv', index_col='SK_ID_CURR')
df_data_preproc = pd.read_csv('X_test_preproc.csv', index_col='SK_ID_CURR')
pickle_in_cl = open('models/model_classifier.pkl','rb')
classifier=pickle.load(pickle_in_cl)

pickle_in_exp = open('explainer/explainer.pkl','rb')
explainer=pickle.load(pickle_in_exp)


shap_values = explainer(df_data_preproc)


option = st.sidebar.selectbox(
    'Please select a CustomerID'
    ,
     df_data.index.tolist(),0)

customer_data = df_data.loc[[option]]


cbx_csdata = st.sidebar.checkbox('Customer datas')

cbx_proba = st.sidebar.checkbox('Customer probability')

    
if cbx_csdata:
    st.dataframe(customer_data)

if cbx_proba:
    
    st.write("""
    # From streamlit directly
    """)
    predict_proba = classifier.predict_proba(customer_data.values.reshape(1,-1))
    predict_proba = predict_proba[:, 1][0]

    st.write('Probability of default : ', predict_proba)


    st.write("""
    # From FlaskAPI
    """)

    # from the flask api
    response = requests.get("https://ocp7implementezunmodele.herokuapp.com/"+ str(option))

    st.write(response.json())
    data_table1 = pd.DataFrame([response.json()])
    st.dataframe(data_table1)
    st.write('Probability of default : ', data_table1['Probability'][0])
    
    