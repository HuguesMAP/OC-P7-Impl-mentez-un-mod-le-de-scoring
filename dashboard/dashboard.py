# To load locally : streamlit run dashboard/dashboard.py

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle
import requests
from imblearn.pipeline import Pipeline as imbpipeline
import shap
import matplotlib.pyplot as plt


# load datas
df_data = pd.read_csv('datas/X_sample.csv', index_col='SK_ID_CURR')
df_data_preproc = pd.read_csv('datas/X_sample_preproc.csv', index_col='SK_ID_CURR')

# load model
pickle_in_cl = open('models/model_classifier.pkl','rb')
classifier=pickle.load(pickle_in_cl)

# load explainer
pickle_in_exp = open('explainer/explainer.pkl','rb')
explainer=pickle.load(pickle_in_exp)

# plot the force plot in streamlit in API
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

shap_values = explainer(df_data_preproc)



# selectbox with customersID
option = st.sidebar.selectbox(
    'Please select a CustomerID'
    ,
     df_data.index.tolist(),0)

# load data of the customer selected
customer_data = df_data.loc[[option]]


# create differents checkbox to display different types of datas
cbx_csdata = st.sidebar.checkbox('Customer datas')
cbx_proba = st.sidebar.checkbox('Customer probability')


# display datas linked to checkbox cbx_csdata
if cbx_csdata:
    st.dataframe(customer_data)

# display datas linked to checkbox cbx_proba
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
    
    
    customer_row = df_data.index.get_loc(option)
    
    
    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    st_shap(shap.force_plot(shap_values[customer_row]))


    
    st.pyplot(shap.force_plot(shap_values[customer_row],matplotlib=True,show=False
                     ,figsize=(15,3)), bbox_inches='tight',dpi=300,pad_inches=0)
    plt.clf()
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values)
    st.pyplot(fig)
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.plots.waterfall(shap_values[customer_row])
    st.pyplot(fig)