# To load locally : streamlit run dashboard/dashboard.py

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle
import requests
from imblearn.pipeline import Pipeline as imbpipeline
import shap
import matplotlib.pyplot as plt
from PIL import Image

# force display to wide
st.set_page_config(layout="wide")

# load datas
df_data = pd.read_csv('datas/X_sample.csv', index_col='SK_ID_CURR')
df_data_preproc = pd.read_csv('datas/X_sample_preproc.csv', index_col='SK_ID_CURR')

features = list(df_data.columns)

# load model
pickle_in_cl = open('models/model_classifier.pkl','rb')
classifier=pickle.load(pickle_in_cl)

# load explainer
pickle_in_exp = open('explainer/explainer.pkl','rb')
explainer=pickle.load(pickle_in_exp)

# load logo
logo = Image.open('datas./logo.png')



# plot the force plot in streamlit in API
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

shap_values = explainer(df_data_preproc)

# display title
st.markdown("<h1 style='text-align: center; color: black;'>Prêt à dépenser customer dashboard<p></p></h1>", unsafe_allow_html=True)

#diplay logo
st.sidebar.image(logo)


# selectbox with customersID
customerid = st.sidebar.selectbox(
    'Please select a CustomerID',
     df_data.index.tolist(),0)

customer_row = df_data.index.get_loc(customerid)

# load data of the customer selected
customer_data = df_data.loc[[customerid]]
customer_data_preproc = df_data_preproc.loc[[customerid]]



# create differents checkbox to display different types of datas
cbx_csdata = st.sidebar.checkbox('Customer datas')
cbx_proba = st.sidebar.checkbox('Customer prediction')
cbx_compare = st.sidebar.checkbox('Customer compare')

# display customerID
st.write('Customer selected :', customerid)

# display datas linked to checkbox cbx_csdata
if cbx_csdata:
    
    st.write("""
    # Customer datas :
    """)
    
    feature_selection = st.multiselect(
        'Select features to display', features, features[:10])
    
    st.dataframe(customer_data[feature_selection])

# display datas linked to checkbox cbx_proba
if cbx_proba:
    
    st.write("""
    # Customer prediction :
    """)
    
    st.write('Predict : ', df_data.loc[customerid]['prediction'])
    st.write('Probability of default : ', round(df_data.loc[customerid]['probability']*100,2), '%')
           
    
    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    st_shap(shap.force_plot(shap_values[customer_row]))

    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values)
    st.pyplot(fig)
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.plots.waterfall(shap_values[customer_row])
    st.pyplot(fig)