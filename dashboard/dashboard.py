# To load locally : streamlit run dashboard/dashboard.py

# import libraries
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle
import requests
import seaborn as sns
from imblearn.pipeline import Pipeline as imbpipeline
import shap
import matplotlib.pyplot as plt
from PIL import Image

# load datas
# data of the customers
df_data = pd.read_csv('datas/X_sample.csv', index_col='SK_ID_CURR')
# data of the customers preprocessed (imputed, normalized)
df_data_preproc = pd.read_csv('datas/X_sample_preproc.csv', index_col='SK_ID_CURR')
# mean and mode of the features
df_mean_mode = pd.read_csv('datas/X_sample_mean_mode.csv',index_col='prediction')

# load model
pickle_classifier = open('models/model_classifier.pkl','rb')
classifier=pickle.load(pickle_classifier)

# load explainer
pickle_explainer = open('explainer/explainer.pkl','rb')
explainer=pickle.load(pickle_explainer)

# load shap_values
pickle_shap_values = open('explainer/shap_values.pkl','rb')
shap_values=pickle.load(pickle_shap_values)

# load logo
logo = Image.open('datas./logo.png')

# thershold optimized for custom score
threshold_optimized = 0.320

# force display to wide
st.set_page_config(layout="wide")

# list of features
features = list(df_data.columns)

# function to display force plot as html
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# display title
st.markdown("<h1 style='text-align: center; color: black;'>Customer Dashboard<p></p></h1>", unsafe_allow_html=True)

#diplay logo
st.sidebar.image(logo)


# selectbox for customersID
customerid = st.sidebar.selectbox(
    'Please select a CustomerID',
     df_data.index.tolist(),0)

# row of the customer in data files
customer_row = df_data.index.get_loc(customerid)

# load data of the customer selected
customer_data = df_data.loc[[customerid]]
customer_data_preproc = df_data_preproc.loc[[customerid]]


# create differents checkbox to display different types of datas
cbx_proba = st.sidebar.checkbox('Customer prediction')
cbx_data = st.sidebar.checkbox('Customer data')
cbx_compare = st.sidebar.checkbox('Customer compare')



# display information how to use dashboard if no checkbox selected
if not cbx_data and not cbx_proba and not cbx_compare:

    st.markdown("<h5 style='text-align: left; color: black;'><u>How to use this dashboard :</u></h5>", unsafe_allow_html=True)
    st.markdown('Select a **CustomerID**')
    st.markdown('<br></br>', unsafe_allow_html=True)
    st.markdown('Information available with the different checkbox :')
    
    st.markdown("<ul><li><h6 style='text-align: left; color: black;'>Customer prediction :</li></h4><ul style='list-style-type:none;'><li>Display the score and the prediction for the customer selected and how to explain them.</li></ul></ul>", unsafe_allow_html=True)

    st.markdown("<ul><li><h6 style='text-align: left; color: black;'>Customer data :</li></h4><ul style='list-style-type:none;'><li>Display descriptive information of the customer selected.</li></ul></ul>", unsafe_allow_html=True)

    st.markdown("<ul><li><h6 style='text-align: left; color: black;'>Customer compare :</li></h4><ul style='list-style-type:none;'><li>Compare the customer selected with other customers.</li></ul></ul>", unsafe_allow_html=True)    
    

# display datas linked to checkbox cbx_csdata
if cbx_proba:
    st.markdown("<h2 style='text-align: left; color: black;'>Customer prediction :</h2>", unsafe_allow_html=True)

    st.markdown('CustomerID selected : **'+ str(customerid) + '**')
    
    if df_data.loc[customerid]['prediction'] == 1:
        customer_status = 'default of repayment 🔴'
    else:
        customer_status = 'non default of repayment 🟢'
    
    st.markdown('For this customer the prediction is : ' + customer_status + ' with a default probability of ' + str(round(df_data.loc[customerid]['probability']*100,2)) + '% (with threshold at : ' + str(threshold_optimized*100) + '%)')
    
    
    col1, col2= st.columns(2)
    
    with col1:
        st.markdown("<h3 style='text-align: center; color: black;'>Features of the customer</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        shap.plots.waterfall(shap_values[customer_row])
        st.pyplot(fig)
        st.markdown("<u>To interpret :</u>", unsafe_allow_html=True)
        st.markdown("<ul><li>f(x) is the output value, it's the prediction for the customer.</li></ul>", unsafe_allow_html=True)
        st.markdown("<ul><li>E[f(x)] is the base value, it's  is the mean prediction.</li></ul>", unsafe_allow_html=True)
        st.markdown("<ul><li>Red/blue: Features that push the prediction higher (to the right) are shown in red, and those pushing the prediction lower are in blue.</li></ul>", unsafe_allow_html=True) 
        
    with col2:
        st.markdown("<h3 style='text-align: center; color: black;'>Features importance</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        shap.summary_plot(shap_values)
        st.pyplot(fig)
        st.markdown("<u>To interpret :</u>", unsafe_allow_html=True)
        st.markdown("<ul><li>Feature importance: Variables are ranked in descending order.</li></ul>", unsafe_allow_html=True)
        st.markdown("<ul><li>Impact: The horizontal location shows whether the effect of that value is associated with a higher or lower prediction.</li></ul>", unsafe_allow_html=True)        
        st.markdown("<ul><li>Original value: Color shows whether that variable is high (in red) or low (in blue) for that observation.</li></ul>", unsafe_allow_html=True)   
        

# display datas linked to checkbox cbx_proba
if cbx_data:    
    
    feature_selection = st.multiselect(
        'Select features to display', features, features[:10])
    
    
    #st.dataframe(customer_data[feature_selection])
    test = customer_data[feature_selection].reset_index().transpose()
    st.dataframe(test.rename(columns={0: "Value"}).astype('string'))
    st.table(test.rename(columns={0: "Value"}).astype('string'))

    chaine = 'Prédiction : **' + str(customerid) +  '** avec **' + str(df_data.loc[customerid]['prediction']) + '%** de risque de défaut (classe réelle : '+ str(round(df_data.loc[customerid]['probability']*100,2)) + ')'
    


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
    
    col1, col2= st.columns([1,1])

    with col1:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        shap.summary_plot(shap_values)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        shap.plots.waterfall(shap_values[customer_row])
        st.pyplot(fig)
    
