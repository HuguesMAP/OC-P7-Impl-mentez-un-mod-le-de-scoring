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
import plotly.graph_objects as go

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

######################################################################################################################################## 

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


######################################################################################################################################## 


# display information how to use dashboard if no checkbox selected
if not cbx_data and not cbx_proba and not cbx_compare:

    st.markdown("<h5 style='text-align: left; color: black;'><u>How to use this dashboard :</u></h5>", unsafe_allow_html=True)
    st.markdown('Select a **CustomerID**')
    st.markdown('<br></br>', unsafe_allow_html=True)
    st.markdown('Information available with the different checkbox :')
    
    st.markdown("<ul><li><h6 style='text-align: left; color: black;'>Customer prediction :</li></h4><ul style='list-style-type:none;'><li>Display the score and the prediction for the customer selected and how to explain them.</li></ul></ul>", unsafe_allow_html=True)

    st.markdown("<ul><li><h6 style='text-align: left; color: black;'>Customer data :</li></h4><ul style='list-style-type:none;'><li>Display descriptive information of the customer selected.</li></ul></ul>", unsafe_allow_html=True)

    st.markdown("<ul><li><h6 style='text-align: left; color: black;'>Customer compare :</li></h4><ul style='list-style-type:none;'><li>Compare the customer selected with other customers.</li></ul></ul>", unsafe_allow_html=True)    

    
######################################################################################################################################## 


# display datas linked to checkbox cbx_csdata
if cbx_proba:
    st.markdown("<h2 style='text-align: left; color: black;'>Customer prediction :</h2>", unsafe_allow_html=True)

    st.markdown('CustomerID selected : **'+ str(customerid) + '**')  
    
    # use plotly to display a gauge for probability of the customer
    fig = go.Figure(go.Indicator(
         mode = "gauge+number",
            value = round(df_data.loc[customerid]['probability']*100,2),
            title = {'text': "Default probability (%)"},

            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {'axis': {'range': [None, 100]},'bar': {'color': "blue"},
                     'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold_optimized*100}}))
    # define size of the display
    fig.update_layout(
            autosize=False,
            width=150,
            height=150,
            margin=dict(
                l=0,
                r=0,
                b=5,
                t=40,
                pad=5))

    st.plotly_chart(fig, use_container_width=True)
    
    if df_data.loc[customerid]['prediction'] == 1:
        customer_status = 'default of repayment 🔴'
    else:
        customer_status = 'non default of repayment 🟢'
    
    # display probability and prediction for customer selected
    st.markdown('For this customer the prediction is : ' + customer_status + ' with a default probability at ' + str(round(df_data.loc[customerid]['probability']*100,2)) + '% (with threshold at : ' + str(threshold_optimized*100) + '%)')
    
    # display graph from shap for customer selected
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

        
########################################################################################################################################     


# display datas linked to checkbox cbx_proba
if cbx_data:    
    st.markdown("<h2 style='text-align: left; color: black;'>Customer data :</h2>", unsafe_allow_html=True)
    
    # use a multiselection to select features to display
    feature_selection = st.multiselect(
        'Select features to display', features, features[:10])
    
    # transpose dataframe in vertical for a better view
    customer_data_transposed = customer_data[feature_selection].reset_index().transpose()
    # display datas of the customer
    st.table(customer_data_transposed.round(2).rename(columns={0: "Value"}).astype('string'))
    
    # display number of missing values
    nb_missing_values = sum(customer_data.isnull().values.any(axis=0))
    percentage_nb_missing_values =  nb_missing_values/customer_data.shape[1]
    
    # use plotly to display a gauge for Missing values
    fig = go.Figure(go.Indicator(
         mode = "gauge+number",
            value = round(percentage_nb_missing_values*100,2),
            title = {'text': "Missing values (%)"},

            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {'axis': {'range': [None, 100]},'bar': {'color': "blue"}}))
    # define size of the display
    fig.update_layout(
            autosize=False,
            width=150,
            height=150,
            margin=dict(
                l=0,
                r=0,
                b=5,
                t=40,
                pad=5))

    st.plotly_chart(fig, use_container_width=True)  
    
    st.markdown("There's : " + str(nb_missing_values) + " missing values (" + str(round(percentage_nb_missing_values*100,2)) + "%).")
    
    # if there's missing values propose to display them with a checkbox
    if nb_missing_values > 0:
        cbx_missing_val = st.checkbox('Display list of missing value')
        
    if cbx_missing_val:    
        st.table(pd.DataFrame(customer_data.columns[customer_data.isnull().any()].to_list()).rename(columns={0: "Feature"}))

        
########################################################################################################################################       
        
# display datas linked to checkbox cbx_compare
if cbx_compare:
    st.markdown("<h2 style='text-align: left; color: black;'>Customer compare :</h2>", unsafe_allow_html=True)
    st.markdown("Compare the data of the customer with the data of the customers predicted with a default of repayment and with customers predicted with a non default of repayment")
    # create a data frame with datas of customer and data for customer with default and customer non default
    df_compare = pd.concat([customer_data, df_mean_mode]).iloc[:, :-1].rename(index={0:"Non-default", 1:"Default", customerid:"Customer"})
    
    nb_feature_to_compare = st.slider('Select a number of features to compare', 0, df_compare.shape[1], 6)
    
    # use a multiselection to select features to compare
    feature_selection = st.multiselect(
        'Select features to compare', features, features[:nb_feature_to_compare])    
    
    # Display features selected to compare in a number of columns
    nb_columns = 4
    cols = st.columns(nb_columns)
    
    # use to create a new line
    j=0    
    for i in feature_selection:
        with cols[j]:
            # if feature is an object display in a table
            if df_compare[i].dtype=='object':
                st.table(df_compare[i])
            # if feature is not an object display in a barplot
            else:
                fig = plt.figure(figsize=(3,1.5))
                sns.barplot(y=df_compare.index, x=i, data=df_compare)
                plt.xlabel(None)
                plt.title(i)
                st.pyplot(fig)            
        if j>nb_columns-2:
            j=0
            st.markdown('')
        else:
            j=j+1

    # display a table of the features selected to compare        
    st.markdown("<h6 style='text-align: left; color: black;'><u>Table of data</u></h6>", unsafe_allow_html=True)
    st.table(df_compare[feature_selection].round(2).transpose().astype('string')) 
    
   