# Library imports
import uvicorn
import gunicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd
from imblearn.pipeline import Pipeline as imbpipeline

     
# Create app and model objects
app = FastAPI()
pickle_in = open('../models/model_classifier.pkl','rb')
classifier=pickle.load(pickle_in)

# load data
df_data = pd.read_csv("../X_test.csv")
#df_data = df_data.drop(columns=['TARGET'])

threshold_optimized = 0.318

@app.get('/{customerID}')
def prediction(customerID: int):
    if customerID in df_data['SK_ID_CURR'].values:
    
        # select datas of the customer
        customer_data = df_data[df_data['SK_ID_CURR']==customerID]

        # calculate the probability with the model for the customer selected
        predict_proba = classifier.predict_proba(customer_data.values.reshape(1,-1))

        # just consider the probability of not repaying a loan
        predict_proba = predict_proba[:, 1][0]

        # use threshold_optimized for prediction
        if predict_proba >= threshold_optimized:
            prediction = 1
        else:
            prediction = 0

        return {
            'CustomerID': customerID,
            'Prediction': prediction,
            'Probability': round(predict_proba,4)

        }
    else:
        return {f'CustomerID : {customerID} doesn\'t exit'}


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0')
#     uvicorn.run(app, host='127.0.0.1', port=8000)
