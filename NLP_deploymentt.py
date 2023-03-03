import pandas as pd
import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('filename', 'rb'))

st.title('Financial Sentiment Analysis :bar_chart:')

def user_input_features():
    Statment = st.text_input('Enter the Statment') 
    data = {'Statment': Statment}
    features = pd.DataFrame(data,index = [0])
    return features 

df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.button('Click!!') 

if prediction_proba > 0:
    st.write('The statment is Positive')
elif prediction_proba == 0:
    st.write('The Statment is Neutral')
else:
    st.write('The Statment is Negative')

