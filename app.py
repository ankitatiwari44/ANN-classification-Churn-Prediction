import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pandas as pd
import pickle

# LOad the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoder and scalers
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geo.pkl','rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

## streamlit app
st.title("Customer Churn Prediction")

# user input
geography = st.selectbox("Geography",one_hot_encoder_geo.categories_[0])
gender = st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider("Age",18,90)
balance = st.number_input("Balance")
credit_score = st.number_input("CreditScore")
estimated_salary = st.number_input("EstimatedSalary")
tenure = st.slider("Tenure",0,10)
num_of_products = st.slider("NumOfProducts",1,4)
has_cr_card = st.selectbox("HasCrCard",[0,1])
is_active_member = st.selectbox("IsActiveMember",[0,1])


if st.button("Predict Churn"):
    # One-hot encode Geography
    geo_encoder = one_hot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoder, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

    # Prepare other input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # Combine One hot encoder columns to original data
    df = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

    # Scaling the input data
    input_data_scaled = scaler.transform(df)

    # Prediction churn
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.write(f'Churn Prediction Probability: {prediction_proba:.2f}')

    if prediction_proba > 0.5:
        st.write("⚠️ The customer is **likely to churn**.")
    else:
        st.write("✅ The customer is **not likely to churn**.")
